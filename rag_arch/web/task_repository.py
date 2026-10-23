"""PostgreSQL-backed task, slot, worker, and event repository.

Every public mutation uses a short database transaction.  In particular,
claiming or finishing a task never holds a transaction across runner/model
work.  User slots and the global execution limit are durable across API and
worker processes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import time
from typing import Any
from uuid import uuid4

from .task_service import ASSISTANTS, BusyError


ACTIVE_TASK_STATUSES = ("queued", "running", "cancelling", "recovering")
ACTIVE_EXECUTION_STATUSES = ("running", "cancelling", "recovering")
TERMINAL_TASK_STATUSES = ("completed", "failed", "cancelled")
PUBLIC_INPUT_FIELDS = (
    "prompt", "duration_seconds", "attachment_ids", "registered_plan_id",
    "plan_action", "use_embeddings", "max_steps", "max_tool_calls", "simulation",
)
DEFAULT_WORKER_LEASE_SECONDS = 15.0


class IdempotencyConflict(ValueError):
    """A request_id was reused with a different structured input payload."""


class StaleAttempt(RuntimeError):
    """A worker call does not belong to the task's current execution attempt."""


class WorkerIdentityConflict(RuntimeError):
    """A worker_instance identifier was reused for a different OS process."""


class ResourceIdentityConflict(RuntimeError):
    """A resource PID/start identity does not match its persisted record."""


def _now() -> float:
    return time.time()


def _iso(value: float | int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def _json_normalize(value: Any) -> tuple[Any, str]:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(encoded), encoded
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("任务数据必须是有限、可 JSON 序列化的数据。") from exc


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数。")
    return value


def _required_text(value: Any, name: str, max_length: int = 255) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{name} 格式无效。")
    return value.strip()


def _valid_limit(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是{'非负' if allow_zero else '正'}整数。")
    return value


class TaskRepository:
    """Short-transaction repository over the existing :class:`Database`.

    Mutating scheduler paths lock in this order: scheduler singleton, workspace,
    assistant slot, then task.  Progress intentionally skips the scheduler and
    slot, taking workspace then task.  Worker heartbeat takes its worker row
    before task lease updates; no task mutation takes a worker row afterward.
    """

    def __init__(self, db) -> None:
        if db is None or not callable(getattr(db, "transaction", None)):
            raise TypeError("db 必须提供 transaction()。")
        self.db = db

    def _ensure_workspace(self, user_id: int) -> str:
        _positive_int(user_id, "user_id")
        with self.db.transaction() as connection:
            existing = connection.execute(
                "SELECT workspace_id FROM workspaces WHERE user_id = %s", (user_id,)
            ).fetchone()
        if existing is not None:
            return existing["workspace_id"]
        # ``ensure_workspace`` also repairs a missing slot row idempotently.
        # Workspace-less users are initialized once; established workspaces
        # use the read-only path above and do not run an upsert transaction.
        return self.db.ensure_workspace(user_id)

    @staticmethod
    def _lock_scheduler(connection):
        row = connection.execute(
            "SELECT global_execution_limit, fairness_cursor_user_id "
            "FROM scheduler_state WHERE singleton_key = 'global' FOR UPDATE"
        ).fetchone()
        if row is None:
            raise RuntimeError("scheduler_state 全局配置行不存在；请先应用数据库迁移。")
        return row

    @staticmethod
    def _lock_workspace(connection, user_id: int, workspace_id: str | None = None):
        if workspace_id is None:
            row = connection.execute(
                "SELECT workspace_id, last_event_seq FROM workspaces "
                "WHERE user_id = %s FOR UPDATE", (user_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT workspace_id, last_event_seq FROM workspaces "
                "WHERE user_id = %s AND workspace_id = %s FOR UPDATE",
                (user_id, workspace_id),
            ).fetchone()
        if row is None:
            raise KeyError(user_id)
        return row

    @staticmethod
    def _lock_slot(connection, user_id: int, assistant_id: str):
        row = connection.execute(
            "SELECT user_id, assistant_id, active_task_id, history_generation "
            "FROM assistant_slots WHERE user_id = %s AND assistant_id = %s FOR UPDATE",
            (user_id, assistant_id),
        ).fetchone()
        if row is None:
            raise KeyError((user_id, assistant_id))
        return row

    @staticmethod
    def _lock_task(connection, user_id: int, task_id: str):
        return connection.execute(
            "SELECT * FROM tasks WHERE user_id = %s AND task_id = %s FOR UPDATE",
            (user_id, task_id),
        ).fetchone()

    @staticmethod
    def _read_task_event_seq(connection, workspace_id: str, task_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) AS seq FROM workspace_events "
            "WHERE workspace_id = %s AND task_id = %s",
            (workspace_id, task_id),
        ).fetchone()
        return int(row["seq"] if row else 0)

    @staticmethod
    def _public_task(row, updated_seq: int, *, now: float | None = None) -> dict:
        inputs = row["inputs"] or {}
        result = deepcopy(row["result"])
        if isinstance(result, dict):
            result.pop("hits", None)
            result.pop("report_history", None)

        started = row["started_at"]
        finished = row["finished_at"]
        if started is None:
            elapsed = 0.0
        else:
            end = float(finished) if finished is not None else (_now() if now is None else now)
            elapsed = round(max(0.0, end - float(started)), 3)

        task = {key: deepcopy(inputs.get(key)) for key in PUBLIC_INPUT_FIELDS if key in inputs}
        task.setdefault("attachment_ids", [])
        task.update({
            "task_id": row["task_id"],
            "assistant_id": row["assistant_id"],
            "request_id": row["request_id"],
            "prompt": inputs.get("prompt", ""),
            "status": row["status"],
            "stage": row["stage"],
            "detail": row["detail"],
            "created_at": _iso(row["created_at"]),
            "started_at": _iso(started),
            "finished_at": _iso(finished),
            "elapsed_seconds": elapsed,
            "duration_seconds": inputs.get("duration_seconds"),
            "updated_seq": updated_seq,
            "result": result,
            "error": row["error"],
            "simulation": bool(inputs.get("simulation", True)),
        })
        return task

    @staticmethod
    def _append_event(connection, user_id: int, workspace_id: str, event_type: str,
                      data: dict, task_id: str | None = None) -> dict:
        now = _now()
        cursor = connection.execute(
            "UPDATE workspaces SET last_event_seq = last_event_seq + 1, updated_at = %s "
            "WHERE user_id = %s AND workspace_id = %s RETURNING last_event_seq",
            (now, user_id, workspace_id),
        ).fetchone()
        if cursor is None:
            raise KeyError(user_id)
        seq = int(cursor["last_event_seq"])
        normalized, encoded = _json_normalize(data)
        connection.execute(
            "INSERT INTO workspace_events(user_id, workspace_id, seq, event_type, task_id, payload, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)",
            (user_id, workspace_id, seq, event_type, task_id, encoded, now),
        )
        return {"seq": seq, "type": event_type, "data": normalized}

    @classmethod
    def _append_task_event(cls, connection, row) -> dict:
        workspace_id = row["workspace_id"]
        event_data = cls._public_task(
            row, int(connection.execute(
                "SELECT last_event_seq + 1 AS seq FROM workspaces "
                "WHERE user_id = %s AND workspace_id = %s",
                (row["user_id"], workspace_id),
            ).fetchone()["seq"]),
        )
        return cls._append_event(
            connection, row["user_id"], workspace_id, "task.updated", event_data, row["task_id"]
        )

    @staticmethod
    def _row_snapshot(connection, row) -> dict:
        seq = TaskRepository._read_task_event_seq(connection, row["workspace_id"], row["task_id"])
        return TaskRepository._public_task(row, seq)

    @staticmethod
    def _normalize_inputs(inputs: Any) -> tuple[dict, str]:
        if not isinstance(inputs, dict):
            raise ValueError("inputs 必须是对象。")
        normalized, encoded = _json_normalize(inputs)
        prompt = normalized.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请输入任务内容。")
        normalized["prompt"] = prompt.strip()
        if "simulation" not in normalized:
            # Match the existing web service's safe default; API facade passes
            # its server-selected mode explicitly for every request.
            normalized["simulation"] = True
        if type(normalized["simulation"]) is not bool:
            raise ValueError("simulation 必须由服务端设置为布尔值。")
        attachments = normalized.get("attachment_ids", [])
        if (not isinstance(attachments, list)
                or any(not isinstance(item, str) or not item for item in attachments)):
            raise ValueError("附件列表格式无效。")
        normalized["attachment_ids"] = attachments
        normalized, encoded = _json_normalize(normalized)
        return normalized, encoded

    def _validate_attachments(self, connection, user_id: int, workspace_id: str, inputs: dict) -> None:
        attachment_ids = list(dict.fromkeys(inputs.get("attachment_ids", [])))
        if not attachment_ids:
            return
        rows = connection.execute(
            "SELECT attachment_id FROM attachments WHERE user_id = %s AND workspace_id = %s "
            "AND status = 'ready' AND attachment_id = ANY(%s)",
            (user_id, workspace_id, attachment_ids),
        ).fetchall()
        found = {row["attachment_id"] for row in rows}
        if found != set(attachment_ids):
            raise KeyError("attachment")

    def submit(self, user_id: int, assistant_id: str, request_id: str, inputs: dict) -> dict:
        _positive_int(user_id, "user_id")
        if assistant_id not in ASSISTANTS:
            raise ValueError("未知助手。")
        request_id = _required_text(request_id, "request_id", 128)
        normalized_inputs, encoded_inputs = self._normalize_inputs(inputs)
        workspace_id = self._ensure_workspace(user_id)

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            self._lock_workspace(connection, user_id, workspace_id)
            slot = self._lock_slot(connection, user_id, assistant_id)
            existing = connection.execute(
                "SELECT * FROM tasks WHERE user_id = %s AND assistant_id = %s AND request_id = %s FOR UPDATE",
                (user_id, assistant_id, request_id),
            ).fetchone()
            if existing is not None:
                if existing["inputs"] != normalized_inputs:
                    raise IdempotencyConflict("request_id 已用于不同的任务输入。")
                return self._row_snapshot(connection, existing)
            if slot["active_task_id"] is not None:
                raise BusyError("这个助手已有任务在运行，请等待或先取消。")

            self._validate_attachments(connection, user_id, workspace_id, normalized_inputs)
            now = _now()
            task_id = uuid4().hex
            row = connection.execute(
                """
                INSERT INTO tasks(
                    task_id, user_id, workspace_id, assistant_id, request_id, inputs,
                    status, generation, cancel_requested, stage, detail,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'queued', %s, FALSE,
                          'queued', %s, %s, %s)
                RETURNING *
                """,
                (task_id, user_id, workspace_id, assistant_id, request_id,
                 encoded_inputs, slot["history_generation"],
                 "模拟任务等待执行" if normalized_inputs.get("simulation") else "任务等待执行",
                 now, now),
            ).fetchone()
            connection.execute(
                "UPDATE assistant_slots SET active_task_id = %s, updated_at = %s "
                "WHERE user_id = %s AND assistant_id = %s",
                (task_id, now, user_id, assistant_id),
            )
            for attachment_id in dict.fromkeys(normalized_inputs.get("attachment_ids", [])):
                connection.execute(
                    "INSERT INTO task_attachments(user_id, workspace_id, task_id, attachment_id, created_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (user_id, workspace_id, task_id, attachment_id, now),
                )
            event = self._append_task_event(connection, row)
            return self._public_task(row, event["seq"])

    def get_task(self, user_id: int, task_id: str) -> dict | None:
        _positive_int(user_id, "user_id")
        if not isinstance(task_id, str) or not task_id:
            return None
        with self.db.transaction() as connection:
            # State and its last event sequence must come from one MVCC
            # snapshot.  A concurrent finish between these two SELECTs must
            # not make an old running state appear to have the terminal seq.
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            row = connection.execute(
                "SELECT * FROM tasks WHERE user_id = %s AND task_id = %s",
                (user_id, task_id),
            ).fetchone()
            return None if row is None else self._row_snapshot(connection, row)

    def snapshot(self, user_id: int, execution_modes: dict | None = None,
                 model_name: str | None = None, limit: int = 50) -> dict:
        _positive_int(user_id, "user_id")
        limit = _valid_limit(limit, "limit")
        overrides = {} if execution_modes is None else execution_modes
        if (not isinstance(overrides, dict) or set(overrides) - set(ASSISTANTS)
                or any(mode not in ("model", "simulation") for mode in overrides.values())):
            raise ValueError("执行模式只能为已登记助手的model或simulation。")
        modes = {assistant: overrides.get(assistant, "simulation") for assistant in ASSISTANTS}
        workspace_id = self._ensure_workspace(user_id)

        with self.db.transaction() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            workspace = connection.execute(
                "SELECT workspace_id, last_event_seq FROM workspaces WHERE user_id = %s",
                (user_id,),
            ).fetchone()
            if workspace is None:
                raise KeyError(user_id)
            slots = connection.execute(
                "SELECT assistant_id, active_task_id, history_generation FROM assistant_slots "
                "WHERE user_id = %s ORDER BY assistant_id", (user_id,),
            ).fetchall()
            slot_by_assistant = {row["assistant_id"]: row for row in slots}
            assistants = []
            for assistant_id, title in ASSISTANTS.items():
                slot = slot_by_assistant.get(assistant_id)
                if slot is None:
                    raise RuntimeError(f"用户助手槽缺失: {assistant_id}")
                rows = connection.execute(
                    "SELECT * FROM tasks WHERE user_id = %s AND workspace_id = %s "
                    "AND assistant_id = %s AND generation = %s "
                    "ORDER BY created_at DESC, task_id DESC LIMIT %s",
                    (user_id, workspace_id, assistant_id, slot["history_generation"], limit),
                ).fetchall()
                task_rows = {row["task_id"]: row for row in rows}
                active_id = slot["active_task_id"]
                if active_id and active_id not in task_rows:
                    active = connection.execute(
                        "SELECT * FROM tasks WHERE user_id = %s AND task_id = %s",
                        (user_id, active_id),
                    ).fetchone()
                    if active is not None:
                        task_rows[active_id] = active
                ordered_rows = sorted(
                    task_rows.values(), key=lambda row: (float(row["created_at"]), row["task_id"])
                )
                task_ids = [row["task_id"] for row in ordered_rows]
                seq_by_task = {}
                if task_ids:
                    seq_rows = connection.execute(
                        "SELECT task_id, MAX(seq) AS seq FROM workspace_events "
                        "WHERE user_id = %s AND workspace_id = %s AND task_id = ANY(%s) "
                        "GROUP BY task_id",
                        (user_id, workspace_id, task_ids),
                    ).fetchall()
                    seq_by_task = {row["task_id"]: int(row["seq"]) for row in seq_rows}
                tasks = [self._public_task(row, seq_by_task.get(row["task_id"], 0))
                         for row in ordered_rows]
                assistants.append({
                    "assistant_id": assistant_id,
                    "title": title,
                    "execution_mode": modes[assistant_id],
                    "active_task_id": active_id,
                    "tasks": tasks,
                })

            return {
                "workspace_id": workspace_id,
                "owner_user_id": user_id,
                "cursor": int(workspace["last_event_seq"]),
                "mode": "chat_preview" if modes["chat"] == "model" else "infrastructure_demo",
                "model_name": model_name,
                "assistants": assistants,
            }

    def cancel(self, user_id: int, task_id: str) -> dict:
        _positive_int(user_id, "user_id")
        if not isinstance(task_id, str) or not task_id:
            raise KeyError(task_id)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            metadata = connection.execute(
                "SELECT workspace_id, assistant_id FROM tasks WHERE user_id = %s AND task_id = %s",
                (user_id, task_id),
            ).fetchone()
            if metadata is None:
                raise KeyError(task_id)
            self._lock_workspace(connection, user_id, metadata["workspace_id"])
            slot = self._lock_slot(connection, user_id, metadata["assistant_id"])
            row = self._lock_task(connection, user_id, task_id)
            if row is None:
                raise KeyError(task_id)
            if row["status"] in TERMINAL_TASK_STATUSES:
                return self._row_snapshot(connection, row)

            now = _now()
            status = row["status"]
            if status == "queued":
                status, stage = "cancelled", "cancelled"
                detail = "任务已在排队时取消"
            elif status == "running":
                status, stage = "cancelling", "cancelling"
                detail = "取消中，等待当前任务阶段结束"
            else:
                stage = status
                detail = "取消请求已登记，等待确认当前执行状态"
            finished_at = now if status == "cancelled" else row["finished_at"]
            updated = connection.execute(
                "UPDATE tasks SET status = %s, cancel_requested = TRUE, stage = %s, detail = %s, "
                "updated_at = %s, finished_at = %s WHERE task_id = %s RETURNING *",
                (status, stage, detail, now, finished_at, task_id),
            ).fetchone()
            if row["status"] == "queued" and slot["active_task_id"] == task_id:
                connection.execute(
                    "UPDATE assistant_slots SET active_task_id = NULL, updated_at = %s "
                    "WHERE user_id = %s AND assistant_id = %s AND active_task_id = %s",
                    (now, user_id, metadata["assistant_id"], task_id),
                )
            event = self._append_task_event(connection, updated)
            return self._public_task(updated, event["seq"])

    def reset(self, user_id: int, assistant_id: str) -> int:
        _positive_int(user_id, "user_id")
        if assistant_id not in ASSISTANTS:
            raise ValueError("未知助手。")
        workspace_id = self._ensure_workspace(user_id)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            self._lock_workspace(connection, user_id, workspace_id)
            slot = self._lock_slot(connection, user_id, assistant_id)
            if slot["active_task_id"] is not None:
                raise BusyError("请先等待或取消当前任务，再清空这个助手。")
            generation = int(slot["history_generation"]) + 1
            now = _now()
            connection.execute(
                "UPDATE assistant_slots SET history_generation = %s, updated_at = %s "
                "WHERE user_id = %s AND assistant_id = %s",
                (generation, now, user_id, assistant_id),
            )
            self._append_event(
                connection, user_id, workspace_id, "assistant.reset", {"assistant_id": assistant_id}
            )
            return generation

    def event_cursor(self, user_id: int) -> int:
        _positive_int(user_id, "user_id")
        workspace_id = self._ensure_workspace(user_id)
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT last_event_seq FROM workspaces WHERE user_id = %s AND workspace_id = %s",
                (user_id, workspace_id),
            ).fetchone()
            if row is None:
                raise KeyError(user_id)
            return int(row["last_event_seq"])

    @staticmethod
    def _load_event_rows(connection, user_id: int, workspace_id: str, cursor: int, limit: int):
        return connection.execute(
            "SELECT seq, event_type, payload FROM workspace_events "
            "WHERE user_id = %s AND workspace_id = %s AND seq > %s "
            "ORDER BY seq LIMIT %s",
            (user_id, workspace_id, cursor, limit),
        ).fetchall()

    def events_after(self, user_id: int, cursor: int, limit: int = 200) -> tuple[list[dict], str | None]:
        _positive_int(user_id, "user_id")
        limit = _valid_limit(limit, "limit")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            return [], "invalid_cursor"
        workspace_id = self._ensure_workspace(user_id)
        with self.db.transaction() as connection:
            # The latest cursor, expiry check, and returned page must be read
            # from one committed view if an event is appended concurrently.
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            workspace = connection.execute(
                "SELECT last_event_seq FROM workspaces WHERE user_id = %s AND workspace_id = %s",
                (user_id, workspace_id),
            ).fetchone()
            if workspace is None:
                raise KeyError(user_id)
            latest = int(workspace["last_event_seq"])
            if cursor > latest:
                return [], "cursor_ahead"
            earliest_row = connection.execute(
                "SELECT MIN(seq) AS seq FROM workspace_events WHERE user_id = %s AND workspace_id = %s",
                (user_id, workspace_id),
            ).fetchone()
            earliest = earliest_row["seq"] if earliest_row else None
            if earliest is None:
                # Retention may have removed every row while the workspace
                # cursor continues to advance.  A lagging client must resync.
                return ([], "cursor_expired") if cursor < latest else ([], None)
            if earliest is not None and cursor < int(earliest) - 1:
                return [], "cursor_expired"
            rows = self._load_event_rows(connection, user_id, workspace_id, cursor, limit)
            events = []
            expected = cursor + 1
            for row in rows:
                seq = int(row["seq"])
                if seq != expected:
                    return [], "cursor_expired"
                events.append({"seq": seq, "type": row["event_type"], "data": row["payload"]})
                expected = seq + 1
            if not events and cursor < latest:
                return [], "cursor_expired"
            return events, None

    def set_global_limit(self, limit: int) -> int:
        """Explicit administrative setting; constructors and workers never call it."""
        limit = _valid_limit(limit, "limit")
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            connection.execute(
                "UPDATE scheduler_state SET global_execution_limit = %s, updated_at = %s "
                "WHERE singleton_key = 'global'",
                (limit, _now()),
            )
        return limit

    def register_worker(self, worker_instance: str, host_id: str, pid: int,
                        process_start_token: str) -> dict:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        host_id = _required_text(host_id, "host_id", 255)
        pid = _positive_int(pid, "pid")
        process_start_token = _required_text(process_start_token, "process_start_token", 255)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            existing = connection.execute(
                "SELECT * FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
                (worker_instance,),
            ).fetchone()
            if existing is None:
                now = _now()
                row = connection.execute(
                    "INSERT INTO worker_instances(worker_instance, host_id, pid, process_start_token, "
                    "state, heartbeat_at, lease_seconds, registered_at) "
                    "VALUES (%s, %s, %s, %s, 'active', %s, %s, %s) RETURNING *",
                    (worker_instance, host_id, pid, process_start_token, now,
                     DEFAULT_WORKER_LEASE_SECONDS, now),
                ).fetchone()
                return dict(row)
            identity = (existing["host_id"], int(existing["pid"]), existing["process_start_token"])
            if identity != (host_id, pid, process_start_token):
                raise WorkerIdentityConflict("worker_instance 已绑定到另一个进程身份。")
            if existing["state"] in ("exited", "stopped"):
                raise WorkerIdentityConflict("已停止的 worker_instance 不能重新注册。")
            now = _now()
            row = connection.execute(
                "UPDATE worker_instances SET state = 'active', heartbeat_at = %s, stale_at = NULL "
                "WHERE worker_instance = %s RETURNING *",
                (now, worker_instance),
            ).fetchone()
            return dict(row)

    def heartbeat_worker(self, worker_instance: str,
                         lease_seconds: float = DEFAULT_WORKER_LEASE_SECONDS) -> bool:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        if (isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float))
                or not math.isfinite(lease_seconds) or lease_seconds <= 0):
            raise ValueError("lease_seconds 必须是有限正数。")
        with self.db.transaction() as connection:
            worker = connection.execute(
                "SELECT state FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
                (worker_instance,),
            ).fetchone()
            if worker is None:
                raise KeyError(worker_instance)
            if worker["state"] in ("exited", "stopped"):
                return False
            # Lock active executions before taking the renewal timestamp so a
            # task-row wait does not consume any part of the new lease.
            active_tasks = connection.execute(
                "SELECT task_id FROM tasks WHERE worker_instance = %s "
                "AND status IN ('running', 'cancelling', 'recovering') "
                "ORDER BY task_id FOR UPDATE",
                (worker_instance,),
            ).fetchall()
            now = _now()
            connection.execute(
                "UPDATE worker_instances SET state = 'active', heartbeat_at = %s, "
                "lease_seconds = %s, stale_at = NULL WHERE worker_instance = %s",
                (now, float(lease_seconds), worker_instance),
            )
            connection.execute(
                "UPDATE tasks SET lease_until = %s WHERE worker_instance = %s "
                "AND status IN ('running', 'cancelling', 'recovering')",
                (now + float(lease_seconds), worker_instance),
            )
            return True

    def stop_worker(self, worker_instance: str) -> bool:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = connection.execute(
                "SELECT state FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
                (worker_instance,),
            ).fetchone()
            if worker is None:
                raise KeyError(worker_instance)
            active = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE worker_instance = %s "
                "AND status IN ('running', 'cancelling', 'recovering')",
                (worker_instance,),
            ).fetchone()["count"]
            if active:
                raise BusyError("worker 仍有活动任务，完成任务后才能停止。")
            pending_resources = connection.execute(
                "SELECT count(*) AS count FROM process_resources "
                "WHERE worker_instance = %s AND state <> 'exited'",
                (worker_instance,),
            ).fetchone()["count"]
            if pending_resources:
                raise BusyError("worker 仍有未确认退出的进程资源，确认全部退出后才能停止。")
            if worker["state"] == "stopped":
                return True
            connection.execute(
                "UPDATE worker_instances SET state = 'stopped', stopped_at = %s "
                "WHERE worker_instance = %s",
                (_now(), worker_instance),
            )
            return True

    def workers(self) -> list[dict]:
        with self.db.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM worker_instances ORDER BY registered_at, worker_instance"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_worker(self, worker_instance: str) -> dict | None:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM worker_instances WHERE worker_instance = %s",
                (worker_instance,),
            ).fetchone()
            return None if row is None else dict(row)

    @staticmethod
    def _lock_worker(connection, worker_instance: str):
        row = connection.execute(
            "SELECT * FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
            (worker_instance,),
        ).fetchone()
        if row is None:
            raise KeyError(worker_instance)
        return row

    def _lock_worker_task(self, connection, worker_instance: str, task_id: str):
        metadata = connection.execute(
            "SELECT user_id, workspace_id, assistant_id FROM tasks WHERE task_id = %s",
            (task_id,),
        ).fetchone()
        if metadata is None:
            raise KeyError(task_id)
        self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
        slot = self._lock_slot(connection, metadata["user_id"], metadata["assistant_id"])
        row = self._lock_task(connection, metadata["user_id"], task_id)
        if row is None or row["worker_instance"] != worker_instance:
            raise StaleAttempt("resource task is not assigned to this worker")
        if row["status"] not in ACTIVE_EXECUTION_STATUSES:
            raise StaleAttempt("resource task is no longer active")
        if slot["active_task_id"] != task_id:
            raise RuntimeError("活动任务与 assistant_slots.active_task_id 不一致。")
        return row

    @staticmethod
    def _task_has_unconfirmed_resources(connection, task_id: str) -> bool:
        row = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM process_resources
                WHERE task_id = %s AND scope = 'task' AND state <> 'exited'
                UNION ALL
                SELECT 1
                FROM process_resource_uses AS u
                JOIN process_resources AS r USING (resource_id)
                WHERE u.task_id = %s AND u.released_at IS NULL
                  AND r.state IN ('starting', 'cleanup_pending', 'unknown')
            ) AS pending
            """,
            (task_id, task_id),
        ).fetchone()
        return bool(row["pending"])

    @staticmethod
    def _task_has_unconfirmed_recovery_resources(connection, task_id: str) -> bool:
        """A dead worker cannot release a task that used any still-live child."""
        row = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM process_resources
                WHERE task_id = %s AND scope = 'task' AND state <> 'exited'
                UNION ALL
                SELECT 1
                FROM process_resource_uses AS u
                JOIN process_resources AS r USING (resource_id)
                WHERE u.task_id = %s AND u.released_at IS NULL AND r.state <> 'exited'
            ) AS pending
            """,
            (task_id, task_id),
        ).fetchone()
        return bool(row["pending"])

    def begin_resource(self, worker_instance: str, kind: str, scope: str,
                       task_id: str | None = None,
                       owner_path: str | None = None, *,
                       resource_id: str | None = None,
                       owner_descriptor_path: str | None = None) -> dict:
        """Record an OS resource before creating it, so crashes remain visible."""
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        kind = _required_text(kind, "kind", 64)
        if scope not in ("worker", "task"):
            raise ValueError("scope 只能是 worker 或 task。")
        if scope == "task" and task_id is None:
            raise ValueError("task_scope resource 必须绑定 task_id。")
        if task_id is not None:
            task_id = _required_text(task_id, "task_id", 128)
        if owner_path is not None:
            owner_path = _required_text(owner_path, "owner_path", 4096)
        if resource_id is None:
            resource_id = uuid4().hex
        else:
            resource_id = _required_text(resource_id, "resource_id", 128)
        if owner_descriptor_path is not None:
            owner_descriptor_path = _required_text(
                owner_descriptor_path, "owner_descriptor_path", 4096,
            )
        owner_nonce = uuid4().hex if owner_descriptor_path is not None else None

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = self._lock_worker(connection, worker_instance)
            if worker["state"] != "active":
                raise StaleAttempt("只有active worker能创建资源记录。")
            if task_id is not None:
                self._lock_worker_task(connection, worker_instance, task_id)
            now = _now()
            row = connection.execute(
                "INSERT INTO process_resources(resource_id, worker_instance, kind, scope, task_id, "
                "owner_path, owner_descriptor_path, owner_nonce, state, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'starting', %s, %s) RETURNING *",
                (resource_id, worker_instance, kind, scope, task_id, owner_path,
                 owner_descriptor_path, owner_nonce, now, now),
            ).fetchone()
            return dict(row)

    def attach_resource(self, worker_instance: str, resource_id: str, *,
                        host_id: str, pid: int, process_start_token: str,
                        executable: str | None = None) -> dict:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        resource_id = _required_text(resource_id, "resource_id", 128)
        host_id = _required_text(host_id, "host_id", 255)
        pid = _positive_int(pid, "pid")
        process_start_token = _required_text(process_start_token, "process_start_token", 255)
        if executable is not None:
            executable = _required_text(executable, "executable", 4096)

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = self._lock_worker(connection, worker_instance)
            row = connection.execute(
                "SELECT * FROM process_resources WHERE resource_id = %s FOR UPDATE",
                (resource_id,),
            ).fetchone()
            if row is None or row["worker_instance"] != worker_instance:
                raise KeyError(resource_id)
            if worker["host_id"] != host_id:
                raise ResourceIdentityConflict("resource host_id 与 worker 登记主机不一致。")
            identity = (row["host_id"], row["pid"], row["process_start_token"])
            requested = (host_id, pid, process_start_token)
            if row["state"] == "starting":
                if worker["state"] != "active":
                    raise StaleAttempt("非active worker不能attach新资源。")
                now = _now()
                updated = connection.execute(
                    "UPDATE process_resources SET state = 'active', host_id = %s, pid = %s, "
                    "process_start_token = %s, executable = %s, updated_at = %s "
                    "WHERE resource_id = %s RETURNING *",
                    (host_id, pid, process_start_token, executable, now, resource_id),
                ).fetchone()
                return dict(updated)
            if identity != requested:
                raise ResourceIdentityConflict("resource PID/start token 与登记身份不一致。")
            if row["state"] not in ("active", "cleanup_pending"):
                raise ResourceIdentityConflict("已终结或unknown的resource不能重新attach。")
            if executable is not None and row["executable"] not in (None, executable):
                raise ResourceIdentityConflict("resource executable 与首次登记值不一致。")
            return dict(row)

    def mark_resource_cleanup_pending(self, worker_instance: str,
                                      resource_id: str) -> bool:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        resource_id = _required_text(resource_id, "resource_id", 128)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            self._lock_worker(connection, worker_instance)
            row = connection.execute(
                "SELECT state FROM process_resources WHERE resource_id = %s "
                "AND worker_instance = %s FOR UPDATE",
                (resource_id, worker_instance),
            ).fetchone()
            if row is None:
                raise KeyError(resource_id)
            if row["state"] == "cleanup_pending":
                return True
            if row["state"] == "exited":
                return True
            if row["state"] not in ("active",):
                return False
            connection.execute(
                "UPDATE process_resources SET state = 'cleanup_pending', updated_at = %s "
                "WHERE resource_id = %s",
                (_now(), resource_id),
            )
            return True

    def use_resource(self, worker_instance: str, resource_id: str,
                     task_id: str) -> bool:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        resource_id = _required_text(resource_id, "resource_id", 128)
        task_id = _required_text(task_id, "task_id", 128)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = self._lock_worker(connection, worker_instance)
            if worker["state"] != "active":
                return False
            task = self._lock_worker_task(connection, worker_instance, task_id)
            resource = connection.execute(
                "SELECT * FROM process_resources WHERE resource_id = %s FOR UPDATE",
                (resource_id,),
            ).fetchone()
            if resource is None or resource["worker_instance"] != worker_instance:
                raise KeyError(resource_id)
            if resource["state"] != "active":
                return False
            if resource["scope"] == "task" and resource["task_id"] != task_id:
                raise ResourceIdentityConflict("task_scope resource 不能被其他 task 使用。")
            token = task["attempt_token"]
            if not token:
                raise StaleAttempt("活动任务缺少attempt token。")
            existing = connection.execute(
                "SELECT released_at FROM process_resource_uses WHERE resource_id = %s "
                "AND task_id = %s AND attempt_token = %s",
                (resource_id, task_id, token),
            ).fetchone()
            if existing is not None:
                return existing["released_at"] is None
            connection.execute(
                "INSERT INTO process_resource_uses(resource_id, worker_instance, task_id, "
                "attempt_token, acquired_at) VALUES (%s, %s, %s, %s, %s)",
                (resource_id, worker_instance, task_id, token, _now()),
            )
            return True

    def resource_exited(self, worker_instance: str, resource_id: str, *,
                        pid: int | None = None,
                        process_start_token: str | None = None) -> bool:
        """Commit OS-confirmed exit; attached resources require the exact identity."""
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        resource_id = _required_text(resource_id, "resource_id", 128)
        if pid is not None:
            pid = _positive_int(pid, "pid")
        if process_start_token is not None:
            process_start_token = _required_text(process_start_token, "process_start_token", 255)
        if (pid is None) != (process_start_token is None):
            raise ValueError("pid 与 process_start_token 必须同时提供或同时省略。")

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            self._lock_worker(connection, worker_instance)
            row = connection.execute(
                "SELECT * FROM process_resources WHERE resource_id = %s FOR UPDATE",
                (resource_id,),
            ).fetchone()
            if row is None or row["worker_instance"] != worker_instance:
                raise KeyError(resource_id)
            registered = (row["pid"], row["process_start_token"])
            requested = (pid, process_start_token)
            if row["state"] == "exited":
                return requested == registered
            if registered == (None, None):
                # No child identity was attached.  This is reserved for an
                # explicit probe confirming creation never succeeded.
                if requested != (None, None) or row["state"] != "starting":
                    return False
            elif requested != registered:
                return False
            now = _now()
            connection.execute(
                "UPDATE process_resources SET state = 'exited', error_code = NULL, "
                "updated_at = %s, exited_at = %s WHERE resource_id = %s",
                (now, now, resource_id),
            )
            connection.execute(
                "UPDATE process_resource_uses SET released_at = GREATEST(COALESCE(released_at, %s), acquired_at) "
                "WHERE resource_id = %s",
                (now, resource_id),
            )
            return True

    def resource_unknown(self, worker_instance: str, resource_id: str,
                         error_code: str) -> bool:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        resource_id = _required_text(resource_id, "resource_id", 128)
        error_code = _required_text(error_code, "error_code", 128)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            self._lock_worker(connection, worker_instance)
            row = connection.execute(
                "SELECT state, error_code FROM process_resources WHERE resource_id = %s "
                "AND worker_instance = %s FOR UPDATE",
                (resource_id, worker_instance),
            ).fetchone()
            if row is None:
                raise KeyError(resource_id)
            if row["state"] == "exited":
                return False
            if row["state"] == "unknown" and row["error_code"] == error_code:
                return True
            connection.execute(
                "UPDATE process_resources SET state = 'unknown', error_code = %s, updated_at = %s "
                "WHERE resource_id = %s",
                (error_code, _now(), resource_id),
            )
            return True

    def backfill_unknown_resource_identity(self, worker_instance: str, resource_id: str, *,
                                           owner_descriptor_path: str,
                                           descriptor: dict,
                                           descriptor_file_identity: dict,
                                           probe_state: str) -> dict:
        """Append trusted owner evidence before attaching an exact child identity.

        This is intentionally limited to unknown rows with no PID, after the
        parent worker has been confirmed exited.  A running or inaccessible
        child may receive its identity for later inspection, but remains
        ``unknown`` and cannot release any task capacity.
        """
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        resource_id = _required_text(resource_id, "resource_id", 128)
        owner_descriptor_path = _required_text(
            owner_descriptor_path, "owner_descriptor_path", 4096,
        )
        if probe_state not in ("running", "exited", "unknown"):
            raise ValueError("probe_state 只能是running、exited或unknown。")
        normalized, _ = _json_normalize(descriptor)
        file_identity, _ = _json_normalize(descriptor_file_identity)
        file_identity_keys = {"device", "inode", "size", "mtime_ns", "mode", "owner_uid"}
        if not isinstance(file_identity, dict) or set(file_identity) != file_identity_keys:
            raise ValueError("descriptor_file_identity字段无效。")
        for name in ("device", "inode", "size", "mtime_ns", "mode"):
            if isinstance(file_identity[name], bool) or not isinstance(file_identity[name], int):
                raise ValueError(f"descriptor_file_identity.{name}必须是整数。")
        if file_identity["size"] < 1 or file_identity["size"] > 65536:
            raise ValueError("descriptor_file_identity.size超出允许范围。")
        if (file_identity["owner_uid"] is not None
                and (isinstance(file_identity["owner_uid"], bool)
                     or not isinstance(file_identity["owner_uid"], int))):
            raise ValueError("descriptor_file_identity.owner_uid必须是整数或None。")
        descriptor_keys = {
            "schema_version", "resource_id", "owner_nonce", "worker_instance",
            "owner_path", "owned", "worker", "resource",
        }
        if not isinstance(normalized, dict) or set(normalized) != descriptor_keys:
            raise ValueError("owner descriptor字段不符合资源身份契约。")
        if normalized["schema_version"] != 1 or normalized["resource_id"] != resource_id:
            raise ResourceIdentityConflict("owner descriptor版本或resource_id不匹配。")
        if normalized["worker_instance"] != worker_instance:
            raise ResourceIdentityConflict("owner descriptor的worker_instance不匹配。")
        if normalized["owned"] is not True:
            raise ResourceIdentityConflict("owner descriptor未确认该worker拥有此进程。")
        worker_proof = normalized["worker"]
        worker_keys = {"host_id", "pid", "process_start_token"}
        if not isinstance(worker_proof, dict) or set(worker_proof) != worker_keys:
            raise ValueError("owner descriptor缺少精确worker进程身份。")
        worker_host = _required_text(worker_proof["host_id"], "descriptor.worker.host_id", 255)
        worker_pid = _positive_int(worker_proof["pid"], "descriptor.worker.pid")
        worker_token = _required_text(
            worker_proof["process_start_token"], "descriptor.worker.process_start_token", 255,
        )
        resource_proof = normalized["resource"]
        resource_keys = {"host_id", "pid", "process_start_token", "executable"}
        if not isinstance(resource_proof, dict) or set(resource_proof) != resource_keys:
            raise ValueError("owner descriptor缺少精确resource进程身份。")
        resource_host = _required_text(resource_proof["host_id"], "descriptor.resource.host_id", 255)
        resource_pid = _positive_int(resource_proof["pid"], "descriptor.resource.pid")
        resource_token = _required_text(
            resource_proof["process_start_token"], "descriptor.resource.process_start_token", 255,
        )
        executable = _required_text(resource_proof["executable"], "descriptor.resource.executable", 4096)
        if normalized["owner_path"] is not None:
            normalized["owner_path"] = _required_text(normalized["owner_path"], "descriptor.owner_path", 4096)

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = self._lock_worker(connection, worker_instance)
            if worker["state"] != "exited":
                raise StaleAttempt("只能为已确认退出的worker补齐资源身份。")
            expected_worker = (worker["host_id"], int(worker["pid"]), worker["process_start_token"])
            supplied_worker = (worker_host, worker_pid, worker_token)
            if expected_worker != supplied_worker:
                raise WorkerIdentityConflict("owner descriptor的worker进程身份与登记不一致。")
            row = connection.execute(
                "SELECT * FROM process_resources WHERE resource_id = %s FOR UPDATE",
                (resource_id,),
            ).fetchone()
            if row is None or row["worker_instance"] != worker_instance:
                raise KeyError(resource_id)
            if row["owner_descriptor_path"] != owner_descriptor_path:
                raise ResourceIdentityConflict("descriptor路径与数据库登记路径不一致。")
            if not row["owner_descriptor_path"] or not row["owner_nonce"]:
                raise ResourceIdentityConflict("资源没有预登记可信owner descriptor。")
            if normalized["owner_nonce"] != row["owner_nonce"]:
                raise ResourceIdentityConflict("owner nonce不匹配。")
            if normalized["owner_path"] != row["owner_path"]:
                raise ResourceIdentityConflict("owner_path与资源预登记不匹配。")
            if row["state"] == "exited":
                identity = (row["host_id"], row["pid"], row["process_start_token"], row["executable"])
                supplied = (resource_host, resource_pid, resource_token, executable)
                if identity != supplied:
                    raise ResourceIdentityConflict("resource身份与已确认退出记录不一致。")
                return dict(row)
            if row["state"] != "unknown":
                raise StaleAttempt("仅unknown资源允许通过descriptor补身份。")
            current_identity = (row["host_id"], row["pid"], row["process_start_token"])
            if current_identity != (None, None, None):
                if current_identity != (resource_host, resource_pid, resource_token):
                    raise ResourceIdentityConflict("resource已有PID身份与descriptor不一致。")
                audit = connection.execute(
                    "SELECT evidence FROM process_resource_identity_audit WHERE resource_id = %s",
                    (resource_id,),
                ).fetchone()
                evidence = {"descriptor": normalized, "file_identity": file_identity}
                if audit is not None and audit["evidence"] == evidence:
                    return dict(row)
                raise ResourceIdentityConflict("已补齐的resource身份不能由另一descriptor覆盖。")
            if row["pid"] is not None or row["process_start_token"] is not None:
                raise ResourceIdentityConflict("resource身份列状态不一致。")

            now = _now()
            resulting_state = "exited" if probe_state == "exited" else "unknown"
            updated = connection.execute(
                "UPDATE process_resources SET host_id = %s, pid = %s, process_start_token = %s, "
                "executable = %s, state = %s, error_code = %s, updated_at = %s, exited_at = %s "
                "WHERE resource_id = %s RETURNING *",
                (resource_host, resource_pid, resource_token, executable, resulting_state,
                 None if probe_state == "exited" else f"owner_descriptor_probe_{probe_state}",
                 now, now if probe_state == "exited" else None, resource_id),
            ).fetchone()
            connection.execute(
                "INSERT INTO process_resource_identity_audit(worker_instance, resource_id, "
                "owner_descriptor_path, owner_nonce, probe_state, evidence, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)",
                (worker_instance, resource_id, owner_descriptor_path, row["owner_nonce"],
                 probe_state,
                 json.dumps({"descriptor": normalized, "file_identity": file_identity},
                            ensure_ascii=False, allow_nan=False, sort_keys=True,
                            separators=(",", ":")), now),
            )
            if probe_state == "exited":
                connection.execute(
                    "UPDATE process_resource_uses SET released_at = GREATEST(COALESCE(released_at, %s), acquired_at) "
                    "WHERE resource_id = %s",
                    (now, resource_id),
                )
            return dict(updated)

    def resources_for_worker(self, worker_instance: str) -> list[dict]:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        with self.db.transaction() as connection:
            worker = connection.execute(
                "SELECT 1 FROM worker_instances WHERE worker_instance = %s",
                (worker_instance,),
            ).fetchone()
            if worker is None:
                raise KeyError(worker_instance)
            rows = connection.execute(
                "SELECT * FROM process_resources WHERE worker_instance = %s "
                "ORDER BY created_at, resource_id",
                (worker_instance,),
            ).fetchall()
            resources = [dict(row) for row in rows]
            if not resources:
                return []
            ids = [row["resource_id"] for row in resources]
            uses = connection.execute(
                "SELECT use_id, resource_id, task_id, attempt_token, acquired_at, released_at "
                "FROM process_resource_uses WHERE worker_instance = %s AND resource_id = ANY(%s) "
                "ORDER BY acquired_at, use_id",
                (worker_instance, ids),
            ).fetchall()
            by_resource: dict[str, list[dict]] = {resource_id: [] for resource_id in ids}
            for use in uses:
                by_resource[use["resource_id"]].append(dict(use))
            for resource in resources:
                resource["uses"] = by_resource[resource["resource_id"]]
            return resources

    def task_has_unconfirmed_resources(self, task_id: str) -> bool:
        task_id = _required_text(task_id, "task_id", 128)
        with self.db.transaction() as connection:
            return self._task_has_unconfirmed_resources(connection, task_id)

    def keep_recovering(self, worker_instance: str, task_id: str,
                        attempt_token: str, error: str) -> bool:
        """Keep both scheduler and user capacity reserved while resources are uncertain."""
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        task_id = _required_text(task_id, "task_id", 128)
        attempt_token = _required_text(attempt_token, "attempt_token", 128)
        error = _required_text(error, "error", 2048)
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = self._lock_worker(connection, worker_instance)
            if worker["state"] not in ("active", "stale"):
                return False
            metadata = connection.execute(
                "SELECT user_id, workspace_id, assistant_id FROM tasks WHERE task_id = %s",
                (task_id,),
            ).fetchone()
            if metadata is None:
                return False
            self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
            slot = self._lock_slot(connection, metadata["user_id"], metadata["assistant_id"])
            row = self._lock_task(connection, metadata["user_id"], task_id)
            if (row is None or row["worker_instance"] != worker_instance
                    or row["attempt_token"] != attempt_token
                    or row["status"] not in ACTIVE_EXECUTION_STATUSES):
                return False
            if slot["active_task_id"] != task_id:
                raise RuntimeError("恢复任务与 assistant_slots.active_task_id 不一致。")
            detail = "资源状态未能确认，任务槽保留等待人工核验"
            if row["status"] == "recovering" and row["error"] == error and row["detail"] == detail:
                return True
            now = _now()
            updated = connection.execute(
                "UPDATE tasks SET status = 'recovering', stage = 'recovering', detail = %s, "
                "error = %s, lease_until = NULL, updated_at = %s WHERE task_id = %s RETURNING *",
                (detail, error, now, task_id),
            ).fetchone()
            self._append_task_event(connection, updated)
            return True

    def resolve_confirmed_task(self, worker_instance: str, task_id: str,
                               attempt_token: str, proof: dict) -> bool:
        """Resolve a held task only after an administrator proves every process exited."""
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        task_id = _required_text(task_id, "task_id", 128)
        attempt_token = _required_text(attempt_token, "attempt_token", 128)
        normalized_proof, encoded_proof = _json_normalize(proof)
        expected_keys = {
            "operator", "reason", "external_work_confirmed_ended", "worker", "resources",
        }
        if not isinstance(normalized_proof, dict) or set(normalized_proof) != expected_keys:
            raise ValueError("proof 字段不符合管理员核验契约。")
        _required_text(normalized_proof["operator"], "proof.operator", 255)
        _required_text(normalized_proof["reason"], "proof.reason", 2048)
        if normalized_proof["external_work_confirmed_ended"] is not True:
            raise ValueError("proof 必须明确 external_work_confirmed_ended=true。")
        worker_proof = normalized_proof["worker"]
        identity_keys = {"worker_instance", "host_id", "pid", "process_start_token"}
        if not isinstance(worker_proof, dict) or set(worker_proof) != identity_keys:
            raise ValueError("proof.worker 必须包含worker_instance、host_id、pid与process_start_token。")
        if worker_proof["worker_instance"] != worker_instance:
            raise WorkerIdentityConflict("proof中的worker_instance与目标worker不匹配。")
        _required_text(worker_proof["host_id"], "proof.worker.host_id", 255)
        _positive_int(worker_proof["pid"], "proof.worker.pid")
        _required_text(worker_proof["process_start_token"], "proof.worker.process_start_token", 255)
        resource_proofs = normalized_proof["resources"]
        if not isinstance(resource_proofs, list):
            raise ValueError("proof.resources 必须是数组。")
        normalized_resource_proofs = {}
        resource_keys = {"resource_id", "host_id", "pid", "process_start_token"}
        for entry in resource_proofs:
            if not isinstance(entry, dict) or set(entry) != resource_keys:
                raise ValueError("每个resource proof必须精确包含resource_id与进程身份。")
            resource_id = _required_text(entry["resource_id"], "proof.resource_id", 128)
            if resource_id in normalized_resource_proofs:
                raise ValueError("proof.resources 存在重复resource_id。")
            if entry["pid"] is None and entry["process_start_token"] is None and entry["host_id"] is None:
                identity = (None, None, None)
            else:
                _required_text(entry["host_id"], "proof.resource.host_id", 255)
                _positive_int(entry["pid"], "proof.resource.pid")
                _required_text(entry["process_start_token"], "proof.resource.process_start_token", 255)
                identity = (entry["host_id"], entry["pid"], entry["process_start_token"])
            normalized_resource_proofs[resource_id] = identity

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = self._lock_worker(connection, worker_instance)
            metadata = connection.execute(
                "SELECT user_id, workspace_id, assistant_id FROM tasks WHERE task_id = %s",
                (task_id,),
            ).fetchone()
            if metadata is None:
                return False
            self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
            slot = self._lock_slot(connection, metadata["user_id"], metadata["assistant_id"])
            row = self._lock_task(connection, metadata["user_id"], task_id)
            if row is None or row["worker_instance"] != worker_instance or row["attempt_token"] != attempt_token:
                return False
            prior = connection.execute(
                "SELECT proof FROM task_resolution_audit WHERE task_id = %s AND attempt_token = %s",
                (task_id, attempt_token),
            ).fetchone()
            if row["status"] in TERMINAL_TASK_STATUSES:
                return prior is not None and prior["proof"] == normalized_proof
            if row["status"] not in ACTIVE_EXECUTION_STATUSES or worker["state"] != "exited":
                return False
            expected_worker = (worker["host_id"], int(worker["pid"]), worker["process_start_token"])
            supplied_worker = (worker_proof["host_id"], worker_proof["pid"], worker_proof["process_start_token"])
            if expected_worker != supplied_worker:
                raise WorkerIdentityConflict("proof 中的worker进程身份与登记信息不匹配。")

            resources = connection.execute(
                "SELECT resource_id, state, host_id, pid, process_start_token "
                "FROM process_resources WHERE worker_instance = %s "
                "ORDER BY resource_id FOR UPDATE",
                (worker_instance,),
            ).fetchall()
            if {resource["resource_id"] for resource in resources} != set(normalized_resource_proofs):
                raise ResourceIdentityConflict("proof.resources 必须覆盖该worker的全部资源。")
            for resource in resources:
                if resource["state"] != "exited":
                    return False
                expected = (resource["host_id"], resource["pid"], resource["process_start_token"])
                if expected != normalized_resource_proofs[resource["resource_id"]]:
                    raise ResourceIdentityConflict(
                        f"resource {resource['resource_id']} 的PID/start身份与proof不一致。"
                    )
            if slot["active_task_id"] != task_id:
                raise RuntimeError("恢复任务与 assistant_slots.active_task_id 不一致。")
            terminal_status = "cancelled" if row["cancel_requested"] else "failed"
            now = _now()
            detail = "管理员已核实worker与全部资源退出；任务结果未恢复"
            error = "administrator confirmed external work ended; result unavailable"
            updated = connection.execute(
                "UPDATE tasks SET status = %s, stage = %s, detail = %s, error = %s, result = NULL, "
                "updated_at = %s, finished_at = %s, lease_until = NULL "
                "WHERE task_id = %s RETURNING *",
                (terminal_status, terminal_status, detail, error, now, now, task_id),
            ).fetchone()
            connection.execute(
                "UPDATE assistant_slots SET active_task_id = NULL, updated_at = %s "
                "WHERE user_id = %s AND assistant_id = %s AND active_task_id = %s",
                (now, metadata["user_id"], metadata["assistant_id"], task_id),
            )
            connection.execute(
                "INSERT INTO task_resolution_audit(worker_instance, task_id, attempt_token, "
                "terminal_status, proof, created_at) VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
                (worker_instance, task_id, attempt_token, terminal_status, encoded_proof, now),
            )
            connection.execute(
                "UPDATE process_resource_uses SET released_at = GREATEST(COALESCE(released_at, %s), acquired_at) "
                "WHERE task_id = %s AND attempt_token = %s",
                (now, task_id, attempt_token),
            )
            self._append_task_event(connection, updated)
            return True

    def resolve_confirmed_task_by_proof(self, task_id: str, proof: dict) -> bool:
        """CLI helper: select the current worker/attempt only from the exact proof."""
        task_id = _required_text(task_id, "task_id", 128)
        if not isinstance(proof, dict) or not isinstance(proof.get("worker"), dict):
            raise ValueError("proof必须包含worker身份对象。")
        worker_instance = _required_text(
            proof["worker"].get("worker_instance"), "proof.worker.worker_instance", 128,
        )
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT worker_instance, attempt_token FROM tasks WHERE task_id = %s",
                (task_id,),
            ).fetchone()
        if row is None or row["worker_instance"] != worker_instance or not row["attempt_token"]:
            return False
        return self.resolve_confirmed_task(
            worker_instance, task_id, row["attempt_token"], proof,
        )

    def claim_next(self, worker_instance: str) -> dict | None:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        with self.db.transaction() as connection:
            scheduler = self._lock_scheduler(connection)
            worker = connection.execute(
                "SELECT * FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
                (worker_instance,),
            ).fetchone()
            if worker is None:
                raise KeyError(worker_instance)
            if worker["state"] != "active":
                return None
            checked_at = _now()
            if float(worker["heartbeat_at"]) + float(worker["lease_seconds"]) < checked_at:
                return None
            executing = connection.execute(
                "SELECT count(*) AS count FROM tasks WHERE status IN ('running', 'cancelling', 'recovering')"
            ).fetchone()["count"]
            if int(executing) >= int(scheduler["global_execution_limit"]):
                return None

            cursor_user = scheduler["fairness_cursor_user_id"]
            candidate = connection.execute(
                """
                SELECT t.task_id, t.user_id, t.workspace_id, t.assistant_id
                FROM tasks AS t
                JOIN assistant_slots AS s
                  ON s.user_id = t.user_id AND s.assistant_id = t.assistant_id
                 AND s.active_task_id = t.task_id
                WHERE t.status = 'queued'
                ORDER BY CASE WHEN %s::bigint IS NULL OR t.user_id > %s::bigint THEN 0 ELSE 1 END,
                         t.user_id, t.created_at, t.task_id
                LIMIT 1
                """,
                (cursor_user, cursor_user),
            ).fetchone()
            if candidate is None:
                return None

            self._lock_workspace(connection, candidate["user_id"], candidate["workspace_id"])
            self._lock_slot(connection, candidate["user_id"], candidate["assistant_id"])
            row = self._lock_task(connection, candidate["user_id"], candidate["task_id"])
            if row is None or row["status"] != "queued":
                return None
            now = _now()
            # The scheduler/workspace/slot/task locks above may have waited.
            # Start the task and its lease from the actual claim moment.
            if float(worker["heartbeat_at"]) + float(worker["lease_seconds"]) < now:
                return None
            token = uuid4().hex
            lease_seconds = float(worker["lease_seconds"])
            updated = connection.execute(
                "UPDATE tasks SET status = 'running', worker_instance = %s, attempt_token = %s, "
                "lease_until = %s, stage = 'starting', detail = %s, started_at = %s, updated_at = %s "
                "WHERE task_id = %s RETURNING *",
                (worker_instance, token, now + lease_seconds,
                 "模拟任务已启动" if (row["inputs"] or {}).get("simulation", True) else "任务已启动",
                 now, now, row["task_id"]),
            ).fetchone()
            connection.execute(
                "UPDATE scheduler_state SET fairness_cursor_user_id = %s, updated_at = %s "
                "WHERE singleton_key = 'global'",
                (row["user_id"], now),
            )
            event = self._append_task_event(connection, updated)
            request = deepcopy(row["inputs"] or {})
            request.update({
                "task_id": row["task_id"],
                "assistant_id": row["assistant_id"],
                "request_id": row["request_id"],
                "user_id": row["user_id"],
                "workspace_id": row["workspace_id"],
                "generation": int(row["generation"]),
                "attempt": token,
                "attempt_token": token,
                "worker_instance": worker_instance,
                "history": self._history_for_task(connection, row),
            })
            return {
                "task_id": row["task_id"],
                "attempt_token": token,
                "worker_instance": worker_instance,
                "lease_until": now + lease_seconds,
                "request": request,
                "event": event,
            }

    @staticmethod
    def _history_for_task(connection, task) -> list[dict]:
        inputs = task["inputs"] or {}
        if inputs.get("simulation", True):
            return []
        rows = connection.execute(
            "SELECT inputs, result FROM tasks WHERE user_id = %s AND assistant_id = %s "
            "AND generation = %s AND status = 'completed' AND task_id <> %s "
            "ORDER BY created_at, task_id",
            (task["user_id"], task["assistant_id"], task["generation"], task["task_id"]),
        ).fetchall()
        wanted_attachments = set(inputs.get("attachment_ids", []))
        wanted_plan = inputs.get("registered_plan_id")
        history = []
        for row in rows:
            previous_inputs = row["inputs"] or {}
            if previous_inputs.get("simulation", True):
                continue
            if set(previous_inputs.get("attachment_ids", [])) != wanted_attachments:
                continue
            if previous_inputs.get("registered_plan_id") != wanted_plan:
                continue
            result = deepcopy(row["result"])
            if result is not None:
                history.append({"question": previous_inputs.get("prompt", ""), "result": result})
        return history

    def execution_control(self, worker_instance: str, task_id: str,
                          attempt_token: str) -> dict:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        task_id = _required_text(task_id, "task_id", 128)
        attempt_token = _required_text(attempt_token, "attempt_token", 128)
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT status, cancel_requested FROM tasks WHERE task_id = %s "
                "AND worker_instance = %s AND attempt_token = %s",
                (task_id, worker_instance, attempt_token),
            ).fetchone()
            if row is None:
                raise StaleAttempt("task attempt is no longer current")
            return {"status": row["status"], "cancel_requested": bool(row["cancel_requested"])}

    def progress(self, worker_instance: str, task_id: str, attempt_token: str,
                 stage: str, detail: str = "") -> dict:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        task_id = _required_text(task_id, "task_id", 128)
        attempt_token = _required_text(attempt_token, "attempt_token", 128)
        if not isinstance(stage, str) or not isinstance(detail, str):
            raise ValueError("stage 与 detail 必须是字符串。")
        with self.db.transaction() as connection:
            metadata = connection.execute(
                "SELECT user_id, workspace_id FROM tasks WHERE task_id = %s "
                "AND worker_instance = %s AND attempt_token = %s",
                (task_id, worker_instance, attempt_token),
            ).fetchone()
            if metadata is None:
                raise StaleAttempt("task attempt is no longer current")
            self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
            row = self._lock_task(connection, metadata["user_id"], task_id)
            if row is None or row["worker_instance"] != worker_instance or row["attempt_token"] != attempt_token:
                raise StaleAttempt("task attempt is no longer current")
            if row["status"] not in ACTIVE_EXECUTION_STATUSES:
                raise StaleAttempt("task attempt is no longer active")
            updated = connection.execute(
                "UPDATE tasks SET stage = %s, detail = %s, updated_at = %s "
                "WHERE task_id = %s RETURNING *",
                (stage, detail, _now(), task_id),
            ).fetchone()
            event = self._append_task_event(connection, updated)
            return self._public_task(updated, event["seq"])

    def finish(self, worker_instance: str, task_id: str, attempt_token: str,
               status: str, *, result: Any = None, error: str | None = None) -> bool:
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        task_id = _required_text(task_id, "task_id", 128)
        attempt_token = _required_text(attempt_token, "attempt_token", 128)
        if status not in TERMINAL_TASK_STATUSES:
            raise ValueError("终态只允许 completed、failed 或 cancelled。")
        normalized_result = None
        encoded_result = None
        if result is not None:
            normalized_result, encoded_result = _json_normalize(result)
        if error is not None and not isinstance(error, str):
            raise ValueError("error 必须是字符串或 None。")

        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            metadata = connection.execute(
                "SELECT user_id, workspace_id, assistant_id FROM tasks WHERE task_id = %s",
                (task_id,),
            ).fetchone()
            if metadata is None:
                return False
            self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
            slot = self._lock_slot(connection, metadata["user_id"], metadata["assistant_id"])
            row = self._lock_task(connection, metadata["user_id"], task_id)
            if row is None or row["worker_instance"] != worker_instance or row["attempt_token"] != attempt_token:
                return False
            effective_status = "cancelled" if row["cancel_requested"] else status
            if row["status"] in TERMINAL_TASK_STATUSES:
                return row["status"] == effective_status
            if row["status"] not in ACTIVE_EXECUTION_STATUSES:
                return False
            # A reusable worker-scoped reader in active/idle state may remain
            # open, but uncertain cleanup or task-scoped resources retain slots.
            if self._task_has_unconfirmed_resources(connection, task_id):
                return False

            now = _now()
            connection.execute(
                "UPDATE process_resource_uses SET released_at = GREATEST(COALESCE(released_at, %s), acquired_at) "
                "WHERE worker_instance = %s AND task_id = %s AND attempt_token = %s",
                (now, worker_instance, task_id, attempt_token),
            )
            if effective_status == "completed":
                stage, detail = "completed", "任务已完成"
                final_result, final_error = encoded_result, None
            elif effective_status == "cancelled":
                stage, detail = "cancelled", "任务已取消"
                final_result, final_error = None, None
            else:
                stage, detail = "failed", "任务失败"
                final_result, final_error = None, error
            updated = connection.execute(
                "UPDATE tasks SET status = %s, stage = %s, detail = %s, result = %s::jsonb, "
                "error = %s, updated_at = %s, finished_at = %s, lease_until = NULL "
                "WHERE task_id = %s RETURNING *",
                (effective_status, stage, detail, final_result, final_error, now, now, task_id),
            ).fetchone()
            if slot["active_task_id"] == task_id:
                connection.execute(
                    "UPDATE assistant_slots SET active_task_id = NULL, updated_at = %s "
                    "WHERE user_id = %s AND assistant_id = %s AND active_task_id = %s",
                    (now, metadata["user_id"], metadata["assistant_id"], task_id),
                )
            else:
                raise RuntimeError("活动任务与 assistant_slots.active_task_id 不一致。")
            self._append_task_event(connection, updated)
            return True

    def mark_stale_workers(self, now: float | None = None) -> list[str]:
        requested_time = None if now is None else float(now)
        if requested_time is not None and not math.isfinite(requested_time):
            raise ValueError("now 必须是有限时间戳。")
        stale_instances: list[str] = []
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            checked_at = _now() if requested_time is None else requested_time
            stale_workers = connection.execute(
                "SELECT * FROM worker_instances WHERE state = 'active' "
                "AND heartbeat_at + lease_seconds <= %s "
                "ORDER BY worker_instance FOR UPDATE",
                (checked_at,),
            ).fetchall()
            for worker in stale_workers:
                instance = worker["worker_instance"]
                connection.execute(
                    "UPDATE worker_instances SET state = 'stale', stale_at = %s "
                    "WHERE worker_instance = %s",
                    (checked_at, instance),
                )
                tasks = connection.execute(
                    "SELECT task_id, user_id, workspace_id, assistant_id FROM tasks "
                    "WHERE worker_instance = %s AND status IN ('running', 'cancelling', 'recovering') "
                    "ORDER BY workspace_id, task_id",
                    (instance,),
                ).fetchall()
                for metadata in tasks:
                    self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
                    self._lock_slot(connection, metadata["user_id"], metadata["assistant_id"])
                    row = self._lock_task(connection, metadata["user_id"], metadata["task_id"])
                    if row is None or row["status"] not in ACTIVE_EXECUTION_STATUSES:
                        continue
                    if row["status"] == "recovering":
                        continue
                    updated = connection.execute(
                        "UPDATE tasks SET status = 'recovering', stage = 'recovering', detail = %s, "
                        "updated_at = %s WHERE task_id = %s RETURNING *",
                        ("执行 worker 租约过期，等待确认旧进程退出", checked_at, row["task_id"]),
                    ).fetchone()
                    self._append_task_event(connection, updated)
                stale_instances.append(instance)
        return stale_instances

    def recover_confirmed_exit(self, worker_instance: str,
                               process_start_token: str) -> list[str]:
        """Recover only tasks whose execution is proven to be local simulation.

        The repository verifies the persisted process-start token.  The local
        process probe must compare both PID and process_start_token before it
        calls this method; lease expiration alone is never sufficient proof.
        A dead parent worker does not prove that an external model request,
        Word process, or PDF child has stopped.  Those tasks stay recovering
        and retain both their user slot and global execution capacity.
        """
        worker_instance = _required_text(worker_instance, "worker_instance", 128)
        process_start_token = _required_text(process_start_token, "process_start_token", 255)
        recovered: list[str] = []
        with self.db.transaction() as connection:
            self._lock_scheduler(connection)
            worker = connection.execute(
                "SELECT * FROM worker_instances WHERE worker_instance = %s FOR UPDATE",
                (worker_instance,),
            ).fetchone()
            if worker is None:
                raise KeyError(worker_instance)
            if worker["process_start_token"] != process_start_token:
                raise WorkerIdentityConflict("进程启动标识不匹配，拒绝释放 worker 任务槽。")
            tasks = connection.execute(
                "SELECT task_id, user_id, workspace_id, assistant_id FROM tasks "
                "WHERE worker_instance = %s AND status IN ('running', 'cancelling', 'recovering') "
                "ORDER BY workspace_id, task_id",
                (worker_instance,),
            ).fetchall()
            for metadata in tasks:
                self._lock_workspace(connection, metadata["user_id"], metadata["workspace_id"])
                slot = self._lock_slot(connection, metadata["user_id"], metadata["assistant_id"])
                row = self._lock_task(connection, metadata["user_id"], metadata["task_id"])
                if (row is None or row["worker_instance"] != worker_instance
                        or row["status"] not in ACTIVE_EXECUTION_STATUSES):
                    continue
                inputs = row["inputs"] or {}
                if inputs.get("simulation") is not True:
                    now = _now()
                    updated = connection.execute(
                        "UPDATE tasks SET status = 'recovering', stage = 'recovering', detail = %s, "
                        "updated_at = %s, lease_until = NULL WHERE task_id = %s RETURNING *",
                        ("worker 父进程已确认退出，但模型调用或子进程状态未核验，任务槽仍保留",
                         now, row["task_id"]),
                    ).fetchone()
                    self._append_task_event(connection, updated)
                    continue
                if self._task_has_unconfirmed_recovery_resources(connection, row["task_id"]):
                    now = _now()
                    updated = connection.execute(
                        "UPDATE tasks SET status = 'recovering', stage = 'recovering', detail = %s, "
                        "updated_at = %s, lease_until = NULL WHERE task_id = %s RETURNING *",
                        ("旧worker已退出，但该任务仍关联未确认退出的资源；任务槽仍保留",
                         now, row["task_id"]),
                    ).fetchone()
                    self._append_task_event(connection, updated)
                    continue
                status = "cancelled" if row["cancel_requested"] else "failed"
                now = _now()
                detail = "旧执行进程已确认退出，任务已取消" if status == "cancelled" else "旧执行进程已确认退出，任务失败"
                updated = connection.execute(
                    "UPDATE tasks SET status = %s, stage = %s, detail = %s, error = %s, "
                    "updated_at = %s, finished_at = %s, lease_until = NULL, result = NULL "
                    "WHERE task_id = %s RETURNING *",
                    (status, status, detail,
                     None if status == "cancelled" else "worker process exited after lease expiry",
                     now, now, row["task_id"]),
                ).fetchone()
                if slot["active_task_id"] == row["task_id"]:
                    connection.execute(
                        "UPDATE assistant_slots SET active_task_id = NULL, updated_at = %s "
                        "WHERE user_id = %s AND assistant_id = %s AND active_task_id = %s",
                        (now, metadata["user_id"], metadata["assistant_id"], row["task_id"]),
                    )
                else:
                    raise RuntimeError("恢复任务与 assistant_slots.active_task_id 不一致。")
                connection.execute(
                    "UPDATE process_resource_uses SET released_at = GREATEST(COALESCE(released_at, %s), acquired_at) "
                    "WHERE worker_instance = %s AND task_id = %s AND attempt_token = %s",
                    (now, worker_instance, row["task_id"], row["attempt_token"]),
                )
                self._append_task_event(connection, updated)
                recovered.append(row["task_id"])
            connection.execute(
                "UPDATE worker_instances SET state = 'exited', exited_at = %s, stale_at = COALESCE(stale_at, %s) "
                "WHERE worker_instance = %s",
                (_now(), _now(), worker_instance),
            )
        return recovered
