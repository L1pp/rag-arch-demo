"""Single-user, in-memory tasks; one lock and one event cursor.

The frontend owns no running task. Archives are written only at termination;
old archives are never loaded and they are not a restart-recovery mechanism.
"""
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from threading import Event, RLock
import time
from uuid import uuid4

from .demo import run_demo


ASSISTANTS = {"chat": "聊天助手", "standards": "规范库问答", "plan": "方案辅助核查"}
TERMINAL = {"completed", "failed", "cancelled"}
DEFAULT_ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "data/runtime/web_demo"


class BusyError(RuntimeError):
    """The selected assistant already has an active task."""


class EventBus:
    """Publish state changes under the same lock used by TaskService."""
    def __init__(self, lock=None, capacity=2000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("事件缓冲容量须为正整数。")
        self._lock = lock if lock is not None else RLock()
        self._events = deque(maxlen=capacity)
        self._sequence = 0
        self._subscribers = {}

    @property
    def cursor(self):
        with self._lock:
            return self._sequence

    def publish(self, event_type, data):
        with self._lock:
            self._sequence += 1
            event = {"seq": self._sequence, "type": event_type, "data": deepcopy(data)}
            self._events.append(event)
            for key, (loop, notify) in list(self._subscribers.items()):
                try:
                    # Worker threads never touch an asyncio Event/Queue directly.
                    loop.call_soon_threadsafe(notify)
                except RuntimeError:
                    # A closed browser/server loop must not fail the task.
                    self._subscribers.pop(key, None)
            return deepcopy(event)

    def read_after(self, cursor):
        with self._lock:
            if type(cursor) is not int or cursor < 0:
                return [], "invalid_cursor"
            if cursor > self._sequence:
                return [], "cursor_ahead"
            if self._events and cursor < self._events[0]["seq"] - 1:
                return [], "cursor_expired"
            return [deepcopy(e) for e in self._events if e["seq"] > cursor], None

    def subscribe(self, loop, notify):
        key = uuid4().hex
        with self._lock:
            self._subscribers[key] = (loop, notify)
        def unsubscribe():
            with self._lock:
                self._subscribers.pop(key, None)
        return unsubscribe


@dataclass
class _Task:
    request: dict
    state: dict
    cancel_event: Event = field(default_factory=Event)
    future: object = None
    started_clock: float | None = None
    events: list[dict] = field(default_factory=list)


def _now():
    return datetime.now(timezone.utc).isoformat()


class ThreadPoolExecutorTaskService:
    def __init__(self, *, runner=None, archive_root=DEFAULT_ARCHIVE_ROOT, event_capacity=2000,
                 execution_modes: dict | None = None, model_name: str | None = None):
        overrides = {} if execution_modes is None else execution_modes
        if (not isinstance(overrides, dict) or set(overrides) - set(ASSISTANTS)
                or any(mode not in ("model", "simulation") for mode in overrides.values())):
            raise ValueError("执行模式只能为已登记助手的model或simulation。")
        self._execution_modes = {assistant: overrides.get(assistant, "simulation") for assistant in ASSISTANTS}
        if "model" in self._execution_modes.values() and runner is None:
            raise ValueError("真实模型模式须提供执行器，不能使用模拟执行器代替。")
        self.model_name = model_name
        self.workspace_id = uuid4().hex
        self.server_instance_id = uuid4().hex
        self._lock = RLock()
        self.events = EventBus(self._lock, event_capacity)
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="web-task")
        self._runner = runner or run_demo
        self._archive_root = Path(archive_root) if archive_root is not None else None
        self._closed = False
        self._tasks = {}
        self._requests = {}
        self._assistants = {key: {"assistant_id": key, "title": title,
                                 "execution_mode": self._execution_modes[key],
                                 "active_task_id": None, "tasks": []}
                            for key, title in ASSISTANTS.items()}

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("任务服务已关闭。")

    def _snapshot_task(self, task):
        value = deepcopy(task.state)
        if isinstance(value.get("result"), dict):
            # Keep detailed history in the worker registry/audit, not every SSE
            # progress snapshot. Frontend citations use the public compact list.
            for key in ("hits", "report_history"):
                value["result"].pop(key, None)
        if task.started_clock is not None and value["status"] not in TERMINAL:
            value["elapsed_seconds"] = round(max(0.0, time.monotonic() - task.started_clock), 3)
        return value

    def _emit_task(self, task):
        task.state["updated_seq"] = self.events.cursor + 1
        event = self.events.publish("task.updated", self._snapshot_task(task))
        task.events.append(event)

    def snapshot(self):
        with self._lock:
            self._ensure_open()
            assistants = []
            for assistant in self._assistants.values():
                assistants.append({**assistant, "tasks": [self._snapshot_task(self._tasks[tid])
                                                           for tid in assistant["tasks"]]})
            # State and cursor are captured atomically, so reconnect has no gap.
            return {"workspace_id": self.workspace_id, "server_instance_id": self.server_instance_id,
                    "cursor": self.events.cursor,
                    "mode": "chat_preview" if self._execution_modes["chat"] == "model" else "infrastructure_demo",
                    "model_name": self.model_name, "assistants": assistants}

    def execution_mode(self, assistant_id):
        return self._execution_modes[assistant_id]

    def submit(self, assistant_id, request_id, prompt, *, duration_seconds=5, fail_at_end=False,
               attachment_ids=None, registered_plan_id=None, plan_action="question",
               use_embeddings=False, max_steps=None, max_tool_calls=None):
        with self._lock:
            self._ensure_open()
            assistant = self._assistants[assistant_id]
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("request_id不能为空。")
            key = (assistant_id, request_id.strip())
            if key in self._requests:
                return self._snapshot_task(self._tasks[self._requests[key]])
            if assistant["active_task_id"] is not None:
                raise BusyError("这个助手已有任务在运行，请等待或先取消。")
            simulation = self._execution_modes[assistant_id] == "simulation"
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("请输入任务内容。")
            if simulation:
                if (type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds)
                        or duration_seconds < 0):
                    raise ValueError("演示时长须为有限非负数。")
                if type(fail_at_end) is not bool:
                    raise ValueError("fail_at_end须为布尔值。")
                duration_seconds = float(duration_seconds)
            else:
                # Request-side demonstration controls never affect a model task.
                duration_seconds, fail_at_end = None, False
            task_id = uuid4().hex
            request = {"task_id": task_id, "assistant_id": assistant_id, "request_id": key[1],
                       "prompt": prompt.strip(), "duration_seconds": duration_seconds,
                       "fail_at_end": fail_at_end, "simulation": simulation,
                       "attachment_ids": list(attachment_ids or []),
                       "registered_plan_id": registered_plan_id, "plan_action": plan_action,
                       "use_embeddings": bool(use_embeddings), "max_steps": max_steps,
                       "max_tool_calls": max_tool_calls}
            if not simulation:
                # Capture before reserving this task, under the state lock.
                # Only completed model answers in this assistant's current view
                # are history; retained records from before reset stay excluded.
                request["history"] = []
                for tid in assistant["tasks"]:
                    old = self._tasks[tid].state
                    same_documents = (set(old.get("attachment_ids", [])) == set(request["attachment_ids"])
                                      and old.get("registered_plan_id") == registered_plan_id)
                    if old["status"] == "completed" and not old["simulation"] and same_documents:
                        request["history"].append({"question": old["prompt"], "result": deepcopy(old["result"])})
            state = {k: request[k] for k in ("task_id", "assistant_id", "request_id", "prompt", "duration_seconds")}
            state.update({k: deepcopy(request[k]) for k in ("attachment_ids", "registered_plan_id",
                         "plan_action", "use_embeddings", "max_steps", "max_tool_calls")})
            state.update(status="queued", stage="queued",
                         detail="模拟任务等待执行" if simulation else "任务等待执行", created_at=_now(),
                         started_at=None, finished_at=None, elapsed_seconds=0.0,
                         result=None, error=None, simulation=simulation, updated_seq=0)
            task = _Task(request=request, state=state)
            self._tasks[task_id], self._requests[key] = task, task_id
            assistant["tasks"].append(task_id)
            assistant["active_task_id"] = task_id
            self._emit_task(task)
            # Reserve and submit while holding the lock; a concurrent duplicate
            # can never start a second task or steal the assistant slot.
            task.future = self._executor.submit(self._run, task_id)
            return self._snapshot_task(task)

    def get_task(self, task_id):
        with self._lock:
            self._ensure_open()
            return self._snapshot_task(self._tasks[task_id])

    def _progress(self, task, stage, detail=""):
        with self._lock:
            if task.cancel_event.is_set() or task.state["status"] in TERMINAL:
                raise CancelledError("任务已取消。")
            task.state.update(stage=str(stage), detail=str(detail))
            self._emit_task(task)

    def _finish(self, task, status, *, result=None, error=None):
        task.state.update(status=status, stage=status,
                          detail=("模拟任务" if task.state["simulation"] else "任务")
                                 + {"completed": "已完成", "failed": "失败", "cancelled": "已取消"}[status],
                          finished_at=_now(), result=deepcopy(result), error=error,
                          elapsed_seconds=round(max(0.0, time.monotonic() - task.started_clock), 3)
                          if task.started_clock is not None else 0.0)
        self._assistants[task.state["assistant_id"]]["active_task_id"] = None
        self._emit_task(task)

    def _run(self, task_id):
        with self._lock:
            task = self._tasks[task_id]
            if task.cancel_event.is_set():
                self._finish(task, "cancelled")
                skip = True
            else:
                task.started_clock = time.monotonic()
                task.state.update(status="running", stage="starting",
                                  detail="模拟任务已启动" if task.state["simulation"] else "任务已启动",
                                  started_at=_now())
                self._emit_task(task)
                skip = False
        if not skip:
            try:
                result = self._runner(deepcopy(task.request),
                                      lambda stage, detail="": self._progress(task, stage, detail), task.cancel_event)
                if isinstance(result, str):
                    result = {"text": result}
                if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                    raise ValueError("执行器须返回text字符串。")
                # Keep model/source/usage/audit metadata, but mode belongs to the
                # server. Reject non-JSON values before publishing any result.
                result = json.loads(json.dumps({**result, "simulation": task.state["simulation"]},
                                               ensure_ascii=False, allow_nan=False))
                with self._lock:
                    if task.cancel_event.is_set():
                        self._finish(task, "cancelled")
                    else:
                        self._finish(task, "completed", result=result)
            except CancelledError:
                with self._lock:
                    self._finish(task, "cancelled")
            except Exception as exc:
                with self._lock:
                    if task.cancel_event.is_set():
                        self._finish(task, "cancelled")
                    else:
                        self._finish(task, "failed", error=f"{type(exc).__name__}: {exc}")
        self._archive(task)

    def _cancel(self, task):
        if task.state["status"] in TERMINAL:
            return False
        task.cancel_event.set()
        if task.state["status"] == "queued" and task.future.cancel():
            self._finish(task, "cancelled")
            return True  # the cancelled Future will never call _run/_archive
        if task.state["status"] != "cancelling":
            task.state.update(status="cancelling", stage="cancelling",
                              detail="取消中，等待当前模拟阶段结束" if task.state["simulation"]
                              else "取消中，等待当前调用返回后停止")
            self._emit_task(task)
        return False

    def cancel(self, task_id):
        with self._lock:
            self._ensure_open()
            task = self._tasks[task_id]
            archive = self._cancel(task)
            result = self._snapshot_task(task)
        if archive:
            self._archive(task)
        return result

    def reset(self, assistant_id):
        with self._lock:
            self._ensure_open()
            assistant = self._assistants[assistant_id]
            if assistant["active_task_id"] is not None:
                raise BusyError("请先等待或取消当前任务，再清空这个助手。")
            # Retain task records for idempotency/download; clear only this view.
            assistant["tasks"].clear()
            self.events.publish("assistant.reset", {"assistant_id": assistant_id})

    def _archive(self, task):
        if self._archive_root is None:
            return
        with self._lock:
            events, snapshot = deepcopy(task.events), self._snapshot_task(task)
        try:
            directory = self._archive_root / self.server_instance_id / task.state["task_id"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "events.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8")
            (directory / "result.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            logging.getLogger(__name__).warning("任务归档失败（%s），内存结果仍保留。", type(exc).__name__)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            archives = [task for task in self._tasks.values() if self._cancel(task)]
        self._executor.shutdown(wait=False, cancel_futures=True)
        for task in archives:
            self._archive(task)


# Public compatibility name used by the Web app and existing callers.  Keep
# the implementation name explicit without breaking the stable import API.
TaskService = ThreadPoolExecutorTaskService
