"""Database-backed task facade and bounded per-user resource cache.

The API process owns no task executor or runner.  Durable task state and event
history live in :class:`TaskRepository`; this module adapts that repository to
the service shape consumed by the existing FastAPI routes and SSE stream.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .task_service import ASSISTANTS


def _execution_modes(overrides: dict[str, str] | None) -> dict[str, str]:
    overrides = {} if overrides is None else overrides
    if (not isinstance(overrides, dict) or set(overrides) - set(ASSISTANTS)
            or any(mode not in ("model", "simulation") for mode in overrides.values())):
        raise ValueError("执行模式只能为已登记助手的model或simulation。")
    return {assistant: overrides.get(assistant, "simulation") for assistant in ASSISTANTS}


class DatabaseEventReader:
    """EventBus-compatible view over persisted events for one owner.

    Reads are synchronous database operations.  ``blocking_reads`` lets the
    async SSE adapter move them to FastAPI's worker pool.  Subscribers use only
    bounded event-loop timers: there is no polling thread per connection.
    """

    blocking_reads = True

    def __init__(self, repository, user_id: int, *, poll_interval: float = 0.5,
                 page_size: int = 200):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be a positive integer")
        self._repository = repository
        self._user_id = user_id
        self._poll_interval = poll_interval
        self._page_size = page_size
        self._lock = RLock()
        self._closed = False
        self._subscriptions: dict[int, dict[str, Any]] = {}
        self._next_subscription_id = 0

    @property
    def cursor(self) -> int:
        return self._repository.event_cursor(self._user_id)

    def read_after(self, cursor: int):
        return self._repository.events_after(self._user_id, cursor, limit=self._page_size)

    def subscribe(self, loop, callback):
        """Wake an SSE reader periodically so it can query persisted events."""
        with self._lock:
            if self._closed:
                raise RuntimeError("事件读取器已关闭。")
            self._next_subscription_id += 1
            subscription_id = self._next_subscription_id
            state = {"loop": loop, "callback": callback, "handle": None, "active": True}
            self._subscriptions[subscription_id] = state

        def tick():
            with self._lock:
                current = self._subscriptions.get(subscription_id)
                if current is None or not current["active"] or self._closed:
                    return
            try:
                callback()
            except Exception:
                # A disconnected consumer must not keep an event-loop callback
                # from rescheduling other active SSE readers.
                unsubscribe()
                return
            with self._lock:
                current = self._subscriptions.get(subscription_id)
                if current is not None and current["active"] and not self._closed:
                    current["handle"] = loop.call_later(self._poll_interval, tick)

        with self._lock:
            state["handle"] = loop.call_later(self._poll_interval, tick)

        def unsubscribe():
            with self._lock:
                current = self._subscriptions.pop(subscription_id, None)
                if current is None:
                    return
                current["active"] = False
                handle = current["handle"]
            if handle is not None:
                handle.cancel()

        return unsubscribe

    def close(self):
        """Release local SSE timers only; persisted tasks remain untouched."""
        with self._lock:
            self._closed = True
            states = list(self._subscriptions.values())
            self._subscriptions.clear()
        for state in states:
            state["active"] = False
            handle = state["handle"]
            if handle is not None:
                handle.cancel()


class DatabaseTaskService:
    """Owner-scoped facade matching the legacy task service's public methods."""

    def __init__(self, repository, *, user_id: int, workspace_id: str,
                 execution_modes: dict[str, str] | None = None,
                 model_name: str | None = None, history_limit: int = 50):
        if type(user_id) is not int or user_id < 1:
            raise ValueError("user_id须为正整数。")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("workspace_id不能为空。")
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("history_limit须为正整数。")
        self.repository = repository
        self.user_id = user_id
        self.workspace_id = workspace_id
        # This value is stable across API instances and process restarts.  The
        # event cursor is persisted per workspace and remains the source of truth.
        self.server_instance_id = f"persisted-{workspace_id}"
        self._execution_modes = _execution_modes(execution_modes)
        self.model_name = model_name
        self._history_limit = history_limit
        self._closed = False
        self.events = DatabaseEventReader(repository, user_id)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("任务服务已关闭。")

    def execution_mode(self, assistant_id: str) -> str:
        return self._execution_modes[assistant_id]

    def snapshot(self):
        self._ensure_open()
        # The repository provides one consistent snapshot/cursor read and must
        # include each active task in addition to the bounded recent history.
        value = self.repository.snapshot(
            self.user_id, self._execution_modes, self.model_name,
            limit=self._history_limit)
        value["server_instance_id"] = self.server_instance_id
        value["workspace_id"] = self.workspace_id
        value["owner_user_id"] = self.user_id
        return value

    def submit(self, assistant_id, request_id, prompt, *, duration_seconds=5,
               fail_at_end=False, attachment_ids=None, registered_plan_id=None,
               plan_action="question", use_embeddings=False, max_steps=None,
               max_tool_calls=None):
        self._ensure_open()
        if assistant_id not in ASSISTANTS:
            raise ValueError("未知助手。")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id不能为空。")
        if len(request_id.strip()) > 128:
            raise ValueError("request_id不能超过128个字符。")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请输入任务内容。")
        prompt = prompt.strip()
        if len(prompt) > 20000:
            raise ValueError("任务内容不能超过20000个字符。")

        simulation = self.execution_mode(assistant_id) == "simulation"
        if simulation:
            import math
            if (type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds)
                    or duration_seconds < 0):
                raise ValueError("演示时长须为有限非负数。")
            if type(fail_at_end) is not bool:
                raise ValueError("fail_at_end须为布尔值。")
            duration_seconds = float(duration_seconds)
        else:
            # Request-side demonstration controls never affect a model task.
            duration_seconds, fail_at_end = None, False

        if attachment_ids is None:
            attachment_ids = []
        if not isinstance(attachment_ids, (list, tuple)) or any(
                not isinstance(item, str) for item in attachment_ids):
            raise ValueError("附件列表格式无效。")
        inputs = {
            "prompt": prompt,
            "duration_seconds": duration_seconds,
            "fail_at_end": fail_at_end,
            "simulation": simulation,
            "attachment_ids": list(attachment_ids),
            "registered_plan_id": registered_plan_id,
            "plan_action": plan_action,
            "use_embeddings": bool(use_embeddings),
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
        }
        return self.repository.submit(
            self.user_id, assistant_id, request_id.strip(), inputs)

    def get_task(self, task_id):
        self._ensure_open()
        value = self.repository.get_task(self.user_id, task_id)
        if value is None:
            raise KeyError(task_id)
        return value

    def cancel(self, task_id):
        self._ensure_open()
        return self.repository.cancel(self.user_id, task_id)

    def reset(self, assistant_id):
        self._ensure_open()
        if assistant_id not in ASSISTANTS:
            raise ValueError("未知助手。")
        self.repository.reset(self.user_id, assistant_id)

    def close(self):
        """Close local readers without cancelling or changing any DB task."""
        if self._closed:
            return
        self._closed = True
        self.events.close()


@dataclass
class PersistentTenantWorkspace:
    user_id: int
    workspace_id: str
    tasks: DatabaseTaskService
    documents: object


class PersistentTenantRegistry:
    """LRU cache of per-user API facades and document stores.

    Eviction only drops this process's references.  It does not delete history,
    cancel a DB task, or signal a worker.  In-flight requests may still hold the
    evicted context until they finish.
    """

    def __init__(self, database, *, root, repository=None, pdf_backend=None,
                 execution_modes=None, model_name=None, max_cached_users=32,
                 document_factory=None, history_limit=50):
        if type(max_cached_users) is not int or max_cached_users < 1:
            raise ValueError("max_cached_users must be a positive integer")
        self.database = database
        self.root = Path(root).resolve()
        self.pdf_backend = pdf_backend
        self.execution_modes = _execution_modes(execution_modes)
        self.model_name = model_name
        self.max_cached_users = max_cached_users
        self.history_limit = history_limit
        self._document_factory = document_factory
        if repository is None:
            from .task_repository import TaskRepository
            repository = TaskRepository(database)
        self.repository = repository
        self._lock = RLock()
        self._closed = False
        self._users: OrderedDict[int, PersistentTenantWorkspace] = OrderedDict()

    def get(self, user_id: int) -> PersistentTenantWorkspace:
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
            raise ValueError("verified user_id must be a positive integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("用户工作区服务已关闭。")
            existing = self._users.get(user_id)
            if existing is not None:
                self._users.move_to_end(user_id)
                return existing

            workspace_id = self.database.ensure_workspace(user_id)
            if self._document_factory is None:
                from .documents import DocumentStore
                factory = DocumentStore
            else:
                factory = self._document_factory
            documents = factory(self.root, workspace_id, self.pdf_backend,
                                user_id=user_id, database=self.database)
            tasks = DatabaseTaskService(
                self.repository, user_id=user_id, workspace_id=workspace_id,
                execution_modes=self.execution_modes, model_name=self.model_name,
                history_limit=self.history_limit)
            context = PersistentTenantWorkspace(user_id, workspace_id, tasks, documents)
            self._users[user_id] = context

            if len(self._users) > self.max_cached_users:
                # Drop only the registry reference.  An in-flight HTTP request
                # can keep using its context, and the DB task remains worker-owned.
                self._users.popitem(last=False)
            return context

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            users = list(self._users.values())
            self._users.clear()
        for context in users:
            context.tasks.close()
            context.documents.close()
