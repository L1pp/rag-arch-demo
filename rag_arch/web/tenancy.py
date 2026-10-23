"""Verified per-user workspaces and their API-local resource caches."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from fastapi import HTTPException

from .persistent_tasks import PersistentTenantRegistry
from .task_service import TaskService


class TenantCapacityError(RuntimeError):
    pass


@dataclass
class TenantWorkspace:
    user_id: int
    workspace_id: str
    tasks: object
    documents: object


class TenantRegistry:
    """Legacy in-memory tenant registry for explicit tests and local previews."""

    def __init__(self, database, *, root, pdf_backend=None, runner_factory=None,
                 execution_modes=None, model_name=None, max_workers=6,
                 max_users=32, max_tasks_per_user=128, document_factory=None):
        for name, value in (("max_workers", max_workers), ("max_users", max_users),
                            ("max_tasks_per_user", max_tasks_per_user)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.database = database
        self.root = Path(root).resolve()
        self.pdf_backend = pdf_backend
        self.runner_factory = runner_factory
        self.execution_modes = execution_modes
        self.model_name = model_name
        self.max_users = max_users
        self.max_tasks_per_user = max_tasks_per_user
        self._document_factory = document_factory
        self._lock = RLock()
        self._closed = False
        self._users = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tenant-task")

    def get(self, user_id):
        if type(user_id) is not int or user_id < 1:
            raise ValueError("verified user_id must be a positive integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("用户工作区服务已关闭。")
            existing = self._users.get(user_id)
            if existing is not None:
                return existing
            if len(self._users) >= self.max_users:
                raise TenantCapacityError("当前工作区服务已达容量上限，请稍后再试。")
            # Initialization is short DB/filesystem work, never a model call.
            # Holding the lock prevents duplicate stores and orphan executors.
            workspace_id = self.database.ensure_workspace(user_id)
            if self._document_factory is None:
                from .documents import DocumentStore
                factory = DocumentStore
            else:
                factory = self._document_factory
            documents = factory(self.root, workspace_id, self.pdf_backend,
                                user_id=user_id, database=self.database)
            try:
                runner = self.runner_factory(documents) if self.runner_factory else None
                tasks = TaskService(
                    runner=runner, execution_modes=self.execution_modes,
                    model_name=self.model_name, user_id=user_id, workspace_id=workspace_id,
                    executor=self._executor, max_tasks=self.max_tasks_per_user,
                    archive_root=self.root / "data/runtime/web_demo/users" / str(user_id) / workspace_id)
            except BaseException:
                documents.close()
                raise
            context = TenantWorkspace(user_id, workspace_id, tasks, documents)
            self._users[user_id] = context
            return context

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            users = list(self._users.values())
        for context in users:
            context.tasks.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        for context in users:
            context.documents.close()


def current_tenant(request):
    registry = getattr(request.app.state, "tenants", None)
    if registry is None:
        return None  # Explicitly injected legacy services / unauthenticated local tests.
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise HTTPException(401, "请先登录。")
    context = getattr(request.state, "tenant", None)
    if context is None or context.user_id != principal.user_id:
        # Normal HTTP requests initialize off the event loop in middleware.
        context = registry.get(principal.user_id)
        request.state.tenant = context
    return context


def current_tasks(request):
    context = current_tenant(request)
    return context.tasks if context is not None else request.app.state.tasks


def current_documents(request):
    context = current_tenant(request)
    return context.documents if context is not None else request.app.state.documents


def check_workspace(request, workspace_id):
    tasks = current_tasks(request)
    if workspace_id != tasks.workspace_id:
        if getattr(tasks, "user_id", None) is not None:
            raise HTTPException(404, "未找到工作区。")
        raise HTTPException(409, "工作区已失效，请重新获取当前工作区。")
