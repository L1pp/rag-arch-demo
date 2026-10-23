"""Async execution with bounded local work and explicit worker ownership.

Network providers use native async APIs. ``run_local`` is only for CPU, file,
database and document operations whose synchronous implementation is retained.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import functools
import json
import threading


class TaskResourceCleanupError(RuntimeError):
    """Completed work whose owned idle clients could not all be released."""

    def __init__(self, component="task", *, exception_types=()):
        self.component = component if component in {"chat", "standards", "plan"} else "task"
        self.exception_types = [str(item)[:80] for item in exception_types
                                if str(item).replace("_", "").isalnum()]
        super().__init__(json.dumps(self.as_error(), ensure_ascii=False))

    def as_error(self):
        return {"component": self.component, "code": "resource_cleanup_failed",
                "message": "任务异步资源未能全部关闭，worker停止领取并等待已有任务结束。",
                "exception_types": self.exception_types}


async def run_local(function, /, *args, **kwargs):
    """Await local work; cancellation waits for the actual thread to return.

    Cancelling an executor Future cannot stop a Python thread. Shielding and
    draining it keeps a task's slot occupied until that work has really ended.
    The worker installs a bounded default executor on its persistent loop.
    """
    context = contextvars.copy_context()
    call = functools.partial(function, *args, **kwargs)
    future = asyncio.get_running_loop().run_in_executor(None, context.run, call)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # Repeated cancellation must not detach a still-running local job.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if future.done() and not future.cancelled():
            future.exception()  # consume any local error; cancellation wins
        raise


class AsyncTaskRuntime:
    """One owned event loop per worker; submissions preserve ContextVars."""

    def __init__(self, *, local_workers=6):
        if type(local_workers) is not int or local_workers < 1:
            raise ValueError("local_workers must be positive")
        self._local_workers = local_workers
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._closed = False
        self._loop = None
        self._thread = threading.Thread(target=self._serve, name="rag-async-worker", daemon=False)
        self._thread.start()
        self._ready.wait()

    def _serve(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_default_executor(ThreadPoolExecutor(
            max_workers=self._local_workers, thread_name_prefix="rag-local-work"))
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def submit(self, coroutine):
        with self._lock:
            if self._closed:
                coroutine.close()
                raise RuntimeError("async worker runtime is closed")
            if threading.current_thread() is self._thread:
                coroutine.close()
                raise RuntimeError("cannot synchronously wait on the worker event loop")
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def run(self, coroutine):
        return self.submit(coroutine).result()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
