"""Task-local guards at selected external request boundaries.

No guard is installed for standalone CLI/retrieval use. Worker threads install
their own guard for the lifetime of a runner, and always restore the context.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_request_control = ContextVar("rag_request_control", default=None)


@contextmanager
def request_control_scope(callback):
    if not callable(callback):
        raise TypeError("request control must be callable")
    token = _request_control.set(callback)
    try:
        yield
    finally:
        _request_control.reset(token)


def check_request_control():
    callback = _request_control.get()
    if callback is not None:
        callback()
