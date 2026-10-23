"""Bounded, owner-scoped SSE replay over durable task events."""
from __future__ import annotations

import asyncio
import inspect
import math
import re
from threading import RLock
import time

from fastapi.sse import ServerSentEvent
from starlette.concurrency import run_in_threadpool


_MAX_EVENT_SEQ = (1 << 63) - 1
_DECIMAL_CURSOR = re.compile(r"[0-9]{1,19}\Z")


class SSEConnectionLimiter:
    """Bound SSE streams per authenticated owner and per API process."""

    def __init__(self, *, max_total: int = 128, max_per_owner: int = 4):
        if type(max_total) is not int or max_total < 1:
            raise ValueError("max_total must be a positive integer")
        if type(max_per_owner) is not int or max_per_owner < 1:
            raise ValueError("max_per_owner must be a positive integer")
        self.max_total = max_total
        self.max_per_owner = max_per_owner
        self._lock = RLock()
        self._leases: dict[object, object] = {}
        self._counts: dict[object, int] = {}

    def reserve(self, owner_key):
        if owner_key is None:
            raise ValueError("owner_key is required")
        with self._lock:
            if len(self._leases) >= self.max_total:
                return None
            if self._counts.get(owner_key, 0) >= self.max_per_owner:
                return None
            lease = object()
            self._leases[lease] = owner_key
            self._counts[owner_key] = self._counts.get(owner_key, 0) + 1
            return lease

    def release(self, lease) -> bool:
        with self._lock:
            owner_key = self._leases.pop(lease, None)
            if owner_key is None:
                return False
            count = self._counts[owner_key] - 1
            if count:
                self._counts[owner_key] = count
            else:
                del self._counts[owner_key]
            return True

    @property
    def active_total(self) -> int:
        with self._lock:
            return len(self._leases)

    def active_for(self, owner_key) -> int:
        with self._lock:
            return self._counts.get(owner_key, 0)


def _parse_cursor(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= _MAX_EVENT_SEQ else None
    if not isinstance(value, str) or not _DECIMAL_CURSOR.fullmatch(value):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed <= _MAX_EVENT_SEQ else None


def resume_cursor(after, last_event_id: str | None, instance: str):
    """Resolve a query cursor or native EventSource reconnect header.

    The browser's Last-Event-ID wins after reconnect.  Its instance component
    binds a sequence to one workspace stream; a cursor from another stream
    must refresh the workspace snapshot instead of replaying unrelated data.
    """
    value = _parse_cursor(after)
    fallback = value if value is not None else 0
    if last_event_id is None:
        return (value, None) if value is not None else (0, "invalid_cursor")

    previous_instance, separator, raw_number = last_event_id.rpartition(":")
    if not separator:
        return fallback, "invalid_cursor"
    value = _parse_cursor(raw_number)
    if value is None:
        return fallback, "invalid_cursor"
    if previous_instance != instance:
        return fallback, "server_restarted"
    return value, None


async def _session_status(session_check):
    if session_check is None:
        return True
    try:
        result = session_check()
        if inspect.isawaitable(result):
            result = await result
    except asyncio.CancelledError:
        raise
    except Exception:
        # A failed identity check must never authorize event delivery.  Let the
        # native EventSource reconnect and let the HTTP layer retry the check.
        return None
    if result is True:
        return True
    if result is False:
        return False
    return None


def _snapshot_required(service, reason: str) -> ServerSentEvent:
    return ServerSentEvent(event="snapshot_required", data={
        "reason": reason,
        "server_instance_id": service.server_instance_id,
    })


async def event_stream(service, after, *, last_event_id=None, session_check=None,
                       session_recheck_interval=0.5, monotonic=None):
    """Replay persisted events in bounded pages and poll without per-client threads.

    ``session_check`` returns True for the same authenticated owner, False for
    a revoked/changed identity, and None (or raises) when identity storage is
    unavailable.  The latter two states fail closed; only a known revocation
    emits ``auth_required``.  Checks happen on each poll and again immediately
    before sending each fetched batch. While a client drains a slow page, the
    monotonic recheck interval also bounds how long it can receive after revoke.
    """
    cursor, reason = resume_cursor(after, last_event_id, service.server_instance_id)
    if reason:
        yield _snapshot_required(service, reason)
        return

    if (isinstance(session_recheck_interval, bool)
            or not isinstance(session_recheck_interval, (int, float))
            or not math.isfinite(session_recheck_interval)
            or session_recheck_interval <= 0):
        raise ValueError("session_recheck_interval must be positive")
    monotonic = time.monotonic if monotonic is None else monotonic
    if not callable(monotonic):
        raise ValueError("monotonic must be callable")

    signal = asyncio.Event()
    unsubscribe = service.events.subscribe(asyncio.get_running_loop(), signal.set)
    session_check_interval = float(session_recheck_interval)
    last_session_check = None

    async def validate_session(*, force=False):
        nonlocal last_session_check
        if session_check is None:
            return True
        now = monotonic()
        if (not force and last_session_check is not None
                and now - last_session_check < session_check_interval):
            return True
        status = await _session_status(session_check)
        last_session_check = monotonic()
        return status

    try:
        while True:
            # Clear before DB reads.  A timer firing during a read leaves the
            # signal set, so the next wait cannot lose that wake-up.
            signal.clear()
            status = await validate_session(force=True)
            if status is False:
                yield ServerSentEvent(event="auth_required")
                return
            if status is None:
                return

            try:
                if getattr(service.events, "blocking_reads", False):
                    events, reason = await run_in_threadpool(
                        service.events.read_after, cursor)
                else:
                    events, reason = service.events.read_after(cursor)
            except asyncio.CancelledError:
                raise
            except Exception:
                yield _snapshot_required(service, "event_store_unavailable")
                return

            if reason:
                yield _snapshot_required(service, reason)
                return

            if events:
                # A session can be revoked while the event query is in flight.
                # Revalidate the original cookie token before exposing the page.
                status = await validate_session(force=True)
                if status is False:
                    yield ServerSentEvent(event="auth_required")
                    return
                if status is None:
                    return
                for item in events:
                    status = await validate_session()
                    if status is False:
                        yield ServerSentEvent(event="auth_required")
                        return
                    if status is None:
                        return
                    cursor = item["seq"]
                    yield ServerSentEvent(
                        event=item["type"],
                        id=f"{service.server_instance_id}:{cursor}",
                        data=item["data"],
                    )
            else:
                await signal.wait()
    finally:
        unsubscribe()
