"""Stable web identities and compatible SQLite/PostgreSQL session stores."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
import math
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
import unicodedata


SESSION_COOKIE_NAME = "rag_arch_session"
DEFAULT_PASSWORD_ITERATIONS = 310_000
DEFAULT_SESSION_TTL_SECONDS = 14 * 24 * 60 * 60
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 1024
_LOGIN_LIMIT_PRUNE_LOCK_NAMESPACE = 0x4C494D49  # "LIMI", separate from the migration lock namespace.


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Stable authenticated identity; usernames are display/login names, not owner keys."""

    user_id: int
    username: str
    credential_version: bytes | None = field(default=None, repr=False, compare=False)


def default_auth_db_path(root: str | Path, override: str | Path | None = None) -> Path:
    """Return the shared runtime path used by the app and account CLI."""
    configured = override if override is not None else os.environ.get("RAG_ARCH_AUTH_DB")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
    else:
        path = Path(root) / "data" / "runtime" / "auth.sqlite3"
    return path.resolve()


def normalize_username(username: str) -> str:
    if not isinstance(username, str):
        raise ValueError("用户名不能为空。")
    value = unicodedata.normalize("NFKC", username.strip())
    if not value or len(value) > 64 or any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("用户名需为 1–64 个字符，且不能包含控制字符。")
    return value


def ascii_username_key(username: str) -> str:
    """Normalize once, then match SQLite NOCASE's ASCII-only case folding."""
    return ascii_username_key_from_normalized(normalize_username(username))


def ascii_username_key_from_normalized(value: str) -> str:
    """Fold an already normalized/stored username without normalizing it again."""
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("用户名需为 1–64 个字符。")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("用户名不能包含控制字符。")
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value)


def validate_password(password: str) -> str:
    if not isinstance(password, str):
        raise ValueError("密码不能为空。")
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise ValueError(f"密码长度需为 {MIN_PASSWORD_LENGTH}–{MAX_PASSWORD_LENGTH} 个字符。")
    if len(password.encode("utf-8")) > 4096:
        raise ValueError("密码过长。")
    return password


class AuthStore:
    """SQLite users/sessions with PBKDF2 password hashes and revocable tokens."""

    def __init__(self, db_path: str | Path, *,
                 session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
                 password_iterations: int = DEFAULT_PASSWORD_ITERATIONS):
        if session_ttl_seconds < 1:
            raise ValueError("session_ttl_seconds must be positive")
        if password_iterations < 1:
            raise ValueError("password_iterations must be positive")
        self.db_path = Path(db_path).expanduser().resolve()
        self.session_ttl_seconds = int(session_ttl_seconds)
        self.password_iterations = int(password_iterations)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._dummy_salt = secrets.token_bytes(16)
        self._dummy_hash = self._derive("rag-arch-invalid-user", self._dummy_salt,
                                        self.password_iterations)

    @staticmethod
    def _derive(password: str, salt: bytes, iterations: int) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY,
                        username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                        password_salt BLOB NOT NULL,
                        password_hash BLOB NOT NULL,
                        password_iterations INTEGER NOT NULL,
                        enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        token_hash TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
                    CREATE INDEX IF NOT EXISTS sessions_expiry_idx ON sessions(expires_at);
                    """
                )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    def create_user(self, username: str, password: str) -> None:
        self._ensure_initialized()
        name = normalize_username(username)
        secret = validate_password(password)
        salt = secrets.token_bytes(16)
        digest = self._derive(secret, salt, self.password_iterations)
        now = time.time()
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO users(username, password_salt, password_hash, password_iterations, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, salt, digest, self.password_iterations, now, now),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError("用户名已存在。") from exc
        finally:
            connection.close()

    def verify_identity(self, username: str, password: str) -> UserIdentity | None:
        self._ensure_initialized()
        try:
            name = normalize_username(username)
            secret = validate_password(password)
        except ValueError:
            # Keep malformed and unknown names on the same password-hash path.
            name = ""
            secret = password if isinstance(password, str) and len(password) <= MAX_PASSWORD_LENGTH else ""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id, username, password_salt, password_hash, password_iterations, enabled "
                "FROM users WHERE username = ? COLLATE NOCASE",
                (name,),
            ).fetchone()
        finally:
            connection.close()
        salt = bytes(row["password_salt"]) if row is not None else self._dummy_salt
        expected = bytes(row["password_hash"]) if row is not None else self._dummy_hash
        iterations = int(row["password_iterations"]) if row is not None else self.password_iterations
        candidate = self._derive(secret, salt, iterations)
        if row is None or not row["enabled"] or not hmac.compare_digest(candidate, expected):
            return None
        return UserIdentity(user_id=int(row["id"]), username=str(row["username"]),
                            credential_version=bytes(row["password_hash"]))

    def verify_user(self, username: str, password: str) -> str | None:
        """Compatibility wrapper returning the verified username."""
        identity = self.verify_identity(username, password)
        return identity.username if identity is not None else None

    def create_session(self, username: str | UserIdentity) -> tuple[str, int]:
        self._ensure_initialized()
        identity = username if isinstance(username, UserIdentity) else None
        name = None if identity is not None else normalize_username(username)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        now = time.time()
        expires_at = now + self.session_ttl_seconds
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if identity is not None:
                if identity.credential_version is None:
                    raise ValueError("身份已失效，请重新登录。")
                row = connection.execute(
                    "SELECT id FROM users WHERE id = ? AND enabled = 1 AND password_hash = ?",
                    (identity.user_id, identity.credential_version),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT id FROM users WHERE username = ? COLLATE NOCASE AND enabled = 1", (name,)
                ).fetchone()
            if row is None:
                if identity is not None:
                    raise ValueError("身份已失效，请重新登录。")
                raise ValueError("用户不存在或已停用。")
            connection.execute("DELETE FROM sessions WHERE user_id = ? AND expires_at <= ?",
                               (row["id"], now))
            connection.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash, row["id"], now, expires_at),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return token, int(expires_at)

    def resolve_identity(self, token: str | None) -> UserIdentity | None:
        self._ensure_initialized()
        if not token or len(token) > 256:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT users.id, users.username, users.enabled, sessions.expires_at "
                "FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if not row["enabled"] or row["expires_at"] <= now:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                connection.commit()
                return None
            return UserIdentity(user_id=int(row["id"]), username=str(row["username"]))
        finally:
            connection.close()

    def resolve_session(self, token: str | None) -> str | None:
        """Compatibility wrapper returning the resolved username."""
        identity = self.resolve_identity(token)
        return identity.username if identity is not None else None

    def revoke_session(self, token: str | None) -> None:
        self._ensure_initialized()
        if not token or len(token) > 256:
            return
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        connection = self._connect()
        try:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            connection.commit()
        finally:
            connection.close()

    def disable_user(self, username: str) -> bool:
        self._ensure_initialized()
        name = normalize_username(username)
        connection = self._connect()
        try:
            row = connection.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (name,)).fetchone()
            if row is None:
                return False
            connection.execute("UPDATE users SET enabled = 0, updated_at = ? WHERE id = ?", (time.time(), row["id"]))
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            connection.commit()
            return True
        finally:
            connection.close()

    def set_password(self, username: str, password: str) -> bool:
        self._ensure_initialized()
        name = normalize_username(username)
        secret = validate_password(password)
        salt = secrets.token_bytes(16)
        digest = self._derive(secret, salt, self.password_iterations)
        connection = self._connect()
        try:
            row = connection.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (name,)).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE users SET password_salt = ?, password_hash = ?, password_iterations = ?, updated_at = ? WHERE id = ?",
                (salt, digest, self.password_iterations, time.time(), row["id"]),
            )
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            connection.commit()
            return True
        finally:
            connection.close()

    def list_users(self) -> list[dict[str, object]]:
        self._ensure_initialized()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT username, enabled, created_at FROM users ORDER BY username COLLATE NOCASE"
            ).fetchall()
            return [{"username": str(row["username"]), "enabled": bool(row["enabled"]),
                     "created_at": float(row["created_at"])} for row in rows]
        finally:
            connection.close()


class PostgresAuthStore:
    """PostgreSQL-backed implementation using the shared Database adapter.

    Construction does not connect or migrate. Call ``migrate()`` explicitly before
    using a new database; the factory never creates or imports a database implicitly.
    """

    def __init__(self, database, *, session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
                 password_iterations: int = DEFAULT_PASSWORD_ITERATIONS):
        if session_ttl_seconds < 1:
            raise ValueError("session_ttl_seconds must be positive")
        if password_iterations < 1:
            raise ValueError("password_iterations must be positive")
        self.database = database
        self.session_ttl_seconds = int(session_ttl_seconds)
        self.password_iterations = int(password_iterations)
        self._dummy_salt = secrets.token_bytes(16)
        self._dummy_hash = AuthStore._derive("rag-arch-invalid-user", self._dummy_salt,
                                             self.password_iterations)

    def migrate(self) -> None:
        """Explicitly run the Database adapter's schema migrations."""
        self.database.migrate()

    @staticmethod
    def _integrity_value(exc: Exception) -> str | None:
        return getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)

    def _fetchone(self, query: str, params: tuple[object, ...]):
        connection = self.database.connect()
        try:
            return connection.execute(query, params).fetchone()
        finally:
            connection.close()

    def create_user(self, username: str, password: str) -> None:
        name = normalize_username(username)
        key = ascii_username_key_from_normalized(name)
        secret = validate_password(password)
        salt = secrets.token_bytes(16)
        digest = AuthStore._derive(secret, salt, self.password_iterations)
        now = time.time()
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO users(username, username_key, password_salt, password_hash, password_iterations, "
                    "created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (name, key, salt, digest, self.password_iterations, now, now),
                )
        except Exception as exc:
            if self._integrity_value(exc) == "23505":
                raise ValueError("用户名已存在。") from exc
            raise

    def verify_identity(self, username: str, password: str) -> UserIdentity | None:
        try:
            name = normalize_username(username)
            key = ascii_username_key_from_normalized(name)
            secret = validate_password(password)
        except ValueError:
            key = ""
            secret = password if isinstance(password, str) and len(password) <= MAX_PASSWORD_LENGTH else ""
        row = self._fetchone(
            "SELECT id, username, password_salt, password_hash, password_iterations, enabled "
            "FROM users WHERE username_key = %s",
            (key,),
        )
        salt = bytes(row["password_salt"]) if row is not None else self._dummy_salt
        expected = bytes(row["password_hash"]) if row is not None else self._dummy_hash
        iterations = int(row["password_iterations"]) if row is not None else self.password_iterations
        candidate = AuthStore._derive(secret, salt, iterations)
        if row is None or not row["enabled"] or not hmac.compare_digest(candidate, expected):
            return None
        return UserIdentity(user_id=int(row["id"]), username=str(row["username"]),
                            credential_version=bytes(row["password_hash"]))

    def verify_user(self, username: str, password: str) -> str | None:
        """Compatibility wrapper returning the verified username."""
        identity = self.verify_identity(username, password)
        return identity.username if identity is not None else None

    def create_session(self, username: str | UserIdentity) -> tuple[str, int]:
        identity = username if isinstance(username, UserIdentity) else None
        key = None if identity is not None else ascii_username_key(username)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        now = time.time()
        expires_at = now + self.session_ttl_seconds
        with self.database.transaction() as connection:
            if identity is not None:
                if identity.credential_version is None:
                    raise ValueError("身份已失效，请重新登录。")
                row = connection.execute(
                    "SELECT id FROM users WHERE id = %s AND enabled = TRUE AND password_hash = %s FOR UPDATE",
                    (identity.user_id, identity.credential_version),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT id FROM users WHERE username_key = %s AND enabled = TRUE FOR UPDATE", (key,)
                ).fetchone()
            if row is None:
                if identity is not None:
                    raise ValueError("身份已失效，请重新登录。")
                raise ValueError("用户不存在或已停用。")
            connection.execute("DELETE FROM sessions WHERE user_id = %s AND expires_at <= %s",
                               (row["id"], now))
            connection.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (%s, %s, %s, %s)",
                (token_hash, row["id"], now, expires_at),
            )
        return token, int(expires_at)

    def resolve_identity(self, token: str | None) -> UserIdentity | None:
        if not token or len(token) > 256:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        row = self._fetchone(
            "SELECT users.id, users.username, users.enabled, sessions.expires_at "
            "FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token_hash = %s",
            (token_hash,),
        )
        if row is None:
            return None
        if not row["enabled"] or row["expires_at"] <= now:
            self.revoke_session(token)
            return None
        return UserIdentity(user_id=int(row["id"]), username=str(row["username"]))

    def resolve_session(self, token: str | None) -> str | None:
        """Compatibility wrapper returning the resolved username."""
        identity = self.resolve_identity(token)
        return identity.username if identity is not None else None

    def revoke_session(self, token: str | None) -> None:
        if not token or len(token) > 256:
            return
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = %s", (token_hash,))

    def disable_user(self, username: str) -> bool:
        key = ascii_username_key(username)
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT id FROM users WHERE username_key = %s", (key,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("UPDATE users SET enabled = FALSE, updated_at = %s WHERE id = %s",
                               (time.time(), row["id"]))
            connection.execute("DELETE FROM sessions WHERE user_id = %s", (row["id"],))
            return True

    def set_password(self, username: str, password: str) -> bool:
        key = ascii_username_key(username)
        secret = validate_password(password)
        salt = secrets.token_bytes(16)
        digest = AuthStore._derive(secret, salt, self.password_iterations)
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT id FROM users WHERE username_key = %s", (key,)
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE users SET password_salt = %s, password_hash = %s, password_iterations = %s, "
                "updated_at = %s WHERE id = %s",
                (salt, digest, self.password_iterations, time.time(), row["id"]),
            )
            connection.execute("DELETE FROM sessions WHERE user_id = %s", (row["id"],))
            return True

    def list_users(self) -> list[dict[str, object]]:
        connection = self.database.connect()
        try:
            rows = connection.execute(
                "SELECT username, enabled, created_at FROM users ORDER BY username_key"
            ).fetchall()
            return [{"username": str(row["username"]), "enabled": bool(row["enabled"]),
                     "created_at": float(row["created_at"])} for row in rows]
        finally:
            connection.close()


def create_auth_store(root: str | Path, override: str | Path | None = None, *,
                      session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
                      password_iterations: int = DEFAULT_PASSWORD_ITERATIONS):
    """Create an auth store without connecting to or migrating a configured database.

    An explicit ``override`` selects SQLite. Otherwise ``RAG_ARCH_DATABASE_URL``
    selects PostgreSQL, with the legacy ``RAG_ARCH_AUTH_DB`` path as fallback.
    """
    if override is not None:
        return AuthStore(default_auth_db_path(root, override),
                         session_ttl_seconds=session_ttl_seconds,
                         password_iterations=password_iterations)
    database_url = os.environ.get("RAG_ARCH_DATABASE_URL")
    if database_url:
        from .db import Database
        database_schema = os.environ.get("RAG_ARCH_DATABASE_SCHEMA", "public")
        return PostgresAuthStore(Database(database_url, schema=database_schema),
                                 session_ttl_seconds=session_ttl_seconds,
                                 password_iterations=password_iterations)
    return AuthStore(default_auth_db_path(root),
                     session_ttl_seconds=session_ttl_seconds,
                     password_iterations=password_iterations)


class LoginRateLimiter:
    """Process-local sliding-window limiter for failed login attempts."""

    def __init__(self, *, max_attempts: int = 10, window_seconds: int = 300):
        if max_attempts < 1 or window_seconds < 1:
            raise ValueError("rate-limit settings must be positive")
        self.max_attempts = int(max_attempts)
        self.window_seconds = int(window_seconds)
        self._failures: dict[str, deque[float]] = {}
        self._inflight: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def _prune(self, failures: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while failures and failures[0] <= cutoff:
            failures.popleft()

    def is_limited(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        current = time.time() if now is None else now
        with self._lock:
            failures = self._failures.get(key)
            if failures is not None:
                self._prune(failures, current)
            if failures is not None and not failures:
                self._failures.pop(key, None)
                failures = None
            inflight = self._inflight.get(key)
            failure_count = len(failures) if failures is not None else 0
            inflight_count = len(inflight) if inflight is not None else 0
            if failure_count + inflight_count < self.max_attempts:
                return False, 0
            retry_after = self._retry_after(failures, current)
            return True, retry_after

    def _retry_after(self, failures: deque[float] | None, now: float) -> int:
        if not failures:
            return 1
        return max(1, int(self.window_seconds - (now - failures[0])))

    def begin_attempt(self, key: str, *, now: float | None = None) -> tuple[str | None, int]:
        """Atomically reserve one rate-limit slot before password verification."""
        current = time.time() if now is None else now
        with self._lock:
            failures = self._failures.get(key)
            if failures is not None:
                self._prune(failures, current)
                if not failures:
                    self._failures.pop(key, None)
                    failures = None
            inflight = self._inflight.get(key)
            failure_count = len(failures) if failures is not None else 0
            inflight_count = len(inflight) if inflight is not None else 0
            if failure_count + inflight_count >= self.max_attempts:
                return None, self._retry_after(failures, current)
            attempt_id = secrets.token_urlsafe(18)
            self._inflight.setdefault(key, set()).add(attempt_id)
            return attempt_id, 0

    def finish_attempt(self, key: str, attempt_id: str, success: bool | None,
                       *, now: float | None = None) -> bool:
        """Release a reservation once, optionally recording failure or success.

        Repeated calls and unknown attempt IDs are harmless and return ``False``.
        ``None`` only releases the in-flight slot, for errors outside credential
        verification where the limiter must not count a failed password.
        """
        if success is not None and type(success) is not bool:
            raise ValueError("success must be bool or None")
        current = time.time() if now is None else now
        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is None or attempt_id not in inflight:
                return False
            inflight.remove(attempt_id)
            if not inflight:
                self._inflight.pop(key, None)
            if success is True:
                self._failures.pop(key, None)
            elif success is False:
                failures = self._failures.setdefault(key, deque())
                self._prune(failures, current)
                failures.append(current)
            return True

    def failed(self, key: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self._lock:
            failures = self._failures.setdefault(key, deque())
            self._prune(failures, current)
            failures.append(current)

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class PostgresLoginRateLimiter(LoginRateLimiter):
    """PostgreSQL-shared sliding-window limiter for multiple API processes.

    The public methods intentionally match ``LoginRateLimiter``. Keys and
    reservation tickets are stored only as SHA-256 digests. Each operation
    serializes on a transaction-scoped advisory lock derived from the key,
    then prunes, checks, and mutates rows in one database transaction.
    """

    def __init__(self, database, *, max_attempts: int = 10, window_seconds: int = 300):
        super().__init__(max_attempts=max_attempts, window_seconds=window_seconds)
        self.database = database
        self._maintenance_lock = threading.Lock()
        self._last_global_prune_monotonic = 0.0

    @staticmethod
    def _key_hash(key: str) -> str:
        if not isinstance(key, str):
            raise ValueError("rate-limit key must be a string")
        try:
            encoded = key.encode("utf-8")
        except UnicodeError:
            raise ValueError("rate-limit key must be UTF-8 text") from None
        return hashlib.sha256(b"rag-arch-login-key-v1\0" + encoded).hexdigest()

    @staticmethod
    def _ticket_hash(ticket: str) -> str:
        if not isinstance(ticket, str):
            return ""
        try:
            encoded = ticket.encode("utf-8")
        except UnicodeError:
            return ""
        return hashlib.sha256(b"rag-arch-login-ticket-v1\0" + encoded).hexdigest()

    @staticmethod
    def _advisory_lock_id(key_hash: str) -> int:
        # A 64-bit digest collision only serializes unrelated keys; it cannot
        # merge their rows or weaken their independent limits.
        digest = hashlib.sha256(bytes.fromhex(key_hash)).digest()
        return int.from_bytes(digest[:8], "big", signed=True)

    @staticmethod
    def _now(connection, now: float | None) -> float:
        if now is not None:
            return float(now)
        row = connection.execute(
            "SELECT EXTRACT(EPOCH FROM clock_timestamp()) AS now"
        ).fetchone()
        return float(row["now"])

    @staticmethod
    def _acquire_key_lock(connection, key_hash: str) -> None:
        # Shared namespace lock lets ordinary keys proceed concurrently while
        # an occasional exclusive transaction removes expired rows globally.
        connection.execute(
            "SELECT pg_advisory_xact_lock_shared(%s, %s)",
            (_LOGIN_LIMIT_PRUNE_LOCK_NAMESPACE, 0),
        )
        connection.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (PostgresLoginRateLimiter._advisory_lock_id(key_hash),),
        )

    @staticmethod
    def _validate_now(now: float | None) -> None:
        if now is None:
            return
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError("now must be a finite timestamp")
        try:
            finite = math.isfinite(now)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise ValueError("now must be a finite timestamp")

    def _maybe_prune_expired(self, *, explicit_now: bool) -> None:
        """Periodically clear abandoned rows without racing any key operation."""
        if explicit_now:
            return
        current_monotonic = time.monotonic()
        with self._maintenance_lock:
            if current_monotonic - self._last_global_prune_monotonic < 60:
                return
            try:
                with self.database.transaction() as connection:
                    # This exclusive transaction lock waits until every normal
                    # operation has committed, then blocks new ones until both
                    # expiry sweeps commit.
                    connection.execute(
                        "SELECT pg_advisory_xact_lock(%s, %s)",
                        (_LOGIN_LIMIT_PRUNE_LOCK_NAMESPACE, 0),
                    )
                    current = self._now(connection, None)
                    connection.execute(
                        "DELETE FROM login_rate_limit_failures WHERE failed_at <= %s",
                        (current - self.window_seconds,),
                    )
                    connection.execute(
                        "DELETE FROM login_rate_limit_attempts WHERE expires_at <= %s",
                        (current,),
                    )
                self._last_global_prune_monotonic = current_monotonic
            except Exception:
                self._raise_unavailable()

    def _prune_locked(self, connection, key_hash: str, now: float) -> None:
        cutoff = now - self.window_seconds
        connection.execute(
            "DELETE FROM login_rate_limit_failures WHERE key_hash = %s AND failed_at <= %s",
            (key_hash, cutoff),
        )
        connection.execute(
            "DELETE FROM login_rate_limit_attempts WHERE key_hash = %s AND expires_at <= %s",
            (key_hash, now),
        )

    @staticmethod
    def _state_locked(connection, key_hash: str) -> tuple[int, int, float | None, float | None]:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM login_rate_limit_failures WHERE key_hash = %s) AS failure_count,
                (SELECT COUNT(*) FROM login_rate_limit_attempts WHERE key_hash = %s) AS inflight_count,
                (SELECT MIN(failed_at) FROM login_rate_limit_failures WHERE key_hash = %s) AS first_failure,
                (SELECT MIN(expires_at) FROM login_rate_limit_attempts WHERE key_hash = %s) AS first_attempt_expiry
            """,
            (key_hash, key_hash, key_hash, key_hash),
        ).fetchone()
        return (
            int(row["failure_count"]),
            int(row["inflight_count"]),
            None if row["first_failure"] is None else float(row["first_failure"]),
            None if row["first_attempt_expiry"] is None else float(row["first_attempt_expiry"]),
        )

    def _retry_after(self, first_failure: float | None, first_attempt_expiry: float | None,
                     now: float) -> int:
        expiries = []
        if first_failure is not None:
            expiries.append(first_failure + self.window_seconds)
        if first_attempt_expiry is not None:
            expiries.append(first_attempt_expiry)
        if not expiries:
            return 1
        remaining = min(expiries) - now
        # A Retry-After value must not tell a client to retry before the
        # oldest slot actually expires. Small epsilon offsets float storage.
        return max(1, math.ceil(remaining - 1e-6))

    @staticmethod
    def _raise_unavailable() -> None:
        # Do not expose database errors, connection strings, identifiers, or
        # ticket material through a login error response or its exception text.
        raise RuntimeError("登录限流存储不可用。") from None

    def is_limited(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        self._validate_now(now)
        key_hash = self._key_hash(key)
        self._maybe_prune_expired(explicit_now=now is not None)
        try:
            with self.database.transaction() as connection:
                self._acquire_key_lock(connection, key_hash)
                current = self._now(connection, now)
                self._prune_locked(connection, key_hash, current)
                failures, inflight, first_failure, first_expiry = self._state_locked(connection, key_hash)
                if failures + inflight < self.max_attempts:
                    return False, 0
                return True, self._retry_after(first_failure, first_expiry, current)
        except Exception:
            self._raise_unavailable()

    def begin_attempt(self, key: str, *, now: float | None = None) -> tuple[str | None, int]:
        """Reserve one shared slot before password verification."""
        self._validate_now(now)
        key_hash = self._key_hash(key)
        ticket = secrets.token_urlsafe(18)
        ticket_hash = self._ticket_hash(ticket)
        self._maybe_prune_expired(explicit_now=now is not None)
        try:
            with self.database.transaction() as connection:
                self._acquire_key_lock(connection, key_hash)
                current = self._now(connection, now)
                self._prune_locked(connection, key_hash, current)
                failures, inflight, first_failure, first_expiry = self._state_locked(connection, key_hash)
                if failures + inflight >= self.max_attempts:
                    return None, self._retry_after(first_failure, first_expiry, current)
                connection.execute(
                    """
                    INSERT INTO login_rate_limit_attempts(key_hash, ticket_hash, reserved_at, expires_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (key_hash, ticket_hash, current, current + self.window_seconds),
                )
                return ticket, 0
        except Exception:
            self._raise_unavailable()

    def finish_attempt(self, key: str, attempt_id: str, success: bool | None,
                       *, now: float | None = None) -> bool:
        """Release only this ticket, optionally adding a failure or clearing failures."""
        if success is not None and type(success) is not bool:
            raise ValueError("success must be bool or None")
        key_hash = self._key_hash(key)
        ticket_hash = self._ticket_hash(attempt_id)
        self._validate_now(now)
        self._maybe_prune_expired(explicit_now=now is not None)
        try:
            with self.database.transaction() as connection:
                self._acquire_key_lock(connection, key_hash)
                current = self._now(connection, now)
                self._prune_locked(connection, key_hash, current)
                released = connection.execute(
                    """
                    DELETE FROM login_rate_limit_attempts
                    WHERE key_hash = %s AND ticket_hash = %s
                    RETURNING ticket_hash
                    """,
                    (key_hash, ticket_hash),
                ).fetchone()
                if released is None:
                    return False
                if success is True:
                    connection.execute(
                        "DELETE FROM login_rate_limit_failures WHERE key_hash = %s",
                        (key_hash,),
                    )
                elif success is False:
                    connection.execute(
                        "INSERT INTO login_rate_limit_failures(key_hash, failed_at) VALUES (%s, %s)",
                        (key_hash, current),
                    )
                return True
        except Exception:
            self._raise_unavailable()

    def failed(self, key: str, *, now: float | None = None) -> None:
        self._validate_now(now)
        key_hash = self._key_hash(key)
        self._maybe_prune_expired(explicit_now=now is not None)
        try:
            with self.database.transaction() as connection:
                self._acquire_key_lock(connection, key_hash)
                current = self._now(connection, now)
                self._prune_locked(connection, key_hash, current)
                connection.execute(
                    "INSERT INTO login_rate_limit_failures(key_hash, failed_at) VALUES (%s, %s)",
                    (key_hash, current),
                )
        except Exception:
            self._raise_unavailable()

    def succeeded(self, key: str) -> None:
        key_hash = self._key_hash(key)
        self._maybe_prune_expired(explicit_now=False)
        try:
            with self.database.transaction() as connection:
                self._acquire_key_lock(connection, key_hash)
                self._prune_locked(connection, key_hash, self._now(connection, None))
                connection.execute(
                    "DELETE FROM login_rate_limit_failures WHERE key_hash = %s",
                    (key_hash,),
                )
        except Exception:
            self._raise_unavailable()
