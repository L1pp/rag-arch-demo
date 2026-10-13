"""Small SQLite-backed account and server-side session store for the web app."""
from __future__ import annotations

from collections import deque
import hashlib
import hmac
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

    def verify_user(self, username: str, password: str) -> str | None:
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
                "SELECT username, password_salt, password_hash, password_iterations, enabled "
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
        return str(row["username"])

    def create_session(self, username: str) -> tuple[str, int]:
        self._ensure_initialized()
        name = normalize_username(username)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        now = time.time()
        expires_at = now + self.session_ttl_seconds
        connection = self._connect()
        try:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            row = connection.execute(
                "SELECT id FROM users WHERE username = ? COLLATE NOCASE AND enabled = 1", (name,)
            ).fetchone()
            if row is None:
                raise ValueError("用户不存在或已停用。")
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

    def resolve_session(self, token: str | None) -> str | None:
        self._ensure_initialized()
        if not token or len(token) > 256:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT users.username, users.enabled, sessions.expires_at "
                "FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if not row["enabled"] or row["expires_at"] <= now:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                connection.commit()
                return None
            return str(row["username"])
        finally:
            connection.close()

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


class LoginRateLimiter:
    """Process-local sliding-window limiter for failed login attempts."""

    def __init__(self, *, max_attempts: int = 10, window_seconds: int = 300):
        if max_attempts < 1 or window_seconds < 1:
            raise ValueError("rate-limit settings must be positive")
        self.max_attempts = int(max_attempts)
        self.window_seconds = int(window_seconds)
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, failures: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while failures and failures[0] <= cutoff:
            failures.popleft()

    def is_limited(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        current = time.time() if now is None else now
        with self._lock:
            failures = self._failures.get(key)
            if failures is None:
                return False, 0
            self._prune(failures, current)
            if not failures:
                self._failures.pop(key, None)
                return False, 0
            if len(failures) < self.max_attempts:
                return False, 0
            retry_after = max(1, int(self.window_seconds - (current - failures[0])))
            return True, retry_after

    def failed(self, key: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self._lock:
            failures = self._failures.setdefault(key, deque())
            self._prune(failures, current)
            failures.append(current)

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
