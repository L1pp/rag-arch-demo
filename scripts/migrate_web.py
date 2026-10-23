"""Explicit PostgreSQL schema, legacy account, and Web history migration.

This module never reads dotenv files and never connects to a database on import.
Run ``python -m scripts.migrate_web schema`` to apply schema migrations,
``python -m scripts.migrate_web import-auth --source-sqlite PATH`` to copy
accounts, or ``python -m scripts.migrate_web legacy-data --owner-user-id ID``
to review old upload/task history. Legacy data is dry-run by default and one
owner must be named for each apply. Existing sessions are intentionally not
copied, so users sign in again after account import.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

if __package__ in (None, ""):
    # Keep direct invocation (python scripts/migrate_web.py ...) usable from
    # outside the repository root without depending on PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_arch.web.auth import ascii_username_key_from_normalized
from rag_arch.web.db import Database


_USER_COLUMNS = (
    "id",
    "username",
    "password_salt",
    "password_hash",
    "password_iterations",
    "enabled",
    "created_at",
    "updated_at",
)
_TARGET_COLUMNS = (
    "id",
    "username",
    "username_key",
    "password_salt",
    "password_hash",
    "password_iterations",
    "enabled",
    "created_at",
    "updated_at",
)


class MigrationError(RuntimeError):
    """A safe, user-facing migration error without account secrets."""


@dataclass(frozen=True)
class LegacyUser:
    user_id: int
    username: str
    username_key: str
    password_salt: bytes
    password_hash: bytes
    password_iterations: int
    enabled: bool
    created_at: float
    updated_at: float

    def target_values(self) -> tuple[object, ...]:
        return (
            self.user_id,
            self.username,
            self.username_key,
            self.password_salt,
            self.password_hash,
            self.password_iterations,
            self.enabled,
            self.created_at,
            self.updated_at,
        )


def _sqlite_readonly_uri(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    # Path.as_uri escapes special characters and emits a valid file URI on
    # Windows as well as POSIX systems. mode=ro prevents creating or changing
    # the legacy source database.
    return f"{source.as_uri()}?mode=ro"


def read_legacy_users(path: str | Path) -> list[LegacyUser]:
    """Read and validate account rows using a read-only SQLite connection."""
    try:
        connection = sqlite3.connect(_sqlite_readonly_uri(path), uri=True)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise MigrationError("无法以只读方式打开指定的 SQLite 账号库。") from exc

    try:
        connection.row_factory = sqlite3.Row
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        if not set(_USER_COLUMNS).issubset(columns):
            raise MigrationError("SQLite 账号库缺少必要的 users 字段。")

        rows = connection.execute(
            "SELECT id, username, password_salt, password_hash, "
            "password_iterations, enabled, created_at, updated_at "
            "FROM users ORDER BY id"
        ).fetchall()
    except MigrationError:
        raise
    except sqlite3.Error as exc:
        raise MigrationError("读取 SQLite 账号记录失败。") from exc
    finally:
        connection.close()

    users: list[LegacyUser] = []
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    try:
        for row in rows:
            user_id = int(row["id"])
            username = str(row["username"])
            # SQLite stores the result of normalize_username(). Reapplying
            # NFKC/strip here can change some already-normalized usernames.
            username_key = ascii_username_key_from_normalized(username)
            salt = bytes(row["password_salt"])
            password_hash = bytes(row["password_hash"])
            iterations = int(row["password_iterations"])
            enabled_value = int(row["enabled"])
            if enabled_value not in (0, 1):
                raise MigrationError("SQLite 账号库包含无效的启用状态。")
            if user_id in seen_ids or username_key in seen_keys:
                raise MigrationError("SQLite 账号库存在重复的用户 ID 或用户名键。")
            if iterations < 1:
                raise MigrationError("SQLite 账号库包含无效的密码参数。")
            seen_ids.add(user_id)
            seen_keys.add(username_key)
            users.append(
                LegacyUser(
                    user_id=user_id,
                    username=username,
                    username_key=username_key,
                    password_salt=salt,
                    password_hash=password_hash,
                    password_iterations=iterations,
                    enabled=bool(enabled_value),
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                )
            )
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, MigrationError):
            raise
        raise MigrationError("SQLite 账号库包含无法导入的字段值。") from exc
    return users


def _set_user_identity_sequence(connection, schema: str) -> None:
    """Advance the identity sequence if imported IDs would otherwise collide."""
    try:
        from psycopg import sql
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise MigrationError("导入账号需要安装 PostgreSQL 驱动 psycopg 3。") from exc

    qualified_table = sql.Identifier(schema, "users").as_string(connection)
    sequence = connection.execute(
        "SELECT n.nspname, c.relname FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.oid = pg_get_serial_sequence(%s, 'id')::regclass",
        (qualified_table,),
    ).fetchone()
    if sequence is None:
        raise MigrationError("目标 users.id 没有关联 identity 序列。")

    sequence_identifier = sql.Identifier(sequence["nspname"], sequence["relname"])
    state = connection.execute(
        sql.SQL("SELECT last_value, is_called FROM {}").format(sequence_identifier)
    ).fetchone()
    maximum = connection.execute(
        "SELECT COALESCE(MAX(id), 0) AS maximum FROM users"
    ).fetchone()["maximum"]
    # setval is intentionally never allowed to move the sequence backwards.
    # If it has not been called, nextval returns last_value itself; otherwise
    # it returns the following value (the identity increment is one).
    needs_advance = (
        int(maximum) >= int(state["last_value"])
        if not state["is_called"]
        else int(maximum) > int(state["last_value"])
    )
    if needs_advance:
        connection.execute(
            "SELECT setval(%s::regclass, %s, TRUE)",
            (sequence_identifier.as_string(connection), int(maximum)),
        )


def import_auth_users(database: Database, source_sqlite: str | Path) -> tuple[int, int]:
    """Import SQLite users atomically, preserving account fields and IDs.

    Returns ``(inserted, already_present)``. Existing rows are accepted only
    when every migrated field matches; conflicts abort the entire transaction.
    The source ``sessions`` table is deliberately never read.
    """
    users = read_legacy_users(source_sqlite)
    inserted = 0
    already_present = 0

    try:
        with database.transaction() as connection:
            # Coordinate with other account writers before comparing rows or
            # adjusting the identity sequence. This lock is scoped to the
            # complete import transaction and does not serialize read-only use.
            connection.execute("LOCK TABLE users IN SHARE ROW EXCLUSIVE MODE")
            for user in users:
                by_id = connection.execute(
                    "SELECT id, username, username_key, password_salt, password_hash, "
                    "password_iterations, enabled, created_at, updated_at "
                    "FROM users WHERE id = %s",
                    (user.user_id,),
                ).fetchone()

                if by_id is not None:
                    actual = tuple(
                        bytes(by_id[name])
                        if name in ("password_salt", "password_hash")
                        else bool(by_id[name])
                        if name == "enabled"
                        else by_id[name]
                        for name in _TARGET_COLUMNS
                    )
                    if actual != user.target_values():
                        raise MigrationError(
                            "目标库存在同 ID 但内容不同的账号；整笔导入已回滚。"
                        )
                    already_present += 1
                    continue

                by_key = connection.execute(
                    "SELECT id FROM users WHERE username_key = %s",
                    (user.username_key,),
                ).fetchone()
                if by_key is not None:
                    raise MigrationError(
                        "目标库存在用户名键冲突；整笔导入已回滚。"
                    )

                connection.execute(
                    "INSERT INTO users(id, username, username_key, password_salt, "
                    "password_hash, password_iterations, enabled, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    user.target_values(),
                )
                inserted += 1

            # Sequences are not advanced on a repeat import. This keeps a
            # no-op import physically read-only with respect to target rows and
            # sequence state while making the next generated user ID safe.
            if inserted:
                _set_user_identity_sequence(connection, database.schema)
    except MigrationError:
        raise
    except Exception as exc:
        # Database exceptions can contain SQL parameters. Do not expose them:
        # account password hashes and salts are among the imported values.
        raise MigrationError(
            "账号导入失败，目标事务已回滚；数据库错误细节已隐藏。"
        ) from exc

    return inserted, already_present


def _database_url(explicit: str | None) -> str:
    value = explicit or os.environ.get("RAG_ARCH_DATABASE_URL")
    if not value:
        raise MigrationError(
            "缺少数据库连接配置，请提供 --database-url 或设置 RAG_ARCH_DATABASE_URL。"
        )
    return value


def _database_schema() -> str:
    # The Database adapter quotes schema identifiers. Its public default is
    # retained for deployments that do not set the optional schema variable.
    return os.environ.get("RAG_ARCH_DATABASE_SCHEMA", "public")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="显式迁移 Web schema、SQLite账号或旧附件和终态任务历史。"
    )
    parser.add_argument(
        "--database-url",
        help="目标 PostgreSQL URL（默认读取 RAG_ARCH_DATABASE_URL）",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    schema_command = commands.add_parser(
        "schema", aliases=["migrate"], help="显式创建并升级数据库 schema"
    )
    schema_command.add_argument(
        "--database-url", dest="command_database_url", help=argparse.SUPPRESS
    )

    import_command = commands.add_parser(
        "import-auth",
        help="维护窗口内从指定 SQLite 库导入 users（不迁移 sessions）",
    )
    import_command.add_argument(
        "--database-url", dest="command_database_url", help=argparse.SUPPRESS
    )
    import_command.add_argument(
        "--source-sqlite",
        "--source",
        dest="source_sqlite",
        required=True,
        help="旧 SQLite 认证数据库路径（只读打开）",
    )

    legacy_command = commands.add_parser(
        "legacy-data",
        help="预览或迁移一个明确用户的旧附件与终态任务历史（默认dry-run）",
    )
    legacy_command.add_argument(
        "--database-url", dest="command_database_url", help=argparse.SUPPRESS
    )
    legacy_command.add_argument(
        "--owner-user-id", type=int, required=True,
        help="本次唯一目标用户ID；归档本身不含可信owner",
    )
    legacy_command.add_argument(
        "--archive-root", required=True,
        help="旧任务result.json归档目录，例如 data/runtime/web_demo",
    )
    legacy_command.add_argument(
        "--upload-root", required=True,
        help="旧附件根目录，例如 data/runtime/web_uploads",
    )
    legacy_command.add_argument(
        "--target-root", required=True,
        help="新附件存储所在项目根目录；写入 data/runtime/web_uploads 下",
    )
    legacy_command.add_argument(
        "--asset-manifest",
        help=("可选附件JSON清单；assets每项含asset_id、source相对路径，"
              "source格式为旧workspace-id/asset-id/source.ext；可用filename恢复原名，"
              "缺省时使用asset_id.ext"),
    )
    mode = legacy_command.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", dest="apply", action="store_false",
        help="只读检查并输出计划（默认）",
    )
    mode.add_argument(
        "--apply", dest="apply", action="store_true",
        help="执行已明确指定owner的迁移",
    )
    legacy_command.set_defaults(apply=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        url = _database_url(
            getattr(args, "command_database_url", None) or args.database_url
        )
        database = Database(url, schema=_database_schema())
        if args.command in ("schema", "migrate"):
            applied = database.migrate()
            if applied:
                print(f"schema 已应用 {len(applied)} 个迁移。")
            else:
                print("schema 已是最新状态。")
            return 0

        if args.command == "legacy-data":
            from rag_arch.web.legacy_import import (
                LegacyImportError,
                apply_legacy_import,
                plan_legacy_import,
            )

            try:
                options = {
                    "archive_root": args.archive_root,
                    "upload_root": args.upload_root,
                    "target_root": args.target_root,
                    "asset_manifest": args.asset_manifest,
                }
                if args.apply:
                    plan = apply_legacy_import(database, args.owner_user_id, **options)
                    print("旧数据迁移结果：")
                    print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
                    if plan.conflicts:
                        return 1
                    print("源归档和旧附件均保留；新workspace事件使用PostgreSQL当前游标。")
                    return 0
                plan = plan_legacy_import(database, args.owner_user_id, **options)
                print("旧数据只读预览（未写入数据库或附件文件）：")
                print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
                if plan.conflicts:
                    return 1
                print("确认owner与来源清单后，添加 --apply 执行迁移。")
                return 0
            except LegacyImportError as exc:
                print(f"操作失败：{exc}", file=sys.stderr)
                return 1

        inserted, already_present = import_auth_users(database, args.source_sqlite)
        print(
            f"账号导入完成：新增 {inserted} 个，已有且内容一致 {already_present} 个。"
        )
        print("旧 SQLite sessions 未迁移；导入的账号需要重新登录。")
        return 0
    except MigrationError as exc:
        print(f"操作失败：{exc}", file=sys.stderr)
        return 1
    except Exception:
        # Driver error strings may include connection details or SQL parameters.
        print("操作失败；为保护凭据，数据库错误细节未输出。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
