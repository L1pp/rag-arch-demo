"""Manage pre-created web accounts without exposing passwords in shell history."""
from __future__ import annotations

import argparse
from datetime import datetime
import getpass
from pathlib import Path
import sys

from .auth import create_auth_store


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理建筑资料助手 Web 登录账号。")
    parser.add_argument("--db", help="显式使用指定的 SQLite 认证文件，优先于 RAG_ARCH_DATABASE_URL")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="添加一个登录账号")
    add.add_argument("username")
    commands.add_parser("list", help="列出账号和启用状态")
    disable = commands.add_parser("disable", help="停用账号并撤销其所有会话")
    disable.add_argument("username")
    set_password = commands.add_parser("set-password", help="重设密码并撤销其所有会话")
    set_password.add_argument("username")
    return parser


def _read_new_password() -> str:
    password = getpass.getpass("新密码（至少 8 个字符）：")
    repeated = getpass.getpass("再次输入新密码：")
    if password != repeated:
        raise ValueError("两次输入的密码不一致。")
    return password


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = create_auth_store(ROOT, args.db)
    try:
        if args.command == "add":
            password = _read_new_password()
            store.create_user(args.username, password)
            print(f"已添加账号：{args.username}")
        elif args.command == "list":
            users = store.list_users()
            if not users:
                print("暂无账号。")
            else:
                print("用户名\t状态\t创建时间")
                for user in users:
                    created = datetime.fromtimestamp(float(user["created_at"])).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                    status = "启用" if user["enabled"] else "停用"
                    print(f"{user['username']}\t{status}\t{created}")
        elif args.command == "disable":
            if not store.disable_user(args.username):
                raise ValueError("未找到该账号。")
            print(f"已停用账号并撤销其会话：{args.username}")
        elif args.command == "set-password":
            password = _read_new_password()
            if not store.set_password(args.username, password):
                raise ValueError("未找到该账号。")
            print(f"已重设密码并撤销其会话：{args.username}")
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
