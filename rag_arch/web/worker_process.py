"""Command line entry point for the independent durable task worker.

This module never migrates the database and never starts a system service.
Production ``run`` uses the configured model runners. ``--simulation`` is an
explicit test switch that routes claimed tasks only to ``run_demo``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
from typing import Any

from .task_worker import ROOT, WorkerRuntime


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def _database(database_url: str | None = None, schema: str | None = None):
    url = database_url or os.environ.get("RAG_ARCH_DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("缺少RAG_ARCH_DATABASE_URL；worker不会自动创建数据库或迁移表。")
    selected_schema = schema or os.environ.get("RAG_ARCH_DATABASE_SCHEMA", "public")
    from .db import Database
    return Database(url, schema=selected_schema)


def _repository(database):
    from .task_repository import TaskRepository
    return TaskRepository(database)


def _check(database, *, simulation: bool) -> dict[str, Any]:
    """Check configuration and schema with SELECT-only database access."""
    settings = None
    if not simulation:
        from ..config import load_settings
        settings = load_settings()
    from .readiness import REQUIRED_MIGRATIONS, REQUIRED_TABLES
    connection = database.connect()
    try:
        connection.execute("SET TRANSACTION READ ONLY")
        row = connection.execute("SELECT current_schema() AS schema_name").fetchone()
        tables = {str(item["table_name"]) for item in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
        ).fetchall()}
        migrations = []
        if "schema_migrations" in tables:
            migrations = [str(item["version"]) for item in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version").fetchall()]
    finally:
        connection.close()
    missing_migrations = sorted(REQUIRED_MIGRATIONS.difference(migrations))
    missing_tables = sorted(REQUIRED_TABLES.difference(tables))
    schema_ready = not missing_migrations and not missing_tables
    model_ready = bool(simulation or (settings is not None and settings.chat_ready))
    return {
        "database_reachable": True,
        "schema": str(row["schema_name"]) if row else database.schema,
        "schema_ready": schema_ready,
        "migrations": migrations,
        "missing_migrations": missing_migrations,
        "missing_tables": missing_tables,
        "model_ready": model_ready,
        "simulation": simulation,
        "ready": schema_ready and model_ready,
    }


def _run(args: argparse.Namespace) -> int:
    database = _database(args.database_url, args.database_schema)
    repository = _repository(database)
    settings = None
    if not args.simulation:
        from ..config import load_settings
        settings = load_settings()

    runtime = WorkerRuntime(
        repository,
        database=database,
        settings=settings,
        root=Path(args.root).resolve() if args.root else ROOT,
        worker_instance=args.worker_instance,
        concurrency=args.concurrency,
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
        lease_seconds=args.lease_seconds,
        simulation=args.simulation,
        task_timeout_seconds=args.task_timeout,
    )

    def request_stop(_signum, _frame):
        runtime.request_stop()

    previous = {}
    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        except (ValueError, OSError):  # signal hooks are only allowed on the main thread
            continue
    try:
        runtime.run(max_tasks=args.max_tasks)
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
    return 0


def _recover(args: argparse.Namespace) -> int:
    if args.cooldown_seconds is not None and not args.isolate_interrupted:
        raise ValueError("--cooldown-seconds 需要同时指定 --isolate-interrupted。")
    root = Path(args.root).resolve() if args.root else ROOT
    repository = _repository(_database(args.database_url, args.database_schema))
    if args.isolate_interrupted:
        from .resource_recovery import maintain_recovery
        cooldown = 120.0 if args.cooldown_seconds is None else args.cooldown_seconds
        outcomes = maintain_recovery(
            repository, cleanup_owned=args.cleanup_owned_resources, root=root,
            cooldown_seconds=cooldown)
        note = ("本机worker和登记资源确认退出并度过冷却期后才隔离中断任务；"
                "远端调用结果仍未知，任务不会自动重试。")
    else:
        from .resource_recovery import recover_resources
        outcomes = recover_resources(repository, cleanup_owned=args.cleanup_owned_resources, root=root)
        note = ("只有simulation/run_demo任务会在确认本机worker进程退出后终结并释放槽；"
                "真实模型任务继续recovering并占用用户槽和全局容量，等待资源及外部调用核验。")
    lease_check = next((row for row in outcomes if row.get("kind") == "lease_check"), {})
    report = {
        "marked_stale_workers": lease_check.get("marked_stale_workers", []),
        "local_worker_checks": lease_check.get("parent_checks", []),
        "local_resource_checks": outcomes,
        "cleanup_owned_resources": args.cleanup_owned_resources,
        "note": note,
    }
    if args.isolate_interrupted:
        report.update(isolate_interrupted=True, cooldown_seconds=cooldown)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _resolve(args: argparse.Namespace) -> int:
    from .resource_recovery import resolve_task
    path = Path(args.proof_file).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("proof必须是普通JSON文件。")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("proof文件不能超过1 MiB。")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("proof不能包含重复JSON字段。")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("proof不能包含非有限数字。")

    proof = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    repository = _repository(_database(args.database_url, args.database_schema))
    accepted = resolve_task(
        repository, args.task_id, proof,
        root=Path(args.root).resolve() if args.root else ROOT)
    print(json.dumps({"task_id": args.task_id, "resolved": bool(accepted)}, ensure_ascii=False))
    return 0 if accepted else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_arch.web.worker_process",
        description="运行、检查或安全恢复本机的RAG Web任务worker。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="启动独立任务worker进程")
    run_parser.add_argument("--database-url", help="数据库连接串；默认读取RAG_ARCH_DATABASE_URL")
    run_parser.add_argument("--database-schema", "--schema", dest="database_schema",
                            help="数据库schema；默认读取RAG_ARCH_DATABASE_SCHEMA")
    run_parser.add_argument("--root", help="项目数据根目录；默认使用本仓库根目录")
    run_parser.add_argument("--worker-instance", help="worker实例编号；默认生成随机编号")
    run_parser.add_argument("--concurrency", type=_positive_int,
                            default=_positive_int(os.environ.get("RAG_ARCH_WORKER_CONCURRENCY", "3")),
                            help="本worker的有限执行线程数（默认3）")
    run_parser.add_argument("--poll-interval", type=_positive_float, default=0.5,
                            help="空闲时的领取轮询间隔秒数")
    run_parser.add_argument("--heartbeat-interval", type=_positive_float, default=5.0,
                            help="worker心跳和取消检查间隔秒数")
    run_parser.add_argument("--lease-seconds", type=_positive_int, default=15,
                            help="任务租约秒数，须大于心跳间隔")
    run_parser.add_argument("--max-tasks", type=_positive_int,
                            help="处理指定数量的新任务后优雅退出")
    run_parser.add_argument("--task-timeout", type=_positive_float,
                            default=_positive_float(os.environ.get("RAG_ARCH_TASK_TIMEOUT_SECONDS", "3600")),
                            help="模型任务总执行时限秒数；到期取消并等待本地工作收尾（默认3600）")
    run_parser.add_argument("--simulation", action="store_true",
                            help="显式测试模式：只运行run_demo，不调用真实模型")
    run_parser.set_defaults(handler=_run)

    check_parser = subparsers.add_parser("check", help="只读检查数据库、schema和模型配置")
    check_parser.add_argument("--database-url", help="数据库连接串；默认读取RAG_ARCH_DATABASE_URL")
    check_parser.add_argument("--database-schema", "--schema", dest="database_schema",
                              help="数据库schema；默认读取RAG_ARCH_DATABASE_SCHEMA")
    check_parser.add_argument("--simulation", action="store_true",
                              help="只检查worker模拟运行所需配置，不要求模型密钥")
    check_parser.set_defaults(handler=None)

    recover_parser = subparsers.add_parser("recover", help="只恢复已确认退出的本机worker任务")
    recover_parser.add_argument("--database-url", help="数据库连接串；默认读取RAG_ARCH_DATABASE_URL")
    recover_parser.add_argument("--database-schema", "--schema", dest="database_schema",
                                help="数据库schema；默认读取RAG_ARCH_DATABASE_SCHEMA")
    recover_parser.add_argument("--root", help="资源身份描述符所属项目根目录")
    recover_parser.add_argument("--cleanup-owned-resources", action="store_true",
                                help="本机父worker退出后允许清理部分身份已核实的自有子进程；LibreOffice进程组只核验退出，不由此命令终止")
    recover_parser.add_argument("--isolate-interrupted", action="store_true",
                                help="显式隔离已确认本机进程退出且冷却完成的中断任务；远端状态仍未知且不自动重试")
    recover_parser.add_argument("--cooldown-seconds", type=_nonnegative_float,
                                help="仅用于 --isolate-interrupted；冷却时间秒数（默认120）")
    recover_parser.set_defaults(handler=_recover)

    resolve_parser = subparsers.add_parser("resolve", help="凭本机退出证据和外部调用核验审计终结recovering任务")
    resolve_parser.add_argument("--database-url", help="数据库连接串；默认读取RAG_ARCH_DATABASE_URL")
    resolve_parser.add_argument("--database-schema", "--schema", dest="database_schema",
                                help="数据库schema；默认读取RAG_ARCH_DATABASE_SCHEMA")
    resolve_parser.add_argument("--root", help="资源身份描述符所属项目根目录")
    resolve_parser.add_argument("--task-id", required=True)
    resolve_parser.add_argument("--proof-file", required=True, help="含operator/reason/外部调用与父子进程身份的JSON证据")
    resolve_parser.set_defaults(handler=_resolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if args.command == "check":
            result = _check(_database(args.database_url, args.database_schema),
                            simulation=args.simulation)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 2
        return args.handler(args)
    except Exception as exc:
        # Driver and SDK error strings can embed the supplied connection URI
        # (including a password). Keep the CLI/journal boundary credential-free.
        print(f"worker {args.command}失败（{type(exc).__name__}）；请检查配置、schema和核验材料。",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
