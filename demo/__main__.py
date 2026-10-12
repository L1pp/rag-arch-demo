"""Command line entry point for the offline-first demonstration."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .runtime import DEFAULT_PLAN, DEFAULT_STANDARDS, run_offline_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行虚构资料驱动的 RAG 离线演示。")
    parser.add_argument("--output", type=Path, default=Path("output"), help="演示结果目录，默认 output。")
    parser.add_argument("--mode", choices=("offline", "live"), default="offline",
                        help="默认 offline；live 会读取本演示目录自己的 .env 并调用已配置服务。")
    parser.add_argument("--standards", type=Path, default=DEFAULT_STANDARDS, help="虚构样例规范JSON。")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="虚构样例方案JSON。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run = run_offline_demo(args.output, standards_path=args.standards,
                               plan_path=args.plan, mode=args.mode)
    except Exception as exc:
        # Provider exceptions can contain request details. Keep the public CLI bounded.
        print(f"演示运行失败：{type(exc).__name__}。请检查演示配置或本地依赖。", file=sys.stderr)
        return 1
    print("离线演示完成。" if args.mode == "offline" else "在线演示完成。")
    print(f"结果文件已写入：{args.output}")
    print("生成：report.md、standard_answer.json、plan_review.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())