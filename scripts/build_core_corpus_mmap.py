"""Offline build of the public standards corpus record/chunk mmap package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_arch.binary_corpus import build_core_corpus_mmap  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线构建按需解码的规范corpus mmap包；不初始化模型或访问网络。")
    parser.add_argument("--source", type=Path, default=ROOT / "data/corpus",
                        help="源语料目录（默认 data/corpus）")
    parser.add_argument("--output", type=Path, default=ROOT / "data/runtime/core_corpus",
                        help="不可变版本输出目录（默认 data/runtime/core_corpus）")
    args = parser.parse_args(argv)
    version = build_core_corpus_mmap(args.source, args.output)
    manifest = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "built",
        "version_dir": str(version),
        "clause_records": manifest["record_counts"]["clauses"],
        "table_records": manifest["record_counts"]["tables"],
        "chunks": manifest["chunk_count"],
        "unique_chunk_texts": manifest["unique_text_count"],
        "source_projection_records": manifest["source_record_count"],
        "files": {name: entry["file_bytes"] for name, entry in manifest["files"].items()},
        "network_requests": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
