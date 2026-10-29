"""Offline build of compact BM25 postings for the public standards corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_arch.binary_bm25 import (  # noqa: E402
    build_core_bm25,
)
from rag_arch.evidence_index import _record_chunk  # noqa: E402
from rag_arch.standards import load_corpus  # noqa: E402


def _core_chunks(corpus):
    """Create the same ordered chunk IDs/text as EvidenceIndex without BM25."""
    collection_id = "standards:core-bm25-build"
    return [
        _record_chunk(record, kind, collection_id)
        for kind, records in (("clause", corpus.clauses), ("table", corpus.tables))
        for record in records
        if record.get("document_part", "body") != "commentary"
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线构建当前公共规范的只读BM25 mmap包；不初始化模型或访问网络。")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus",
                        help="规范语料目录（默认 data/corpus）")
    parser.add_argument("--output", type=Path, default=ROOT / "data/runtime/core_bm25",
                        help="只读mmap版本输出目录（默认 data/runtime/core_bm25）")
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    chunks = _core_chunks(corpus)
    version = build_core_bm25(args.output, chunks=chunks)
    manifest = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "built",
        "version_dir": str(version),
        "documents": manifest["document_count"],
        "vocabulary": manifest["vocabulary_count"],
        "postings": manifest["posting_count"],
        "average_length": manifest["average_length"],
        "array_bytes": manifest["array_bytes"],
        "network_requests": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
