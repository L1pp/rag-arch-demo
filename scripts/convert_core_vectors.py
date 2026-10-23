"""Offline conversion of the public standards embedding cache to read-only NPY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_arch.binary_vectors import (  # noqa: E402
    convert_core_vectors,
    read_core_embedding_identity,
)
from rag_arch.evidence_index import EvidenceIndex  # noqa: E402
from rag_arch.standards import load_corpus  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线流式转换公共规范向量；不加载.env、不创建provider或发起网络请求。")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/runtime/core_embeddings.json",
                        help="只读源JSON缓存（默认 data/runtime/core_embeddings.json）")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/runtime/core_vectors",
                        help="二进制版本目录（默认 data/runtime/core_vectors）")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64",
                        help="默认float64以保持当前归一化检索分数；float32为显式折中")
    parser.add_argument("--identity-json", type=Path, default=None,
                        help="可选的期望identity JSON文件，须含base_url/model/dimensions")
    args = parser.parse_args(argv)

    if args.identity_json is None:
        identity = read_core_embedding_identity(args.source)
    else:
        try:
            identity = json.loads(args.identity_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error(f"无法读取identity JSON：{exc}")
    corpus = load_corpus(ROOT / "data/corpus")
    index = EvidenceIndex(corpus)
    version_dir = convert_core_vectors(
        args.source, args.output, identity=identity, chunks=index.chunks, dtype=args.dtype)
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "converted",
        "version_dir": str(version_dir),
        "dtype": manifest["dtype"],
        "shape": manifest["shape"],
        "vector_bytes": manifest["vector_bytes"],
        "metadata_bytes": manifest["metadata_bytes"],
        "source_document_records_scanned": manifest["source_document_records_scanned"],
        "source_query_records_scanned": manifest["source_query_records_scanned"],
        "matched_unique_embedding_texts": manifest["matched_unique_embedding_texts"],
        "network_requests": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
