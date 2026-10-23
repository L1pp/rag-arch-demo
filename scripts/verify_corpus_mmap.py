"""Offline comparison of JSON and lazy mmap corpus backends.

The default stage builds only a small synthetic corpus in a temporary
directory. Real-corpus stages require --confirm-large and reuse existing
immutable BM25/vector mmap packages. This tool does not build those packages,
call a model provider, use PostgreSQL, or start an HTTP service.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the existing BM25 verifier's offline and measurement helpers without
# leaking its import-time BLAS environment changes into test callers.
_BLAS_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_old_blas_env = {key: os.environ.get(key) for key in _BLAS_ENV}
from scripts import verify_bm25_mmap as _bm25
for _key, _value in _old_blas_env.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value
del _old_blas_env, _key, _value

import numpy as np
import psutil

from rag_arch.binary_bm25 import build_core_bm25
from rag_arch.binary_corpus import BinaryCorpusStore, build_core_corpus_mmap
from rag_arch.binary_vectors import BinaryVectorStore
from rag_arch.evidence_index import EvidenceIndex, _record_chunk
from rag_arch.standards import load_corpus


_SOURCE_NAMES = (
    "clauses.jsonl", "tables.jsonl", "summary.json",
    "catalog.json", "relations.json",
)
_BOUNDARY_QUERIES = (
    "1.8≤18", "1.8 >= 18", "≤ ≥ ≠", "", "   ",
    "bm25_mmap_unseen_probe",
)
_EXPECTED_CACHED_QUERY_COUNT = 28
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_CONTROL_POLL_SECONDS = 0.05


def _jsonl_write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(
                record, ensure_ascii=False, allow_nan=False,
                separators=(",", ":")) + "\n")


def _make_synthetic_records(filler_count: int = 128):
    """Small source fixture shared by the CLI smoke stage and unittest tests."""
    if isinstance(filler_count, bool) or not isinstance(filler_count, int) or filler_count < 32:
        raise ValueError("synthetic filler_count must be an integer >= 32")
    std_id = "GB50000-2020"
    std_no = "GB 50000-2020"
    second_id = "GB50000-2024"
    second_no = "GB 50000-2024"
    owner_id = f"{std_id}/4.2.1"
    table_id = f"{std_id}/table/4.2.1-1"
    formula_id = f"{std_id}/formula/4.2.1-1"
    revision_owner_id = f"{std_id}/5.1"
    revised_formula_id = f"{std_id}/amendment/2024/formula-5.1"

    def clause(clause_id, number, text, *, standard_id=std_id,
               standard_no=std_no, **fields):
        return {
            "clause_id": clause_id, "std_id": standard_id,
            "std_no": standard_no, "clause_no": number, "text": text,
            "document_part": "body", "content_status": "text_available",
            "content_readable": True,
            "source_spans": [{
                "source_id": f"FILE_{standard_id}",
                "source_path": f"synthetic/{standard_id}.pdf",
                "page_start": 1, "page_end": 2,
            }],
            **fields,
        }

    clauses = [
        clause(
            owner_id, "4.2.1",
            "主体结构共同过滤词，应按已核公式计算并同时查阅表格。",
            table_ids=[table_id], formula_ids=[formula_id],
            heading_path=["4 主体结构", "4.2 计算"],
        ),
        clause(
            formula_id, "4.2.1-1",
            "共同公式 F 等于恒载与活载之和。",
            record_kind="formula", formula_no="(4.2.1-1)",
            formula_latex="F=G+Q", formula_text="F=G+Q",
            variables=[{"symbol": "G", "meaning": "恒载"},
                       {"symbol": "Q", "meaning": "活载"}],
            units="kN", heading_path=["4 主体结构", "4.2 计算"],
        ),
        clause(
            revision_owner_id, "5.1",
            "修订前基础条文，需结合2024公式修订。",
            import_part_id="body-2020",
            heading_path=["5 修订对象"],
        ),
        clause(
            revised_formula_id, "5.1",
            "2024修订公式为 M 等于设计弯矩。",
            document_part="amendment", import_part_id="amendment-2024",
            record_kind="formula", formula_no="(5.1-2024)",
            formula_latex="M=M_d", formula_text="M=M_d",
            variables=[{"symbol": "M_d", "meaning": "设计弯矩"}],
            units="kN·m", heading_path=["2024修订", "5.1"],
        ),
        clause(
            f"{std_id}/commentary/4.2.2", "4.2.2",
            "commentary_only_sentinel 解释性文字不进入正文检索。",
            document_part="commentary", heading_path=["条文说明"],
        ),
        clause(
            f"{std_id}/appendix-a/9.9", "9.9",
            "附录 A 限定条号共同过滤词。",
            document_part="appendix", exact_clause_lookup=False,
            heading_path=["附录 A"],
        ),
        clause(
            f"{std_id}/appendix-b/9.9", "9.9",
            "附录 B 限定条号共同过滤词。",
            document_part="appendix", exact_clause_lookup=False,
            heading_path=["附录 B"],
        ),
        clause(
            f"{second_id}/4.2.1", "4.2.1",
            "另一个版本的主体条文共同过滤词。",
            standard_id=second_id, standard_no=second_no,
        ),
    ]
    for number in range(1000, 1000 + filler_count):
        clauses.append(clause(
            f"{std_id}/synthetic/{number}", f"8.0.{number}",
            "共同过滤词用于稳定并列排序的合成正文。",
            heading_path=["8 合成索引"],
        ))

    tables = [{
        "table_id": table_id, "std_id": std_id, "std_no": std_no,
        "table_no": "4.2.1-1", "caption": "共同过滤词验算表",
        "markdown": "| 项目 | 取值 |\n| --- | --- |\n| 设计值 | 18 kN |",
        "html": "", "flat_text": "项目 取值 设计值 18 kN",
        "notes": ["表格说明"], "owner_clause_id": owner_id,
        "document_part": "body", "content_status": "text_available",
        "content_readable": True,
        "source_spans": [{
            "source_id": f"FILE_{std_id}",
            "source_path": f"synthetic/{std_id}.pdf",
            "page_start": 2, "page_end": 2,
        }],
    }]
    summary = {
        "schema_version": "core_corpus_v1",
        "document_count": 2,
        "clause_count": len(clauses),
        "table_count": len(tables),
        "documents": [
            {"std_id": std_id, "std_no": std_no, "record_count": len(clauses),
             "table_count": 1},
            {"std_id": second_id, "std_no": second_no, "record_count": 1,
             "table_count": 0},
        ],
        "issues": [],
    }
    catalog = {"standards": [
        {"std_id": std_id, "std_no": std_no, "title": "合成标准 2020",
         "library_class": "A", "text_available": True, "source_files": []},
        {"std_id": second_id, "std_no": second_no, "title": "合成标准 2024",
         "library_class": "A", "text_available": True, "source_files": []},
    ]}
    relations = {"relations": [{
        "relation_id": f"{std_id}/revision/2024-formula",
        "relation": "amends", "scope": "explicit_formula",
        "source_std_id": std_id, "target_std_id": std_id,
        "source_part_id": "amendment-2024", "target_part_id": "body-2020",
        "target_clause_ids": [revision_owner_id],
        "target_clause_nos": ["5.1"],
        "source_evidence_ids": [revised_formula_id],
        "source_clause_nos": [], "source_physical_pages": [],
        "target_page_ranges": [], "whole_clause": False,
    }]}
    return clauses, tables, summary, catalog, relations


def _write_synthetic_corpus(corpus_dir: str | Path, *, filler_count: int = 128) -> Path:
    directory = Path(corpus_dir)
    directory.mkdir(parents=True, exist_ok=True)
    clauses, tables, summary, catalog, relations = _make_synthetic_records(filler_count)
    _jsonl_write(directory / "clauses.jsonl", clauses)
    _jsonl_write(directory / "tables.jsonl", tables)
    for name, payload in (
        ("summary.json", summary), ("catalog.json", catalog),
        ("relations.json", relations),
    ):
        (directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return directory


def _source_snapshot(directory: Path) -> dict:
    return {
        name: (_bm25._source_snapshot(directory / name)
               if (directory / name).is_file() else None)
        for name in _SOURCE_NAMES
    }


def _iter_source_records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{line_number} must contain JSON objects")
                yield value


def _canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _chunk_payload(chunk) -> dict:
    locator = copy.deepcopy(chunk.locator)
    locator.pop("collection_id", None)
    return {
        "doc_id": chunk.doc_id, "filename": chunk.filename,
        "chunk_id": chunk.chunk_id, "text": chunk.text,
        "locator": locator, "source_kind": chunk.source_kind,
    }


def _hit_payload(hit) -> dict:
    return {"score_hex": float(hit.score).hex(), **_chunk_payload(hit.chunk)}


def _retrieval_payload(result) -> dict:
    return {
        "query": result.query, "mode": result.mode, "reranked": result.reranked,
        "raw_hits": [_hit_payload(hit) for hit in result.raw_hits],
        "context_hits": [_hit_payload(hit) for hit in result.context_hits],
        "missing_evidence": copy.deepcopy(result.missing_evidence),
    }


def _digest(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _compare_records_and_chunks(source_dir: Path, store: BinaryCorpusStore) -> dict:
    """Stream source JSON rows against mapped getters and generated chunks."""
    corpus = store.corpus
    stats_open = store.cache_stats
    if any(stats_open[key] for key in (
            "record_decode_count", "chunk_decode_count", "id_text_decode_count",
            "source_projection_reads", "cache_bytes")):
        raise AssertionError("opening a ready corpus package decoded runtime payload rows")

    record_count = 0
    for name, mapped in (("clauses.jsonl", corpus.clauses),
                         ("tables.jsonl", corpus.tables)):
        row = 0
        for original in _iter_source_records(source_dir / name):
            actual = mapped[row]
            if actual != original:
                raise AssertionError(f"{name} mapped record differs at row {row}")
            row += 1
            record_count += 1
        if row != len(mapped):
            raise AssertionError(f"{name} source/mmap row count differs")

    chunk_count = 0
    sequence = corpus.chunk_sequence("corpus-mmap-verifier")
    row = 0
    for kind, filename in (("clause", "clauses.jsonl"), ("table", "tables.jsonl")):
        for record in _iter_source_records(source_dir / filename):
            if record.get("document_part", "body") == "commentary":
                continue
            expected = _record_chunk(record, kind, "json-source-reference")
            if not expected.text.strip():
                continue
            actual = sequence[row]
            expected_payload = _chunk_payload(expected)
            actual_payload = _chunk_payload(actual)
            if actual_payload != expected_payload:
                raise AssertionError(f"lazy chunk differs from JSON source at row {row}")
            row += 1
            chunk_count += 1
    if row != len(sequence):
        raise AssertionError("source-generated chunk count differs from lazy chunk sequence")

    stats_after = store.cache_stats
    if stats_after["cache_bytes"] > stats_after["cache_limit_bytes"]:
        raise AssertionError("record cache exceeded its configured byte limit")
    return {
        "source_record_rows_compared": record_count,
        "source_record_equal": True,
        "chunk_rows_compared": chunk_count,
        "chunk_text_locator_equal": True,
        "package_source_validation_rows": stats_open["source_validation_records"],
        "open_decode_counts": {
            key: stats_open[key] for key in (
                "record_decode_count", "chunk_decode_count", "id_text_decode_count")
        },
        "post_compare_decode_counts": {
            key: stats_after[key] for key in (
                "record_decode_count", "chunk_decode_count", "id_text_decode_count")
        },
        "record_cache_bytes": stats_after["cache_bytes"],
        "record_cache_limit_bytes": stats_after["cache_limit_bytes"],
    }


def _semantic_snapshot(index: EvidenceIndex, queries: list[str]) -> dict:
    scores, bm25_top30, rrf_top8, filtered = {}, {}, {}, {}
    boundaries = {}
    for query in queries:
        key = _bm25._query_id(query)
        values = np.asarray(index.local._bm25_scores(query), dtype="<f8")
        scores[key] = hashlib.sha256(values.tobytes()).hexdigest()
        bm25_top30[key] = _digest(_retrieval_payload(index.retrieve(
            query, mode="bm25", top_k=30)))
        rrf_top8[key] = _digest(_retrieval_payload(index.retrieve(
            query, mode="rrf", top_k=8)))

    for query in _BOUNDARY_QUERIES:
        key = _bm25._query_id(query)
        values = np.asarray(index.local._bm25_scores(query), dtype="<f8")
        boundaries[key] = {
            "score_array_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "top30_result_sha256": _digest(_retrieval_payload(index.retrieve(
                query, mode="bm25", top_k=30))),
        }

    metadata = getattr(index.chunks, "metadata", None)
    std_no = metadata[0]["std_no"] if metadata is not None and len(metadata) else (
        index.chunks[0].locator["std_no"] if len(index.chunks) else None)
    filter_cases = [
        {"kind": "table"},
        {"std_no": std_no} if std_no else {"document_part": "body"},
        {"document_part": "body"},
    ]
    for query in queries[:3]:
        for filters in filter_cases:
            case = f"{_bm25._query_id(query)}:{_digest(filters)[:12]}"
            filtered[case] = _digest(_retrieval_payload(index.retrieve(
                query, mode="rrf", top_k=8, filters=filters)))
    return {
        "bm25_score_array_sha256": scores,
        "bm25_top30_full_hits_sha256": bm25_top30,
        "rrf_top8_full_hits_sha256": rrf_top8,
        "filtered_rrf_expansion_sha256": filtered,
        "six_boundary_probes": boundaries,
    }


class _SyntheticQueryProvider:
    dimensions = 4

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.query_requests = 0
        self.document_requests = 0

    def embed_query(self, text):
        self.query_requests += 1
        return self._vectors[text]

    def embed_documents(self, _texts):
        self.document_requests += 1
        raise AssertionError("synthetic mmap verifier cannot request embeddings")


def _synthetic_vectors(chunks):
    rows = []
    for row in range(len(chunks)):
        raw = np.asarray([1.0 + row % 3, 1.0 + row % 5,
                          1.0 + row % 7, 1.0], dtype=np.float64)
        rows.append(raw / np.linalg.norm(raw))
    matrix = np.asarray(rows, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix, {
        "主体结构公式": [1.0, 0.0, 0.0, 0.0],
        "表格关键值": [0.0, 1.0, 0.0, 0.0],
        "共同过滤词": [0.0, 0.0, 1.0, 0.0],
    }


def _build_synthetic_packages(source_dir: Path, output_root: Path,
                              *, filler_count: int = 128):
    _write_synthetic_corpus(source_dir, filler_count=filler_count)
    core_dir = output_root / "core_corpus"
    bm25_dir = output_root / "core_bm25"
    build_core_corpus_mmap(source_dir, core_dir)
    build_index = EvidenceIndex(load_corpus(source_dir))
    try:
        build_core_bm25(bm25_dir, chunks=build_index.chunks)
    finally:
        build_index.local.close_bm25()
        del build_index
        gc.collect()
    return core_dir, bm25_dir


def _synthetic_stage() -> dict:
    queries = ["主体结构公式", "表格关键值", "共同过滤词"]
    with tempfile.TemporaryDirectory(prefix="rag-corpus-mmap-smoke-") as temporary:
        root = Path(temporary)
        source_dir = root / "source"
        core_dir, bm25_dir = _build_synthetic_packages(
            source_dir, root / "package", filler_count=128)

        equivalence_store = BinaryCorpusStore.open(
            core_dir, source_dir=source_dir, cache_bytes=8 * 1024)
        try:
            corpus_compare = _compare_records_and_chunks(source_dir, equivalence_store)
        finally:
            equivalence_store.close()

        store = BinaryCorpusStore.open(
            core_dir, source_dir=source_dir, cache_bytes=8 * 1024)
        legacy = EvidenceIndex(load_corpus(source_dir), bm25_directory=bm25_dir)
        mapped = EvidenceIndex(store.corpus, bm25_directory=bm25_dir)
        providers = []
        try:
            if not getattr(mapped.chunks, "is_lazy_chunk_sequence", False):
                raise AssertionError("EvidenceIndex did not retain the lazy chunk sequence")
            if len(legacy.chunks) != len(mapped.chunks):
                raise AssertionError("legacy/mapped EvidenceIndex chunk count differs")
            matrix, query_vectors = _synthetic_vectors(legacy.chunks)
            for index in (legacy, mapped):
                provider = _SyntheticQueryProvider(query_vectors)
                providers.append(provider)
                index.local.bind_prepared_vectors(matrix, embeddings=provider)
                index.embeddings = provider

            before_filter = store.cache_stats
            mapped._allowed_indices({"std_no": "GB 50000-2020", "kind": "clause"})
            after_filter = store.cache_stats
            if after_filter["chunk_decode_count"] != before_filter["chunk_decode_count"]:
                raise AssertionError("metadata filtering decoded lazy chunks")

            before_top8 = store.cache_stats["chunk_decode_count"]
            tied_first = mapped.retrieve("共同过滤词", mode="bm25", top_k=8)
            tied_second = mapped.retrieve("共同过滤词", mode="bm25", top_k=8)
            if _retrieval_payload(tied_first) != _retrieval_payload(tied_second):
                raise AssertionError("repeated tied BM25 top-8 ordering changed")
            top8_decodes = store.cache_stats["chunk_decode_count"] - before_top8
            if top8_decodes >= len(mapped.chunks):
                raise AssertionError("a bounded top-8 query decoded the full lazy sequence")

            legacy_results = _semantic_snapshot(legacy, queries)
            mapped_results = _semantic_snapshot(mapped, queries)
            if legacy_results != mapped_results:
                raise AssertionError("JSON and mmap synthetic retrieval results differ")
            stats_after = store.cache_stats
            if stats_after["cache_bytes"] > stats_after["cache_limit_bytes"]:
                raise AssertionError("synthetic record cache exceeded its byte limit")
            if stats_after["cache_evictions"] < 1:
                raise AssertionError("small synthetic cache did not exercise eviction")
            for provider in providers:
                if provider.document_requests or provider.query_requests != (
                        len(queries) + len(queries[:3]) * 3):
                    raise AssertionError("synthetic query-only provider request count differs")
        finally:
            for index in (legacy, mapped):
                index.local._vectors = None
                index.local.close_bm25()
            del legacy, mapped
            gc.collect()
            store.close()
        return {
            "stage": "synthetic",
            "corpus_equivalence": corpus_compare,
            "retrieval_modes": ["bm25_top30", "rrf_top8", "filtered_rrf_expansion"],
            "boundary_probe_count": len(_BOUNDARY_QUERIES),
            "full_hit_text_locator_equal": True,
            "filter_did_not_decode_chunks": True,
            "top8_decode_count": top8_decodes,
            "synthetic_chunk_count": corpus_compare["chunk_rows_compared"],
            "record_cache_bytes": stats_after["cache_bytes"],
            "record_cache_limit_bytes": stats_after["cache_limit_bytes"],
            "record_cache_evictions": stats_after["cache_evictions"],
            "cached_query_vector_lookups": sum(
                provider.query_requests for provider in providers),
            "document_embedding_requests": sum(provider.document_requests for provider in providers),
            "real_model_requests": 0,
            "external_network_requests": 0,
        }


@contextmanager
def _offline_network_scope():
    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
    )
    try:
        socket.socket.connect = _bm25._guarded_connect
        socket.socket.connect_ex = _bm25._guarded_connect_ex
        socket.create_connection = _bm25._guarded_create_connection
        yield
    finally:
        (socket.socket.connect, socket.socket.connect_ex,
         socket.create_connection) = originals


def _real_source_equivalence(args) -> dict:
    before = _source_snapshot(args.corpus_dir)
    store = BinaryCorpusStore.open(
        args.core_corpus_dir, source_dir=args.corpus_dir,
        cache_bytes=args.corpus_cache_bytes)
    try:
        comparison = _compare_records_and_chunks(args.corpus_dir, store)
        if _source_snapshot(args.corpus_dir) != before:
            raise AssertionError("JSON source files changed while comparing corpus mmap")
        comparison.update({
            "stage": "corpus_equivalence",
            "source_file_stats_unchanged": True,
            "comparison_strategy": (
                "one source JSON record and one mapped record/chunk at a time; "
                "no corpus-sized verification copy"),
            "store_state_before_close": store.cache_stats["state"],
        })
        return comparison
    finally:
        store.close()


def _semantic_backend(args, backend: str, queries: list[str],
                      query_vectors: dict[str, list[float]], identity: dict) -> dict:
    store = vector_store = index = provider = None
    stats_open = stats_bound = stats_query = None
    try:
        if backend == "json":
            corpus = load_corpus(args.corpus_dir)
        elif backend == "mmap":
            store = BinaryCorpusStore.open(
                args.core_corpus_dir, source_dir=args.corpus_dir,
                cache_bytes=args.corpus_cache_bytes)
            corpus = store.corpus
            stats_open = store.cache_stats
            if any(stats_open[key] for key in (
                    "record_decode_count", "chunk_decode_count",
                    "id_text_decode_count", "cache_bytes")):
                raise AssertionError("opening mapped corpus decoded runtime rows")
        else:
            raise ValueError("unknown corpus backend")

        index = EvidenceIndex(corpus, bm25_directory=args.bm25_dir)
        vector_store = BinaryVectorStore.open(
            args.vector_dir, identity=identity, chunks=index.chunks)
        provider = _bm25._CachedQueryProvider(query_vectors, identity["dimensions"])
        index.local.bind_prepared_vectors(vector_store.vectors, embeddings=provider)
        index.embeddings = provider
        if store is not None:
            stats_bound = store.cache_stats
            if stats_bound["record_decode_count"] or stats_bound["chunk_decode_count"]:
                raise AssertionError("BM25/vector startup materialized records or Chunk objects")

        retrieval = _semantic_snapshot(index, queries)
        if provider.document_requests != 0:
            raise AssertionError("semantic comparison attempted document embeddings")
        result = {
            "retrieval": retrieval,
            "cached_query_vector_lookups": provider.query_requests,
            "document_embedding_requests": provider.document_requests,
        }
        if store is not None:
            stats_query = store.cache_stats
            if stats_query["cache_bytes"] > stats_query["cache_limit_bytes"]:
                raise AssertionError("mapped record cache exceeded byte limit")
            result["corpus_runtime_stats"] = {
                "open": {
                    key: stats_open[key] for key in (
                        "record_decode_count", "chunk_decode_count",
                        "id_text_decode_count", "cache_bytes")
                },
                "after_index_and_vector_bind": {
                    key: stats_bound[key] for key in (
                        "record_decode_count", "chunk_decode_count",
                        "id_text_decode_count", "cache_bytes", "cache_limit_bytes")
                },
                "after_queries": {
                    key: stats_query[key] for key in (
                        "record_decode_count", "chunk_decode_count",
                        "id_text_decode_count", "cache_bytes",
                        "cache_limit_bytes")
                },
                "note": (
                    "id_text_decode_count includes bounded streaming for BM25/vector "
                    "mmap metadata binding; it is not a Chunk/record object decode count"
                ),
            }
        return result
    finally:
        if index is not None:
            index.local._vectors = None
            index.local.embeddings = None
            index.embeddings = None
            index.local.close_bm25()
            del index
            gc.collect()
        if vector_store is not None:
            vector_store.close()
        if store is not None:
            store.close()


def _semantic_stage(args) -> dict:
    memory_budget = _bm25._check_memory_budget(
        1, args.memory_per_worker_gib, args.reserve_gib)
    identity, query_vectors, cache_before = _bm25._load_full_queries(
        SimpleNamespace(embedding_cache=args.embedding_cache))
    if len(query_vectors) != _EXPECTED_CACHED_QUERY_COUNT:
        raise AssertionError(
            f"expected {_EXPECTED_CACHED_QUERY_COUNT} cached query vectors, "
            f"found {len(query_vectors)}")
    queries = list(query_vectors)
    expected_requests = len(queries) + len(queries[:3]) * 3
    source_before = _source_snapshot(args.corpus_dir)
    json_result = _semantic_backend(args, "json", queries, query_vectors, identity)
    mmap_result = _semantic_backend(args, "mmap", queries, query_vectors, identity)
    if (_source_snapshot(args.corpus_dir) != source_before
            or _bm25._source_snapshot(args.embedding_cache) != cache_before):
        raise AssertionError("corpus or core embedding source stats changed during verification")
    if json_result["retrieval"] != mmap_result["retrieval"]:
        raise AssertionError("JSON and mmap corpus BM25/RRF/full-hit results differ")
    for result in (json_result, mmap_result):
        if result["cached_query_vector_lookups"] != expected_requests:
            raise AssertionError("unexpected cached query-vector request count")
        if result["document_embedding_requests"] != 0:
            raise AssertionError("offline semantic verification requested document vectors")
    return {
        "stage": "semantic",
        "baseline": "JSON StandardCorpus + existing BM25 mmap",
        "candidate": "BinaryCorpusStore mmap + same BM25 mmap",
        "memory_preflight": memory_budget,
        "query_count": len(queries),
        "query_ids": [_bm25._query_id(query) for query in queries],
        "cached_query_vector_lookups": {
            "json": json_result["cached_query_vector_lookups"],
            "mmap": mmap_result["cached_query_vector_lookups"],
        },
        "boundary_probe_count": len(_BOUNDARY_QUERIES),
        "source_file_stats_unchanged": True,
        "embedding_cache_stats_unchanged": True,
        "query_vectors_streamed_only": True,
        "document_embedding_requests": 0,
        "real_model_requests": 0,
        "external_network_requests": 0,
        "full_hits_text_locator_and_expansion_equal": True,
        "bm25_score_arrays_top30_filters_and_rrf_top8_equal": True,
        "comparison_sha256": _digest(json_result["retrieval"]),
        "mapped_corpus_runtime_stats": mmap_result["corpus_runtime_stats"],
    }


def _result_digest(result) -> str:
    return _digest(_retrieval_payload(result))


def _load_worker_corpus(backend: str, source_dir: str | Path,
                        core_corpus_dir: str | Path,
                        corpus_cache_bytes: int):
    started = time.perf_counter()
    if backend == "json":
        return load_corpus(source_dir), None, time.perf_counter() - started
    if backend == "mmap":
        store = BinaryCorpusStore.open(
            core_corpus_dir, source_dir=source_dir,
            cache_bytes=corpus_cache_bytes)
        return store.corpus, store, time.perf_counter() - started
    raise ValueError("unknown corpus backend")


def _corpus_worker_entry(connection, *, backend: str, source_dir: str,
                         core_corpus_dir: str, bm25_dir: str, vector_dir: str,
                         identity: dict, query_vectors: dict[str, list[float]],
                         queries: list[str], repeats: int, corpus_cache_bytes: int,
                         expected: dict[str, str] | None) -> None:
    """Spawn target: wait for all peers before real EvidenceIndex retrieval."""
    _bm25._disable_external_network()
    store = vector_store = index = provider = None
    try:
        started = time.perf_counter()
        corpus, store, corpus_load_seconds = _load_worker_corpus(
            backend, source_dir, core_corpus_dir, corpus_cache_bytes)
        corpus_stats = None
        if store is not None:
            stats_open = store.cache_stats
            if any(stats_open[key] for key in (
                    "record_decode_count", "chunk_decode_count",
                    "id_text_decode_count", "cache_bytes")):
                raise AssertionError("corpus mmap open decoded runtime rows")
            corpus_stats = {"open": {
                key: stats_open[key] for key in (
                    "record_decode_count", "chunk_decode_count",
                    "id_text_decode_count", "cache_bytes", "cache_limit_bytes")
            }}

        index_started = time.perf_counter()
        index = EvidenceIndex(corpus, bm25_directory=bm25_dir)
        vector_store = BinaryVectorStore.open(
            vector_dir, identity=identity, chunks=index.chunks)
        provider = _bm25._CachedQueryProvider(query_vectors, identity["dimensions"])
        index.local.bind_prepared_vectors(vector_store.vectors, embeddings=provider)
        index.embeddings = provider
        index_seconds = time.perf_counter() - index_started
        if store is not None:
            stats_bound = store.cache_stats
            if stats_bound["record_decode_count"] or stats_bound["chunk_decode_count"]:
                raise AssertionError("mmap worker startup materialized records or Chunk objects")
            corpus_stats["after_index_and_vector_bind"] = {
                key: stats_bound[key] for key in (
                    "record_decode_count", "chunk_decode_count",
                    "id_text_decode_count", "cache_bytes", "cache_limit_bytes")
            }

        memory = _bm25._proc_memory()
        connection.send({
            "status": "ready", "pid": os.getpid(), "backend": backend,
            "chunk_count": len(index.chunks),
            "corpus_load_seconds": corpus_load_seconds,
            "bm25_and_vector_bind_seconds": index_seconds,
            "resource_load_seconds": time.perf_counter() - started,
            "ready_uss_bytes": memory["uss_bytes"],
            "hashseed": os.environ.get("PYTHONHASHSEED"),
            "corpus_stats": corpus_stats,
        })
        if connection.recv() != "go":
            raise RuntimeError("parent did not release query barrier")

        query_started = time.perf_counter()
        digests = {}
        query_latencies = {}
        mismatch_query_ids = []
        for _ in range(repeats):
            for query in queries:
                call_started = time.perf_counter()
                result = index.retrieve(query, mode="bm25", top_k=8)
                query_id = _bm25._query_id(query)
                digest = _result_digest(result)
                query_latencies.setdefault(query_id, []).append(
                    time.perf_counter() - call_started)
                digests.setdefault(query_id, []).append(digest)
                if expected is not None and digest != expected.get(query_id):
                    mismatch_query_ids.append(query_id)
        query_seconds = time.perf_counter() - query_started
        cached_query_vector_lookups = provider.query_requests
        document_embedding_requests = provider.document_requests
        if document_embedding_requests:
            raise AssertionError("BM25 concurrency stage requested document embeddings")
        if cached_query_vector_lookups:
            raise AssertionError("BM25 concurrency stage looked up cached query vectors")
        if store is not None:
            stats_query = store.cache_stats
            if stats_query["cache_bytes"] > stats_query["cache_limit_bytes"]:
                raise AssertionError("corpus record cache exceeded byte limit")
            corpus_stats["after_query_phase"] = {
                key: stats_query[key] for key in (
                    "record_decode_count", "chunk_decode_count",
                    "id_text_decode_count", "cache_bytes", "cache_limit_bytes")
            }
        result_message = {
            "status": "done", "pid": os.getpid(), "backend": backend,
            "query_count": len(queries) * repeats,
            "query_seconds": query_seconds,
            "query_latencies_seconds": query_latencies,
            "result_digests": digests,
            "mismatch_query_ids": mismatch_query_ids,
            "cached_query_vector_lookups": cached_query_vector_lookups,
            "document_embedding_requests": document_embedding_requests,
            "real_model_requests": 0,
            "external_network_requests": 0,
            "corpus_stats": corpus_stats,
        }
        index.local._vectors = None
        index.local.embeddings = None
        index.embeddings = None
        index.local.close_bm25()
        del index
        index = None
        gc.collect()
        vector_store.close()
        vector_store = None
        if store is not None:
            store.close()
            result_message["corpus_store_closed"] = store.cache_stats["state"] == "closed"
            store = None
        connection.send(result_message)
    except BaseException as exc:
        try:
            connection.send({
                "status": "error", "pid": os.getpid(),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        if index is not None:
            try:
                index.local._vectors = None
                index.local.close_bm25()
            except Exception:
                pass
        if vector_store is not None:
            try:
                vector_store.close()
            except Exception:
                pass
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        connection.close()


def _start_corpus_worker(ctx, *, backend, source_dir, core_corpus_dir,
                         bm25_dir, vector_dir, identity, query_vectors,
                         queries, repeats, corpus_cache_bytes, expected):
    parent, child = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_corpus_worker_entry,
        kwargs={
            "connection": child, "backend": backend,
            "source_dir": str(source_dir),
            "core_corpus_dir": str(core_corpus_dir),
            "bm25_dir": str(bm25_dir), "vector_dir": str(vector_dir),
            "identity": identity, "query_vectors": query_vectors,
            "queries": queries, "repeats": repeats,
            "corpus_cache_bytes": corpus_cache_bytes, "expected": expected,
        },
        name=f"corpus-mmap-{backend}-worker",
    )
    with patch.dict(os.environ, {"PYTHONHASHSEED": "0", **_THREAD_ENV}):
        process.start()
    child.close()
    return process, parent


def _sum_uss_sample(processes) -> dict:
    sample = _bm25._group_memory_sample(processes)
    return {
        "sum_uss_bytes": sample["sum_uss_bytes"],
        "available_system_bytes": sample["available_system_bytes"],
    }


def _run_corpus_group(args, *, backend, worker_count, queries,
                      identity, query_vectors, expected=None, repeats=3) -> dict:
    ctx = mp.get_context("spawn")
    processes, connections, results = [], [], {}
    started = time.perf_counter()
    try:
        for _ in range(worker_count):
            process, connection = _start_corpus_worker(
                ctx, backend=backend, source_dir=args.corpus_dir,
                core_corpus_dir=args.core_corpus_dir,
                bm25_dir=args.bm25_dir, vector_dir=args.vector_dir,
                identity=identity, query_vectors=query_vectors,
                queries=queries, repeats=repeats,
                corpus_cache_bytes=args.corpus_cache_bytes,
                expected=expected)
            processes.append(process)
            connections.append(connection)
        ready = _bm25._collect_ready(processes, connections, args.timeout_seconds)
        ready_sample = _sum_uss_sample(processes)
        query_phase_started = time.perf_counter()
        for connection in connections:
            connection.send("go")

        deadline = time.monotonic() + args.timeout_seconds
        pending = set(range(worker_count))
        peak_uss = ready_sample["sum_uss_bytes"]
        min_available = ready_sample["available_system_bytes"]
        while pending:
            sample = _sum_uss_sample(processes)
            if sample["sum_uss_bytes"] is not None:
                peak_uss = max(peak_uss or 0, sample["sum_uss_bytes"])
            min_available = min(min_available, sample["available_system_bytes"])
            for position in list(pending):
                connection = connections[position]
                if connection.poll(0):
                    message = connection.recv()
                    if message.get("status") != "done":
                        raise RuntimeError(f"corpus worker failed: {message}")
                    results[position] = message
                    pending.remove(position)
                elif processes[position].exitcode is not None:
                    raise RuntimeError(
                        f"corpus worker exited before result: pid={processes[position].pid}, "
                        f"exit={processes[position].exitcode}")
            if pending and time.monotonic() >= deadline:
                raise TimeoutError("corpus worker query phase exceeded timeout")
            if pending:
                time.sleep(_CONTROL_POLL_SECONDS)

        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                raise TimeoutError(f"corpus worker did not exit after result: pid={process.pid}")
            if process.exitcode != 0:
                raise RuntimeError(
                    f"corpus worker exited nonzero: pid={process.pid}, exit={process.exitcode}")
        query_phase_seconds = time.perf_counter() - query_phase_started
        workers = [results[position] for position in range(worker_count)]
        expected_per_query = expected or {
            query_id: digests[0]
            for query_id, digests in workers[0]["result_digests"].items()
        }
        mismatches = []
        for worker in workers:
            mismatches.extend(worker["mismatch_query_ids"])
            for query_id, digests in worker["result_digests"].items():
                if any(value != expected_per_query.get(query_id) for value in digests):
                    mismatches.append(query_id)
        if mismatches:
            raise AssertionError(
                "EvidenceIndex.retrieve JSON/mmap result digests differ for query IDs: "
                + ",".join(sorted(set(mismatches)))
            )
        query_latency_values = []
        query_latency_by_id = {}
        for worker in workers:
            for query_id, values in worker["query_latencies_seconds"].items():
                query_latency_by_id.setdefault(query_id, []).extend(values)
                query_latency_values.extend(values)
        total_query_calls = sum(worker["query_count"] for worker in workers)
        if not query_latency_values or query_phase_seconds <= 0:
            raise AssertionError("query latency or query phase timing samples are empty")

        def latency_summary(values):
            return {
                "sample_count": len(values),
                "p50_seconds": float(np.percentile(values, 50)),
                "p95_seconds": float(np.percentile(values, 95)),
            }

        return {
            "backend": backend,
            "worker_count": worker_count,
            "ready_workers": ready,
            "ready_phase_sum_uss_bytes": ready_sample["sum_uss_bytes"],
            "query_phase_peak_sampled_sum_uss_bytes": peak_uss,
            "query_phase_min_available_system_bytes": min_available,
            "query_phase_sample_interval_seconds": _CONTROL_POLL_SECONDS,
            "rss_is_not_summed": True,
            "query_rounds_per_worker": repeats,
            "queries_per_worker": len(queries),
            "query_calls_per_worker": len(queries) * repeats,
            "query_phase_seconds_including_result_collection_and_worker_close": (
                query_phase_seconds),
            "query_phase_total_calls": total_query_calls,
            "query_phase_calls_per_second_including_worker_close": (
                total_query_calls / query_phase_seconds),
            "cached_query_vector_lookups": sum(
                worker["cached_query_vector_lookups"] for worker in workers),
            "real_model_requests": 0,
            "external_network_requests": 0,
            "retrieve_plus_digest_latency_seconds": {
                **latency_summary(query_latency_values),
                "per_query_id": {
                    query_id: latency_summary(values)
                    for query_id, values in sorted(query_latency_by_id.items())
                },
            },
            "all_results_match_json_baseline": True,
            "all_workers_exited_before_group_return": True,
            "barrier_released_after_all_workers_ready": True,
            "workers": workers,
            "group_seconds": time.perf_counter() - started,
        }
    finally:
        for connection in connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in processes:
            if process.is_alive():
                # This handle belongs to the exact Process this helper started.
                process.terminate()
                process.join(timeout=5)


def _concurrency_stage(args) -> dict:
    memory_preflights = []
    identity, query_vectors, embedding_source_before = _bm25._load_full_queries(
        SimpleNamespace(embedding_cache=args.embedding_cache))
    if len(query_vectors) != _EXPECTED_CACHED_QUERY_COUNT:
        raise AssertionError(
            f"expected {_EXPECTED_CACHED_QUERY_COUNT} cached query vectors, "
            f"found {len(query_vectors)}")
    source_before = _source_snapshot(args.corpus_dir)
    queries = list(query_vectors)
    groups = []
    baseline_by_query = None
    for worker_count in args.process_counts:
        memory_preflights.append(_bm25._check_memory_budget(
            worker_count, args.memory_per_worker_gib, args.reserve_gib))
        json_group = _run_corpus_group(
            args, backend="json", worker_count=worker_count,
            queries=queries, identity=identity, query_vectors=query_vectors,
            expected=baseline_by_query, repeats=args.repeats)
        if baseline_by_query is None:
            baseline_by_query = {
                query_id: digests[0]
                for query_id, digests in json_group["workers"][0]["result_digests"].items()
            }
        mapped_group = _run_corpus_group(
            args, backend="mmap", worker_count=worker_count,
            queries=queries, identity=identity, query_vectors=query_vectors,
            expected=baseline_by_query, repeats=args.repeats)
        groups.extend([json_group, mapped_group])
    if (_bm25._source_snapshot(args.embedding_cache) != embedding_source_before
            or _source_snapshot(args.corpus_dir) != source_before):
        raise AssertionError("corpus or cached query source stats changed during concurrency stage")
    return {
        "stage": "concurrency",
        "baseline": "JSON StandardCorpus + existing BM25 mmap",
        "candidate": "BinaryCorpusStore mmap + same BM25 mmap",
        "process_counts": args.process_counts,
        "hashseed": "0",
        "thread_environment": _THREAD_ENV,
        "query_count": len(queries),
        "query_ids": [_bm25._query_id(query) for query in queries],
        "query_rounds_per_worker": args.repeats,
        "memory_preflights": memory_preflights,
        "groups": groups,
        "source_stats_unchanged": True,
        "memory_metric_note": (
            "USS is summed only for concurrently active workers; the query-phase "
            "peak is sampled, not a lifecycle peak; RSS is not summed"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("synthetic", "semantic", "memory", "concurrency", "all"),
        default="synthetic",
        help="default runs a small temporary corpus; real source stages require --confirm-large")
    parser.add_argument(
        "--confirm-large", action="store_true",
        help="explicitly allow reading real corpus/vector packages and spawning measurement workers")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument(
        "--core-corpus-dir", type=Path,
        default=ROOT / "data" / "runtime" / "core_corpus")
    parser.add_argument(
        "--bm25-dir", type=Path,
        default=ROOT / "data" / "runtime" / "core_bm25")
    parser.add_argument(
        "--vector-dir", type=Path,
        default=ROOT / "data" / "runtime" / "core_vectors")
    parser.add_argument(
        "--embedding-cache", type=Path,
        default=ROOT / "data" / "runtime" / "core_embeddings.json")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "corpus_mmap_optimization")
    parser.add_argument("--process-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--memory-per-worker-gib", type=float, default=1.5)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--corpus-cache-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def _validate_args(args) -> None:
    if args.stage != "synthetic" and not args.confirm_large:
        raise SystemExit("real corpus/vector/process stages require --confirm-large")
    if (not args.process_counts
            or any(isinstance(value, bool) or value < 1 for value in args.process_counts)):
        raise SystemExit("process counts must be positive")
    if (args.repeats < 1 or not math.isfinite(args.timeout_seconds)
            or args.timeout_seconds <= 0
            or not math.isfinite(args.memory_per_worker_gib)
            or args.memory_per_worker_gib <= 0
            or not math.isfinite(args.reserve_gib)
            or args.reserve_gib <= 0
            or args.corpus_cache_bytes < 0):
        raise SystemExit("repeats, timeout, RAM budget, and corpus cache limit are invalid")


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    for name in ("corpus_dir", "core_corpus_dir", "bm25_dir",
                 "vector_dir", "embedding_cache"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with patch.dict(os.environ, _THREAD_ENV), _offline_network_scope():
        if args.stage == "synthetic":
            _bm25._write_result(
                args.output_dir, "corpus_mmap_synthetic", _synthetic_stage())
            return 0

        if not args.core_corpus_dir.is_dir() or not (args.core_corpus_dir / "CURRENT").is_file():
            raise SystemExit(
                "prebuilt core_corpus mmap package is required; build it before this confirmed stage")
        for path, name in ((args.bm25_dir, "BM25"), (args.vector_dir, "vector")):
            if not path.is_dir():
                raise SystemExit(f"prebuilt {name} mmap package is missing")
        if not args.embedding_cache.is_file():
            raise SystemExit("the existing core embedding query cache is required")

        _bm25._check_memory_budget(
            1, args.memory_per_worker_gib, args.reserve_gib)
        _bm25._write_result(
            args.output_dir, "corpus_source_and_chunk_equivalence",
            _real_source_equivalence(args))
        if args.stage in {"semantic", "all"}:
            _bm25._write_result(
                args.output_dir, "corpus_mmap_semantic", _semantic_stage(args))
        if args.stage == "memory":
            args.process_counts = [1]
            _bm25._write_result(
                args.output_dir, "corpus_mmap_memory", _concurrency_stage(args))
        elif args.stage in {"concurrency", "all"}:
            _bm25._write_result(
                args.output_dir, "corpus_mmap_concurrency", _concurrency_stage(args))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
