"""Offline correctness and isolated-process measurements for BM25 mmap.

Default invocation runs a small synthetic smoke check. Real-corpus stages are
explicit and require --confirm-large; this module never builds the package or
calls an embedding provider.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import ipaddress
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

# Keep process-level CPU thread counts comparable. These are scoped to this
# verifier process and its children, not persisted to the caller's environment.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_env] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ijson
import numpy as np
import psutil

from rag_arch.binary_bm25 import build_core_bm25
from rag_arch.binary_vectors import (BinaryVectorStore,
                                     read_core_embedding_identity)
from rag_arch.evidence_index import EvidenceIndex
from rag_arch.models import Chunk
from rag_arch.retrieval import LocalIndex, tokenize
from rag_arch.standards import load_corpus


_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


def _is_loopback_address(address) -> bool:
    if isinstance(address, str):
        # Local named-pipe/Unix-domain endpoints are not external TCP hosts.
        return True
    if not isinstance(address, tuple) or not address:
        return True
    host = str(address[0]).lower().split("%", 1)[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guarded_connect(sock, address):
    if not _is_loopback_address(address):
        raise OSError("offline BM25 verification blocks external network connections")
    return _ORIGINAL_CONNECT(sock, address)


def _guarded_connect_ex(sock, address):
    if not _is_loopback_address(address):
        raise OSError("offline BM25 verification blocks external network connections")
    return _ORIGINAL_CONNECT_EX(sock, address)


def _guarded_create_connection(address, *args, **kwargs):
    if not _is_loopback_address(address):
        raise OSError("offline BM25 verification blocks external network connections")
    return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)


def _disable_external_network() -> None:
    # Keep loopback available for Windows asyncio/test infrastructure.
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.create_connection = _guarded_create_connection


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def _source_snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "file_id": getattr(stat, "st_ino", 0)}


def _query_id(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _load_cached_query_vectors(source: Path, expected_dimensions: int) -> dict[str, list[float]]:
    """Stream the large legacy JSON cache and retain only its query vectors.

    The large ``documents`` object is parsed as a stream and is never assembled
    into a Python dictionary or list. Only the (currently 28) query vectors are
    kept in memory for local RRF verification.
    """
    if not source.is_file():
        raise FileNotFoundError(f"query-vector source cache is missing: {source}")
    query_vectors: dict[str, list[float]] = {}
    with source.open("rb") as handle:
        for query, raw in ijson.kvitems(handle, "queries", buf_size=64 * 1024,
                                        use_float=True):
            if not isinstance(query, str) or query in query_vectors:
                raise ValueError("legacy query cache contains a duplicate or invalid key")
            vector = np.asarray(raw, dtype=np.float64)
            if (vector.ndim != 1 or vector.shape[0] != expected_dimensions
                    or not np.isfinite(vector).all()):
                raise ValueError("legacy cached query vector has invalid shape or values")
            query_vectors[query] = [float(value) for value in vector]
    if len(query_vectors) != 28:
        raise ValueError(f"expected the 28 existing cached query vectors; found {len(query_vectors)}")
    return query_vectors


class _CachedQueryProvider:
    """Query-only adapter over vectors streamed from the existing local cache."""
    def __init__(self, vectors: dict[str, list[float]], dimensions: int):
        self.vectors = vectors
        self.dimensions = dimensions
        self.query_requests = 0
        self.document_requests = 0

    def embed_documents(self, texts):
        self.document_requests += 1
        raise AssertionError("offline verifier cannot request document embeddings")

    def embed_query(self, text):
        self.query_requests += 1
        try:
            return self.vectors[text]
        except KeyError:
            raise AssertionError("offline verifier cannot request uncached query embeddings") from None


def _retrieval_snapshot(index: EvidenceIndex, query: str, *, mode: str,
                        top_k: int, filters: dict | None = None) -> dict:
    result = index.retrieve(query, mode=mode, top_k=top_k, filters=filters)
    return {
        "raw": [(hit.chunk.chunk_id, float(hit.score).hex()) for hit in result.raw_hits],
        "context": [(hit.chunk.chunk_id, float(hit.score).hex()) for hit in result.context_hits],
        "missing": result.missing_evidence,
    }


def _find_tied_query(index: EvidenceIndex) -> tuple[str, int] | None:
    counts = index.local._counts
    lengths = index.local._lengths
    if counts is None or lengths is None:
        return None
    seen: dict[tuple[str, int, int], int] = {}
    for doc_index, document_counts in enumerate(counts):
        for term, frequency in document_counts.items():
            if tokenize(term) != [term]:
                continue
            key = (term, frequency, lengths[doc_index])
            previous = seen.get(key)
            if previous is not None:
                # Both rows have identical tf and length for the same token,
                # so their BM25 score is exactly tied under the current formula.
                return term, previous
            seen[key] = doc_index
    return None


def _index_snapshots(index: EvidenceIndex, queries: list[str], *,
                     tie_probe: str | None) -> dict:
    lexical_queries = list(queries) + [
        "1.8≤18", "1.8 >= 18", "≤ ≥ ≠", "", "bm25_mmap_unseen_probe"]
    if tie_probe:
        lexical_queries.append(tie_probe)
    score_hashes = {}
    bm25_top30 = {}
    rrf_top8 = {}
    for query in lexical_queries:
        scores = np.asarray(index.local._bm25_scores(query), dtype="<f8")
        query_key = _query_id(query)
        score_hashes[query_key] = hashlib.sha256(scores.tobytes()).hexdigest()
        bm25_top30[query_key] = _retrieval_snapshot(
            index, query, mode="bm25", top_k=30)

    for query in queries:
        rrf_top8[_query_id(query)] = _retrieval_snapshot(
            index, query, mode="rrf", top_k=8)

    filter_cases = [
        {"kind": "table"},
        {"std_no": index.chunks[0].locator.get("std_no")},
        {"document_part": "body"},
    ]
    filtered = {}
    for query in queries[:3]:
        for filters in filter_cases:
            key = f"{_query_id(query)}:{hashlib.sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:8]}"
            filtered[key] = _retrieval_snapshot(
                index, query, mode="rrf", top_k=8, filters=filters)
    return {"score_array_hashes": score_hashes, "bm25_top30": bm25_top30,
            "rrf_top8": rrf_top8, "filtered_rrf_and_expansion": filtered}


def _signature(index: EvidenceIndex, query: str) -> dict:
    scores = np.asarray(index.local._bm25_scores(query), dtype="<f8")
    eligible = [i for i, chunk in enumerate(index.chunks)
                if not chunk.locator.get("excluded_from_current_retrieval", False)
                and scores[i] > 0]
    ordered = sorted(eligible, key=lambda i: (-float(scores[i]), index.chunks[i].chunk_id))[:30]
    return {"score_array_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
            "bm25_top30_ids": [index.chunks[i].chunk_id for i in ordered]}


def _proc_memory(pid: int | None = None) -> dict:
    process = psutil.Process(pid or os.getpid())
    info = process.memory_info()
    try:
        full = process.memory_full_info()
        uss = getattr(full, "uss", None)
        private_commit = getattr(full, "private", getattr(info, "private", None))
    except (psutil.Error, AttributeError):
        uss = None
        private_commit = getattr(info, "private", None)
    return {"rss_bytes": info.rss, "uss_bytes": uss,
            "private_commit_bytes": private_commit}


def _mapped_state(index: EvidenceIndex) -> dict:
    store = index.local._bm25_store
    if store is None:
        return {"backend": "memory", "all_arrays_mmap": False,
                "all_arrays_readonly": False}
    arrays = [store.offsets, store.postings, store.doc_lengths, store.doc_freqs]
    return {"backend": "mmap",
            "all_arrays_mmap": all(isinstance(array, np.memmap) for array in arrays),
            "all_arrays_readonly": all(not array.flags.writeable for array in arrays),
            "manifest_format": store.manifest.get("format"),
            "document_count": store.document_count,
            "vocabulary_count": store.vocabulary_count,
            "posting_count": store.posting_count}


def _worker_entry(connection, backend: str, corpus_dir: str, bm25_dir: str,
                  queries: list[str], repeats: int,
                  expected: dict[str, dict] | None) -> None:
    """Spawn target. Waits at a parent-controlled barrier after index loading."""
    _disable_external_network()
    index = None
    try:
        from rag_arch.evidence_index import EvidenceIndex as WorkerEvidenceIndex
        from rag_arch.standards import load_corpus as worker_load_corpus

        started = time.perf_counter()
        corpus_started = time.perf_counter()
        corpus = worker_load_corpus(corpus_dir)
        corpus_seconds = time.perf_counter() - corpus_started
        index_started = time.perf_counter()
        index = WorkerEvidenceIndex(
            corpus, bm25_directory=None if backend == "memory" else bm25_dir)
        index_seconds = time.perf_counter() - index_started
        ready = {
            "status": "ready", "pid": os.getpid(), "backend": backend,
            "chunk_count": len(index.chunks), "corpus_load_seconds": corpus_seconds,
            "bm25_index_seconds": index_seconds,
            "resource_load_seconds": time.perf_counter() - started,
            "memory_after_load": _proc_memory(), "mapping": _mapped_state(index),
            "hashseed": os.environ.get("PYTHONHASHSEED"),
        }
        connection.send(ready)
        if connection.recv() != "go":
            raise RuntimeError("parent did not release query barrier")

        latencies = []
        signatures = {}
        mismatch_ids = []
        for _ in range(repeats):
            for query in queries:
                before = time.perf_counter()
                signature = _signature(index, query)
                latencies.append(time.perf_counter() - before)
                query_key = _query_id(query)
                signatures[query_key] = signature
                if expected is not None and expected.get(query) != signature:
                    mismatch_ids.append(query_key)

        index.local.close_bm25()
        if index.local._bm25_store is not None and index.local._bm25_store._state != "closed":
            raise RuntimeError("BM25 mmap did not close before worker exit")
        connection.send({
            "status": "done", "pid": os.getpid(), "backend": backend,
            "query_count": len(queries) * repeats, "latencies_seconds": latencies,
            "signatures": signatures, "mismatch_query_ids": mismatch_ids,
            "mapping_closed": True,
        })
    except BaseException as exc:
        try:
            connection.send({"status": "error", "pid": os.getpid(),
                             "error_type": type(exc).__name__,
                             "error": str(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        if index is not None:
            try:
                index.local.close_bm25()
            except Exception:
                pass
        connection.close()


def _start_worker(ctx, backend, corpus_dir, bm25_dir, queries, repeats, expected):
    parent_connection, child_connection = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_worker_entry,
        args=(child_connection, backend, str(corpus_dir), str(bm25_dir),
              queries, repeats, expected))
    # PYTHONHASHSEED is read only when the spawned interpreter starts. Scoped
    # patching gives every child the same token accumulation order while leaving
    # the caller's Python process and environment unchanged afterward.
    with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
        process.start()
    child_connection.close()
    return process, parent_connection


def _collect_ready(processes, connections, timeout_seconds: float) -> list[dict]:
    deadline = time.monotonic() + timeout_seconds
    ready = []
    for process, connection in zip(processes, connections):
        while not connection.poll(0.05):
            if process.exitcode is not None:
                raise RuntimeError(f"worker exited before ready: pid={process.pid}, exit={process.exitcode}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"worker did not reach ready barrier: pid={process.pid}")
        message = connection.recv()
        if message.get("status") != "ready":
            raise RuntimeError(f"worker failed before ready: {message}")
        ready.append(message)
    return ready


def _group_memory_sample(processes) -> dict:
    uss_values = []
    private_values = []
    for process in processes:
        try:
            snapshot = _proc_memory(process.pid)
        except (psutil.Error, OSError):
            continue
        if snapshot["uss_bytes"] is not None:
            uss_values.append(snapshot["uss_bytes"])
        if snapshot["private_commit_bytes"] is not None:
            private_values.append(snapshot["private_commit_bytes"])
    return {"sum_uss_bytes": sum(uss_values) if len(uss_values) == len(processes) else None,
            "sum_private_commit_bytes": (sum(private_values)
                                         if len(private_values) == len(processes) else None),
            "available_system_bytes": psutil.virtual_memory().available}


def _run_group(*, backend: str, workers: int, corpus_dir: Path, bm25_dir: Path,
               queries: list[str], repeats: int, expected: dict[str, dict] | None,
               timeout_seconds: float) -> dict:
    ctx = mp.get_context("spawn")
    processes, connections = [], []
    start_memory = psutil.virtual_memory().available
    try:
        for _ in range(workers):
            process, connection = _start_worker(
                ctx, backend, corpus_dir, bm25_dir, queries, repeats, expected)
            processes.append(process)
            connections.append(connection)
        ready = _collect_ready(processes, connections, timeout_seconds)
        ready_sample = _group_memory_sample(processes)

        query_started = time.perf_counter()
        for connection in connections:
            connection.send("go")
        query_sample = _group_memory_sample(processes)
        deadline = time.monotonic() + timeout_seconds
        pending = set(range(workers))
        results = {}
        peak_query_sum_uss = query_sample["sum_uss_bytes"]
        peak_query_sum_private = query_sample["sum_private_commit_bytes"]
        min_available = min(ready_sample["available_system_bytes"],
                            query_sample["available_system_bytes"])
        while pending:
            sample = _group_memory_sample(processes)
            if sample["sum_uss_bytes"] is not None:
                peak_query_sum_uss = max(peak_query_sum_uss or 0,
                                         sample["sum_uss_bytes"])
            if sample["sum_private_commit_bytes"] is not None:
                peak_query_sum_private = max(peak_query_sum_private or 0,
                                             sample["sum_private_commit_bytes"])
            min_available = min(min_available, sample["available_system_bytes"])
            for index in list(pending):
                if connections[index].poll():
                    message = connections[index].recv()
                    if message.get("status") != "done":
                        raise RuntimeError(f"worker query phase failed: {message}")
                    results[index] = message
                    pending.remove(index)
            if time.monotonic() >= deadline:
                raise TimeoutError("worker query group exceeded timeout")
            if pending:
                time.sleep(0.05)

        query_seconds = time.perf_counter() - query_started
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                raise TimeoutError(f"worker did not exit after close: pid={process.pid}")
            if process.exitcode != 0:
                raise RuntimeError(f"worker exited abnormally: pid={process.pid}, exit={process.exitcode}")

        all_results = [results[i] for i in range(workers)]
        mismatch_ids = sorted({item for result in all_results
                               for item in result["mismatch_query_ids"]})
        if mismatch_ids:
            raise AssertionError(f"scores or Top30 differ from legacy reference: {mismatch_ids}")
        latencies = [value for result in all_results
                     for value in result["latencies_seconds"]]
        query_total = sum(result["query_count"] for result in all_results)
        sorted_latencies = sorted(latencies)
        p95 = (sorted_latencies[max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)]
               if sorted_latencies else 0.0)
        return {
            "backend": backend, "worker_count": workers,
            "barrier_ready_count": len(ready), "barrier_released_after_all_ready": len(ready) == workers,
            "all_workers_exited_before_next_group": all(not process.is_alive() for process in processes),
            "hashseed": "0", "query_count_per_worker": len(queries) * repeats,
            "input_query_count": len(queries), "repeats": repeats,
            "group_query_seconds": query_seconds,
            "total_queries_per_second": query_total / query_seconds if query_seconds else None,
            "verification_workload_latency_p95_seconds": p95,
            "latency_scope_note": "each sample includes a full score array, Top30 ordering, and digest; it is not a pure BM25 kernel or HTTP latency",
            "ready_sum_uss_bytes": ready_sample["sum_uss_bytes"],
            "peak_query_phase_sum_uss_bytes": peak_query_sum_uss,
            "peak_query_phase_sum_private_commit_bytes": peak_query_sum_private,
            "query_phase_memory_sampling_interval_seconds": 0.05,
            "memory_peak_scope_note": "sampled only after the ready barrier is released; loading transients are excluded, and this is not a continuous lifetime peak",
            "system_available_before_bytes": start_memory,
            "system_available_min_bytes": min_available,
            "workers": [{key: value for key, value in result.items()
                          if key not in {"latencies_seconds", "signatures"}}
                         | {"signatures": result["signatures"]}
                         for result in all_results],
            "ready_worker_metrics": [{
                "pid": item["pid"], "resource_load_seconds": item["resource_load_seconds"],
                "corpus_load_seconds": item["corpus_load_seconds"],
                "bm25_index_seconds": item["bm25_index_seconds"],
                "memory_after_load": item["memory_after_load"],
                "mapping": item["mapping"], "hashseed": item["hashseed"],
            } for item in ready],
        }
    finally:
        for connection in connections:
            try:
                connection.close()
            except Exception:
                pass
        for process in processes:
            if process.is_alive():
                # These are Process handles created and owned by this verifier.
                process.terminate()
                process.join(timeout=5)
            try:
                process.close()
            except ValueError:
                pass


def _check_memory_budget(worker_count: int, per_worker_gib: float,
                         reserve_gib: float) -> dict:
    available = psutil.virtual_memory().available
    required = int((worker_count * per_worker_gib + reserve_gib) * (1024 ** 3))
    result = {"worker_count": worker_count, "available_bytes": available,
              "required_bytes": required, "per_worker_budget_gib": per_worker_gib,
              "reserve_gib": reserve_gib, "passed": available >= required}
    if not result["passed"]:
        raise MemoryError(
            f"not enough available RAM for {worker_count} workers: "
            f"available={available / (1024 ** 3):.2f} GiB, "
            f"required={required / (1024 ** 3):.2f} GiB "
            f"({worker_count}*{per_worker_gib}+{reserve_gib})")
    return result


def _load_full_queries(args):
    source = args.embedding_cache
    before = _source_snapshot(source)
    identity = read_core_embedding_identity(source)
    queries = _load_cached_query_vectors(source, identity["dimensions"])
    return identity, queries, before


def _semantic_stage(args, query_vectors: dict[str, list[float]], identity: dict,
                    source_before: dict) -> dict:
    memory_budget = _check_memory_budget(1, args.memory_per_worker_gib,
                                         args.reserve_gib)
    source = args.embedding_cache
    corpus = load_corpus(args.corpus_dir)
    queries = list(query_vectors)
    lexical_probes = ["1.8≤18", "1.8 >= 18", "≤ ≥ ≠", "", "bm25_mmap_unseen_probe"]
    reference_index = EvidenceIndex(corpus)
    tie = _find_tied_query(reference_index)
    tie_query = tie[0] if tie else None
    vector_store = None
    baseline_snapshots = None
    mapped_snapshots = None
    query_requests = document_requests = 0
    baseline_chunk_rows = [(chunk.chunk_id, chunk.text) for chunk in reference_index.chunks]
    try:
        vector_store = BinaryVectorStore.open(
            args.vector_dir, identity=identity, chunks=reference_index.chunks)
        baseline_provider = _CachedQueryProvider(query_vectors, identity["dimensions"])
        reference_index.local.bind_prepared_vectors(vector_store.vectors,
                                                    embeddings=baseline_provider)
        reference_index.embeddings = baseline_provider
        baseline_snapshots = _index_snapshots(
            reference_index, queries, tie_probe=tie_query)
        query_requests += baseline_provider.query_requests
        document_requests += baseline_provider.document_requests
        del baseline_provider
        reference_index.local.close_bm25()
        reference_index.local._vectors = None
        del reference_index
        reference_index = None
        gc.collect()

        mapped_index = EvidenceIndex(corpus, bm25_directory=args.bm25_dir)
        try:
            if baseline_chunk_rows != [(chunk.chunk_id, chunk.text) for chunk in mapped_index.chunks]:
                raise AssertionError("memory and mmap EvidenceIndex chunk sequences differ")
            mapped_provider = _CachedQueryProvider(query_vectors, identity["dimensions"])
            mapped_index.local.bind_prepared_vectors(vector_store.vectors,
                                                      embeddings=mapped_provider)
            mapped_index.embeddings = mapped_provider
            mapped_snapshots = _index_snapshots(
                mapped_index, queries, tie_probe=tie_query)
            query_requests += mapped_provider.query_requests
            document_requests += mapped_provider.document_requests
            del mapped_provider
        finally:
            mapped_index.local.close_bm25()
            del mapped_index
            gc.collect()
    finally:
        if reference_index is not None:
            reference_index.local.close_bm25()
            reference_index.local._vectors = None
            del reference_index
            gc.collect()
        if vector_store is not None:
            vector_store.close()

    if baseline_snapshots != mapped_snapshots:
        raise AssertionError("BM25 score arrays, Top30, RRF Top8, filters, or evidence expansion differ")
    after = _source_snapshot(source)
    if source_before != after:
        raise AssertionError("source embedding cache metadata changed during offline verification")
    # Two index passes, each with 28 unfiltered RRF and 3x3 filtered queries.
    expected_requests = 2 * (len(queries) + len(queries[:3]) * 3)
    if query_requests != expected_requests:
        raise AssertionError(f"unexpected local cached-query count: {query_requests}")
    if document_requests != 0:
        raise AssertionError("offline semantic stage attempted document embedding")
    return {
        "stage": "semantic", "corpus_chunk_count": len(baseline_chunk_rows),
        "memory_preflight": memory_budget,
        "cached_query_count": len(queries), "cached_query_ids": [_query_id(query) for query in queries],
        "lexical_probe_count": len(lexical_probes) + (1 if tie_query else 0),
        "tie_probe_found": tie is not None,
        "tie_probe_id": _query_id(tie_query) if tie_query else None,
        "bm25_score_arrays_exact": True, "bm25_top30_exact": True,
        "rrf_top8_exact": True, "filter_and_expansion_exact": True,
        "query_embedding_requests": 0, "document_embedding_requests": 0,
        "query_vectors_streamed_only": True,
        "source_embedding_cache_unchanged": True,
        "source_cache_size_bytes": source_before["size_bytes"],
    }


def _reference_signatures(args, queries: list[str], timeout_seconds: float) -> tuple[dict, dict]:
    _check_memory_budget(1, args.memory_per_worker_gib, args.reserve_gib)
    result = _run_group(
        backend="memory", workers=1, corpus_dir=args.corpus_dir,
        bm25_dir=args.bm25_dir, queries=queries, repeats=1, expected=None,
        timeout_seconds=timeout_seconds)
    signatures = result["workers"][0]["signatures"]
    if len(signatures) != len(queries):
        raise AssertionError("legacy process did not return every query signature")
    # Key reference results by text for workers; source query text is never
    # copied into persisted JSON or printed to the console.
    mapped = {query: signatures[_query_id(query)] for query in queries}
    return mapped, result


def _run_memory_stage(args, queries: list[str], expected: dict[str, dict]) -> dict:
    groups = []
    preflights = []
    for backend in ("memory", "mmap"):
        preflights.append(_check_memory_budget(
            1, args.memory_per_worker_gib, args.reserve_gib))
        group = _run_group(
            backend=backend, workers=1, corpus_dir=args.corpus_dir,
            bm25_dir=args.bm25_dir, queries=queries, repeats=1,
            expected=expected, timeout_seconds=args.timeout_seconds)
        groups.append(group)
    if not all(worker["mapping_closed"] for group in groups for worker in group["workers"]):
        raise AssertionError("a measurement worker exited without closing its BM25 resource")
    return {"stage": "memory", "memory_preflights": preflights,
            "groups": groups,
            "scope": "one spawned process; corpus plus BM25 only; excludes dense vectors, catalog, PDF and API runtime",
            "load_order_note": "memory then mmap; OS file cache is not flushed, so report load times as sequential warm-cache observations"}


def _run_concurrency_stage(args, queries: list[str], expected: dict[str, dict]) -> dict:
    groups = []
    preflights = []
    for count in args.process_counts:
        for backend in ("memory", "mmap"):
            budget = _check_memory_budget(count, args.memory_per_worker_gib,
                                          args.reserve_gib)
            preflights.append(budget)
            group = _run_group(
                backend=backend, workers=count, corpus_dir=args.corpus_dir,
                bm25_dir=args.bm25_dir, queries=queries, repeats=args.repeats,
                expected=expected, timeout_seconds=args.timeout_seconds)
            if not group["barrier_released_after_all_ready"]:
                raise AssertionError("query group did not wait for every worker at the barrier")
            if not group["all_workers_exited_before_next_group"]:
                raise AssertionError("previous worker group was not fully stopped")
            groups.append(group)
    return {"stage": "concurrency", "process_counts": args.process_counts,
            "hashseed": "0", "thread_environment": {
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1"},
            "input_query_ids": [_query_id(query) for query in queries],
            "input_query_count": len(queries), "repeats": args.repeats,
            "memory_preflights": preflights, "groups": groups,
            "memory_metric_note": "sum_uss is reported per simultaneously active group; RSS is not summed or labeled physical memory"}


def _small_synthetic_stage(args) -> dict:
    chunks = [Chunk("synthetic", "sample.txt", f"synthetic/{i:04d}",
                    f"模板支撑项目 {i % 37} 的设计值为1.8，校核应符合要求。",
                    {}, "synthetic", "smoke") for i in range(1200)]
    queries = ["模板支撑", "1.8", "设计值", "bm25_mmap_oov"]
    with tempfile.TemporaryDirectory(prefix="bm25-mmap-smoke-") as temporary:
        directory = Path(temporary) / "core_bm25"
        build_core_bm25(directory, chunks=chunks)
        memory = LocalIndex(chunks)
        mapped = LocalIndex(chunks, bm25_directory=directory)
        try:
            for query in queries:
                if memory._bm25_scores(query) != mapped._bm25_scores(query):
                    raise AssertionError(f"synthetic score mismatch for query id {_query_id(query)}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                actual = list(pool.map(mapped._bm25_scores, queries * 4))
            expected = [memory._bm25_scores(query) for query in queries * 4]
            if actual != expected:
                raise AssertionError("synthetic concurrent mmap reads differ from memory reference")
            readonly = _mapped_state(SimpleNamespace(local=mapped))
        finally:
            mapped.close_bm25()
    return {"stage": "synthetic", "chunk_count": len(chunks),
            "query_count": len(queries), "thread_read_count": len(queries) * 4,
            "scores_exact": True, "concurrent_thread_reads_exact": True,
            "readonly_mapping": readonly}


def _write_result(output_dir: Path, name: str, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version.split()[0], "stage_result": result}
    _atomic_json(output_dir / f"{name}.json", result)
    _append_log(output_dir / "verify.log",
                f"stage={name} status=passed result={json.dumps(result['stage_result'], ensure_ascii=False, separators=(',', ':'))}")
    print(json.dumps(result, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("synthetic", "semantic", "memory",
                                              "concurrency", "all"),
                        default="synthetic",
                        help="default is a small synthetic smoke test; real stages are explicit")
    parser.add_argument("--confirm-large", action="store_true",
                        help="required to load the real corpus or start measurement processes")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts" / "bm25_mmap_optimization")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--bm25-dir", type=Path,
                        default=ROOT / "data" / "runtime" / "core_bm25")
    parser.add_argument("--vector-dir", type=Path,
                        default=ROOT / "data" / "runtime" / "core_vectors")
    parser.add_argument("--embedding-cache", type=Path,
                        default=ROOT / "data" / "runtime" / "core_embeddings.json")
    parser.add_argument("--process-counts", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--memory-per-worker-gib", type=float, default=1.5)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stage != "synthetic" and not args.confirm_large:
        raise SystemExit("real corpus/process stages require --confirm-large")
    if not args.process_counts or any(count < 1 for count in args.process_counts):
        raise SystemExit("process counts must be positive")
    if (args.repeats < 1 or not math.isfinite(args.timeout_seconds)
            or args.timeout_seconds <= 0
            or not math.isfinite(args.memory_per_worker_gib)
            or args.memory_per_worker_gib <= 0
            or not math.isfinite(args.reserve_gib) or args.reserve_gib <= 0):
        raise SystemExit("repeats and timeout must be positive")
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    for name in ("corpus_dir", "bm25_dir", "vector_dir", "embedding_cache"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    _disable_external_network()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "synthetic":
        _write_result(args.output_dir, "synthetic", _small_synthetic_stage(args))
        return 0

    identity, query_vectors, source_before = _load_full_queries(args)
    queries = list(query_vectors)
    if args.stage in {"semantic", "all"}:
        result = _semantic_stage(args, query_vectors, identity, source_before)
        _write_result(args.output_dir, "semantic", result)
    if args.stage in {"memory", "concurrency", "all"}:
        expected, reference = _reference_signatures(args, queries, args.timeout_seconds)
        _write_result(args.output_dir, "legacy_reference", {
            "stage": "legacy_reference", "query_count": len(queries),
            "hashseed": "0", "worker_count": 1,
            "all_signatures_generated": len(expected) == len(queries),
            "resource_load_seconds": [item["resource_load_seconds"]
                                      for item in reference["ready_worker_metrics"]],
        })
        if args.stage in {"memory", "all"}:
            _write_result(args.output_dir, "memory",
                          _run_memory_stage(args, queries, expected))
        if args.stage in {"concurrency", "all"}:
            _write_result(args.output_dir, "concurrency",
                          _run_concurrency_stage(args, queries, expected))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
