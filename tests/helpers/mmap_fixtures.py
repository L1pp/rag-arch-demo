"""Synthetic-only fixtures shared by the public mmap regression tests."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import time

from rag_arch.binary_bm25 import build_core_bm25
from rag_arch.binary_corpus import BinaryCorpusStore, build_core_corpus_mmap
from rag_arch.evidence_index import EvidenceIndex, _record_chunk
from rag_arch.standards import load_corpus

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

def _iter_source_records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{line_number} must contain JSON objects")
                yield value

def _chunk_payload(chunk) -> dict:
    locator = copy.deepcopy(chunk.locator)
    locator.pop("collection_id", None)
    return {
        "doc_id": chunk.doc_id, "filename": chunk.filename,
        "chunk_id": chunk.chunk_id, "text": chunk.text,
        "locator": locator, "source_kind": chunk.source_kind,
    }

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
