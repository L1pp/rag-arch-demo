"""Immutable, read-only mmap package for the public standards corpus.

The package keeps source records and generated chunk text in separate blobs.
Runtime access decodes one record/chunk at a time; the complete standards JSONL
is never reconstructed as Python dictionaries or Counters.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np

from .standards import StandardCorpus, _standard_key


_FORMAT = "rag_arch.core_corpus_mmap"
_FORMAT_VERSION = 1
# Bump this version when _record_chunk or its _readable_* helper semantics
# change; persisted chunk text must be regenerated under the new formatter.
_CHUNK_FORMAT_VERSION = 1
_CHUNK_BUILDER_ID = "rag_arch.evidence_index._record_chunk"
_CURRENT_RE = re.compile(r"^v1-[0-9a-f]{32}$")
_SOURCE_FILES = ("clauses.jsonl", "tables.jsonl", "summary.json", "catalog.json", "relations.json")
_PACKAGE_FILES = (
    "clauses.records.bin", "tables.records.bin",
    "clauses.record_offsets.npy", "tables.record_offsets.npy",
    "clauses.metadata.jsonl", "tables.metadata.jsonl",
    "clauses.status_proof.bin", "clauses.status_proof_offsets.npy",
    "chunks.text.bin", "chunks.text_offsets.npy", "chunks.map.npy",
    "chunks.metadata.jsonl", "source_records.jsonl", "corpus_meta.json",
)
_SOURCE_LOCATION_FIELDS = ("source_spans", "image_refs", "asset_refs", "source_page_images")
_SOURCE_PATH_KEYS = ("source_path", "asset_path", "image_path", "file")
_STATUS_FIELDS = ("std_no", "clause_no", "document_part", "import_origin",
                  "original_mandatory", "exact_clause_lookup")
_META_FIELDS = (
    "clause_id", "table_id", "std_no", "std_id", "clause_no", "table_no",
    "document_part", "import_origin", "exact_clause_lookup", "original_mandatory",
    "original_mandatory_evidence_present", "record_kind", "import_part_id",
    "content_status", "content_readable", "excluded_from_current_retrieval",
)
_CHUNK_META_FIELDS = (
    "chunk_id", "std_no", "std_id", "evidence_kind", "document_part",
    "excluded_from_current_retrieval", "import_part_id", "clause_no", "table_id",
)
_LOW_CARD_METADATA_FIELDS = frozenset({
    "std_no", "std_id", "document_part", "record_kind", "content_status",
    "evidence_kind",
})
_CANONICAL_METADATA_KEY_NAMES = frozenset({
    *_META_FIELDS, *_CHUNK_META_FIELDS, "chunk_included", "source_kind", "source_row",
    "std_no", *_SOURCE_LOCATION_FIELDS, *_SOURCE_PATH_KEYS,
    "source_id", "logical_page", "document_page", "document_page_start",
    "document_page_end", "page", "page_start", "page_end", "block",
    "block_start", "block_end", "candidate_path", "role",
})
_CANONICAL_METADATA_KEYS = {
    name: sys.intern(name) for name in _CANONICAL_METADATA_KEY_NAMES
}
_CACHE_STATS_FIELDS = (
    "record_decode_count", "chunk_decode_count", "id_text_decode_count",
    "source_projection_reads", "source_validation_records", "cache_hits",
    "cache_misses", "cache_evictions", "cache_bytes", "cache_limit_bytes",
    "active_readers", "state",
)
_UINT64_MAX = int(np.iinfo(np.uint64).max)


class BinaryCorpusStoreError(ValueError):
    """The current corpus package is absent, stale, malformed, or closed."""


def _validate_summary_documents(summary: dict) -> None:
    """Validate the small metadata projection before it is constructed."""
    documents = summary.get("documents", [])
    if not isinstance(documents, list) or any(not isinstance(row, dict) for row in documents):
        raise BinaryCorpusStoreError("summary.documents须为对象列表。")


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and bool(is_junction()))


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        raise BinaryCorpusStoreError("corpus包文件缺失或无法stat。") from exc


def _json_no_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BinaryCorpusStoreError("JSON对象包含重复字段。")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BinaryCorpusStoreError("corpus manifest不是有效JSON。") from exc
    if not isinstance(value, dict):
        raise BinaryCorpusStoreError("corpus manifest格式无效。")
    return value


def _load_small_source_metadata(source_dir: Path) -> dict:
    summary_path = source_dir / "summary.json"
    catalog_path = source_dir / "catalog.json"
    relations_path = source_dir / "relations.json"
    try:
        summary = (json.loads(summary_path.read_text(encoding="utf-8"),
                              object_pairs_hook=_json_no_duplicate_pairs)
                   if summary_path.exists() else {})
        catalog = (json.loads(catalog_path.read_text(encoding="utf-8-sig"),
                              object_pairs_hook=_json_no_duplicate_pairs).get("standards", [])
                   if catalog_path.exists() else [])
        relations = (json.loads(relations_path.read_text(encoding="utf-8-sig"),
                                object_pairs_hook=_json_no_duplicate_pairs).get("relations", [])
                     if relations_path.exists() else [])
    except (OSError, UnicodeError, json.JSONDecodeError, BinaryCorpusStoreError) as exc:
        raise BinaryCorpusStoreError("summary/catalog/relations源文件无效。") from exc
    if not isinstance(summary, dict) or not isinstance(catalog, list) or not isinstance(relations, list):
        raise BinaryCorpusStoreError("summary/catalog/relations源数据形状无效。")
    _validate_summary_documents(summary)
    return {"summary": summary, "catalog": catalog, "relations": relations}


def _check_source_files(source_dir: Path) -> None:
    for name in _SOURCE_FILES:
        path = source_dir / name
        if _is_link_or_junction(path):
            raise BinaryCorpusStoreError("corpus源文件不得为符号链接或junction。")
        try:
            exists = path.exists()
            if not exists:
                if name in {"clauses.jsonl", "tables.jsonl"}:
                    raise BinaryCorpusStoreError(f"缺少必要语料文件：{name}")
                continue
            if not path.is_file():
                raise BinaryCorpusStoreError(f"corpus源路径不是普通文件：{name}")
        except OSError as exc:
            raise BinaryCorpusStoreError(f"无法读取corpus源文件stat：{name}") from exc


def _resolve_current(directory: Path) -> tuple[str, Path]:
    if _is_link_or_junction(directory):
        raise BinaryCorpusStoreError("corpus mmap目录不得为符号链接或junction。")
    pointer = directory / "CURRENT"
    if _is_link_or_junction(pointer):
        raise BinaryCorpusStoreError("corpus CURRENT不得为符号链接。")
    try:
        content = pointer.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise BinaryCorpusStoreError("corpus CURRENT缺失或不可读。") from exc
    version_id = content[:-1] if content.endswith("\n") else content
    if (content not in (version_id, version_id + "\n")
            or not _CURRENT_RE.fullmatch(version_id)):
        raise BinaryCorpusStoreError("corpus CURRENT格式无效。")
    versions = directory / "versions"
    version_dir = versions / version_id
    try:
        root_resolved = directory.resolve(strict=True)
        versions_resolved = versions.resolve(strict=True)
        version_resolved = version_dir.resolve(strict=True)
        versions_resolved.relative_to(root_resolved)
        version_resolved.relative_to(versions_resolved)
    except (OSError, ValueError) as exc:
        raise BinaryCorpusStoreError("corpus CURRENT指向目录外或不存在的版本。") from exc
    if any(_is_link_or_junction(path) for path in (versions, version_dir)):
        raise BinaryCorpusStoreError("corpus版本目录不得为junction或符号链接。")
    return version_id, version_dir


def _validate_manifest(manifest: dict, version_id: str) -> None:
    if (manifest.get("format") != _FORMAT
            or type(manifest.get("format_version")) is not int
            or manifest.get("format_version") != _FORMAT_VERSION
            or manifest.get("version_id") != version_id
            or manifest.get("chunk_builder_id") != _CHUNK_BUILDER_ID
            or type(manifest.get("chunk_format_version")) is not int
            or manifest.get("chunk_format_version") != _CHUNK_FORMAT_VERSION):
        raise BinaryCorpusStoreError("corpus包格式、版本或Chunk构建规则不匹配。")
    counts = manifest.get("record_counts")
    if (not isinstance(counts, dict) or set(counts) != {"clauses", "tables"}
            or any(type(counts[name]) is not int or counts[name] < 0 for name in counts)):
        raise BinaryCorpusStoreError("manifest record_counts无效。")
    for field in ("chunk_count", "unique_text_count", "source_record_count"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise BinaryCorpusStoreError(f"manifest {field}无效。")
    if manifest["unique_text_count"] > manifest["chunk_count"]:
        raise BinaryCorpusStoreError("unique_text_count不能大于chunk_count。")
    row_ids = manifest.get("row_ids")
    if (type(row_ids) is not list or len(row_ids) != manifest["chunk_count"]
            or any(not isinstance(value, str) or not value for value in row_ids)
            or len(set(row_ids)) != len(row_ids)):
        raise BinaryCorpusStoreError("manifest chunk row_ids无效。")


def _validate_package_files(version_dir: Path, files: dict) -> None:
    if not isinstance(files, dict) or set(files) != set(_PACKAGE_FILES):
        raise BinaryCorpusStoreError("manifest派生文件清单不完整或包含未知路径。")
    for name in _PACKAGE_FILES:
        path = version_dir / name
        if _is_link_or_junction(path):
            raise BinaryCorpusStoreError(f"corpus包文件不得为链接：{name}")
        entry = files[name]
        if not isinstance(entry, dict):
            raise BinaryCorpusStoreError(f"manifest文件条目无效：{name}")
        if type(entry.get("file_bytes")) is not int or entry["file_bytes"] != _file_size(path):
            raise BinaryCorpusStoreError(f"corpus包文件尺寸无效：{name}")
        # Older packages may include a digest field; it is neither required nor checked.
        fields = set(entry) - {"sha256"}
        if name.endswith(".bin"):
            logical = entry.get("logical_bytes")
            if (fields != {"file_bytes", "logical_bytes"}
                    or type(logical) is not int or logical < 0
                    or entry["file_bytes"] != max(1, logical)):
                raise BinaryCorpusStoreError(f"corpus blob逻辑长度无效：{name}")
        elif name.endswith(".npy"):
            if not isinstance(entry.get("shape"), list) or any(
                    type(value) is not int or value < 0 for value in entry["shape"]):
                raise BinaryCorpusStoreError(f"corpus NPY shape无效：{name}")
            if name == "chunks.map.npy":
                if (fields != {"file_bytes", "shape", "fields"}
                        or entry.get("fields") != {"kind": "uint8", "row": "uint64"}):
                    raise BinaryCorpusStoreError("chunks.map.npy dtype无效。")
            elif (fields != {"file_bytes", "shape", "dtype"}
                    or entry.get("dtype") != np.dtype(np.uint64).str):
                raise BinaryCorpusStoreError(f"corpus offsets dtype无效：{name}")
        elif fields != {"file_bytes"}:
            raise BinaryCorpusStoreError(f"corpus sidecar文件字段无效：{name}")


def _validate_offset_array(array: np.memmap, count: int, logical_bytes: int, name: str,
                           *, allow_empty: bool = False) -> None:
    if (array.shape != (count + 1,) or array.dtype != np.dtype(np.uint64)
            or array.flags.writeable or not array.flags.c_contiguous
            or int(array[0]) != 0 or int(array[-1]) != logical_bytes
            or np.any(array[1:] < array[:-1])):
        raise BinaryCorpusStoreError(f"{name} shape/dtype/offset边界无效。")
    if count and not allow_empty and np.any(array[1:] == array[:-1]):
        raise BinaryCorpusStoreError(f"{name}含空记录偏移。")


def _map_blob(path: Path, logical_bytes: int) -> np.memmap:
    try:
        blob = np.memmap(path, mode="r", dtype=np.uint8, shape=(max(1, logical_bytes),))
    except (OSError, ValueError) as exc:
        raise BinaryCorpusStoreError(f"{path.name}无法只读映射。") from exc
    if blob.flags.writeable or blob.nbytes != max(1, logical_bytes):
        mmap = getattr(blob, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()
        raise BinaryCorpusStoreError(f"{path.name}映射长度无效。")
    if logical_bytes == 0 and int(blob[0]) != 0:
        blob._mmap.close()
        raise BinaryCorpusStoreError(f"{path.name}空blob sentinel无效。")
    return blob


def _close_mmap(array) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None and not mmap.closed:
        mmap.close()


def _validate_projection_shapes(manifest: dict, record_meta: dict[str, list[dict]],
                                chunk_meta: list[dict], source_records: list[dict],
                                arrays: dict[str, np.memmap]) -> None:
    counts = manifest["record_counts"]
    files = manifest["files"]
    for kind, id_field in (("clauses", "clause_id"), ("tables", "table_id")):
        rows = record_meta[kind]
        if len(rows) != counts[kind]:
            raise BinaryCorpusStoreError(f"{kind}.metadata.jsonl行数不匹配。")
        _validate_offset_array(
            arrays[f"{kind}.record_offsets.npy"], counts[kind],
            files[f"{kind}.records.bin"]["logical_bytes"],
            f"{kind}.record_offsets.npy")
        seen = set()
        expected_fields = set(_META_FIELDS) | {"chunk_included"}
        for row in rows:
            if (not isinstance(row, dict) or set(row) != expected_fields
                    or not isinstance(row.get(id_field), str) or not row[id_field]
                    or not isinstance(row.get("std_no"), str) or not row["std_no"]
                    or type(row.get("chunk_included")) is not bool
                    or type(row.get("original_mandatory_evidence_present")) is not bool):
                raise BinaryCorpusStoreError(f"{kind}.metadata.jsonl投影字段无效。")
            if row[id_field] in seen:
                raise BinaryCorpusStoreError(f"{kind} source record ID重复。")
            seen.add(row[id_field])

    expected_chunk_map = []
    for kind_code, kind, id_field in ((0, "clauses", "clause_id"),
                                      (1, "tables", "table_id")):
        for source_row, meta in enumerate(record_meta[kind]):
            if meta["chunk_included"]:
                expected_chunk_map.append((kind_code, source_row, kind, id_field, meta))
    chunk_count = manifest["chunk_count"]
    if len(chunk_meta) != chunk_count or len(expected_chunk_map) != chunk_count:
        raise BinaryCorpusStoreError("chunk metadata映射行数不匹配。")
    _validate_offset_array(
        arrays["chunks.text_offsets.npy"], chunk_count,
        files["chunks.text.bin"]["logical_bytes"], "chunks.text_offsets.npy")
    proof_offsets = arrays["clauses.status_proof_offsets.npy"]
    _validate_offset_array(
        proof_offsets, counts["clauses"], files["clauses.status_proof.bin"]["logical_bytes"],
        "clauses.status_proof_offsets.npy", allow_empty=True)
    for row, record in enumerate(record_meta["clauses"]):
        proof_bytes = int(proof_offsets[row + 1] - proof_offsets[row])
        if (record["original_mandatory_evidence_present"] and proof_bytes == 0
                or not record["original_mandatory_evidence_present"] and proof_bytes != 0):
            raise BinaryCorpusStoreError("status proof presence与offset sidecar不一致。")
    expected_fields = set(_CHUNK_META_FIELDS) | {"source_kind", "source_row"}
    chunk_ids = []
    for row, (kind_code, source_row, kind, id_field, record) in zip(chunk_meta, expected_chunk_map):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise BinaryCorpusStoreError("chunks.metadata.jsonl字段无效。")
        expected = {
            "chunk_id": record[id_field],
            "std_no": record["std_no"],
            "std_id": record.get("std_id"),
            "evidence_kind": "clause" if kind == "clauses" else "table",
            "document_part": record.get("document_part", "body"),
            "excluded_from_current_retrieval": record["excluded_from_current_retrieval"],
            "import_part_id": record.get("import_part_id"),
            "clause_no": record.get("clause_no"),
            "table_id": record.get("table_id"),
            "source_kind": kind_code,
            "source_row": source_row,
        }
        if row != expected:
            raise BinaryCorpusStoreError("chunk metadata与source record行映射不一致。")
        chunk_ids.append(row["chunk_id"])
    if (chunk_ids != manifest["row_ids"]
            or len(set(chunk_ids)) != len(chunk_ids)):
        raise BinaryCorpusStoreError("chunk ID顺序或唯一性与manifest不一致。")

    chunk_map = arrays["chunks.map.npy"]
    physical_chunk_count = max(1, chunk_count)
    expected_map_dtype = np.dtype([("kind", np.uint8), ("row", np.uint64)])
    if (not isinstance(chunk_map, np.memmap) or chunk_map.shape != (physical_chunk_count,)
            or chunk_map.dtype != expected_map_dtype or chunk_map.flags.writeable):
        raise BinaryCorpusStoreError("chunks.map.npy shape/dtype无效。")
    for index, (kind_code, source_row, _, _, _) in enumerate(expected_chunk_map):
        mapped = chunk_map[index]
        if int(mapped["kind"]) != kind_code or int(mapped["row"]) != source_row:
            raise BinaryCorpusStoreError("chunks.map.npy行与chunk metadata不一致。")
    if not chunk_count and (int(chunk_map[0]["kind"]) != 0 or int(chunk_map[0]["row"]) != 0):
        raise BinaryCorpusStoreError("空chunks.map.npy sentinel无效。")

    if (not isinstance(source_records, list)
            or len(source_records) != manifest["source_record_count"]):
        raise BinaryCorpusStoreError("source_records.jsonl行数与manifest不一致。")
    for record in source_records:
        if (not isinstance(record, dict) or not isinstance(record.get("std_no"), str)
                or not record["std_no"]
                or set(record) - ({"std_no"} | set(_SOURCE_LOCATION_FIELDS))):
            raise BinaryCorpusStoreError("source-only location投影字段无效。")
        for field in _SOURCE_LOCATION_FIELDS:
            values = record.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
                raise BinaryCorpusStoreError("source-only location条目无效。")

    corpus_meta_path_shape = manifest.get("corpus_meta_shape")
    if corpus_meta_path_shape != {"keys": ["summary", "catalog", "relations"]}:
        raise BinaryCorpusStoreError("corpus_meta.json manifest字段无效。")


def _write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _finish_blob(handle, logical_bytes: int) -> None:
    if logical_bytes == 0:
        handle.write(b"\0")
    handle.flush()
    os.fsync(handle.fileno())


def _file_entry(path: Path, *, logical_bytes: int | None = None,
                shape: list[int] | None = None,
                dtype: str | None = None,
                fields: dict | None = None) -> dict:
    entry = {"file_bytes": _file_size(path)}
    if logical_bytes is not None:
        entry["logical_bytes"] = logical_bytes
    if shape is not None:
        entry["shape"] = shape
    if dtype is not None:
        entry["dtype"] = dtype
    if fields is not None:
        entry["fields"] = fields
    return entry


def build_core_corpus_mmap(source_dir: str | Path, output_dir: str | Path) -> Path:
    """Stream source JSONL into an immutable corpus-record/chunk mmap package.

    No StandardCorpus, EvidenceIndex, or LocalIndex is constructed. At most one
    source record and its generated Chunk are held at a time; light projections
    and text uniqueness are accumulated for package metadata.
    """
    source = Path(source_dir)
    output = Path(output_dir)
    _check_source_files(source)
    corpus_meta = _load_small_source_metadata(source)
    try:
        if _is_link_or_junction(output):
            raise BinaryCorpusStoreError("corpus输出目录不得为符号链接或junction。")
        output.mkdir(parents=True, exist_ok=True)
        if _is_link_or_junction(output):
            raise BinaryCorpusStoreError("corpus输出目录不得为符号链接或junction。")
        versions = output / "versions"
        if _is_link_or_junction(versions):
            raise BinaryCorpusStoreError("corpus versions目录不得为符号链接或junction。")
        if versions.exists():
            versions.resolve(strict=True).relative_to(output.resolve(strict=True))
        versions.mkdir(exist_ok=True)
        resolved_output = output.resolve(strict=True)
        resolved_versions = versions.resolve(strict=True)
        resolved_versions.relative_to(resolved_output)
    except BinaryCorpusStoreError:
        raise
    except (OSError, ValueError) as exc:
        raise BinaryCorpusStoreError("无法在corpus输出子树创建versions目录。") from exc
    if _is_link_or_junction(output / "CURRENT"):
        raise BinaryCorpusStoreError("corpus CURRENT不得为符号链接或junction。")

    version_id = "v1-" + uuid4().hex
    temporary = versions / (".tmp-" + uuid4().hex)
    published = versions / version_id
    try:
        temporary.mkdir()
        paths = {name: temporary / name for name in _PACKAGE_FILES}
        record_counts = {"clauses": 0, "tables": 0}
        record_offsets = {"clauses": [0], "tables": [0]}
        record_blob_bytes = {"clauses": 0, "tables": 0}
        status_proof_offsets = [0]
        status_proof_bytes = 0
        chunk_text_offsets = [0]
        chunk_text_bytes = 0
        chunk_map_values: list[tuple[int, int]] = []
        chunk_metadata_rows: list[dict] = []
        row_ids: list[str] = []
        unique_texts: set[str] = set()
        seen_ids = {"clauses": set(), "tables": set()}
        seen_chunk_ids: set[str] = set()
        seen_source_paths: set[tuple[str, str]] = set()
        source_record_count = 0
        from .evidence_index import _record_chunk

        with (paths["clauses.records.bin"].open("xb") as clause_blob,
              paths["tables.records.bin"].open("xb") as table_blob,
              paths["clauses.status_proof.bin"].open("xb") as proof_blob,
              paths["chunks.text.bin"].open("xb") as text_blob,
              paths["clauses.metadata.jsonl"].open("x", encoding="utf-8", newline="\n") as clause_meta_file,
              paths["tables.metadata.jsonl"].open("x", encoding="utf-8", newline="\n") as table_meta_file,
              paths["chunks.metadata.jsonl"].open("x", encoding="utf-8", newline="\n") as chunk_meta_file,
              paths["source_records.jsonl"].open("x", encoding="utf-8", newline="\n") as source_meta_file):
            for kind, source_name, id_field, kind_code, blob, meta_file in (
                    ("clause", "clauses.jsonl", "clause_id", 0, clause_blob, clause_meta_file),
                    ("table", "tables.jsonl", "table_id", 1, table_blob, table_meta_file)):
                key = "clauses" if kind == "clause" else "tables"
                source_path = source / source_name
                with source_path.open("rb") as source_handle:
                    for line_number, raw_line in enumerate(source_handle, 1):
                        record = _parse_jsonl_line(raw_line, filename=source_name,
                                                   line_number=line_number)
                        if record is None:
                            continue
                        metadata = _project_record(record, kind)
                        identifier = metadata[id_field]
                        if identifier in seen_ids[key]:
                            raise BinaryCorpusStoreError(f"{source_name}含重复{ id_field }。")
                        seen_ids[key].add(identifier)

                        record_bytes = raw_line.rstrip(b"\r\n")
                        if not record_bytes:
                            raise BinaryCorpusStoreError(f"{source_name}:{line_number}为空记录。")
                        blob.write(record_bytes)
                        record_blob_bytes[key] += len(record_bytes)
                        record_offsets[key].append(record_blob_bytes[key])

                        proof_present = (kind == "clause"
                                         and "original_mandatory_evidence" in record)
                        metadata["original_mandatory_evidence_present"] = proof_present
                        proof_bytes = (json.dumps(record["original_mandatory_evidence"],
                                                  ensure_ascii=False, separators=(",", ":"))
                                       .encode("utf-8") if proof_present else b"")
                        if proof_bytes:
                            proof_blob.write(proof_bytes)
                            status_proof_bytes += len(proof_bytes)
                        if kind == "clause":
                            status_proof_offsets.append(status_proof_bytes)

                        chunk_included = record.get("document_part", "body") != "commentary"
                        chunk = None
                        if chunk_included:
                            chunk = _record_chunk(
                                record, kind, "standards:binary-corpus-build")
                            chunk_included = bool(chunk.text.strip())
                        metadata["chunk_included"] = chunk_included
                        json_line = json.dumps(metadata, ensure_ascii=False,
                                               separators=(",", ":"))
                        meta_file.write(json_line + "\n")

                        if chunk_included:
                            if chunk.chunk_id in seen_chunk_ids:
                                raise BinaryCorpusStoreError("clauses/tables Chunk ID重复。")
                            seen_chunk_ids.add(chunk.chunk_id)
                            text_bytes = chunk.text.encode("utf-8")
                            if not text_bytes:
                                raise BinaryCorpusStoreError("nonblank Chunk缺少UTF-8文本。")
                            text_blob.write(text_bytes)
                            chunk_text_bytes += len(text_bytes)
                            chunk_text_offsets.append(chunk_text_bytes)
                            chunk_map_values.append((kind_code, record_counts[key]))
                            row_ids.append(chunk.chunk_id)
                            unique_texts.add(chunk.text)
                            chunk_meta = {
                                "chunk_id": chunk.chunk_id,
                                "std_no": metadata["std_no"],
                                "std_id": metadata.get("std_id"),
                                "evidence_kind": kind,
                                "document_part": metadata["document_part"],
                                "excluded_from_current_retrieval": metadata[
                                    "excluded_from_current_retrieval"],
                                "import_part_id": metadata.get("import_part_id"),
                                "clause_no": metadata.get("clause_no"),
                                "table_id": metadata.get("table_id"),
                                "source_kind": kind_code,
                                "source_row": record_counts[key],
                            }
                            chunk_meta_file.write(json.dumps(
                                chunk_meta, ensure_ascii=False, separators=(",", ":")) + "\n")

                        source_projection = _source_projection(record, seen_source_paths)
                        if source_projection is not None:
                            source_meta_file.write(json.dumps(
                                source_projection, ensure_ascii=False, separators=(",", ":")) + "\n")
                            source_record_count += 1
                        record_counts[key] += 1

            _finish_blob(clause_blob, record_blob_bytes["clauses"])
            _finish_blob(table_blob, record_blob_bytes["tables"])
            _finish_blob(proof_blob, status_proof_bytes)
            _finish_blob(text_blob, chunk_text_bytes)
            for handle in (clause_meta_file, table_meta_file, chunk_meta_file, source_meta_file):
                handle.flush()
                os.fsync(handle.fileno())

        small_meta_path = paths["corpus_meta.json"]
        with small_meta_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(corpus_meta, handle, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        chunk_count = len(row_ids)
        chunk_map_dtype = np.dtype([("kind", np.uint8), ("row", np.uint64)])
        physical_map = max(1, chunk_count)
        chunk_map_array = np.zeros(physical_map, dtype=chunk_map_dtype)
        if chunk_count:
            chunk_map_array["kind"] = np.asarray([value[0] for value in chunk_map_values], dtype=np.uint8)
            chunk_map_array["row"] = np.asarray([value[1] for value in chunk_map_values], dtype=np.uint64)
        _write_npy(paths["clauses.record_offsets.npy"],
                   np.asarray(record_offsets["clauses"], dtype=np.uint64))
        _write_npy(paths["tables.record_offsets.npy"],
                   np.asarray(record_offsets["tables"], dtype=np.uint64))
        _write_npy(paths["clauses.status_proof_offsets.npy"],
                   np.asarray(status_proof_offsets, dtype=np.uint64))
        _write_npy(paths["chunks.text_offsets.npy"],
                   np.asarray(chunk_text_offsets, dtype=np.uint64))
        _write_npy(paths["chunks.map.npy"], chunk_map_array)

        file_entries = {}
        for name in _PACKAGE_FILES:
            path = paths[name]
            if name.endswith(".bin"):
                logical_bytes = {
                    "clauses.records.bin": record_blob_bytes["clauses"],
                    "tables.records.bin": record_blob_bytes["tables"],
                    "clauses.status_proof.bin": status_proof_bytes,
                    "chunks.text.bin": chunk_text_bytes,
                }[name]
                file_entries[name] = _file_entry(path, logical_bytes=logical_bytes)
            elif name == "chunks.map.npy":
                file_entries[name] = _file_entry(
                    path, shape=[physical_map],
                    fields={"kind": "uint8", "row": "uint64"})
            elif name.endswith(".npy"):
                shape = {
                    "clauses.record_offsets.npy": [len(record_offsets["clauses"])],
                    "tables.record_offsets.npy": [len(record_offsets["tables"])],
                    "clauses.status_proof_offsets.npy": [len(status_proof_offsets)],
                    "chunks.text_offsets.npy": [len(chunk_text_offsets)],
                }[name]
                file_entries[name] = _file_entry(
                    path, shape=shape, dtype=np.dtype(np.uint64).str)
            else:
                file_entries[name] = _file_entry(path)

        manifest = {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "chunk_builder_id": _CHUNK_BUILDER_ID,
            "chunk_format_version": _CHUNK_FORMAT_VERSION,
            "version_id": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record_counts": record_counts,
            "chunk_count": chunk_count,
            "unique_text_count": len(unique_texts),
            "source_record_count": source_record_count,
            "row_ids": row_ids,
            "corpus_meta_shape": {"keys": ["summary", "catalog", "relations"]},
            "files": file_entries,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        temporary.replace(published)
        _fsync_directory(versions)
        pointer_temp = output / (".CURRENT-" + uuid4().hex + ".tmp")
        try:
            with pointer_temp.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(version_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            pointer_temp.replace(output / "CURRENT")
            _fsync_directory(output)
        finally:
            pointer_temp.unlink(missing_ok=True)
        return published
    except BinaryCorpusStoreError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise BinaryCorpusStoreError("corpus mmap构建失败；CURRENT未切换。") from exc
    finally:
        if temporary.exists():
            try:
                if (not _is_link_or_junction(temporary)
                        and re.fullmatch(r"\.tmp-[0-9a-f]{32}", temporary.name)):
                    current_output = output.resolve(strict=True)
                    current_versions = versions.resolve(strict=True)
                    resolved_temp = temporary.resolve(strict=True)
                    if (current_output == resolved_output and current_versions == resolved_versions
                            and resolved_temp.parent == resolved_versions):
                        shutil.rmtree(temporary)
            except (OSError, ValueError):
                pass


def _parse_jsonl_line(raw_line: bytes, *, filename: str, line_number: int) -> dict:
    content = raw_line.rstrip(b"\r\n")
    if not content.strip():
        return None
    try:
        value = json.loads(content, object_pairs_hook=_json_no_duplicate_pairs)
    except (UnicodeError, json.JSONDecodeError, BinaryCorpusStoreError) as exc:
        raise BinaryCorpusStoreError(f"{filename}:{line_number}不是有效JSON。") from exc
    if not isinstance(value, dict):
        raise BinaryCorpusStoreError(f"{filename}:{line_number}必须是JSON对象。")
    return value


def _project_record(record: dict, kind: str) -> dict:
    identifier_field = "clause_id" if kind == "clause" else "table_id"
    identifier = record.get(identifier_field)
    if not isinstance(identifier, str) or not identifier:
        raise BinaryCorpusStoreError(f"{kind}记录缺少{identifier_field}。")
    if not isinstance(record.get("std_no"), str) or not record["std_no"]:
        raise BinaryCorpusStoreError(f"{kind}记录缺少std_no。")
    result = {field: deepcopy(record.get(field)) for field in _META_FIELDS}
    result[identifier_field] = identifier
    result["document_part"] = record.get("document_part", "body")
    result["original_mandatory_evidence_present"] = "original_mandatory_evidence" in record
    result["excluded_from_current_retrieval"] = _excluded_from_current(record)
    return result


def _excluded_from_current(record: dict) -> bool:
    return bool(
        record.get("superseded_by") or record.get("deleted_by")
        or record.get("source_reproduction_superseded_by")
        or record.get("source_view_superseded_by")
        or record.get("edition_role") in {
            "verified_2015_historical_baseline", "historical_2015_ocr_baseline"})


def _location_values(record: dict, field: str) -> list[dict]:
    values = record.get(field, []) or []
    if isinstance(values, dict):
        values = [values]
    locations = []
    for value in values:
        if isinstance(value, str):
            locations.append({"source_path": value})
        elif isinstance(value, dict):
            locations.append(deepcopy(value))
    return locations


def _source_projection(record: dict, seen: set[tuple[str, str]]) -> dict | None:
    std_no = record.get("std_no")
    if not isinstance(std_no, str) or not std_no:
        return None
    projection = {"std_no": std_no}
    has_locations = False
    for field in _SOURCE_LOCATION_FIELDS:
        retained = []
        for location in _location_values(record, field):
            for path_key in _SOURCE_PATH_KEYS:
                path_value = location.get(path_key)
                if (not isinstance(path_value, str) or not path_value
                        or "://" in path_value or (std_no, path_value) in seen):
                    continue
                seen.add((std_no, path_value))
                # Keep one original path identity per projection row while
                # retaining source_id/page/logical-page and every other field.
                projected_location = deepcopy(location)
                for other_key in _SOURCE_PATH_KEYS:
                    if other_key != path_key:
                        projected_location.pop(other_key, None)
                retained.append(projected_location)
        if retained:
            projection[field] = retained
            has_locations = True
    return projection if has_locations else None


def _open_json_lines(path: Path) -> list[dict]:
    values = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, object_pairs_hook=_canonical_metadata_pairs)
                except (json.JSONDecodeError, BinaryCorpusStoreError) as exc:
                    raise BinaryCorpusStoreError(f"{path.name}包含无效JSON行。") from exc
                if not isinstance(value, dict):
                    raise BinaryCorpusStoreError(f"{path.name}行必须是对象。")
                for field in _LOW_CARD_METADATA_FIELDS:
                    field_value = value.get(field)
                    if isinstance(field_value, str) and len(field_value) <= 256:
                        value[field] = sys.intern(field_value)
                values.append(value)
    except (OSError, UnicodeError) as exc:
        raise BinaryCorpusStoreError(f"无法读取corpus元数据文件：{path.name}") from exc
    return values


def _canonical_metadata_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BinaryCorpusStoreError("JSON对象包含重复字段。")
        result[_CANONICAL_METADATA_KEYS.get(key, key)] = value
    return result


def _read_offsets(path: Path, expected_count: int, expected_last: int) -> np.memmap:
    try:
        offsets = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        raise BinaryCorpusStoreError(f"{path.name}不是有效的安全NPY offsets。") from exc
    if (not isinstance(offsets, np.memmap) or offsets.dtype != np.dtype(np.uint64)
            or offsets.shape != (expected_count + 1,) or offsets.flags.writeable
            or not offsets.flags.c_contiguous):
        mmap = getattr(offsets, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()
        raise BinaryCorpusStoreError(f"{path.name}的offset shape/dtype无效。")
    if (int(offsets[0]) != 0 or int(offsets[-1]) != expected_last
            or np.any(offsets[1:] < offsets[:-1])):
        offsets._mmap.close()
        raise BinaryCorpusStoreError(f"{path.name}的offset边界或单调性无效。")
    return offsets


class _ReadLease:
    def __init__(self, store: "BinaryCorpusStore"):
        self.store = store

    def __enter__(self):
        with self.store._condition:
            if self.store._state != "open":
                raise BinaryCorpusStoreError("BinaryCorpusStore正在关闭或已关闭。")
            self.store._active_readers += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        with self.store._condition:
            self.store._active_readers -= 1
            if self.store._active_readers == 0:
                self.store._condition.notify_all()


class _ProjectionSequence(Sequence):
    """Small preloaded projection; callers receive independent dictionaries."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.filter_metadata = tuple(
            MappingProxyType(row) if all(_is_scalar(value) for value in row.values())
            else _freeze_value(row)
            for row in rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [deepcopy(row) for row in self._rows[index]]
        return deepcopy(self._rows[index])


def _freeze_value(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _is_scalar(value) -> bool:
    return value is None or type(value) in (str, int, float, bool)


class _MappedRecordSequence(Sequence):
    def __init__(self, store: "BinaryCorpusStore", kind: str,
                 metadata: list[dict], offsets: np.memmap):
        self._store = store
        self._kind = kind
        self._metadata_rows = metadata
        self._offsets = offsets
        self.metadata = _ProjectionSequence(metadata)

    def __len__(self):
        return len(self._metadata_rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("record index须为整数或slice。")
        row = index + len(self) if index < 0 else index
        if row < 0 or row >= len(self):
            raise IndexError(index)
        with self._store._read_lease():
            return self._store._decode_record_unleased(self._kind, row)

    def status_records(self):
        if self._kind != "clause":
            return iter(())
        return self._store._status_record_iterator()


class _LazyRecordMapping(Mapping):
    """O(1) ID-to-row mapping that decodes a new JSON object on each lookup."""

    def __init__(self, sequence: _MappedRecordSequence, key_field: str):
        self._sequence = sequence
        self._key_field = key_field
        self._rows = {meta[key_field]: row for row, meta in enumerate(sequence._metadata_rows)}

    def __getitem__(self, key):
        try:
            row = self._rows[key]
        except KeyError:
            raise KeyError(key) from None
        return self._sequence[row]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class _StatusRecord(Mapping):
    def __init__(self, store: "BinaryCorpusStore", row: int, metadata: dict):
        self._store = store
        self._row = row
        self._metadata = metadata
        self._fields = tuple(key for key in _STATUS_FIELDS if key in metadata)
        self._has_proof = bool(metadata.get("original_mandatory_evidence_present"))

    def __getitem__(self, key):
        if key == "original_mandatory_evidence":
            if not self._has_proof:
                raise KeyError(key)
            with self._store._read_lease():
                return self._store._decode_status_proof_unleased(self._row)
        if key in self._fields:
            return deepcopy(self._metadata[key])
        raise KeyError(key)

    def __iter__(self):
        yield from self._fields
        if self._has_proof:
            yield "original_mandatory_evidence"

    def __len__(self):
        return len(self._fields) + int(self._has_proof)


class _IdTextRows(Sequence):
    def __init__(self, store: "BinaryCorpusStore", row_ids: list[str],
                 offsets: np.memmap):
        self._store = store
        self._row_ids = row_ids
        self._offsets = offsets

    def __len__(self):
        return len(self._row_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("id_text_rows index须为整数或slice。")
        row = index + len(self) if index < 0 else index
        if row < 0 or row >= len(self):
            raise IndexError(index)
        with self._store._read_lease():
            text = self._store._decode_chunk_text_unleased(row)
            self._store._cache_stats["id_text_decode_count"] += 1
            return self._row_ids[row], text


class _LazyChunkSequence(Sequence):
    is_lazy_chunk_sequence = True

    def __init__(self, store: "BinaryCorpusStore", collection_id: str,
                 rows: list[dict], row_ids: list[str], mapping: np.memmap,
                 text_offsets: np.memmap):
        self._store = store
        self.collection_id = collection_id
        self._metadata_rows = rows
        self._row_ids = row_ids
        self._mapping = mapping
        self.metadata = _ProjectionSequence(rows)
        self.filter_metadata = self.metadata.filter_metadata
        self.id_text_rows = _IdTextRows(store, row_ids, text_offsets)
        self.unique_text_count = store._manifest["unique_text_count"]

    @property
    def row_ids(self) -> list[str]:
        return list(self._row_ids)

    def __len__(self):
        return len(self._row_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("chunk index须为整数或slice。")
        row = index + len(self) if index < 0 else index
        if row < 0 or row >= len(self):
            raise IndexError(index)
        with self._store._read_lease():
            kind_code = int(self._mapping[row]["kind"])
            source_row = int(self._mapping[row]["row"])
            kind = "clause" if kind_code == 0 else "table"
            record = self._store._decode_record_unleased(kind, source_row)
            from .evidence_index import _record_chunk

            chunk = _record_chunk(record, kind, self.collection_id)
            expected = self._metadata_rows[row]
            if (chunk.chunk_id != expected["chunk_id"]
                    or chunk.text != self._store._decode_chunk_text_unleased(row)):
                raise BinaryCorpusStoreError("生成的Chunk与当前预建ID/text投影不一致。")
            self._store._cache_stats["chunk_decode_count"] += 1
            return chunk


class _StatusProofView:
    def __init__(self, store: "BinaryCorpusStore"):
        self._store = store


class MappedStandardCorpus(StandardCorpus):
    """Lazy StandardCorpus-compatible facade; deliberately skips eager base init."""

    def __init__(self, store: "BinaryCorpusStore", *, clauses: _MappedRecordSequence,
                 tables: _MappedRecordSequence, metadata: list[dict], corpus_meta: dict):
        self._store = store
        self.clauses = clauses
        self.tables = tables
        self.summary = corpus_meta["summary"]
        self.catalog = corpus_meta["catalog"]
        self.relations = corpus_meta["relations"]
        self.metadata = _ProjectionSequence(metadata)
        self._clauses = _LazyRecordMapping(clauses, "clause_id")
        self._tables = _LazyRecordMapping(tables, "table_id")
        self._aliases: dict[str, set[str]] = {}
        self._clause_lookup: dict[tuple[str, str], list[int]] = {}
        for sequence, kind in ((clauses, "clause"), (tables, "table")):
            id_field = "clause_id" if kind == "clause" else "table_id"
            for row, record in enumerate(sequence._metadata_rows):
                identifier = record[id_field]
                std_no = record["std_no"]
                std_id = record.get("std_id") or identifier.split("/", 1)[0]
                full = _standard_key(std_no)
                for alias in {full, _standard_key(std_id), re.sub(r"-\d{4}$", "", full)}:
                    self._aliases.setdefault(alias, set()).add(std_id)
                if kind == "clause":
                    clause_no = record.get("clause_no")
                    if isinstance(clause_no, str):
                        self._clause_lookup.setdefault((std_id, clause_no.strip()), []).append(row)

    def get_clause(self, std_no: str, clause_no: str) -> dict | None:
        std_id = self.resolve_standard_id(std_no)
        indexes = self._clause_lookup.get((std_id, clause_no.strip()), [])
        matches = [row for row in indexes
                   if self.clauses._metadata_rows[row].get("exact_clause_lookup") is not False]
        if len(matches) == 1:
            return self.clauses[matches[0]]
        qualified = self._clauses.get(f"{std_id}/{clause_no.strip()}") if std_id else None
        if (qualified and qualified.get("exact_clause_lookup") is False
                and "/" not in clause_no):
            return None
        return qualified

    def get_table(self, table_id: str) -> dict | None:
        return self._tables.get(table_id)

    def status_records(self):
        return self.clauses.status_records()

    def iter_source_records(self, selected=None):
        return self._store.iter_source_records(selected=selected)

    def chunk_sequence(self, collection_id: str) -> Sequence:
        return self._store.chunk_sequence(collection_id)

    def search(self, query: str, limit: int = 5, std_no: str | None = None,
               include_tables: bool = True) -> list[dict]:
        """Stream records and retain only top hits, never rebuild full record lists."""
        if limit < 1 or not query.strip():
            return []
        from .standards import _tokens

        query_tokens = _tokens(query)
        heap = []
        serial = 0
        for kind, sequence in (("clause", self.clauses), ("table", self.tables)):
            if kind == "table" and not include_tables:
                continue
            identifier_field = "clause_id" if kind == "clause" else "table_id"
            number_field = "clause_no" if kind == "clause" else "table_no"
            for row_index, projection in enumerate(sequence._metadata_rows):
                if (std_no and _standard_key(projection["std_no"]) != _standard_key(std_no)):
                    continue
                if (kind == "table"
                        and projection.get("content_status") != "text_available"):
                    continue
                record = sequence[row_index]
                body = record["text"] if kind == "clause" else record["flat_text"]
                words = _tokens(body)
                overlap = query_tokens & words
                if not overlap:
                    continue
                score = len(overlap) / math.sqrt(max(1, len(words)))
                number = record[number_field]
                if number in query_tokens:
                    score += 5.0
                if kind == "table":
                    score += len(query_tokens & _tokens(record["caption"])) / max(1, len(query_tokens))
                evidence_id = record[identifier_field]
                ranked_score = round(score, 6)
                hit = {
                    "evidence_id": evidence_id, "kind": kind,
                    "score": ranked_score, "record": record,
                }
                candidate = _SearchCandidate(ranked_score, evidence_id, serial, hit)
                serial += 1
                if len(heap) < limit:
                    heapq.heappush(heap, candidate)
                elif heap[0] < candidate:
                    heapq.heapreplace(heap, candidate)
        return sorted((candidate.hit for candidate in heap),
                      key=lambda hit: (-hit["score"], hit["evidence_id"]))


class _SearchCandidate:
    __slots__ = ("score", "evidence_id", "serial", "hit")

    def __init__(self, score: float, evidence_id: str, serial: int, hit: dict):
        self.score = score
        self.evidence_id = evidence_id
        self.serial = serial
        self.hit = hit

    def __lt__(self, other):
        if self.score != other.score:
            return self.score < other.score
        if self.evidence_id != other.evidence_id:
            return self.evidence_id > other.evidence_id
        return self.serial > other.serial


class BinaryCorpusStore:
    """Owner for immutable corpus blobs and their read-only mmap handles."""

    @property
    def cache_stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                **self._cache_stats,
                "cache_bytes": self._cache_current_bytes,
                "cache_limit_bytes": self._cache_limit,
                "active_readers": self._active_readers,
                "state": self._state,
            }

    @contextmanager
    def _read_lease(self):
        lease = _ReadLease(self)
        lease.__enter__()
        try:
            yield
        finally:
            lease.__exit__(None, None, None)

    def _blob_slice_unleased(self, blob: np.memmap, offsets: np.memmap, row: int) -> bytes:
        start, end = int(offsets[row]), int(offsets[row + 1])
        if start < 0 or end < start or end > blob.shape[0]:
            raise BinaryCorpusStoreError("record/text blob offset越界。")
        return blob[start:end].tobytes()

    def _record_bytes_unleased(self, kind: str, row: int) -> bytes:
        key = (kind, row)
        with self._condition:
            cached = self._record_cache.get(key)
            if cached is not None:
                self._record_cache.move_to_end(key)
                self._cache_stats["cache_hits"] += 1
                return cached
            self._cache_stats["cache_misses"] += 1
        blob = self._record_blob(kind)
        offsets = self._record_offsets(kind)
        value = self._blob_slice_unleased(blob, offsets, row)
        if self._cache_limit and len(value) <= self._cache_limit:
            with self._condition:
                # Another reader may have missed the same key while this
                # thread was reading the mmap outside the lock. Keep the
                # existing entry instead of double-counting its bytes.
                cached = self._record_cache.get(key)
                if cached is not None:
                    self._record_cache.move_to_end(key)
                    return cached
                self._record_cache[key] = value
                self._cache_current_bytes += len(value)
                while self._cache_current_bytes > self._cache_limit and self._record_cache:
                    _, removed = self._record_cache.popitem(last=False)
                    self._cache_current_bytes -= len(removed)
                    self._cache_stats["cache_evictions"] += 1
        return value

    def _record_blob(self, kind: str) -> np.memmap:
        return self._clauses_blob if kind == "clause" else self._tables_blob

    def _record_offsets(self, kind: str) -> np.memmap:
        return self._clauses_offsets if kind == "clause" else self._tables_offsets

    def _decode_record_unleased(self, kind: str, row: int) -> dict:
        raw = self._record_bytes_unleased(kind, row)
        try:
            record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BinaryCorpusStoreError("source record blob含无效JSON。") from exc
        metadata = self._clauses._metadata_rows if kind == "clause" else self._tables._metadata_rows
        id_field = "clause_id" if kind == "clause" else "table_id"
        if not isinstance(record, dict) or record.get(id_field) != metadata[row][id_field]:
            raise BinaryCorpusStoreError("source record ID与metadata行不一致。")
        with self._condition:
            self._cache_stats["record_decode_count"] += 1
        return record

    def _decode_chunk_text_unleased(self, row: int) -> str:
        raw = self._blob_slice_unleased(self._chunk_text_blob, self._chunk_text_offsets, row)
        try:
            return raw.decode("utf-8")
        except UnicodeError as exc:
            raise BinaryCorpusStoreError("chunk text blob不是有效UTF-8。") from exc

    def _decode_status_proof_unleased(self, row: int):
        raw = self._blob_slice_unleased(self._status_proof_blob, self._status_proof_offsets, row)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BinaryCorpusStoreError("status proof blob包含无效JSON。") from exc

    def _status_record_iterator(self):
        for row, metadata in enumerate(self._clauses._metadata_rows):
            with self._read_lease():
                record = _StatusRecord(self, row, metadata)
            yield record

    def chunk_sequence(self, collection_id: str) -> Sequence:
        if not isinstance(collection_id, str) or not collection_id:
            raise ValueError("collection_id须为非空文本。")
        return _LazyChunkSequence(
            self, collection_id, self._chunk_metadata, self._row_ids,
            self._chunk_map, self._chunk_text_offsets)

    def iter_source_records(self, selected=None):
        selected_set = set(selected or [])
        for record in self._source_records:
            std_no = record.get("std_no")
            if selected_set and std_no not in selected_set:
                continue
            with self._read_lease():
                self._cache_stats["source_projection_reads"] += 1
                projected = deepcopy(record)
            yield projected

    def close(self) -> None:
        with self._condition:
            if self._state == "closed":
                return
            self._state = "closing"
            while self._active_readers:
                self._condition.wait()
            for array in self._maps:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None and not mmap.closed:
                    try:
                        mmap.close()
                    except BufferError as exc:
                        raise BinaryCorpusStoreError(
                            "corpus mmap仍有外部视图；释放后可重试close。") from exc
            self._record_cache.clear()
            self._cache_current_bytes = 0
            self._state = "closed"
            self._condition.notify_all()

    def __enter__(self):
        with self._condition:
            if self._state != "open":
                raise BinaryCorpusStoreError("BinaryCorpusStore正在关闭或已关闭。")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    @classmethod
    def open(cls, directory: str | Path, *, source_dir: str | Path | None = None,
             cache_bytes: int = 8 * 1024 * 1024) -> "BinaryCorpusStore":
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError("cache_bytes须为非负整数。")
        root = Path(directory)
        if _is_link_or_junction(root):
            raise BinaryCorpusStoreError("corpus mmap目录不得为符号链接或junction。")
        version_id, version_dir = _resolve_current(root)
        manifest = _read_json_object(version_dir / "manifest.json")
        _validate_manifest(manifest, version_id)
        files = manifest["files"]
        _validate_package_files(version_dir, files)
        maps: list[np.memmap] = []
        try:
            loaded_arrays = {}
            for name, info in files.items():
                if not name.endswith(".npy"):
                    continue
                path = version_dir / name
                try:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError, TypeError, EOFError) as exc:
                    raise BinaryCorpusStoreError(f"{name}不是安全的NPY数组。") from exc
                maps.append(array)
                expected_dtype = (np.dtype([("kind", np.uint8), ("row", np.uint64)])
                                  if name == "chunks.map.npy" else np.dtype(info["dtype"]))
                if (not isinstance(array, np.memmap) or array.flags.writeable
                        or not array.flags.c_contiguous or list(array.shape) != info["shape"]
                        or array.dtype != expected_dtype):
                    raise BinaryCorpusStoreError(f"{name}的header与manifest不匹配。")
                if array.offset + array.nbytes != info["file_bytes"]:
                    raise BinaryCorpusStoreError(f"{name}数据区长度与文件大小不一致。")
                loaded_arrays[name] = array
            record_meta = {
                "clauses": _open_json_lines(version_dir / "clauses.metadata.jsonl"),
                "tables": _open_json_lines(version_dir / "tables.metadata.jsonl"),
            }
            chunk_meta = _open_json_lines(version_dir / "chunks.metadata.jsonl")
            source_records = _open_json_lines(version_dir / "source_records.jsonl")
            corpus_meta = _read_json_object(version_dir / "corpus_meta.json")
            if (set(corpus_meta) != {"summary", "catalog", "relations"}
                    or not isinstance(corpus_meta.get("summary"), dict)
                    or not isinstance(corpus_meta.get("catalog"), list)
                    or not isinstance(corpus_meta.get("relations"), list)):
                raise BinaryCorpusStoreError("corpus_meta.json形状无效。")
            _validate_summary_documents(corpus_meta["summary"])
            _validate_projection_shapes(manifest, record_meta, chunk_meta, source_records,
                                        loaded_arrays)

            blobs = {}
            for name in ("clauses.records.bin", "tables.records.bin", "clauses.status_proof.bin",
                         "chunks.text.bin"):
                logical_bytes = files[name]["logical_bytes"]
                physical_bytes = max(1, logical_bytes)
                try:
                    blob = np.memmap(version_dir / name, mode="r", dtype=np.uint8,
                                     shape=(physical_bytes,))
                except (OSError, ValueError) as exc:
                    raise BinaryCorpusStoreError(f"{name}无法只读映射。") from exc
                maps.append(blob)
                if blob.flags.writeable or blob.nbytes != physical_bytes:
                    raise BinaryCorpusStoreError(f"{name}映射长度无效。")
                if logical_bytes == 0 and int(blob[0]) != 0:
                    raise BinaryCorpusStoreError(f"{name}空blob sentinel无效。")
                blobs[name] = blob

            # The optional source path is retained for API compatibility; loading
            # does not compare the current source against a packaged snapshot.
            validation_count = 0

            clauses = _MappedRecordSequence(
                None, "clause", record_meta["clauses"], loaded_arrays["clauses.record_offsets.npy"])
            tables = _MappedRecordSequence(
                None, "table", record_meta["tables"], loaded_arrays["tables.record_offsets.npy"])
            store = cls(
                version_dir=version_dir, manifest=manifest, maps=maps,
                clauses=clauses, tables=tables, corpus_meta=corpus_meta,
                chunk_map=loaded_arrays["chunks.map.npy"],
                chunk_text_offsets=loaded_arrays["chunks.text_offsets.npy"],
                chunk_text_blob=blobs["chunks.text.bin"],
                row_ids=[row["chunk_id"] for row in chunk_meta],
                chunk_metadata=chunk_meta, source_records=source_records,
                status_proof_blob=blobs["clauses.status_proof.bin"],
                status_proof_offsets=loaded_arrays["clauses.status_proof_offsets.npy"],
                clauses_blob=blobs["clauses.records.bin"],
                tables_blob=blobs["tables.records.bin"],
                clauses_offsets=loaded_arrays["clauses.record_offsets.npy"],
                tables_offsets=loaded_arrays["tables.record_offsets.npy"],
                cache_bytes=cache_bytes, source_validation_records=validation_count)
            clauses._store = store
            tables._store = store
            return store
        except BaseException:
            for array in maps:
                _close_mmap(array)
            raise

    def __init__(self, *, version_dir, manifest, maps, clauses, tables,
                 corpus_meta, chunk_map, chunk_text_offsets, chunk_text_blob,
                 row_ids, chunk_metadata, source_records, status_proof_blob,
                 status_proof_offsets, clauses_blob, tables_blob,
                 clauses_offsets, tables_offsets, cache_bytes,
                 source_validation_records):
        self.version_dir = version_dir
        self._manifest = manifest
        self._maps = maps
        self._clauses = clauses
        self._tables = tables
        self._corpus_meta = corpus_meta
        self._chunk_map = chunk_map
        self._chunk_text_offsets = chunk_text_offsets
        self._chunk_text_blob = chunk_text_blob
        self._row_ids = row_ids
        self._chunk_metadata = chunk_metadata
        self._source_records = source_records
        self._status_proof_blob = status_proof_blob
        self._status_proof_offsets = status_proof_offsets
        self._clauses_blob = clauses_blob
        self._tables_blob = tables_blob
        self._clauses_offsets = clauses_offsets
        self._tables_offsets = tables_offsets
        self._condition = threading.Condition(threading.RLock())
        self._active_readers = 0
        self._state = "open"
        self._cache_limit = cache_bytes
        self._record_cache = OrderedDict()
        self._cache_current_bytes = 0
        self._cache_stats = {
            "record_decode_count": 0, "chunk_decode_count": 0,
            "id_text_decode_count": 0, "source_projection_reads": 0,
            "source_validation_records": source_validation_records,
            "cache_hits": 0, "cache_misses": 0, "cache_evictions": 0,
        }
        self.corpus = MappedStandardCorpus(
            self, clauses=clauses, tables=tables,
            metadata=deepcopy(corpus_meta["summary"].get("documents", [])),
            corpus_meta=corpus_meta)
