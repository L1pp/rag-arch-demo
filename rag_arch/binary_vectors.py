"""Immutable, read-only binary storage for the public standards vectors.

The source embedding cache remains an ordinary JSON cache. Conversion streams
that cache and publishes a version directory only after validating the complete
JSON document. Runtime loading never reads the source cache or contacts a model
provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import ijson
import numpy as np

from .embedding_cache import _identity


_FORMAT = "rag_arch.core_vectors"
_FORMAT_VERSION = 1
_SOURCE_SCHEMA_VERSION = 1
_CURRENT_RE = re.compile(r"^v1-[0-9a-f]{32}$")
_SOURCE_KEYS = {"schema_version", "identity", "vector_dimensions", "documents", "queries"}
_IDENTITY_KEYS = {"base_url", "model", "dimensions"}
_GROUP_KEYS = {"documents", "queries"}


class BinaryVectorStoreError(ValueError):
    """The published vector store or conversion source cannot be trusted."""


class _Container:
    __slots__ = ("kind", "path", "pending_key", "next_index", "seen_keys")

    def __init__(self, kind: str, path: tuple[Any, ...]):
        self.kind = kind
        self.path = path
        self.pending_key: str | None = None
        self.next_index = 0
        self.seen_keys: set[str] = set()


def _effective_chunks(chunks: Sequence, *, materialize_lazy: bool = False):
    try:
        from .retrieval import _validated_lazy_chunk_sequence
        lazy_rows = _validated_lazy_chunk_sequence(chunks, BinaryVectorStoreError)
    except BinaryVectorStoreError:
        raise
    except (TypeError, ValueError) as exc:
        raise BinaryVectorStoreError("向量chunks序列无效。") from exc
    if lazy_rows is not None:
        # Only the explicit offline converter requests materialization below.
        # Runtime open keeps this view and decodes each metadata row as it is
        # validated, avoiding a corpus-sized list of text payloads.
        if not len(lazy_rows):
            raise BinaryVectorStoreError("当前规范语料没有可用的稠密检索chunk。")
        return list(lazy_rows) if materialize_lazy else lazy_rows
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks须为 EvidenceIndex/LocalIndex chunk 序列。")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None)
        text = getattr(chunk, "text", None)
        if not isinstance(chunk_id, str) or not chunk_id:
            raise BinaryVectorStoreError("chunk_id须为非空文本。")
        if not isinstance(text, str):
            raise BinaryVectorStoreError("chunk embedding text须为文本。")
        # LocalIndex removes whitespace-only chunks before constructing its
        # dense matrix, so the binary row set follows that same effective order.
        if not text.strip():
            continue
        if chunk_id in seen:
            raise BinaryVectorStoreError("当前chunk_id不唯一，拒绝绑定向量行。")
        seen.add(chunk_id)
        rows.append((chunk_id, text))
    if not rows:
        raise BinaryVectorStoreError("当前规范语料没有可用的稠密检索chunk。")
    return rows


def _row_ids(rows) -> list[str]:
    if getattr(rows, "is_lazy_chunk_rows", False) is True:
        return rows.row_ids
    return [chunk_id for chunk_id, _ in rows]


def _unique_embedding_text_count(rows) -> int:
    if getattr(rows, "is_lazy_chunk_rows", False) is True:
        return rows.unique_text_count
    return len({text for _, text in rows})


def _json_no_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BinaryVectorStoreError("JSON对象包含重复字段。")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BinaryVectorStoreError("向量manifest不是有效JSON。") from exc
    if not isinstance(value, dict):
        raise BinaryVectorStoreError("向量manifest格式无效。")
    return value


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _is_symlink_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and bool(is_junction()))


def _fsync_directory(path: Path) -> None:
    # Directory fsync is available on POSIX; Windows does not expose a portable
    # equivalent through Python. Data files and CURRENT itself are fsynced on
    # every platform before their atomic same-volume rename.
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


def _check_source_file(path: Path) -> None:
    try:
        path.stat()
        regular_file = path.is_file()
        is_symlink = path.is_symlink()
    except OSError as exc:
        raise BinaryVectorStoreError("源embedding缓存缺失或无法读取stat信息。") from exc
    if not regular_file or is_symlink:
        raise BinaryVectorStoreError("源embedding缓存必须是普通文件，不可为符号链接。")


def _current_version(output_root: Path) -> str:
    pointer = output_root / "CURRENT"
    if _is_symlink_or_junction(pointer):
        raise BinaryVectorStoreError("CURRENT不得为符号链接。")
    try:
        contents = pointer.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise BinaryVectorStoreError("向量CURRENT指针缺失或不可读。") from exc
    version = contents[:-1] if contents.endswith("\n") else contents
    if contents not in (version, version + "\n"):
        raise BinaryVectorStoreError("向量CURRENT指针格式无效。")
    if not _CURRENT_RE.fullmatch(version):
        raise BinaryVectorStoreError("向量CURRENT指针格式无效。")
    return version


def read_core_embedding_identity(source_json: str | Path) -> dict:
    """Read only the source cache identity for the offline CLI default.

    Conversion performs its own strict full-file validation, including
    duplicate-key detection, before publishing any version.
    """
    source = Path(source_json)
    try:
        with source.open("rb") as handle:
            values = ijson.items(handle, "identity", buf_size=64 * 1024, use_float=True)
            identity = next(values)
        return _identity(identity)
    except StopIteration as exc:
        raise BinaryVectorStoreError("源embedding缓存缺少identity。") from exc
    except BinaryVectorStoreError:
        raise
    except Exception as exc:
        raise BinaryVectorStoreError("无法从源embedding缓存读取identity。") from exc


def _scan_source(source: Path, expected_identity: dict,
                 rows_by_text: Mapping[str, list[int]],
                 *, output_matrix: np.memmap | None = None,
                 output_dtype: np.dtype | None = None,
                 expected_dimensions: int | None = None) -> dict:
    """Validate the entire source cache while retaining at most one vector.

    As soon as identity, cache metadata, or a prior vector supplies a width,
    oversized vectors are rejected before the extra value is appended. If all
    three widths are absent before the first vector, that single vector must
    be buffered to infer its width; subsequent vectors are bounded by it.
    """
    root_keys: set[str] = set()
    identity_values: dict[str, Any] = {}
    schema_version = None
    declared_dimensions = None
    actual_dimensions = None
    document_records = 0
    query_records = 0
    matched_texts: set[str] = set()
    active_vector: dict[str, Any] | None = None
    stack: list[_Container] = []
    root_started = False
    root_finished = False
    matrix = output_matrix
    dtype = output_dtype

    def take_path() -> tuple[Any, ...]:
        if not stack:
            if root_started:
                raise BinaryVectorStoreError("JSON包含多个顶层值。")
            return ()
        parent = stack[-1]
        if parent.kind == "map":
            if parent.pending_key is None:
                raise BinaryVectorStoreError("JSON对象值缺少字段名。")
            part = parent.pending_key
            parent.pending_key = None
        else:
            part = parent.next_index
            parent.next_index += 1
        return parent.path + (part,)

    def start_vector(path: tuple[Any, ...]) -> None:
        nonlocal active_vector, document_records, query_records
        if (len(path) != 2 or path[0] not in _GROUP_KEYS
                or not isinstance(path[1], str) or active_vector is not None):
            raise BinaryVectorStoreError("源缓存的documents/queries结构无效。")
        group, text = path
        if group == "documents":
            document_records += 1
        else:
            query_records += 1
        active_vector = {"group": group, "text": text, "values": []}

    def finish_vector() -> None:
        nonlocal active_vector, actual_dimensions, matrix
        if active_vector is None:
            raise BinaryVectorStoreError("源缓存包含未配对的向量数组。")
        values = active_vector["values"]
        if not values:
            raise BinaryVectorStoreError("源缓存包含空向量。")
        dimension = len(values)
        if expected_dimensions is not None and dimension != expected_dimensions:
            raise BinaryVectorStoreError("源向量维度与当前embedding identity不符。")
        if declared_dimensions is not None and dimension != declared_dimensions:
            raise BinaryVectorStoreError("源向量维度与vector_dimensions不符。")
        if actual_dimensions is None:
            actual_dimensions = dimension
        elif dimension != actual_dimensions:
            raise BinaryVectorStoreError("源缓存中的向量维度不一致。")
        row = np.asarray([values], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(row, axis=1)
        if (not np.isfinite(row).all() or not np.isfinite(norm).all()
                or np.any(norm == 0)):
            raise BinaryVectorStoreError("源缓存包含非有限值、零向量或无法归一化的向量。")
        group, text = active_vector["group"], active_vector["text"]
        if group == "documents" and text in rows_by_text:
            if text in matched_texts:
                raise BinaryVectorStoreError("源documents文本键重复。")
            if matrix is not None:
                normalized = row / norm[:, None]
                stored = normalized.astype(dtype, copy=False)
                for row_number in rows_by_text[text]:
                    matrix[row_number, :] = stored[0]
            matched_texts.add(text)
        active_vector = None

    try:
        with source.open("rb") as handle:
            events = ijson.basic_parse(handle, buf_size=64 * 1024, use_float=True)
            for event, value in events:
                if root_finished:
                    raise BinaryVectorStoreError("JSON顶层对象结束后仍有额外内容。")
                if event == "map_key":
                    if not stack or stack[-1].kind != "map":
                        raise BinaryVectorStoreError("JSON字段位置无效。")
                    container = stack[-1]
                    if value in container.seen_keys:
                        raise BinaryVectorStoreError("源JSON包含重复字段。")
                    container.seen_keys.add(value)
                    if container.path == ():
                        if value not in _SOURCE_KEYS:
                            raise BinaryVectorStoreError("源缓存包含未知顶层字段。")
                        root_keys.add(value)
                    elif container.path == ("identity",):
                        if value not in _IDENTITY_KEYS:
                            raise BinaryVectorStoreError("源identity包含未知字段。")
                    elif container.path not in (("documents",), ("queries",)):
                        raise BinaryVectorStoreError("源缓存包含意外对象层级。")
                    container.pending_key = value
                    continue

                if event in ("start_map", "start_array", "string", "number", "boolean", "null"):
                    path = take_path()
                    if path == ():
                        if event != "start_map" or root_started:
                            raise BinaryVectorStoreError("源缓存顶层必须是单个JSON对象。")
                        root_started = True
                        stack.append(_Container("map", ()))
                        continue

                    if event == "start_map":
                        if path == ("identity",) and stack[-1].path == ():
                            stack.append(_Container("map", path))
                            continue
                        if path in (("documents",), ("queries",)) and stack[-1].path == ():
                            stack.append(_Container("map", path))
                            continue
                        raise BinaryVectorStoreError("源缓存对象层级无效。")

                    if event == "start_array":
                        if len(path) == 2 and path[0] in _GROUP_KEYS:
                            start_vector(path)
                            stack.append(_Container("array", path))
                            continue
                        raise BinaryVectorStoreError("源缓存包含非向量数组。")

                    if event == "end_map" or event == "end_array":
                        raise AssertionError("unreachable")

                    if len(path) == 2 and path[0] == "identity":
                        key = path[1]
                        if key == "dimensions":
                            if value is not None and (type(value) is not int or value < 1):
                                raise BinaryVectorStoreError("源identity维度无效。")
                        elif not isinstance(value, str) or not value.strip():
                            raise BinaryVectorStoreError("源identity地址或模型无效。")
                        identity_values[key] = value
                        continue

                    if path == ("schema_version",):
                        if type(value) is not int:
                            raise BinaryVectorStoreError("源缓存schema_version无效。")
                        schema_version = value
                        continue
                    if path == ("vector_dimensions",):
                        if value is not None and (type(value) is not int or value < 1):
                            raise BinaryVectorStoreError("源缓存vector_dimensions无效。")
                        declared_dimensions = value
                        continue

                    if (active_vector is not None and len(path) == 3
                            and path[:2] == (active_vector["group"], active_vector["text"])):
                        if event != "number" or isinstance(value, bool):
                            raise BinaryVectorStoreError("源向量必须仅含JSON数值。")
                        try:
                            number = float(value)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise BinaryVectorStoreError("源向量数值无法表示为float64。") from exc
                        if not math.isfinite(number):
                            raise BinaryVectorStoreError("源向量包含非有限数值。")
                        known_width = (expected_dimensions if expected_dimensions is not None
                                       else declared_dimensions if declared_dimensions is not None
                                       else actual_dimensions)
                        if known_width is not None and len(active_vector["values"]) >= known_width:
                            raise BinaryVectorStoreError("源向量元素数超过已知embedding维度。")
                        active_vector["values"].append(number)
                        continue
                    raise BinaryVectorStoreError("源缓存包含意外标量字段。")

                if event in ("end_map", "end_array"):
                    if not stack:
                        raise BinaryVectorStoreError("源JSON容器结束标记无效。")
                    container = stack[-1]
                    expected_kind = "map" if event == "end_map" else "array"
                    if container.kind != expected_kind or container.pending_key is not None:
                        raise BinaryVectorStoreError("源JSON容器结构不完整。")
                    stack.pop()
                    if expected_kind == "array" and len(container.path) == 2:
                        finish_vector()
                    if not stack:
                        root_finished = True
                    continue

                raise BinaryVectorStoreError("ijson返回了未知JSON事件。")
    except BinaryVectorStoreError:
        raise
    except (ijson.JSONError, OSError, UnicodeError, ValueError, TypeError, OverflowError) as exc:
        raise BinaryVectorStoreError("源embedding缓存不是完整、有效的JSON。") from exc

    if not root_started or not root_finished or stack:
        raise BinaryVectorStoreError("源embedding缓存被截断或缺少顶层对象。")
    if root_keys != _SOURCE_KEYS:
        raise BinaryVectorStoreError("源embedding缓存缺少必要字段。")
    if schema_version != _SOURCE_SCHEMA_VERSION:
        raise BinaryVectorStoreError("源embedding缓存schema版本不支持。")
    try:
        stored_identity = _identity(identity_values)
    except (TypeError, ValueError) as exc:
        raise BinaryVectorStoreError("源embedding identity无效。") from exc
    if stored_identity != expected_identity:
        raise BinaryVectorStoreError("源embedding identity与当前配置不一致。")
    if (declared_dimensions is not None and actual_dimensions is not None
            and declared_dimensions != actual_dimensions):
        raise BinaryVectorStoreError("源缓存实际维度与vector_dimensions不一致。")
    if (expected_dimensions is not None and actual_dimensions is not None
            and expected_dimensions != actual_dimensions):
        raise BinaryVectorStoreError("源实际向量维度与当前embedding identity不一致。")
    if set(identity_values) != _IDENTITY_KEYS:
        raise BinaryVectorStoreError("源embedding identity缺少必要字段。")
    if matrix is not None and actual_dimensions != matrix.shape[1]:
        raise BinaryVectorStoreError("创建向量矩阵时维度不一致。")
    return {
        "identity": stored_identity,
        "actual_dimensions": actual_dimensions,
        "declared_dimensions": declared_dimensions,
        "document_records": document_records,
        "query_records": query_records,
        "matched_texts": matched_texts,
    }


def convert_core_vectors(source_json: str | Path, output_root: str | Path, *,
                         identity: Mapping[str, Any], chunks: Sequence,
                         dtype: str = "float64") -> Path:
    """Stream the JSON cache into an immutable, normalized NPY version.

    Only vectors whose exact document text occurs in the current effective
    chunk sequence are written. Every source vector, including old documents
    and queries, is still parsed and validated. No provider is constructed or
    called. Temporary files are created only below ``output_root``.
    """
    try:
        normalized_identity = _identity(identity)
    except (TypeError, ValueError) as exc:
        raise BinaryVectorStoreError("当前embedding identity无效。") from exc
    if dtype not in {"float64", "float32"}:
        raise ValueError("dtype仅支持float64或显式选择的float32。")
    numpy_dtype = np.dtype(dtype)
    source = Path(source_json)
    root = Path(output_root)
    _check_source_file(source)
    # Vector conversion is an explicit offline job. Keep its established
    # text->row lookup, while the runtime open path below remains lazy.
    rows = _effective_chunks(chunks, materialize_lazy=True)
    row_ids = _row_ids(rows)
    rows_by_text: dict[str, list[int]] = {}
    for row_number, (_, text) in enumerate(rows):
        rows_by_text.setdefault(text, []).append(row_number)
    expected_dimensions = normalized_identity["dimensions"]

    try:
        if _is_symlink_or_junction(root):
            raise BinaryVectorStoreError("向量输出目录不得为符号链接。")
        root.mkdir(parents=True, exist_ok=True)
        if _is_symlink_or_junction(root):
            raise BinaryVectorStoreError("向量输出目录不得为符号链接。")
        versions = root / "versions"
        if _is_symlink_or_junction(versions):
            raise BinaryVectorStoreError("向量versions目录不得为符号链接。")
        if versions.exists():
            versions.resolve(strict=True).relative_to(root.resolve(strict=True))
        versions.mkdir(exist_ok=True)
        try:
            root_resolved = root.resolve(strict=True)
            versions_resolved = versions.resolve(strict=True)
            versions_resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise BinaryVectorStoreError("向量versions目录不得通过junction或重解析点离开输出目录。") from exc
    except BinaryVectorStoreError:
        raise
    except (OSError, ValueError) as exc:
        raise BinaryVectorStoreError("无法创建向量输出目录。") from exc
    if _is_symlink_or_junction(root / "CURRENT"):
        raise BinaryVectorStoreError("向量CURRENT指针不得为符号链接。")

    version_id = "v1-" + uuid4().hex
    temporary = versions / (".tmp-" + uuid4().hex)
    published = versions / version_id
    matrix: np.memmap | None = None
    try:
        temporary.mkdir()

        # Build lazily after the first source vector tells us its actual width.
        # This preserves bounded memory even when identity.dimensions is None.
        source_metadata = _scan_source(
            source, normalized_identity, rows_by_text,
            output_matrix=None, expected_dimensions=expected_dimensions)
        actual_dimensions = source_metadata["actual_dimensions"]
        if actual_dimensions is None or actual_dimensions < 1:
            raise BinaryVectorStoreError("源缓存没有可用的实际向量维度。")
        if source_metadata["matched_texts"] != set(rows_by_text):
            missing = len(set(rows_by_text) - source_metadata["matched_texts"])
            raise BinaryVectorStoreError(f"当前规范chunk有{missing}条embedding文本未命中源缓存。")

        # Second pass writes rows while revalidating the full source stream.
        # The first pass obtains shape without retaining any cache vectors.
        matrix = np.lib.format.open_memmap(
            temporary / "vectors.npy", mode="w+", dtype=numpy_dtype,
            shape=(len(rows), actual_dimensions), fortran_order=False)
        written_metadata = _scan_source(
            source, normalized_identity, rows_by_text,
            output_matrix=matrix, output_dtype=numpy_dtype,
            expected_dimensions=expected_dimensions)
        if (written_metadata["actual_dimensions"] != actual_dimensions
                or written_metadata["matched_texts"] != set(rows_by_text)):
            raise BinaryVectorStoreError("源缓存两次读取结果不一致，拒绝发布。")
        matrix.flush()
        matrix._mmap.close()
        del matrix
        matrix = None
        vectors_path = temporary / "vectors.npy"
        _fsync_file(vectors_path)

        metadata_path = temporary / "metadata.jsonl"
        with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
            for chunk_id, text in rows:
                handle.write(json.dumps({"chunk_id": chunk_id, "text": text},
                                        ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        vector_bytes = len(rows) * actual_dimensions * numpy_dtype.itemsize
        manifest = {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "version_id": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": normalized_identity,
            "actual_dimensions": actual_dimensions,
            "dtype": dtype,
            "shape": [len(rows), actual_dimensions],
            "normalized": "float64 LocalIndex._document_vectors L2 row normalization",
            "source_schema_version": _SOURCE_SCHEMA_VERSION,
            "source_name": source.name,
            "source_is_output_sibling": source.parent.resolve() == root.parent.resolve(),
            "source_vector_dimensions": written_metadata["declared_dimensions"],
            "source_document_records_scanned": written_metadata["document_records"],
            "source_query_records_scanned": written_metadata["query_records"],
            "matched_unique_embedding_texts": len(rows_by_text),
            "row_ids": row_ids,
            "vector_bytes": vector_bytes,
            "vectors_file_bytes": vectors_path.stat().st_size,
            "metadata_bytes": metadata_path.stat().st_size,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        temporary.replace(published)
        _fsync_directory(versions)

        pointer_temp = root / (".CURRENT-" + uuid4().hex + ".tmp")
        try:
            with pointer_temp.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(version_id + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            pointer_temp.replace(root / "CURRENT")
            _fsync_directory(root)
        finally:
            pointer_temp.unlink(missing_ok=True)
        return published
    except BinaryVectorStoreError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise BinaryVectorStoreError("二进制向量转换失败；CURRENT未切换。") from exc
    finally:
        if matrix is not None:
            try:
                matrix.flush()
            except Exception:
                pass
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
            del matrix
        if temporary.exists():
            try:
                # A build may have failed after the directory was redirected.
                # Only remove the original direct child of the verified root.
                if (not any(_is_symlink_or_junction(path)
                            for path in (root, versions, temporary))
                        and re.fullmatch(r"\.tmp-[0-9a-f]{32}", temporary.name)):
                    current_root = root.resolve(strict=True)
                    current_versions = versions.resolve(strict=True)
                    resolved_temporary = temporary.resolve(strict=True)
                    if (current_root == root_resolved
                            and current_versions == versions_resolved
                            and resolved_temporary.parent == versions_resolved):
                        shutil.rmtree(temporary)
            except (OSError, ValueError):
                # An incomplete unpublished directory is harmless and is never
                # selected by CURRENT. Do not touch any previous version.
                pass


class BinaryVectorStore:
    """Validated read-only mmap of the CURRENT immutable vector version."""

    def __init__(self, version_dir: Path, vectors: np.memmap,
                 row_by_chunk_id: dict[str, int], manifest: dict):
        self.version_dir = version_dir
        self.vectors = vectors
        self.row_by_chunk_id = row_by_chunk_id
        self.manifest = manifest
        self._closed = False

    def close(self) -> None:
        """Idempotently release this process's read-only mmap handle."""
        if self._closed:
            return
        mmap = getattr(self.vectors, "_mmap", None)
        if mmap is not None and not mmap.closed:
            mmap.close()
        self._closed = True

    def __enter__(self) -> "BinaryVectorStore":
        if self._closed:
            raise BinaryVectorStoreError("BinaryVectorStore已关闭。")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @classmethod
    def open(cls, output_root: str | Path, *, identity: Mapping[str, Any],
             chunks: Sequence) -> "BinaryVectorStore":
        try:
            expected_identity = _identity(identity)
        except (TypeError, ValueError) as exc:
            raise BinaryVectorStoreError("当前embedding identity无效。") from exc
        rows = _effective_chunks(chunks)
        row_ids = _row_ids(rows)
        root = Path(output_root)
        if _is_symlink_or_junction(root):
            raise BinaryVectorStoreError("向量输出目录不得为符号链接。")
        version_id = _current_version(root)
        versions = root / "versions"
        version_dir = versions / version_id
        try:
            resolved_root = root.resolve(strict=True)
            resolved_versions = versions.resolve(strict=True)
            resolved_version = version_dir.resolve(strict=True)
            resolved_version.relative_to(resolved_versions)
            resolved_versions.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise BinaryVectorStoreError("CURRENT指向versions目录之外或目标不存在。") from exc
        if (_is_symlink_or_junction(version_dir)
                or _is_symlink_or_junction(versions)):
            raise BinaryVectorStoreError("向量版本路径不得为符号链接。")

        manifest_path = version_dir / "manifest.json"
        metadata_path = version_dir / "metadata.jsonl"
        vectors_path = version_dir / "vectors.npy"
        if any(_is_symlink_or_junction(path)
               for path in (manifest_path, metadata_path, vectors_path)):
            raise BinaryVectorStoreError("向量版本文件不得为符号链接。")
        manifest = _read_json_object(manifest_path)
        vectors = None
        try:
            manifest_identity = _identity(manifest.get("identity"))
        except (TypeError, ValueError) as exc:
            raise BinaryVectorStoreError("向量manifest的embedding identity无效。") from exc
        if (manifest.get("format") != _FORMAT
                or type(manifest.get("format_version")) is not int
                or manifest.get("format_version") != _FORMAT_VERSION
                or manifest.get("version_id") != version_id
                or manifest_identity != expected_identity
                or type(manifest.get("source_schema_version")) is not int
                or manifest.get("source_schema_version") != _SOURCE_SCHEMA_VERSION):
            raise BinaryVectorStoreError("向量manifest格式、版本或embedding identity不匹配。")
        source_name = manifest.get("source_name")
        if (not isinstance(source_name, str) or not source_name
                or Path(source_name).name != source_name
                or "/" in source_name or "\\" in source_name or ":" in source_name
                or type(manifest.get("source_is_output_sibling")) is not bool):
            raise BinaryVectorStoreError("manifest源缓存信息无效。")
        if manifest.get("source_is_output_sibling") is True:
            sibling_source = root.parent / source_name
            if sibling_source.exists() or _is_symlink_or_junction(sibling_source):
                if _is_symlink_or_junction(sibling_source):
                    raise BinaryVectorStoreError("同目录源embedding缓存不得为符号链接。")
                _check_source_file(sibling_source)
        dtype_name = manifest.get("dtype")
        if dtype_name not in {"float64", "float32"}:
            raise BinaryVectorStoreError("向量manifest dtype无效。")
        try:
            dtype = np.dtype(dtype_name)
            shape = manifest.get("shape")
            dimensions = manifest.get("actual_dimensions")
            expected_shape = [len(rows), dimensions]
            if (type(dimensions) is not int or dimensions < 1
                    or type(shape) is not list or shape != expected_shape
                    or any(type(value) is not int for value in shape)
                    or manifest.get("normalized") != "float64 LocalIndex._document_vectors L2 row normalization"
                    or manifest.get("row_ids") != row_ids):
                raise BinaryVectorStoreError("向量manifest的维度、归一化声明或行顺序无效。")
            if (type(manifest.get("matched_unique_embedding_texts")) is not int
                    or manifest["matched_unique_embedding_texts"] != _unique_embedding_text_count(rows)):
                raise BinaryVectorStoreError("manifest匹配的embedding文本计数无效。")
            source_dimensions = manifest.get("source_vector_dimensions")
            if (source_dimensions is not None
                    and (type(source_dimensions) is not int or source_dimensions != dimensions)):
                raise BinaryVectorStoreError("manifest源向量维度与矩阵维度不一致。")
            for field in ("source_document_records_scanned", "source_query_records_scanned"):
                if type(manifest.get(field)) is not int or manifest[field] < 0:
                    raise BinaryVectorStoreError("manifest源记录计数无效。")
            if manifest["source_document_records_scanned"] < manifest["matched_unique_embedding_texts"]:
                raise BinaryVectorStoreError("manifest文档记录数小于匹配文本数。")
            if (expected_identity["dimensions"] is not None
                    and expected_identity["dimensions"] != dimensions):
                raise BinaryVectorStoreError("向量实际维度与当前embedding identity不匹配。")
            expected_bytes = len(rows) * dimensions * dtype.itemsize
            vectors_file_bytes = manifest.get("vectors_file_bytes")
            if (type(manifest.get("vector_bytes")) is not int
                    or manifest["vector_bytes"] != expected_bytes
                    or type(vectors_file_bytes) is not int
                    or vectors_file_bytes != vectors_path.stat().st_size
                    or vectors_path.stat().st_size < expected_bytes):
                raise BinaryVectorStoreError("vectors.npy文件长度与manifest不匹配。")
            metadata_bytes = metadata_path.stat().st_size
            if (type(manifest.get("metadata_bytes")) is not int
                    or manifest["metadata_bytes"] != metadata_bytes):
                raise BinaryVectorStoreError("metadata.jsonl文件长度与manifest不匹配。")
            row_by_chunk_id: dict[str, int] = {}
            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                for row_number, expected_row in enumerate(rows):
                    line = handle.readline()
                    if not line:
                        raise BinaryVectorStoreError("metadata记录数少于当前chunk数。")
                    if not line.endswith("\n"):
                        raise BinaryVectorStoreError("metadata.jsonl最后一行不完整。")
                    try:
                        item = json.loads(line, object_pairs_hook=_json_no_duplicate_pairs)
                    except (json.JSONDecodeError, BinaryVectorStoreError) as exc:
                        raise BinaryVectorStoreError("metadata.jsonl包含无效记录。") from exc
                    if (not isinstance(item, dict) or set(item) != {"chunk_id", "text"}
                            or item.get("chunk_id") != expected_row[0]
                            or item.get("text") != expected_row[1]):
                        raise BinaryVectorStoreError("metadata的chunk ID、embedding text或顺序已变化。")
                    if item["chunk_id"] in row_by_chunk_id:
                        raise BinaryVectorStoreError("metadata包含重复chunk ID。")
                    row_by_chunk_id[item["chunk_id"]] = row_number
                if handle.readline() or len(row_by_chunk_id) != len(rows):
                    raise BinaryVectorStoreError("metadata记录数与当前chunk数不一致。")

            vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
            if (not isinstance(vectors, np.memmap) or vectors.shape != tuple(expected_shape)
                    or vectors.dtype != dtype or vectors.flags.writeable
                    or not vectors.flags.c_contiguous):
                raise BinaryVectorStoreError("vectors.npy header与manifest不一致。")
            if vectors.offset + vectors.nbytes != vectors_file_bytes:
                raise BinaryVectorStoreError("vectors.npy header长度与文件长度不一致。")
            # Scan with a small bounded block so silent file corruption (NaN,
            # Inf, zero rows, or broken normalization) fails closed at startup.
            block_rows = 256
            with vectors_path.open("rb") as vector_file:
                header = vector_file.read(vectors.offset)
            if len(header) != vectors.offset:
                raise BinaryVectorStoreError("vectors.npy header被截断。")
            for start in range(0, len(rows), block_rows):
                block = np.asarray(vectors[start:start + block_rows])
                if not np.isfinite(block).all():
                    raise BinaryVectorStoreError("vectors.npy包含非有限值。")
                with np.errstate(over="ignore", invalid="ignore"):
                    norms = np.linalg.norm(block.astype(np.float64, copy=False), axis=1)
                tolerance = 5e-6 if dtype == np.dtype("float32") else 1e-12
                if (not np.isfinite(norms).all() or np.any(norms == 0)
                        or not np.allclose(norms, 1.0, rtol=tolerance, atol=tolerance)):
                    raise BinaryVectorStoreError("vectors.npy包含零向量或未归一化行。")
            store = cls(version_dir, vectors, row_by_chunk_id, manifest)
            vectors = None  # ownership moved to the returned store
            return store
        except BinaryVectorStoreError:
            raise
        except (OSError, ValueError, TypeError, UnicodeError, OverflowError) as exc:
            raise BinaryVectorStoreError("向量版本损坏或无法读取。") from exc
        finally:
            if vectors is not None:
                mmap = getattr(vectors, "_mmap", None)
                if mmap is not None and not mmap.closed:
                    mmap.close()
