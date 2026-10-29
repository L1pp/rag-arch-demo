"""Compact immutable BM25 postings stored in read-only NumPy mmap files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any
import unicodedata
from uuid import uuid4

import numpy as np


_FORMAT = "rag_arch.core_bm25"
_FORMAT_VERSION = 1
_CURRENT_RE = re.compile(r"^v1-[0-9a-f]{32}$")
_MAX_VOCABULARY_TERMS = 1_000_000
_POSTING_BLOCK = 65_536
_SCORE_BLOCK = 32_768
_ID_BLOCK = 65_536

# Bump rules_id whenever tokenize's NFKC/lowercase, CJK unigram/bigram, or
# operator normalization behavior changes. The current regex is also recorded
# so a pattern-only code change invalidates older packages automatically.
_TOKENIZER_RULES_ID = "nfkc-lower-cjk-unigram-bigram-numeric-operators-v1"


class BinaryBM25StoreError(ValueError):
    """The current BM25 package is missing, stale, malformed, or closed."""


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
        raise BinaryBM25StoreError("BM25包文件缺失或无法stat。") from exc


def _json_no_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BinaryBM25StoreError("JSON对象包含重复字段。")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BinaryBM25StoreError("BM25 manifest不是有效JSON。") from exc
    if not isinstance(value, dict):
        raise BinaryBM25StoreError("BM25 manifest格式无效。")
    return value


def _tokenizer_parts():
    # Lazy import avoids retrieval -> binary_bm25 -> retrieval import cycles.
    from .retrieval import _TOKENS, tokenize

    contract = {
        "name": "rag_arch.retrieval.tokenize",
        "rules_id": _TOKENIZER_RULES_ID,
        "pattern": _TOKENS.pattern,
        "unicode_version": unicodedata.unidata_version,
        "normalization": "NFKC then lower; <=, >=, != normalize from ≤, ≥, ≠",
        "cjk": "unigram plus adjacent bigram",
    }
    return tokenize, contract


def _effective_chunks(chunks: Sequence):
    try:
        from .retrieval import _validated_lazy_chunk_sequence
        lazy_rows = _validated_lazy_chunk_sequence(chunks, BinaryBM25StoreError)
    except BinaryBM25StoreError:
        raise
    except (TypeError, ValueError) as exc:
        raise BinaryBM25StoreError("BM25 chunks序列无效。") from exc
    if lazy_rows is not None:
        return lazy_rows
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise TypeError("chunks须为EvidenceIndex/LocalIndex chunk序列。")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None)
        text = getattr(chunk, "text", None)
        if not isinstance(chunk_id, str) or not chunk_id:
            raise BinaryBM25StoreError("chunk_id须为非空文本。")
        if not isinstance(text, str):
            raise BinaryBM25StoreError("chunk text须为文本。")
        if not text.strip():
            continue
        if chunk_id in seen:
            raise BinaryBM25StoreError("当前chunk_id不唯一，拒绝绑定BM25行。")
        seen.add(chunk_id)
        rows.append((chunk_id, text))
    return rows


def _row_ids(rows) -> list[str]:
    if getattr(rows, "is_lazy_chunk_rows", False) is True:
        return rows.row_ids
    return [chunk_id for chunk_id, _ in rows]


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    mmap = getattr(array, "_mmap", None)
    if mmap is not None and not mmap.closed:
        mmap.close()


def _collect_statistics(rows, tokenize):
    vocabulary: dict[str, int] = {}
    document_frequencies: list[int] = []
    document_lengths: list[int] = []
    posting_count = 0
    for _, text in rows:
        counts = Counter(tokenize(text))
        document_lengths.append(sum(counts.values()))
        posting_count += len(counts)
        for term in counts:
            term_id = vocabulary.get(term)
            if term_id is None:
                if len(vocabulary) >= _MAX_VOCABULARY_TERMS:
                    raise BinaryBM25StoreError(
                        f"词汇表超过安全上限{_MAX_VOCABULARY_TERMS}。")
                term_id = len(vocabulary)
                vocabulary[term] = term_id
                document_frequencies.append(0)
            document_frequencies[term_id] += 1
    if posting_count > np.iinfo(np.uint64).max:
        raise BinaryBM25StoreError("倒排项数量超出uint64范围。")
    return vocabulary, document_frequencies, document_lengths, posting_count


def _manifest_path(directory: Path) -> tuple[str, Path]:
    if _is_link_or_junction(directory):
        raise BinaryBM25StoreError("BM25输出目录不得为符号链接或junction。")
    pointer = directory / "CURRENT"
    if _is_link_or_junction(pointer):
        raise BinaryBM25StoreError("BM25 CURRENT不得为符号链接。")
    try:
        content = pointer.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise BinaryBM25StoreError("BM25 CURRENT缺失或不可读。") from exc
    version_id = content[:-1] if content.endswith("\n") else content
    if (content not in (version_id, version_id + "\n")
            or not _CURRENT_RE.fullmatch(version_id)):
        raise BinaryBM25StoreError("BM25 CURRENT格式无效。")
    versions = directory / "versions"
    version_dir = versions / version_id
    try:
        resolved_directory = directory.resolve(strict=True)
        resolved_versions = versions.resolve(strict=True)
        resolved_version = version_dir.resolve(strict=True)
        resolved_versions.relative_to(resolved_directory)
        resolved_version.relative_to(resolved_versions)
    except (OSError, ValueError) as exc:
        raise BinaryBM25StoreError("BM25 CURRENT指向输出目录之外或目标不存在。") from exc
    if any(_is_link_or_junction(path) for path in (directory, versions, version_dir)):
        raise BinaryBM25StoreError("BM25版本目录不得为符号链接或junction。")
    return version_id, version_dir


def build_core_bm25(directory: str | Path, *, chunks: Sequence,
                    source_paths=(), expected_source_stats=None) -> Path:
    """Build and atomically publish compact BM25 postings for current chunks.

    Only one chunk's ``Counter`` is alive at a time. Two passes build document
    frequencies first, then scatter postings through mmap cursors. The legacy
    source_paths and expected_source_stats arguments are accepted for caller
    compatibility but do not trigger source snapshots or comparisons.
    """
    rows = _effective_chunks(chunks)
    tokenize, tokenizer_contract = _tokenizer_parts()
    root = Path(directory)
    if _is_link_or_junction(root):
        raise BinaryBM25StoreError("BM25输出目录不得为符号链接或junction。")
    try:
        root.mkdir(parents=True, exist_ok=True)
        if _is_link_or_junction(root):
            raise BinaryBM25StoreError("BM25输出目录不得为符号链接或junction。")
        versions = root / "versions"
        if _is_link_or_junction(versions):
            raise BinaryBM25StoreError("BM25 versions目录不得为符号链接或junction。")
        if versions.exists():
            versions.resolve(strict=True).relative_to(root.resolve(strict=True))
        versions.mkdir(exist_ok=True)
        resolved_root = root.resolve(strict=True)
        resolved_versions = versions.resolve(strict=True)
        resolved_versions.relative_to(resolved_root)
    except BinaryBM25StoreError:
        raise
    except (OSError, ValueError) as exc:
        raise BinaryBM25StoreError("无法在BM25输出子树创建versions目录。") from exc
    if _is_link_or_junction(root / "CURRENT"):
        raise BinaryBM25StoreError("BM25 CURRENT不得为符号链接或junction。")

    version_id = "v1-" + uuid4().hex
    temporary = versions / (".tmp-" + uuid4().hex)
    published = versions / version_id
    mapped: list[np.memmap] = []
    try:
        temporary.mkdir()
        vocabulary, document_frequencies, document_lengths, posting_count = (
            _collect_statistics(rows, tokenize))
        document_count = len(rows)
        vocabulary_count = len(vocabulary)
        max_uint32 = int(np.iinfo(np.uint32).max)
        max_uint64 = int(np.iinfo(np.uint64).max)
        total_document_tokens = sum(document_lengths)
        max_document_length = max(document_lengths, default=0)
        if (document_count > max_uint64 or total_document_tokens > max_uint64
                or max_document_length > max_uint64):
            raise BinaryBM25StoreError("文档数或token计数超出uint64范围。")
        id_dtype = np.dtype(np.uint32 if document_count <= max_uint32 + 1 else np.uint64)
        df_dtype = np.dtype(np.uint32 if document_count <= max_uint32 else np.uint64)
        tf_dtype = np.dtype(np.uint32 if max_document_length <= max_uint32 else np.uint64)
        length_dtype = tf_dtype

        offsets = np.zeros(vocabulary_count + 1, dtype=np.uint64)
        if vocabulary_count:
            np.cumsum(np.asarray(document_frequencies, dtype=np.uint64),
                      out=offsets[1:])
        if int(offsets[-1]) != posting_count:
            raise BinaryBM25StoreError("内部倒排统计不一致。")

        physical_docs = max(1, document_count)
        physical_vocab = max(1, vocabulary_count)
        physical_postings = max(1, posting_count)
        postings_dtype = np.dtype([("doc_id", id_dtype), ("tf", tf_dtype)])
        array_specs = (
            ("offsets.npy", np.dtype(np.uint64), (vocabulary_count + 1,)),
            ("doc_lengths.npy", length_dtype, (physical_docs,)),
            ("doc_freqs.npy", df_dtype, (physical_vocab,)),
            ("postings.npy", postings_dtype, (physical_postings,)),
        )
        array_paths = {}
        for filename, dtype, shape in array_specs:
            array = np.lib.format.open_memmap(
                temporary / filename, mode="w+", dtype=dtype, shape=shape)
            mapped.append(array)
            array_paths[filename] = array
        offsets_map = array_paths["offsets.npy"]
        lengths_map = array_paths["doc_lengths.npy"]
        dfs_map = array_paths["doc_freqs.npy"]
        postings_map = array_paths["postings.npy"]
        offsets_map[:] = offsets
        if document_count:
            lengths_map[:document_count] = np.asarray(document_lengths, dtype=length_dtype)
        else:
            lengths_map[0] = 0
        if vocabulary_count:
            dfs_map[:vocabulary_count] = np.asarray(document_frequencies, dtype=df_dtype)
        else:
            dfs_map[0] = 0
        if posting_count == 0:
            postings_map[0] = (0, 0)

        cursors = offsets[:-1].copy()
        for doc_id, (_, text) in enumerate(rows):
            counts = Counter(tokenize(text))
            if sum(counts.values()) != document_lengths[doc_id]:
                raise BinaryBM25StoreError("构建期间tokenizer输出发生变化。")
            for term, tf in counts.items():
                term_id = vocabulary.get(term)
                if term_id is None:
                    raise BinaryBM25StoreError("第二遍tokenizer产生了新词。")
                position = int(cursors[term_id])
                if position >= int(offsets[term_id + 1]):
                    raise BinaryBM25StoreError("倒排cursor越界。")
                postings_map[position] = (doc_id, tf)
                cursors[term_id] += 1
        if vocabulary_count and not np.array_equal(cursors, offsets[1:]):
            raise BinaryBM25StoreError("倒排cursor未完整填充。")
        for array in mapped:
            array.flush()
            _close_memmap(array)
        mapped.clear()
        for filename in array_paths:
            _fsync_file(temporary / filename)

        vocabulary_path = temporary / "vocabulary.jsonl"
        with vocabulary_path.open("x", encoding="utf-8", newline="\n") as handle:
            for term in vocabulary:
                handle.write(json.dumps(term, ensure_ascii=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        metadata_path = temporary / "metadata.jsonl"
        with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
            for chunk_id, text in rows:
                handle.write(json.dumps({"chunk_id": chunk_id, "text": text},
                                        ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        average_length = (sum(document_lengths) / document_count
                          if document_count else 0.0)
        manifest = {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "version_id": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer": tokenizer_contract,
            "document_count": document_count,
            "average_length": average_length,
            "total_document_tokens": total_document_tokens,
            "vocabulary_count": vocabulary_count,
            "posting_count": posting_count,
            "id_dtype": id_dtype.name,
            "tf_dtype": tf_dtype.name,
            "length_dtype": length_dtype.name,
            "df_dtype": df_dtype.name,
            "row_ids": _row_ids(rows),
            "arrays": {
                "offsets.npy": {"shape": [vocabulary_count + 1], "dtype": "uint64"},
                "doc_lengths.npy": {"shape": [physical_docs], "dtype": length_dtype.name},
                "doc_freqs.npy": {"shape": [physical_vocab], "dtype": df_dtype.name},
                "postings.npy": {
                    "shape": [physical_postings],
                    "doc_id_dtype": id_dtype.name,
                    "tf_dtype": tf_dtype.name,
                },
            },
            "array_bytes": {
                filename: (temporary / filename).stat().st_size for filename in array_paths
            },
            "vocabulary_bytes": vocabulary_path.stat().st_size,
            "metadata_bytes": metadata_path.stat().st_size,
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
    except BinaryBM25StoreError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise BinaryBM25StoreError("BM25构建失败；CURRENT未切换。") from exc
    finally:
        for array in mapped:
            try:
                array.flush()
            except Exception:
                pass
            _close_memmap(array)
        if temporary.exists():
            try:
                # Do not recursively remove a path that was redirected while
                # the build ran. The UUID temp directory must still resolve as
                # a direct child of the same versions directory we created.
                if (not _is_link_or_junction(temporary)
                        and re.fullmatch(r"\.tmp-[0-9a-f]{32}", temporary.name)):
                    current_root = root.resolve(strict=True)
                    current_versions = versions.resolve(strict=True)
                    resolved_temporary = temporary.resolve(strict=True)
                    if (current_root == resolved_root
                            and current_versions == resolved_versions
                            and resolved_temporary.parent == resolved_versions):
                        shutil.rmtree(temporary)
            except (OSError, ValueError):
                pass


class BinaryBM25Store:
    """Validated BM25 postings mapped read-only for concurrent scoring."""

    def __init__(self, *, version_dir: Path, manifest: dict,
                 vocabulary: dict[str, int], offsets: np.memmap,
                 postings: np.memmap, doc_lengths: np.ndarray,
                 doc_freqs: np.ndarray, maps: list[np.memmap]):
        self.version_dir = version_dir
        self.manifest = manifest
        self.term_to_id = vocabulary
        self.offsets = offsets
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_freqs = doc_freqs
        self.document_count = manifest["document_count"]
        self.average_length = manifest["average_length"]
        self.vocabulary_count = manifest["vocabulary_count"]
        self.posting_count = manifest["posting_count"]
        self._maps = maps
        self._condition = threading.Condition(threading.RLock())
        self._state = "open"
        self._active_scores = 0

    @contextmanager
    def _score_lease(self):
        with self._condition:
            if self._state != "open":
                raise BinaryBM25StoreError("BinaryBM25Store正在关闭或已关闭。")
            self._active_scores += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_scores -= 1
                if self._active_scores == 0:
                    self._condition.notify_all()

    def scores(self, query: str) -> list[float]:
        """Return scores with LocalIndex's exact Python-float accumulation order."""
        tokenize, _ = _tokenizer_parts()
        query_terms = set(tokenize(query))
        scores = [0.0] * self.document_count
        with self._score_lease():
            if not self.average_length:
                return scores
            n = self.document_count
            k1, b = 1.5, 0.75
            for term in query_terms:
                term_id = self.term_to_id.get(term)
                if term_id is None:
                    continue
                frequency = int(self.doc_freqs[term_id])
                if not frequency:
                    continue
                idf = math.log1p((n - frequency + 0.5) / (frequency + 0.5))
                start = int(self.offsets[term_id])
                end = int(self.offsets[term_id + 1])
                for block_start in range(start, end, _SCORE_BLOCK):
                    block_end = min(end, block_start + _SCORE_BLOCK)
                    # Structured ndarray indexing creates a NumPy scalar per
                    # posting. A bounded Python tuple block is materially
                    # faster and preserves the exact document accumulation
                    # order of LocalIndex's enumerate loop.
                    for doc_id_value, tf_value in self.postings[block_start:block_end].tolist():
                        doc_id = int(doc_id_value)
                        tf = int(tf_value)
                        denominator = tf + k1 * (
                            1 - b + b * int(self.doc_lengths[doc_id]) / self.average_length)
                        scores[doc_id] += idf * tf * (k1 + 1) / denominator
        return scores

    def close(self) -> None:
        """Stop new scorers, drain active readers, then close every mmap."""
        with self._condition:
            if self._state == "closed":
                return
            self._state = "closing"
            while self._active_scores:
                self._condition.wait()
            for array in self._maps:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None and not mmap.closed:
                    try:
                        mmap.close()
                    except BufferError as exc:
                        # Keep closing state so no new score can race the map;
                        # callers may release exported views and retry close.
                        raise BinaryBM25StoreError(
                            "BM25 mmap仍有外部视图；释放视图后可重试close。") from exc
            self._state = "closed"
            self._condition.notify_all()

    def __enter__(self) -> "BinaryBM25Store":
        with self._condition:
            if self._state != "open":
                raise BinaryBM25StoreError("BinaryBM25Store正在关闭或已关闭。")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @classmethod
    def open(cls, directory: str | Path, *, chunks: Sequence) -> "BinaryBM25Store":
        rows = _effective_chunks(chunks)
        row_ids = _row_ids(rows)
        root = Path(directory)
        version_id, version_dir = _manifest_path(root)
        filenames = ("offsets.npy", "postings.npy", "doc_lengths.npy",
                     "doc_freqs.npy", "vocabulary.jsonl", "metadata.jsonl", "manifest.json")
        paths = {name: version_dir / name for name in filenames}
        if any(_is_link_or_junction(path) for path in paths.values()):
            raise BinaryBM25StoreError("BM25版本文件不得为符号链接或junction。")
        manifest = _read_json_object(paths["manifest.json"])
        tokenize, tokenizer_contract = _tokenizer_parts()
        del tokenize
        if (manifest.get("format") != _FORMAT
                or type(manifest.get("format_version")) is not int
                or manifest.get("format_version") != _FORMAT_VERSION
                or manifest.get("version_id") != version_id
                or manifest.get("tokenizer") != tokenizer_contract):
            raise BinaryBM25StoreError("BM25格式、版本或tokenizer identity不匹配。")
        document_count = manifest.get("document_count")
        vocabulary_count = manifest.get("vocabulary_count")
        posting_count = manifest.get("posting_count")
        average_length = manifest.get("average_length")
        if (type(document_count) is not int or document_count != len(rows)
                or type(vocabulary_count) is not int or not 0 <= vocabulary_count <= _MAX_VOCABULARY_TERMS
                or type(posting_count) is not int or posting_count < 0
                or not isinstance(average_length, (int, float))
                or isinstance(average_length, bool) or not math.isfinite(average_length)
                or average_length < 0
                or manifest.get("row_ids") != row_ids):
            raise BinaryBM25StoreError("BM25 manifest的语料行数、顺序或尺寸无效。")
        if type(manifest.get("total_document_tokens")) is not int or manifest["total_document_tokens"] < 0:
            raise BinaryBM25StoreError("BM25 manifest的总token数无效。")

        expected_dtypes = {"uint32": np.dtype(np.uint32), "uint64": np.dtype(np.uint64)}
        for field in ("id_dtype", "tf_dtype", "length_dtype", "df_dtype"):
            if manifest.get(field) not in expected_dtypes:
                raise BinaryBM25StoreError("BM25 manifest中的整数dtype无效。")
        id_dtype = expected_dtypes[manifest["id_dtype"]]
        tf_dtype = expected_dtypes[manifest["tf_dtype"]]
        length_dtype = expected_dtypes[manifest["length_dtype"]]
        df_dtype = expected_dtypes[manifest["df_dtype"]]
        max_uint32 = int(np.iinfo(np.uint32).max)
        expected_id_dtype = np.dtype(np.uint32 if document_count <= max_uint32 + 1
                                     else np.uint64)
        expected_df_dtype = np.dtype(np.uint32 if document_count <= max_uint32
                                     else np.uint64)
        if (length_dtype != tf_dtype or id_dtype != expected_id_dtype
                or df_dtype != expected_df_dtype):
            raise BinaryBM25StoreError("BM25 manifest中的数组dtype关系无效。")

        expected_shapes = {
            "offsets.npy": (vocabulary_count + 1,),
            "doc_lengths.npy": (max(1, document_count),),
            "doc_freqs.npy": (max(1, vocabulary_count),),
            "postings.npy": (max(1, posting_count),),
        }
        expected_dtypes_by_file = {
            "offsets.npy": np.dtype(np.uint64),
            "doc_lengths.npy": length_dtype,
            "doc_freqs.npy": df_dtype,
            "postings.npy": np.dtype([("doc_id", id_dtype), ("tf", tf_dtype)]),
        }
        arrays_manifest = manifest.get("arrays")
        file_bytes = manifest.get("array_bytes")
        if not isinstance(arrays_manifest, dict) or not isinstance(file_bytes, dict):
            raise BinaryBM25StoreError("BM25 manifest数组目录无效。")
        expected_array_names = set(expected_shapes)
        if (set(arrays_manifest) != expected_array_names
                or set(file_bytes) != expected_array_names):
            raise BinaryBM25StoreError("BM25 manifest数组文件集合无效。")
        for filename, shape in expected_shapes.items():
            expected_meta = arrays_manifest[filename]
            if filename == "postings.npy":
                valid_meta = (expected_meta == {
                    "shape": list(shape), "doc_id_dtype": id_dtype.name,
                    "tf_dtype": tf_dtype.name})
            else:
                valid_meta = (expected_meta == {
                    "shape": list(shape), "dtype": expected_dtypes_by_file[filename].name})
            if not valid_meta:
                raise BinaryBM25StoreError("BM25 manifest数组shape或dtype无效。")
            if (type(file_bytes.get(filename)) is not int
                    or file_bytes[filename] != _file_size(paths[filename])):
                raise BinaryBM25StoreError("BM25数组文件长度无效。")

        if (type(manifest.get("vocabulary_bytes")) is not int
                or manifest["vocabulary_bytes"] != _file_size(paths["vocabulary.jsonl"])
                or type(manifest.get("metadata_bytes")) is not int
                or manifest["metadata_bytes"] != _file_size(paths["metadata.jsonl"])):
            raise BinaryBM25StoreError("BM25词表或元数据长度无效。")

        vocabulary: dict[str, int] = {}
        with paths["vocabulary.jsonl"].open("r", encoding="utf-8", newline="") as handle:
            for term_id in range(vocabulary_count):
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    raise BinaryBM25StoreError("BM25词表记录缺失或不完整。")
                try:
                    term = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BinaryBM25StoreError("BM25词表包含无效JSON。") from exc
                if not isinstance(term, str) or not term or term in vocabulary:
                    raise BinaryBM25StoreError("BM25词表含无效或重复term。")
                vocabulary[term] = term_id
            if handle.readline():
                raise BinaryBM25StoreError("BM25词表行数超过manifest声明。")

        if len(row_ids) != document_count:
            raise BinaryBM25StoreError("BM25 metadata行数与manifest不一致。")
        with paths["metadata.jsonl"].open("r", encoding="utf-8", newline="") as handle:
            for line_number, expected_row in enumerate(rows):
                line = handle.readline()
                if not line:
                    raise BinaryBM25StoreError("BM25 metadata行数少于当前chunk数。")
                if not line.endswith("\n"):
                    raise BinaryBM25StoreError("BM25 metadata行不完整。")
                try:
                    record = json.loads(line, object_pairs_hook=_json_no_duplicate_pairs)
                except (json.JSONDecodeError, BinaryBM25StoreError) as exc:
                    raise BinaryBM25StoreError("BM25 metadata包含无效记录。") from exc
                if (not isinstance(record, dict) or set(record) != {"chunk_id", "text"}
                        or record.get("chunk_id") != expected_row[0]
                        or record.get("text") != expected_row[1]):
                    raise BinaryBM25StoreError("BM25 metadata的chunk ID、文本或行顺序已变化。")
            if next(handle, None) is not None:
                raise BinaryBM25StoreError("BM25 metadata行数超过当前chunk数。")

        maps: list[np.memmap] = []
        try:
            loaded = {}
            for filename in expected_shapes:
                try:
                    array = np.load(paths[filename], mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError, TypeError, EOFError) as exc:
                    raise BinaryBM25StoreError(f"{filename}不是可读取的安全NPY文件。") from exc
                maps.append(array)
                if (not isinstance(array, np.memmap) or array.shape != expected_shapes[filename]
                        or array.dtype != expected_dtypes_by_file[filename]
                        or array.flags.writeable or not array.flags.c_contiguous):
                    raise BinaryBM25StoreError("BM25 NPY数组header与manifest不一致。")
                if array.offset + array.nbytes != file_bytes[filename]:
                    raise BinaryBM25StoreError("BM25 NPY header数据区长度与文件长度不一致。")
                loaded[filename] = array
            offsets = loaded["offsets.npy"]
            doc_lengths_base = loaded["doc_lengths.npy"]
            doc_freqs_base = loaded["doc_freqs.npy"]
            postings_base = loaded["postings.npy"]
            doc_lengths = doc_lengths_base[:document_count]
            doc_freqs = doc_freqs_base[:vocabulary_count]
            postings = postings_base[:posting_count]
            if document_count == 0 and int(doc_lengths_base[0]) != 0:
                raise BinaryBM25StoreError("空BM25库的doc_lengths sentinel无效。")
            if vocabulary_count == 0 and int(doc_freqs_base[0]) != 0:
                raise BinaryBM25StoreError("空BM25词表的df sentinel无效。")
            if posting_count == 0 and (int(postings_base[0]["doc_id"]) != 0
                                      or int(postings_base[0]["tf"]) != 0):
                raise BinaryBM25StoreError("空BM25倒排表sentinel无效。")

            if int(offsets[0]) != 0 or int(offsets[-1]) != posting_count:
                raise BinaryBM25StoreError("BM25 offsets边界无效。")
            if vocabulary_count:
                if np.any(offsets[1:] < offsets[:-1]):
                    raise BinaryBM25StoreError("BM25 offsets不是单调递增。")
                for start in range(0, vocabulary_count, _ID_BLOCK):
                    end = min(vocabulary_count, start + _ID_BLOCK)
                    delta = offsets[start + 1:end + 1] - offsets[start:end]
                    if not np.array_equal(delta, doc_freqs[start:end]):
                        raise BinaryBM25StoreError("BM25 doc_freqs与offsets不一致。")

            totals = np.zeros(document_count, dtype=np.uint64)
            previous_doc = -1
            previous_term = -1
            for start in range(0, posting_count, _POSTING_BLOCK):
                end = min(posting_count, start + _POSTING_BLOCK)
                block = postings[start:end]
                ids = block["doc_id"]
                freqs = block["tf"]
                if (np.any(ids >= document_count) or np.any(freqs == 0)):
                    raise BinaryBM25StoreError("BM25 posting包含越界doc_id或零tf。")
                positions = np.arange(start, end, dtype=np.uint64)
                term_ids = np.searchsorted(offsets[1:], positions, side="right")
                if len(ids):
                    if int(term_ids[0]) == previous_term and int(ids[0]) <= previous_doc:
                        raise BinaryBM25StoreError("BM25 term倒排doc_id未严格递增。")
                    same_term = term_ids[1:] == term_ids[:-1]
                    if np.any((ids[1:] <= ids[:-1]) & same_term):
                        raise BinaryBM25StoreError("BM25 term倒排doc_id未严格递增。")
                    previous_doc = int(ids[-1])
                    previous_term = int(term_ids[-1])
                    np.add.at(totals, ids, freqs.astype(np.uint64, copy=False))
            if document_count and not np.array_equal(totals, doc_lengths.astype(np.uint64, copy=False)):
                raise BinaryBM25StoreError("BM25倒排tf合计与doc_lengths不一致。")
            token_total = sum(int(value) for value in doc_lengths)
            max_document_length = max((int(value) for value in doc_lengths), default=0)
            expected_tf_dtype = np.dtype(np.uint32 if max_document_length <= max_uint32
                                         else np.uint64)
            expected_average = token_total / document_count if document_count else 0.0
            if (token_total != manifest["total_document_tokens"]
                    or tf_dtype != expected_tf_dtype
                    or expected_average != average_length):
                raise BinaryBM25StoreError("BM25平均长度或总token数与文档长度不一致。")

            return cls(version_dir=version_dir, manifest=manifest, vocabulary=vocabulary,
                       offsets=offsets, postings=postings, doc_lengths=doc_lengths,
                       doc_freqs=doc_freqs, maps=maps)
        except BaseException:
            for array in maps:
                _close_memmap(array)
            raise
