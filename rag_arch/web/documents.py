"""Workspace-scoped uploaded documents and registered plan resources.

The Web layer keeps uploaded bytes in a workspace-owned runtime directory, while
the registry and the parsed/indexed objects stay in memory.  This module is
deliberately independent of FastAPI so routes and workers can share the same
resource semantics without taking a lock across model or PDF-worker calls.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from threading import RLock
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid4, uuid5

from ..collection import (DIRECT_TEXT_MAX_TOKENS, UploadCollection, build_collection,
                          extracted_text_tokens)
from ..config import ROOT
from ..document_pages import SUPPORTED_SUFFIXES
from ..loaders import _upload_name_kind, load_document, validate_upload_sizes
from ..page_sources import PageSources, plan_page_sources
from ..retrieval import LocalIndex
from ..plan_documents import (
    PlanDocument,
    load_registered_document,
    load_uploaded_document,
    registered_plan_labels,
)


REGISTERED_PLAN_IDS = ("P01", "P02", "R01")


def _document_cache_weight(document, original_bytes=0) -> int:
    """Conservative text/object estimate, including OCR expansion of small PDFs."""
    weight = 4096
    for unit in getattr(document, "units", ()):
        body = unit.get("text", "") if isinstance(unit, dict) else getattr(unit, "text", "")
        locator = unit.get("locator", {}) if isinstance(unit, dict) else getattr(unit, "locator", {})
        weight += len(body) * 4 + 4096
        weight += len(json.dumps(locator, ensure_ascii=False, default=str)) * 4
    weight += sum(len(str(note)) * 4 for note in getattr(document, "warnings", ()))
    weight += len(json.dumps(getattr(document, "ocr_pages", []), ensure_ascii=False, default=str)) * 4
    return max(1, original_bytes * 4, weight)


def _chat_cache_weight(prepared, original_bytes=0) -> int:
    collection = prepared.collection
    weight = sum(_document_cache_weight(doc) for doc in getattr(collection, "documents", ()))
    # Count retained retrieval chunks conservatively even if some strings share
    # storage with source units. The LRU estimate is not a process RSS promise.
    weight += sum(len(getattr(chunk, "text", "")) * 4 + 4096
                  for chunk in getattr(collection, "chunks", ()))
    return max(1, original_bytes * 8, weight)


@dataclass
class PreparedChat:
    """The cached material prepared for one chat attachment selection.

    ``assets`` and ``document_assets`` are convenience metadata for workers
    that need to build page links.  The collection and page service remain the
    primary contract.
    """

    collection: UploadCollection
    pages: PageSources
    assets: list[dict[str, Any]]
    asset_ids: tuple[str, ...]
    workspace_id: str
    document_assets: dict[str, dict[str, Any]]


@dataclass
class PlanContext:
    document: PlanDocument
    asset: dict[str, str]
    context_id: str


@dataclass(frozen=True)
class UploadAsset:
    """Internal asset metadata; ``path`` is intentionally not public API."""

    asset_id: str
    filename: str
    size_bytes: int
    path: Path


def _filename(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("文件名不能为空。")
    if "\x00" in value:
        raise ValueError("文件名含无效字符。")
    name = PurePosixPath(value.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("文件名不能为空。")
    return name


def _ids(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError("附件编号必须是列表。") from exc
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError("附件编号不能为空。")
    if len(set(result)) != len(result):
        raise ValueError("同一个附件不能重复选择。")
    return result


def pages_for_plan(document: PlanDocument, asset: dict[str, Any], corpus=None,
                   *, root: str | Path = ROOT, pdf_backend=None):
    """Bind a plan document to its original pages without reading them early."""
    return plan_page_sources(document, asset, corpus=corpus, root=Path(root),
                             pdf_backend=pdf_backend)


class DocumentStore:
    """Workspace-scoped uploads with optional durable tenant metadata.

    The legacy constructor keeps its in-memory registry for isolated callers.
    A tenant store must receive both ``user_id`` and ``database``; attachment
    metadata then comes from PostgreSQL and every query is owner/workspace
    scoped. Parsed and page/index objects share a bounded LRU, while in-flight
    builders use futures so identical concurrent work is still single-flight.
    """

    def __init__(self, root: str | Path = ROOT, workspace_id: str = "default",
                 pdf_backend=None, *, user_id: int | None = None, database=None,
                 direct_text_max_tokens: int = DIRECT_TEXT_MAX_TOKENS,
                 cache_max_entries: int = 64,
                 cache_max_bytes: int = 64 * 1024 * 1024,
                 ocr_client=None):
        if type(direct_text_max_tokens) is not int or direct_text_max_tokens < 0:
            raise ValueError("direct_text_max_tokens须为不小于0的整数。")
        if type(cache_max_entries) is not int or cache_max_entries < 1:
            raise ValueError("cache_max_entries须为正整数。")
        if type(cache_max_bytes) is not int or cache_max_bytes < 1:
            raise ValueError("cache_max_bytes须为正整数。")
        if (user_id is None) != (database is None):
            raise ValueError("user_id和database必须同时提供。")
        if user_id is not None and (isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1):
            raise ValueError("user_id须为正整数。")
        self.root = Path(root).resolve()
        self.workspace_id = self._workspace_id(workspace_id)
        self.user_id = user_id
        self.database = database
        self.pdf_backend = pdf_backend
        # Only the durable worker injects this capability. API/preview stores
        # keep None even when the server OCR feature flag is enabled.
        self.ocr_client = ocr_client
        self.direct_text_max_tokens = direct_text_max_tokens
        self.cache_max_entries = cache_max_entries
        self.cache_max_bytes = cache_max_bytes
        self._storage_root = (self.root / "data/runtime/web_uploads").resolve()
        if self.user_id is None:
            self.upload_root = (self._storage_root / self.workspace_id).resolve()
            expected_parts = (self.workspace_id,)
        else:
            self.upload_root = (self._storage_root / str(self.user_id) /
                                self.workspace_id).resolve()
            expected_parts = (str(self.user_id), self.workspace_id)
        try:
            actual_parts = self.upload_root.relative_to(self._storage_root).parts
        except ValueError:
            actual_parts = ()
        if actual_parts != expected_parts:
            raise ValueError("上传目录无效。")

        self._lock = RLock()
        self._assets: dict[str, UploadAsset] = {}
        self._cache: OrderedDict[tuple[str, Any], tuple[Any, int]] = OrderedDict()
        self._cache_bytes = 0
        self._inflight: dict[tuple[str, Any], Future] = {}
        self._async_inflight_tasks: dict[tuple[str, Any], asyncio.Task] = {}
        self._async_inflight_waiters: dict[tuple[str, Any], int] = {}
        self._closed = False

    @staticmethod
    def _workspace_id(value: str) -> str:
        value = str(value).strip() if value is not None else ""
        if (not value or value in {".", ".."} or
                any(separator in value for separator in ("/", "\\", ":"))):
            raise ValueError("workspace_id 无效。")
        return value

    @staticmethod
    def _public(record: UploadAsset) -> dict[str, Any]:
        return {"asset_id": record.asset_id, "filename": record.filename,
                "size_bytes": record.size_bytes}

    def _cache_get(self, namespace: str, key: Any):
        token = (namespace, key)
        with self._lock:
            entry = self._cache.get(token)
            if entry is None:
                return None
            self._cache.move_to_end(token)
            return entry[0]

    def _cache_put(self, namespace: str, key: Any, value: Any, weight_bytes: int):
        weight = max(1, int(weight_bytes))
        if weight > self.cache_max_bytes:
            return value
        token = (namespace, key)
        with self._lock:
            if self._closed:
                return value
            previous = self._cache.pop(token, None)
            if previous is not None:
                self._cache_bytes -= previous[1]
            self._cache[token] = (value, weight)
            self._cache_bytes += weight
            while (len(self._cache) > self.cache_max_entries or
                   self._cache_bytes > self.cache_max_bytes):
                _, (_, evicted_weight) = self._cache.popitem(last=False)
                self._cache_bytes -= evicted_weight
        return value

    def _memoized(self, namespace: str, key: Any, build, weight):
        cached = self._cache_get(namespace, key)
        if cached is not None:
            return cached
        token = (namespace, key)
        with self._lock:
            # The fast lookup above and the in-flight reservation are separate
            # critical sections. A builder can finish and remove its Future in
            # between them, so recheck the LRU while holding the same lock used
            # to reserve the next build. Cache insertion/eviction also uses this
            # lock, making the check-and-reserve decision atomic.
            entry = self._cache.get(token)
            if entry is not None:
                self._cache.move_to_end(token)
                return entry[0]
            future = self._inflight.get(token)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[token] = future
        if not owner:
            return future.result()
        try:
            value = build()
            self._cache_put(namespace, key, value, weight(value))
            future.set_result(value)
            return value
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                if self._inflight.get(token) is future:
                    self._inflight.pop(token, None)

    async def _amemoized(self, namespace: str, key: Any, build, weight):
        """Share async cache builds without letting a cancelled waiter cancel peers."""
        cached = self._cache_get(namespace, key)
        if cached is not None:
            return cached
        token = (namespace, key)
        with self._lock:
            entry = self._cache.get(token)
            if entry is not None:
                self._cache.move_to_end(token)
                return entry[0]
            future = self._inflight.get(token)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[token] = future
            self._async_inflight_waiters[token] = self._async_inflight_waiters.get(token, 0) + 1

        if owner:
            async def complete_build():
                try:
                    value = await build()
                    self._cache_put(namespace, key, value, weight(value))
                    if not future.done():
                        future.set_result(value)
                except BaseException as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    with self._lock:
                        self._async_inflight_tasks.pop(token, None)
                        if self._inflight.get(token) is future:
                            self._inflight.pop(token, None)

            try:
                task = asyncio.create_task(complete_build())
                with self._lock:
                    self._async_inflight_tasks[token] = task
            except BaseException:
                with self._lock:
                    if self._inflight.get(token) is future:
                        self._inflight.pop(token, None)
                raise

        # shield prevents a cancelled waiter from cancelling a shared build.
        wrapped = asyncio.wrap_future(future)
        def consume_unobserved(done):
            if done.cancelled():
                return
            try:
                done.exception()
            except BaseException:
                pass
        wrapped.add_done_callback(consume_unobserved)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            # An exception raised by the shared build is distinct from this
            # caller receiving a cancellation request; a new caller may retry.
            if current is not None and current.cancelling() == 0:
                return await self._amemoized(namespace, key, build, weight)
            with self._lock:
                remaining = self._async_inflight_waiters.get(token, 1) - 1
                if remaining > 0:
                    self._async_inflight_waiters[token] = remaining
                else:
                    self._async_inflight_waiters.pop(token, None)
                task = self._async_inflight_tasks.get(token)
                if (owner or remaining <= 0) and task is not None and not task.done():
                    task.cancel()
            if (owner or remaining <= 0) and task is not None:
                # A cancelled builder retains its task slot until the shared
                # task and any run_local physical operation have drained. Peers
                # retry the completed shared cancellation with their own closure.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                # A Task cancelled before its coroutine's first step cannot run
                # complete_build's finally block, so settle the shared Future here.
                if task.cancelled() and not future.done():
                    with self._lock:
                        if not future.done():
                            future.set_exception(asyncio.CancelledError())
                        if self._inflight.get(token) is future:
                            self._inflight.pop(token, None)
                        if self._async_inflight_tasks.get(token) is task:
                            self._async_inflight_tasks.pop(token, None)
            raise
        finally:
            # Cancelled callers released their borrower in the except block.
            current = asyncio.current_task()
            if current is None or current.cancelling() == 0:
                with self._lock:
                    remaining = self._async_inflight_waiters.get(token, 0) - 1
                    if remaining > 0:
                        self._async_inflight_waiters[token] = remaining
                    else:
                        self._async_inflight_waiters.pop(token, None)

    def _private_path(self, asset_id: str, filename: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", asset_id):
            raise ValueError("上传文件登记信息无效。")
        suffix = Path(filename).suffix.lower()
        try:
            path = (self.upload_root / asset_id / ("source" + suffix)).resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError("上传路径无效。") from exc
        try:
            if path.relative_to(self.upload_root).parts != (asset_id, "source" + suffix):
                raise ValueError("上传路径无效。")
        except ValueError as exc:
            raise ValueError("上传路径无效。") from exc
        return path

    def _write_original(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=False)
        temporary = path.parent / (".upload-" + uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_attachment_row(self, asset_id: str, filename: str, size_bytes: int,
                              storage_key: str, *, status: str) -> None:
        now = time.time()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO attachments(
                    attachment_id, user_id, workspace_id, filename, storage_key,
                    content_type, size_bytes, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (asset_id, self.user_id, self.workspace_id, filename, storage_key,
                 None, size_bytes, status, now, now),
            )

    def _update_attachment_status(self, asset_id: str, expected: str, status: str) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute(
                """
                UPDATE attachments SET status = %s, updated_at = %s
                WHERE attachment_id = %s AND user_id = %s AND workspace_id = %s
                  AND status = %s
                """,
                (status, time.time(), asset_id, self.user_id, self.workspace_id, expected),
            )
            return result.rowcount == 1

    def _record_from_row(self, row) -> UploadAsset:
        asset_id = row["attachment_id"]
        filename = _filename(row["filename"])
        suffix = Path(filename).suffix.lower()
        expected_key = f"{self.user_id}/{self.workspace_id}/{asset_id}/source{suffix}"
        if row["storage_key"] != expected_key:
            raise ValueError("上传文件登记信息无效。")
        path = self._private_path(asset_id, filename)
        record = UploadAsset(asset_id, filename, int(row["size_bytes"]), path)
        self._record_path(record)
        return record

    def add(self, filename: str, data: bytes) -> dict[str, Any]:
        """Persist one upload and return metadata safe for public responses."""
        name = _filename(filename)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("上传内容必须是字节数据。")
        payload = bytes(data)
        validate_upload_sizes((len(payload),))
        if not payload:
            raise ValueError("文件为空，或上传内容不是字节数据。")
        if Path(name).suffix.lower() not in {".pdf", ".doc", ".docx", ".txt", ".md"}:
            raise ValueError("目前支持 PDF、DOC、DOCX、TXT 和 MD 文字文件。")
        if Path(name).suffix.lower() == ".doc":
            # Reject a mislabeled file before persisting the original. The
            # converter also validates the actual Word document on opening.
            _upload_name_kind(name, payload)

        asset_id = uuid4().hex
        path = self._private_path(asset_id, name)
        if self.user_id is None:
            try:
                self._write_original(path, payload)
            except Exception:
                shutil.rmtree(path.parent, ignore_errors=True)
                raise
            record = UploadAsset(asset_id, name, len(payload), path)
            with self._lock:
                self._assets[asset_id] = record
            return self._public(record)

        storage_key = f"{self.user_id}/{self.workspace_id}/{asset_id}/source{Path(name).suffix.lower()}"
        pending_created = False
        directory_created = False
        ready_update_started = False
        try:
            self._write_attachment_row(asset_id, name, len(payload), storage_key,
                                       status="processing")
            pending_created = True
            # A processing row is not visible to readers. The source is fully
            # written and atomically renamed before the row becomes ready.
            path.parent.mkdir(parents=True, exist_ok=False)
            directory_created = True
            temporary = path.parent / (".upload-" + uuid4().hex + ".tmp")
            try:
                with temporary.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            ready_update_started = True
            if not self._update_attachment_status(asset_id, "processing", "ready"):
                raise RuntimeError("附件登记状态已变化，未能完成上传。")
        except Exception:
            # Before the ready transition, cleanup cannot invalidate a file
            # that another reader could observe. Once the ready transaction
            # starts, preserve the original because its commit may be unknown.
            if not ready_update_started and directory_created:
                shutil.rmtree(path.parent, ignore_errors=True)
            if pending_created and not ready_update_started:
                try:
                    self._update_attachment_status(asset_id, "processing", "failed")
                except Exception:
                    pass
            raise
        record = UploadAsset(asset_id, name, len(payload), path)
        return self._public(record)

    def get(self, asset_id: str) -> UploadAsset:
        """Return an internal registration, including its approved path."""
        if not isinstance(asset_id, str) or not asset_id:
            raise KeyError(asset_id)
        if self.database is not None:
            with self.database.transaction() as connection:
                row = connection.execute(
                    """
                    SELECT attachment_id, filename, storage_key, size_bytes
                    FROM attachments
                    WHERE attachment_id = %s AND user_id = %s AND workspace_id = %s
                      AND status = 'ready'
                    """,
                    (asset_id, self.user_id, self.workspace_id),
                ).fetchone()
            if row is None:
                raise KeyError(asset_id)
            return self._record_from_row(row)
        with self._lock:
            record = self._assets.get(asset_id)
            if record is None:
                raise KeyError(asset_id)
            return record

    def list(self) -> list[dict[str, Any]]:
        """Return only public asset metadata in registration order."""
        if self.database is not None:
            with self.database.transaction() as connection:
                rows = connection.execute(
                    """
                    SELECT attachment_id, filename, storage_key, size_bytes
                    FROM attachments
                    WHERE user_id = %s AND workspace_id = %s AND status = 'ready'
                    ORDER BY created_at, attachment_id
                    """,
                    (self.user_id, self.workspace_id),
                ).fetchall()
            result = []
            for row in rows:
                try:
                    result.append(self._public(self._record_from_row(row)))
                except (OSError, ValueError):
                    # A ready row with a missing or invalid source is not a
                    # usable upload and must not be advertised as downloadable.
                    continue
            return result
        with self._lock:
            return [self._public(record) for record in self._assets.values()]

    def _record_path(self, record: UploadAsset) -> Path:
        try:
            path = record.path.resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError("上传文件已不可用，请重新上传。") from exc
        if (path != record.path or not path.is_relative_to(self.upload_root) or
                not path.is_file()):
            raise ValueError("上传文件已不可用，请重新上传。")
        try:
            if path.stat().st_size != record.size_bytes:
                raise ValueError("上传文件已更新，请重新上传。")
        except OSError as exc:
            raise ValueError("上传文件已不可用，请重新上传。") from exc
        return path

    def _load_bytes(self, record: UploadAsset) -> tuple[bytes, Path]:
        path = self._record_path(record)
        try:
            return path.read_bytes(), path
        except OSError as exc:
            raise ValueError("上传文件已不可用，请重新上传。") from exc

    def _normalized_docx(self, record: UploadAsset) -> Path:
        """Share a legacy DOC's normalized file with its page service."""
        from ..office_conversion import OfficeConverter
        source = self._record_path(record)
        target = source.with_name("normalized.docx")
        OfficeConverter(self.root).ensure_docx(source, target)
        return target

    def _parse_upload(self, record: UploadAsset, data: bytes):
        if Path(record.filename).suffix.lower() == ".doc":
            normalized = self._normalized_docx(record)
            try:
                converted = normalized.read_bytes()
            except OSError as exc:
                raise ValueError("DOC 转换后的文件暂不可读取。") from exc
            return load_document(record.filename, data, pdf_backend=self.pdf_backend,
                                 normalized_docx=converted)
        return load_document(record.filename, data, pdf_backend=self.pdf_backend)

    def load(self, asset_id: str):
        """Parse one upload once and reuse its ``LoadedDocument`` instance."""
        record = self.get(asset_id)
        def build():
            data, _ = self._load_bytes(record)
            loaded = self._parse_upload(record, data)
            if self.ocr_client is not None and loaded.source_kind in {"pdf", "doc", "docx"}:
                from .ocr_preparation import OCRDocumentUnreadable, supplement_document
                pages = self.pages_for_asset(asset_id)
                if pages is not None:
                    loaded = supplement_document(
                        loaded, pages, f"upload:{asset_id}", self.ocr_client,
                        record.path.parent / "ocr")
                    if not loaded.units and loaded.ocr_pages:
                        raise OCRDocumentUnreadable(loaded.ocr_pages, source_kind=loaded.source_kind)
            return loaded

        return self._memoized("loaded", asset_id, build,
                              lambda value: _document_cache_weight(value, record.size_bytes))

    def _build_collection(self, records: list[UploadAsset], embeddings=None,
                          *, loaded_documents=None):
        files = [(record.filename, self._load_bytes(record)[0]) for record in records]
        if loaded_documents is None:
            loaded_documents = [self.load(record.asset_id) for record in records]
        return build_collection(files, embeddings=embeddings,
                               direct_text_max_tokens=self.direct_text_max_tokens,
                               documents=loaded_documents)

    def _chat_pages(self, records: list[UploadAsset], collection_id: str) -> PageSources:
        service = PageSources(self.root, collection_id=collection_id,
                              pdf_backend=self.pdf_backend)
        for record in records:
            if Path(record.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            normalized = (self._normalized_docx(record)
                          if Path(record.filename).suffix.lower() == ".doc" else None)
            service.add(f"upload:{record.asset_id}", record.filename,
                        source_kind="upload", path=record.path,
                        normalized_docx_path=normalized, asset_id=record.asset_id)
        return service

    def prepare_chat(self, attachment_ids: Iterable[str] | str,
                     *, embeddings=None, embeddings_factory=None) -> PreparedChat:
        ids = _ids(attachment_ids)
        if not ids:
            raise ValueError("请先选择附件。")
        if embeddings_factory is not None and not callable(embeddings_factory):
            raise TypeError("embeddings_factory 须为可调用对象。")
        # Attachment selection is a set for cache identity.  Building in the
        # same canonical order keeps collection/doc-to-asset mapping stable
        # when a client submits the same selection in another order.
        ids = tuple(sorted(ids))
        records = [self.get(asset_id) for asset_id in ids]
        validate_upload_sizes(record.size_bytes for record in records)
        loaded_documents = [self.load(record.asset_id) for record in records]
        text_tokens = extracted_text_tokens(loaded_documents)
        should_embed = (text_tokens > self.direct_text_max_tokens
                        and (embeddings is not None or callable(embeddings_factory)))
        key = (ids, should_embed)

        def build():
            if should_embed and embeddings is None:
                resolved_embeddings = embeddings_factory()
                if resolved_embeddings is None:
                    raise ValueError(f"附件提取正文合计超过 {self.direct_text_max_tokens:,} 估算 Token，向量服务不可用；未改用关键词检索。")
            else:
                resolved_embeddings = embeddings
            collection = self._build_collection(records, embeddings=resolved_embeddings if should_embed else None,
                                                loaded_documents=loaded_documents)
            # Switching retrieval mode changes the index, not the selected documents.
            # Preserve their history/page scope across keyword and vector variants.
            identity = "\0".join((str(self.root), str(self.user_id), self.workspace_id, *ids))
            collection_id = uuid5(NAMESPACE_URL, "rag-arch-upload-set:" + identity).hex
            collection.collection_id = collection_id
            for chunk in collection.chunks:
                chunk.collection_id = collection_id
            pages = self._chat_pages(records, collection.collection_id)
            public_assets = [self._public(record) for record in records]
            documents = list(getattr(collection, "documents", ()) or ())
            document_assets = {
                document.doc_id: public_assets[index]
                for index, document in enumerate(documents)
                if index < len(public_assets) and getattr(document, "doc_id", None)
            }
            prepared = PreparedChat(collection, pages, public_assets, ids,
                                    self.workspace_id, document_assets)
            return prepared

        return self._memoized("chat", key, build,
                              lambda value: _chat_cache_weight(value, sum(record.size_bytes for record in records)))

    async def aprepare_chat(self, attachment_ids: Iterable[str] | str,
                            *, embeddings=None, embeddings_factory=None) -> PreparedChat:
        """Prepare uploaded chat context while keeping embedding calls async."""
        from ..async_support import run_local
        ids = _ids(attachment_ids)
        if not ids:
            raise ValueError("请先选择附件。")
        if embeddings_factory is not None and not callable(embeddings_factory):
            raise TypeError("embeddings_factory 须为可调用对象。")
        ids = tuple(sorted(ids))
        records = await run_local(lambda: [self.get(asset_id) for asset_id in ids])
        await run_local(validate_upload_sizes, (record.size_bytes for record in records))
        loaded_documents = await run_local(
            lambda: [self.load(record.asset_id) for record in records])
        text_tokens = await run_local(extracted_text_tokens, loaded_documents)
        should_embed = (text_tokens > self.direct_text_max_tokens
                        and (embeddings is not None or callable(embeddings_factory)))
        key = (ids, should_embed)

        async def build():
            resolved_embeddings = embeddings
            if should_embed and resolved_embeddings is None:
                resolved_embeddings = await run_local(embeddings_factory)
                if resolved_embeddings is None:
                    raise ValueError(f"附件提取正文合计超过 {self.direct_text_max_tokens:,} 估算 Token，向量服务不可用；未改用关键词检索。")

            files = await run_local(lambda: [
                (record.filename, self._load_bytes(record)[0]) for record in records])
            collection = await run_local(
                build_collection, files, None,
                direct_text_max_tokens=self.direct_text_max_tokens,
                documents=loaded_documents)
            if should_embed:
                collection.index = await LocalIndex.acreate(collection.chunks, resolved_embeddings)
                collection.uses_embeddings = True

            identity = "\0".join((str(self.root), str(self.user_id), self.workspace_id, *ids))
            collection_id = uuid5(NAMESPACE_URL, "rag-arch-upload-set:" + identity).hex
            collection.collection_id = collection_id
            for chunk in collection.chunks:
                chunk.collection_id = collection_id
            pages = await run_local(self._chat_pages, records, collection.collection_id)
            public_assets = await run_local(lambda: [self._public(record) for record in records])
            documents = list(getattr(collection, "documents", ()) or ())
            document_assets = {
                document.doc_id: public_assets[index]
                for index, document in enumerate(documents)
                if index < len(public_assets) and getattr(document, "doc_id", None)
            }
            return PreparedChat(collection, pages, public_assets, ids,
                                self.workspace_id, document_assets)

        return await self._amemoized(
            "chat", key, build,
            lambda value: _chat_cache_weight(value, sum(record.size_bytes for record in records)))

    def pages_for_asset(self, asset_id: str):
        """Return the cached single-upload page service, if it is pageable."""
        record = self.get(asset_id)
        if Path(record.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            return None

        def build():
            service = PageSources(self.root, collection_id=f"upload:{asset_id}",
                                  pdf_backend=self.pdf_backend)
            normalized = (self._normalized_docx(record)
                          if Path(record.filename).suffix.lower() == ".doc" else None)
            service.add(f"upload:{asset_id}", record.filename,
                        source_kind="upload", path=record.path,
                        normalized_docx_path=normalized, asset_id=asset_id)
            return service

        return self._memoized("asset_pages", asset_id, build,
                              lambda _: max(1, record.size_bytes))

    def _uploaded_plan(self, asset_id: str, record: UploadAsset) -> PlanDocument:
        loaded = self.load(asset_id)
        data, _ = self._load_bytes(record)
        kwargs = {"document_id": f"UPLOAD-{asset_id}",
                  "source_revision": f"upload:{asset_id}",
                  "pdf_backend": self.pdf_backend}
        # The extended loader consumes this parsed object and therefore does
        # not parse the same bytes a second time.
        return load_uploaded_document(record.filename, data,
                                      loaded_document=loaded, **kwargs)

    def _registered_asset(self, document: PlanDocument) -> dict[str, str]:
        allowed_root = (self.root / "data/raw/plans").resolve()
        paths = []
        for unit in document.units:
            source = unit.get("source_path")
            if not isinstance(source, str) or not source or "://" in source:
                continue
            candidate = Path(source)
            candidate = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
            if candidate.is_relative_to(allowed_root) and candidate.name == document.filename:
                paths.append(candidate)
        if not paths:
            raise ValueError("登记方案原件不在允许的 data/raw/plans 目录内。")
        path = paths[0]
        if not path.is_file():
            raise ValueError("登记方案原件当前不可用。")
        return {"filename": document.filename, "path": str(path)}

    def get_plan(self, registered_plan_id: str | None = None,
                 attachment_ids: Iterable[str] | str | None = None) -> PlanContext:
        ids = _ids(attachment_ids)
        if registered_plan_id is not None:
            if ids:
                raise ValueError("登记方案和上传附件不能同时选择。")
            plan_id = str(registered_plan_id).strip()
            if plan_id not in REGISTERED_PLAN_IDS:
                raise KeyError("方案未登记。")
            key = ("registered", plan_id)

            def build_registered():
                document = load_registered_document(plan_id, root=self.root)
                asset = self._registered_asset(document)
                context_id = f"plan:{document.document_id}:{document.source_revision}"
                return PlanContext(document, asset, context_id)

            return self._memoized("plan", key, build_registered,
                                  lambda value: _document_cache_weight(value.document))
        else:
            if len(ids) != 1:
                raise ValueError("方案上传只支持一个附件。")
            asset_id = ids[0]
            key = ("upload", asset_id)
            record = self.get(asset_id)

            def build_uploaded():
                asset = {"filename": record.filename, "path": str(record.path)}
                document = self._uploaded_plan(asset_id, record)
                if Path(record.filename).suffix.lower() == ".doc":
                    asset["normalized_docx_path"] = str(self._normalized_docx(record))
                context_id = f"plan:{document.document_id}:{document.source_revision}"
                return PlanContext(document, asset, context_id)

            return self._memoized("plan", key, build_uploaded,
                                  lambda value: _document_cache_weight(value.document, record.size_bytes))

    def pages_for_plan(self, document: PlanDocument, asset: dict[str, Any], corpus=None):
        path = str(asset.get("path") or "")
        key = (document.document_id, document.source_revision, path, id(corpus))

        def build():
            return pages_for_plan(document, asset, corpus, root=self.root,
                                  pdf_backend=self.pdf_backend)

        try:
            weight = max(1, Path(path).stat().st_size)
        except OSError:
            weight = 1
        return self._memoized("plan_pages", key, build, lambda _: weight)

    def registered_plans(self) -> list[dict[str, str]]:
        """List the public registered-plan allowlist without parsing originals."""
        directory = self.root / "data/processed/plans_v0"
        info = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
        rows = {row.get("plan_id"): row for row in info.get("plans", [])
                if isinstance(row, dict)}
        labels = registered_plan_labels(self.root)
        result = []
        for plan_id in REGISTERED_PLAN_IDS:
            row = rows.get(plan_id)
            if not row or not isinstance(row.get("filename"), str):
                continue
            result.append({"id": plan_id, "plan_id": plan_id,
                           "label": labels.get(plan_id, plan_id),
                           "filename": row["filename"]})
        return result

    def close(self) -> None:
        """Release this store's parsed/index/page references only.

        Uploaded originals belong to the workspace runtime and are intentionally
        left in place.  A shared PDF backend is owned by its caller.
        """
        with self._lock:
            self._closed = True
            self._cache.clear()
            self._cache_bytes = 0


__all__ = [
    "DocumentStore", "UploadAsset", "PreparedChat", "PlanContext",
    "REGISTERED_PLAN_IDS",
    "pages_for_plan",
]
