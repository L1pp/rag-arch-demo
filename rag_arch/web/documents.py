"""Workspace-scoped uploaded documents and registered plan resources.

The Web layer keeps uploaded bytes in a workspace-owned runtime directory, while
the registry and the parsed/indexed objects stay in memory.  This module is
deliberately independent of FastAPI so routes and workers can share the same
resource semantics without taking a lock across model or PDF-worker calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, Iterable
from uuid import uuid4

from ..collection import (DIRECT_TEXT_MAX_TOKENS, UploadCollection, build_collection,
                          extracted_text_tokens)
from ..config import ROOT
from ..document_pages import SUPPORTED_SUFFIXES
from ..loaders import load_document, validate_upload_sizes
from ..page_sources import PageSources, plan_page_sources
from ..plan_documents import (
    PlanDocument,
    load_registered_document,
    load_uploaded_document,
    registered_plan_labels,
)


REGISTERED_PLAN_IDS = ("P01", "P02", "R01")


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
    """In-memory registry plus workspace-scoped persisted upload files."""

    def __init__(self, root: str | Path = ROOT, workspace_id: str = "default",
                 pdf_backend=None, *, direct_text_max_tokens: int = DIRECT_TEXT_MAX_TOKENS):
        if type(direct_text_max_tokens) is not int or direct_text_max_tokens < 0:
            raise ValueError("direct_text_max_tokens须为不小于0的整数。")
        self.root = Path(root).resolve()
        self.workspace_id = self._workspace_id(workspace_id)
        self.pdf_backend = pdf_backend
        self.direct_text_max_tokens = direct_text_max_tokens
        self.upload_root = (self.root / "data/runtime/web_uploads" /
                            self.workspace_id).resolve()
        if not self.upload_root.is_relative_to((self.root / "data/runtime/web_uploads").resolve()):
            raise ValueError("上传目录无效。")

        self._lock = RLock()
        self._assets: dict[str, UploadAsset] = {}
        self._loaded: dict[str, Any] = {}
        self._load_locks: dict[str, Lock] = {}
        self._chat_cache: dict[tuple[tuple[str, ...], bool], PreparedChat] = {}
        self._chat_locks: dict[tuple[tuple[str, ...], bool], Lock] = {}
        self._collection_ids: dict[tuple[str, ...], str] = {}
        self._asset_pages: dict[str, PageSources] = {}
        self._plan_cache: dict[tuple[str, str], PlanContext] = {}
        self._plan_pages: dict[tuple[str, str, str, int], PageSources] = {}

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

    def add(self, filename: str, data: bytes) -> dict[str, Any]:
        """Persist one upload and return metadata safe for public responses."""
        name = _filename(filename)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("上传内容必须是字节数据。")
        payload = bytes(data)
        validate_upload_sizes((len(payload),))
        if not payload:
            raise ValueError("文件为空，或上传内容不是字节数据。")
        if Path(name).suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            raise ValueError("目前支持 PDF、DOCX、TXT 和 MD 文字文件。")

        asset_id = uuid4().hex
        directory = (self.upload_root / asset_id).resolve()
        if not directory.is_relative_to(self.upload_root):
            raise ValueError("上传路径无效。")
        path = directory / ("source" + Path(name).suffix.lower())
        directory.mkdir(parents=True, exist_ok=False)
        try:
            path.write_bytes(payload)
        except Exception:
            # The unique directory contains no user data before this write;
            # leave cleanup to the caller/normal runtime cleanup on failure.
            raise
        record = UploadAsset(asset_id, name, len(payload), path)
        with self._lock:
            self._assets[asset_id] = record
        return self._public(record)

    def get(self, asset_id: str) -> UploadAsset:
        """Return an internal registration, including its approved path."""
        if not isinstance(asset_id, str) or not asset_id:
            raise KeyError(asset_id)
        with self._lock:
            record = self._assets.get(asset_id)
            if record is None:
                raise KeyError(asset_id)
            return record

    def list(self) -> list[dict[str, Any]]:
        """Return only public asset metadata in registration order."""
        with self._lock:
            return [self._public(record) for record in self._assets.values()]

    def _record_path(self, record: UploadAsset) -> Path:
        path = record.path.resolve()
        if not path.is_relative_to(self.upload_root) or not path.is_file():
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

    def _parse_upload(self, record: UploadAsset, data: bytes):
        return load_document(record.filename, data, pdf_backend=self.pdf_backend)

    def load(self, asset_id: str):
        """Parse one upload once and reuse its ``LoadedDocument`` instance."""
        record = self.get(asset_id)
        with self._lock:
            cached = self._loaded.get(asset_id)
            parse_lock = self._load_locks.setdefault(asset_id, Lock())
        if cached is not None:
            return cached
        with parse_lock:
            with self._lock:
                cached = self._loaded.get(asset_id)
            if cached is not None:
                return cached
            data, _ = self._load_bytes(record)
            loaded = self._parse_upload(record, data)
            with self._lock:
                # A registry entry is immutable; this also avoids replacing a
                # result that another worker stored while parsing outside the lock.
                cached = self._loaded.setdefault(asset_id, loaded)
            return cached

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
            service.add(f"upload:{record.asset_id}", record.filename,
                        source_kind="upload", path=record.path,
                        asset_id=record.asset_id)
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
        with self._lock:
            cached = self._chat_cache.get(key)
            build_lock = self._chat_locks.setdefault(key, Lock())
        if cached is not None:
            return cached

        with build_lock:
            with self._lock:
                cached = self._chat_cache.get(key)
            if cached is not None:
                return cached
            if should_embed and embeddings is None:
                embeddings = embeddings_factory()
                if embeddings is None:
                    raise ValueError(f"附件提取正文合计超过 {self.direct_text_max_tokens:,} 估算 Token，向量服务不可用；未改用关键词检索。")
            collection = self._build_collection(records, embeddings=embeddings if should_embed else None,
                                                loaded_documents=loaded_documents)
            # Switching retrieval mode changes the index, not the selected documents.
            # Preserve their history/page scope across keyword and vector variants.
            with self._lock:
                collection_id = self._collection_ids.setdefault(ids, uuid4().hex)
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
            with self._lock:
                return self._chat_cache.setdefault(key, prepared)

    def pages_for_asset(self, asset_id: str):
        """Return the cached single-upload page service, if it is pageable."""
        record = self.get(asset_id)
        if Path(record.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            return None
        with self._lock:
            cached = self._asset_pages.get(asset_id)
        if cached is not None:
            return cached
        service = PageSources(self.root, collection_id=f"upload:{asset_id}",
                              pdf_backend=self.pdf_backend)
        service.add(f"upload:{asset_id}", record.filename,
                    source_kind="upload", path=record.path, asset_id=asset_id)
        with self._lock:
            return self._asset_pages.setdefault(asset_id, service)

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
            with self._lock:
                cached = self._plan_cache.get(key)
            if cached is not None:
                return cached
            document = load_registered_document(plan_id, root=self.root)
            asset = self._registered_asset(document)
        else:
            if len(ids) != 1:
                raise ValueError("方案上传只支持一个附件。")
            asset_id = ids[0]
            key = ("upload", asset_id)
            with self._lock:
                cached = self._plan_cache.get(key)
            if cached is not None:
                return cached
            record = self.get(asset_id)
            asset = {"filename": record.filename, "path": str(record.path)}
            document = self._uploaded_plan(asset_id, record)

        context_id = f"plan:{document.document_id}:{document.source_revision}"
        result = PlanContext(document, asset, context_id)
        with self._lock:
            return self._plan_cache.setdefault(key, result)

    def pages_for_plan(self, document: PlanDocument, asset: dict[str, Any], corpus=None):
        path = str(asset.get("path") or "")
        key = (document.document_id, document.source_revision, path, id(corpus))
        with self._lock:
            cached = self._plan_pages.get(key)
        if cached is not None:
            return cached
        service = pages_for_plan(document, asset, corpus, root=self.root,
                                 pdf_backend=self.pdf_backend)
        with self._lock:
            return self._plan_pages.setdefault(key, service)

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
            self._loaded.clear()
            self._load_locks.clear()
            self._chat_cache.clear()
            self._chat_locks.clear()
            self._collection_ids.clear()
            self._asset_pages.clear()
            self._plan_cache.clear()
            self._plan_pages.clear()


__all__ = [
    "DocumentStore", "UploadAsset", "PreparedChat", "PlanContext",
    "REGISTERED_PLAN_IDS",
    "pages_for_plan",
]
