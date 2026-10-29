"""Explicitly registered files share one cached PDF for page text and page images."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable
from uuid import uuid4
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pymupdf

from .visual_page_index import build_visual_page_index, paginate_visual_page_index
from .page_regions import (normalize_region, page_image_id, render_page_image,
                           validate_page_result_identity)


MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_EDGE = 2400
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PAGE_CACHE_BYTES = 24 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
SUPPORTED_SUFFIXES = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"}
_DOCX_RENDER_LOCK = threading.Lock()


@dataclass
class _Source:
    source_ref: str
    filename: str
    label: str
    registered_path: Path | None
    data: bytes | None
    stamp: tuple[int, int] | None
    page_count: int | None = None
    cached_input: Path | None = None
    rendered_pdf: Path | None = None
    validated_pdf_stamp: tuple[int, int] | None = None
    normalized_docx_path: Path | None = None
    normalized_docx_stamp: tuple[int, int] | None = None
    cached_normalized_docx: Path | None = None

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()


def _check_docx(source: Path) -> None:
    """Allow ordinary hyperlinks/internal fields, but never load active external content."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with ZipFile(source) as package:
            if "word/document.xml" not in package.namelist():
                raise ValueError("文件不是有效的 DOCX。")
            if sum(x.file_size for x in package.infolist()) > MAX_SOURCE_BYTES:
                raise ValueError("DOCX 解压内容过大，暂不支持分页。")
            for name in package.namelist():
                if "vbaproject" in name.lower():
                    raise ValueError("不支持含宏的文档，请使用不含宏的 DOCX。")
                if name.endswith(".rels"):
                    for relation in ElementTree.fromstring(package.read(name)):
                        if (relation.get("TargetMode", "").lower() == "external"
                                and not relation.get("Type", "").endswith("/hyperlink")):
                            raise ValueError("文档含外部加载链接，请先将链接内容嵌入 DOCX。")
                if name.startswith("word/") and name.endswith(".xml"):
                    xml = ElementTree.fromstring(package.read(name))
                    instructions = [x.text or "" for x in xml.iter(ns + "instrText")]
                    instructions += [x.get(ns + "instr", "") for x in xml.iter(ns + "fldSimple")]
                    if re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK)\b",
                                 " ".join(instructions), re.IGNORECASE):
                        raise ValueError("文档含外部加载字段，不能在后台自动排版。")
    except (BadZipFile, ElementTree.ParseError, KeyError) as exc:
        raise ValueError("DOCX 文件损坏或无法解析。") from exc


class DocumentPageService:
    """Only register() accepts local paths; model-facing methods accept source_ref only.

    docx_renderer(source_path, output_pdf) is injectable for tests/another local renderer.
    cache_dir, when supplied, must stay under data/runtime/document_pages.
    """

    def __init__(self, project_root: str | Path, cache_dir: str | Path | None = None,
                 *, docx_renderer: Callable[[Path, Path], None] | None = None,
                 conversion_timeout: float = 120, pdf_backend: Any | None = None,
                 office_converter: Any | None = None):
        self.project_root = Path(project_root).resolve()
        self.cache_root = (self.project_root / "data/runtime/document_pages").resolve()
        candidate = Path(cache_dir) if cache_dir is not None else self.cache_root
        self.cache_dir = (candidate if candidate.is_absolute()
                          else self.project_root / candidate).resolve()
        if not self.cache_dir.is_relative_to(self.cache_root):
            raise ValueError("分页缓存必须位于项目 data/runtime/document_pages 内。")
        if not math.isfinite(conversion_timeout) or conversion_timeout <= 0:
            raise ValueError("转换超时必须是正数。")
        self.conversion_timeout = float(conversion_timeout)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sources: dict[str, _Source] = {}
        self._lock = threading.RLock()
        self._docx_renderer = docx_renderer or self._render_office
        self._custom_docx_renderer = docx_renderer is not None
        self._office_converter = office_converter
        self.pdf_backend = pdf_backend
        self._page_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._page_cache_bytes = 0
        self._visual_index_cache: dict[tuple[str, object], dict] = {}

    def register(self, source_ref: str, filename: str, path: str | Path | None = None,
                 data: bytes | None = None, label: str | None = None,
                 normalized_docx_path: str | Path | None = None) -> dict:
        """Register only: paths use stat; uploaded bytes stay in memory until first read."""
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise ValueError("source_ref 不能为空。")
        if not isinstance(filename, str):
            raise ValueError("文件名必须是文字。")
        filename = Path(filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("分页仅支持 PDF、DOC、DOCX、PNG 和 JPEG。")
        normalized_path = None
        normalized_stamp = None
        if normalized_docx_path is not None:
            if suffix != ".doc":
                raise ValueError("只有 DOC 来源可指定派生 DOCX。")
            normalized_path = Path(normalized_docx_path)
            normalized_path = (normalized_path if normalized_path.is_absolute()
                               else self.project_root / normalized_path).resolve()
            if normalized_path.suffix.lower() != ".docx" or not normalized_path.is_file():
                raise ValueError("DOC 派生文字文档不存在或格式无效。")
            normalized_status = normalized_path.stat()
            normalized_stamp = (normalized_status.st_size, normalized_status.st_mtime_ns)
        approved = None
        stamp = None
        if data is None:
            if path is None:
                raise ValueError("必须提供已选定文件的 path 或 data。")
            approved = Path(path)
            approved = (approved if approved.is_absolute()
                        else self.project_root / approved).resolve()
            if not approved.is_file():
                raise ValueError("已登记的来源文件不存在。")
            status = approved.stat()
            stamp = (status.st_size, status.st_mtime_ns)
        else:
            if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
                raise ValueError("来源文件内容为空或不是字节数据。")
            data = bytes(data)
            if len(data) > MAX_SOURCE_BYTES:
                raise ValueError("上传内容超过分页服务的 128 MB 上限。")
            if suffix == ".doc" and not data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
                raise ValueError("文件不是有效的 DOC 二进制文档。")
        with self._lock:
            previous = self._sources.get(source_ref)
            if (previous is not None and previous.filename == filename
                    and previous.registered_path == approved
                    and previous.stamp == stamp and previous.data == data
                    and previous.normalized_docx_path == normalized_path
                    and previous.normalized_docx_stamp == normalized_stamp):
                new_label = str(label or filename)
                if previous.label != new_label:
                    previous.label = new_label
                    for key in list(self._page_cache):
                        if key[0] == source_ref:
                            self._drop_cached_page(key)
                    for key in list(self._visual_index_cache):
                        if key[0] == source_ref:
                            self._visual_index_cache.pop(key, None)
                return self._describe(previous)
            source = _Source(source_ref, filename, str(label or filename), approved, data, stamp,
                             1 if suffix in {".png", ".jpg", ".jpeg"} else None)
            source.normalized_docx_path = normalized_path
            source.normalized_docx_stamp = normalized_stamp
            self._sources[source_ref] = source
            for key in list(self._page_cache):
                if key[0] == source_ref:
                    self._drop_cached_page(key)
            for key in list(self._visual_index_cache):
                if key[0] == source_ref:
                    self._visual_index_cache.pop(key, None)
            return self._describe(source)

    def _source(self, source_ref: str) -> _Source:
        if not isinstance(source_ref, str) or source_ref not in self._sources:
            raise ValueError("来源未登记；请从已登记来源目录选择 source_ref。")
        return self._sources[source_ref]

    @staticmethod
    def _describe(source: _Source) -> dict:
        return {"source_ref": source.source_ref, "filename": source.filename,
                "label": source.label, "page_count": source.page_count,
                "format": source.suffix.lstrip("."),
                "page_numbering": ("system_rendered_page" if source.suffix in {".doc", ".docx"}
                                   else "physical_page_of_cached_pdf")}

    def sources(self) -> list[dict]:
        """List registrations without starting Word or enumerating the filesystem."""
        return [self._describe(source) for source in self._sources.values()]

    def describe(self, source_ref: str | None = None) -> dict:
        return ({"sources": self.sources()} if source_ref is None
                else self._describe(self._source(source_ref)))

    @staticmethod
    def _validate_pdf(path: Path) -> int:
        try:
            with pymupdf.open(path) as document:
                if not document.is_pdf:
                    raise ValueError("来源不是有效 PDF。")
                if document.needs_pass:
                    raise ValueError("加密 PDF 暂不支持分页。")
                if len(document) == 0:
                    raise ValueError("PDF 没有可读取页面。")
                return len(document)
        except (OSError, RuntimeError) as exc:
            raise ValueError("PDF 无法打开，请检查来源文件。") from exc

    @staticmethod
    def _assert_unchanged(source: _Source) -> None:
        if source.registered_path is not None:
            try:
                status = source.registered_path.stat()
            except OSError as exc:
                raise ValueError("已登记的来源文件已不可用，请重新登记。") from exc
            if (status.st_size, status.st_mtime_ns) != source.stamp:
                raise ValueError("来源文件已更新，请重新登记后读取页面。")
        if source.normalized_docx_path is not None:
            try:
                status = source.normalized_docx_path.stat()
            except OSError as exc:
                raise ValueError("DOC 派生文字文档已不可用，请重新登记。") from exc
            if (status.st_size, status.st_mtime_ns) != source.normalized_docx_stamp:
                raise ValueError("DOC 派生文字文档已更新，请重新登记后读取页面。")

    @staticmethod
    def _path_stamp(path: Path) -> tuple[int, int] | None:
        try:
            status = path.stat()
        except OSError:
            return None
        return status.st_size, status.st_mtime_ns

    def _ensure_page_count(self, source: _Source, pdf_path: Path) -> None:
        if self.pdf_backend is None:
            source.page_count = self._validate_pdf(pdf_path)
            return
        stamp = (source.stamp if source.registered_path is not None and source.suffix == ".pdf"
                 else self._path_stamp(pdf_path))
        if source.page_count is None or source.validated_pdf_stamp != stamp:
            source.page_count = self.pdf_backend.validate(pdf_path)
            source.validated_pdf_stamp = stamp

    def _snapshot(self, source: _Source) -> Path:
        if source.cached_input is None:
            folder = (self.cache_dir / uuid4().hex).resolve()
            if not folder.is_relative_to(self.cache_root):
                raise ValueError("分页缓存路径无效。")
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / ("source" + source.suffix)
            if source.data is not None:
                target.write_bytes(source.data)
            else:
                shutil.copyfile(source.registered_path, target)
                self._assert_unchanged(source)
            source.cached_input = target
        return source.cached_input

    def _office(self):
        if self._office_converter is None:
            from .office_conversion import OfficeConverter
            self._office_converter = OfficeConverter(
                self.project_root, timeout=self.conversion_timeout)
        return self._office_converter

    def _render_office(self, source: Path, output: Path) -> None:
        if not self._uses_libreoffice():
            self._render_with_word(source, output)
        else:
            self._office().render_pdf(source, output)

    @staticmethod
    def _uses_libreoffice() -> bool:
        return os.name != "nt"

    def _shared_office_pdf(self, source: _Source) -> Path | None:
        """Tenant uploads share a PDF across API, chat and plan page services."""
        if (self._custom_docx_renderer or not self._uses_libreoffice()
                or source.registered_path is None or source.suffix not in {".doc", ".docx"}):
            return None
        upload_root = (self.project_root / "data/runtime/web_uploads").resolve()
        if not source.registered_path.is_relative_to(upload_root):
            return None
        if source.suffix == ".doc":
            normalized = source.normalized_docx_path
            if normalized is None:
                normalized = source.registered_path.with_name("normalized.docx")
                self._office().ensure_docx(source.registered_path, normalized)
            render_input = normalized
        else:
            render_input = source.registered_path
        _check_docx(render_input)
        target = source.registered_path.with_name("pages.pdf")
        self._office().ensure_pdf(render_input, target)
        self._assert_unchanged(source)
        return target

    def _normalized_snapshot(self, source: _Source, original: Path) -> Path:
        if source.cached_normalized_docx is None:
            target = original.with_name("normalized.docx")
            if source.normalized_docx_path is None:
                self._office().ensure_docx(original, target)
            else:
                shutil.copyfile(source.normalized_docx_path, target)
                self._assert_unchanged(source)
            _check_docx(target)
            source.cached_normalized_docx = target
        return source.cached_normalized_docx

    def _pdf(self, source: _Source) -> Path:
        with self._lock:
            self._assert_unchanged(source)
            validated_page_count = None
            if source.rendered_pdf is None:
                shared_pdf = self._shared_office_pdf(source)
                if shared_pdf is not None:
                    source.rendered_pdf = shared_pdf
            if source.suffix == ".pdf":
                # Registered library PDFs are opened directly, never copied or hashed.
                pdf_path = source.registered_path or self._snapshot(source)
            elif source.rendered_pdf is not None:
                pdf_path = source.rendered_pdf
            else:
                input_path = self._snapshot(source)
                pdf_path = input_path.with_name("pages.pdf")
                temp = pdf_path.with_name("render-" + uuid4().hex + ".pdf")
                try:
                    if source.suffix in {".doc", ".docx"}:
                        with _DOCX_RENDER_LOCK:
                            render_input = (self._normalized_snapshot(source, input_path)
                                            if source.suffix == ".doc" else input_path)
                            _check_docx(render_input)
                            self._docx_renderer(render_input, temp)
                    elif self.pdf_backend is not None:
                        temp.write_bytes(self.pdf_backend.image_to_pdf(input_path, source.suffix))
                    else:
                        with pymupdf.open(input_path) as image:
                            if image.is_pdf or len(image) != 1:
                                raise ValueError("图片来源不是有效的单页图片。")
                            temp.write_bytes(image.convert_to_pdf())
                    if self.pdf_backend is None:
                        self._validate_pdf(temp)
                    else:
                        validated_page_count = self.pdf_backend.validate(temp)
                    temp.replace(pdf_path)
                    source.rendered_pdf = pdf_path
                except (OSError, RuntimeError) as exc:
                    from .resource_supervision import ResourceCleanupUncertain
                    if isinstance(exc, ResourceCleanupUncertain):
                        raise
                    raise ValueError("文件排版失败，未生成可用分页 PDF。") from exc
                finally:
                    temp.unlink(missing_ok=True)
            if self.pdf_backend is not None and validated_page_count is not None:
                source.page_count = validated_page_count
                source.validated_pdf_stamp = self._path_stamp(pdf_path)
            else:
                self._ensure_page_count(source, pdf_path)
            return pdf_path

    @staticmethod
    def _positive_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + "必须是从 1 开始的整数。")

    @staticmethod
    def _check_page_image(result: dict) -> None:
        image = result.get("image_bytes")
        if not isinstance(image, bytes) or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("页面图像读取失败，缓存内容无效。")

    @staticmethod
    def _source_token(source: _Source) -> object:
        return source.stamp if source.stamp is not None else id(source)

    def _visual_index(self, source_ref: str, source: _Source, pdf_path: Path) -> dict:
        """Build one metadata-only index for the current registered source."""
        cache_key = (source_ref, self._source_token(source))
        with self._lock:
            cached = self._visual_index_cache.get(cache_key)
            if cached is not None:
                return cached
            if self.pdf_backend is not None:
                scanner = getattr(self.pdf_backend, "scan_visual_pages", None)
                if scanner is None:
                    scanner = getattr(self.pdf_backend, "visual_page_index", None)
                if scanner is None:
                    raise ValueError("PDF worker 不支持视觉页目录扫描。")
                index = scanner(pdf_path)
            else:
                with pymupdf.open(pdf_path) as document:
                    index = build_visual_page_index(document)
            if not isinstance(index, dict):
                raise ValueError("视觉页目录扫描返回内容无效。")
            self._visual_index_cache[cache_key] = index
            return index

    def _list_visual_pages(self, source_ref: str, source: _Source, pdf_path: Path,
                           start: int, limit: int, query: str) -> dict:
        index = self._visual_index(source_ref, source, pdf_path)
        result = paginate_visual_page_index(index, start, limit, query)
        return {"ok": True, **self._describe(source), "pages": result["pages"],
                "start": start, "limit": limit, "query": query, "visual_only": True,
                "visual_candidate_count": result["visual_candidate_count"],
                "visual_index_complete": result["visual_index_complete"],
                "next_start": result["next_start"]}

    def ocr_candidates(self, source_ref: str) -> list[dict]:
        """Inspect the cached PDF for pages that may need supplementary OCR.

        DOC/DOCX page numbers refer to the server-rendered PDF layout. Results
        are heuristics, not proof that remaining image content was read.
        """
        source = self._source(source_ref)
        pdf_path = self._pdf(source)
        if self.pdf_backend is not None:
            scanner = getattr(self.pdf_backend, "ocr_candidates", None)
            if not callable(scanner):
                raise ValueError("PDF worker 不支持识别候选页检测。")
            rows = scanner(pdf_path)
        else:
            from .ocr_page_policy import build_ocr_page_candidates
            with pymupdf.open(pdf_path) as document:
                rows = build_ocr_page_candidates(document)
        if (not isinstance(rows, list) or len(rows) != source.page_count
                or any(not isinstance(row, dict)
                       or type(row.get("page")) is not int
                       or row["page"] != index
                       or type(row.get("needs_ocr")) is not bool
                       or not isinstance(row.get("reason"), str)
                       or type(row.get("native_chars")) is not int
                       or row["native_chars"] < 0
                       or type(row.get("body_chars")) is not int
                       or row["body_chars"] < 0
                       or type(row.get("blank")) is not bool
                       or type(row.get("image_coverage")) not in (int, float)
                       or not math.isfinite(row["image_coverage"])
                       or not 0 <= row["image_coverage"] <= 1
                       for index, row in enumerate(rows, 1))):
            raise ValueError("PDF worker 返回的候选页检测结果无效。")
        return rows

    def list_pages(self, source_ref: str, start: int = 1, limit: int = 20,
                   query: str = "", visual_only: bool = False) -> dict:
        self._positive_int(start, "起始页码")
        self._positive_int(limit, "结果数量")
        if limit > 50:
            raise ValueError("每次最多列出 50 页。")
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("页内定位词必须是长度不超过 200 的文字。")
        if not isinstance(visual_only, bool):
            raise ValueError("visual_only 必须是布尔值。")
        source = self._source(source_ref)
        pdf_path = self._pdf(source)
        if visual_only:
            return self._list_visual_pages(source_ref, source, pdf_path, start, limit, query)
        if self.pdf_backend is not None:
            result = self.pdf_backend.list_pages(pdf_path, start, limit, query)
            return {"ok": True, **self._describe(source), "pages": result["pages"],
                    "start": start, "limit": limit, "query": query,
                    "next_start": result["next_start"]}
        results = []
        with pymupdf.open(pdf_path) as document:
            if start > len(document):
                raise ValueError("起始页码超出来源页面范围。")
            last_page = start - 1
            for index in range(start - 1, len(document)):
                last_page = index + 1
                text = document[index].get_text().strip()
                position = text.casefold().find(query.casefold()) if query else 0
                if position < 0:
                    continue
                preview_start = max(0, position - 80)
                results.append({"page": index + 1, "has_text": bool(text),
                                "text_preview": text[preview_start:preview_start + 500]})
                if len(results) >= limit:
                    break
        return {"ok": True, **self._describe(source), "pages": results,
                "start": start, "limit": limit, "query": query,
                "next_start": last_page + 1 if last_page < source.page_count else None}

    @staticmethod
    def _cached_page_weight(result: dict) -> int:
        return len(result.get("image_bytes", b"")) + 4 * len(result.get("text", ""))

    def _drop_cached_page(self, key) -> None:
        removed = self._page_cache.pop(key, None)
        if removed is not None:
            self._page_cache_bytes -= self._cached_page_weight(removed)

    def _cache_page(self, key, result: dict) -> None:
        """Bound retained encoded pixels and extracted text, as well as entry count."""
        weight = self._cached_page_weight(result)
        with self._lock:
            self._drop_cached_page(key)
            if weight <= MAX_PAGE_CACHE_BYTES:
                self._page_cache[key] = deepcopy(result)
                self._page_cache_bytes += weight
            while len(self._page_cache) > 16 or self._page_cache_bytes > MAX_PAGE_CACHE_BYTES:
                self._drop_cached_page(next(iter(self._page_cache)))

    def read_page(self, source_ref: str, page: int, scale: float = 1.5, region=None) -> dict:
        self._positive_int(page, "物理页码")
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not 0.25 <= scale <= 4 or not math.isfinite(scale)):
            raise ValueError("页面缩放必须在 0.25 到 4 之间。")
        region = normalize_region(region)
        source = self._source(source_ref)
        pdf_path = self._pdf(source)
        # Path registrations use the existing stat stamp; uploaded sources use
        # the registration object's identity, so a concurrent re-registration
        # cannot make an older render look current.
        source_token = source.stamp if source.stamp is not None else id(source)
        cache_key = (source_ref, source_token, page, float(scale), region)
        with self._lock:
            cached = self._page_cache.get(cache_key)
            if cached is not None:
                self._page_cache.move_to_end(cache_key)
                validate_page_result_identity(cached, source_ref, page, region)
                self._check_page_image(cached)
                return deepcopy(cached)
        if self.pdf_backend is not None:
            # Keep old injected whole-page backends compatible; region support
            # is required only when a caller explicitly asks for a crop.
            kwargs = {"region": region} if region is not None else {}
            result = self.pdf_backend.read_page(pdf_path, page, float(scale), **kwargs)
            result = {"ok": True, **self._describe(source), **result}
        else:
            with pymupdf.open(pdf_path) as document:
                if page > len(document):
                    raise ValueError("物理页码超出来源页面范围。")
                result = {"ok": True, **self._describe(source), "page": page,
                          **render_page_image(document[page - 1], scale, region,
                              max_pixels=MAX_IMAGE_PIXELS, max_edge=MAX_IMAGE_EDGE,
                              max_bytes=MAX_IMAGE_BYTES)}
        validate_page_result_identity(result, source_ref, page, region)
        result["evidence_id"] = page_image_id(source_ref, page, region)
        self._check_page_image(result)
        self._cache_page(cache_key, result)
        return result

    @staticmethod
    def _resource_api():
        from .resource_supervision import ResourceCleanupUncertain, begin_resource, probe_process
        return ResourceCleanupUncertain, begin_resource, probe_process

    @staticmethod
    def _decode_process_output(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _merge_process_output(complete: Any, partial: Any) -> str:
        complete_text = DocumentPageService._decode_process_output(complete)
        partial_text = DocumentPageService._decode_process_output(partial)
        if not partial_text or partial_text in complete_text:
            return complete_text
        separator = "" if not complete_text or complete_text.endswith("\n") else "\n"
        return complete_text + separator + partial_text

    @staticmethod
    def _read_word_owner(owner: Path) -> dict:
        try:
            value = json.loads(owner.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Word 所有者文件缺失或无法解析。") from exc
        if (not isinstance(value, dict) or value.get("owned") is not True
                or isinstance(value.get("pid"), bool) or not isinstance(value.get("pid"), int)
                or value["pid"] < 1
                or not isinstance(value.get("start_ticks"), str)
                or not value["start_ticks"].isascii() or not value["start_ticks"].isdigit()
                or not isinstance(value.get("executable"), str)
                or not os.path.isabs(value["executable"])
                or Path(value["executable"]).name.casefold() != "winword.exe"):
            raise ValueError("Word 所有者文件中的 PID、StartTicks 或可执行文件身份无效。")
        try:
            unix_seconds = float(
                (Decimal(value["start_ticks"]) - Decimal(621355968000000000))
                / Decimal(10000000))
        except (InvalidOperation, OverflowError, ValueError) as exc:
            raise ValueError("Word 所有者文件中的 StartTicks 无效。") from exc
        if not math.isfinite(unix_seconds) or unix_seconds < 0:
            raise ValueError("Word 所有者文件中的 StartTicks 无效。")
        # Match task_worker.process_start_token and psutil's six-decimal epoch
        # representation, including its float rounding at half-microsecond ticks.
        value["process_start_token"] = f"{unix_seconds:.6f}"
        return value

    @staticmethod
    def _stop_powershell(process: subprocess.Popen, timeout: float = 5) -> tuple[str, str]:
        """Stop and reap only the exact child represented by this Popen handle."""
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                if process.poll() is None:
                    process.kill()
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=timeout)
        if process.poll() is None:
            raise TimeoutError("The owned PowerShell process is still running after termination.")
        return (DocumentPageService._decode_process_output(stdout),
                DocumentPageService._decode_process_output(stderr))

    def _render_with_word(self, source: Path, output: Path) -> None:
        executable = shutil.which("powershell.exe") or shutil.which("pwsh")
        script = self.project_root / "scripts/render_docx_pages.ps1"
        if not executable or not script.is_file():
            raise ValueError("未找到本地 Word 转换脚本或 PowerShell。")
        owner = output.with_suffix(".owner.json")
        if owner.exists():
            raise ValueError("本次 Word 所有者文件已存在，拒绝覆盖。")
        # Scope execution policy to this trusted converter process; Group Policy
        # still takes precedence and no user or machine policy is changed.
        common = [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                  "-ExecutionPolicy", "Bypass", "-File", str(script)]
        temp_dir = output.parent / ("word-temp-" + uuid4().hex)
        temp_dir.mkdir()
        child_env = {**os.environ, "TEMP": str(temp_dir), "TMP": str(temp_dir)}
        kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                  "text": True, "encoding": "utf-8", "errors": "replace", "env": child_env}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        ResourceCleanupUncertain, begin_resource, probe_process = self._resource_api()
        word_resource = begin_resource(kind="word", scope="task", owner_path=str(owner))
        main_resource = None
        powershell_resources = []
        main_process = None
        main_attached = False
        main_completed = False
        main_stdout = ""
        main_stderr = ""
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        cleanup_failures: list[BaseException] = []
        word_record = None
        word_attached = False
        cleanup_stdout = ""
        cleanup_stderr = ""
        cleanup_exit_code: int | None = None
        word_confirmed_exited = False

        def record_unknown(handle, code: str) -> None:
            try:
                handle.unknown(code)
            except BaseException as exc:
                cleanup_errors.append(exc)

        def record_cleanup_failure(exc: BaseException) -> None:
            if isinstance(exc, ResourceCleanupUncertain):
                cleanup_errors.append(exc)
            else:
                cleanup_failures.append(exc)

        def record_exited(handle, code: str) -> bool:
            try:
                handle.exited()
                return True
            except BaseException as exc:
                cleanup_errors.append(exc)
                record_unknown(handle, code)
                return False

        def attach_word_record() -> bool:
            nonlocal word_record, word_attached, word_confirmed_exited
            if word_record is not None:
                return word_attached
            if not owner.is_file():
                return False
            try:
                word_record = self._read_word_owner(owner)
                status = probe_process(
                    word_record["pid"], word_record["process_start_token"],
                    executable=word_record["executable"])
                if status == "exited":
                    word_resource.attach_exited_process(
                        word_record["pid"], word_record["process_start_token"],
                        executable=word_record["executable"])
                    word_confirmed_exited = True
                elif status == "running":
                    word_resource.attach_process(
                        word_record["pid"],
                        process_start_token=word_record["process_start_token"],
                        executable=word_record["executable"])
                    word_resource.used()
                else:
                    word_resource.unknown("word_owner_process_identity_unknown")
                word_attached = True
            except BaseException as exc:
                cleanup_errors.append(exc)
                record_unknown(word_resource, "word_owner_identity_unverified")
            return word_attached

        def run_cleanup_process() -> None:
            nonlocal cleanup_stdout, cleanup_stderr, cleanup_exit_code
            cleanup_resource = begin_resource(
                kind="powershell", scope="task", owner_path=str(owner))
            powershell_resources.append(cleanup_resource)
            process = None
            attached = False
            try:
                process = subprocess.Popen(common + ["-OwnerFile", str(owner), "-CleanupOnly"],
                                           **kwargs)
            except BaseException:
                cleanup_resource.exited()
                raise
            try:
                cleanup_resource.attach_process(process.pid, executable=str(executable))
                cleanup_resource.used()
                attached = True
                cleanup_stdout, cleanup_stderr = process.communicate(timeout=15)
                cleanup_exit_code = process.returncode
            except BaseException as exc:
                cleanup_failures.append(exc)
                try:
                    extra_out, extra_err = self._stop_powershell(process)
                    cleanup_stdout = cleanup_stdout or extra_out
                    cleanup_stderr = cleanup_stderr or extra_err
                except BaseException as stop_exc:
                    cleanup_failures.append(stop_exc)
                    record_unknown(cleanup_resource, "word_cleanup_powershell_exit_unverified")
                cleanup_exit_code = process.returncode
            finally:
                if attached and process.poll() is not None:
                    record_exited(cleanup_resource, "word_cleanup_powershell_exit_unverified")
                elif not attached and process.poll() is not None:
                    record_unknown(cleanup_resource, "word_cleanup_powershell_attach_failed")
                elif process.poll() is None:
                    record_unknown(cleanup_resource, "word_cleanup_powershell_exit_unverified")

        try:
            main_resource = begin_resource(
                kind="powershell", scope="task", owner_path=str(owner))
            powershell_resources.append(main_resource)
            try:
                main_process = subprocess.Popen(
                    common + ["-InputPath", str(source), "-OutputPath", str(output),
                              "-OwnerFile", str(owner)], **kwargs)
            except BaseException:
                main_resource.exited()
                raise
            try:
                main_resource.attach_process(main_process.pid, executable=str(executable))
                main_resource.used()
                main_attached = True
            except BaseException as exc:
                primary_error = exc
                record_unknown(main_resource, "word_renderer_powershell_attach_failed")
            if primary_error is None:
                deadline = time.monotonic() + self.conversion_timeout
                timeout_error = None
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timeout_error = subprocess.TimeoutExpired(
                            main_process.args, self.conversion_timeout,
                            output=main_stdout, stderr=main_stderr)
                        break
                    try:
                        from .request_control import check_request_control
                        check_request_control()
                        main_stdout, main_stderr = main_process.communicate(
                            timeout=min(0.2, remaining))
                        timeout_error = None
                        main_completed = True
                        if main_process.returncode:
                            primary_error = ValueError(
                                "DOCX 的 Word 后台排版失败；详细原因已保存在本次缓存日志。")
                        if main_attached:
                            record_exited(main_resource,
                                          "word_renderer_powershell_exit_unverified")
                        break
                    except subprocess.TimeoutExpired as exc:
                        timeout_error = exc
                        main_stdout = self._decode_process_output(exc.stdout)
                        main_stderr = self._decode_process_output(exc.stderr)
                        attach_word_record()
                        if time.monotonic() >= deadline:
                            break
                    except BaseException as exc:
                        primary_error = exc
                        attach_word_record()
                        break
                if timeout_error is not None and primary_error is None:
                    primary_error = ValueError(
                        f"Word 排版超时（{self.conversion_timeout:g}s）；"
                        "详细过程已保存在本次缓存日志。")
                    primary_error.__cause__ = timeout_error
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc

        # The script publishes the PID+StartTicks record immediately after
        # creating Word, before opening the DOCX. Give that atomic write a short,
        # bounded chance to complete after cancellation/timeout.
        if not owner.exists() and main_process is not None and main_process.poll() is None:
            deadline = time.monotonic() + 2.0
            while not owner.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
        attach_word_record()

        # On timeout/cancellation, clean the exact Word PID first. A missing or
        # malformed owner record is never treated as proof of exit.
        if word_record is not None and not word_confirmed_exited:
            try:
                run_cleanup_process()
            except BaseException as exc:
                record_cleanup_failure(exc)

        # If PowerShell was still running, stop only its Popen child handle after
        # the Word cleanup attempt, wait boundedly, and verify the same child.
        if main_process is not None and main_process.poll() is None:
            try:
                out, err = self._stop_powershell(main_process)
                main_stdout = self._merge_process_output(out, main_stdout)
                main_stderr = self._merge_process_output(err, main_stderr)
                main_completed = True
            except BaseException as exc:
                cleanup_failures.append(exc)
                record_unknown(main_resource, "word_renderer_powershell_exit_unverified")

        if main_resource is not None and main_attached and main_process is not None \
                and main_process.poll() is not None \
                and not getattr(main_resource, "_verified_exited", False):
                record_exited(main_resource, "word_renderer_powershell_exit_unverified")

        # If Word published its record while the converter was being stopped,
        # read and clean that now. If the process reached the creation stage but
        # no verifiable record exists, keep the resource unresolved.
        if word_record is None and owner.exists():
            attach_word_record()
            if word_record is not None:
                try:
                    run_cleanup_process()
                except BaseException as exc:
                    record_cleanup_failure(exc)

        if word_attached:
            if not word_confirmed_exited:
                word_confirmed_exited = record_exited(
                    word_resource, "owned_word_exit_unverified")
        elif main_process is None:
            word_confirmed_exited = record_exited(
                word_resource, "word_not_started_state_unverified")
        elif main_completed and "DOCX_PAGE_RENDER_STAGE:create_word" not in main_stdout:
            # The converter exited before its explicit pre-COM creation marker.
            word_confirmed_exited = record_exited(word_resource, "word_not_started_unverified")
        else:
            record_unknown(word_resource, "word_owner_file_missing_or_invalid")

        all_processes_confirmed = (
            word_confirmed_exited
            and all(getattr(handle, "_verified_exited", False)
                    for handle in powershell_resources))
        if owner.exists() and not all_processes_confirmed:
            # Keep the evidence file available for operator recovery whenever
            # any process identity or wait result remains uncertain.
            pass
        elif owner.exists():
            owner.unlink(missing_ok=True)
        if all_processes_confirmed:
            shutil.rmtree(temp_dir, ignore_errors=True)

        log_parts = [self._decode_process_output(main_stdout),
                     self._decode_process_output(main_stderr)]
        if cleanup_stdout or cleanup_stderr:
            log_parts.extend(["Word cleanup:", self._decode_process_output(cleanup_stdout),
                              self._decode_process_output(cleanup_stderr)])
        if isinstance(primary_error, ValueError) and primary_error.__cause__ is not None \
                and isinstance(primary_error.__cause__, subprocess.TimeoutExpired):
            log_parts.append(str(primary_error))
        log_error = None
        try:
            output.with_suffix(".log").write_text("\n".join(log_parts), encoding="utf-8")
        except OSError as exc:
            log_error = exc

        if cleanup_errors:
            uncertain = next((error for error in cleanup_errors
                              if isinstance(error, ResourceCleanupUncertain)), None)
            if uncertain is not None:
                raise uncertain from primary_error
            raise ResourceCleanupUncertain("word_cleanup_unconfirmed") from cleanup_errors[0]

        if (word_record is not None and cleanup_exit_code is not None
                and (cleanup_exit_code != 0
                     or "DOCX_PAGE_WORD_CLEANUP_VERIFIED" not in cleanup_stdout)):
            if primary_error is None:
                primary_error = ValueError(
                    "Word 清理脚本未返回可验证的退出确认；详细过程已保存在本次缓存日志。")
        if cleanup_failures and primary_error is None:
            primary_error = ValueError(
                "Word 清理子进程未能正常完成；详细过程已保存在本次缓存日志。")
            primary_error.__cause__ = cleanup_failures[0]
        if log_error is not None and primary_error is None:
            primary_error = ValueError("Word 转换日志无法写入本次缓存。")
            primary_error.__cause__ = log_error
        if primary_error is not None:
            raise primary_error
