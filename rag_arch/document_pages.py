"""Explicitly registered files share one cached PDF for page text and page images."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any, Callable
from uuid import uuid4
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pymupdf

from .visual_page_index import build_visual_page_index, paginate_visual_page_index


MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_EDGE = 2400
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
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
                 conversion_timeout: float = 120, pdf_backend: Any | None = None):
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
        self._docx_renderer = docx_renderer or self._render_with_word
        self.pdf_backend = pdf_backend
        self._page_cache: OrderedDict[tuple[str, object, int, float], dict] = OrderedDict()
        self._visual_index_cache: dict[tuple[str, object], dict] = {}

    def register(self, source_ref: str, filename: str, path: str | Path | None = None,
                 data: bytes | None = None, label: str | None = None) -> dict:
        """Register only: paths use stat; uploaded bytes stay in memory until first read."""
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise ValueError("source_ref 不能为空。")
        if not isinstance(filename, str):
            raise ValueError("文件名必须是文字。")
        filename = Path(filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("分页仅支持 PDF、DOCX、PNG 和 JPEG。")
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
        with self._lock:
            previous = self._sources.get(source_ref)
            if (previous is not None and previous.filename == filename
                    and previous.registered_path == approved
                    and previous.stamp == stamp and previous.data == data):
                new_label = str(label or filename)
                if previous.label != new_label:
                    previous.label = new_label
                    for key in list(self._page_cache):
                        if key[0] == source_ref:
                            self._page_cache.pop(key, None)
                    for key in list(self._visual_index_cache):
                        if key[0] == source_ref:
                            self._visual_index_cache.pop(key, None)
                return self._describe(previous)
            source = _Source(source_ref, filename, str(label or filename), approved, data, stamp,
                             1 if suffix in {".png", ".jpg", ".jpeg"} else None)
            self._sources[source_ref] = source
            for key in list(self._page_cache):
                if key[0] == source_ref:
                    self._page_cache.pop(key, None)
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
                "page_numbering": "physical_page_of_cached_pdf"}

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

    def _pdf(self, source: _Source) -> Path:
        with self._lock:
            self._assert_unchanged(source)
            validated_page_count = None
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
                    if source.suffix == ".docx":
                        _check_docx(input_path)
                        with _DOCX_RENDER_LOCK:
                            self._docx_renderer(input_path, temp)
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

    def read_page(self, source_ref: str, page: int, scale: float = 1.5) -> dict:
        self._positive_int(page, "物理页码")
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or not 0.25 <= scale <= 4):
            raise ValueError("页面缩放必须在 0.25 到 4 之间。")
        source = self._source(source_ref)
        pdf_path = self._pdf(source)
        # Path registrations use the existing stat stamp; uploaded sources use
        # the registration object's identity, so a concurrent re-registration
        # cannot make an older render look current.
        source_token = source.stamp if source.stamp is not None else id(source)
        cache_key = (source_ref, source_token, page, float(scale))
        with self._lock:
            cached = self._page_cache.get(cache_key)
            if cached is not None:
                self._page_cache.move_to_end(cache_key)
                self._check_page_image(cached)
                return dict(cached)
        if self.pdf_backend is not None:
            result = self.pdf_backend.read_page(pdf_path, page, float(scale))
            result = {"ok": True, **self._describe(source), **result}
            self._check_page_image(result)
            with self._lock:
                self._page_cache[cache_key] = dict(result)
                self._page_cache.move_to_end(cache_key)
                while len(self._page_cache) > 16:
                    self._page_cache.popitem(last=False)
            return result
        with pymupdf.open(pdf_path) as document:
            if page > len(document):
                raise ValueError("物理页码超出来源页面范围。")
            original = document[page - 1]
            width, height = original.rect.width, original.rect.height
            render_scale = min(float(scale), (MAX_IMAGE_EDGE - 1) / max(width, height),
                               math.sqrt(MAX_IMAGE_PIXELS / (width * height)) * 0.99)
            for _ in range(3):
                pixmap = original.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale),
                                             colorspace=pymupdf.csRGB, alpha=False)
                png = pixmap.tobytes("png")
                if len(png) <= MAX_IMAGE_BYTES:
                    break
                render_scale *= 0.7
            else:
                raise ValueError("页面图像过大，请使用更小的缩放比例。")
            result = {"ok": True, **self._describe(source), "page": page,
                    "text": original.get_text().strip(), "image_bytes": png,
                    "mime_type": "image/png", "width": pixmap.width, "height": pixmap.height,
                    "rendered_scale": render_scale}
            self._check_page_image(result)
            with self._lock:
                self._page_cache[cache_key] = dict(result)
                self._page_cache.move_to_end(cache_key)
                while len(self._page_cache) > 16:
                    self._page_cache.popitem(last=False)
            return result

    def _render_with_word(self, source: Path, output: Path) -> None:
        executable = shutil.which("powershell.exe") or shutil.which("pwsh")
        script = self.project_root / "scripts/render_docx_pages.ps1"
        if not executable or not script.is_file():
            raise ValueError("未找到本地 Word 转换脚本或 PowerShell。")
        owner = output.with_suffix(".owner.json")
        # Scope execution policy to this trusted converter process; Group Policy
        # still takes precedence and no user or machine policy is changed.
        common = [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                  "-ExecutionPolicy", "Bypass", "-File", str(script)]
        temp_dir = output.parent / ("word-temp-" + uuid4().hex)
        temp_dir.mkdir()
        child_env = {**os.environ, "TEMP": str(temp_dir), "TMP": str(temp_dir)}
        kwargs = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace",
                  "env": child_env}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            result = subprocess.run(common + ["-InputPath", str(source), "-OutputPath", str(output),
                                              "-OwnerFile", str(owner)],
                                    timeout=self.conversion_timeout, **kwargs)
            output.with_suffix(".log").write_text(
                result.stdout + "\n" + result.stderr, encoding="utf-8")
            if result.returncode:
                raise ValueError("DOCX 的 Word 后台排版失败；详细原因已保存在本次缓存日志。")
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired may hold bytes even when subprocess.run uses text=True.
            captured = [part.decode("utf-8", errors="replace") if isinstance(part, bytes)
                        else (part or "") for part in (exc.stdout, exc.stderr)]
            output.with_suffix(".log").write_text(
                "\n".join(captured + [f"Word conversion timed out after {exc.timeout}s."]),
                encoding="utf-8")
            raise ValueError("Word 排版超时，已停止本次转换；未操作其他已打开的 Word。") from exc
        finally:
            # Clean up only the recorded PID and creation time, including normal failures.
            try:
                cleanup = subprocess.run(common + ["-OwnerFile", str(owner), "-CleanupOnly"],
                                         timeout=15, **kwargs)
                if cleanup.returncode == 0:
                    owner.unlink(missing_ok=True)
            except (OSError, subprocess.TimeoutExpired):
                pass
