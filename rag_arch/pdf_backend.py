"""Run PyMuPDF work in one long-lived process for concurrent web requests."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import threading

import pymupdf

from .models import LoadedDocument
from .visual_page_index import build_visual_page_index


MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_EDGE = 2400
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _open_pdf(source):
    if isinstance(source, str):
        return pymupdf.open(source)
    return pymupdf.open(stream=source, filetype="pdf")


def _check_pdf(document) -> int:
    if not document.is_pdf:
        raise ValueError("来源不是有效 PDF。")
    if document.needs_pass:
        raise ValueError("加密 PDF 暂不支持分页。")
    if len(document) == 0:
        raise ValueError("PDF 没有可读取页面。")
    return len(document)


def _validate(source) -> int:
    try:
        with _open_pdf(source) as document:
            return _check_pdf(document)
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _image_to_pdf(source, suffix: str) -> bytes:
    try:
        image = (pymupdf.open(source) if isinstance(source, str)
                 else pymupdf.open(stream=source, filetype=suffix))
        with image:
            if image.is_pdf or len(image) != 1:
                raise ValueError("图片来源不是有效的单页图片。")
            return image.convert_to_pdf()
    except (OSError, RuntimeError) as exc:
        raise ValueError("文件排版失败，未生成可用分页 PDF。") from exc


def _list_pages(source, start: int, limit: int, query: str) -> dict:
    try:
        with _open_pdf(source) as document:
            page_count = _check_pdf(document)
            if start > page_count:
                raise ValueError("起始页码超出来源页面范围。")
            results = []
            last_page = start - 1
            for index in range(start - 1, page_count):
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
            return {"page_count": page_count, "pages": results,
                    "next_start": last_page + 1 if last_page < page_count else None}
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _scan_visual_pages(source) -> dict:
    """Scan one PDF's page geometry/text metadata without rasterising it."""
    try:
        with _open_pdf(source) as document:
            _check_pdf(document)
            return build_visual_page_index(document)
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _read_page(source, page: int, scale: float) -> dict:
    try:
        with _open_pdf(source) as document:
            page_count = _check_pdf(document)
            if page > page_count:
                raise ValueError("物理页码超出来源页面范围。")
            original = document[page - 1]
            width, height = original.rect.width, original.rect.height
            render_scale = min(float(scale), (MAX_IMAGE_EDGE - 1) / max(width, height),
                               (MAX_IMAGE_PIXELS / (width * height)) ** 0.5 * 0.99)
            for _ in range(3):
                pixmap = original.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale),
                                             colorspace=pymupdf.csRGB, alpha=False)
                png = pixmap.tobytes("png")
                if len(png) <= MAX_IMAGE_BYTES:
                    break
                render_scale *= 0.7
            else:
                raise ValueError("页面图像过大，请使用更小的缩放比例。")
            return {"page_count": page_count, "page": page,
                    "text": original.get_text().strip(), "image_bytes": png,
                    "mime_type": "image/png", "width": pixmap.width, "height": pixmap.height,
                    "rendered_scale": render_scale}
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _worker_call(operation: str, payload: dict):
    if operation == "validate":
        result = _validate(payload["source"])
    elif operation == "extract_document":
        from .loaders import load_document
        result = load_document(payload["filename"], payload["data"], pdf_backend=None)
    elif operation == "image_to_pdf":
        result = _image_to_pdf(payload["source"], payload["suffix"])
    elif operation == "list_pages":
        result = _list_pages(payload["source"], payload["start"], payload["limit"], payload["query"])
    elif operation in {"scan_visual_pages", "visual_page_index"}:
        result = _scan_visual_pages(payload["source"])
    elif operation == "read_page":
        result = _read_page(payload["source"], payload["page"], payload["scale"])
    else:
        raise ValueError("未知 PDF worker 操作。")
    return os.getpid(), result


class ProcessPdfBackend:
    """A small, thread-safe client for one spawned PyMuPDF worker process."""

    def __init__(self):
        context = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
        self._lock = threading.RLock()
        self._closed = False
        self._worker_pids: set[int] = set()

    def _bytes(self, value, name: str) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(name + "必须是字节数据。")
        return bytes(value)

    def _source(self, value, name: str):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._bytes(value, name)
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
        raise TypeError(name + "必须是字节数据或本地路径。")

    def _call(self, operation: str, payload: dict):
        with self._lock:
            if self._closed:
                raise RuntimeError("PDF worker 已关闭。")
            future = self._executor.submit(_worker_call, operation, payload)
        pid, result = future.result()
        with self._lock:
            self._worker_pids.add(pid)
        return result

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Worker identities observed by this client; useful for diagnostics/tests."""
        with self._lock:
            return tuple(sorted(self._worker_pids))

    def validate(self, source) -> int:
        return self._call("validate", {"source": self._source(source, "PDF")})

    def extract_document(self, filename: str, data) -> LoadedDocument:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("文件名不能为空。")
        return self._call("extract_document", {
            "filename": filename, "data": self._bytes(data, "PDF")})

    def image_to_pdf(self, source, suffix: str) -> bytes:
        if not isinstance(suffix, str) or not suffix:
            raise ValueError("图片格式不能为空。")
        return self._call("image_to_pdf", {
            "source": self._source(source, "图片"), "suffix": suffix.lstrip(".").lower()})

    def list_pages(self, source, start: int = 1, limit: int = 20, query: str = "") -> dict:
        return self._call("list_pages", {
            "source": self._source(source, "PDF"), "start": start,
            "limit": limit, "query": query})

    def scan_visual_pages(self, source) -> dict:
        """Return a complete metadata-only visual candidate index for one PDF."""
        return self._call("scan_visual_pages", {"source": self._source(source, "PDF")})

    def visual_page_index(self, source) -> dict:
        """Compatibility spelling for callers that treat the result as an index."""
        return self.scan_visual_pages(source)

    def read_page(self, source, page: int, scale: float = 1.5) -> dict:
        return self._call("read_page", {
            "source": self._source(source, "PDF"), "page": page, "scale": scale})

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:  # pragma: no cover - compatibility with older Python
            executor.shutdown(wait=True)


# Retain the shorter name for local callers that do not need to spell out the process boundary.
PdfBackend = ProcessPdfBackend
