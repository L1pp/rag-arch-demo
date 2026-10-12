"""Read uploaded text locally; never follow links, run OCR, or execute content."""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
import re
from uuid import uuid4
from zipfile import BadZipFile, ZipFile
import xml.etree.ElementTree as ET

from .models import LoadedDocument, TextUnit

MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def validate_upload_sizes(sizes) -> None:
    """A small local-use limit: both one file and one batch fit in 10 MiB."""
    total = 0
    for size in sizes:
        if size > MAX_UPLOAD_BYTES:
            raise ValueError("单个文件不能超过 10 MB，请拆分或使用较小的文件。")
        total += size
    if total > MAX_UPLOAD_BYTES:
        raise ValueError("本批附件合计不能超过 10 MB，请减少文件后再试。")


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_WATERMARKS = {
    "住房城乡建设部信息公开", "住房和城乡建设部信息公开",
    "浏览专用", "仅供浏览", "仅供学习参考", "仅供内部参考",
}


def _upload_name_kind(filename: str, data: bytes) -> tuple[str, str]:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("文件名不能为空。")
    if not isinstance(data, bytes) or not data:
        raise ValueError("文件为空，或上传内容不是字节数据。")
    validate_upload_sizes((len(data),))
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        raise ValueError("文件名不能为空。")
    kind = PurePosixPath(name).suffix.lower().lstrip(".")
    if kind not in {"pdf", "docx", "txt", "md"}:
        raise ValueError("目前支持 PDF、DOCX、TXT 和 MD 文字文件。")
    return name, kind


def load_document(filename: str, data: bytes, *, pdf_backend=None) -> LoadedDocument:
    """Load PDF, DOCX, TXT or MD bytes and keep only the basename, not a path.

    An image-only PDF is a valid upload with no readable units and unread pages;
    malformed/unsupported files raise ValueError. No file links are dereferenced.
    """
    name, kind = _upload_name_kind(filename, data)
    if kind == "pdf" and pdf_backend is not None:
        loaded = pdf_backend.extract_document(name, data)
        if not isinstance(loaded, LoadedDocument):
            raise ValueError("PDF worker 返回了无效文档。")
        return loaded
    doc = LoadedDocument(uuid4().hex, name, kind, [], [], [])
    if kind == "pdf":
        _load_pdf(doc, data)
    elif kind == "docx":
        _load_docx(doc, data)
    else:
        _load_text(doc, data)
    if not doc.units:
        doc.warnings.append("没有提取到可用于问答的文字；扫描件自动 OCR 尚未提供。")
    return doc


def _load_text(doc: LoadedDocument, data: bytes) -> None:
    binary_signatures = (b"%PDF-", b"PK\x03\x04", b"\x89PNG", b"\xff\xd8\xff", b"\xd0\xcf\x11\xe0")
    if data.startswith(binary_signatures):
        raise ValueError("文件内容是二进制文档，与 TXT/MD 扩展名不符。")
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("gb18030")
                doc.warnings.append("该文件按 GB18030 编码读取；文字若与原件不符，请另存为 UTF-8 后上传。")
    except UnicodeDecodeError as exc:
        raise ValueError("文字编码无法完整解码，请使用 UTF-8 或 GB18030 文本。") from exc
    if any((ord(c) < 32 and c not in "\t\n\r\f") or c == "\x7f" for c in text):
        raise ValueError("文本含异常控制字符，可能是二进制文件或内容损坏。")
    if "\ufffd" in text:
        raise ValueError("文本含解码替换字符，无法确认原字，请从原件重新导出。")
    if text.strip():
        doc.units.append(TextUnit(text, {"line_start": 1, "line_end": _last_line(text)}))


def _last_line(text: str) -> int:
    return max(1, text.count("\n") + (0 if text.endswith("\n") else 1))


def _image_coverage(page) -> float:
    """Union of image rectangles, clipped to the page (not a sum of overlays)."""
    bounds = page.rect
    rects = []
    for item in page.get_image_info():
        x0, y0, x1, y1 = item["bbox"]
        x0, y0 = max(x0, bounds.x0), max(y0, bounds.y0)
        x1, y1 = min(x1, bounds.x1), min(y1, bounds.y1)
        if x1 > x0 and y1 > y0:
            rects.append((x0, y0, x1, y1))
    if not rects or bounds.width * bounds.height <= 0:
        return 0.0
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        ranges = sorted((r[1], r[3]) for r in rects if r[0] < right and r[2] > left)
        merged_height = 0.0
        low = high = None
        for bottom, top in ranges:
            if low is None:
                low, high = bottom, top
            elif bottom <= high:
                high = max(high, top)
            else:
                merged_height += high - low
                low, high = bottom, top
        if low is not None:
            merged_height += high - low
        area += (right - left) * merged_height
    return min(1.0, area / (bounds.width * bounds.height))


def _without_watermark(text: str) -> tuple[str, bool]:
    lines = []
    removed = False
    for line in text.splitlines(keepends=True):
        compact = re.sub(r"\s+", "", line)
        remainder = compact
        for phrase in sorted(_WATERMARKS, key=len, reverse=True):
            remainder = remainder.replace(phrase, "")
        # PDF text sorting can concatenate several rotated watermark spans.
        # Remove only lines fully explained by known watermarks/page numbers.
        if remainder != compact and re.fullmatch(r"[\d-]*", remainder):
            removed = True
        else:
            lines.append(line)
    result = "".join(lines)
    # A watermark plus a printed page number is still not readable body text.
    if removed and re.fullmatch(r"[\s\d-]*", result):
        result = ""
    return result, removed


def _load_pdf(doc: LoadedDocument, data: bytes) -> None:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise ValueError("PDF 读取依赖 PyMuPDF 尚未安装。") from exc
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("无法打开 PDF，文件可能损坏或格式不符。") from exc
    with pdf:
        if pdf.needs_pass:
            raise ValueError("该 PDF 已加密，请上传可直接打开的副本。")
        doc.page_count = len(pdf)
        if not doc.page_count:
            raise ValueError("PDF 没有页面。")
        has_images = False
        removed_watermark = False
        sparse_image_pages = []
        for number, page in enumerate(pdf, 1):
            try:
                text, removed = _without_watermark(page.get_text("text", sort=True))
                coverage = _image_coverage(page)
            except Exception:
                doc.unread_pages.append(number)
                doc.warnings.append(f"第 {number} 页读取失败，未作为完整文字页使用。")
                continue
            has_images = has_images or coverage > 0
            removed_watermark = removed_watermark or removed
            count = len(re.sub(r"\s+", "", text))
            if not count:
                doc.unread_pages.append(number)
                continue
            if text.count("\ufffd") > max(1, count * 0.1):
                doc.unread_pages.append(number)
                doc.warnings.append(f"第 {number} 页文字层含较多无法解码字符，未用于问答。")
                continue
            # Image-dominant pages with little text may contain only a caption.
            # Short text on a page without a large image is deliberately retained.
            partial = coverage >= 0.5 and count < 80
            locator: dict[str, object] = {"page": number}
            if partial:
                doc.unread_pages.append(number)
                sparse_image_pages.append(number)
                locator["partial_page"] = True
            doc.units.append(TextUnit(text, locator))
        if removed_watermark:
            doc.warnings.append("已排除识别到的浏览水印；水印文字不计作正文。")
        if sparse_image_pages:
            doc.warnings.append("大图占页至少一半且可提取文字不足 80 字符的页面，按可能仅有标题/说明标为未完整读取；保留的少量文字仍可引用。")
        if has_images:
            doc.warnings.append("PDF 中的图片、扫描正文和图片表格没有自动识别；当前只使用已有文字层。")
        if doc.unread_pages:
            doc.warnings.append("以下物理页未完整读取：" + "、".join(map(str, doc.unread_pages)) + "。问答不能覆盖这些页面的图像正文。")


def _children(node):
    """Use one Word AlternateContent representation, never both fallbacks."""
    if node.tag == MC + "AlternateContent":
        choices = [c for c in node if c.tag == MC + "Choice"]
        fallback = next((c for c in node if c.tag == MC + "Fallback"), None)
        choice = next((c for c in choices if any(e.tag == W + "t" for e in c.iter())), None)
        selected = choice if choice is not None else fallback
        if selected is None and choices:
            selected = choices[0]
        return list(selected) if selected is not None else []
    return list(node)


def _visible_nodes(node):
    yield node
    for child in _children(node):
        yield from _visible_nodes(child)


def _paragraph_text(paragraph) -> str:
    result = []

    def visit(node):
        if node.tag == W + "t":
            result.append(node.text or "")
        elif node.tag == W + "tab":
            result.append("\t")
        elif node.tag in {W + "br", W + "cr"}:
            result.append("\n")
        else:
            nested_paragraph = node is not paragraph and node.tag == W + "p"
            if nested_paragraph and result and not result[-1].endswith("\n"):
                result.append("\n")
            for child in _children(node):
                visit(child)
            if nested_paragraph:
                result.append("\n")

    visit(paragraph)
    return "".join(result).strip()


def _blocks(container):
    for child in _children(container):
        if child.tag in {W + "p", W + "tbl"}:
            yield child
        else:
            yield from _blocks(child)


def _table_rows(table):
    for row in (c for c in _children(table) if c.tag == W + "tr"):
        values = []
        for cell in (c for c in _children(row) if c.tag == W + "tc"):
            parts = []
            for block in _blocks(cell):
                if block.tag == W + "p":
                    parts.append(_paragraph_text(block))
                else:
                    parts.extend(_table_rows(block))
            values.append("\n".join(parts))
        yield "\t".join(values)


def _load_docx(doc: LoadedDocument, data: bytes) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if sum(x.file_size for x in entries) > 256 * 1024 * 1024:
                raise ValueError("DOCX 解压内容过大，请使用较小的文字副本。")
            if "[Content_Types].xml" not in archive.namelist():
                raise ValueError("文件不是完整的 DOCX 文档。")
            entry = archive.getinfo("word/document.xml")
            if entry.file_size > 16 * 1024 * 1024:
                raise ValueError("DOCX 正文过大，请拆分后上传。")
            document = ET.fromstring(archive.read(entry))
            media = sum(x.filename.startswith("word/media/") and not x.is_dir() for x in entries)
            auxiliary = any(re.match(r"word/(?:header\d+|footer\d+|footnotes|endnotes|comments)\.xml$", x.filename) for x in entries)
    except (BadZipFile, KeyError, ET.ParseError, RuntimeError, OSError) as exc:
        raise ValueError("无法读取 DOCX 正文，文件可能损坏或格式不符。") from exc
    body = document.find(W + "body")
    if body is None:
        raise ValueError("DOCX 缺少正文节点。")
    visible = list(_visible_nodes(body))
    paragraphs = {id(n): i for i, n in enumerate((n for n in visible if n.tag == W + "p"), 1)}
    tables = {id(n): i for i, n in enumerate((n for n in visible if n.tag == W + "tbl"), 1)}
    for block in _blocks(body):
        if block.tag == W + "p":
            text = _paragraph_text(block)
            if text.strip():
                doc.units.append(TextUnit(text, {"paragraph": paragraphs[id(block)]}))
        else:
            for row_number, text in enumerate(_table_rows(block), 1):
                if text.strip():
                    doc.units.append(TextUnit(text, {"table": tables[id(block)], "row": row_number}))
    if media or any(n.tag in {W + "drawing", W + "pict"} for n in visible):
        doc.warnings.append("DOCX 含图片或绘图；仅提取已有文字与文本框文字，没有识别图片中的正文、流程图或表格。")
    if any(n.tag in {W + "vMerge", W + "gridSpan"} for n in visible) or any(len(list(n.iter(W + "tbl"))) > 1 for n in visible if n.tag == W + "tbl"):
        doc.warnings.append("DOCX 含合并单元格或嵌套表格，已保留可读文字；复杂行列对应关系未重建。")
    if any(n.tag == W + "txbxContent" for n in visible):
        doc.warnings.append("文本框文字归入所属段落；段落编号是正文 XML 中的顺序索引，包含文本框段落，不是页码。")
    if any(n.tag == W + "altChunk" for n in visible):
        doc.warnings.append("DOCX 包含嵌入式外部正文块（altChunk），当前未解析该部分，请另存为普通 DOCX 或 TXT。")
    if auxiliary:
        doc.warnings.append("本次读取正文，不包含独立页眉、页脚、脚注、尾注或批注部分。")
