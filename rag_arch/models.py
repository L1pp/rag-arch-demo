"""Small shared records; uploaded documents need no engineering clause schema."""
from dataclasses import dataclass, field


@dataclass
class TextUnit:
    text: str
    locator: dict[str, object]


@dataclass
class LoadedDocument:
    doc_id: str
    filename: str
    source_kind: str
    units: list[TextUnit]
    warnings: list[str]
    unread_pages: list[int]
    page_count: int | None = None
    # OCR text is a supplement, never proof that the original page is accurate.
    ocr_pages: list[dict] = field(default_factory=list)


def ocr_evidence_warnings(ocr_pages, *, source_kind: str = "", units=None) -> list[str]:
    """Describe OCR coverage without claiming that recognised text is verified."""
    rows = list(ocr_pages) if isinstance(ocr_pages, list) else []
    # An OCR unit is authoritative even when page-summary metadata is missing.
    # Derive only a cautionary page entry; never infer accuracy from its text.
    recorded = {row.get("page") for row in rows if isinstance(row, dict)
                and row.get("status") == "completed"
                and type(row.get("page")) is int and row["page"] > 0}
    ocr_without_page = False
    for unit in units or []:
        locator = unit.get("locator") if isinstance(unit, dict) else getattr(unit, "locator", None)
        if not isinstance(locator, dict) or locator.get("text_source") != "cloud_ocr":
            continue
        page = locator.get("page")
        if type(page) is int and page > 0:
            if page not in recorded:
                rows.append({"page": page, "status": "completed"})
                recorded.add(page)
        else:
            ocr_without_page = True
    if not rows and not ocr_without_page:
        return []
    pages_by_status = {"completed": set(), "failed": set(), "unknown": set(), "not_attempted": set()}
    for row in rows:
        if not isinstance(row, dict):
            continue
        page = row.get("page")
        if type(page) is not int or page < 1:
            continue
        status = row.get("status")
        key = status if isinstance(status, str) and status in pages_by_status else "unknown"
        pages_by_status[key].add(page)
    warnings = []
    if pages_by_status["completed"]:
        pages = "、".join(map(str, sorted(pages_by_status["completed"])))
        warnings.append(f"第 {pages} 页含 OCR 补充文字，识别结果未经原页逐字核实；数字、单位和表格关系需回看原页。")
    for status, label in (("failed", "失败"), ("unknown", "状态未确认"), ("not_attempted", "未尝试")):
        if pages_by_status[status]:
            pages = "、".join(map(str, sorted(pages_by_status[status])))
            warnings.append(f"第 {pages} 页 OCR {label}，未读内容不能直接认定为方案未提供。")
    if ocr_without_page:
        warnings.append("部分 OCR 补充文字缺少有效页码，内容尚需核实原页。")
    if not warnings:
        warnings.append("部分页面的 OCR 状态无法确认，请回看原页。")
    if source_kind in {"doc", "docx"}:
        warnings.append("Word 的 OCR 页码为系统排版生成的预览页，不是原 Word 原生页码。")
    return warnings


@dataclass
class Chunk:
    doc_id: str
    filename: str
    chunk_id: str
    text: str
    locator: dict[str, object]
    source_kind: str
    collection_id: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    hits: list[SearchHit]
    warnings: list[str]
    grounded: bool
    citation_check: dict | None = None
    audit_log: dict | None = None
    page_sources: list[dict] = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    native_tool_observations: list[dict] = field(default_factory=list)
    native_tool_usage: dict = field(default_factory=dict)
    catalog_matches: list[dict] = field(default_factory=list)
    answer_scope: str = "evidence"
    tool_calls: int = 0


@dataclass
class ConversationTurn:
    question: str
    answer: Answer
