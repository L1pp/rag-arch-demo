"""Supplement uploaded pages only inside a claimed durable worker task."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path

from ..models import LoadedDocument, TextUnit, ocr_evidence_warnings
from ..request_control import check_request_control


class OCRPreparationDeferred(RuntimeError):
    pass


class OCRDocumentUnreadable(ValueError):
    def __init__(self, ocr_pages, *, source_kind=""):
        self.notes = ocr_evidence_warnings(ocr_pages, source_kind=source_kind)
        super().__init__("OCR 未取得可用正文。")

    def as_error(self):
        return {"code": "ocr_reading_unavailable",
                "message": "OCR 未取得可用于问答或审查的正文，请查看原页或上传更清晰的副本。"
                           + " ".join(self.notes)}


@dataclass
class _TaskOCR:
    guard: object
    maximum: int | None
    attempted: int = 0
    blocked: str = ""


_task_ocr = ContextVar("rag_task_ocr", default=None)
MAX_DOCUMENT_OCR_CHARACTERS = 1_000_000


def _page_limit(raw):
    try:
        maximum = int(raw)
    except (ValueError, TypeError):
        raise ValueError("OCR 每任务页数须为1至500的整数。") from None
    if isinstance(raw, bool) or str(maximum) != str(raw).strip() or not 1 <= maximum <= 500:
        raise ValueError("OCR 每任务页数须为1至500的整数。")
    return maximum


def _ensure_page_limit(state):
    if state.maximum is None:
        state.maximum = _page_limit(os.environ.get("RAG_ARCH_OCR_MAX_PAGES_PER_TASK", "100"))
    return state.maximum


@contextmanager
def ocr_preparation_scope(guard, *, max_pages=None):
    if not callable(guard):
        raise TypeError("OCR task guard must be callable")
    # Explicit limits are checked immediately. The optional environment value
    # is read only if a task actually needs to handle an OCR candidate.
    maximum = None if max_pages is None else _page_limit(max_pages)
    token = _task_ocr.set(_TaskOCR(guard, maximum))
    try:
        yield
    finally:
        _task_ocr.reset(token)


def _check_budget(state, text_characters=0):
    if text_characters >= MAX_DOCUMENT_OCR_CHARACTERS:
        raise OCRPreparationDeferred("document_text_budget")
    if state.blocked:
        raise OCRPreparationDeferred(state.blocked)
    if state.attempted >= _ensure_page_limit(state):
        raise OCRPreparationDeferred("task_page_budget")


def _before_request(state, page, text_characters=0):
    check_request_control()
    # This callback is the durable worker's fresh execution_control check,
    # not merely a feature flag or an unscoped no-op request guard.
    state.guard(page)
    _check_budget(state, text_characters)
    state.attempted += 1


def supplement_document(loaded: LoadedDocument, pages, source_ref: str, client,
                        cache_dir: Path) -> LoadedDocument:
    """Keep native units unchanged; append explicitly unverified OCR units.

    Appending preserves the existing native unit ordinals and citations. PDF
    physical page metadata makes the source order explicit; Word paragraphs
    retain their XML locators and never acquire guessed page numbers.
    """
    state = _task_ocr.get()
    if state is None:
        raise RuntimeError("在线OCR只能由已领取的后台任务准备。")
    check_request_control()
    try:
        candidates = pages.ocr_candidates(source_ref)
    except (ValueError, OSError):
        loaded.warnings.append("OCR 页图检查未完成，保留已有文字；图片内容请查看原页。")
        return loaded
    loaded.ocr_pages = []
    if candidates:
        loaded.page_count = len(candidates)
    unread = set(loaded.unread_pages)
    supplements = []
    text_characters = 0
    text_budget_exhausted = False
    for candidate in candidates:
        check_request_control()
        page = candidate.get("page")
        if type(page) is not int or page < 1:
            raise ValueError("OCR 候选页信息无效。")
        reason = candidate.get("reason", "")
        if candidate.get("blank") is True:
            unread.discard(page)
            continue
        if candidate.get("needs_ocr") is not True:
            if reason in {"page_analysis_failed", "minor_drawing_without_text", "small_image_without_text"}:
                unread.add(page)
                loaded.ocr_pages.append({"page": page, "status": "not_attempted",
                                         "reason": reason, "recognition_verified": False})
            continue
        # Invalid configured budgets fail explicitly when OCR is actually used,
        # rather than being converted into an ordinary page failure.
        _ensure_page_limit(state)
        try:
            peek = getattr(client, "peek_cached_page", None)
            result = peek(page=page, cache_dir=cache_dir) if callable(peek) else None
            if result is None:
                current_text = MAX_DOCUMENT_OCR_CHARACTERS if text_budget_exhausted else text_characters
                _check_budget(state, current_text)
                # Same bounded original-page renderer used for previews. Retry
                # rasterization only, never the external OCR request.
                png = None
                for scale in (200 / 72, 2.0, 1.5):
                    rendered = pages.read_page(source_ref, page, scale=scale)
                    png = rendered.get("image_bytes")
                    if isinstance(png, bytes) and 0 < len(png) <= 6 * 1024 * 1024:
                        break
                if not isinstance(png, bytes) or not 0 < len(png) <= 6 * 1024 * 1024:
                    raise ValueError("page image too large")
                result = client.recognize_page(
                    png, page=page, cache_dir=cache_dir,
                    before_request=lambda page=page: _before_request(state, page, current_text))
        except OCRPreparationDeferred as exc:
            unread.add(page)
            loaded.ocr_pages.append({"page": page, "status": "not_attempted",
                                     "reason": str(exc), "recognition_verified": False})
            continue
        except (ValueError, OSError):
            unread.add(page)
            loaded.ocr_pages.append({"page": page, "status": "failed",
                                     "reason": "page_preparation_failed", "recognition_verified": False})
            continue
        row = result.to_dict()
        row.pop("text", None)
        row["reason"] = reason
        if loaded.source_kind in {"doc", "docx"}:
            row["page_numbering"] = "system_rendered_page"
        loaded.ocr_pages.append(row)
        if result.status == "completed" and result.text.strip():
            if text_characters + len(result.text) > MAX_DOCUMENT_OCR_CHARACTERS:
                # The recognizer returned, so retain usage and do not label the
                # paid attempt as if no request had happened.
                row.update(status="failed", reason="document_text_budget", request_completed=True)
                text_budget_exhausted = True
                unread.add(page)
                continue
            locator = {"page": page, "text_source": "cloud_ocr", "ocr_model": result.model,
                       "ocr_status": "completed", "recognition_verified": False,
                       "partial_page": True}
            if loaded.source_kind in {"doc", "docx"}:
                locator["page_numbering"] = "system_rendered_page"
            supplements.append(TextUnit(result.text, locator))
            text_characters += len(result.text)
            unread.discard(page)
        else:
            unread.add(page)
            code = result.error_code or ""
            if (result.status == "unknown" or "Quota" in code or "Auth" in code
                    or "Access" in code or code in {"http_401", "http_403", "http_429"}):
                state.blocked = "previous_request_unconfirmed_or_service_denied"
    loaded.units.extend(supplements)
    loaded.unread_pages = sorted(unread)
    if loaded.ocr_pages:
        if loaded.source_kind in {"doc", "docx"}:
            for row in loaded.ocr_pages:
                row["page_numbering"] = "system_rendered_page"
        # Keep extraction limitations, but make their phase explicit after OCR.
        native_notes = []
        for warning in loaded.warnings:
            if warning.startswith("以下物理页未完整读取："):
                continue
            if "扫描件自动 OCR 尚未提供" in warning:
                warning = "原生提取阶段未取得可用文字，已按候选页尝试 OCR 补充；识别结果仍需核对原页。"
            elif "没有自动识别" in warning:
                warning = "原生提取只覆盖已有文字层；图像正文按候选页补充 OCR，其他图像仍需查看原页。"
            elif "没有识别图片" in warning:
                warning = "Word 原生提取只覆盖已有文字与文本框；图像内容按候选预览页补充 OCR，仍需核对原页。"
            native_notes.append(warning)
        loaded.warnings = native_notes
        loaded.warnings.extend(ocr_evidence_warnings(loaded.ocr_pages, source_kind=loaded.source_kind))
        if any(row.get("reason") == "document_text_budget" for row in loaded.ocr_pages):
            loaded.warnings.append("OCR 补充文字已达到单份文档的文本量限制，部分页面未用于本轮问答或审查；请拆分文档后继续，并查看原页。")
        if loaded.unread_pages:
            loaded.warnings.append("仍未取得完整文字的物理页：" + "、".join(map(str, loaded.unread_pages)) + "；请查看原页，不能据此认定方案缺少内容。")
    if supplements:
        loaded.warnings.append("OCR 为页图补充，可能与原生文字重复或冲突；关键数字、单位、合并表格和完整公式仍需核对原页。")
    loaded.warnings = list(dict.fromkeys(loaded.warnings))
    return loaded
