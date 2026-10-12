"""Separate returned text, proofreading, and related-image delivery.

These are source facts, not an engineering verdict. Legacy `complete` flags
remain unchanged and must not be interpreted as a text-truncation flag.
"""


def text_delivery_status(record):
    if record.get("text_truncated") is True:
        return "truncated"
    if record.get("content_status") in {"image_only", "source_page_only"}:
        return "reference_only"
    count = record.get("stored_text_chars")
    if type(count) is int:
        return "full_stored_text" if count > 0 and record.get("text_truncated") is False else "not_available"
    if isinstance(record.get("text"), str) and record["text"].strip():
        return "returned_untruncated" if record.get("text_truncated") is False else "unknown"
    return "not_available"


def source_checks(record, *, sent_pages=(), source_ref_for_path=None):
    """Only an exact registered source + physical page confirms image delivery."""
    provided = {(page.get("source_ref"), page.get("page")) for page in sent_pages
                if page.get("sent_to_model") is True and page.get("source_kind") == "standard"
                and page.get("source_ref")}
    required, unlocated = set(), 0
    refs = record.get("image_refs") or record.get("asset_refs") or []
    for ref in refs:
        if not isinstance(ref, dict):
            unlocated += 1
            continue
        source = ref.get("source_ref")
        if not isinstance(source, str):
            source = None
        if not source and callable(source_ref_for_path):
            for key in ("source_path", "asset_path", "image_path", "file"):
                if isinstance(ref.get(key), str) and ref[key]:
                    source = source_ref_for_path(ref[key])
                    if source:
                        break
        start = ref.get("source_page", ref.get("page", ref.get("page_start")))
        end = start if "source_page" in ref else ref.get("page_end", start)
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            unlocated += 1
            continue
        if not source:
            unlocated += 1
        required.update((source or "", number) for number in range(start, end + 1))
    received, pending = required & provided, required - provided
    if not refs:
        image_state = "not_linked"
    elif not pending and not unlocated:
        image_state = "pages_provided"
    elif received:
        image_state = "partly_provided"
    else:
        image_state = "not_confirmed" if unlocated and provided else "not_provided"
    review_pending = bool(record.get("review_unverified")) or record.get("review_status") in {
        "unreviewed_ocr", "needs_review"} or record.get("content_status") in {
        "needs_review", "text_plus_image_needs_review", "image_with_unreviewed_ocr", "unreviewed_ocr", "ocr_candidate"}
    page_rows = lambda values: [{"source_ref": source or None, "page": page} for source, page in sorted(values)]
    return {"text_delivery": text_delivery_status(record),
            "proofreading_pending": review_pending, "review_status": record.get("review_status"),
            "related_images": {"status": image_state, "required_pages": page_rows(required),
                               "provided_pages": page_rows(received), "pending_pages": page_rows(pending),
                               "unlocated_references": unlocated}}


def source_status_notice(record, checks, *, legacy_complete=False):
    """Explain an actual limitation without calling every limitation missing text."""
    text_state = checks["text_delivery"]
    images = checks["related_images"]
    image_gap = images["status"] not in {"not_linked", "pages_provided"}
    text_gap = text_state not in {"full_stored_text", "returned_untruncated"}
    if legacy_complete and not text_gap and not image_gap and not checks["proofreading_pending"]:
        return None
    pieces = [{
        "full_stored_text": "库内正文已完整返回，未截断",
        "returned_untruncated": "文字已返回，未标记截断",
        "truncated": "本次返回的文字存在截断",
        "reference_only": "当前仅取得图片或原页线索，未取得已登记正文/表体",
        "not_available": "尚未取得正文/表体文字",
        "unknown": "文字返回范围尚未确认",
    }[text_state]]
    if images["status"] != "not_linked":
        description = {
            "pages_provided": "关联规范附图原页已提供，图像识别未逐字核验",
            "partly_provided": "关联规范附图仅部分原页已提供",
            "not_provided": "关联规范附图尚未读取",
            "not_confirmed": "关联规范附图的实际读取范围尚未确认",
        }[images["status"]]
        missing_numbers = sorted({page["page"] for page in images["pending_pages"]})
        if missing_numbers:
            description += "（尚未提供物理第" + "、".join(map(str, missing_numbers)) + "页）"
        pieces.append(description)
    if checks["proofreading_pending"]:
        pieces.append("文字校勘状态仍待确认；正文送达不代表OCR逐字无误")
    elif not legacy_complete and not image_gap and not text_gap:
        pieces.append("原有整条证据完整性标记尚未确认，不等同正文缺失")
    return "；".join(pieces) + "。"
