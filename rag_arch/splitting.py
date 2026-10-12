"""Split readable units without normalizing document numbers or symbols."""
from __future__ import annotations

from .models import Chunk, LoadedDocument

_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", ",", " ")


def _spans(text: str, chunk_size: int, overlap: int):
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            minimum = start + max(overlap + 1, chunk_size // 2)
            for separator in _SEPARATORS:
                found = text.rfind(separator, minimum, end)
                if found >= 0:
                    end = found + len(separator)
                    break
        yield start, end
        if end == len(text):
            break
        start = end - overlap


def split_document(
    doc: LoadedDocument,
    collection_id: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Split each unit separately, keeping its page, paragraph, row or lines.

    char_start/char_end are a half-open range within the original TextUnit, never
    a global document index. No image or Markdown link is opened while splitting.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size 必须是正整数。")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size。")
    if not isinstance(collection_id, str) or not collection_id.strip():
        raise ValueError("collection_id 不能为空。")
    chunks = []
    for unit_number, unit in enumerate(doc.units, 1):
        if not unit.text.strip():
            continue
        is_text = doc.source_kind in {"txt", "md"}
        previous_start = 0
        line_start = int(unit.locator.get("line_start", 1)) if is_text else 0
        for part_number, (start, end) in enumerate(_spans(unit.text, chunk_size, overlap), 1):
            text = unit.text[start:end]
            locator = dict(unit.locator)
            locator.update({"char_start": start, "char_end": end})
            if is_text:
                # Count only the newly traversed prefix, then this short chunk.
                # Overlapping chunks keep exact source lines without rescanning
                # the entire document or storing an extra newline index.
                line_start += unit.text.count("\n", previous_start, start)
                previous_start = start
                locator["line_start"] = line_start
                locator["line_end"] = line_start + unit.text.count("\n", start, end - 1)
            if not text.strip():
                continue
            chunks.append(Chunk(
                doc_id=doc.doc_id,
                filename=doc.filename,
                chunk_id=f"{doc.doc_id}:{unit_number}:{part_number}",
                text=text,
                locator=locator,
                source_kind=doc.source_kind,
                collection_id=collection_id,
            ))
    return chunks
