"""Model-authored working notes; never a source of evidence."""
from __future__ import annotations
from copy import deepcopy
import json

NOTE_POLICY_VERSION = "source-linked-working-notes-v1"
MAX_NOTES = 24
MAX_NOTE_CHARS = 12000  # Legacy structured experiments only.
NATURAL_NOTE_MAX_CHARS = 50000
STATUSES = {"needs_check", "supported", "withdrawn", "needs_source"}

class ReviewNotebook:
    def __init__(self):
        self.notes = {}
        # One model-maintained record, kept separately from dialogue summaries.
        # Oversized writes are rejected whole, never truncated into new notes.
        self._natural_text = ""

    @property
    def text(self) -> str:
        """Current free-form continuation record."""
        return self._natural_text

    @text.setter
    def text(self, value: str):
        self._natural_text = self._normalize_natural_text(value)

    @staticmethod
    def _normalize_natural_text(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("自然笔记须为文本。")
        value = text.strip()
        if len(value) > NATURAL_NOTE_MAX_CHARS:
            raise ValueError("工作笔记最多50000字符；本次未保存，原笔记不变。请合并精简后重新提交完整笔记。")
        return value

    def write(self, text: str) -> dict:
        """Replace the current natural-language continuation record.

        The single ``text`` argument is intentionally sufficient for a tool
        call. Empty text clears the record. It is a working aid only; it does
        not create evidence or validate any claim. Rewriting the same text is
        treated as unchanged, which lets callers avoid duplicate injection.
        """
        try:
            value = self._normalize_natural_text(text)
        except ValueError as exc:
            return {"ok": False, "error": {"code": "invalid_working_notes", "message": str(exc)}}
        changed = value != self._natural_text
        self._natural_text = value
        return {
            "ok": True,
            "changed": changed,
            "cleared": not bool(value),
            "char_count": len(value),
            "note": "仅保存自然语言接续记录，不取得原文，不确认事实或规范完整性。",
        }

    def readable(self) -> str:
        """Return natural notes as prompt-ready text, or ``""`` when empty."""
        return self._natural_text

    def clear(self, *, structured: bool = False) -> dict:
        """Clear natural notes, and optionally the legacy structured notes."""
        had_natural = bool(self._natural_text)
        self._natural_text = ""
        if structured:
            self.notes.clear()
        return {
            "ok": True,
            "natural_cleared": had_natural,
            "structured_cleared": bool(structured),
        }

    def update(self, notes: list[dict], remove_ids: list[str] | None = None) -> dict:
        if not isinstance(notes, list) or len(notes) > MAX_NOTES:
            raise ValueError("notes须为最多24项的列表。")
        if remove_ids is not None and (not isinstance(remove_ids, list) or any(not isinstance(x,str) for x in remove_ids)):
            raise ValueError("remove_ids须为字符串列表。")
        proposed = deepcopy(self.notes)
        for key in remove_ids or []:
            proposed.pop(key, None)
        for raw in notes:
            if not isinstance(raw, dict):
                raise ValueError("每条笔记须为对象。")
            note_id = raw.get("note_id")
            summary = raw.get("summary")
            if not isinstance(note_id,str) or not 1 <= len(note_id.strip()) <= 60:
                raise ValueError("note_id须为1至60字符。")
            if not isinstance(summary,str) or not 1 <= len(summary.strip()) <= 500:
                raise ValueError("summary须为1至500字符的待核事项或进展，不代替原文。")
            status = raw.get("status", "needs_check")
            if status not in STATUSES:
                raise ValueError("status须为needs_check、supported、withdrawn或needs_source。")
            item = {"note_id":note_id.strip(),"summary":summary.strip(),"status":status}
            for field in ("plan_unit_ids", "standard_evidence_ids", "pending_checks"):
                values = raw.get(field, [])
                if not isinstance(values,list) or len(values)>12 or any(not isinstance(v,str) or len(v)>300 for v in values):
                    raise ValueError(field+"须为最多12项的短字符串列表。")
                item[field] = list(dict.fromkeys(v.strip() for v in values if v.strip()))
            proposed[note_id.strip()] = item
        if len(proposed)>MAX_NOTES or len(json.dumps(list(proposed.values()),ensure_ascii=False))>MAX_NOTE_CHARS:
            raise ValueError("笔记总量超限，请合并或移除已处理项；原记录未改变。")
        self.notes = proposed
        return {"ok":True,"saved_note_ids":[n["note_id"] for n in notes],
                "note_count":len(proposed),"note":"仅保存模型待核工作笔记，不取得任何原文，不增加覆盖率，不确认事实或规范完整性。"}

    def snapshot(self):
        """Return a natural record when present, otherwise the legacy snapshot.

        The no-argument form is useful to root-side prompt assembly: a written
        natural record is returned as text.  An empty natural notebook keeps
        the historical structured shape so older experiments remain usable.
        """
        if self._natural_text:
            return self._natural_text
        return {"policy":NOTE_POLICY_VERSION,"notes":deepcopy(list(self.notes.values())),
                "is_evidence":False,"note":"这些是本模型的工作线索，可能错误或已过时；结论与引文仍须回到本轮实际原文。supported只是模型自记状态，不是程序验证或工程结论。"}
