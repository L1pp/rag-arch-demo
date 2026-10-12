"""Readable plan documents, independent of benchmark assertions and rule packs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

from .config import ROOT
from .loaders import _upload_name_kind, load_document
from .models import LoadedDocument


@dataclass
class PlanDocument:
    document_id: str
    filename: str
    units: list[dict]
    source_kind: str = ""
    page_count: int | None = None
    unread_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_revision: str = ""

    def __post_init__(self):
        if not self.document_id or not self.filename:
            raise ValueError("方案缺少文档标识或文件名。")
        clean, ids = [], set()
        for unit in self.units:
            uid, body = unit.get("unit_id"), unit.get("text")
            if not isinstance(uid, str) or not uid or uid in ids:
                raise ValueError("方案原文单元ID缺失或重复。")
            if not isinstance(body, str):
                raise ValueError("方案原文必须为文字。")
            if unit.get("plan_id", self.document_id) != self.document_id:
                raise ValueError("不能将另一方案的单元混入当前文档。")
            ids.add(uid)
            # Deliberately exclude candidate_scope, inferred frames and answers.
            clean.append({"unit_id": uid, "text": body,
                          "locator": deepcopy(unit.get("locator", {})),
                          "source_path": str(unit.get("source_path") or ""),
                          "source_kind": str(unit.get("source_kind") or self.source_kind)})
        self.units = clean
        if not self.source_revision:
            self.source_revision = uuid4().hex

    @property
    def by_id(self):
        return {u["unit_id"]: u for u in self.units}

    @property
    def text_chars(self):
        return sum(len(u["text"]) for u in self.units)

    def metadata(self):
        return {"document_id": self.document_id, "filename": self.filename,
                "unit_count": len(self.units), "text_chars": self.text_chars, "source_revision": self.source_revision,
                "page_count": self.page_count, "unread_pages": list(self.unread_pages),
                "warnings": list(self.warnings)}

    @classmethod
    def from_bundle(cls, bundle: dict):
        return cls(bundle["document_id"], bundle["filename"], bundle["units"],
                   bundle.get("source_kind", ""), bundle.get("page_count"),
                   list(bundle.get("unread_pages", [])), list(bundle.get("warnings", [])),
                   str(bundle.get("source_revision") or ""))

    def navigation(self, start: int = 1, limit: int = 30, max_chars: int = 8000) -> dict:
        """Bounded source navigation; previews are not acquired evidence."""
        if type(start) is not int or not 1 <= start <= max(1, len(self.units)):
            raise ValueError("start须为现有单元序号，从1开始。")
        if type(limit) is not int or not 1 <= limit <= 60:
            raise ValueError("limit须为1至60。")
        self._check_budget(max_chars)
        result = {"document_id": self.document_id, "entries": [], "total_units": len(self.units),
                  "next_start": start, "navigation_only": True,
                  "note": "仅定位导航，短预览不作为完整原文或引用依据；用read_plan按序号/偏移取证，也可search_plan定位。"}
        for pos in range(start - 1, min(len(self.units), start - 1 + limit)):
            unit = self.units[pos]
            entry = {"ordinal": pos + 1, "unit_id": unit["unit_id"], "locator": unit["locator"],
                     "text_chars": len(unit["text"]), "preview": unit["text"][:48]}
            candidate = deepcopy(result)
            candidate["entries"].append(entry)
            candidate["next_start"] = pos + 2 if pos + 1 < len(self.units) else None
            if len(self._json(candidate)) > max_chars:
                break
            result = candidate
        if self.units and not result["entries"]:
            raise ValueError("当前位置导航元数据超过本次预算，请提高max_chars。")
        return result

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _check_budget(max_chars):
        if type(max_chars) is not int or not 1000 <= max_chars <= 32000:
            raise ValueError("max_chars须为1000至32000，包含序列化返回值。")

    def source_ref_for(self, unit_id: str) -> str | None:
        if getattr(self, "_anchor_index", None) is None:
            from .plan_text_view import document_anchor_index
            self._anchor_index = document_anchor_index(self.units)
        return self._anchor_index["by_unit"].get(unit_id)

    def source_units_for_ref(self, source_ref: str) -> list[str]:
        if getattr(self, "_anchor_index", None) is None:
            from .plan_text_view import document_anchor_index
            self._anchor_index = document_anchor_index(self.units)
        return list(self._anchor_index["by_ref"].get(source_ref, []))

    def _fragment(self, pos, offset, length):
        unit = self.units[pos]
        end = min(len(unit["text"]), offset + length)
        fragment = {"ordinal": pos + 1, "unit_id": unit["unit_id"], "locator": deepcopy(unit["locator"]),
                    "text": unit["text"][offset:end], "text_start": offset, "text_end": end,
                    "unit_text_chars": len(unit["text"]),
                    "complete_unit": offset == 0 and end == len(unit["text"])}
        source_ref = self.source_ref_for(unit["unit_id"])
        if source_ref:
            fragment["source_ref"] = source_ref
        return fragment

    def _append_fragment(self, result, pos, offset, max_chars):
        """Fit a source slice inside the serialized envelope, not a text-only cap."""
        total = len(self.units[pos]["text"])
        def with_size(size):
            candidate = deepcopy(result)
            fragment = self._fragment(pos, offset, size)
            candidate["units"].append(fragment)
            cursor = ({"start": pos + 1, "offset": fragment["text_end"]} if fragment["text_end"] < total
                      else {"start": pos + 2, "offset": 0} if pos + 1 < len(self.units) else None)
            candidate.update(next_cursor=cursor, next_start=cursor["start"] if cursor else None)
            return candidate
        whole = with_size(total - offset)
        if len(self._json(whole)) <= max_chars:
            return whole
        low, high, best = 0, min(total - offset, max_chars), None
        while low <= high:
            size = (low + high) // 2
            candidate = with_size(size)
            if len(self._json(candidate)) <= max_chars:
                best = candidate
                low = size + 1
            else:
                high = size - 1
        if best is None or (total > offset and not best["units"][-1]["text"]):
            return None
        return best

    def read(self, start: int = 1, max_chars: int = 16000, offset: int = 0) -> dict:
        """One-based units, zero-based Unicode character offsets, bounded JSON."""
        if type(start) is not int or not 1 <= start <= max(1, len(self.units)):
            raise ValueError("start须为现有单元的序号，从1开始。")
        self._check_budget(max_chars)
        if type(offset) is not int or offset < 0 or (self.units and offset > len(self.units[start - 1]["text"])):
            raise ValueError("offset须在当前单元字符范围内，从0开始。")
        result = {"document_id": self.document_id, "units": [], "total_units": len(self.units),
                  "next_start": start if self.units else None,
                  "next_cursor": {"start": start, "offset": offset} if self.units else None,
                  "note": "只提供text_start到text_end的原文；偏移按Unicode字符计。next_cursor可接续，部分单元不代表全文已读。"}
        for pos in range(start - 1, len(self.units)):
            chunk = self._append_fragment(result, pos, offset if pos == start - 1 else 0, max_chars)
            if chunk is None:
                break
            result = chunk
            if result["units"][-1]["text_end"] < len(self.units[pos]["text"]):
                break
        if self.units and not result["units"]:
            raise ValueError("当前位置元数据超过本次预算，请提高max_chars。")
        return result

    def search(self, query: str, limit: int = 5, max_chars: int = 16000) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query须为非空关键词。")
        if len(query) > 1000:
            raise ValueError("query最多1000字符。")
        if type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("limit须为1至12。")
        self._check_budget(max_chars)
        from .retrieval import LocalIndex
        from .models import Chunk
        chunks = [Chunk(self.document_id, self.filename, u["unit_id"], u["text"],
                        u["locator"], u["source_kind"], self.document_id) for u in self.units if u["text"].strip()]
        hits = LocalIndex(chunks).search(query, top_k=limit) if chunks else []
        result = {"document_id": self.document_id, "query": query, "units": [], "total_units": len(self.units),
                  "next_start": None, "next_cursor": None,
                  "note": "关键词定位可返回部分原文；每项按ordinal和text_start/text_end用read_plan续读。命中不代表整份核查覆盖。"}
        if len(self._json(result)) > max_chars:
            raise ValueError("query与返回元数据超过本次预算，请缩短query或提高max_chars。")
        positions = {u["unit_id"]: i for i, u in enumerate(self.units)}
        for hit in hits:
            pos = positions[hit.chunk.chunk_id]
            body = self.units[pos]["text"]
            terms = [query.strip(), *re.findall(r"[\w]+", query)]
            found = [body.lower().find(term.lower()) for term in terms if term]
            found = [value for value in found if value >= 0]
            offset = max(0, min(found) - 240) if found else 0
            chunk = self._append_fragment(result, pos, offset, max_chars)
            if chunk is None:
                break
            result = chunk
        return result

    def quote_span(self, unit_id: str, quote: str) -> dict | None:
        unit = self.by_id.get(unit_id)
        if unit is None or not isinstance(quote, str) or not quote.strip():
            return None
        body = unit["text"]
        start = body.find(quote)
        if start >= 0:
            return {"start": start, "end": start + len(quote), "quote": quote, "whitespace_normalized": False}
        # Accept only layout-space and Unicode-width normalization, never numbers
        # guessed from OCR, dropped words, ellipses or semantic paraphrases.
        chars, positions = [], []
        for pos, char in enumerate(body):
            for normal in unicodedata.normalize("NFKC", char):
                if not normal.isspace():
                    chars.append(normal)
                    positions.append(pos)
        needle = "".join(c for c in unicodedata.normalize("NFKC", quote) if not c.isspace())
        start = "".join(chars).find(needle)
        if not needle or start < 0:
            return None
        left, right = positions[start], positions[start + len(needle) - 1] + 1
        return {"start": left, "end": right, "quote": body[left:right], "whitespace_normalized": True}



def registered_plan_labels(root: Path = ROOT) -> dict[str, str]:
    """Display the registered filenames without hiding their Chinese names."""
    info = json.loads((root / "data/processed/plans_v0/summary.json").read_text(encoding="utf-8-sig"))
    labels = {}
    for row in info["plans"]:
        plan_id = row["plan_id"]
        name = Path(row["filename"]).stem.removeprefix(plan_id + "_").replace("_", " · ")
        pages = row.get("page_count")
        if pages and name.endswith(f"{pages}p"):
            name = name.removesuffix(f"{pages}p") + f"{pages}页"
        labels[plan_id] = f"{plan_id} · {name}"
    return labels


def load_registered_document(plan_id: str, root: Path = ROOT) -> PlanDocument:
    """Read raw text units only. No assertions, benchmark, references or rules."""
    directory = root / "data/processed/plans_v0"
    info = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
    row = next((p for p in info["plans"] if p["plan_id"] == plan_id), None)
    if row is None:
        raise ValueError("方案未登记。")
    units = [u for line in (directory / "text_units.jsonl").read_text(encoding="utf-8-sig").splitlines()
             if line.strip() for u in [json.loads(line)] if u["plan_id"] == plan_id]
    return PlanDocument(plan_id, row["filename"], units, "registered", row.get("page_count"),
                        row.get("unread_pages", []), row.get("warnings", []),
                        ":".join(str((directory / f).stat().st_mtime_ns) for f in ("text_units.jsonl", "summary.json")))


def load_uploaded_document(filename: str, data: bytes, *, document_id: str | None = None,
                           source_revision: str = "", pdf_backend=None,
                           loaded_document=None) -> PlanDocument:
    name, kind = _upload_name_kind(filename, data)
    if loaded_document is None:
        loaded = load_document(name, data, pdf_backend=pdf_backend)
    else:
        if (not isinstance(loaded_document, LoadedDocument)
                or loaded_document.filename != name
                or loaded_document.source_kind != kind):
            raise ValueError("已解析文档与上传文件不一致。")
        loaded = loaded_document
    pid = document_id or "UPLOAD-" + loaded.doc_id
    units = []
    for original in loaded.units:
        if loaded.source_kind in {"txt", "md"}:
            lines = [(line, {"line_start": n, "line_end": n, "line_numbering": "source_file"})
                     for n, line in enumerate(original.text.splitlines(keepends=True),
                                               int(original.locator.get("line_start", 1))) if line.strip()]
        else:
            lines = [(original.text, original.locator)]
        for body, locator in lines:
            units.append({"unit_id": f"{pid}/u{len(units) + 1:04d}", "plan_id": pid,
                          "text": body, "locator": deepcopy(locator),
                          "source_kind": loaded.source_kind,
                          "source_path": f"session/{loaded.doc_id}/{loaded.filename}"})
    return PlanDocument(pid, loaded.filename, units, loaded.source_kind, loaded.page_count,
                        loaded.unread_pages, loaded.warnings, source_revision)
