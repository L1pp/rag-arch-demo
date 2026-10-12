"""Offline clause/table parsing for the reviewed, repaired body candidates.

This parser preserves source identity and does not infer applicability, current
legal status, or the values of an image/formula. It never follows document links.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class BodySourceSpec:
    std_id: str
    std_no: str
    candidate_path: str
    sources: dict[str, dict[str, str]]
    official_source: dict[str, str]
    commentary_path: str = ""
    table_header_row_counts: dict[str, int] = field(default_factory=dict)


def source_specs() -> list[BodySourceSpec]:
    specs = [
        BodySourceSpec(
            std_id=name, std_no=name.replace("GB", "GB ", 1),
            candidate_path=f"data/extracted/repaired/{name}_body.md",
            commentary_path=f"data/extracted/repaired/{name}_commentary.md",
            sources={"tuchuang": {
                "source_id": f"FILE_{name.split('-')[0]}_TUCHUANG",
                "source_path": f"data/raw/standards/text_candidates/{name}_图创文字导出.pdf",
            }},
            official_source={
                "source_id": f"MOHURD_{name.replace('-', '_')}",
                "source_path": f"data/raw/standards/{name}_住建部.pdf",
            },
        )
        for name in ("GB55023-2022", "GB55034-2022")
    ]
    specs.append(BodySourceSpec(
        std_id="GB55008-2021", std_no="GB 55008-2021",
        candidate_path="data/extracted/repaired/GB55008-2021_body.md",
        commentary_path="data/extracted/repaired/GB55008-2021_commentary.md",
        sources={"yuanlin8": {
            "source_id": "FILE_GB55008_HTML_YUANLIN8",
            "source_path": "data/incoming/yuanlin8/GB55008-2021_yuanlin8.html",
            "source_url": "https://yuanlin8.com/guifan/jiegou/f79ac9fc89.html",
        }, "mohurd": {
            "source_id": "MOHURD_GB55008_2021",
            "source_path": "data/raw/standards/GB55008-2021_住建部.pdf",
        }},
        official_source={
            "source_id": "MOHURD_GB55008_2021",
            "source_path": "data/raw/standards/GB55008-2021_住建部.pdf",
        },
        # Reviewed header rows in the already expanded source Markdown.
        # This preserves table structure, not engineering interpretation.
        table_header_row_counts={
            "3.1.8": 2, "3.2.1": 2, "3.2.2": 2, "3.3.2": 1,
            "4.4.6": 1, "4.4.8-1": 2, "4.4.8-2": 1,
            "4.4.9-1": 2, "4.4.9-2": 1,
        },
    ))
    return specs


_CLAUSE = re.compile(r"^(\d+\.\d+\.\d+)\s+")
_HEADING = re.compile(r"^(\d+(?:\.\d+)?)\s+(.+)$")
_SOURCE = re.compile(
    r"^<!--\s*source:\s*([^;]+);\s*(source_pdf_page|body_block_index):\s*(\d+);"
    r"\s*document_part:\s*([^; >]+)\s*-->$"
)
_MANUAL_PAGE = re.compile(r"^\[按住建部原PDF物理第(\d+)页人工补录[^]]*\]$")
_CAPTION = re.compile(r"^(?:\*\*|#{1,6}\s*)?表\s*(\d+\.\d+\.\d+(?:-\d+)?)\s*(.*?)\**$")
_TABLE_REF = re.compile(r"表\s*(\d+\.\d+\.\d+(?:-\d+)?)")
_STRENGTH = re.compile(
    r"不应|不得|不宜|严禁|必须|应当|(?<![相对效反响感供适])应(?![急力变])"
    r"|宜|可(?![靠燃变调塑溶逆])"
)
_LINK = re.compile(r"!?\[([^]]*)\]\(([^)]+)\)")


def _standard_key(value: str) -> str:
    return re.sub(r"\s+", "", value).upper().replace("－", "-")


def _numbered_heading(lines: list[str], index: int) -> tuple[str, str] | None:
    """Distinguish chapter/section headings from numbered items by the next ID."""
    line = re.sub(r"^#{1,6}\s+", "", lines[index].strip())
    is_markdown_heading = lines[index].lstrip().startswith("#")
    match = (re.match(r"^(\d+(?:\.\d+)?)\s*([^\d.\s].*)$", line)
             if is_markdown_heading else _HEADING.match(line))
    if not match:
        return None
    number, title = match.groups()
    # A final numbered requirement may share the chapter number and be followed
    # by its next clause. Sentence punctuation distinguishes it from the reviewed
    # short chapter/section titles; do not turn e.g. "4 其他永久荷载。" into a title.
    if re.search(r"[。；;：:，,]", title):
        return None
    for following in lines[index + 1:]:
        is_following_heading = following.lstrip().startswith("#")
        following = re.sub(r"^#{1,6}\s+", "", following.strip())
        if not following or following.startswith("<!--"):
            continue
        next_id = re.match(
            r"^(\d+(?:\.\d+){1,2})(?=[^\d.]|$)" if is_following_heading
            else r"^(\d+(?:\.\d+){1,2})(?:\s|$)", following)
        if next_id and next_id.group(1).startswith(number + "."):
            return number, f"{number} {title}"
        return None
    return None


def _spans(entries: list[tuple[int, str, dict]], candidate_path: str) -> list[dict]:
    spans: list[dict] = []
    for line_no, _, location in entries:
        key = (location.get("source_id"), location.get("source_path"),
               location.get("page"), location.get("block"))
        if spans and key == spans[-1]["_key"]:
            spans[-1]["candidate_line_end"] = line_no
        else:
            spans.append({
                "_key": key,
                "source_id": key[0], "source_path": key[1],
                "page_start": key[2], "page_end": key[2],
                "candidate_path": candidate_path,
                "candidate_line_start": line_no, "candidate_line_end": line_no,
                "role": "text_source",
            })
            if key[3] is not None:
                spans[-1].update(block_start=key[3], block_end=key[3])
            if location.get("source_url"):
                spans[-1]["source_url"] = location["source_url"]
    for span in spans:
        span.pop("_key")
    return spans


def _page_bounds(spans: list[dict]) -> tuple[int | None, int | None]:
    paths = {s["source_path"] for s in spans}
    if len(paths) != 1 or None in paths or any(s["page_start"] is None for s in spans):
        return None, None
    return min(s["page_start"] for s in spans), max(s["page_end"] for s in spans)


def _cells(line: str) -> list[str]:
    # The reviewed tables contain no escaped pipes. Reject unsupported structure
    # instead of silently treating an escaped pipe as another column.
    if r"\|" in line:
        raise ValueError("escaped_pipe")
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _marker_location(marker: re.Match, spec: BodySourceSpec) -> tuple[dict, str, str]:
    source_name, locator_type, value, part = marker.groups()
    source = spec.sources.get(source_name.strip(), {})
    locator = "page" if locator_type == "source_pdf_page" else "block"
    return {**source, locator: int(value)}, part, source_name


def _next_table_content(lines: list[str], index: int, location: dict,
                        spec: BodySourceSpec) -> tuple[int, dict]:
    """Look ahead through whitespace/source markers, committing only with content."""
    current = dict(location)
    while index < len(lines):
        value = lines[index].strip()
        if not value:
            index += 1
            continue
        marker = _SOURCE.match(value)
        if marker:
            new_location, part, source_name = _marker_location(marker, spec)
            if part != "body" or source_name not in spec.sources:
                break
            current = new_location
            index += 1
            continue
        break
    return index, current


def parse_body(text: str, spec: BodySourceSpec) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse repaired Markdown without changing its source files or following links."""
    lines = text.lstrip("\ufeff").splitlines()
    clauses: list[dict] = []
    tables: list[dict] = []
    issues: list[dict] = []
    location: dict = {}
    part = "body"
    headings: list[str] = []
    active: dict | None = None
    seen_ids: set[str] = set()

    def issue(code: str, line: int, **extra) -> None:
        issues.append({"std_id": spec.std_id, "code": code, "candidate_line": line, **extra})

    def finish() -> None:
        nonlocal active
        if active is None:
            return
        entries = active.pop("_entries")
        spans = _spans(entries, spec.candidate_path)
        page_start, page_end = _page_bounds(spans)
        clause_text = "\n\n".join(value for _, value, _ in entries)
        tokens = list(dict.fromkeys(_STRENGTH.findall(clause_text)))
        schema_tokens = [t if t in {"应", "不应", "不得", "宜", "可"} else "mixed" for t in tokens]
        strength = schema_tokens[0] if len(set(schema_tokens)) == 1 else "mixed" if tokens else "none"
        active.update(
            text=clause_text, strength=strength, strength_tokens=tokens,
            strength_method="lexical_marker_candidates",
            source_spans=spans, page_start=page_start, page_end=page_end,
            line_start=entries[0][0], line_end=entries[-1][0],
            table_refs=list(dict.fromkeys(
                f"{spec.std_id}/table/{no}" for no in _TABLE_REF.findall(clause_text)
            )),
        )
        if any(s["source_path"] is None or
               (s["page_start"] is None and s.get("block_start") is None) for s in spans):
            issue("missing_source_locator", entries[0][0], evidence_id=active["clause_id"])
        clauses.append(active)
        active = None

    i = 0
    while i < len(lines):
        line_no, raw = i + 1, lines[i]
        value = raw.strip()
        marker = _SOURCE.match(value)
        if marker:
            location, part, source_name = _marker_location(marker, spec)
            if part != "body":
                finish()
                issue("non_body_section_excluded", line_no, document_part=part)
            elif not location.get("source_id"):
                issue("unknown_source_identity", line_no, source_name=source_name)
            i += 1
            continue
        manual = _MANUAL_PAGE.match(value)
        if manual:
            location = {**spec.official_source, "page": int(manual.group(1))}
            i += 1
            continue
        if not value or part != "body":
            i += 1
            continue
        if value.startswith("<!--"):
            issue("unsupported_source_marker", line_no)
            # Do not assign the preceding source page to following unknown text.
            location = {}
            i += 1
            continue
        heading = _numbered_heading(lines, i)
        if heading:
            finish()
            number, label = heading
            if "." not in number:
                headings = [label]
            else:
                headings = headings[:1] + [label]
            i += 1
            continue
        start = _CLAUSE.match(value)
        if start:
            finish()
            clause_no = start.group(1)
            clause_id = f"{spec.std_id}/{clause_no}"
            if clause_id in seen_ids:
                raise ValueError(f"Duplicate clause ID: {clause_id} at line {line_no}")
            seen_ids.add(clause_id)
            active = {
                "clause_id": clause_id, "std_id": spec.std_id, "std_no": spec.std_no,
                "clause_no": clause_no, "heading_path": list(headings),
                "document_part": "body", "candidate_path": spec.candidate_path,
                "original_mandatory": None, "current_mandatory_status": "unknown",
                "table_ids": [], "image_refs": [], "parse_note": "",
                "_entries": [(line_no, value, dict(location))],
            }
            i += 1
            continue
        caption = _CAPTION.match(value)
        if caption:
            table_no, title = caption.groups()
            table_id = f"{spec.std_id}/table/{table_no}"
            if table_id in seen_ids:
                raise ValueError(f"Duplicate table ID: {table_id} at line {line_no}")
            seen_ids.add(table_id)
            j = i + 1
            table_lines: list[str] = []
            table_line_numbers: list[int] = []
            table_location = dict(location)
            table_entries = [(line_no, value, dict(location))]
            while j < len(lines):
                next_index, next_location = _next_table_content(lines, j, table_location, spec)
                if next_index >= len(lines) or not lines[next_index].strip().startswith("|"):
                    break
                table_lines.append(lines[next_index])
                table_line_numbers.append(next_index + 1)
                table_entries.append((next_index + 1, lines[next_index], dict(next_location)))
                table_location = next_location
                j = next_index + 1
            notes: list[str] = []
            provenance: list[str] = []
            provenance_spans: list[dict] = []
            evidence_end = table_line_numbers[-1] if table_line_numbers else line_no
            numbered_notes = False
            while j < len(lines):
                next_index, next_location = _next_table_content(lines, j, table_location, spec)
                if next_index >= len(lines):
                    break
                following = lines[next_index].strip()
                if following.startswith(("注：", "注:")):
                    notes.append(following)
                    numbered_notes = following in {"注：", "注:"}
                    evidence_end = next_index + 1
                    table_entries.append((evidence_end, following, dict(next_location)))
                elif numbered_notes and re.match(r"^\d+[.、]\s", following):
                    notes.append(following)
                    evidence_end = next_index + 1
                    table_entries.append((evidence_end, following, dict(next_location)))
                elif following.startswith(("> 图片转录依据：", "转录说明：")):
                    annotation = following.removeprefix("> ").strip()
                    provenance.append(annotation)
                    provenance_spans.append({
                        "candidate_path": spec.candidate_path,
                        "line_start": next_index + 1, "line_end": next_index + 1, "text": annotation,
                    })
                    numbered_notes = False
                else:
                    break
                table_location = next_location
                j = next_index + 1
            content_status, headers, rows = "text_available", [], []
            try:
                if len(table_lines) < 2:
                    raise ValueError("missing_markdown_table")
                headers = _cells(table_lines[0])
                separator = _cells(table_lines[1])
                if len(separator) != len(headers) or not all(re.fullmatch(r":?-+:?", c) for c in separator):
                    raise ValueError("unsupported_table_header")
                rows = [_cells(row) for row in table_lines[2:]]
                if any(len(row) != len(headers) for row in rows):
                    raise ValueError("inconsistent_table_columns")
            except ValueError as exc:
                content_status = "unsupported"
                headers, rows = [], []
                issue("unsupported_table_structure", line_no, evidence_id=table_id, reason=str(exc))
            owner = f"{spec.std_id}/{table_no.split('-')[0]}"
            if active is None or active["clause_id"] != owner:
                issue("table_owner_context_mismatch", line_no, evidence_id=table_id, owner_clause_id=owner)
            spans = _spans(table_entries, spec.candidate_path)
            source_notes = "\n".join(provenance)
            crossing = re.search(r"图创跨物理第\s*(\d+)[–—~-](\d+)页", source_notes)
            if crossing:
                spans[0]["page_start"], spans[0]["page_end"] = map(int, crossing.groups())
                spans[0]["source_mapping_note_lines"] = [
                    p["line_start"] for p in provenance_spans if "图创跨物理第" in p["text"]
                ]
            verified = re.search(r"住建部原件\]\([^)]+\)物理第\s*(\d+)\s*页", source_notes)
            if verified:
                verification_line = next(
                    p["line_start"] for p in provenance_spans if "住建部原件" in p["text"]
                )
                spans.append({
                    **spec.official_source, "page_start": int(verified.group(1)),
                    "page_end": int(verified.group(1)), "role": "table_visual_verification",
                    "candidate_path": spec.candidate_path,
                    "candidate_line_start": verification_line, "candidate_line_end": verification_line,
                })
            header_metadata = {}
            flat_headers = headers
            if spec.table_header_row_counts and content_status == "text_available":
                header_count = spec.table_header_row_counts.get(table_no, 1)
                if header_count < 1 or header_count > len(rows) + 1:
                    raise ValueError(f"Invalid reviewed header row count for {table_id}")
                header_rows = [headers] + rows[:header_count - 1]
                rows = rows[header_count - 1:]
                column_paths = [
                    list(dict.fromkeys(row[column] for row in header_rows))
                    for column in range(len(headers))
                ]
                flat_headers = [" / ".join(path) for path in column_paths]
                header_metadata = {"header_rows": header_rows, "column_paths": column_paths}
            flat = [f"表 {table_no} {title}"]
            flat.extend("；".join(f"{h}：{c}" for h, c in zip(flat_headers, row)) for row in rows)
            flat.extend(notes)
            tables.append({
                "table_id": table_id, "std_id": spec.std_id, "std_no": spec.std_no,
                "table_no": table_no, "caption": f"表 {table_no} {title}",
                "document_part": "body", "markdown": "\n".join(table_lines),
                "html": None, "headers": headers, "rows": rows, "notes": notes,
                **header_metadata,
                "flat_text": "\n".join(flat) if content_status == "text_available" else "",
                "owner_clause_id": owner, "source_spans": spans,
                "page_start": spans[0]["page_start"], "page_end": spans[0]["page_end"],
                "page_source_id": spans[0]["source_id"],
                "candidate_path": spec.candidate_path, "line_start": line_no,
                "line_end": evidence_end, "source_note": source_notes,
                "provenance_spans": provenance_spans,
                "asset_refs": [{"label": label, "href": href} for label, href in _LINK.findall(source_notes)],
                "content_status": content_status,
            })
            location = table_location
            i = j
            continue
        if value.startswith("!["):
            refs = [{"label": label, "href": href, "candidate_line": line_no}
                    for label, href in _LINK.findall(value)]
            if active:
                active["image_refs"].extend(refs)
            issue("image_content_preserved_not_transcribed", line_no, image_refs=refs)
            i += 1
            continue
        if value.startswith(("#", "> ", "[按", "本页图表/公式原图")):
            if active:
                issue("editorial_or_unsupported_markup_excluded", line_no, text=value)
            i += 1
            continue
        if active is not None:
            active["_entries"].append((line_no, value, dict(location)))
        elif location:
            issue("unassigned_body_text", line_no, text=value)
        # Initial candidate title and repair description intentionally stay out.
        i += 1
    finish()
    by_clause = {c["clause_id"]: c for c in clauses}
    by_table = {t["table_id"]: t for t in tables}
    for table in tables:
        owner = by_clause.get(table["owner_clause_id"])
        if owner:
            owner["table_ids"].append(table["table_id"])
        else:
            issue("table_owner_missing", table["line_start"], evidence_id=table["table_id"])
    for clause in clauses:
        clause["unresolved_table_refs"] = [ref for ref in clause["table_refs"] if ref not in by_table]
        for ref in clause["unresolved_table_refs"]:
            issue("missing_table_reference", clause["line_start"], evidence_id=clause["clause_id"], table_id=ref)
    return clauses, tables, issues


def build_corpus(project_root: str | Path, output_dir: str | Path | None = None) -> dict:
    root = Path(project_root).resolve()
    output = Path(output_dir) if output_dir is not None else root / "data/processed/standards_v0"
    all_clauses: list[dict] = []
    all_tables: list[dict] = []
    all_issues: list[dict] = []
    documents = []
    expected = {"GB55023-2022": (77, 3), "GB55034-2022": (129, 0), "GB55008-2021": (99, 9)}
    for spec in source_specs():
        clauses, tables, issues = parse_body((root / spec.candidate_path).read_text(encoding="utf-8-sig"), spec)
        if (len(clauses), len(tables)) != expected[spec.std_id]:
            issues.append({"std_id": spec.std_id, "code": "reviewed_count_mismatch",
                           "expected": list(expected[spec.std_id]), "actual": [len(clauses), len(tables)]})
        documents.append({
            "std_id": spec.std_id, "std_no": spec.std_no,
            "candidate_path": spec.candidate_path, "excluded_commentary_path": spec.commentary_path,
            "clause_count": len(clauses), "table_count": len(tables), "issue_count": len(issues),
        })
        all_clauses.extend(clauses)
        all_tables.extend(tables)
        all_issues.extend(issues)
    summary = {
        "schema_version": "standards_v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "indexed_document_parts": ["body"],
        "clause_count": len(all_clauses), "table_count": len(all_tables),
        "documents": documents, "issues": all_issues,
        "issue_counts": dict(Counter(issue["code"] for issue in all_issues)),
        "limitations": [
            "只解析 GB55023/GB55034/GB55008 已修复正文；其余规范与条文说明未索引。",
            "GB55008主要按园林吧网页块定位，未知PDF页保持null；表的已记录官方核验页另存。",
            "原强条及当前强制状态未推断，当前字段为 null/unknown，等待关联登记数据。",
            "修复候选并非全本逐字校勘；图片链接不会自动读取、OCR或发送给模型。",
            "表格保留现有 Markdown、单位、表注与来源，不执行公式或工程比较。",
            "离线 search 是确定性关键词检索，不是正式 BM25/Dense/Hybrid 评测。",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("clauses.jsonl", all_clauses), ("tables.jsonl", all_tables)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _tokens(text: str) -> set[str]:
    normalized = text.lower()
    result = set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", normalized))
    for run in re.findall(r"[\u3400-\u9fff]+", normalized):
        result.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return result


class StandardCorpus:
    """Read-only local getter and lexical search; records use plain JSON types."""

    def __init__(self, clauses: list[dict], tables: list[dict], summary: dict | None = None,
                 catalog: list[dict] | None = None, relations: list[dict] | None = None):
        self.clauses = clauses
        self.tables = tables
        self.summary = summary or {}
        self.catalog = catalog or []
        self.relations = relations or []
        self._clauses = {c["clause_id"]: c for c in clauses}
        self._tables = {t["table_id"]: t for t in tables}
        if len(self._clauses) != len(clauses) or len(self._tables) != len(tables):
            raise ValueError("Duplicate evidence IDs in corpus")
        self._aliases: dict[str, set[str]] = {}
        self._clause_lookup: dict[tuple[str, str], list[dict]] = {}
        for record in clauses + tables:
            std_id = record.get("std_id") or (record.get("clause_id") or record["table_id"]).split("/", 1)[0]
            full = _standard_key(record["std_no"])
            for alias in {full, _standard_key(std_id), re.sub(r"-\d{4}$", "", full)}:
                self._aliases.setdefault(alias, set()).add(std_id)
        for record in clauses:
            std_id = record.get("std_id") or record["clause_id"].split("/", 1)[0]
            self._clause_lookup.setdefault((std_id, record["clause_no"].strip()), []).append(record)

    def resolve_standard_id(self, std_no: str) -> str | None:
        matches = self._aliases.get(_standard_key(std_no), set())
        return next(iter(matches)) if len(matches) == 1 else None

    def get_clause(self, std_no: str, clause_no: str) -> dict | None:
        std_id = self.resolve_standard_id(std_no)
        matches = [row for row in self._clause_lookup.get((std_id, clause_no.strip()), [])
                   if row.get("exact_clause_lookup") is not False]
        if len(matches) == 1:
            return matches[0]
        # Fully qualified appendix suffixes remain usable even if its printed
        # numbering is also used in the body. Never choose another edition.
        qualified = self._clauses.get(f"{std_id}/{clause_no.strip()}") if std_id else None
        if qualified and qualified.get("exact_clause_lookup") is False and "/" not in clause_no:
            return None
        return qualified

    def get_table(self, table_id: str) -> dict | None:
        return self._tables.get(table_id)

    def search(self, query: str, limit: int = 5, std_no: str | None = None,
               include_tables: bool = True) -> list[dict]:
        if limit < 1 or not query.strip():
            return []
        query_tokens = _tokens(query)
        hits = []
        records = [("clause", c) for c in self.clauses]
        if include_tables:
            records += [("table", t) for t in self.tables if t["content_status"] == "text_available"]
        for kind, record in records:
            if std_no and _standard_key(record["std_no"]) != _standard_key(std_no):
                continue
            body = record["text"] if kind == "clause" else record["flat_text"]
            words = _tokens(body)
            overlap = query_tokens & words
            if not overlap:
                continue
            score = len(overlap) / math.sqrt(max(1, len(words)))
            number = record["clause_no"] if kind == "clause" else record["table_no"]
            if number in query_tokens:
                score += 5.0
            if kind == "table":
                score += len(query_tokens & _tokens(record["caption"])) / max(1, len(query_tokens))
            hits.append({
                "evidence_id": record["clause_id"] if kind == "clause" else record["table_id"],
                "kind": kind, "score": round(score, 6), "record": record,
            })
        return sorted(hits, key=lambda h: (-h["score"], h["evidence_id"]))[:limit]


def load_corpus(output_dir: str | Path) -> StandardCorpus:
    directory = Path(output_dir)

    def records(name):
        return [json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines() if line.strip()]

    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    catalog_path, relations_path = directory / "catalog.json", directory / "relations.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig")).get("standards", []) if catalog_path.exists() else []
    relations = json.loads(relations_path.read_text(encoding="utf-8-sig")).get("relations", []) if relations_path.exists() else []
    return StandardCorpus(records("clauses.jsonl"), records("tables.jsonl"), summary, catalog, relations)
