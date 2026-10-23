"""Shared catalog discovery helpers; never turn registration text into clauses."""
from __future__ import annotations
from copy import deepcopy
import re
from .standard_identifiers import parse_standard_references

CANDIDATE_FIELDS = ("candidate_only", "match_method", "match_notes", "requested_identifier")

def explicit_standard_requests(query):
    """Keep visibly written years, including abbreviated or malformed suffixes.

    In particular, GB50212-91 must not become the yearless GB50212 and silently
    select a newer edition. Candidate resolution can later explain differences.
    """
    return list(dict.fromkeys(re.sub(r"\s+", "", row["normalized"])
                              for row in parse_standard_references(query)))


def project_catalog(entry, *, brief=False):
    """Bound tool context while leaving the server-side source inventory intact."""
    if brief:
        return {key: deepcopy(entry[key]) for key in ("std_no", "title") if key in entry}
    fields = ("catalog_id", "std_id", "code", "std_no", "title", "library_class",
              "text_available", "fulltext_available", "original_available", "page_reading_available",
              "discovery_only", "usage_note", "metadata_verification")
    value = {key: deepcopy(entry[key]) for key in fields if key in entry}
    value.update({key: deepcopy(entry[key]) for key in CANDIDATE_FIELDS if key in entry})
    for key, limit in (("source_files", 2), ("metadata_sources", 3)):
        sources = entry.get(key, [])
        sources = sources if isinstance(sources, list) else []
        value[key + "_count"] = entry.get(key + "_count", len(sources))
        shown = sources[:limit]
        value[key] = deepcopy(shown)
        value[key + "_truncated"] = len(shown) < value[key + "_count"]
    if value["source_files_truncated"]:
        value["source_inventory_note"] = "这里只显示代表来源；完整原件及页序登记由原页服务保留，未删除或丢页。"
    return value


def discover_catalog(index, query, filters=None, limit=10):
    filters = filters or {}
    selected = filters.get("std_nos", [filters["std_no"]] if filters.get("std_no") else [])
    selected = [selected] if isinstance(selected, str) else selected
    requested = explicit_standard_requests(query)
    if requested or selected:
        matches = []
        names = [row["raw"] for row in parse_standard_references(query)] if requested else selected
        for name in names:
            rows = index.candidates(name, limit=limit)
            exact = [row for row in rows if row.get("candidate_only") is False]
            matches.extend(exact or rows)
    else:
        matches = index.candidates(query, limit=limit)
    if selected:
        allowed = {entry["catalog_id"] for name in selected for entry in index.get(name)["matches"]}
        matches = [entry for entry in matches if entry["catalog_id"] in allowed]
    return _project_catalog_matches(matches, limit)


def _project_catalog_matches(matches, limit):
    seen, result = {}, []
    for entry in matches:
        identity = entry["catalog_id"]
        if identity not in seen:
            seen[identity] = len(result)
            result.append(project_catalog(entry))
        elif result[seen[identity]].get("candidate_only") and not entry.get("candidate_only"):
            result[seen[identity]] = project_catalog(entry)
    return result[:limit]


def _selected_catalog_ids(index, selected):
    return {entry["catalog_id"] for name in selected for entry in index.get(name)["matches"]}


async def adiscover_catalog(index, query, filters=None, limit=10):
    """Async catalog discovery that never runs a synchronous embedding call on the loop."""
    from .async_support import run_local

    filters = filters or {}
    selected = filters.get("std_nos", [filters["std_no"]] if filters.get("std_no") else [])
    selected = [selected] if isinstance(selected, str) else selected
    requested = explicit_standard_requests(query)
    if requested or selected:
        matches = []
        names = [row["raw"] for row in parse_standard_references(query)] if requested else selected
        candidates = getattr(index, "acandidates", None)
        for name in names:
            if callable(candidates):
                rows = await candidates(name, limit=limit)
            else:
                # Legacy in-memory catalogs can still be queried off-loop. If
                # embeddings are configured, candidates() may perform network
                # I/O and must have an explicit async adapter instead.
                if getattr(index, "embeddings", None) is not None:
                    raise RuntimeError("启用了目录向量服务，但目录索引没有异步候选接口。")
                rows = await run_local(index.candidates, name, limit=limit)
            exact = [row for row in rows if row.get("candidate_only") is False]
            matches.extend(exact or rows)
    else:
        candidates = getattr(index, "acandidates", None)
        if callable(candidates):
            matches = await candidates(query, limit=limit)
        else:
            if getattr(index, "embeddings", None) is not None:
                raise RuntimeError("启用了目录向量服务，但目录索引没有异步候选接口。")
            matches = await run_local(index.candidates, query, limit=limit)
    if selected:
        allowed = await run_local(_selected_catalog_ids, index, selected)
        matches = [entry for entry in matches if entry["catalog_id"] in allowed]
    return await run_local(_project_catalog_matches, matches, limit)


def catalog_metadata_query(query):
    """Conservative directory intent; technical questions still require text/pages."""
    # Explicit technical wording takes precedence over words such as '名称'.
    if re.search(r"条文|条款|\d+\.\d+\.\d+|强制|是否现行|是否废止|合规|应当|不应|要求|允许|限值|"
                 r"规定.*(?:压力|厚度|温度|荷载)|(?:压力|厚度|温度|荷载).*(?:多少|多大|数值|上限|下限)", query):
        return False
    return bool(re.search(r"目录|编号|全称|叫什么|名称|收录|库存|全文|原件|文字版|扫描件|资料完整|下载|来源|(?:查找|推荐|列出|找).*(?:标准|规范)|哪[些本].*(?:标准|规范)|(?:标准|规范).*哪[些本]|是什么(?:标准|规范)|有[没有无].*(?:标准|规范)", query))


def catalog_answer(matches, warnings=None, *, missing_text=False):
    """Bounded deterministic answer: identity, availability, and source pointers only."""
    from .models import Answer
    rows = deepcopy(matches)
    lines = ["本地找到以下资料："]
    for number, row in enumerate(rows, 1):
        state = ("已有检索文字" if row.get("text_available") else
                 "已保存原件，尚未准备检索文字" if row.get("original_available") else
                 "只有目录信息，暂无正文")
        lines.append(f"[CATALOG{number}] {row.get('std_no', row.get('code', ''))}《{row.get('title', '')}》：{state}。")
        if row.get("candidate_only"):
            lines.append("这是待核对的目录候选，尚未确认与原引用为同一版本。")
        for note in row.get("match_notes", []):
            if isinstance(note, str):
                lines.append(note)
        if row.get("page_reading_available"):
            lines.append("可使用已登记的原页读取接口查看原件；原页实际读取前不推定条款内容。")
        if not row.get("original_available"):
            lines.append("可按目录登记来源获取原件；本地暂不能回答该标准的具体条款或数值。")
    if missing_text:
        lines.append("本轮未取得可支持所问内容的正文；目录命中不等于已取得条文依据。")
    return Answer("\n".join(lines), [], list(warnings or []), False,
                  {"answer_scope": "catalog", "catalog_reference_ids": [row["catalog_id"] for row in rows],
                   "semantic_support_checked": False, "engineering_evidence": False},
                  catalog_matches=rows, answer_scope="catalog")
