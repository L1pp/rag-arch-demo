"""Readable export of actual whole-plan findings and source locations."""
from __future__ import annotations

import json
import re
from .plan_location_display import format_physical_locator
from .page_regions import normalize_region


def locator_text(locator):
    physical = format_physical_locator(locator)
    if physical:
        return physical
    if locator.get("paragraph"):
        return f"段落{locator['paragraph']}"
    if locator.get("table"):
        row = locator.get("row", locator.get("table_row", ""))
        return f"表{locator['table']} 行{row}"
    if locator.get("line_start"):
        end = locator.get("line_end", locator["line_start"])
        return f"源文件第{locator['line_start']}–{end}行"
    return json.dumps(locator, ensure_ascii=False) if locator else "位置未登记"


_VERDICT_ORDER = ("符合", "不符合", "无法判断", "需要补充资料")
_VISIBLE_REFERENCES = 3
_EXCERPT_CHARS = 240


def _excerpt(text):
    """Presentation only: never replace the evidence stored in the result."""
    text = str(text or "").strip()
    if len(text) <= _EXCERPT_CHARS:
        return text
    return text[:_EXCERPT_CHARS].rstrip() + "…（节选，完整内容见JSON）"


def _unique_plan_refs(refs):
    seen = []
    for ref in refs:
        if ref not in seen:
            seen.append(ref)
            yield ref


def _plan_location(ref):
    if ref.get("evidence_kind") == "page_image":
        source = ref.get("page_source") or {"page": ref.get("locator", {}).get("page")}
        return f"{ref['unit_id']} · {_page_location(source)}（图像识别，未逐字核验）"
    if ref.get("location_verified") is False:
        label = "定位未核实"
        if ref.get("source_ref"):
            label += f"（模型锚点 {ref['source_ref']}）"
        elif ref.get("unit_id"):
            label += f"（模型位置 {ref['unit_id']}）"
        if ref.get("resolution_basis") == "unverified_duplicate_quote":
            label += "：相同引句出现多处，未默认选取"
        return label
    location = f"{ref['unit_id']} · {locator_text(ref['locator'])}"
    if ref.get("resolution_basis") == "unique_quote_in_provided_units":
        location += (f"（原模型位置 {ref.get('model_unit_id')} 未直接命中，"
                     "已按本轮已提供单元中的唯一原文定位）")
    return location


def finding_summary_text(result):
    """Summarize the final, locally validated findings without changing the result."""
    row = result.to_dict() if hasattr(result, "to_dict") else result
    findings = row.get("findings") or []
    if not findings:
        return ""

    counts = {verdict: 0 for verdict in _VERDICT_ORDER}
    extra_counts = {}
    for finding in findings:
        verdict = str(finding.get("verdict") or "未分类")
        if verdict in counts:
            counts[verdict] += 1
        else:
            extra_counts[verdict] = extra_counts.get(verdict, 0) + 1

    parts = [f"{verdict}{counts[verdict]}" for verdict in _VERDICT_ORDER if counts[verdict]]
    parts.extend(f"{verdict}{count}" for verdict, count in extra_counts.items())
    return f"本轮{len(findings)}项：" + "，".join(parts)


def _literal_text_block(value):
    """Fence stored model text so it remains literal when rendered as Markdown."""
    text = "" if value is None else str(value)
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + "text", text, fence]


def candidate_details_markdown(result):
    """Export the model-authored candidate ledger without validating or rewriting it.

    The output is a companion detail view for experiment artifacts. It is
    deliberately separate from :func:`report_markdown` so ordinary reports
    do not grow with every internal candidate record.
    """
    row = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(row, dict):
        return None
    runtime = row.get("runtime") or {}
    ledger = runtime.get("candidate_ledger")
    candidates = ledger.get("candidates") if isinstance(ledger, dict) else None
    candidates = [item for item in candidates if isinstance(item, dict)] if isinstance(candidates, list) else []
    if not candidates:
        return None

    audit = runtime.get("candidate_audit_review")
    audit = audit if isinstance(audit, dict) else {}
    reviewed_ids = {str(item) for item in audit.get("reviewed_ids", []) if item is not None}
    pending_ids = {str(item) for item in audit.get("pending_ids", []) if item is not None}
    inherited_decisions = audit.get("inherited_decisions")
    inherited_decisions = inherited_decisions if isinstance(inherited_decisions, dict) else {}
    current_decisions = audit.get("current_decisions")
    current_decisions = current_decisions if isinstance(current_decisions, dict) else {}

    lines = [
        "# 候选台账明细", "",
        "以下内容来自模型台账记录；程序仅按字段原样展示，不验证其工程结论，不做语义归并或改写。",
        "主审裁决不代表本轮复核完成；复核状态按本轮单独记录列示。未裁决、待复核和驳回项目均保留。", "",
    ]
    for index, item in enumerate(candidates, 1):
        candidate_id = str(item.get("id", ""))
        if candidate_id in reviewed_ids:
            review_status = "已记录本轮复核（模型记录）"
        elif candidate_id in pending_ids:
            review_status = "本轮待复核（尚未记录处理）"
        elif audit:
            review_status = "未见本轮复核记录"
        else:
            review_status = "未提供本轮复核记录"
        lines.extend([f"## 候选项 {index}", "", "候选编号：", *_literal_text_block(candidate_id), "",
                      "本轮复核记录：" + review_status, "",
                      "模型当前裁决：", *_literal_text_block(item.get("decision") or "未记录"), "",
                      "模型初始描述：", *_literal_text_block(item.get("initial_description")), "",
                      "模型当前文本：", *_literal_text_block(item.get("text")), "",
                      "模型裁决理由：", *_literal_text_block(item.get("reason") or "未记录"), ""])
        if audit:
            if candidate_id in inherited_decisions:
                lines.extend(["主审继承裁决：", *_literal_text_block(
                    inherited_decisions.get(candidate_id) or "未记录"), ""])
            if candidate_id in current_decisions:
                lines.extend(["本轮复核裁决：", *_literal_text_block(
                    current_decisions.get(candidate_id) or "未记录"), ""])
        if item.get("revised_conclusion") is not None:
            lines.extend(["模型修订结论：", *_literal_text_block(item["revised_conclusion"]), ""])
    return "\n".join(lines)


def runtime_error_notice(result):
    """Explain service failures from saved codes without altering findings."""
    row = result.to_dict() if hasattr(result, "to_dict") else result
    if row.get("stop_reason") != "model_error":
        return ""
    error = row.get("runtime", {}).get("last_model_error") or {}
    codes = {str(code).lower() for code in error.get("provider_codes", [])}
    if codes & {"insufficient_quota", "insufficient_balance"}:
        reason = "模型服务额度或余额不足，请恢复额度或切换可用服务通道后重试"
    elif error.get("http_status") == 429 or error.get("code") == "quota_or_rate_limit":
        reason = "模型服务的配额或请求频率受限，请检查服务额度与限流状态后重试"
    elif error.get("code") == "authentication_failed":
        reason = "模型服务鉴权失败，请检查该通道的密钥和权限"
    elif error.get("code") == "timeout":
        reason = "模型服务请求超时，可稍后重试"
    elif error.get("code") == "stream_incomplete":
        reason = "模型连接结束但未收到完整结束标记，可稍后重试"
    else:
        reason = "模型服务调用失败，请检查服务连接及运行记录"
    return "运行中断：" + reason + "。本轮未完成，已取得资料保留。"



def _page_location(source):
    filename = str(source.get("filename") or "来源文件")
    page = source.get("page", "未记录")
    numbering = ("系统排版第" if source.get("page_numbering") == "system_rendered_page"
                  or filename.lower().endswith((".doc", ".docx")) else "物理第")
    total = source.get("page_count")
    return f"{filename} · {numbering}{page}页" + (f" / 共{total}页" if total else "")


def _region_location(source):
    try:
        region = normalize_region(source.get("region"))
    except ValueError:
        return " · 局部区域（位置未核实）"
    if region is None:
        return ""
    return " · 局部区域[" + ", ".join(f"{coordinate:.2%}" for coordinate in region) + "]（显示页左、上、右、下）"


def _runtime_source_lines(row):
    """Export source provenance only; image delivery/search results do not certify their claims."""
    runtime = row.get("runtime") or {}
    lines = []
    pages = runtime.get("page_sources") or []
    visual = runtime.get("visual_review", runtime.get("visual_review_ledger"))
    visual = visual if isinstance(visual, dict) else {}
    images = visual.get("images")
    if isinstance(images, list) and images:
        pages = [source for source in images if isinstance(source, dict)]
    directory = runtime.get("visual_page_index") or {}
    if directory.get("ok"):
        candidate_count = directory.get("visual_candidate_count")
        provided = sorted({p["page"] for p in pages if p.get("source_ref") == "plan:current"
                           and p.get("sent_to_model") is True and isinstance(p.get("page"), int)})
        count_text = str(candidate_count) if isinstance(candidate_count, int) else "未统计"
        lines.extend([f"图表候选目录：{count_text}页；本轮实际提供方案页图{len(provided)}页。"
                      "候选目录与图像发送均不代表已完成全部图纸核验。", ""])
    if visual:
        automatic = [source for source in pages if source.get("automatic") is True]
        confirmed = [source for source in automatic if source.get("response_received") is True]
        lines.extend([f"首批自动附图：已取得{len(automatic)}个页图视图；其中{len(confirmed)}个随成功返回的模型请求提供。"
                      "自动选图按关键内容与预算取舍，不表示全部图表已核查。", ""])
        attempts = visual.get("attempts") or []
        unconfirmed = [attempt for attempt in attempts if isinstance(attempt, dict)
                       and attempt.get("response_received") is not True]
        if unconfirmed:
            lines.extend([f"另有{len(unconfirmed)}次带图请求未确认成功返回；请求尝试不证明服务端收到图片，也不证明模型已经判读。", ""])
    if pages:
        lines.extend(["## 本轮页图来源", "",
                      "页图可能由首批自动附入或模型按需读取；提供图片、成功返回和引用均不证明识别准确或整页全部核查，图像识别引文未逐字核验，也不计入可读文字的完整提供范围。DOC和DOCX页码以本次系统排版为准。", ""])
        for source in pages:
            identifier = source.get("evidence_id") or source.get("citation") or source.get("source_ref") or "未编号"
            if source.get("response_received") is True:
                delivery = "已提供给模型（请求成功返回）"
            elif source.get("request_attempted") is True:
                delivery = "已尝试请求，投递未确认"
            elif source.get("acquired") is True:
                delivery = "已取得，尚未进入模型请求"
            else:
                delivery = "已发送给模型" if source.get("sent_to_model") is True else "未确认发送给模型"
            origin = (" · 首批自动附图" if source.get("automatic") is True else
                      " · 模型按需读取" if "automatic" in source else "")
            citation = (" · 模型明确引用（不代表识别已验证）" if source.get("cited") is True else "")
            lines.append(f"- {identifier} · {_page_location(source)}{_region_location(source)} · 来源标识：{source.get('source_ref') or '未记录'} · {delivery}{origin}{citation}")
        lines.append("")
    failures = visual.get("read_failures") or []
    if failures:
        lines.extend(["## 页图读取未完成", ""])
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            lines.append(f"- 来源{failure.get('source_ref') or '未记录'} · 第{failure.get('page') or '未记录'}页"
                         f"{_region_location(failure)} · 读取失败（{failure.get('error_type') or 'ReadError'}）；相关内容仍待核实。")
        lines.append("")
    web = runtime.get("web_sources") or []
    if web:
        lines.extend(["## 本轮外部网页参考", "",
                      "以下是搜索服务返回的摘要与链接；未自动入库，也未逐字核验网页全文或确认规范现行状态。", ""])
        for source in web:
            identifier = source.get("citation") or source.get("display_id") or source.get("evidence_id") or "未编号"
            title = " ".join(str(source.get("title") or "网页来源").split())
            title = title.replace("[", "\\[").replace("]", "\\]")
            url = str(source.get("url") or "")
            label = f"[{title}](<{url}>)" if url.startswith(("https://", "http://")) else title
            lines.append(f"- {identifier} {label}")
            if source.get("evidence_id") and source.get("evidence_id") != identifier:
                lines.append("  来源编号：" + str(source["evidence_id"]))
            if source.get("snippet"):
                lines.extend(["", "> " + str(source["snippet"]).replace("\n", "\n> "), ""])
        lines.append("")
    return lines

def _unparsed_lines(row):
    values = (row.get("runtime") or {}).get("unparsed_output", [])
    values = [value for value in values if value != row.get("answer")]
    if not values:
        return []
    lines = ["## 未结构化原稿", "", "以下内容未通过格式检查，原样保留；程序未补造判断。", ""]
    for value in values:
        body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        fence = "`" * max(3, max(map(len, re.findall(r"`+", body)), default=0) + 1)
        lines.extend([fence, body, fence, ""])
    return lines


def question_markdown(row):
    lines = [row["answer"], ""]
    refs = (row.get("runtime") or {}).get("answer_plan_evidence", [])
    if refs:
        lines.extend(["### 方案依据", ""])
        for ref in list(_unique_plan_refs(refs))[:_VISIBLE_REFERENCES]:
            lines.extend([_plan_location(ref), "", "> " + _excerpt(ref["quote"]).replace("\n", "\n> "), ""])
    warnings = list(dict.fromkeys(row.get("limitations", []) + row.get("validation_notes", [])))
    if warnings:
        lines.extend(["### 说明", "", *["- " + warning for warning in warnings], ""])
    lines.extend(_unparsed_lines(row))
    return "\n".join(lines)


def report_markdown(result):
    row = result.to_dict() if hasattr(result, "to_dict") else result
    if (row.get("runtime") or {}).get("report_format") == "markdown":
        # The model writes the report. The program adds only observed run data,
        # without guessing how many professional findings the prose contains.
        lines = [row["answer"], ""]
        notice = runtime_error_notice(row)
        if notice:
            lines.extend([notice, ""])
        coverage = row.get("source_coverage") or {}
        lines.extend(["---", "", "运行记录（程序生成）：", "",
            f"- 模型调用 {row.get('model_calls', 0)} 次；工具尝试 {row.get('tool_calls', 0)} 次。"])
        if (row.get("runtime") or {}).get("review_mode") == "review":
            lines.append(f"- 可读文本提供范围：{coverage.get('provided_units', 0)}/{coverage.get('total_units', 0)}个单元；此统计不代表工程问题全部发现。")
        warnings = list(dict.fromkeys(row.get("limitations", []) + row.get("validation_notes", [])))
        if warnings:
            lines.extend(["", "资料与运行说明：", *[f"- {item}" for item in warnings]])
        explicit = set(row.get("evidence_ids") or [])
        evidence = {item["evidence_id"]: item for item in row.get("evidence", [])
                    if isinstance(item, dict) and item.get("evidence_id")}
        for title, selected in (
            ("模型明确标注的来源", [item for eid, item in evidence.items() if eid in explicit]),
            ("本轮查阅资料（不等于模型逐条引用）", [item for eid, item in evidence.items() if eid not in explicit]),
        ):
            if selected:
                lines.extend(["", "## " + title, ""])
            for item in selected:
                label = " · ".join(str(item[k]) for k in ("std_no", "title", "clause_no") if item.get(k))
                lines.append(f"- {label or item['evidence_id']}（{item['evidence_id']}）")
        lines.extend(["", *_runtime_source_lines(row)])
        audit = (row.get("runtime") or {}).get("audit_log") or {}
        if audit.get("run_id"):
            lines.extend(["", "运行日志编号：" + str(audit["run_id"])])
        return "\n".join(lines)
    if (row.get("runtime") or {}).get("review_mode") == "question":
        return question_markdown(row)
    coverage = row["source_coverage"]
    notice = runtime_error_notice(row)
    format_warning = (
        "本轮模型回复未通过本地格式校验；以下保留原始可读文本供阅读，未视为已验证报告。"
        if row.get("stop_reason") in {"invalid_result", "invalid_tool_call"}
        and isinstance(row.get("raw_final_text"), str) and row["raw_final_text"].strip()
        else "")
    lines = [f"# {row['document']['filename']}：方案资料核查",
             "", f"核验日期：{row['verification_date']}；模型：{row.get('model') or '未记录'}",
             "", "依据本次收录规范和登记日期提出资料核验结果，不是工程审批或现场安全结论。",
             "", "## 本次回应", "", *([notice, ""] if notice else []),
             *([format_warning, ""] if format_warning else []), row["answer"], "",
             f"原文提供范围：完整提供{coverage['provided_units']}/{coverage['total_units']}个可读单元，"
             f"另有{coverage.get('partially_provided_units', 0)}个单元仅提供部分文字。"
             "提供范围不代表逐句审阅或全部问题均已发现。",
             "", "## 发现与依据", ""]
    summary = finding_summary_text(row)
    if summary:
        lines.extend([summary, ""])
    if row["findings"]:
        lines.extend(["每项按模型排列保留至多3处方案引文和3条规范节选，其余来源简列；"
                      "完整引文、依据及引用检查见JSON，重复规范不再全文展开。", ""])
    evidence = {e["evidence_id"]: e for e in row.get("evidence", [])}
    shown_standards = set()
    for finding in row["findings"]:
        lines.extend([f"### {finding['finding_id']} {finding['title']}", "",
                      f"模型结论：**{finding['verdict']}**", "", finding["description"], ""])
        check = finding["citation_check"]
        if check["issues"]:
            lines.extend(["本项引用仍有待核对事项；以下区分来源定位、正文、校勘及关联图表状态，保留模型原判断：",
                          *[f"- {issue}" for issue in check["issues"]], ""])
        plan_refs = list(_unique_plan_refs(finding["plan_evidence"]))
        for ref in plan_refs[:_VISIBLE_REFERENCES]:
            location = "方案位置：" + _plan_location(ref)
            if ref.get("evidence_kind") == "page_image":
                location = "方案页图：" + _plan_location(ref)
                lines.extend([location, "", "图像识别引文（模型识别，未逐字核验）：", "",
                              "> " + _excerpt(ref["quote"]).replace("\n", "\n> "), ""])
            else:
                lines.extend([location, "", "> " + _excerpt(ref["quote"]).replace("\n", "\n> "), ""])
        other_locations = list(dict.fromkeys(
            _plan_location(ref)
            for ref in plan_refs[_VISIBLE_REFERENCES:]))
        if other_locations:
            lines.extend(["其他方案位置：" + "；".join(other_locations) + "。完整引文见JSON。", ""])
        standard_ids = list(dict.fromkeys(finding["standard_evidence_ids"]))
        for eid in standard_ids[:_VISIBLE_REFERENCES]:
            rec = evidence.get(eid)
            kind = rec.get("kind") if rec else None
            label = {"page_image": "规范页图依据：", "web": "外部网页参考："}.get(kind, "规范依据：")
            lines.append(label + eid + ("（本轮未取得）" if rec is None else ""))
            if rec:
                if eid in shown_standards:
                    lines.extend(["（节选及来源已列于前文，完整内容见JSON。）", ""])
                    continue
                shown_standards.add(eid)
                if kind == "page_image":
                    lines.extend(["", "图像识别内容未逐字核验；这里只确认取得对应页图。"])
                elif kind == "web":
                    lines.extend(["", "搜索服务返回的外部参考，未作为已逐字核验的本地规范正文。"])
                lines.extend(["", _excerpt(rec.get("text", "")), ""])
                sources = list(dict.fromkeys(
                    str(span.get("source_path") or span.get("source_url") or span.get("source_id") or "") +
                    " · " + locator_text(span) for span in rec.get("source_spans", [])))
                if sources:
                    lines.append("来源：" + "；".join(sources))
            lines.append("")
        if standard_ids[_VISIBLE_REFERENCES:]:
            lines.extend(["其他关联依据：" + "；".join(
                eid + ("（本轮未取得）" if eid not in evidence else "")
                for eid in standard_ids[_VISIBLE_REFERENCES:]) + "。完整内容见JSON。", ""])
        if finding["conditions"]:
            lines.extend(["适用条件：", *[f"- {x}" for x in finding["conditions"]], ""])
        if finding["recommendation"]:
            lines.extend(["建议：" + finding["recommendation"], ""])
    if not row["findings"]:
        lines.extend(["本轮未形成独立发现条目；不据此判断整份方案通过。", ""])
    lines.extend(_runtime_source_lines(row))
    lines.extend(_unparsed_lines(row))
    lines.extend(["## 范围与待补证", ""])
    lines.extend(f"- {x}" for x in row.get("limitations", []) + row.get("validation_notes", []))
    if coverage.get("unread_pages"):
        lines.append("- 未取得文字的物理页：" + "、".join(map(str, coverage["unread_pages"])))
    lines.extend(["", "引用检查核对的是实际取得的内容和原文位置，不代替语义、适用条件或工程判断复核。", "",
                  f"模型调用{row['model_calls']}次，工具调用{row['tool_calls']}次；结束状态：{row['status']}。", ""])
    return "\n".join(lines)
