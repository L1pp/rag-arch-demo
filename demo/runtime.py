"""Offline-first public demo runtime around the copied project agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import blake2b
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_arch.evidence_index import EvidenceIndex
from rag_arch.graph_agent import StandardAgent
from rag_arch.plan_documents import PlanDocument, load_uploaded_document
from rag_arch.plan_review_agent import PlanReviewAgent
from rag_arch.plan_review_report import report_markdown
from rag_arch.reranking import RerankResult
from rag_arch.standard_status import StandardStatusRegistry
from rag_arch.standard_tools import StandardTools
from rag_arch.standards import StandardCorpus
from rag_arch.retrieval import tokenize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STANDARDS = ROOT / "examples" / "standards.json"
DEFAULT_PLAN = ROOT / "examples" / "demo_plan.json"
OFFLINE_NOTICE = (
    "离线脚本演示：向量由本地特征映射生成，重排由词项重合脚本完成，"
    "ScriptedChat 按固定规则返回；没有真实模型推理或网络请求。"
)
PUBLIC_MATERIAL_NOTICE = "演示中的资料、方案、标准编号及数值均为自编虚构内容。"


class FeatureVectorEmbeddings:
    """Small deterministic local feature vectors; not a trained embedding model."""

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        for token in tokenize(text):
            raw = blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(raw[:4], "little") % self.dimensions
            sign = 1.0 if raw[4] & 1 else -1.0
            vector[slot] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LexicalReranker:
    """Deterministic overlap scoring used only to make the offline flow runnable."""

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        query_terms = set(tokenize(query))
        ranked = []
        for index, document in enumerate(documents):
            terms = set(tokenize(document))
            score = len(query_terms & terms) / max(1, len(query_terms))
            ranked.append(RerankResult(index=index, score=float(score)))
        ranked.sort(key=lambda item: (-item.score, item.index))
        return ranked[:top_n] if top_n is not None else ranked


@dataclass
class DemoRuntime:
    corpus: StandardCorpus
    index: EvidenceIndex
    tools: StandardTools
    mode: str
    settings: Any = None


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("演示输入文件必须是JSON对象。")
    return value


def _build_corpus(source: dict, source_label: str = "examples/standards.json") -> StandardCorpus:
    meta = source.get("standard") or {}
    std_id = str(meta.get("std_id") or "DEMO-STRUCT-2026")
    std_no = str(meta.get("std_no") or std_id)
    title = str(meta.get("title") or "虚构演示资料")
    clauses = []
    for row in source.get("clauses", []):
        clause_no = str(row["clause_no"])
        clauses.append({
            "clause_id": str(row.get("evidence_id") or f"{std_id}/{clause_no}"),
            "std_id": std_id,
            "std_no": std_no,
            "clause_no": clause_no,
            "title": str(row.get("title") or "演示条文"),
            "text": str(row["text"]),
            "heading_path": [title],
            "document_part": "body",
            "record_kind": "clause",
            "content_status": "text_available",
            "content_readable": True,
            "source_spans": [{
                "source_id": "DEMO-STANDARD",
                "source_path": source_label,
                "page_start": 1,
                "page_end": 1,
            }],
        })
    catalog = ([{
        "std_id": std_id,
        "std_no": std_no,
        "title": title,
        "library_class": "A",
        "entry_kind": "local_text",
        "text_available": bool(clauses),
        "record_count": len(clauses),
        "source_files": [],
    }] if meta else [])
    return StandardCorpus(
        clauses=clauses,
        tables=[],
        summary={"generated_at": "public-demo-v1", "limitations": [PUBLIC_MATERIAL_NOTICE]},
        catalog=catalog,
        relations=[],
    )


def build_runtime(
    mode: str = "offline",
    *,
    standards_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> DemoRuntime:
    """Build the offline runtime by default. Live construction is explicit."""
    if mode not in {"offline", "live"}:
        raise ValueError("mode 只能是 offline 或 live。")
    standards_file = Path(standards_path) if standards_path else DEFAULT_STANDARDS
    standards_data = _read_json(standards_file)
    try:
        source_label = standards_file.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        source_label = "自定义演示资料"
    corpus = _build_corpus(standards_data, source_label)
    settings = None
    if mode == "offline":
        embeddings = FeatureVectorEmbeddings()
        reranker = LexicalReranker()
    else:
        from rag_arch.config import load_settings
        from rag_arch.providers import create_embeddings, create_reranker

        # Read only the public demo's own .env; do not inherit RAG_* process state.
        settings = load_settings(env_path=Path(config_path) if config_path else ROOT / ".env", environ={})
        if not settings.chat_ready or not settings.embedding_ready or not settings.reranker_ready:
            raise ValueError("live 模式需要在演示仓库自己的 .env 中配置 Chat、Embedding 与 Reranker。")
        if settings.reranker_provider != "bailian":
            raise ValueError("live 演示要求使用云端 Reranker；当前配置未选择 bailian。")
        embeddings = create_embeddings(settings)
        reranker = create_reranker(settings)
        if embeddings is None or reranker is None:
            raise ValueError("live 模式未能构造云端 Embedding 或 Reranker 客户端。")
    index = EvidenceIndex(corpus, embeddings=embeddings, reranker=reranker)
    registry = StandardStatusRegistry(
        project_root=ROOT,
        clauses=corpus.clauses,
        catalog=corpus.catalog,
        relations=corpus.relations,
    )
    tools = StandardTools(index, registry, default_mode="hybrid", top_k=6)
    return DemoRuntime(corpus=corpus, index=index, tools=tools, mode=mode, settings=settings)


def _chat(runtime: DemoRuntime, workflow: str):
    if runtime.mode == "offline":
        return ScriptedChat(workflow=workflow)
    from rag_arch.tool_chat import create_tool_chat_model

    return create_tool_chat_model(runtime.settings, json_output=True)


def run_standard_question(runtime: DemoRuntime, question: str):
    agent = StandardAgent(_chat(runtime, "qa"), runtime.tools, max_steps=4, max_tool_calls=4)
    return agent.run(question, corpus_id="public-demo-v1")


def make_plan_document(plan_data: dict | None = None, *, plan_path: str | Path | None = None,
                       text: str | None = None) -> PlanDocument:
    source = plan_data if plan_data is not None else _read_json(plan_path or DEFAULT_PLAN)
    document_id = str(source.get("document_id") or "DEMO-PLAN-001")
    filename = str(source.get("filename") or "演示方案.txt")
    authored_units = [dict(unit) for unit in source.get("units", [])]
    if text is not None:
        if authored_units:
            authored_units[0]["text"] = text
            authored_units = authored_units[:1]
        else:
            authored_units = [{"unit_id": f"{document_id}/u1", "text": text,
                               "locator": {"line_start": 1, "line_end": 1}}]
    physical_lines = []
    for unit in authored_units:
        body = unit.get("text")
        if not isinstance(body, str):
            continue
        physical_lines.extend((unit, line) for line in body.splitlines() if line.strip())
    if not physical_lines:
        return PlanDocument(document_id, filename, [], source_kind="txt",
                            warnings=[PUBLIC_MATERIAL_NOTICE])

    # Pass the authored text through the same in-memory text loader used for uploads.
    text_bytes = "\n".join(line for _, line in physical_lines).encode("utf-8")
    loaded = load_uploaded_document(filename, text_bytes, document_id=document_id)
    if len(loaded.units) != len(physical_lines):
        raise RuntimeError("样例文字经资料处理后行数不一致。")
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for unit, _ in physical_lines:
        base = str(unit.get("unit_id") or f"{document_id}/u{len(totals) + 1}")
        totals[base] = totals.get(base, 0) + 1
    processed_units = []
    for index, (loaded_unit, authored) in enumerate(zip(loaded.units, physical_lines), 1):
        source_unit, _ = authored
        base = str(source_unit.get("unit_id") or f"{document_id}/u{index}")
        counts[base] = counts.get(base, 0) + 1
        unit_id = base if totals[base] == 1 else f"{base}/line-{counts[base]}"
        locator = dict(loaded_unit["locator"])
        original_locator = source_unit.get("locator")
        if isinstance(original_locator, dict):
            locator["demo_input_locator"] = dict(original_locator)
        processed_units.append({
            "unit_id": unit_id,
            "plan_id": document_id,
            "text": loaded_unit["text"],
            "locator": locator,
            "source_kind": loaded_unit.get("source_kind", "txt"),
            "source_path": "examples/demo_plan.json",
        })
    return PlanDocument(
        document_id=document_id,
        filename=loaded.filename,
        units=processed_units,
        source_kind=loaded.source_kind,
        warnings=[PUBLIC_MATERIAL_NOTICE],
    )

def run_plan_review(runtime: DemoRuntime, plan: PlanDocument | None = None):
    document = plan or make_plan_document()
    agent = PlanReviewAgent(
        _chat(runtime, "plan"),
        document,
        runtime.tools,
        max_steps=4,
        max_tool_calls=6,
        initial_plan_chars=6000,
        context_budget_tokens=64000,
        allow_context_compaction=False,
        mode="review",
        report_format="structured",
    )
    result = agent.run("请用本次演示资料展示方案原文定位、规范检索和证据引用。")
    result.runtime["execution_mode"] = runtime.mode
    result.runtime["chat_adapter"] = "ScriptedChat" if runtime.mode == "offline" else "configured provider"
    return result


def render_plan_report(result) -> str:
    """Use the original renderer while making offline call counts truthful."""
    row = result.to_dict() if callable(getattr(result, "to_dict", None)) else result_to_dict(result)
    offline = (row.get("runtime") or {}).get("execution_mode") == "offline"
    if offline:
        scripted_turns = row.get("model_calls", 0)
        row["model"] = "ScriptedChat（离线脚本替身）"
        row["model_calls"] = 0
        row.setdefault("runtime", {})["scripted_decision_turns"] = scripted_turns
        row["runtime"]["execution_mode"] = "offline"
    body = report_markdown(row)
    if offline:
        return f"> {OFFLINE_NOTICE}\n> {PUBLIC_MATERIAL_NOTICE}\n\n{body}\n"
    return body


def result_to_dict(result: Any) -> dict:
    if callable(getattr(result, "to_dict", None)):
        return result.to_dict()
    if is_dataclass(result):
        return asdict(result)
    raise TypeError("演示结果无法序列化为对象。")


def run_offline_demo(
    output: str | Path,
    *,
    standards_path: str | Path | None = None,
    plan_path: str | Path | None = None,
    mode: str = "offline",
) -> dict:
    runtime = build_runtime(mode, standards_path=standards_path)
    question = "虚构演示条文中，支撑间距的上限是多少？"
    answer = run_standard_question(runtime, question)
    plan = make_plan_document(plan_path=plan_path)
    review = run_plan_review(runtime, plan)
    report = render_plan_report(review)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.md").write_text(report, encoding="utf-8")
    (destination / "standard_answer.json").write_text(
        json.dumps({"execution_mode": mode, "question": question,
                    "network_model_calls": 0 if mode == "offline" else answer.model_calls,
                    "scripted_chat_turns": answer.model_calls if mode == "offline" else None,
                    "result": result_to_dict(answer)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "plan_review.json").write_text(
        json.dumps({"execution_mode": mode,
                    "network_model_calls": 0 if mode == "offline" else review.model_calls,
                    "scripted_chat_turns": review.model_calls if mode == "offline" else None,
                    "result": result_to_dict(review)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"runtime": runtime, "answer": answer, "review": review, "report": report,
            "output_dir": destination}


class ScriptedChat:
    """A deterministic workflow substitute; it does not infer or call a model."""

    model_name = "ScriptedChat-local-demo"
    max_tokens = None
    request_timeout = 1

    def __init__(self, workflow: str):
        if workflow not in {"qa", "plan"}:
            raise ValueError("未知离线脚本流程。")
        self.workflow = workflow
        self._call_number = 0
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def _call(self, name: str, args: dict) -> AIMessage:
        self._call_number += 1
        return AIMessage(content="", tool_calls=[{
            "name": name,
            "args": args,
            "id": f"script-call-{self._call_number}",
        }])

    @staticmethod
    def _payload(message: ToolMessage) -> dict:
        try:
            value = json.loads(message.content)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _plan_source_unit(messages) -> dict | None:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            units = (getattr(message, "additional_kwargs", None) or {}).get("plan_source_units")
            if isinstance(units, list):
                return next((unit for unit in units
                             if isinstance(unit, dict) and unit.get("complete_unit") is True
                             and isinstance(unit.get("text"), str) and unit["text"].strip()), None)
        return None

    def invoke(self, messages, **kwargs) -> AIMessage:
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if self.workflow == "qa":
            prior = next((message for message in reversed(tool_messages) if message.name == "search"), None)
            if prior is None:
                question = next((str(message.content) for message in reversed(messages)
                                 if isinstance(message, HumanMessage)), "演示问题")
                return self._call("search", {"query": question, "mode": "hybrid", "scope": "evidence"})
            payload = self._payload(prior)
            record = _eligible_evidence(payload)
            if record:
                evidence_id = record["evidence_id"]
                text = str(record.get("text") or "样例证据已取得。")
                answer = f"离线脚本从虚构演示资料中取得：{text} [{evidence_id}]"
                final = {"status": "answered", "answer": answer, "evidence_ids": [evidence_id]}
            else:
                final = {
                    "status": "insufficient_evidence",
                    "answer": "虚构演示资料中没有取得可引用的完整证据；本轮不补造条文或引用。",
                    "evidence_ids": [],
                }
            return AIMessage(content=json.dumps(final, ensure_ascii=False))

        plan_message = next((message for message in reversed(tool_messages)
                             if message.name == "search_plan"), None)
        standard_message = next((message for message in reversed(tool_messages)
                                 if message.name == "search"), None)
        plan_unit = self._plan_source_unit(messages)
        if plan_message is None:
            return self._call("search_plan", {"query": "支撑间距", "limit": 3})
        if standard_message is None:
            if plan_unit is None:
                return AIMessage(content=json.dumps({
                    "answer": "方案原文没有完整提供可引用单元，本轮不生成方案发现。",
                    "findings": [], "evidence_ids": [],
                    "limitations": ["没有取得完整方案原文单元。"],
                    "reviewed_topics": [],
                }, ensure_ascii=False))
            return self._call("search", {
                "query": ("支撑间距" if "支撑间距" in plan_unit["text"] else plan_unit["text"]),
                "mode": "hybrid", "scope": "evidence"})

        standard_payload = self._payload(standard_message)
        record = _eligible_evidence(standard_payload)
        if plan_unit is None:
            findings = []
            evidence_ids = []
            limitations = ["没有取得完整方案原文单元，未生成方案定位引用。"]
            answer = "当前脚本没有取得可核对的方案原文。"
        else:
            quote = str(plan_unit["text"]).strip()
            plan_ref = {"unit_id": plan_unit["unit_id"], "quote": quote}
            if record:
                evidence_id = record["evidence_id"]
                excerpt = str(record.get("text") or "")[:260]
                findings = [{
                    "title": "虚构示例资料对照",
                    "category": "standard_comparison",
                    "verdict": "需要补充资料",
                    "description": (f"脚本展示方案原文「{quote}」与样例条文「{excerpt}」的关联。"
                                    "本结果由预设演示逻辑生成，没有执行模型推理或工程合规判断。"),
                    "plan_evidence": [plan_ref],
                    "standard_evidence_ids": [evidence_id],
                    "conditions": [PUBLIC_MATERIAL_NOTICE],
                    "recommendation": "将本项仅用于查看检索和引用链路；真实工程结论须另行专业复核。",
                }]
                evidence_ids = [evidence_id]
                limitations = [OFFLINE_NOTICE, PUBLIC_MATERIAL_NOTICE]
                answer = "脚本已完成样例方案定位、虚构条文检索和引用校验展示。"
            else:
                findings = [{
                    "title": "未取得虚构规范依据",
                    "category": "source_gap",
                    "verdict": "需要补充资料",
                    "description": "脚本没有从样例规范资料中取得可用条文，因此不生成规范结论。",
                    "plan_evidence": [plan_ref],
                    "standard_evidence_ids": [],
                    "conditions": [PUBLIC_MATERIAL_NOTICE],
                    "recommendation": "补充可引用的演示条文后再运行；不得将本轮空缺解释为工程缺项。",
                }]
                evidence_ids = []
                limitations = ["没有取得可引用的规范证据；未伪造证据编号。", PUBLIC_MATERIAL_NOTICE]
                answer = "方案原文已定位，但演示资料中没有可引用规范证据。"
        final = {
            "answer": answer,
            "findings": findings,
            "evidence_ids": evidence_ids,
            "limitations": limitations,
            "reviewed_topics": ["样例原文定位", "虚构资料检索"],
        }
        return AIMessage(content=json.dumps(final, ensure_ascii=False))

def _eligible_evidence(payload: dict) -> dict | None:
    complete_ids = payload.get("complete_context_ids")
    complete_ids = set(complete_ids) if isinstance(complete_ids, list) else set()
    records = payload.get("evidence")
    if not isinstance(records, list):
        return None
    for record in records:
        if (isinstance(record, dict) and record.get("evidence_id") in complete_ids
                and record.get("complete") is True):
            return record
    return None