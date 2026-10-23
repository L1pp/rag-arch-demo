"""Small synthetic objects used by tenant-scoped audit regression tests."""
from __future__ import annotations

from copy import deepcopy
import json

from langchain_core.messages import AIMessage
from rag_arch.standards import StandardCorpus


CLAUSE = "GB99999-2026/4.1.1"
STD = "GB 99999-2026"
TABLE1 = "GB99999-2026/table/4.1.1-1"
TABLE2 = "GB99999-2026/table/4.1.1-2"


def final(ids=None, answer=None, status="answered", required=None):
    ids = [CLAUSE] if ids is None else ids
    answer = "演示条文回答。" + "".join(f"[{item}]" for item in ids) if answer is None else answer
    payload = {"status": status, "answer": answer, "evidence_ids": ids}
    if required is not None:
        payload["required_evidence_ids"] = required
    return AIMessage(
        content=json.dumps(payload, ensure_ascii=False),
        additional_kwargs={"reasoning_content": "synthetic fixture"},
        usage_metadata={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
    )


class FakeChat:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []
        self.bound_tools = []
        self.request_options = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages, **kwargs):
        self.requests.append(deepcopy(messages))
        self.request_options.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def corpus_fixture():
    clauses = [{
        "clause_id": CLAUSE,
        "std_id": "GB99999-2026",
        "std_no": STD,
        "clause_no": "4.1.1",
        "heading_path": ["4 合成规定"],
        "text": "仅用于隔离测试的合成条文。",
        "table_ids": [TABLE1, TABLE2],
        "document_part": "body",
        "source_spans": [{
            "source_id": "SYNTHETIC",
            "source_path": "synthetic.pdf",
            "page_start": 1,
            "page_end": 1,
        }],
    }]
    tables = [{
        "table_id": table_id,
        "std_no": STD,
        "caption": f"合成表格 {index}",
        "markdown": "| 条件 | 数值 |\n|---|---|\n| 测试 | 1 |",
        "notes": ["测试数据，不构成工程依据。"],
        "content_status": "text_available",
        "source_spans": [{
            "source_id": "SYNTHETIC",
            "source_path": "synthetic.pdf",
            "page_start": 1,
            "page_end": 1,
        }],
        "owner_clause_id": CLAUSE,
        "document_part": "body",
    } for index, table_id in enumerate((TABLE1, TABLE2), 1)]
    return StandardCorpus(clauses, tables)


class FakeStatus:
    def __init__(self):
        self.calls = []

    def check(self, std_no, clause_no=None):
        self.calls.append((std_no, clause_no))
        return {
            "found": True,
            "std_no": std_no,
            "clause_no": clause_no,
            "standard_status": "unknown",
            "registered_status": "unknown",
            "original_mandatory": None,
            "current_mandatory_status": "unknown",
            "verification_date": "2026-10-23",
            "source_ids": ["SYNTHETIC"],
            "sources": [],
            "matched_relations": [],
            "related_subclause_relations": [],
            "compliance_conclusion": None,
        }


def reply(text="synthetic answer"):
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    )


class FakeModel:
    max_tokens = 256
    native_tool_names = ()

    def __init__(self, response=None, *, action=None):
        self.response = response if response is not None else reply()
        self.action = action
        self.calls = []
        self.bindings = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.action:
            self.action()
        return self.response

    def bind_tools(self, *args, **kwargs):
        self.bindings.append((args, kwargs))
        child = FakeModel(self.response)
        child.calls = self.calls
        return child

    def plain(self):
        child = FakeModel(self.response)
        child.calls = self.calls
        return child
