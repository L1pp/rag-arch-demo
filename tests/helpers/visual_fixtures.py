"""Minimal self-authored fakes for public visual review tests."""
import asyncio
from copy import deepcopy
from io import BytesIO
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from PIL import Image

from rag_arch.config import Settings
from rag_arch.plan_documents import PlanDocument


def call(name="get_clause", id="call-1", args=None):
    return {"name": name, "id": id, "args": {} if args is None else args, "type": "tool_call"}


def action(*calls):
    return AIMessage(content="", tool_calls=list(calls),
                     usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def document(body="虚构构件间距为1.5m。", units=None, **kwargs):
    return PlanDocument("DEMO-PLAN", "虚构方案.txt", units or [{
        "unit_id": "DEMO-PLAN/u1", "plan_id": "DEMO-PLAN", "text": body,
        "locator": {"line_start": 1, "line_end": 1}, "source_path": "fictional.txt",
        "candidate_scope": False}], **kwargs)


def final(findings=None, **kwargs):
    return AIMessage(content=json.dumps({"answer": "本轮得到一项待复核的示例发现。",
        "findings": [] if findings is None else findings, "limitations": [],
        "reviewed_topics": ["示例参数"], "evidence_ids": [], **kwargs}, ensure_ascii=False),
        usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})


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


class FakeTools:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [{"ok": True, "status": "ok", "found": True,
                                        "raw_ids": [], "evidence": [],
                                        "complete_context_ids": [],
                                        "context_complete": True, "missing": []}])
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def as_langchain_tools(self):
        def get_clause(std_no: str, clause_no: str):
            return self.dispatch("get_clause", {"std_no": std_no, "clause_no": clause_no})

        def search(query: str, filters: dict | str | None = None):
            args = {"query": query}
            if filters is not None:
                args["filters"] = filters
            return self.dispatch("search", args)

        def check_standard_status(std_no: str, clause_no: str | None = None):
            return self.dispatch("check_standard_status", {"std_no": std_no, "clause_no": clause_no})

        return [StructuredTool.from_function(f, name=f.__name__, description=f.__name__)
                for f in (get_clause, search, check_standard_status)]


class Pages:
    collection_id = "test-plan-pages"

    def sources(self):
        return [{"source_ref": "plan:current", "filename": "fictional.pdf", "page_count": 1}]

    def list_pages(self, source_ref, start=1, limit=20, query="", visual_only=False):
        return {"ok": True, "source_ref": source_ref, "page_count": 1,
                "pages": [{"page": 1, "text_preview": ""}]}

    def read_page(self, source_ref, page):
        stream = BytesIO()
        Image.new("RGB", (120, 100), "blue").save(stream, format="PNG")
        return {"ok": True, "source_ref": source_ref, "filename": "fictional.pdf",
                "page": page, "page_count": 1, "text": "", "image_bytes": stream.getvalue(),
                "mime_type": "image/png", "width": 120, "height": 100,
                "source_kind": "plan", "collection_id": self.collection_id}


class AsyncOnlyModel:
    native_tool_names = ()

    def __init__(self, *responses, delay=0):
        self.responses = list(responses)
        self.calls = []
        self.delay = delay
        self.bound_schemas = []

    def bind_tools(self, schemas):
        self.bound_schemas.append(schemas)
        return self

    def invoke(self, _messages):
        raise AssertionError("async QA called a synchronous model")

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response_settings():
    return Settings(chat_model="qwen3.8-flash", chat_api_key="synthetic-responses-secret",
                    chat_base_url="https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                    chat_max_tokens=8192, chat_thinking_budget=4096)


def payload(output):
    return {"id": "response-1", "object": "response", "model": "qwen3.8-flash", "status": "completed",
            "output": output, "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300,
                "input_tokens_details": {"cached_tokens": 42}, "output_tokens_details": {"reasoning_tokens": 50}}}


def text_output(text="完成"):
    return {"type": "message", "id": "msg-1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
