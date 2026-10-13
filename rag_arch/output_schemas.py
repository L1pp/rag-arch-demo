"""Provider response schemas for the two production Agent outputs.

These constrain syntax and fields only. Citation acquisition and semantic
support remain separate checks; model-supplied verification flags are absent.
"""
from copy import deepcopy
from .config import is_qwen38_model

_STRINGS = {"type": "array", "items": {"type": "string"}}

STANDARD_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
        "answer": {"type": "string"},
        "evidence_ids": deepcopy(_STRINGS),
        "required_evidence_ids": deepcopy(_STRINGS),
        "answer_scope": {"type": "string", "enum": ["evidence", "catalog"],
                         "description": "catalog仅回答目录身份和本地资料可得性；不能支持工程条文结论。"},
    },
    "required": ["status", "answer", "evidence_ids"],
    "additionalProperties": False,
}

_PLAN_QUOTE = {
    "type": "object",
    "properties": {
        "quote": {"type": "string"},
        "source_ref": {"type": "string"},
        "unit_id": {"type": "string"},
    },
    "required": ["quote"],
    "additionalProperties": False,
}
_PLAN_FINDING = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "category": {"type": "string", "enum": ["standard_comparison", "document_consistency", "source_gap"]},
        "verdict": {"type": "string", "enum": ["符合", "不符合", "无法判断", "需要补充资料"]},
        "description": {"type": "string"},
        "plan_evidence": {"type": "array", "items": _PLAN_QUOTE},
        "standard_evidence_ids": deepcopy(_STRINGS),
        "conditions": deepcopy(_STRINGS),
        "recommendation": {"type": "string"},
    },
    "required": ["title", "category", "verdict", "description", "plan_evidence",
                 "standard_evidence_ids", "conditions", "recommendation"],
    "additionalProperties": False,
}
PLAN_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "findings": {"type": "array", "items": _PLAN_FINDING},
        "evidence_ids": {**deepcopy(_STRINGS),
                         "description": "仅列本轮取得的规范证据ID；方案单元ID应放入findings[].plan_evidence，不放在这里。"},
        "limitations": deepcopy(_STRINGS),
        "reviewed_topics": deepcopy(_STRINGS),
        "context_facts": {"type": "array", "description": "多批次审查中供跨章核对的语义事实，可省略；不是程序统计或已确认缺陷。",
            "items": {"type": "object", "properties": {
                "summary": {"type": "string"},
                "plan_evidence": {"type": "array", "items": deepcopy(_PLAN_QUOTE)}},
                "required": ["summary", "plan_evidence"], "additionalProperties": False}},
    },
    "required": ["answer", "findings", "evidence_ids", "limitations", "reviewed_topics"],
    "additionalProperties": False,
}

PLAN_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "plan_evidence": {"type": "array", "items": deepcopy(_PLAN_QUOTE)},
        "evidence_ids": deepcopy(_STRINGS),
        "limitations": deepcopy(_STRINGS),
    },
    "required": ["answer", "plan_evidence", "evidence_ids", "limitations"],
    "additionalProperties": False,
}


def schema_for_model(model_name, schema):
    """Use the verified Qwen3.8 capability; other compatible providers keep JSON Object."""
    return deepcopy(schema) if is_qwen38_model(model_name) else None
