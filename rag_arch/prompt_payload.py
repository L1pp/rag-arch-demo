"""Lossless, response-local sharing of repeated standard status metadata."""
from __future__ import annotations

from copy import deepcopy
import json


PAYLOAD_VARIANT = "standard-status-shared-json-v1"
_TABLE = "shared_status_metadata"
_REF = "shared_metadata_ref"
_FIELDS = ("source_ids", "source_urls", "note", "full_status_tool")


def _status_records(payload):
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [record["status_info"] for record in evidence
            if isinstance(record, dict) and isinstance(record.get("status_info"), dict)]


def _json_value(value):
    """Reject non-JSON values rather than sharing two unequal Python objects."""
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("共享状态字段须为JSON值，不能转换其他Python对象。")


def _metadata_key(metadata):
    _json_value(metadata)
    try:
        # Content comparison only: no hash. In particular 1, 1.0 and true are
        # distinct JSON literals even though Python dictionary equality is not.
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("共享状态字段须为有限、可序列化的JSON值。") from exc


def _copy_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("规范工具payload须为JSON对象。")
    result = deepcopy(payload)
    if isinstance(payload.get("evidence"), list):
        # A Python caller may reuse a record object at two JSON array positions.
        # Give each position its own copy before moving or restoring fields.
        result["evidence"] = [deepcopy(record) for record in payload["evidence"]]
    return result


def compact_standard_payload(payload):
    """Share exact duplicate metadata groups occurring at least twice.

    Only the four fields in ``evidence[*].status_info`` may move. Missing keys
    remain distinct from null. Reserved-key collisions fail before any change
    to the caller's object. No repeat means no format change, only a deep copy.
    """
    result = _copy_payload(payload)
    records = _status_records(result)
    if _TABLE in result or any(_REF in record for record in records):
        raise ValueError("payload含共享状态预留字段，不能覆盖或重复压缩。")
    groups = {}
    for status in records:
        metadata = {key: status[key] for key in _FIELDS if key in status}
        if not metadata:
            continue
        key = _metadata_key(metadata)
        if key not in groups:
            groups[key] = (metadata, [])
        groups[key][1].append(status)
    shared = {}
    for metadata, occurrences in groups.values():
        if len(occurrences) < 2:
            continue
        reference = f"s{len(shared) + 1}"
        shared[reference] = deepcopy(metadata)
        for status in occurrences:
            for key in metadata:
                del status[key]
            status[_REF] = reference
    if shared:
        result[_TABLE] = shared
    return result


def expand_standard_payload(payload):
    """Restore metadata on a deep copy; never overwrite a local field.

    Unknown/bad references, malformed metadata, collisions and unused table
    entries raise ValueError rather than silently dropping or guessing data.
    Ordinary uncompressed payloads are copied unchanged.
    """
    result = _copy_payload(payload)
    records = _status_records(result)
    references = [record for record in records if _REF in record]
    if _TABLE not in result:
        if references:
            raise ValueError("共享状态引用缺少响应内元数据表。")
        return result
    shared = result[_TABLE]
    if not isinstance(shared, dict) or not shared:
        raise ValueError("共享状态元数据表须为非空对象。")
    for reference, metadata in shared.items():
        if not isinstance(reference, str) or not reference:
            raise ValueError("共享状态表的引用ID须为非空字符串。")
        if not isinstance(metadata, dict) or not metadata or set(metadata) - set(_FIELDS):
            raise ValueError("共享状态表只能包含指定的非空元数据字段组。")
        _metadata_key(metadata)
    used = set()
    for status in references:
        reference = status[_REF]
        if not isinstance(reference, str) or reference not in shared:
            raise ValueError("共享状态引用无效或未在本次响应中登记。")
        metadata = shared[reference]
        if any(key in status for key in metadata):
            raise ValueError("共享状态字段与证据本地字段碰撞，不能覆盖。")
        del status[_REF]
        status.update(deepcopy(metadata))
        used.add(reference)
    if used != set(shared):
        raise ValueError("共享状态表含未被证据引用的条目，不能静默删除。")
    del result[_TABLE]
    return result
