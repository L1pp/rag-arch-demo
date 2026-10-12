"""Ordered, lossless batches of already extracted plan text.

The limit applies to each serialized batch, not to the model context window.
This module does not load files, extract images, or change the source document.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .plan_review_context import estimate_tokens
from .model_input_view import model_plan_payload

if TYPE_CHECKING:
    from .plan_documents import PlanDocument


def plan_text_batches(document: PlanDocument, *, token_counter=None,
                      max_tokens: int = 32000) -> list[dict]:
    """Keep source order and character offsets while fitting whole units first.

    ``token_counter.count(text)`` may supply a local tokenizer. Otherwise the
    existing approximate estimator is used. Counts include the JSON envelope
    serialized with ``json.dumps(batch, ensure_ascii=False)``. Batch indices are
    one-based; a document without units yields no batches. Empty units are kept
    as complete zero-length fragments, matching the source coverage contract.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens 须为正整数，表示每批正文的估算 Token 预算。")
    if not document.units:
        return []
    count = estimate_tokens if token_counter is None else token_counter.count

    def tokens(payload):
        serialized = json.dumps(payload, ensure_ascii=False)
        if token_counter is None:
            size = estimate_tokens(json.dumps(model_plan_payload(payload), ensure_ascii=False, separators=(",", ":")))
        else:
            size = count(serialized)
        if type(size) is not int or size < 0:
            raise ValueError("Token 计数器须返回非负整数。")
        return size

    def pack(budget):
        batches, current = [], []

        def payload(units):
            return {"units": units, "batch_index": len(batches) + 1,
                    "total_batches": 1}

        for pos, unit in enumerate(document.units):
            offset, total = 0, len(unit["text"])
            while True:
                whole = document._fragment(pos, offset, total - offset)
                if tokens(payload([*current, whole])) <= budget:
                    current.append(whole)
                    break
                if current:
                    # Prefer moving an intact unit to a fresh batch over
                    # splitting it just to consume the previous batch's space.
                    batches.append(payload(current))
                    current = []
                    continue
                first = document._fragment(pos, offset, min(1, total - offset))
                if tokens(payload([first])) > budget:
                    raise ValueError(
                        f"方案单元 {unit['unit_id']} 的定位元数据及最小原文片段超过 "
                        f"{max_tokens} Token 批次预算；请提高每批预算。")
                # The whole remainder did not fit; find a positive prefix.
                # Every accepted candidate is counted independently, so an
                # approximate/nonlinear tokenizer cannot create an oversize
                # fragment or prevent progress.
                best = first
                low, high = 2, total - offset - 1
                while low <= high:
                    length = (low + high) // 2
                    candidate = document._fragment(pos, offset, length)
                    if tokens(payload([candidate])) <= budget:
                        best = candidate
                        low = length + 1
                    else:
                        high = length - 1
                batches.append(payload([best]))
                offset = best["text_end"]
        if current:
            batches.append(payload(current))
        return batches

    reserve = 0
    while reserve < max_tokens:
        batches = pack(max_tokens - reserve)
        for batch in batches:
            batch["total_batches"] = len(batches)
        overflow = max(tokens(batch) - max_tokens for batch in batches)
        if overflow <= 0:
            return batches
        # Final batch counts can add digits to the envelope. Repack with room
        # for them and validate the actual labels rather than assuming they
        # cost the same as the placeholder. Growing reserve bounds retries.
        reserve = max(reserve * 2, reserve + overflow)
    raise ValueError("批次元数据超过 Token 预算；请提高每批预算。")
