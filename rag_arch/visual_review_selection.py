"""High-precision, bounded automatic first-page-image selection.

The caller supplies all candidate rows, not only the first directory page.
Uncertain photos, scans, diagrams and title cues remain available for deliberate
tool reads; they never qualify for automatic attachment by themselves.
"""
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy


DEFAULT_INITIAL_VISUAL_PAGES = 1
MAX_INITIAL_VISUAL_PAGES = 2


def select_initial_visual_pages(candidates: Iterable[dict], *, max_pages: int = DEFAULT_INITIAL_VISUAL_PAGES) -> list[dict]:
    """Select at most one numeric table by default, with an absolute cap of two.

    Only explicit grid evidence produced by the index qualifies.  Missing or
    old metadata conservatively produces an empty list.  The caller separately
    limits actual image bytes, visual tokens, rendering time and source access.
    """
    if type(max_pages) is not int or not 0 <= max_pages <= MAX_INITIAL_VISUAL_PAGES:
        raise ValueError("首次自动页图数量必须为0至2的整数。")
    eligible = {}
    for row in candidates:
        if not isinstance(row, dict) or row.get("auto_attach_eligible") is not True:
            continue
        page = row.get("page")
        reasons = row.get("auto_attach_reasons")
        signals = row.get("visual_signals")
        if (type(page) is not int or page < 1 or not isinstance(reasons, list)
                or "populated_numeric_table_grid" not in reasons
                or not isinstance(signals, list) or "table_grid" not in signals):
            continue
        priority = row.get("auto_attach_priority", 0)
        if type(priority) is not int or priority < 0:
            priority = 0
        item = deepcopy(row)
        item["selection_reason"] = "检测到含数字的已填充网格表；保守自动附图，不代表识别已核验。"
        item["auto_attach_priority"] = priority
        if page not in eligible or priority > eligible[page]["auto_attach_priority"]:
            eligible[page] = item
    ranked = sorted(eligible.values(), key=lambda row: (-row["auto_attach_priority"], row["page"]))
    return ranked[:max_pages]
