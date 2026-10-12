"""Local lookup of explicit standard identifiers, without model/network calls."""
from .standard_identifiers import parse_standard_references


def plan_standard_preflight(document, registry):
    if registry is None or not callable(getattr(registry, "check", None)):
        return []
    references = {}
    for unit in document.units:
        for item in parse_standard_references(unit["text"]):
            identifier = item["normalized"]
            row = references.setdefault(identifier, {"requested": identifier, "plan_unit_ids": []})
            if unit["unit_id"] not in row["plan_unit_ids"]:
                row["plan_unit_ids"].append(unit["unit_id"])
    for row in references.values():
        try:
            row["status_info"] = registry.check(row["requested"])
        except Exception:
            row["status_info"] = {"found": False, "query_std_no": row["requested"],
                                  "standard_status": "unknown", "requires_status_recheck": True}
            row["warning"] = "本地状态预查失败，不能认定规范不存在或已经废止。"
    return list(references.values())
