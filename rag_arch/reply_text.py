"""Read a model's visible reply without inventing missing result fields."""
from .json_output import parse_single_json_object


def readable_reply(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    try:
        payload = parse_single_json_object(text)
    except (ValueError, TypeError):
        return text
    if isinstance(payload, dict):
        for name in ("answer", "text", "summary", "report"):
            value = payload.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return text
