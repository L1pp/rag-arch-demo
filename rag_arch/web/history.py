"""Read-only HTTP view of the published history snapshot; never rebuild data."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = Path("data/meta/history/active_preparation.json")
PAGE = "标准历史关系图.html"
MEDIA_TYPES = {
    "current_history.json": "application/json",
    "relations.csv": "text/csv",
    "标准历史替代关系.md": "text/markdown",
}
_BOOTSTRAP = "choose(); return {model, render};"

# The current snapshot controller already handles node ids, aliases, ambiguous
# matches and unknown input, and clears the selected edge on focus changes.
# Seed its existing input before its first choose(); never rewrite graph logic.
_ADAPTER = r"""
/* history-web-adapter-v1 */
const historyParams = new URLSearchParams(win.location.search);
if (historyParams.has('std')) input.value = historyParams.get('std');
"""


def _artifact(root: Path, filename: str) -> Path:
    """Only the active pointer and four fixed adjacent files are reachable."""
    if filename not in {PAGE, *MEDIA_TYPES}:
        raise HTTPException(404, "未提供此历史关系附件。")
    try:
        metadata_root = (root / "data/meta").resolve()
        active_path = (root / ACTIVE).resolve()
        if not active_path.is_relative_to(metadata_root):
            raise ValueError("invalid active pointer")
        pointer = json.loads(active_path.read_text(encoding="utf-8-sig"))
        graph_value = pointer["history_graph"]
        if not isinstance(graph_value, str) or not graph_value:
            raise ValueError("missing history graph")
        graph = Path(graph_value)
        graph = (graph if graph.is_absolute() else root / graph).resolve()
        if (not graph.is_relative_to(metadata_root) or graph.name != "current_history.json"
                or not graph.is_file()):
            raise ValueError("invalid history graph location")
        target = (graph.parent / filename).resolve()
        if not target.is_relative_to(graph.parent) or not target.is_file():
            raise ValueError("missing history snapshot")
        return target
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(404, "当前历史关系快照或附件不可用。") from None


def _adapt_page(path: Path) -> str:
    try:
        source = path.read_text(encoding="utf-8-sig")
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OSError, UnicodeError):
        raise HTTPException(404, "当前历史关系页面不可读取。") from None
    if source.count(_BOOTSTRAP) != 1 or "HistoryGraphApp" not in source or "</h1>" not in source:
        raise HTTPException(503, "历史关系页面结构已变化，暂不能应用标准定位适配。")
    note = (f'<p id="history-snapshot" class="muted">快照文件更新时间：{escape(modified)}。'
            '本页展示已生成快照，不会自动重读或研究规范关系。</p>')
    return source.replace("</h1>", "</h1>" + note, 1).replace(_BOOTSTRAP, _ADAPTER + "\n" + _BOOTSTRAP, 1)


def create_history_router(root=ROOT):
    root = Path(root).resolve()
    routes = APIRouter(prefix="/viewer/history", tags=["history viewer"])

    @routes.get("", include_in_schema=False)
    def trailing_slash(request: Request):
        query = "?" + request.url.query if request.url.query else ""
        return RedirectResponse("/viewer/history/" + query, status_code=307)

    @routes.get("/", response_class=HTMLResponse)
    def history_page():
        html = _adapt_page(_artifact(root, PAGE))
        return HTMLResponse(html, headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    def attachment(filename):
        return FileResponse(_artifact(root, filename), media_type=MEDIA_TYPES[filename],
                            filename=filename, headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    @routes.get("/current_history.json")
    def history_json():
        return attachment("current_history.json")

    @routes.get("/relations.csv")
    def history_csv():
        return attachment("relations.csv")

    @routes.get("/标准历史替代关系.md")
    def history_markdown():
        return attachment("标准历史替代关系.md")

    @routes.get("/{unsupported:path}", include_in_schema=False)
    def unsupported_file(unsupported: str):
        raise HTTPException(404, "未提供此历史关系附件。")

    return routes


router = create_history_router()
