"""Export the fixed, read-only history viewer snapshot for a static web root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = Path("output/history_static")
DEPLOY_OUTPUT = Path("frontend/public/viewer/history")
MANIFEST_PATH = Path("output/history_static_export_manifest.json")
MANIFEST_FORMAT = "rag-arch-history-static-export-v1"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_FILENAMES = {"index.html", "current_history.json", "relations.csv", "标准历史替代关系.md"}


class ExportError(RuntimeError):
    """The export target is unsafe, ambiguous, or not owned by this exporter."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_symlink_components(root: Path, target: Path) -> None:
    relative = target.relative_to(root)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ExportError("导出路径含符号链接，已拒绝写入。")


def _resolve_output(root: Path, output: Path | str | None) -> tuple[Path, str]:
    root = root.resolve()
    requested = Path(output) if output is not None else DEFAULT_OUTPUT
    candidate = requested if requested.is_absolute() else root / requested
    lexical = Path(os.path.abspath(candidate))
    try:
        _reject_symlink_components(root, lexical)
    except ValueError:
        raise ExportError("输出目录必须位于项目根目录中。") from None
    resolved = candidate.resolve(strict=False)
    allowed = {root / DEFAULT_OUTPUT, root / DEPLOY_OUTPUT}
    if resolved not in allowed:
        raise ExportError("输出目录只允许项目验收目录或 frontend/public/viewer/history。")
    _reject_symlink_components(root, resolved)
    relative_name = resolved.relative_to(root).as_posix()
    return resolved, relative_name


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"format": MANIFEST_FORMAT, "targets": {}}
    if path.is_symlink() or not path.is_file():
        raise ExportError("静态导出归属清单不是普通文件，已拒绝写入。")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ExportError("静态导出归属清单不可读取，已拒绝覆盖。") from None
    if (not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT
            or not isinstance(manifest.get("targets"), dict)):
        raise ExportError("静态导出归属清单格式不受支持，已拒绝覆盖。")
    return manifest


def _target_files(root: Path) -> dict[str, bytes]:
    """Resolve only the four public snapshot files and adapt the HTML response."""
    from fastapi import HTTPException

    from rag_arch.web.history import MEDIA_TYPES, PAGE, _adapt_page, _artifact

    graph = _artifact(root, "current_history.json")
    source_paths = {PAGE: _artifact(root, PAGE)}
    source_paths.update({name: _artifact(root, name) for name in MEDIA_TYPES})
    if any(path.parent != graph.parent for path in source_paths.values()):
        raise ExportError("导出期间活动快照发生变化，请重新运行导出。")

    try:
        adapted = _adapt_page(source_paths[PAGE]).encode("utf-8")
        files = {"index.html": adapted}
        for filename in MEDIA_TYPES:
            files[filename] = source_paths[filename].read_bytes()
    except HTTPException as exc:
        if exc.status_code == 503:
            raise ExportError("历史页面模板结构变化，不能安全应用标准定位适配。") from None
        raise ExportError("历史快照或附件当前不可用。") from None
    except OSError:
        raise ExportError("历史快照或附件当前不可用。") from None
    return files


def _unavailable_html() -> bytes:
    return ("""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>历史关系快照不可用</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:54rem;margin:4rem auto;padding:0 1.2rem;color:#263238}h1{font-size:1.6rem}.muted{color:#546e7a}</style>
</head><body><main><h1>标准历史替代关系</h1>
<p role="status">当前没有可发布的历史关系快照。</p>
<p class="muted">此静态页面没有图表或附件数据。请在发布快照后重新运行静态导出。</p>
</main></body></html>
""").encode("utf-8")


def _existing_owned_files(output: Path, record: Any) -> dict[str, str]:
    recorded: dict[str, Any] = {}
    if record is None:
        recorded = {}
    elif isinstance(record, dict) and isinstance(record.get("files"), dict):
        recorded = record["files"]
        if any(name not in OUTPUT_FILENAMES for name in recorded):
            raise ExportError("归属清单包含未知文件名，已拒绝写入。")
        if any(not isinstance(value, str) or not HASH_RE.fullmatch(value)
               for value in recorded.values()):
            raise ExportError("归属清单包含无效文件记录，已拒绝写入。")
    else:
        raise ExportError("静态导出归属记录格式不受支持。")
    if not output.exists():
        return {}
    if output.is_symlink() or not output.is_dir():
        raise ExportError("输出目标不是普通目录，已拒绝写入。")

    present: dict[str, Path] = {}
    for child in output.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ExportError("输出目录含非预期文件或子目录，已拒绝写入。")
        present[child.name] = child

    if present and record is None:
        raise ExportError("输出目录已有文件，但归属清单中没有本工具的记录。")

    if any(name not in recorded for name in present):
        raise ExportError("输出目录含未由本工具登记的文件，已拒绝覆盖。")

    hashes: dict[str, str] = {}
    for name, file_path in present.items():
        expected_hash = recorded.get(name)
        if not isinstance(expected_hash, str) or not HASH_RE.fullmatch(expected_hash):
            raise ExportError("输出文件缺少有效归属记录，已拒绝覆盖。")
        actual_hash = _sha256(file_path.read_bytes())
        if actual_hash != expected_hash:
            raise ExportError("已登记的输出文件被修改，已拒绝覆盖。")
        hashes[name] = actual_hash
    return hashes


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".history-export-", dir=path.parent,
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ExportError("写入静态历史页面失败。") from exc


def export_history(root: Path | str = ROOT, output: Path | str | None = None) -> dict[str, Any]:
    """Write the public snapshot into one of the two explicitly supported targets.

    The acceptance directory is the default. The deployment directory is enabled
    only when supplied explicitly. Source paths are always resolved through the
    allowlist in ``rag_arch.web.history``.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ExportError("项目根目录不可用。")
    output_path, target_name = _resolve_output(root, output)

    manifest_path = root / MANIFEST_PATH
    _reject_symlink_components(root, manifest_path)
    manifest = _load_manifest(manifest_path)
    target_record = manifest["targets"].get(target_name)
    if target_record is not None and not isinstance(target_record, dict):
        raise ExportError("静态导出归属记录格式不受支持。")

    try:
        files = _target_files(root)
        status = "available"
    except Exception as exc:
        from fastapi import HTTPException

        if isinstance(exc, HTTPException) and exc.status_code == 404:
            files = {"index.html": _unavailable_html()}
            status = "unavailable"
        elif isinstance(exc, ExportError):
            raise
        else:
            raise

    _existing_owned_files(output_path, target_record)
    output_path.mkdir(parents=True, exist_ok=True)

    previous_files = set(target_record.get("files", {})) if isinstance(target_record, dict) else set()
    for stale_name in previous_files - set(files):
        stale_path = output_path / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for filename, payload in files.items():
        _atomic_write(output_path / filename, payload)

    manifest.setdefault("targets", {})[target_name] = {
        "status": status,
        "files": {filename: _sha256(payload) for filename, payload in files.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"status": status, "output": target_name, "files": sorted(files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出只读公开历史快照供静态站点使用。")
    parser.add_argument("--output", type=Path,
                        help="显式输出目录；仅允许 frontend/public/viewer/history。默认写入被忽略的 output/history_static。")
    args = parser.parse_args(argv)
    try:
        result = export_history(ROOT, args.output)
    except ExportError as exc:
        print(f"历史静态导出失败：{exc}", file=sys.stderr)
        return 2
    except OSError:
        print("历史快照读写失败；出于安全考虑不显示本地路径。", file=sys.stderr)
        return 2
    except Exception:
        print("历史静态导出遇到未预期错误；未输出本地路径。", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
