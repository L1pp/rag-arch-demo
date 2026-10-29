"""Explicit, owner-scoped import of legacy Web uploads and terminal tasks.

Legacy task archives and upload files do not contain a trustworthy owner.  A
caller must therefore provide one target user for the whole invocation and
explicitly choose apply before writing anything.  This module never scans the
corpus, deletes source files, imports sessions, or reuses legacy event cursors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from .task_repository import TaskRepository
from .task_service import ASSISTANTS, TERMINAL


class LegacyImportError(RuntimeError):
    """A safe migration error that avoids printing source data or credentials."""


_ASSET_ID = re.compile(r"^[0-9a-f]{32}$")
_LEGACY_ID = re.compile(r"^[0-9a-f]{32}$")
_SUPPORTED_SUFFIXES = {".pdf", ".doc", ".docx", ".txt", ".md"}
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_STORAGE_KEY = 512
_COPY_CHUNK_BYTES = 1024 * 1024
_DROP_RESULT_KEYS = {
    "audit", "audit_log", "debug", "debug_info", "debug_path",
    "local_path", "absolute_path", "source_path", "filesystem_path",
}
_ABSOLUTE_PATH_FRAGMENT = re.compile(
    r"(?i)(?:[A-Z]:[\\/][^\s\"'<>|,;]+|\\\\[^\s\"'<>|,;]+)"
)


@dataclass(frozen=True)
class LegacyAsset:
    attachment_id: str
    legacy_workspace_id: str
    relative_source: str
    source_path: Path
    filename: str
    size_bytes: int
    storage_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegacyTask:
    legacy_workspace_id: str
    legacy_task_id: str
    source_relative: str
    task_id: str
    request_id: str
    assistant_id: str
    inputs: dict[str, Any]
    status: str
    detail: str
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float
    result: Any
    error: str | None


@dataclass
class LegacyImportPlan:
    owner_user_id: int
    workspace_id: str | None
    assets: list[LegacyAsset] = field(default_factory=list)
    tasks: list[LegacyTask] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)
    already_present_assets: list[str] = field(default_factory=list)
    already_present_tasks: list[str] = field(default_factory=list)

    @property
    def ready_asset_count(self) -> int:
        return len(self.assets) - len(self.already_present_assets)

    @property
    def ready_task_count(self) -> int:
        return len(self.tasks) - len(self.already_present_tasks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "owner_user_id": self.owner_user_id,
            "workspace_id": self.workspace_id,
            "workspace_will_be_created_on_apply": self.workspace_id is None,
            "asset_count": len(self.assets),
            "task_count": len(self.tasks),
            "already_present_assets": list(self.already_present_assets),
            "already_present_tasks": list(self.already_present_tasks),
            "ready_asset_count": self.ready_asset_count,
            "ready_task_count": self.ready_task_count,
            "ready_assets": [
                {"asset_id": asset.attachment_id, "filename": asset.filename,
                 "source": asset.relative_source}
                for asset in self.assets
                if asset.attachment_id not in self.already_present_assets
            ],
            "ready_tasks": [
                {"source": task.source_relative, "assistant_id": task.assistant_id,
                 "status": task.status, "target_task_id": task.task_id}
                for task in self.tasks
                if task.task_id not in self.already_present_tasks
            ],
            "skipped": list(self.skipped),
            "conflicts": list(self.conflicts),
        }


def _validate_owner(owner_user_id: int) -> int:
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) or owner_user_id < 1:
        raise LegacyImportError("必须明确提供一个有效的 --owner-user-id。")
    return owner_user_id


def _root(path: str | Path, label: str) -> Path:
    try:
        value = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LegacyImportError(f"{label}目录不存在或不可访问。") from exc
    if not value.is_dir():
        raise LegacyImportError(f"{label}必须是目录。")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _read_json(path: Path) -> Any:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("unsupported file")
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise LegacyImportError("JSON文件无效、过大或不可读。") from exc


def _manifest_assets(manifest_path: str | Path | None, upload_root: Path,
                     owner_user_id: int, skipped: list[dict[str, str]],
                     conflicts: list[dict[str, str]]) -> dict[str, LegacyAsset]:
    if manifest_path is None:
        return {}
    try:
        manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LegacyImportError("附件manifest不存在或不可访问。") from exc
    value = _read_json(manifest_file)
    if (not isinstance(value, dict) or value.get("version") != 1
            or not isinstance(value.get("assets"), list)):
        raise LegacyImportError("附件manifest须包含 version=1 和 assets 列表。")

    assets: dict[str, LegacyAsset] = {}
    for index, row in enumerate(value["assets"]):
        item_label = f"manifest.assets[{index}]"
        if not isinstance(row, dict):
            conflicts.append({"kind": "asset", "source": item_label,
                              "reason": "manifest_entry_invalid"})
            continue
        attachment_id = row.get("asset_id")
        relative = row.get("source")
        owner = row.get("owner_user_id")
        if owner is not None and (isinstance(owner, bool) or owner != owner_user_id):
            skipped.append({"kind": "asset", "source": item_label,
                             "reason": "owner_conflict"})
            continue
        if not isinstance(attachment_id, str) or not _ASSET_ID.fullmatch(attachment_id):
            conflicts.append({"kind": "asset", "source": item_label,
                              "reason": "asset_id_invalid"})
            continue
        if not isinstance(relative, str) or "\\" in relative or ":" in relative:
            conflicts.append({"kind": "asset", "source": item_label,
                              "reason": "source_path_invalid"})
            continue
        pure = PurePosixPath(relative)
        parts = pure.parts
        if (pure.is_absolute() or len(parts) != 3 or any(p in ("", ".", "..") for p in parts)
                or not _LEGACY_ID.fullmatch(parts[0]) or parts[1] != attachment_id
                or not parts[2].startswith("source")):
            conflicts.append({"kind": "asset", "source": item_label,
                              "reason": "source_path_invalid"})
            continue
        suffix = Path(parts[2]).suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES or parts[2] != "source" + suffix:
            conflicts.append({"kind": "asset", "source": item_label,
                              "reason": "source_suffix_invalid"})
            continue
        source_path = upload_root.joinpath(*parts)
        if not _inside(source_path, upload_root) or source_path.is_symlink():
            conflicts.append({"kind": "asset", "source": item_label,
                              "reason": "source_path_escapes_root"})
            continue
        if not source_path.is_file():
            skipped.append({"kind": "asset", "source": relative,
                             "reason": "source_file_missing"})
            continue
        try:
            size_bytes = source_path.stat().st_size
        except OSError:
            skipped.append({"kind": "asset", "source": relative,
                             "reason": "source_file_unreadable"})
            continue
        if size_bytes <= 0:
            skipped.append({"kind": "asset", "source": relative,
                             "reason": "source_file_empty"})
            continue

        filename = row.get("filename")
        if filename is None:
            filename = attachment_id + suffix
        if not isinstance(filename, str) or "\x00" in filename:
            conflicts.append({"kind": "asset", "source": relative,
                              "reason": "filename_invalid"})
            continue
        filename = PurePosixPath(filename.replace("\\", "/")).name.strip()
        if (not filename or filename in {".", ".."} or len(filename) > 255
                or Path(filename).suffix.lower() != suffix):
            conflicts.append({"kind": "asset", "source": relative,
                              "reason": "filename_invalid"})
            continue
        try:
            _validate_json_text(filename)
        except (UnicodeEncodeError, ValueError):
            conflicts.append({"kind": "asset", "source": relative,
                              "reason": "filename_invalid"})
            continue
        metadata = {"legacy_import": {
            "legacy_workspace_id": parts[0],
            "asset_id": attachment_id,
        }}
        asset = LegacyAsset(
            attachment_id=attachment_id,
            legacy_workspace_id=parts[0],
            relative_source=relative,
            source_path=source_path,
            filename=filename,
            size_bytes=size_bytes,
            metadata=metadata,
        )
        previous = assets.get(attachment_id)
        if previous is not None:
            if (previous.relative_source, previous.filename) != (relative, filename):
                conflicts.append({"kind": "asset", "source": relative,
                                  "reason": "duplicate_asset_id_conflict"})
            continue
        assets[attachment_id] = asset
    return assets


def _parse_timestamp(value: Any, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp timezone")
    epoch = parsed.timestamp()
    if not math.isfinite(epoch):
        raise ValueError("timestamp finite")
    return epoch


def _absolute_path(value: str) -> bool:
    return bool(re.match(r"^(?:[A-Za-z]:[\\/]|\\\\|/)", value))


def _clean_message(value: str) -> str:
    if _absolute_path(value.strip()):
        return "[本机绝对路径已省略]"
    return _ABSOLUTE_PATH_FRAGMENT.sub("[本机绝对路径已省略]", value)


def _validate_json_text(value: Any) -> None:
    """Reject values PostgreSQL text/JSONB cannot faithfully store."""
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError("NUL text")
        value.encode("utf-8", errors="strict")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_json_text(key)
            _validate_json_text(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_text(item)


def _clean_result(value: Any) -> Any:
    """Remove old audit/debug path material before it enters user-visible JSON."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if (normalized_key in _DROP_RESULT_KEYS or normalized_key.endswith("_path")
                    or normalized_key in {"path", "paths", "absolute_paths"}):
                continue
            cleaned = _clean_result(item)
            if isinstance(cleaned, str) and _absolute_path(cleaned.strip()):
                continue
            result[str(key)] = cleaned
        return result
    if isinstance(value, list):
        return [_clean_result(item) for item in value]
    if isinstance(value, str) and _absolute_path(value.strip()):
        return "[本机绝对路径已省略]"
    return value


def _stable_task_ids(owner_user_id: int, legacy_workspace_id: str,
                     legacy_task_id: str) -> tuple[str, str]:
    identity = f"rag-arch-legacy-task-v1:{owner_user_id}:{legacy_workspace_id}:{legacy_task_id}"
    task_id = uuid5(NAMESPACE_URL, identity).hex
    request_id = "legacy-v1:" + task_id
    if len(task_id) > 128 or len(request_id) > 128:
        raise LegacyImportError("旧任务来源标识超过允许长度。")
    return task_id, request_id


def _task_from_archive(path: Path, archive_root: Path, owner_user_id: int,
                       skipped: list[dict[str, str]]) -> LegacyTask | None:
    relative = path.relative_to(archive_root).as_posix()
    source_workspace_id = path.parent.parent.name
    source_task_id = path.parent.name
    try:
        if (not _LEGACY_ID.fullmatch(source_workspace_id)
                or not _LEGACY_ID.fullmatch(source_task_id)):
            raise ValueError("source ids")
        row = _read_json(path)
        if not isinstance(row, dict):
            raise ValueError("root object")
        if row.get("task_id") != source_task_id:
            raise ValueError("task id mismatch")
        for owner_key in ("owner_user_id", "user_id"):
            archived_owner = row.get(owner_key)
            if archived_owner is not None and (
                isinstance(archived_owner, bool) or archived_owner != owner_user_id
            ):
                skipped.append({"kind": "task", "source": relative,
                                "reason": "owner_conflict"})
                return None

        status = row.get("status")
        if status not in TERMINAL:
            skipped.append({"kind": "task", "source": relative,
                            "reason": "task_not_terminal"})
            return None
        assistant_id = row.get("assistant_id")
        prompt = row.get("prompt")
        simulation = row.get("simulation")
        if (assistant_id not in ASSISTANTS or not isinstance(prompt, str)
                or not prompt.strip() or type(simulation) is not bool):
            raise ValueError("required fields")
        request_inputs: dict[str, Any] = {
            "prompt": prompt.strip(),
            "simulation": simulation,
        }
        duration = row.get("duration_seconds")
        if duration is not None:
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration < 0):
                raise ValueError("duration")
            request_inputs["duration_seconds"] = float(duration)
        attachment_ids = row.get("attachment_ids", [])
        if (not isinstance(attachment_ids, list)
                or any(not isinstance(item, str) or not _ASSET_ID.fullmatch(item)
                       for item in attachment_ids)
                or len(set(attachment_ids)) != len(attachment_ids)):
            raise ValueError("attachments")
        request_inputs["attachment_ids"] = list(attachment_ids)
        for key in ("registered_plan_id", "plan_action", "use_embeddings",
                    "max_steps", "max_tool_calls"):
            if key in row:
                value = row[key]
                if key == "use_embeddings" and type(value) is not bool:
                    raise ValueError("use_embeddings")
                if key in ("registered_plan_id", "plan_action") and value is not None and not isinstance(value, str):
                    raise ValueError(key)
                if key in ("max_steps", "max_tool_calls") and value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                ):
                    raise ValueError(key)
                request_inputs[key] = value

        result = row.get("result")
        error = row.get("error")
        if status == "completed":
            if (not isinstance(result, dict) or not isinstance(result.get("text"), str)
                    or not result.get("text").strip()):
                raise ValueError("completed result")
        if status == "failed" and not (
            isinstance(error, str) and error.strip()
        ):
            raise ValueError("failed error")
        if error is not None and not isinstance(error, str):
            raise ValueError("error type")
        cleaned_result = _clean_result(result) if result is not None else None
        if isinstance(cleaned_result, dict):
            cleaned_result["simulation"] = simulation
        # Validate the exact JSON payload that will be inserted into JSONB.
        _validate_json_text(request_inputs)
        _validate_json_text(cleaned_result)
        json.dumps(cleaned_result, ensure_ascii=False, allow_nan=False)
        created_at = _parse_timestamp(row.get("created_at"))
        started_at = _parse_timestamp(row.get("started_at"), optional=True)
        finished_at = _parse_timestamp(row.get("finished_at"))
        assert created_at is not None and finished_at is not None
        if (finished_at < created_at
                or (started_at is not None and not created_at <= started_at <= finished_at)
                or (status in ("completed", "failed") and started_at is None)):
            raise ValueError("timestamp order")
        elapsed = row.get("elapsed_seconds")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("elapsed")
        detail = row.get("detail", "")
        if not isinstance(detail, str):
            raise ValueError("detail")
        detail = _clean_message(detail)
        if isinstance(error, str):
            error = _clean_message(error)
        _validate_json_text(detail)
        if isinstance(error, str):
            _validate_json_text(error)
        task_id, request_id = _stable_task_ids(owner_user_id, source_workspace_id,
                                               source_task_id)
        return LegacyTask(
            legacy_workspace_id=source_workspace_id,
            legacy_task_id=source_task_id,
            source_relative=relative,
            task_id=task_id,
            request_id=request_id,
            assistant_id=assistant_id,
            inputs=request_inputs,
            status=status,
            detail=detail,
            created_at=created_at,
            updated_at=finished_at,
            started_at=started_at,
            finished_at=finished_at,
            result=cleaned_result,
            error=error,
        )
    except (OSError, ValueError, TypeError, OverflowError, AssertionError,
            LegacyImportError, RecursionError):
        skipped.append({"kind": "task", "source": relative,
                        "reason": "task_structure_invalid"})
        return None


def _scan_archives(archive_root: Path, owner_user_id: int,
                   skipped: list[dict[str, str]]) -> list[LegacyTask]:
    tasks: list[LegacyTask] = []
    try:
        workspace_dirs = list(archive_root.iterdir())
    except OSError as exc:
        raise LegacyImportError("无法读取旧任务归档目录。") from exc
    for workspace_dir in workspace_dirs:
        if workspace_dir.is_symlink() or not workspace_dir.is_dir():
            continue
        try:
            task_dirs = list(workspace_dir.iterdir())
        except OSError:
            skipped.append({"kind": "workspace", "source": workspace_dir.name,
                            "reason": "archive_directory_unreadable"})
            continue
        for task_dir in task_dirs:
            if task_dir.is_symlink() or not task_dir.is_dir():
                continue
            archive_file = task_dir / "result.json"
            if not archive_file.exists():
                continue
            if archive_file.is_symlink() or not _inside(archive_file, archive_root):
                skipped.append({"kind": "task",
                                "source": f"{workspace_dir.name}/{task_dir.name}/result.json",
                                "reason": "archive_path_invalid"})
                continue
            task = _task_from_archive(archive_file, archive_root, owner_user_id, skipped)
            if task is not None:
                tasks.append(task)
    return tasks


def _load_inputs(owner_user_id: int, archive_root: str | Path,
                 upload_root: str | Path,
                 asset_manifest: str | Path | None) -> tuple[LegacyImportPlan, dict[str, LegacyAsset]]:
    owner_user_id = _validate_owner(owner_user_id)
    archive = _root(archive_root, "旧任务归档")
    uploads = _root(upload_root, "旧附件")
    skipped: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    assets = _manifest_assets(asset_manifest, uploads, owner_user_id, skipped, conflicts)
    tasks = _scan_archives(archive, owner_user_id, skipped)
    usable_tasks: list[LegacyTask] = []
    available_asset_ids = set(assets)
    for task in tasks:
        missing = sorted(set(task.inputs["attachment_ids"]) - available_asset_ids)
        if missing:
            skipped.append({"kind": "task", "source": task.source_relative,
                            "reason": "attachment_missing_from_manifest"})
            continue
        usable_tasks.append(task)
    plan = LegacyImportPlan(owner_user_id, None, list(assets.values()), usable_tasks,
                            skipped, conflicts)
    return plan, assets


def _existing_workspace(database, owner_user_id: int) -> str | None:
    with database.transaction() as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        owner = connection.execute("SELECT id FROM users WHERE id = %s", (owner_user_id,)).fetchone()
        if owner is None:
            raise LegacyImportError("指定的目标用户不存在。")
        workspace = connection.execute(
            "SELECT workspace_id FROM workspaces WHERE user_id = %s", (owner_user_id,)
        ).fetchone()
        return None if workspace is None else str(workspace["workspace_id"])


def _safe_target_path(target_root: Path, user_id: int, workspace_id: str,
                      asset: LegacyAsset) -> tuple[Path, str]:
    suffix = Path(asset.filename).suffix.lower()
    key = f"{user_id}/{workspace_id}/{asset.attachment_id}/source{suffix}"
    if not key or len(key) > _MAX_STORAGE_KEY:
        raise LegacyImportError("新附件storage_key超过允许长度。")
    base = (target_root / "data" / "runtime" / "web_uploads").resolve(strict=False)
    path = base / str(user_id) / workspace_id / asset.attachment_id / ("source" + suffix)
    if not _inside(path, base):
        raise LegacyImportError("新附件目标路径超出web_uploads目录。")
    return path, key


def _same_bytes(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as a, right.open("rb") as b:
            while True:
                a_chunk = a.read(_COPY_CHUNK_BYTES)
                b_chunk = b.read(_COPY_CHUNK_BYTES)
                if a_chunk != b_chunk:
                    return False
                if not a_chunk:
                    return True
    except OSError:
        return False


def _asset_metadata_matches(row, asset: LegacyAsset, storage_key: str) -> bool:
    return (
        row["filename"] == asset.filename
        and row["storage_key"] == storage_key
        and row["size_bytes"] == asset.size_bytes
        and row["status"] == "ready"
        and row["content_type"] is None
        and row["metadata"] == asset.metadata
    )


def _task_values(task: LegacyTask, owner_user_id: int, workspace_id: str,
                 generation: int) -> tuple[Any, ...]:
    return (
        task.task_id, owner_user_id, workspace_id, task.assistant_id, task.request_id,
        json.dumps(task.inputs, ensure_ascii=False, allow_nan=False), task.status,
        generation, task.status, task.detail, task.created_at, task.updated_at,
        task.started_at, task.finished_at,
        None if task.result is None else json.dumps(task.result, ensure_ascii=False, allow_nan=False),
        task.error,
    )


def _task_row_matches(row, task: LegacyTask, owner_user_id: int,
                      workspace_id: str) -> bool:
    return (
        row["task_id"] == task.task_id
        and row["user_id"] == owner_user_id
        and row["workspace_id"] == workspace_id
        and row["assistant_id"] == task.assistant_id
        and row["request_id"] == task.request_id
        and row["inputs"] == task.inputs
        and row["status"] == task.status
        and row["stage"] == task.status
        and row["detail"] == task.detail
        and row["created_at"] == task.created_at
        and row["updated_at"] == task.updated_at
        and row["started_at"] == task.started_at
        and row["finished_at"] == task.finished_at
        and row["result"] == task.result
        and row["error"] == task.error
    )


def _ensure_workspace(connection, owner_user_id: int, now: float) -> str:
    candidate = uuid4().hex
    connection.execute(
        "INSERT INTO workspaces(workspace_id, user_id, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (user_id) DO NOTHING",
        (candidate, owner_user_id, now, now),
    )
    row = connection.execute(
        "SELECT workspace_id FROM workspaces WHERE user_id = %s FOR UPDATE",
        (owner_user_id,),
    ).fetchone()
    if row is None:
        raise LegacyImportError("指定的目标用户不存在。")
    workspace_id = str(row["workspace_id"])
    connection.execute(
        "INSERT INTO assistant_slots(user_id, assistant_id, updated_at) "
        "VALUES (%s, 'chat', %s), (%s, 'standards', %s), (%s, 'plan', %s) "
        "ON CONFLICT (user_id, assistant_id) DO NOTHING",
        (owner_user_id, now, owner_user_id, now, owner_user_id, now),
    )
    return workspace_id


def _stage_atomic(asset: LegacyAsset, base: Path) -> Path:
    """Copy and fsync outside the scheduler transaction on the target volume."""
    base.mkdir(parents=True, exist_ok=True)
    temporary = base / (".legacy-import-" + uuid4().hex + ".tmp")
    if not _inside(temporary, base):
        raise LegacyImportError("附件暂存路径超出web_uploads目录。")
    try:
        with asset.source_path.open("rb") as source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output, _COPY_CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != asset.size_bytes:
            raise LegacyImportError("旧附件在迁移过程中发生变化，未完成导入。")
        if not _same_bytes(temporary, asset.source_path):
            raise LegacyImportError("旧附件在迁移过程中发生变化，未完成导入。")
        return temporary
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _publish_staged(temporary: Path, destination: Path, base: Path) -> None:
    """Atomically link a staged file without replacing any existing target."""
    if destination.exists() or destination.is_symlink():
        raise LegacyImportError("目标附件路径已存在但没有一致的目标登记，拒绝覆盖。")
    if (not _inside(destination, base) or not _inside(destination.parent, base)
            or not _inside(temporary, base)):
        raise LegacyImportError("新附件路径超出web_uploads目录。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (not _inside(destination, base) or not _inside(destination.parent, base)
            or not _inside(temporary, base)):
        raise LegacyImportError("新附件路径超出web_uploads目录。")
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise LegacyImportError("目标附件在迁移过程中出现冲突，已停止导入。") from exc
    except OSError as exc:
        raise LegacyImportError("目标文件系统不支持安全的原子无覆盖写入。") from exc


def _cleanup_staged(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _cleanup_created(paths: list[tuple[Path, Path]]) -> None:
    for path, base in reversed(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        parent = path.parent
        while parent != base and _inside(parent, base):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _resolve_commit_outcome(database, plan: LegacyImportPlan,
                            created_assets: list[tuple[LegacyAsset, Path, str]],
                            inserted_tasks: list[LegacyTask], target_root: Path) -> bool | None:
    """Return committed/rolled-back when provable, otherwise preserve files."""
    workspace_id = plan.workspace_id
    if workspace_id is None:
        return None
    base = (target_root / "data" / "runtime" / "web_uploads").resolve(strict=False)
    try:
        with database.transaction() as connection:
            # The import holds this same scheduler row lock through COMMIT.
            # Waiting for it prevents a fresh MVCC snapshot from mistaking an
            # in-flight COMMIT for a rollback and deleting files too early.
            lock = connection.execute(
                "SELECT singleton_key FROM scheduler_state "
                "WHERE singleton_key = 'global' FOR UPDATE"
            ).fetchone()
            if lock is None:
                return None
            for asset in plan.assets:
                destination, storage_key = _safe_target_path(
                    target_root, plan.owner_user_id, workspace_id, asset
                )
                row = connection.execute(
                    "SELECT filename, storage_key, size_bytes, status, content_type, metadata "
                    "FROM attachments WHERE user_id = %s AND workspace_id = %s AND attachment_id = %s",
                    (plan.owner_user_id, workspace_id, asset.attachment_id),
                ).fetchone()
                if (row is None or not _asset_metadata_matches(row, asset, storage_key)
                        or destination.is_symlink() or not destination.is_file()
                        or not _inside(destination, base)
                        or not _same_bytes(asset.source_path, destination)):
                    break
            else:
                for task in plan.tasks:
                    rows = connection.execute(
                        "SELECT * FROM tasks WHERE task_id = %s AND user_id = %s",
                        (task.task_id, plan.owner_user_id),
                    ).fetchall()
                    if len(rows) != 1 or not _task_row_matches(
                        rows[0], task, plan.owner_user_id, workspace_id
                    ):
                        break
                    event = connection.execute(
                        "SELECT 1 FROM workspace_events WHERE user_id = %s AND workspace_id = %s "
                        "AND task_id = %s LIMIT 1",
                        (plan.owner_user_id, workspace_id, task.task_id),
                    ).fetchone()
                    if event is None:
                        break
                else:
                    return True

            new_rows_present = 0
            for asset, destination, storage_key in created_assets:
                row = connection.execute(
                    "SELECT filename, storage_key, size_bytes, status, content_type, metadata "
                    "FROM attachments WHERE user_id = %s AND workspace_id = %s AND attachment_id = %s",
                    (plan.owner_user_id, workspace_id, asset.attachment_id),
                ).fetchone()
                if row is not None:
                    new_rows_present += 1
                    if (not _asset_metadata_matches(row, asset, storage_key)
                            or destination.is_symlink() or not destination.is_file()
                            or not _same_bytes(asset.source_path, destination)):
                        return None
            for task in inserted_tasks:
                rows = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = %s AND user_id = %s",
                    (task.task_id, plan.owner_user_id),
                ).fetchall()
                if rows:
                    new_rows_present += 1
                    if (len(rows) != 1 or not _task_row_matches(
                        rows[0], task, plan.owner_user_id, workspace_id
                    )):
                        return None
            expected_new = len(created_assets) + len(inserted_tasks)
            if new_rows_present == 0:
                return False
            if new_rows_present == expected_new:
                # A concurrent change or incomplete event write is not enough
                # proof; the all-plan check above must pass to accept success.
                return None
            return None
    except Exception:
        return None


def _inspect_targets(database, plan: LegacyImportPlan, assets: dict[str, LegacyAsset],
                     workspace_id: str | None, target_root: Path,
                     conflicts: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    already_assets: list[str] = []
    already_tasks: list[str] = []
    base = (target_root / "data" / "runtime" / "web_uploads").resolve(strict=False)
    with database.transaction() as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        if workspace_id is not None:
            for asset in assets.values():
                try:
                    destination, storage_key = _safe_target_path(
                        target_root, plan.owner_user_id, workspace_id, asset
                    )
                except LegacyImportError:
                    conflicts.append({"kind": "asset", "source": asset.relative_source,
                                      "reason": "storage_key_invalid"})
                    continue
                row = connection.execute(
                    "SELECT filename, storage_key, size_bytes, status, content_type, metadata "
                    "FROM attachments WHERE user_id = %s AND workspace_id = %s AND attachment_id = %s",
                    (plan.owner_user_id, workspace_id, asset.attachment_id),
                ).fetchone()
                if row is None:
                    if destination.exists() or destination.is_symlink():
                        conflicts.append({"kind": "asset", "source": asset.relative_source,
                                          "reason": "unregistered_target_file_exists"})
                    continue
                if (not _asset_metadata_matches(row, asset, storage_key)
                        or destination.is_symlink() or not destination.is_file()
                        or not _inside(destination, base)
                        or not _same_bytes(asset.source_path, destination)):
                    conflicts.append({"kind": "asset", "source": asset.relative_source,
                                      "reason": "target_asset_conflict"})
                else:
                    already_assets.append(asset.attachment_id)

        for task in plan.tasks:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE task_id = %s OR "
                "(user_id = %s AND assistant_id = %s AND request_id = %s)",
                (task.task_id, plan.owner_user_id, task.assistant_id, task.request_id),
            ).fetchall()
            if not rows:
                continue
            if len(rows) != 1 or workspace_id is None:
                conflicts.append({"kind": "task", "source": task.source_relative,
                                  "reason": "target_task_conflict"})
                continue
            row = rows[0]
            if _task_row_matches(row, task, plan.owner_user_id, workspace_id):
                already_tasks.append(task.task_id)
            else:
                conflicts.append({"kind": "task", "source": task.source_relative,
                                  "reason": "target_task_conflict"})
    return already_assets, already_tasks


def plan_legacy_import(database, owner_user_id: int, *, archive_root: str | Path,
                       upload_root: str | Path, target_root: str | Path,
                       asset_manifest: str | Path | None = None) -> LegacyImportPlan:
    """Build a read-only import plan; missing/unsafe history is reported, not copied."""
    target = Path(target_root).expanduser().resolve(strict=False)
    plan, assets = _load_inputs(owner_user_id, archive_root, upload_root,
                                asset_manifest)
    workspace_id = _existing_workspace(database, plan.owner_user_id)
    plan.workspace_id = workspace_id
    already_assets, already_tasks = _inspect_targets(
        database, plan, assets, workspace_id, target, plan.conflicts
    )
    plan.already_present_assets = already_assets
    plan.already_present_tasks = already_tasks
    return plan


def apply_legacy_import(database, owner_user_id: int, *, archive_root: str | Path,
                        upload_root: str | Path,
                        target_root: str | Path,
                        asset_manifest: str | Path | None = None) -> LegacyImportPlan:
    """Apply one explicitly owner-scoped migration atomically in PostgreSQL."""
    owner_user_id = _validate_owner(owner_user_id)
    target = Path(target_root).expanduser().resolve(strict=False)
    plan, assets = _load_inputs(owner_user_id, archive_root, upload_root,
                                asset_manifest)
    plan.workspace_id = _existing_workspace(database, owner_user_id)
    already_assets, already_tasks = _inspect_targets(
        database, plan, assets, plan.workspace_id, target, plan.conflicts
    )
    plan.already_present_assets = already_assets
    plan.already_present_tasks = already_tasks
    if plan.conflicts:
        return plan

    created_files: list[tuple[Path, Path]] = []
    created_assets: list[tuple[LegacyAsset, Path, str]] = []
    inserted_tasks: list[LegacyTask] = []
    staged_files: list[Path] = []
    staged_by_asset: dict[str, Path] = {}
    commit_may_have_started = False
    repository = TaskRepository(database)
    try:
        staging_base = (target / "data" / "runtime" / "web_uploads").resolve(strict=False)
        for asset in assets.values():
            if asset.attachment_id in plan.already_present_assets:
                continue
            temporary = _stage_atomic(asset, staging_base)
            staged_files.append(temporary)
            staged_by_asset[asset.attachment_id] = temporary

        with database.transaction() as connection:
            owner = connection.execute("SELECT id FROM users WHERE id = %s", (owner_user_id,)).fetchone()
            if owner is None:
                raise LegacyImportError("指定的目标用户不存在。")
            repository._lock_scheduler(connection)
            now = time.time()
            workspace_id = _ensure_workspace(connection, owner_user_id, now)
            plan.workspace_id = workspace_id
            repository._lock_workspace(connection, owner_user_id, workspace_id)
            slot_rows = {}
            for assistant_id in ASSISTANTS:
                slot_rows[assistant_id] = repository._lock_slot(connection, owner_user_id, assistant_id)

            base = (target / "data" / "runtime" / "web_uploads").resolve(strict=False)
            new_assets: list[tuple[LegacyAsset, Path, str]] = []
            for asset in assets.values():
                destination, storage_key = _safe_target_path(
                    target, owner_user_id, workspace_id, asset
                )
                row = connection.execute(
                    "SELECT filename, storage_key, size_bytes, status, content_type, metadata "
                    "FROM attachments WHERE user_id = %s AND workspace_id = %s AND attachment_id = %s "
                    "FOR UPDATE",
                    (owner_user_id, workspace_id, asset.attachment_id),
                ).fetchone()
                if row is not None:
                    if (not _asset_metadata_matches(row, asset, storage_key)
                            or destination.is_symlink() or not destination.is_file()
                            or not _inside(destination, base)
                            or not _same_bytes(asset.source_path, destination)):
                        raise LegacyImportError("目标已有不一致附件；整笔迁移已回滚。")
                    if asset.attachment_id not in plan.already_present_assets:
                        plan.already_present_assets.append(asset.attachment_id)
                    continue
                if asset.attachment_id not in staged_by_asset:
                    raise LegacyImportError("目标附件在dry-run后发生变化；请重新预览。")
                if destination.exists() or destination.is_symlink():
                    raise LegacyImportError("目标存在未登记附件；整笔迁移已回滚。")
                _publish_staged(staged_by_asset[asset.attachment_id], destination, base)
                created_files.append((destination, base))
                new_assets.append((asset, destination, storage_key))
                created_assets.append((asset, destination, storage_key))

            for asset, _destination, storage_key in new_assets:
                connection.execute(
                    "INSERT INTO attachments(attachment_id, user_id, workspace_id, filename, "
                    "storage_key, content_type, size_bytes, status, metadata, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, NULL, %s, 'ready', %s::jsonb, %s, %s)",
                    (asset.attachment_id, owner_user_id, workspace_id, asset.filename,
                     storage_key, asset.size_bytes,
                     json.dumps(asset.metadata, ensure_ascii=False, allow_nan=False), now, now),
                )

            new_tasks: list[tuple[LegacyTask, Any]] = []
            for task in plan.tasks:
                existing_rows = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = %s OR "
                    "(user_id = %s AND assistant_id = %s AND request_id = %s) FOR UPDATE",
                    (task.task_id, owner_user_id, task.assistant_id, task.request_id),
                ).fetchall()
                generation = int(slot_rows[task.assistant_id]["history_generation"])
                if existing_rows:
                    if len(existing_rows) != 1:
                        raise LegacyImportError("目标任务ID或请求ID冲突；整笔迁移已回滚。")
                    existing = existing_rows[0]
                    if _task_row_matches(existing, task, owner_user_id, workspace_id):
                        if task.task_id not in plan.already_present_tasks:
                            plan.already_present_tasks.append(task.task_id)
                        continue
                    raise LegacyImportError("目标已有同来源但内容不同的任务；整笔迁移已回滚。")
                task_values = _task_values(task, owner_user_id, workspace_id, generation)
                row = connection.execute(
                    "INSERT INTO tasks(task_id, user_id, workspace_id, assistant_id, request_id, inputs, "
                    "status, generation, cancel_requested, stage, detail, created_at, updated_at, "
                    "started_at, finished_at, result, error) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, FALSE, %s, %s, %s, %s, %s, %s, "
                    "%s::jsonb, %s) RETURNING *",
                    task_values,
                ).fetchone()
                for attachment_id in task.inputs["attachment_ids"]:
                    connection.execute(
                        "INSERT INTO task_attachments(user_id, workspace_id, task_id, attachment_id, created_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (owner_user_id, workspace_id, task.task_id, attachment_id, now),
                    )
                new_tasks.append((task, row))
                inserted_tasks.append(task)

            for _task, row in new_tasks:
                repository._append_task_event(connection, row)

            # If context exit fails after this point, PostgreSQL may already
            # have committed. Resolve using a fresh read-only transaction
            # before considering any filesystem cleanup.
            commit_may_have_started = True

        return plan
    except Exception as exc:
        if commit_may_have_started:
            outcome = _resolve_commit_outcome(
                database, plan, created_assets, inserted_tasks, target
            )
            if outcome is True:
                return plan
            if outcome is None:
                raise LegacyImportError(
                    "数据库提交状态暂不能确认；为避免破坏可能已提交的附件，目标文件已保留，"
                    "请先重新dry-run核对后再处理。"
                ) from exc
        _cleanup_created(created_files)
        raise
    finally:
        _cleanup_staged(staged_files)
