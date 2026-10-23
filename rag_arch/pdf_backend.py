"""Run PyMuPDF work in one long-lived process for concurrent web requests."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import signal
import tempfile
import threading
import time

import pymupdf
import psutil

from .models import LoadedDocument
from .request_control import check_request_control
from .resource_supervision import (
    ResourceCleanupUncertain,
    begin_resource,
    process_token,
    probe_process,
)
from .visual_page_index import build_visual_page_index


MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_EDGE = 2400
MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_PDF_TIMEOUT_SECONDS = 120.0
DEFAULT_PDF_CLEANUP_TIMEOUT_SECONDS = 2.0
PDF_TIMEOUT_ENV = "RAG_ARCH_PDF_TIMEOUT_SECONDS"
PDF_TEMP_DIR_ENV = "RAG_ARCH_PDF_TEMP_DIR"
PDF_CONTROL_POLL_SECONDS = 0.1


def _positive_finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + "必须是有限正数。") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + "必须是有限正数。")
    return number


def _configured_call_timeout(value) -> float:
    if value is not None:
        return _positive_finite(value, "PDF worker 超时")
    raw = os.environ.get(PDF_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_PDF_TIMEOUT_SECONDS
    return _positive_finite(raw, PDF_TIMEOUT_ENV)


def _process_identity(pid: int):
    """Return (create-time token, executable), or None if the PID is absent."""
    try:
        process = psutil.Process(pid)
        return process_token(pid), process.exe()
    except psutil.NoSuchProcess:
        return None


def _open_pdf(source):
    if isinstance(source, str):
        return pymupdf.open(source)
    return pymupdf.open(stream=source, filetype="pdf")


def _check_pdf(document) -> int:
    if not document.is_pdf:
        raise ValueError("来源不是有效 PDF。")
    if document.needs_pass:
        raise ValueError("加密 PDF 暂不支持分页。")
    if len(document) == 0:
        raise ValueError("PDF 没有可读取页面。")
    return len(document)


def _validate(source) -> int:
    try:
        with _open_pdf(source) as document:
            return _check_pdf(document)
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _image_to_pdf(source, suffix: str) -> bytes:
    try:
        image = (pymupdf.open(source) if isinstance(source, str)
                 else pymupdf.open(stream=source, filetype=suffix))
        with image:
            if image.is_pdf or len(image) != 1:
                raise ValueError("图片来源不是有效的单页图片。")
            return image.convert_to_pdf()
    except (OSError, RuntimeError) as exc:
        raise ValueError("文件排版失败，未生成可用分页 PDF。") from exc


def _list_pages(source, start: int, limit: int, query: str) -> dict:
    try:
        with _open_pdf(source) as document:
            page_count = _check_pdf(document)
            if start > page_count:
                raise ValueError("起始页码超出来源页面范围。")
            results = []
            last_page = start - 1
            for index in range(start - 1, page_count):
                last_page = index + 1
                text = document[index].get_text().strip()
                position = text.casefold().find(query.casefold()) if query else 0
                if position < 0:
                    continue
                preview_start = max(0, position - 80)
                results.append({"page": index + 1, "has_text": bool(text),
                                "text_preview": text[preview_start:preview_start + 500]})
                if len(results) >= limit:
                    break
            return {"page_count": page_count, "pages": results,
                    "next_start": last_page + 1 if last_page < page_count else None}
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _scan_visual_pages(source) -> dict:
    """Scan one PDF's page geometry/text metadata without rasterising it."""
    try:
        with _open_pdf(source) as document:
            _check_pdf(document)
            return build_visual_page_index(document)
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _read_page(source, page: int, scale: float) -> dict:
    try:
        with _open_pdf(source) as document:
            page_count = _check_pdf(document)
            if page > page_count:
                raise ValueError("物理页码超出来源页面范围。")
            original = document[page - 1]
            width, height = original.rect.width, original.rect.height
            render_scale = min(float(scale), (MAX_IMAGE_EDGE - 1) / max(width, height),
                               (MAX_IMAGE_PIXELS / (width * height)) ** 0.5 * 0.99)
            for _ in range(3):
                pixmap = original.get_pixmap(matrix=pymupdf.Matrix(render_scale, render_scale),
                                             colorspace=pymupdf.csRGB, alpha=False)
                png = pixmap.tobytes("png")
                if len(png) <= MAX_IMAGE_BYTES:
                    break
                render_scale *= 0.7
            else:
                raise ValueError("页面图像过大，请使用更小的缩放比例。")
            return {"page_count": page_count, "page": page,
                    "text": original.get_text().strip(), "image_bytes": png,
                    "mime_type": "image/png", "width": pixmap.width, "height": pixmap.height,
                    "rendered_scale": render_scale}
    except (OSError, RuntimeError) as exc:
        raise ValueError("PDF 无法打开，请检查来源文件。") from exc


def _worker_call(operation: str, payload: dict):
    if operation == "validate":
        result = _validate(payload["source"])
    elif operation == "extract_document":
        from .loaders import load_document
        if "data_path" in payload:
            data = Path(payload["data_path"]).read_bytes()
        else:  # Backwards-compatible internal spelling for direct callers.
            data = payload["data"]
        result = load_document(payload["filename"], data, pdf_backend=None)
    elif operation == "image_to_pdf":
        result = _image_to_pdf(payload["source"], payload["suffix"])
    elif operation == "list_pages":
        result = _list_pages(payload["source"], payload["start"], payload["limit"], payload["query"])
    elif operation in {"scan_visual_pages", "visual_page_index"}:
        result = _scan_visual_pages(payload["source"])
    elif operation == "read_page":
        result = _read_page(payload["source"], payload["page"], payload["scale"])
    else:
        raise ValueError("未知 PDF worker 操作。")
    return os.getpid(), result


def _write_response(path: str, envelope: tuple) -> None:
    temporary = path + ".part"
    with open(temporary, "wb") as stream:
        pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
    os.replace(temporary, path)


_SAFE_WORKER_ERRORS = {
    "来源不是有效 PDF。",
    "加密 PDF 暂不支持分页。",
    "PDF 没有可读取页面。",
    "起始页码超出来源页面范围。",
    "图片来源不是有效的单页图片。",
    "物理页码超出来源页面范围。",
    "页面图像过大，请使用更小的缩放比例。",
    "文件排版失败，未生成可用分页 PDF。",
    "PDF 无法打开，请检查来源文件。",
    "未知 PDF worker 操作。",
    "PDF 读取依赖 PyMuPDF 尚未安装。",
    "无法打开 PDF，文件可能损坏或格式不符。",
    "该 PDF 已加密，请上传可直接打开的副本。",
    "PDF 没有页面。",
}


def _safe_worker_error(exc: BaseException) -> tuple[str, str, str]:
    """Keep established public validation messages and hide arbitrary details."""
    message = str(exc)
    if isinstance(exc, ValueError) and message in _SAFE_WORKER_ERRORS:
        return "builtins", "ValueError", message
    return "builtins", "RuntimeError", "PDF worker 处理失败。"


def _pdf_process_main(connection, worker_call) -> None:
    """Serve one request at a time; pipe messages contain paths only."""
    try:
        while True:
            try:
                message = pickle.loads(connection.recv_bytes(maxlength=16 * 1024))
            except (EOFError, OSError):
                return
            if message == ("stop",):
                try:
                    connection.send_bytes(pickle.dumps(("stopped", os.getpid())))
                except (EOFError, OSError):
                    pass
                return
            if not (isinstance(message, tuple) and len(message) == 2):
                return
            request_path, response_path = message
            try:
                with open(request_path, "rb") as stream:
                    operation, payload = pickle.load(stream)
                _pid, result = worker_call(operation, payload)
                _write_response(response_path, ("ok", result))
            except BaseException as exc:
                try:
                    module, name, message = _safe_worker_error(exc)
                    _write_response(response_path, (
                        "error", module, name, message))
                except BaseException:
                    # A missing response file is treated as a worker failure by the parent.
                    return
            try:
                connection.send_bytes(pickle.dumps(("result", os.getpid())))
            except (EOFError, OSError):
                return
    finally:
        try:
            connection.close()
        except OSError:
            pass


def _raise_worker_error(module: str, name: str, message: str):
    """Recreate the common builtin exceptions raised by the legacy worker."""
    if module == "builtins":
        builtins_obj = __builtins__
        candidate = (builtins_obj.get(name) if isinstance(builtins_obj, dict)
                     else getattr(builtins_obj, name, None))
        if isinstance(candidate, type) and issubclass(candidate, Exception):
            raise candidate(message)
    raise RuntimeError(message)


def _remove_owned_directory(path: str) -> None:
    # This directory is created per request by this backend and contains only
    # its staged request/result files.
    shutil.rmtree(path, ignore_errors=False)


class ProcessPdfBackend:
    """Thread-safe client for one spawned PyMuPDF worker with bounded cleanup."""

    def __init__(self, *, call_timeout=None,
                 cleanup_timeout: float = DEFAULT_PDF_CLEANUP_TIMEOUT_SECONDS,
                 temp_root=None, _worker_callable=None, _identity_probe=None,
                 _process_probe=None):
        self.call_timeout = _configured_call_timeout(call_timeout)
        self.cleanup_timeout = _positive_finite(cleanup_timeout, "PDF 清理超时")
        self._context = mp.get_context("spawn")
        self._call_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._closed = False
        self._broken = False
        self._cleanup_error: ResourceCleanupUncertain | None = None
        self._process = None
        self._connection = None
        self._resource = None
        self._process_token = None
        configured_temp_root = (temp_root if temp_root is not None
                                else os.environ.get(PDF_TEMP_DIR_ENV))
        self._temp_root = (os.fspath(configured_temp_root)
                           if configured_temp_root else None)
        self._worker_callable = _worker_callable or _worker_call
        self._identity_probe = _identity_probe or _process_identity
        self._process_probe = _process_probe or probe_process
        self._preserved_temp_dirs: list[str] = []
        self._worker_pids: set[int] = set()

    def _new_call_directory(self) -> str:
        if self._temp_root is not None:
            os.makedirs(self._temp_root, exist_ok=True)
        return tempfile.mkdtemp(prefix="rag-arch-pdf-", dir=self._temp_root)

    @staticmethod
    def _stage_request(operation: str, payload: dict, directory: str) -> tuple[str, str]:
        staged = dict(payload)
        if operation == "extract_document":
            data = staged.pop("data")
            suffix = Path(staged["filename"]).suffix or ".bin"
            data_path = os.path.join(directory, f"input{suffix}")
            Path(data_path).write_bytes(data)
            staged["data_path"] = data_path
        elif isinstance(staged.get("source"), bytes):
            source = staged["source"]
            suffix = "." + staged["suffix"] if operation == "image_to_pdf" else ".pdf"
            source_path = os.path.join(directory, f"input{suffix}")
            Path(source_path).write_bytes(source)
            staged["source"] = source_path
        request_path = os.path.join(directory, "request.pickle")
        response_path = os.path.join(directory, "response.pickle")
        with open(request_path, "wb") as stream:
            pickle.dump((operation, staged), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
        return request_path, response_path

    def _ensure_process(self, deadline: float) -> None:
        with self._state_lock:
            if self._process is not None:
                return
            if self._closed or self._broken:
                raise RuntimeError("PDF worker 已关闭。")
        # Process startup and resource persistence happen outside the state
        # lock so close() can set its fence while startup is in progress.
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_pdf_process_main,
            args=(child, self._worker_callable),
            name="rag-arch-pdf-worker",
            daemon=True,
        )
        resource = begin_resource(kind="pdf", scope="worker", owner_path=None)
        try:
            process.start()
        except BaseException:
            child.close()
            parent.close()
            if process.pid is None:
                # No child was published, so this resource never attached
                # to an operating-system process.
                resource.exited()
            else:
                with self._state_lock:
                    self._process = process
                    self._connection = parent
                    self._resource = resource
                self._resource_unknown("pdf_worker_start_identity_unknown")
            raise RuntimeError("PDF worker 启动失败。") from None
        child.close()
        pid = process.pid
        with self._state_lock:
            self._process = process
            self._connection = parent
            self._resource = resource
        if pid is None:
            self._resource_unknown("pdf_worker_pid_unavailable")
        try:
            identity = self._identity_probe(pid)
        except (psutil.Error, OSError):
            self._resource_unknown("pdf_worker_identity_probe_failed")
        if identity is None:
            self._resource_unknown("pdf_worker_identity_unavailable")
        token, executable = identity
        with self._state_lock:
            self._process_token = token
        try:
            resource.attach_process(pid, token, executable)
        except ResourceCleanupUncertain as exc:
            with self._state_lock:
                self._cleanup_error = exc
                self._broken = True
                self._closed = True
            raise
        with self._state_lock:
            self._worker_pids.add(pid)
        # Spawning is normally quick, but include it in the request's time budget.
        if time.monotonic() > deadline:
            self._abort_process("pdf_worker_start_timeout", retire_owner=True)
            raise TimeoutError("PDF worker 启动超时。")

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _acquire_call_lock(self, deadline: float) -> bool:
        while True:
            check_request_control()
            remaining = self._remaining(deadline)
            if remaining <= 0:
                return False
            if self._call_lock.acquire(timeout=min(remaining, PDF_CONTROL_POLL_SECONDS)):
                return True

    def _bytes(self, value, name: str) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(name + "必须是字节数据。")
        return bytes(value)

    def _source(self, value, name: str):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._bytes(value, name)
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
        raise TypeError(name + "必须是字节数据或本地路径。")

    def _call(self, operation: str, payload: dict):
        deadline = time.monotonic() + self.call_timeout
        if not self._acquire_call_lock(deadline):
            raise TimeoutError("等待 PDF worker 超时。")
        call_directory = None
        preserve_directory = False
        try:
            check_request_control()
            with self._state_lock:
                if self._cleanup_error is not None:
                    raise self._cleanup_error
                if self._closed or self._broken:
                    raise RuntimeError("PDF worker 已关闭。")
            try:
                call_directory = self._new_call_directory()
                request_path, response_path = self._stage_request(
                    operation, payload, call_directory)
            except (OSError, pickle.PickleError):
                raise RuntimeError("PDF worker 临时文件准备失败。") from None
            check_request_control()
            self._ensure_process(deadline)
            if self._remaining(deadline) <= 0:
                self._abort_process("pdf_worker_call_timeout", retire_owner=True)
                raise TimeoutError("PDF worker 调用超时。")
            self._resource.used()
            check_request_control()
            connection = self._connection
            try:
                connection.send_bytes(pickle.dumps(
                    (request_path, response_path), protocol=pickle.HIGHEST_PROTOCOL))
            except (EOFError, OSError, BrokenPipeError) as exc:
                self._abort_process("pdf_worker_transport_failed", retire_owner=True)
                raise RuntimeError("PDF worker 已异常退出。") from exc
            while True:
                try:
                    check_request_control()
                except BaseException:
                    self._abort_process("pdf_worker_request_cancelled", retire_owner=True)
                    raise
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    self._abort_process("pdf_worker_call_timeout", retire_owner=True)
                    raise TimeoutError("PDF worker 调用超时。")
                if connection.poll(min(remaining, PDF_CONTROL_POLL_SECONDS)):
                    break
            try:
                response = pickle.loads(connection.recv_bytes(maxlength=1024))
            except (EOFError, OSError, pickle.PickleError, ValueError) as exc:
                self._abort_process("pdf_worker_response_invalid", retire_owner=True)
                raise RuntimeError("PDF worker 返回数据无效。") from exc
            if not (isinstance(response, tuple) and len(response) == 2
                    and response[0] == "result"):
                self._abort_process("pdf_worker_response_invalid", retire_owner=True)
                raise RuntimeError("PDF worker 返回数据无效。")
            if time.monotonic() > deadline:
                self._abort_process("pdf_worker_result_timeout", retire_owner=True)
                raise TimeoutError("PDF worker 返回超时。")
            try:
                with open(response_path, "rb") as stream:
                    envelope = pickle.load(stream)
            except (OSError, EOFError, pickle.PickleError) as exc:
                self._abort_process("pdf_worker_result_missing", retire_owner=True)
                raise RuntimeError("PDF worker 返回文件无效。") from exc
            if not isinstance(envelope, tuple) or not envelope:
                self._abort_process("pdf_worker_result_invalid", retire_owner=True)
                raise RuntimeError("PDF worker 返回数据无效。")
            if envelope[0] == "error" and len(envelope) == 4:
                _raise_worker_error(envelope[1], envelope[2], envelope[3])
            if envelope[0] != "ok" or len(envelope) != 2:
                self._abort_process("pdf_worker_result_invalid", retire_owner=True)
                raise RuntimeError("PDF worker 返回数据无效。")
            check_request_control()
            return envelope[1]
        except ResourceCleanupUncertain:
            preserve_directory = True
            raise
        finally:
            try:
                if call_directory is not None:
                    if preserve_directory:
                        self._preserved_temp_dirs.append(call_directory)
                    else:
                        try:
                            _remove_owned_directory(call_directory)
                        except FileNotFoundError:
                            pass
                        except OSError:
                            try:
                                self._abort_process(
                                    "pdf_worker_temp_cleanup_failed", retire_owner=True)
                            except ResourceCleanupUncertain:
                                self._preserved_temp_dirs.append(call_directory)
                                raise
                            try:
                                _remove_owned_directory(call_directory)
                            except FileNotFoundError:
                                pass
                            except OSError:
                                self._preserved_temp_dirs.append(call_directory)
                            raise RuntimeError("PDF worker 临时文件清理失败。") from None
            finally:
                self._call_lock.release()

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Worker identities observed by this client; useful for diagnostics/tests."""
        with self._state_lock:
            return tuple(sorted(self._worker_pids))

    def validate(self, source) -> int:
        return self._call("validate", {"source": self._source(source, "PDF")})

    def extract_document(self, filename: str, data) -> LoadedDocument:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("文件名不能为空。")
        return self._call("extract_document", {
            "filename": filename, "data": self._bytes(data, "PDF")})

    def image_to_pdf(self, source, suffix: str) -> bytes:
        if not isinstance(suffix, str) or not suffix:
            raise ValueError("图片格式不能为空。")
        return self._call("image_to_pdf", {
            "source": self._source(source, "图片"), "suffix": suffix.lstrip(".").lower()})

    def list_pages(self, source, start: int = 1, limit: int = 20, query: str = "") -> dict:
        return self._call("list_pages", {
            "source": self._source(source, "PDF"), "start": start,
            "limit": limit, "query": query})

    def scan_visual_pages(self, source) -> dict:
        """Return a complete metadata-only visual candidate index for one PDF."""
        return self._call("scan_visual_pages", {"source": self._source(source, "PDF")})

    def visual_page_index(self, source) -> dict:
        """Compatibility spelling for callers that treat the result as an index."""
        return self.scan_visual_pages(source)

    def read_page(self, source, page: int, scale: float = 1.5) -> dict:
        return self._call("read_page", {
            "source": self._source(source, "PDF"), "page": page, "scale": scale})

    def _clear_process(self) -> None:
        with self._state_lock:
            connection = self._connection
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            self._process = None
            self._connection = None
            self._resource = None
            self._process_token = None

    def _resource_unknown(self, code: str):
        with self._state_lock:
            self._broken = True
            self._closed = True
            resource = self._resource
        if resource is None:
            error = ResourceCleanupUncertain(code)
            with self._state_lock:
                self._cleanup_error = error
            raise error
        try:
            resource.unknown(code)
        except ResourceCleanupUncertain as exc:
            with self._state_lock:
                self._cleanup_error = exc
            raise
        error = ResourceCleanupUncertain(code)
        with self._state_lock:
            self._cleanup_error = error
        raise error

    def _mark_resource_exited(self, resource, *, retire_owner: bool) -> None:
        if resource is not None:
            try:
                resource.exited()
                if retire_owner:
                    resource.retire_owner()
            except ResourceCleanupUncertain:
                raise
            except Exception:
                self._resource_unknown("pdf_worker_exit_record_failed")

    def _abort_process(self, error_code: str, *, retire_owner: bool) -> None:
        with self._state_lock:
            process = self._process
            resource = self._resource
            expected_token = self._process_token
        if process is None:
            if resource is None:
                # No child has ever been published for this backend. A failed
                # request-file cleanup must not permanently poison a reusable
                # backend or retire a worker without an owned process.
                return
            with self._state_lock:
                self._broken = True
                self._closed = True
            self._mark_resource_exited(resource, retire_owner=retire_owner)
            self._clear_process()
            return
        with self._state_lock:
            self._broken = True
            self._closed = True
        pid = process.pid
        if pid is None or expected_token is None:
            self._resource_unknown(error_code + "_identity_missing")
        try:
            identity_state = self._process_probe(
                pid, expected_token,
                executable=resource.executable if resource is not None else None)
        except (psutil.Error, OSError):
            self._resource_unknown(error_code + "_identity_probe_failed")
        if identity_state == "exited":
            # The original process has exited or the PID now names another
            # process. The supervision handle performs its own exact check.
            self._mark_resource_exited(resource, retire_owner=retire_owner)
            try:
                process.join(timeout=0)
            except (AssertionError, OSError):
                pass
            self._clear_process()
            return
        if identity_state != "running":
            self._resource_unknown(error_code + "_identity_unknown")
        pidfd = None
        try:
            if os.name == "nt":
                # Windows Popen owns a kernel process handle; terminate() targets
                # this exact child and cannot hit a later PID reuse.
                process.terminate()
            elif hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                pidfd = os.pidfd_open(pid)
                current_state = self._process_probe(
                    pid, expected_token,
                    executable=resource.executable if resource is not None else None)
                if current_state == "exited":
                    self._mark_resource_exited(resource, retire_owner=retire_owner)
                    self._clear_process()
                    return
                if current_state != "running":
                    self._resource_unknown(error_code + "_identity_unknown")
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            else:
                self._resource_unknown(error_code + "_safe_termination_unavailable")
            process.join(timeout=self.cleanup_timeout)
            if process.is_alive():
                if os.name == "nt":
                    process.kill()
                elif pidfd is not None:
                    current_state = self._process_probe(
                        pid, expected_token,
                        executable=resource.executable if resource is not None else None)
                    if current_state == "exited":
                        self._mark_resource_exited(resource, retire_owner=retire_owner)
                        self._clear_process()
                        return
                    if current_state != "running":
                        self._resource_unknown(error_code + "_identity_unknown")
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                process.join(timeout=self.cleanup_timeout)
            if process.is_alive():
                self._resource_unknown(error_code + "_termination_unconfirmed")
        except ResourceCleanupUncertain:
            raise
        except (OSError, psutil.Error, AssertionError):
            self._resource_unknown(error_code + "_termination_failed")
        finally:
            if pidfd is not None:
                try:
                    os.close(pidfd)
                except OSError:
                    pass
        self._mark_resource_exited(resource, retire_owner=retire_owner)
        self._clear_process()

    def _stop_process(self) -> None:
        with self._state_lock:
            process = self._process
            connection = self._connection
            resource = self._resource
        if process is None:
            return
        try:
            connection.send_bytes(pickle.dumps(("stop",)))
            if not connection.poll(self.cleanup_timeout):
                self._abort_process("pdf_worker_close_timeout", retire_owner=True)
                return
            response = pickle.loads(connection.recv_bytes(maxlength=1024))
            if not (isinstance(response, tuple) and len(response) == 2
                    and response[0] == "stopped"):
                self._abort_process("pdf_worker_close_response_invalid", retire_owner=True)
                return
            process.join(timeout=self.cleanup_timeout)
            if process.is_alive():
                self._abort_process("pdf_worker_close_timeout", retire_owner=True)
                return
        except ResourceCleanupUncertain:
            raise
        except (EOFError, OSError, pickle.PickleError, ValueError):
            self._abort_process("pdf_worker_close_failed", retire_owner=True)
            return
        self._mark_resource_exited(resource, retire_owner=False)
        self._clear_process()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                if self._cleanup_error is not None:
                    raise self._cleanup_error
                return
            self._closed = True
        # Active calls own the call lock and have their own finite deadline.
        # Closing waits one request timeout plus bounded process cleanup.
        wait = self.call_timeout + (2 * self.cleanup_timeout) + 0.5
        if not self._call_lock.acquire(timeout=wait):
            self._abort_process("pdf_worker_close_wait_timeout", retire_owner=True)
            self._resource_unknown("pdf_worker_close_wait_unconfirmed")
        try:
            self._stop_process()
        finally:
            self._call_lock.release()


# Retain the shorter name for local callers that do not need to spell out the process boundary.
PdfBackend = ProcessPdfBackend
