"""Loopback TCP storage-service simulator with deterministic upload throttling."""

from __future__ import annotations

import json
import socketserver
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import Workbench
from .errors import E_CONFLICT, E_INPUT, E_INTERNAL, BusinessError
from .logging import JsonlLogger
from .protocol import MAX_FRAME_BYTES, PROTOCOL_NAME
from .util import canonical_text, mono_ms


class UploadThrottle:
    def __init__(self, global_limit: int, session_limit: int) -> None:
        if global_limit <= 0 or session_limit <= 0:
            raise BusinessError(E_INPUT, "concurrency limits must be positive integers")
        self.global_limit = global_limit
        self.session_limit = session_limit
        self._condition = threading.Condition()
        self._active_total = 0
        self._active_by_session: dict[str, int] = {}
        self.peak_global = 0
        self.peak_by_session: dict[str, int] = {}

    def acquire(self, session_id: str) -> tuple[int, int]:
        queued_at = mono_ms()
        with self._condition:
            while (
                self._active_total >= self.global_limit
                or self._active_by_session.get(session_id, 0) >= self.session_limit
            ):
                self._condition.wait()
            started_at = mono_ms()
            self._active_total += 1
            current_session = self._active_by_session.get(session_id, 0) + 1
            self._active_by_session[session_id] = current_session
            self.peak_global = max(self.peak_global, self._active_total)
            self.peak_by_session[session_id] = max(
                self.peak_by_session.get(session_id, 0), current_session
            )
            return started_at - queued_at, started_at

    def release(self, session_id: str) -> None:
        with self._condition:
            self._active_total -= 1
            current = self._active_by_session.get(session_id, 0) - 1
            if current <= 0:
                self._active_by_session.pop(session_id, None)
            else:
                self._active_by_session[session_id] = current
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "global_limit": self.global_limit,
                "session_limit": self.session_limit,
                "active_global": self._active_total,
                "active_by_session": dict(self._active_by_session),
                "peak_global": self.peak_global,
                "peak_by_session": dict(self.peak_by_session),
            }


@dataclass
class ServerContext:
    engine: Workbench
    throttle: UploadThrottle
    logger: JsonlLogger
    host: str
    port: int
    started_mono_ms: int = field(default_factory=mono_ms)
    request_count: int = 0
    error_count: int = 0
    _counter_lock: threading.Lock = field(default_factory=threading.Lock)

    def count_request(self, failed: bool) -> None:
        with self._counter_lock:
            self.request_count += 1
            if failed:
                self.error_count += 1


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        context: ServerContext = self.server.context  # type: ignore[attr-defined]
        line = self.rfile.readline(MAX_FRAME_BYTES + 1)
        if not line:
            return
        request_id = "unknown"
        op = "unknown"
        try:
            if len(line) > MAX_FRAME_BYTES:
                raise BusinessError(E_INPUT, "request frame exceeds the size limit")
            request = json.loads(line.decode("utf-8"))
            request_id = str(request.get("request_id", ""))
            op = str(request.get("op", ""))
            if request.get("protocol") != PROTOCOL_NAME:
                raise BusinessError(
                    E_INPUT,
                    "unsupported protocol",
                    {"expected": PROTOCOL_NAME, "actual": request.get("protocol")},
                )
            if not request_id:
                raise BusinessError(E_INPUT, "request_id is required")
            params = request.get("params")
            if not isinstance(params, dict):
                raise BusinessError(E_INPUT, "params must be a JSON object")
            result = dispatch(context, op, params)
            response = {"request_id": request_id, "ok": True, "result": result}
            context.count_request(False)
            context.logger.event(
                "rpc_success",
                request_id=request_id,
                op=op,
                session_id=params.get("session_id"),
                object_key=params.get("object_key"),
            )
        except BusinessError as exc:
            context.count_request(True)
            response = {
                "request_id": request_id,
                "ok": False,
                "error": exc.as_dict(),
            }
            context.logger.event(
                "rpc_error",
                request_id=request_id,
                op=op,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        except Exception as exc:  # noqa: BLE001 - service boundary must answer clients
            context.count_request(True)
            error = BusinessError(
                E_INTERNAL,
                "unhandled storage service error",
                {"type": type(exc).__name__, "reason": str(exc)},
            )
            response = {
                "request_id": request_id,
                "ok": False,
                "error": error.as_dict(),
            }
            context.logger.event(
                "rpc_error",
                request_id=request_id,
                op=op,
                code=error.code,
                message=error.message,
                details=error.details,
            )
        wire = (canonical_text(response) + "\n").encode("utf-8")
        try:
            self.wfile.write(wire)
            self.wfile.flush()
        except OSError:
            pass


def _summarize_params(params: dict[str, Any]) -> dict[str, Any]:
    summary = dict(params)
    if "data_b64" in summary:
        summary["data_b64"] = f"<{len(str(summary['data_b64']))} base64 chars>"
    return summary


def dispatch(context: ServerContext, op: str, params: dict[str, Any]) -> dict[str, Any]:
    engine = context.engine
    if op == "ping":
        return {
            "ok": True,
            "service": "chunkdeps-storage-simulator",
            "protocol": PROTOCOL_NAME,
            "host": context.host,
            "port": context.port,
            "uptime_ms": mono_ms() - context.started_mono_ms,
            "throttle": context.throttle.snapshot(),
            "recovered_sessions": len(engine.recovered_on_start),
            "request_count": context.request_count,
            "error_count": context.error_count,
        }

    if op == "upload_chunk":
        session_id = str(params.get("session_id", ""))
        queued_ms, started = context.throttle.acquire(session_id)
        result: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            result = engine.upload_chunk(params)
        except Exception as exc:  # release must always occur
            error = exc
        finally:
            ended = mono_ms()
            context.throttle.release(session_id)
        index = params.get("chunk_index")
        chunk_index = index if isinstance(index, int) and not isinstance(index, bool) else -1
        schedule = engine.ledger.record_schedule(
            session_id=session_id,
            chunk_index=chunk_index,
            queue_ms=queued_ms,
            start_mono_ms=started,
            end_mono_ms=ended,
            global_limit=context.throttle.global_limit,
            session_limit=context.throttle.session_limit,
            injection=params.get("inject"),
        )
        if error is not None:
            raise error
        assert result is not None
        result["schedule"] = schedule
        result["throttle"] = context.throttle.snapshot()
        return result

    methods = {
        "create_session": engine.create_session,
        "get_session": engine.get_session,
        "resume_session": engine.resume_session,
        "abort_session": engine.abort_session,
        "commit_session": engine.commit_session,
        "delete_object": engine.delete_object,
        "audit": engine.audit,
        "snapshot_save": engine.snapshot_save,
        "snapshot_restore": engine.snapshot_restore,
        "list_sessions": engine.list_sessions,
        "list_objects": engine.list_objects,
        "get_graph": engine.get_graph,
        "invalidations": engine.invalidations,
        "status": engine.status,
        "recovery_status": engine.recovery_status,
        "schedules": engine.schedules,
        "list_views": engine.list_views,
    }
    method = methods.get(op)
    if method is None:
        raise BusinessError(E_INPUT, "unknown operation", {"op": op})
    return method(params)


class StorageService:
    def __init__(
        self,
        root: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        global_concurrency: int = 4,
        session_concurrency: int = 2,
        log_path: str | Path | None = None,
        merge_pause_ms: int | None = None,
    ) -> None:
        logger = JsonlLogger(Path(log_path) if log_path else Path(root) / "logs" / "service.jsonl")
        engine = Workbench(root, logger=logger, merge_pause_ms=merge_pause_ms)
        self.context = ServerContext(
            engine=engine,
            throttle=UploadThrottle(global_concurrency, session_concurrency),
            logger=logger,
            host=host,
            port=port,
        )
        self.server = ThreadedTCPServer((host, port), RequestHandler)
        self.server.context = self.context  # type: ignore[attr-defined]
        self.context.port = int(self.server.server_address[1])
        self.context.logger.event(
            "service_started",
            host=host,
            port=self.context.port,
            global_concurrency=global_concurrency,
            session_concurrency=session_concurrency,
            root=str(engine.ledger.root),
        )

    @property
    def host(self) -> str:
        return self.context.host

    @property
    def port(self) -> int:
        return self.context.port

    @property
    def engine(self) -> Workbench:
        return self.context.engine

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=0.1)

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.context.logger.event("service_stopped", port=self.port)
        self.engine.close()

    def __enter__(self) -> "StorageService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
