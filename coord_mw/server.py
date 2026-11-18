"""Loopback TCP JSON-lines coordination server."""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from .engine import CoordinatorEngine
from .errors import CoordError, port_error, protocol_error
from .util import JsonlLogger, canonical_json, is_int, read_json


MAX_LINE_BYTES = 1_048_576


def _required_string(request: dict[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value:
        raise protocol_error(f"missing or invalid field: {field}", field=field)
    return value


def _required_int(request: dict[str, Any], field: str) -> int:
    value = request.get(field)
    if not is_int(value):
        raise protocol_error(f"missing or invalid integer field: {field}", field=field)
    return value


class CoordinationRequestHandler(socketserver.StreamRequestHandler):
    MAX_LINE_BYTES = MAX_LINE_BYTES

    def handle(self) -> None:
        while True:
            line = self.rfile.readline(self.MAX_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > self.MAX_LINE_BYTES:
                self._send_error(protocol_error("request line exceeds protocol limit"))
                return
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_error(protocol_error(f"invalid JSON line: {exc}"))
                continue
            if not isinstance(request, dict):
                self._send_error(protocol_error("request must be a JSON object"))
                continue
            response = self.server.app.dispatch(request)  # type: ignore[attr-defined]
            self._send(response)
            if self.server.app.stop_event.is_set():  # type: ignore[attr-defined]
                return

    def _send_error(self, error: CoordError) -> None:
        self.server.app.logger.write(  # type: ignore[attr-defined]
            "request_rejected",
            code=error.code,
            message=error.message,
            details=error.details,
        )
        self._send(error.as_dict())

    def _send(self, response: dict[str, Any]) -> None:
        try:
            self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


class ServerApplication:
    def __init__(self, engine: CoordinatorEngine, logger: JsonlLogger) -> None:
        self.engine = engine
        self.logger = logger
        self.stop_event = threading.Event()

    @staticmethod
    def _validate_common(request: dict[str, Any]) -> str:
        op = request.get("op")
        if not isinstance(op, str) or not op:
            raise protocol_error("op must be a non-empty string", field="op")
        return op

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            op = self._validate_common(request)
            if op == "vote":
                result = self.engine.vote(
                    _required_int(request, "term"),
                    _required_string(request, "member"),
                    _required_string(request, "candidate"),
                )
            elif op == "heartbeat":
                result = self.engine.heartbeat(
                    _required_int(request, "term"),
                    _required_string(request, "member"),
                    _required_int(request, "lease_version"),
                )
            elif op == "query_leader":
                result = self.engine.query_leader()
            elif op == "query_state":
                result = self.engine.query_state()
            elif op == "apply_change":
                changes = request.get("changes")
                if changes is None:
                    changes = {
                        key: value
                        for key, value in request.items()
                        if key != "op"
                    }
                if not isinstance(changes, dict) or not changes:
                    raise protocol_error("apply_change requires a non-empty changes object")
                result = self.engine.apply_changeset(changes)
            elif op == "invalidate_token":
                result = self.engine.invalidate_token(_required_string(request, "token"))
            elif op == "advance_term":
                member = request.get("member")
                if member is not None and (not isinstance(member, str) or not member):
                    raise protocol_error("member must be a non-empty string", field="member")
                result = self.engine.advance_term(member)
            elif op == "snapshot":
                path = request.get("path")
                if path is not None and not isinstance(path, str):
                    raise protocol_error("path must be a string", field="path")
                result = self.engine.create_snapshot(path)
            elif op == "restore_snapshot":
                result = self.engine.restore_snapshot(_required_string(request, "path"))
            elif op == "reap":
                result = {"reaped": self.engine.reap_expired()}
            elif op == "ping":
                result = {"pong": True, "state_version": self.engine.state_version}
            elif op == "shutdown":
                result = {"shutdown": True, "state_version": self.engine.state_version}
                self.stop_event.set()
            else:
                raise protocol_error(f"unsupported operation: {op}", op=op)
            self.logger.write("request_completed", op=op)
            return {"ok": True, **result}
        except CoordError as exc:
            self.logger.write(
                "request_rejected",
                code=exc.code,
                message=exc.message,
                details=exc.details,
                request_op=request.get("op") if isinstance(request, dict) else None,
            )
            return exc.as_dict()
        except Exception as exc:  # defensive boundary for a long-lived local server
            self.logger.write(
                "request_internal_error",
                code="E_INTERNAL",
                message=str(exc),
                request_op=request.get("op") if isinstance(request, dict) else None,
            )
            return CoordError("E_INTERNAL", "internal server error", cause=str(exc)).as_dict()


class CoordinatorTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True
    request_queue_size = 64


def build_engine(args: argparse.Namespace) -> CoordinatorEngine:
    definition = None
    if args.members:
        definition = read_json(Path(args.members))
    return CoordinatorEngine(
        args.root,
        definition=definition,
        mark_unknown_on_load=True,
    )


def serve(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    server_logger = JsonlLogger(root / "logs" / "server.jsonl")
    try:
        engine = build_engine(args)
    except CoordError as exc:
        server_logger.write(
            "server_start_rejected",
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
        print(canonical_json(exc.as_dict()), flush=True)
        return 2
    except Exception as exc:
        error = CoordError("E_STATE", f"cannot initialize server: {exc}")
        server_logger.write("server_start_rejected", **error.as_dict())
        print(canonical_json(error.as_dict()), flush=True)
        return 2

    app = ServerApplication(engine, server_logger)
    stop_event = app.stop_event
    try:
        server = CoordinatorTCPServer((args.host, args.port), CoordinationRequestHandler)
    except OSError as exc:
        error = port_error(
            "cannot bind coordination port",
            host=args.host,
            port=args.port,
            cause=str(exc),
        )
        server_logger.write("server_start_rejected", **error.as_dict())
        print(canonical_json(error.as_dict()), flush=True)
        return 3
    server.app = app  # type: ignore[attr-defined]
    server.timeout = 0.1

    def reaper() -> None:
        while not stop_event.wait(0.025):
            try:
                engine.reap_expired()
            except Exception as exc:
                server_logger.write(
                    "reaper_error",
                    code="E_INTERNAL",
                    message=str(exc),
                )

    reaper_thread = threading.Thread(target=reaper, name="lease-reaper", daemon=True)
    reaper_thread.start()
    ready = {
        "ready": True,
        "host": args.host,
        "port": server.server_address[1],
        "root": str(root),
        "state_version": engine.state_version,
    }
    server_logger.write("server_ready", **ready)
    if args.ready_file:
        Path(args.ready_file).write_text(canonical_json(ready) + "\n", encoding="utf-8")
    print(canonical_json(ready), flush=True)

    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        server_logger.write("server_stopped", state_version=engine.state_version)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local coordination node server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28765)
    parser.add_argument("--root", default="runtime")
    parser.add_argument("--members", default=None)
    parser.add_argument("--ready-file", default=None)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        print(
            canonical_json(
                CoordError(
                    "E_PROTO",
                    "the lab only accepts the 127.0.0.1 loopback address",
                    host=args.host,
                ).as_dict()
            ),
            flush=True,
        )
        return 2
    if not (1024 <= args.port <= 65535):
        print(
            canonical_json(
                CoordError(
                    "E_PORT",
                    "port must be a high TCP port in [1024, 65535]",
                    port=args.port,
                ).as_dict()
            ),
            flush=True,
        )
        return 2
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())

