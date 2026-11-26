from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .determinism import deterministic_body, failure_selected
from .protocol import PROTOCOL_NAME, encode_line, monotonic_now_ns
from .specs import DiagnosticConfig, HopSpec, OriginSpec, load_bundle


Handler = Callable[[dict[str, Any]], dict[str, Any]]


class JsonlNodeServer:
    def __init__(self, *, host: str, port: int, name: str, handler: Handler,
                 connect_timeout_seconds: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.name = name
        self.handler = handler
        self.connect_timeout_seconds = connect_timeout_seconds
        self._stop_event = threading.Event()
        self._sock: socket.socket | None = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def serve_forever(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(128)
        server.settimeout(0.25)
        self._sock = server
        try:
            while not self._stop_event.is_set():
                try:
                    conn, _address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                thread = threading.Thread(
                    target=self._safe_handle,
                    args=(conn,),
                    name=f"jsonl-{self.name}",
                    daemon=True,
                )
                thread.start()
        finally:
            try:
                server.close()
            except OSError:
                pass

    def _safe_handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(self.connect_timeout_seconds)
            conn.sendall(encode_line({
                "type": "hello_ack",
                "protocol": PROTOCOL_NAME,
                "node_id": self.name,
            }))
            chunks = bytearray()
            while True:
                char = conn.recv(1)
                if not char or char == b"\n":
                    break
                chunks.extend(char)
                if len(chunks) > 4 * 1024 * 1024:
                    raise ValueError("request line too large")
            if not chunks:
                return
            message = json.loads(chunks.decode("utf-8"))
            response = self.handler(message)
            conn.sendall(encode_line(response))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = {
                "type": "node_error",
                "ok": False,
                "error": {"code": "E_NODE_PROTOCOL", "message": str(exc)},
            }
            try:
                conn.sendall(encode_line(error))
            except OSError:
                pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _elapsed_ms(start_ns: int) -> int:
    return max(0, (monotonic_now_ns() - start_ns) // 1_000_000)


def make_hop_handler(hop: HopSpec) -> Handler:
    def handle(message: dict[str, Any]) -> dict[str, Any]:
        request_id = str(message.get("request_id", ""))
        start_ns = monotonic_now_ns()
        if hop.queue_ms:
            time.sleep(hop.queue_ms / 1000.0)
        if hop.process_ms:
            time.sleep(hop.process_ms / 1000.0)

        failed = failure_selected(hop.hop_id, request_id, hop.failure_ratio_bp)
        receipt = {
            "schema": "hopledger.receipt/1",
            "type": "hop_receipt",
            "kind": "hop",
            "hop_id": hop.hop_id,
            "node_id": hop.hop_id,
            "request_id": request_id,
            "config_version": hop.config_version,
            "scale_factor": hop.scale_factor,
            "queue_ms": hop.queue_ms,
            "process_ms": hop.process_ms,
            "forward_ms": hop.forward_estimate_ms,
            "monotonic_ms": _elapsed_ms(start_ns),
            "status": "forward_failed" if failed else "ok",
            "failure_injected": failed,
        }
        if failed:
            return {
                "type": "hop_error",
                "ok": False,
                "error": {
                    "code": "E_FORWARD_INJECTED",
                    "hop_id": hop.hop_id,
                    "request_id": request_id,
                },
                "receipt": receipt,
            }
        return {"type": "hop_receipt", "ok": True, "receipt": receipt}

    return handle


def make_origin_handler(origin: OriginSpec, diagnostic: DiagnosticConfig) -> Handler:
    def handle(message: dict[str, Any]) -> dict[str, Any]:
        request_id = str(message.get("request_id", ""))
        start_ns = monotonic_now_ns()
        sample = origin.sample_for(request_id)
        body = deterministic_body(origin.seed, sample.seed_offset, origin.body_bytes)
        receipt = {
            "schema": "hopledger.receipt/1",
            "type": "origin_receipt",
            "kind": "origin",
            "hop_id": origin.origin_id,
            "node_id": origin.origin_id,
            "request_id": request_id,
            "config_version": origin.config_version,
            "scale_factor": diagnostic.session_scale,
            "queue_ms": 0,
            "process_ms": 0,
            "forward_ms": 0,
            "monotonic_ms": _elapsed_ms(start_ns),
            "status": "ok",
            "body_bytes": len(body),
            "body_digest": __import__("hashlib").sha256(body).hexdigest(),
        }
        return {
            "type": "origin_receipt",
            "ok": True,
            "receipt": receipt,
            "body_b64": base64.b64encode(body).decode("ascii"),
        }

    return handle


def run_node(config_dir: str | Path, *, kind: str, hop_id: str | None = None) -> int:
    bundle = load_bundle(config_dir)
    if kind == "hop":
        if not hop_id:
            raise SystemExit("--hop-id is required for hop nodes")
        spec = bundle.chain.hop(hop_id)
        host, port, name = "127.0.0.1", spec.port, spec.hop_id
        handler = make_hop_handler(spec)
    elif kind == "origin":
        spec = bundle.origin
        host, port, name = "127.0.0.1", spec.port, spec.origin_id
        handler = make_origin_handler(spec, bundle.diagnostic)
    else:
        raise SystemExit(f"unknown node kind {kind!r}")

    server = JsonlNodeServer(host=host, port=port, name=name, handler=handler)
    try:
        # Bind inside serve_forever; an early preflight gives a structured E_PORT line.
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            test_socket.bind((host, port))
        except OSError as exc:
            print(json.dumps({
                "level": "error",
                "code": "E_PORT",
                "node_id": name,
                "port": port,
                "error": str(exc),
            }), file=sys.stderr, flush=True)
            return 2
        finally:
            test_socket.close()
        # Closing the preflight socket can leave a tiny race; the real bind still
        # reports E_PORT through the exception path if another process wins.
        print(json.dumps({
            "level": "ready",
            "node_id": name,
            "kind": kind,
            "pid": __import__("os").getpid(),
            "host": host,
            "port": port,
        }), flush=True)
        server.serve_forever()
    except OSError as exc:
        print(json.dumps({
            "level": "error",
            "code": "E_PORT",
            "node_id": name,
            "port": port,
            "error": str(exc),
        }), file=sys.stderr, flush=True)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hopledger-node")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--kind", choices=("hop", "origin"), required=True)
    parser.add_argument("--hop-id")
    args = parser.parse_args(argv)
    return run_node(args.config_dir, kind=args.kind, hop_id=args.hop_id)


if __name__ == "__main__":
    raise SystemExit(main())
