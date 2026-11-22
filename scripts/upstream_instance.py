#!/usr/bin/env python3
"""Deterministic loopback upstream instance for Meshwork demos.

Protocol: one JSON object per TCP connection followed by a newline. The process
does not access the network beyond binding the requested 127.0.0.1 port.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
from typing import Any

PROTOCOL_VERSION = 1


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def emit_error(code: str, message: str, **fields: Any) -> None:
    record = {"code": code, "message": message, "process": "upstream_instance"}
    record.update(fields)
    sys.stderr.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    sys.stderr.flush()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def classify(seed: int, request_id: str, attempt_no: int, behavior: dict[str, Any]) -> tuple[str, str, int]:
    material = canonical(
        {"seed": seed, "request_id": request_id, "attempt_no": attempt_no}
    )
    digest = hashlib.sha256(material).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    latency_class = "fast"
    outcome = "success"
    success_percent = int(behavior["success_percent"])
    slow_percent = int(behavior["slow_percent"])
    error_percent = int(behavior["error_percent"])
    if bucket < success_percent:
        outcome = "success"
        latency_class = "fast"
    elif bucket < success_percent + slow_percent:
        outcome = "success"
        latency_class = "slow"
    elif bucket < success_percent + slow_percent + error_percent:
        outcome = "error"
        latency_class = "fast"
    for rule in behavior.get("rules", []):
        if request_id.startswith(str(rule["request_prefix"])):
            outcome = str(rule["outcome"])
            latency_class = str(rule.get("latency_class", "fast"))
            break
    delay_ms = int(behavior["slow_ms"] if latency_class == "slow" else behavior["fast_ms"])
    return outcome, latency_class, delay_ms


def send_line(conn: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    conn.sendall(payload)


def recv_line(conn: socket.socket) -> bytes:
    data = bytearray()
    while len(data) <= 1_048_576:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    return bytes(data).partition(b"\n")[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--behavior-json", required=True)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        emit_error("E_SPEC", "host must be 127.0.0.1", instance_id=args.instance_id)
        return 2
    try:
        behavior = json.loads(args.behavior_json)
    except json.JSONDecodeError as exc:
        emit_error("E_SPEC", f"invalid behavior JSON: {exc}", instance_id=args.instance_id)
        return 2
    stop_event = threading.Event()

    def parent_watchdog() -> None:
        try:
            while True:
                chunk = sys.stdin.buffer.read(4096)
                if not chunk:
                    break
        except Exception:
            pass
        stop_event.set()

    watchdog = threading.Thread(target=parent_watchdog, name="parent-watchdog", daemon=True)
    watchdog.start()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(64)
    except OSError as exc:
        emit_error(
            "E_PORT",
            f"cannot bind {args.host}:{args.port}: {exc}",
            instance_id=args.instance_id,
            port=args.port,
        )
        server.close()
        return 3
    server.settimeout(0.2)
    try:
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(5.0)
                try:
                    line = recv_line(conn)
                    if not line:
                        continue
                    request = json.loads(line.decode("utf-8"))
                    if request.get("type") == "ping":
                        send_line(
                            conn,
                            {
                                "protocol": PROTOCOL_VERSION,
                                "type": "pong",
                                "instance_id": args.instance_id,
                                "service_id": args.service_id,
                                "pid": os.getpid(),
                            },
                        )
                        continue
                    if request.get("type") != "request":
                        send_line(
                            conn,
                            {
                                "protocol": PROTOCOL_VERSION,
                                "request_id": str(request.get("request_id", "")),
                                "attempt_no": int(request.get("attempt_no", 0)),
                                "instance_id": args.instance_id,
                                "status": "error",
                                "latency_class": "fast",
                                "delay_ms": 0,
                                "detail": "unsupported message type",
                            },
                        )
                        continue
                    request_id = str(request["request_id"])
                    attempt_no = int(request["attempt_no"])
                    outcome, latency_class, delay_ms = classify(
                        args.seed, request_id, attempt_no, behavior
                    )
                    time.sleep(delay_ms / 1000.0)
                    send_line(
                        conn,
                        {
                            "protocol": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "attempt_no": attempt_no,
                            "instance_id": args.instance_id,
                            "service_id": args.service_id,
                            "status": "ok" if outcome == "success" else "error",
                            "outcome": outcome,
                            "latency_class": latency_class,
                            "delay_ms": delay_ms,
                            "seed": args.seed,
                            "detail": "deterministic_sample",
                        },
                    )
                except (BrokenPipeError, ConnectionResetError, socket.timeout):
                    continue
                except Exception as exc:
                    try:
                        send_line(
                            conn,
                            {
                                "protocol": PROTOCOL_VERSION,
                                "request_id": "",
                                "attempt_no": 0,
                                "instance_id": args.instance_id,
                                "status": "error",
                                "latency_class": "fast",
                                "delay_ms": 0,
                                "detail": f"upstream exception: {exc}",
                            },
                        )
                    except Exception:
                        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
