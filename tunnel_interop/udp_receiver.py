"""Windows-compatible loopback UDP receiver process used by real forwarding tests."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from pathlib import Path
from typing import Any

from .errors import TunnelError
from .util import atomic_write_json, sha256_hex, wall_time_ms
from .wire import peek_wire_identity


def _jsonl(path: str | None, event: MappingLike) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


MappingLike = dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic 127.0.0.1 UDP tunnel receiver")
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--ack-mode", choices=("normal", "missing-field"), default="normal")
    parser.add_argument("--drop-indices", default="")
    parser.add_argument("--drop-all", action="store_true")
    parser.add_argument("--idle-timeout-ms", type=int, default=10_000)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--log-file")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    drop_indices = {
        int(value) for value in args.drop_indices.split(",") if value.strip() != ""
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    sock.settimeout(0.05)
    stop = False

    def on_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, on_stop)
            except (OSError, ValueError):
                pass

    try:
        sock.bind((args.host, args.port))
        actual_port = sock.getsockname()[1]
        atomic_write_json(
            args.ready_file,
            {
                "ok": True,
                "event": "ready",
                "endpoint_id": args.endpoint_id,
                "host": args.host,
                "port": actual_port,
                "pid": os.getpid(),
                "ack_mode": args.ack_mode,
            },
        )
    except Exception as exc:
        atomic_write_json(
            args.ready_file,
            {
                "ok": False,
                "event": "bind_error",
                "endpoint_id": args.endpoint_id,
                "host": args.host,
                "port": args.port,
                "error": str(exc),
            },
        )
        sock.close()
        return 2

    _jsonl(
        args.log_file,
        {
            "event": "receiver_ready",
            "endpoint_id": args.endpoint_id,
            "host": args.host,
            "port": actual_port,
            "timestamp_ms": wall_time_ms(),
        },
    )
    last_receive = time.monotonic()
    receiver_seq = 0
    try:
        while not stop:
            if (time.monotonic() - last_receive) * 1000 > args.idle_timeout_ms:
                _jsonl(
                    args.log_file,
                    {
                        "event": "receiver_idle_exit",
                        "endpoint_id": args.endpoint_id,
                        "timestamp_ms": wall_time_ms(),
                    },
                )
                break
            try:
                datagram, peer = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                if stop:
                    break
                raise
            last_receive = time.monotonic()
            receiver_seq += 1
            received_at_ms = wall_time_ms()
            try:
                identity = peek_wire_identity(datagram)
            except TunnelError as exc:
                ack = {
                    "ack": False,
                    "endpoint_id": args.endpoint_id,
                    "error": exc.to_dict()["error"],
                    "datagram_sha256": sha256_hex(datagram),
                    "receiver_seq": receiver_seq,
                    "received_at_ms": received_at_ms,
                }
            else:
                ack = {
                    "ack": True,
                    "endpoint_id": args.endpoint_id,
                    "fragment_index": identity["fragment_index"],
                    "tunnel_id": identity["tunnel_id"],
                    "message_id": identity["message_id"],
                    "format": identity["format"],
                    "fragment_offset": identity["fragment_offset"],
                    "fragment_length": identity["fragment_length"],
                    "datagram_sha256": sha256_hex(datagram),
                    "datagram_length": len(datagram),
                    "received_at_ms": received_at_ms,
                    "receiver_seq": receiver_seq,
                }
                if args.ack_mode == "missing-field":
                    ack.pop("received_at_ms", None)
            if args.drop_all or (ack.get("fragment_index") in drop_indices):
                _jsonl(
                    args.log_file,
                    {
                        "event": "receiver_drop_ack",
                        "endpoint_id": args.endpoint_id,
                        "fragment_index": ack.get("fragment_index"),
                        "receiver_seq": receiver_seq,
                    },
                )
                continue
            sock.sendto(json.dumps(ack, sort_keys=True, separators=(",", ":")).encode("utf-8"), peer)
            _jsonl(
                args.log_file,
                {
                    "event": "receiver_ack",
                    "endpoint_id": args.endpoint_id,
                    "fragment_index": ack.get("fragment_index"),
                    "receiver_seq": receiver_seq,
                    "ok": ack.get("ack"),
                },
            )
    finally:
        sock.close()
        _jsonl(
            args.log_file,
            {
                "event": "receiver_exit",
                "endpoint_id": args.endpoint_id,
                "timestamp_ms": wall_time_ms(),
            },
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
