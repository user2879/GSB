"""One isolated token-bucket process speaking JSON lines over loopback TCP."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canon import canonical_json
from .fixed import saturate_int64, saturating_mul
from .spec import Bucket, Spec, load_spec


PROTOCOL_VERSION = 1
FIXED_VERSION = "fixed-int64-v1"
STATUSES = {"ALLOW", "SHAPE", "DROP", "SKIP"}


def _send_line(sock: socket.socket, value: dict[str, Any]) -> None:
    sock.sendall(canonical_json(value).encode("utf-8") + b"\n")


def _read_line(sock: socket.socket) -> dict[str, Any] | None:
    buffer = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return None
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _, _remainder = bytes(buffer).partition(b"\n")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON line: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError("JSON line must contain an object")
            return value


@dataclass
class ServerBucketState:
    bucket: Bucket
    scale_exponent: int
    balance: np.int64
    queue_fp: list[int] = field(default_factory=list)
    last_ms: int | None = None
    refill_count: int = 0
    saturated_refills: int = 0

    @property
    def scale_factor(self) -> int:
        return 1 << self.scale_exponent

    @property
    def capacity_fp(self) -> int:
        return saturating_mul(self.bucket.burst_capacity, self.scale_factor)

    @property
    def needed_fp(self) -> int:
        return saturating_mul(self.bucket.packet_len, self.scale_factor)

    def advance(self, now_ms: int) -> tuple[int, int]:
        if self.last_ms is None:
            elapsed = 0
            self.last_ms = now_ms
        else:
            elapsed = now_ms - self.last_ms
            if elapsed < 0:
                raise ValueError("arrival clock moved backwards")
            self.last_ms = now_ms
        if elapsed <= 0 or self.bucket.rate_per_sec == 0:
            return 0, 0
        operands = np.asarray(
            [self.bucket.rate_per_sec, self.scale_factor, elapsed],
            dtype=object,
        )
        raw = np.prod(operands)
        delta = int(raw // 1000)
        residual = int(raw % 1000)
        candidate = int(self.balance) + delta
        was_saturated = candidate > int(np.iinfo(np.int64).max)
        if was_saturated:
            candidate = int(np.iinfo(np.int64).max)
        candidate = np.minimum(np.int64(candidate), np.int64(self.capacity_fp)).item()
        self.balance = np.int64(candidate)
        self.refill_count += 1
        if was_saturated:
            self.saturated_refills += 1
        return elapsed, residual

    def drain(self) -> None:
        while self.queue_fp and int(self.balance) >= self.queue_fp[0]:
            self.balance = np.int64(int(self.balance) - self.queue_fp.pop(0))

    def decide(self, eligible: bool) -> tuple[str, int, bool, int, int]:
        before = int(self.balance)
        queue_before = len(self.queue_fp)
        if not eligible:
            return "SKIP", before, False, queue_before, int(self.balance)
        needed = self.needed_fp
        overflow = False
        if np.greater_equal(self.balance, np.int64(needed)).item():
            status = "ALLOW"
            self.balance = np.int64(int(self.balance) - needed)
        elif len(self.queue_fp) < self.bucket.queue_limit:
            status = "SHAPE"
            self.queue_fp.append(needed)
        else:
            status = "DROP"
            overflow = True
        return status, before, overflow, queue_before, int(self.balance)


def _required_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def serve(spec_path: str, level: int) -> int:
    spec = load_spec(spec_path)
    bucket = spec.bucket_for_level(level)
    state = ServerBucketState(
        bucket=bucket,
        scale_exponent=spec.scale_exponent,
        balance=np.int64(saturating_mul(bucket.burst_capacity, spec.scale_factor)),
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bucket.host, bucket.port))
        listener.listen(1)
    except OSError as exc:
        print(
            canonical_json(
                {
                    "event": "error",
                    "code": "E_PORT",
                    "file": str(spec_path),
                    "entry": f"buckets[{level - 1}]",
                    "field": "port",
                    "message": str(exc),
                    "port": bucket.port,
                }
            ),
            flush=True,
        )
        listener.close()
        return 2

    print(
        canonical_json(
            {
                "event": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "level": level,
                "bucket_id": bucket.bucket_id,
                "host": bucket.host,
                "port": bucket.port,
                "scale_exponent": spec.scale_exponent,
                "fixed_version": FIXED_VERSION,
            }
        ),
        flush=True,
    )
    listener.settimeout(max(1.0, spec.decision.tcp_timeout_ms / 1000.0))
    try:
        connection, _address = listener.accept()
    except socket.timeout:
        listener.close()
        return 0
    except OSError:
        listener.close()
        return 1
    listener.close()
    connection.settimeout(max(0.001, spec.decision.tcp_timeout_ms / 1000.0))
    try:
        hello = _read_line(connection)
        if hello is None:
            connection.close()
            return 0
        if hello.get("type") != "hello":
            _send_line(
                connection,
                {"type": "protocol_error", "code": "E_FIELD", "field": "type", "message": "hello required"},
            )
            connection.close()
            return 1
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            _send_line(
                connection,
                {
                    "type": "protocol_error",
                    "code": "E_FIELD",
                    "field": "protocol_version",
                    "message": "unsupported protocol version",
                },
            )
            connection.close()
            return 1
        _send_line(
            connection,
            {
                "type": "hello_ack",
                "protocol_version": PROTOCOL_VERSION,
                "level": level,
                "bucket_id": bucket.bucket_id,
                "scale_exponent": spec.scale_exponent,
                "fixed_version": FIXED_VERSION,
                "caliber_version": f"{FIXED_VERSION}/scale-exp-{spec.scale_exponent}",
            },
        )
        while True:
            request = _read_line(connection)
            if request is None:
                break
            started_ns = time.perf_counter_ns()
            if request.get("type") != "decide":
                _send_line(
                    connection,
                    {
                        "type": "protocol_error",
                        "code": "E_FIELD",
                        "field": "type",
                        "message": "decide request required",
                    },
                )
                break
            seq = _required_int(request.get("seq"))
            arrival_ms = _required_int(request.get("arrival_ms"))
            eligible = request.get("eligible")
            if seq is None or arrival_ms is None or not isinstance(eligible, bool):
                bad_field = "seq" if seq is None else "arrival_ms" if arrival_ms is None else "eligible"
                _send_line(
                    connection,
                    {
                        "type": "protocol_error",
                        "code": "E_FIELD",
                        "field": bad_field,
                        "message": "decision request field is missing or invalid",
                    },
                )
                break
            try:
                elapsed, residual = state.advance(arrival_ms)
                state.drain()
                status, before, overflow, queue_before, after = state.decide(eligible)
            except ValueError as exc:
                _send_line(
                    connection,
                    {
                        "type": "protocol_error",
                        "code": "E_FIELD",
                        "field": "arrival_ms",
                        "message": str(exc),
                    },
                )
                break
            elapsed_ms = max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
            _send_line(
                connection,
                {
                    "type": "decision",
                    "seq": seq,
                    "level": level,
                    "level_name": bucket.name,
                    "bucket_id": bucket.bucket_id,
                    "status": status,
                    "before_fp": before,
                    "after_fp": after,
                    "needed_fp": state.needed_fp,
                    "overflow": overflow,
                    "queue_used_before": queue_before,
                    "queue_used_after": len(state.queue_fp),
                    "refill_elapsed_ms": elapsed,
                    "refill_residual_numerator": residual,
                    "refill_count_after": state.refill_count,
                    "saturated_refills_after": state.saturated_refills,
                    "scale_exponent": spec.scale_exponent,
                    "fixed_version": FIXED_VERSION,
                    "caliber_version": f"{FIXED_VERSION}/scale-exp-{spec.scale_exponent}",
                    "elapsed_ms": elapsed_ms,
                },
            )
    except (socket.timeout, ConnectionError, ValueError, OSError):
        pass
    finally:
        try:
            connection.close()
        except OSError:
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="single token-bucket JSON-lines server")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--level", required=True, type=int, choices=(1, 2, 3))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return serve(args.spec, args.level)


if __name__ == "__main__":
    sys.exit(main())
