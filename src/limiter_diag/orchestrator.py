"""Parent-side orchestration for three real loopback bucket processes."""

from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .canon import canonical_bytes, canonical_json
from .engine import (
    ALLOW,
    DROP,
    SHAPE,
    SKIP,
    RunOutcome,
    choose_path,
    compare_results,
    divergence_error,
    run_path,
)
from .errors import DiagnosticError, field_error
from .fixed import saturating_mul
from .spec import Event, EventStream, Spec


RECEIPT_FIELDS = {
    "type",
    "seq",
    "level",
    "level_name",
    "bucket_id",
    "status",
    "before_fp",
    "after_fp",
    "needed_fp",
    "overflow",
    "queue_used_before",
    "queue_used_after",
    "refill_elapsed_ms",
    "refill_residual_numerator",
    "refill_count_after",
    "scale_exponent",
    "fixed_version",
    "caliber_version",
    "elapsed_ms",
}


def _read_json_line(sock: socket.socket) -> dict[str, Any]:
    buffer = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("bucket server closed the connection")
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _, _rest = bytes(buffer).partition(b"\n")
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON line must be an object")
            return value


def _send_json_line(sock: socket.socket, value: dict[str, Any]) -> None:
    sock.sendall(canonical_json(value).encode("utf-8") + b"\n")


@dataclass
class BucketProcess:
    level: int
    port: int
    process: subprocess.Popen[str]
    socket: socket.socket | None = None
    ready: dict[str, Any] = field(default_factory=dict)


class BucketProcessGroup:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.processes: list[BucketProcess] = []
        self.started = False
        self.telemetry_ms = 0

    def start(self) -> None:
        try:
            for bucket in self.spec.buckets:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "limiter_diag.bucket_server",
                        "--spec",
                        self.spec.source_file,
                        "--level",
                        str(bucket.level),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=str(os.path.dirname(os.path.abspath(self.spec.source_file))),
                )
                holder = BucketProcess(level=bucket.level, port=bucket.port, process=process)
                self.processes.append(holder)
                ready = self._wait_for_ready(holder, self.spec.decision.startup_timeout_ms)
                holder.ready = ready
                client = self._connect_and_handshake(holder)
                holder.socket = client
            self.started = True
        except Exception:
            self.stop()
            raise

    def _wait_for_ready(self, holder: BucketProcess, timeout_ms: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        assert holder.process.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stderr = self._safe_stderr(holder.process)
                raise DiagnosticError(
                    "E_TIMEOUT",
                    "bucket subprocess did not announce readiness before startup timeout",
                    file=self.spec.source_file,
                    entry=f"buckets[{holder.level - 1}]",
                    field="startup_timeout_ms",
                    details={"level": holder.level, "port": holder.port, "stderr": stderr},
                )
            readable, _, _ = select.select([holder.process.stdout], [], [], min(remaining, 0.25))
            if not readable:
                if holder.process.poll() is not None:
                    stderr = self._safe_stderr(holder.process)
                    raise DiagnosticError(
                        "E_PORT",
                        "bucket subprocess exited before binding its declared loopback port",
                        file=self.spec.source_file,
                        entry=f"buckets[{holder.level - 1}]",
                        field="port",
                        details={"level": holder.level, "port": holder.port, "stderr": stderr},
                    )
                continue
            line = holder.process.stdout.readline()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DiagnosticError(
                    "E_PORT",
                    "bucket subprocess emitted an invalid startup line",
                    file=self.spec.source_file,
                    entry=f"buckets[{holder.level - 1}]",
                    field="stdout",
                    details={"level": holder.level, "line": line, "parse_error": str(exc)},
                )
            if value.get("event") == "error":
                code = str(value.get("code", "E_PORT"))
                raise DiagnosticError(
                    code,
                    str(value.get("message", "bucket subprocess failed before readiness")),
                    file=str(value.get("file", self.spec.source_file)),
                    entry=str(value.get("entry", f"buckets[{holder.level - 1}]")),
                    field=str(value.get("field", "port")),
                    details={"level": holder.level, "port": holder.port},
                )
            if value.get("event") != "ready":
                continue
            required = {
                "event",
                "protocol_version",
                "level",
                "bucket_id",
                "host",
                "port",
                "scale_exponent",
                "fixed_version",
            }
            for field_name in sorted(required - set(value)):
                raise field_error(
                    self.spec.source_file,
                    f"startup[level={holder.level}]",
                    field_name,
                    "readiness line is missing a required field",
                )
            if value.get("protocol_version") != 1:
                raise field_error(
                    self.spec.source_file,
                    f"startup[level={holder.level}]",
                    "protocol_version",
                    "unsupported protocol version",
                )
            if value.get("level") != holder.level or value.get("port") != holder.port:
                raise field_error(
                    self.spec.source_file,
                    f"startup[level={holder.level}]",
                    "level/port",
                    "readiness does not match the declared bucket",
                )
            return value

    def _connect_and_handshake(self, holder: BucketProcess) -> socket.socket:
        deadline = time.monotonic() + self.spec.decision.startup_timeout_ms / 1000.0
        timeout = max(0.001, self.spec.decision.tcp_timeout_ms / 1000.0)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(timeout)
            try:
                client.connect((self.spec.bucket_for_level(holder.level).host, holder.port))
            except OSError as exc:
                last_error = exc
                client.close()
                time.sleep(0.01)
                continue
            try:
                _send_json_line(client, {"type": "hello", "protocol_version": 1})
                ack = _read_json_line(client)
            except socket.timeout as exc:
                client.close()
                raise DiagnosticError(
                    "E_TIMEOUT",
                    "timed out during bucket handshake",
                    file=self.spec.source_file,
                    entry=f"level={holder.level}",
                    field="handshake",
                    details={"level": holder.level, "port": holder.port},
                ) from exc
            except (ConnectionError, ValueError, json.JSONDecodeError) as exc:
                client.close()
                last_error = exc
                time.sleep(0.01)
                continue
            if ack.get("type") == "protocol_error":
                code = str(ack.get("code", "E_FIELD"))
                field_name = str(ack.get("field", "protocol"))
                client.close()
                raise DiagnosticError(
                    code,
                    str(ack.get("message", "protocol handshake failed")),
                    file=self.spec.source_file,
                    entry=f"level={holder.level}",
                    field=field_name,
                )
            for field_name in ("type", "protocol_version", "level", "bucket_id", "scale_exponent", "fixed_version"):
                if field_name not in ack:
                    client.close()
                    raise field_error(
                        self.spec.source_file,
                        f"handshake[level={holder.level}]",
                        field_name,
                        "handshake receipt is missing a required field",
                    )
            if ack.get("type") != "hello_ack":
                client.close()
                raise field_error(
                    self.spec.source_file,
                    f"handshake[level={holder.level}]",
                    "type",
                    "expected hello_ack",
                )
            return client
        raise DiagnosticError(
            "E_PORT",
            "could not connect to the declared loopback bucket port",
            file=self.spec.source_file,
            entry=f"level={holder.level}",
            field="port",
            details={"level": holder.level, "port": holder.port, "error": str(last_error)},
        )

    def _validate_receipt(self, receipt: dict[str, Any], level: int, seq: int) -> dict[str, Any]:
        entry = f"receipt[level={level},seq={seq}]"
        if receipt.get("type") == "protocol_error":
            raise DiagnosticError(
                str(receipt.get("code", "E_FIELD")),
                str(receipt.get("message", "bucket protocol error")),
                file=self.spec.source_file,
                entry=entry,
                field=str(receipt.get("field", "receipt")),
            )
        for field_name in sorted(RECEIPT_FIELDS - set(receipt)):
            raise field_error(
                self.spec.source_file,
                entry,
                field_name,
                "bucket receipt is missing a required field",
            )
        int_fields = (
            "seq",
            "level",
            "before_fp",
            "after_fp",
            "needed_fp",
            "queue_used_before",
            "queue_used_after",
            "refill_elapsed_ms",
            "refill_residual_numerator",
            "refill_count_after",
            "scale_exponent",
            "elapsed_ms",
        )
        for field_name in int_fields:
            value = receipt[field_name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise field_error(
                    self.spec.source_file,
                    entry,
                    field_name,
                    "bucket receipt field has the wrong type",
                )
        if receipt["type"] != "decision":
            raise field_error(self.spec.source_file, entry, "type", "expected decision receipt")
        if receipt["seq"] != seq or receipt["level"] != level:
            raise field_error(
                self.spec.source_file,
                entry,
                "seq/level",
                "receipt does not match the request",
            )
        if receipt["status"] not in {ALLOW, SHAPE, DROP, SKIP}:
            raise field_error(self.spec.source_file, entry, "status", "unknown status")
        if not isinstance(receipt["overflow"], bool):
            raise field_error(self.spec.source_file, entry, "overflow", "must be boolean")
        if receipt["scale_exponent"] != self.spec.scale_exponent:
            raise field_error(
                self.spec.source_file,
                entry,
                "scale_exponent",
                "caliber mismatch",
            )
        if receipt["elapsed_ms"] < 0:
            raise field_error(self.spec.source_file, entry, "elapsed_ms", "must be nonnegative")
        return receipt

    def decide(self, level: int, seq: int, arrival_ms: int, eligible: bool) -> dict[str, Any]:
        holder = self.processes[level - 1]
        if holder.socket is None:
            raise DiagnosticError(
                "E_PORT",
                "bucket socket is not connected",
                file=self.spec.source_file,
                entry=f"level={level}",
                field="socket",
            )
        request = {
            "type": "decide",
            "seq": seq,
            "arrival_ms": arrival_ms,
            "eligible": eligible,
        }
        try:
            _send_json_line(holder.socket, request)
            receipt = _read_json_line(holder.socket)
        except socket.timeout as exc:
            raise DiagnosticError(
                "E_TIMEOUT",
                "timed out while waiting for a bucket decision receipt",
                file=self.spec.source_file,
                entry=f"receipt[level={level},seq={seq}]",
                field="socket_read",
                details={"level": level, "seq": seq},
            ) from exc
        except ConnectionError as exc:
            raise DiagnosticError(
                "E_TIMEOUT",
                "bucket connection closed before a complete receipt was read",
                file=self.spec.source_file,
                entry=f"receipt[level={level},seq={seq}]",
                field="socket_read",
                details={"level": level, "seq": seq, "error": str(exc)},
            ) from exc
        self.telemetry_ms += int(receipt.get("elapsed_ms", 0))
        return self._validate_receipt(receipt, level, seq)

    def _safe_stderr(self, process: subprocess.Popen[str]) -> str:
        if process.stderr is None:
            return ""
        try:
            if process.poll() is None:
                return ""
            return process.stderr.read().strip()
        except OSError:
            return ""

    def shutdown(self) -> None:
        for holder in self.processes:
            if holder.socket is not None:
                try:
                    holder.socket.close()
                except OSError:
                    pass
                holder.socket = None
        for holder in self.processes:
            process = holder.process
            if process.poll() is None:
                process.terminate()
        for holder in self.processes:
            process = holder.process
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        self.processes = []
        self.started = False

    def force_kill(self) -> list[int]:
        """Kill children without closing their sockets first, for crash tests."""
        pids: list[int] = []
        for holder in self.processes:
            if holder.process.poll() is None:
                pids.append(holder.process.pid)
                holder.process.kill()
        for holder in self.processes:
            try:
                holder.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            if holder.socket is not None:
                try:
                    holder.socket.close()
                except OSError:
                    pass
                holder.socket = None
        self.started = False
        return pids

    def stop(self) -> None:
        self.shutdown()

    def __enter__(self) -> "BucketProcessGroup":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown()

    def lease_dict(self) -> dict[str, Any]:
        return {
            "pids": [holder.process.pid for holder in self.processes],
            "ports": [holder.port for holder in self.processes],
        }


def _annotate_from_receipts(
    spec: Spec,
    event: Event,
    receipts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    window_index = event.arrival_ms // spec.decision.window_ms
    boundary = event.arrival_ms % spec.decision.window_ms == 0
    levels: list[dict[str, Any]] = []
    reached_levels: list[int] = []
    verdict = ALLOW
    for index, receipt in enumerate(receipts):
        status = str(receipt["status"])
        refill_lsb = 1 if int(receipt["refill_elapsed_ms"]) > 0 else 0
        carry_lsb = index
        window_lsb = 1 if boundary else 0
        uncertainty = refill_lsb + carry_lsb + window_lsb
        ambiguous = False
        tiebreak: str | None = None
        if status != SKIP:
            reached_levels.append(index + 1)
            if int(abs(int(receipt["before_fp"]) - int(receipt["needed_fp"]))) <= uncertainty and uncertainty > 0:
                ambiguous = True
                tiebreak = f"level={index + 1};seq={event.seq};rule=conservative_numeric_then_level_then_seq"
            if status != ALLOW and verdict == ALLOW:
                verdict = status
        cause = "none"
        if ambiguous:
            candidates = (("refill", refill_lsb), ("carry", carry_lsb), ("window", window_lsb))
            largest = max(value for _, value in candidates)
            cause = next(name for name, value in candidates if value == largest)
        levels.append(
            {
                "level": index + 1,
                "name": receipt["level_name"],
                "bucket_id": receipt["bucket_id"],
                "reached": status != SKIP,
                "status": status,
                "before_fp": int(receipt["before_fp"]),
                "after_fp": int(receipt["after_fp"]),
                "needed_fp": int(receipt["needed_fp"]),
                "overflow": bool(receipt["overflow"]),
                "queue_used_before": int(receipt["queue_used_before"]),
                "queue_used_after": int(receipt["queue_used_after"]),
                "refill_elapsed_ms": int(receipt["refill_elapsed_ms"]),
                "refill_residual_numerator": int(receipt["refill_residual_numerator"]),
                "refill_count_after": int(receipt["refill_count_after"]),
                "refill_lsb": refill_lsb,
                "carry_lsb": carry_lsb,
                "window_lsb": window_lsb,
                "uncertainty_lsb": uncertainty,
                "error_affected": ambiguous,
                "cause": cause,
                "tiebreak": tiebreak,
            }
        )
    pair_flags: list[dict[str, Any]] = []
    for left_level, right_level in zip(reached_levels, reached_levels[1:]):
        left = levels[left_level - 1]
        right = levels[right_level - 1]
        combined = int(left["uncertainty_lsb"]) + int(right["uncertainty_lsb"])
        delta = abs(int(left["before_fp"]) - int(right["before_fp"]))
        if combined > 0 and delta <= combined:
            pair_flags.append(
                {
                    "left_level": left_level,
                    "right_level": right_level,
                    "delta_fp": delta,
                    "bound_lsb": combined,
                    "tiebreak": f"level={left_level};seq={event.seq};rule=level_order",
                }
            )
    causes = sorted(
        {level["cause"] for level in levels if level["error_affected"] and level["cause"] != "none"}
    )
    return {
        "seq": event.seq,
        "arrival_ms": event.arrival_ms,
        "window_index": window_index,
        "verdict": verdict,
        "levels": levels,
        "indistinguishable_level_pairs": pair_flags,
        "error_affected": any(level["error_affected"] for level in levels) or bool(pair_flags),
        "causes": causes,
        "window_boundary_affected": False,
    }


@dataclass
class TcpRunOutcome:
    outcome: RunOutcome
    telemetry: dict[str, Any]
    receipts: list[list[dict[str, Any]]]


def run_tcp_path(spec: Spec, events: EventStream, path: str) -> TcpRunOutcome:
    path_choice = choose_path(spec, events)
    group = BucketProcessGroup(spec)
    receipts: list[list[dict[str, Any]]] = []
    try:
        group.start()
        for event in events.events:
            eligible = True
            event_receipts: list[dict[str, Any]] = []
            for level in (1, 2, 3):
                receipt = group.decide(level, event.seq, event.arrival_ms, eligible)
                event_receipts.append(receipt)
                eligible = eligible and receipt["status"] == ALLOW
            receipts.append(event_receipts)
    finally:
        group.shutdown()

    annotated = [
        _annotate_from_receipts(spec, event, event_receipts)
        for event, event_receipts in zip(events.events, receipts)
    ]
    local = run_path(spec, events, path, path_choice=path_choice)
    comparison = compare_results(annotated, local.results)
    if not comparison["equal"]:
        raise divergence_error({**comparison, "mode": "tcp_vs_local"})
    telemetry = {
        "transport": "loopback-tcp-json-lines",
        "decision_receipts": len(receipts) * 3,
        "sum_receipt_elapsed_ms": sum(
            int(receipt["elapsed_ms"]) for event_receipts in receipts for receipt in event_receipts
        ),
        "max_receipt_elapsed_ms": max(
            (int(receipt["elapsed_ms"]) for event_receipts in receipts for receipt in event_receipts),
            default=0,
        ),
    }
    return TcpRunOutcome(outcome=local, telemetry=telemetry, receipts=receipts)


def verify_tcp_paths(spec: Spec, events: EventStream) -> dict[str, Any]:
    eventwise = run_tcp_path(spec, events, "eventwise")
    window = run_tcp_path(spec, events, "window")
    comparison = compare_results(eventwise.outcome.results, window.outcome.results)
    if not comparison["equal"]:
        raise divergence_error(comparison)
    choice = choose_path(spec, events)
    return {
        "equal": True,
        "selected_path": choice.path,
        "path_choice": choice.canonical_dict(),
        "eventwise_digest": eventwise.outcome.analysis_digest,
        "window_digest": window.outcome.analysis_digest,
        "eventwise_telemetry": eventwise.telemetry,
        "window_telemetry": window.telemetry,
        "results": window.outcome.results,
        "checkpoint": window.outcome.checkpoint,
        "error_report": window.outcome.error_report,
    }

