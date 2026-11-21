"""Loopback TCP peer replay and crash-state recovery."""

from __future__ import annotations

import json
import socket
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnosis import REORDERING, RETRANSMISSION, TRUE_LOSS
from .errors import DiagnosticError, echo_error, format_error
from .model import FLAG_DATA, FLAG_REORDER, FLAG_RETRANSMIT
from .pipeline import DiagnosisSession
from .util import canonical_json, content_digest


ECHO_PROTOCOL = "echo-replay-v1"
MAX_MESSAGE = 16 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class EchoStateStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS echo_jobs (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT,
                response_digest TEXT,
                acked_at TEXT
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def prepare(self, request_id: str, request: dict[str, Any]) -> None:
        now = _now()
        self.connection.execute(
            "INSERT OR REPLACE INTO echo_jobs("
            "request_id, created_at, updated_at, status, request_json, response_json, response_digest, acked_at"
            ") VALUES (?, ?, ?, 'prepared', ?, NULL, NULL, NULL)",
            (request_id, now, now, canonical_json(request)),
        )
        self.connection.commit()

    def mark_sent(self, request_id: str, response: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE echo_jobs SET updated_at=?, status='sent', response_json=?, response_digest=? WHERE request_id=?",
            (_now(), canonical_json(response), content_digest(response), request_id),
        )
        self.connection.commit()

    def mark_acknowledged(self, request_id: str) -> None:
        self.connection.execute(
            "UPDATE echo_jobs SET updated_at=?, status='acknowledged', acked_at=? WHERE request_id=?",
            (_now(), _now(), request_id),
        )
        self.connection.commit()

    def mark_failed(self, request_id: str) -> None:
        self.connection.execute(
            "UPDATE echo_jobs SET updated_at=?, status='failed' WHERE request_id=?",
            (_now(), request_id),
        )
        self.connection.commit()

    def jobs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM echo_jobs ORDER BY created_at ASC, request_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, 4)
    if header is None:
        raise ConnectionError("peer closed before sending a message header")
    (length,) = struct.unpack("!I", header)
    if length <= 0 or length > MAX_MESSAGE:
        raise ValueError("invalid echo message length")
    body = _recv_exact(sock, length)
    if body is None:
        raise ConnectionError("peer closed during message body")
    return json.loads(body.decode("utf-8"))


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    body = canonical_json(message).encode("utf-8")
    if len(body) > MAX_MESSAGE:
        raise ValueError("echo message too large")
    sock.sendall(struct.pack("!I", len(body)) + body)


def _recv_exact(sock: socket.socket, length: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < length:
        piece = sock.recv(length - len(chunks))
        if not piece:
            return None
        chunks.extend(piece)
    return bytes(chunks)


def build_echo_plan(session: DiagnosisSession) -> dict[str, Any]:
    flows = []
    for diagnosis in session.diagnoses:
        flows.append(
            {
                "flow_id": diagnosis.flow_id,
                "sequence_envelope": diagnosis.sequence_envelope,
                "gaps": [
                    {
                        "start_seq": gap.start_seq,
                        "end_seq": gap.end_seq,
                        "classification": gap.classification,
                    }
                    for gap in diagnosis.gaps
                ],
            }
        )
    return {
        "protocol": ECHO_PROTOCOL,
        "flows": flows,
    }


def build_echo_response(
    request: dict[str, Any],
    *,
    inject_mismatch: bool = False,
) -> dict[str, Any]:
    if request.get("protocol") != ECHO_PROTOCOL:
        raise format_error(
            "unsupported echo protocol",
            record_number=0,
            field="protocol",
            actual=request.get("protocol"),
        )
    flows = request.get("flows")
    if not isinstance(flows, list) or not flows:
        raise format_error(
            "echo request has no flows",
            record_number=0,
            field="flows",
        )
    replayed_flows = []
    for flow_index, flow in enumerate(flows):
        envelope = flow.get("sequence_envelope", {})
        start = envelope.get("start_seq")
        end = envelope.get("end_seq")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise format_error(
                "echo flow has an invalid sequence envelope",
                record_number=flow_index + 1,
                field="sequence_envelope",
            )
        gap_class: dict[int, str] = {}
        for gap in flow.get("gaps", []):
            classification = gap.get("classification")
            low = gap.get("start_seq")
            high = gap.get("end_seq")
            if classification not in (TRUE_LOSS, RETRANSMISSION, REORDERING):
                raise format_error(
                    "echo gap has an invalid classification",
                    record_number=flow_index + 1,
                    field="gaps.classification",
                )
            if not isinstance(low, int) or not isinstance(high, int) or high < low:
                raise format_error(
                    "echo gap has an invalid interval",
                    record_number=flow_index + 1,
                    field="gaps.interval",
                )
            for seq in range(low, high + 1):
                gap_class[seq] = classification
        observations = []
        arrival = 1
        for seq in range(start, end + 1):
            classification = gap_class.get(seq)
            if classification == TRUE_LOSS:
                continue
            flags = FLAG_DATA
            if classification == RETRANSMISSION:
                flags |= FLAG_RETRANSMIT
            elif classification == REORDERING:
                flags |= FLAG_REORDER
            observations.append(
                {
                    "seq": seq,
                    "flags": flags,
                    "arrival_ordinal": arrival,
                }
            )
            arrival += 1
        if inject_mismatch and observations:
            observations.pop(0)
        replayed_flows.append(
            {
                "flow_id": flow["flow_id"],
                "sequence_envelope": {"start_seq": start, "end_seq": end},
                "observations": observations,
            }
        )
    response = {
        "protocol": ECHO_PROTOCOL,
        "request_id": request.get("request_id"),
        "flows": replayed_flows,
        "generated_by": "loopback-peer",
    }
    response["response_digest"] = content_digest(
        {key: value for key, value in response.items() if key != "response_digest"}
    )
    return response


def _actual_gap_intervals(flow: dict[str, Any]) -> list[dict[str, Any]]:
    observed = {
        int(observation["seq"]): int(observation["flags"])
        for observation in flow.get("observations", [])
    }
    start = int(flow["sequence_envelope"]["start_seq"])
    end = int(flow["sequence_envelope"]["end_seq"])
    by_seq: dict[int, str] = {}
    for seq in range(start, end + 1):
        flags = observed.get(seq)
        if flags is None:
            by_seq[seq] = TRUE_LOSS
        elif flags & FLAG_RETRANSMIT:
            by_seq[seq] = RETRANSMISSION
        elif flags & FLAG_REORDER:
            by_seq[seq] = REORDERING
    intervals = []
    for seq in sorted(by_seq):
        classification = by_seq[seq]
        if (
            intervals
            and intervals[-1]["end_seq"] + 1 == seq
            and intervals[-1]["classification"] == classification
        ):
            intervals[-1]["end_seq"] = seq
            intervals[-1]["count"] += 1
        else:
            intervals.append(
                {
                    "start_seq": seq,
                    "end_seq": seq,
                    "classification": classification,
                    "count": 1,
                }
            )
    return intervals


def verify_echo_response(session: DiagnosisSession, response: dict[str, Any]) -> dict[str, Any]:
    if response.get("protocol") != ECHO_PROTOCOL:
        raise echo_error(
            "echo response protocol mismatch",
            flow_id=None,
            gap_interval=None,
            field="protocol",
            expected=ECHO_PROTOCOL,
            actual=response.get("protocol"),
        )
    expected_flows = {
        diagnosis.flow_id: {
            "sequence_envelope": diagnosis.sequence_envelope,
            "gaps": [
                {
                    "start_seq": gap.start_seq,
                    "end_seq": gap.end_seq,
                    "classification": gap.classification,
                    "count": gap.count,
                }
                for gap in diagnosis.gaps
            ],
        }
        for diagnosis in session.diagnoses
    }
    actual_flows = {flow["flow_id"]: flow for flow in response.get("flows", [])}
    for flow_id in sorted(set(expected_flows) | set(actual_flows)):
        if flow_id not in expected_flows:
            raise echo_error(
                "echo response contains an unexpected flow",
                flow_id=flow_id,
                gap_interval=None,
                field="flow_presence",
                expected=None,
                actual=actual_flows.get(flow_id),
            )
        if flow_id not in actual_flows:
            raise echo_error(
                "echo response is missing a flow",
                flow_id=flow_id,
                gap_interval=None,
                field="flow_presence",
                expected=expected_flows[flow_id],
                actual=None,
            )
        expected = expected_flows[flow_id]
        actual = actual_flows[flow_id]
        if expected["sequence_envelope"] != actual.get("sequence_envelope"):
            raise echo_error(
                "echo response sequence envelope differs",
                flow_id=flow_id,
                gap_interval=None,
                field="sequence_envelope",
                expected=expected["sequence_envelope"],
                actual=actual.get("sequence_envelope"),
            )
        actual_gaps = _actual_gap_intervals(actual)
        expected_by_interval = {
            (gap["start_seq"], gap["end_seq"]): gap for gap in expected["gaps"]
        }
        actual_by_interval = {
            (gap["start_seq"], gap["end_seq"]): gap for gap in actual_gaps
        }
        for interval in sorted(set(expected_by_interval) | set(actual_by_interval)):
            gap_interval = {"start_seq": interval[0], "end_seq": interval[1]}
            expected_gap = expected_by_interval.get(interval)
            actual_gap = actual_by_interval.get(interval)
            if expected_gap is None or actual_gap is None:
                raise echo_error(
                    "echo replay gap shape differs",
                    flow_id=flow_id,
                    gap_interval=gap_interval,
                    field="gap_presence",
                    expected=expected_gap,
                    actual=actual_gap,
                )
            for field_name in ("classification", "count"):
                if expected_gap[field_name] != actual_gap[field_name]:
                    raise echo_error(
                        "echo replay gap field differs",
                        flow_id=flow_id,
                        gap_interval=gap_interval,
                        field=field_name,
                        expected=expected_gap[field_name],
                        actual=actual_gap[field_name],
                    )
    return {
        "ok": True,
        "matches": True,
        "flow_count": len(expected_flows),
        "gap_count": sum(len(flow["gaps"]) for flow in expected_flows.values()),
    }


def serve_peer(
    *,
    host: str,
    port: int,
    state_db: str,
    once: bool = False,
    ack_timeout_seconds: float = 5.0,
    ready_event: Any = None,
) -> None:
    if host != "127.0.0.1":
        raise ValueError("echo peer is restricted to 127.0.0.1")
    store = EchoStateStore(state_db)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(4)
            print(f"echo-peer listening on {host}:{port}", flush=True)
            if ready_event is not None:
                ready_event.set()
            while True:
                connection, _address = server.accept()
                with connection:
                    connection.settimeout(ack_timeout_seconds)
                    try:
                        request = recv_message(connection)
                    except Exception:
                        continue
                    op = request.get("op")
                    if op == "ack":
                        request_id = str(request.get("request_id", ""))
                        store.mark_acknowledged(request_id)
                        send_message(
                            connection,
                            {
                                "ok": True,
                                "op": "ack",
                                "request_id": request_id,
                                "status": "acknowledged_persisted",
                            },
                        )
                        continue
                    if op != "replay":
                        send_message(
                            connection,
                            {"ok": False, "code": "E_ECHO", "message": "unsupported operation"},
                        )
                        continue
                    request_id = str(request.get("request_id") or f"echo-{uuid.uuid4().hex}")
                    request = dict(request)
                    request["request_id"] = request_id
                    store.prepare(request_id, request)
                    try:
                        response = build_echo_response(
                            request,
                            inject_mismatch=bool(request.get("inject_mismatch", False)),
                        )
                    except DiagnosticError as exc:
                        store.mark_failed(request_id)
                        send_message(connection, exc.to_dict())
                        continue
                    store.mark_sent(request_id, response)
                    send_message(connection, response)
                    try:
                        ack = recv_message(connection)
                    except Exception:
                        ack = None
                    if isinstance(ack, dict) and ack.get("op") == "ack" and ack.get("request_id") == request_id:
                        store.mark_acknowledged(request_id)
                        send_message(
                            connection,
                            {
                                "ok": True,
                                "op": "ack",
                                "request_id": request_id,
                                "status": "acknowledged_persisted",
                            },
                        )
                if once:
                    break
    finally:
        store.close()


def run_echo_replay(
    session: DiagnosisSession,
    *,
    host: str,
    port: int,
    request_id: str,
    send_ack: bool = True,
    inject_mismatch: bool = False,
) -> dict[str, Any]:
    if host != "127.0.0.1":
        raise ValueError("echo client is restricted to 127.0.0.1")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    plan = build_echo_plan(session)
    request = {
        "op": "replay",
        "protocol": ECHO_PROTOCOL,
        "request_id": request_id,
        "flows": plan["flows"],
        "inject_mismatch": inject_mismatch,
    }
    with socket.create_connection((host, port), timeout=10.0) as connection:
        connection.settimeout(10.0)
        send_message(connection, request)
        response = recv_message(connection)
        if response.get("ok") is False:
            raise DiagnosticError(
                str(response.get("code", "E_ECHO")),
                str(response.get("message", "echo peer rejected request")),
                **response.get("details", {}),
            )
        verification = verify_echo_response(session, response)
        ack_result = None
        if send_ack:
            send_message(
                connection,
                {
                    "op": "ack",
                    "request_id": request_id,
                },
            )
            ack_result = recv_message(connection)
    return {
        "ok": True,
        "peer": {"host": host, "port": port},
        "request_id": request_id,
        "request_digest": content_digest(request),
        "response_digest": response.get("response_digest"),
        "verification": verification,
        "ack": ack_result,
    }


def recover_echo_state(state_db: str) -> dict[str, Any]:
    store = EchoStateStore(state_db)
    try:
        jobs = store.jobs()
    finally:
        store.close()
    recovered = []
    for job in jobs:
        status = job["status"]
        unconfirmed = status != "acknowledged"
        recovered.append(
            {
                "request_id": job["request_id"],
                "persisted_status": status,
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "response_digest": job["response_digest"],
                "peer_state_confirmed": not unconfirmed,
                "state_confirmation": "unconfirmed" if unconfirmed else "acknowledged",
                "unconfirmed_fields": (
                    [
                        "peer_process_liveness_at_termination",
                        "socket_delivery_completion",
                        "client_ack",
                    ]
                    if unconfirmed
                    else []
                ),
                "request": json.loads(job["request_json"]),
            }
        )
    return {
        "ok": True,
        "recovery_source": "sqlite_persisted_echo_state_only",
        "complete_incident_reconstruction": False,
        "notice": (
            "Forced termination is not reconstructed as a complete incident scene. "
            "Only durably persisted peer state is reported."
        ),
        "jobs": recovered,
    }
