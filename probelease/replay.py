"""Replay and byte-level logical splice verification."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .declarations import DeclarationSet, RATIO_SCALE
from .errors import ReplayError, SpliceError
from .ledger import Ledger, canonical_json
from .logging import StructuredLogger
from .probes import ProbeSupervisor
from .transport import ACK, build_probe_batch, transmit_and_collect


@dataclass(frozen=True)
class ReplayReport:
    sessions_replayed: tuple[str, ...]
    rounds_verified: int
    probes_verified: int
    elapsed_monotonic_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions_replayed": list(self.sessions_replayed),
            "rounds_verified": self.rounds_verified,
            "probes_verified": self.probes_verified,
            "elapsed_monotonic_ns": self.elapsed_monotonic_ns,
        }


@dataclass(frozen=True)
class SpliceReport:
    equal: bool
    left_sha256: str
    right_sha256: str
    bytes_compared: int
    sessions_compared: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "equal": self.equal,
            "left_sha256": self.left_sha256,
            "right_sha256": self.right_sha256,
            "bytes_compared": self.bytes_compared,
            "sessions_compared": list(self.sessions_compared),
        }


def replay_verify(
    ledger: str | os.PathLike[str] | Ledger,
    declarations: DeclarationSet,
    *,
    session_id: str | None = None,
    supervisor: ProbeSupervisor | None = None,
    logger: StructuredLogger | None = None,
) -> ReplayReport:
    """Replay committed rounds over real UDP and compare each checkpoint.

    Callers must stop any probe supervisor bound to the same declared ports.
    When ``supervisor`` is supplied it is reused and not closed here.
    """

    logger = logger or StructuredLogger()
    owns_ledger = not isinstance(ledger, Ledger)
    ledger_obj = (
        Ledger(ledger, declarations, logger=logger, recover_incomplete=False)
        if owns_ledger
        else ledger
    )
    owns_supervisor = supervisor is None
    supervisor_obj = supervisor
    started = time.monotonic_ns()
    try:
        if supervisor_obj is None:
            supervisor_obj = ProbeSupervisor(declarations, logger=logger)
        if not supervisor_obj.running:
            try:
                supervisor_obj.start()
            except Exception as exc:
                raise ReplayError(
                    "failed to start deterministic probes for replay",
                    "replay.probes",
                    {"reason": str(exc)},
                ) from exc
        target_sessions = (
            [declarations.session(session_id)]
            if session_id is not None
            else list(declarations.sessions)
        )
        rounds_verified = 0
        probes_verified = 0
        for session in target_sessions:
            round_records = ledger_obj.round_records(session.session_id)
            if not round_records:
                if session_id is not None:
                    raise ReplayError(
                        "no committed rounds exist for the requested session",
                        f"sessions[{session.session_id}].rounds",
                        {"session_id": session.session_id},
                    )
                continue
            if round_records[0]["cycle"] != 1:
                raise ReplayError(
                    "replay requires committed history from cycle 1",
                    f"sessions[{session.session_id}].cycle",
                    {
                        "session_id": session.session_id,
                        "first_recorded_cycle": round_records[0]["cycle"],
                    },
                )
            events = ledger_obj.pause_events(session.session_id)
            event_index = 0
            lease_version = 1
            consecutive_unrenewed = 0
            status = "ACTIVE"
            pause_reason: str | None = None
            latest_ack: dict[str, int | None] = {
                probe_id: None for probe_id in session.probe_ids
            }
            for record in round_records:
                while (
                    event_index < len(events)
                    and int(events[event_index]["mono_ns"]) < int(record["start_mono_ns"])
                ):
                    event = events[event_index]
                    event_index += 1
                    if event["kind"] == "PAUSE":
                        status = "PAUSED"
                        pause_reason = event["reason"]
                    elif event["kind"] == "RESUME":
                        status = "ACTIVE"
                        pause_reason = None
                        if event["reason"] != "DEPENDENCY_RESTORED":
                            consecutive_unrenewed = 0
                    elif event["kind"] == "TERMINATE":
                        status = "TERMINATED"
                        pause_reason = event["reason"]
                if status != "ACTIVE":
                    raise ReplayError(
                        "ledger contains a committed round while replay state is inactive",
                        f"sessions[{session.session_id}].status",
                        {
                            "session_id": session.session_id,
                            "cycle": record["cycle"],
                            "replay_status": status,
                            "pause_reason": pause_reason,
                            "basis": "committed round sequence and pause/resume events",
                        },
                    )
                cycle = int(record["cycle"])
                attempt = 1
                request_lease_version = lease_version
                batch = build_probe_batch(
                    session,
                    declarations.probes,
                    cycle=cycle,
                    attempt=attempt,
                    lease_version=lease_version,
                    planned_mono_ns=time.monotonic_ns(),
                )
                actual_records = transmit_and_collect(
                    session,
                    batch,
                    cycle=cycle,
                    lease_version=lease_version,
                    attempt=attempt,
                )
                actual_by_probe = {item["probe_id"]: item for item in actual_records}
                expected_by_probe = record["probes"]
                for probe_id in session.probe_ids:
                    expected_probe = expected_by_probe.get(probe_id)
                    actual_probe = actual_by_probe.get(probe_id)
                    if expected_probe is None or actual_probe is None:
                        raise ReplayError(
                            "probe set differs from committed round",
                            f"sessions[{session.session_id}].probes[{probe_id}]",
                            {
                                "session_id": session.session_id,
                                "probe_id": probe_id,
                                "cycle": cycle,
                                "expected_present": expected_probe is not None,
                                "actual_present": actual_probe is not None,
                                "basis": "ledger round probe inventory",
                            },
                        )
                    if expected_probe["outcome"] != actual_probe["outcome"]:
                        raise _replay_mismatch(
                            session.session_id,
                            probe_id,
                            cycle,
                            "outcome",
                            expected_probe["outcome"],
                            actual_probe["outcome"],
                        )
                    if expected_probe["reason"] != actual_probe["reason"]:
                        raise _replay_mismatch(
                            session.session_id,
                            probe_id,
                            cycle,
                            "reason",
                            expected_probe["reason"],
                            actual_probe["reason"],
                        )
                    if actual_probe["outcome"] == ACK:
                        latest_ack[probe_id] = int(actual_probe["response_seq"])
                    expected_latest = expected_probe["response_seq"] if expected_probe["outcome"] == ACK else None
                    if expected_latest is None:
                        expected_latest = _latest_ack_from_ledger(
                            ledger_obj, session.session_id, probe_id, cycle
                        )
                    if expected_latest != latest_ack[probe_id]:
                        raise _replay_mismatch(
                            session.session_id,
                            probe_id,
                            cycle,
                            "latest_ack_seq",
                            expected_latest,
                            latest_ack[probe_id],
                        )
                    probes_verified += 1

                acked = sum(1 for item in actual_records if item["outcome"] == ACK)
                total = len(actual_records)
                ratio_bp = (acked * RATIO_SCALE + total // 2) // total
                if ratio_bp >= session.ack_floor_bp:
                    lease_version += 1
                    consecutive_unrenewed = 0
                    status = "ACTIVE"
                    pause_reason = None
                else:
                    consecutive_unrenewed += 1
                    if consecutive_unrenewed >= session.lease_miss_limit:
                        status = "PAUSED"
                        pause_reason = "LEASE_EXPIRED"
                    else:
                        status = "ACTIVE"
                        pause_reason = None
                if request_lease_version != record["request_lease_version"]:
                    raise _replay_mismatch(
                        session.session_id,
                        "*",
                        cycle,
                        "request_lease_version",
                        record["request_lease_version"],
                        request_lease_version,
                    )
                if ratio_bp != record["ratio_bp"]:
                    raise _replay_mismatch(
                        session.session_id,
                        "*",
                        cycle,
                        "ratio_bp",
                        record["ratio_bp"],
                        ratio_bp,
                    )
                if lease_version != record["lease_version"]:
                    raise _replay_mismatch(
                        session.session_id,
                        "*",
                        cycle,
                        "lease_version",
                        record["lease_version"],
                        lease_version,
                    )
                if status != record["status_after"]:
                    raise _replay_mismatch(
                        session.session_id,
                        "*",
                        cycle,
                        "status",
                        record["status_after"],
                        status,
                    )
                if pause_reason != record["pause_reason_after"]:
                    raise _replay_mismatch(
                        session.session_id,
                        "*",
                        cycle,
                        "pause_reason",
                        record["pause_reason_after"],
                        pause_reason,
                    )
                rounds_verified += 1
            logger.emit(
                "replay.session_verified",
                session_id=session.session_id,
                rounds=len(round_records),
                latest_ack=latest_ack,
                lease_version=lease_version,
                status=status,
                pause_reason=pause_reason,
            )
        elapsed = time.monotonic_ns() - started
        if rounds_verified == 0:
            raise ReplayError(
                "no committed rounds are available for replay",
                "replay.rounds",
                {"sessions": [session.session_id for session in target_sessions]},
            )
        report = ReplayReport(
            sessions_replayed=tuple(session.session_id for session in target_sessions),
            rounds_verified=rounds_verified,
            probes_verified=probes_verified,
            elapsed_monotonic_ns=elapsed,
        )
        logger.emit("replay.verified", report=report.to_dict())
        return report
    finally:
        if owns_supervisor and supervisor_obj is not None:
            supervisor_obj.close()
        if owns_ledger:
            ledger_obj.close()


def _latest_ack_from_ledger(
    ledger: Ledger, session_id: str, probe_id: str, through_cycle: int
) -> int | None:
    latest: int | None = None
    for record in ledger.round_records(
        session_id, start_cycle=1, end_cycle=through_cycle
    ):
        probe = record["probes"].get(probe_id)
        if probe is not None and probe["outcome"] == ACK:
            latest = probe["response_seq"]
    return latest


def _replay_mismatch(
    session_id: str,
    probe_id: str,
    cycle: int,
    field: str,
    expected: Any,
    actual: Any,
) -> ReplayError:
    field_path = f"sessions[{session_id}]"
    if probe_id != "*":
        field_path += f".probes[{probe_id}]"
    field_path += f".{field}"
    return ReplayError(
        "replayed round differs from committed ledger",
        field_path,
        {
            "session_id": session_id,
            "probe_id": probe_id,
            "cycle": cycle,
            "field": field,
            "expected": expected,
            "actual": actual,
            "basis": "committed ledger checkpoint vs real UDP replay",
        },
    )


def compare_splice_ledgers(
    left: Ledger,
    right: Ledger,
    *,
    session_ids: Iterable[str] | None = None,
    start_cycle: int | None = None,
    end_cycle: int | None = None,
) -> SpliceReport:
    """Compare stable logical ledger bytes (timing and transport ids excluded)."""

    if session_ids is None:
        left_ids = {record["session_id"] for record in left.logical_round_records()}
        right_ids = {record["session_id"] for record in right.logical_round_records()}
        target_ids = tuple(sorted(left_ids | right_ids))
    else:
        target_ids = tuple(sorted(set(session_ids)))
    left_bytes = _logical_bytes_for_sessions(
        left, target_ids, start_cycle=start_cycle, end_cycle=end_cycle
    )
    right_bytes = _logical_bytes_for_sessions(
        right, target_ids, start_cycle=start_cycle, end_cycle=end_cycle
    )
    left_digest = hashlib.sha256(left_bytes).hexdigest()
    right_digest = hashlib.sha256(right_bytes).hexdigest()
    if left_bytes != right_bytes:
        details = _first_ledger_difference(
            left, right, target_ids, start_cycle=start_cycle, end_cycle=end_cycle
        )
        raise SpliceError(
            "spliced ledger bytes differ from uninterrupted ledger bytes",
            details.get("field_path"),
            {
                **details,
                "left_sha256": left_digest,
                "right_sha256": right_digest,
                "left_bytes": len(left_bytes),
                "right_bytes": len(right_bytes),
            },
        )
    report = SpliceReport(
        equal=True,
        left_sha256=left_digest,
        right_sha256=right_digest,
        bytes_compared=len(left_bytes),
        sessions_compared=target_ids,
    )
    return report


def _logical_bytes_for_sessions(
    ledger: Ledger,
    session_ids: tuple[str, ...],
    *,
    start_cycle: int | None,
    end_cycle: int | None,
) -> bytes:
    return b"".join(
        ledger.logical_ledger_bytes(
            session_id, start_cycle=start_cycle, end_cycle=end_cycle
        )
        for session_id in session_ids
    )


def _first_ledger_difference(
    left: Ledger,
    right: Ledger,
    session_ids: tuple[str, ...],
    *,
    start_cycle: int | None,
    end_cycle: int | None,
) -> dict[str, Any]:
    for session_id in session_ids:
        left_records = left.logical_round_records(
            session_id, start_cycle=start_cycle, end_cycle=end_cycle
        )
        right_records = right.logical_round_records(
            session_id, start_cycle=start_cycle, end_cycle=end_cycle
        )
        left_map = {record["cycle"]: record for record in left_records}
        right_map = {record["cycle"]: record for record in right_records}
        cycles = sorted(set(left_map) | set(right_map))
        for cycle in cycles:
            if cycle not in left_map or cycle not in right_map:
                return {
                    "session_id": session_id,
                    "probe_id": "*",
                    "cycle": cycle,
                    "field": "record_presence",
                    "field_path": f"sessions[{session_id}].cycles[{cycle}]",
                    "expected": cycle in left_map,
                    "actual": cycle in right_map,
                }
            path, expected, actual = _first_mapping_difference(
                left_map[cycle], right_map[cycle], f"sessions[{session_id}].cycles[{cycle}]"
            )
            if path is not None:
                probe_id = "*"
                marker = ".probes["
                if marker in path:
                    probe_id = path.split(marker, 1)[1].split("]", 1)[0]
                return {
                    "session_id": session_id,
                    "probe_id": probe_id,
                    "cycle": cycle,
                    "field": path.rsplit(".", 1)[-1],
                    "field_path": path,
                    "expected": expected,
                    "actual": actual,
                }
    return {"field_path": "ledger", "expected": "equal", "actual": "different"}


def _first_mapping_difference(
    left: Any, right: Any, path: str
) -> tuple[str | None, Any, Any]:
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}", None, right[key]
            if key not in right:
                return f"{path}.{key}", left[key], None
            child_path, expected, actual = _first_mapping_difference(
                left[key], right[key], f"{path}.{key}"
            )
            if child_path is not None:
                return child_path, expected, actual
        return None, None, None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child_path, expected, actual = _first_mapping_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if child_path is not None:
                return child_path, expected, actual
        return None, None, None
    if left != right:
        return path, left, right
    return None, None, None
