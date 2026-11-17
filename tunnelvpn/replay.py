"""Deterministic restore replay and first-mismatch reporting."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codec import FrameDecoder, Reassembler, encode_message
from .config import PeerDefinition, TunnelConfig
from .errors import ErrorCode, TunnelError
from .metrics import (
    Ledger,
    SessionRecord,
    compare_statistics,
)
from .session import SessionState
from .snapshot import RestoredState, Snapshot, restore_snapshot


@dataclass(frozen=True)
class ReplayMismatch:
    tunnel_id: str | None
    session_id: str | None
    field: str
    expected: Any
    actual: Any
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tunnel_id": self.tunnel_id,
            "session_id": self.session_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class ReplayReport:
    code: str
    ok: bool
    first_mismatch: ReplayMismatch | None
    sessions_replayed: int
    session_results: tuple[dict[str, Any], ...]
    expected_statistics: dict[str, Any]
    replay_statistics: dict[str, Any]
    restored_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "ok": self.ok,
            "sessions_replayed": self.sessions_replayed,
            "first_mismatch": (
                None if self.first_mismatch is None else self.first_mismatch.to_dict()
            ),
            "session_results": list(self.session_results),
            "expected_statistics": self.expected_statistics,
            "replay_statistics": self.replay_statistics,
            "restored_report": self.restored_report,
        }


class ReplayVerifier:
    """Strict deterministic replay using the snapshot's clock observations.

    The replay re-encodes every saved payload, decodes and reassembles every
    frame, re-applies the saved peer behavior, then rebuilds the ledger using
    the recorded start/end/queue observations.  Physical network timing is not
    used, making the full ledger comparison byte-stable after restart.
    """

    def __init__(self, snapshot: Snapshot, *, current_config: TunnelConfig | None = None) -> None:
        self.snapshot = snapshot
        self.current_config = current_config if current_config is not None else snapshot.config()
        self.restored: RestoredState = restore_snapshot(snapshot, self.current_config)

    def verify(self) -> ReplayReport:
        config = self.restored.config
        routes = config.route_map()
        peers = config.peer_map()
        records: list[SessionRecord] = []
        session_results: list[dict[str, Any]] = []
        mismatch: ReplayMismatch | None = None

        sessions = sorted(self.restored.sessions, key=lambda item: (item.start_ms, item.session_id))
        for state in sessions:
            result = self._replay_session(state, config, routes, peers)
            session_results.append(result)
            if result["outcome"] == "REPLAY_ERROR":
                if mismatch is None:
                    mismatch = ReplayMismatch(
                        tunnel_id=state.route_id,
                        session_id=state.session_id,
                        field="session",
                        expected="replayable",
                        actual="REPLAY_ERROR",
                        basis=result["basis"],
                    )
                continue
            if state.requires_rerun or state.status == "UNKNOWN":
                if mismatch is None:
                    mismatch = ReplayMismatch(
                        tunnel_id=state.route_id,
                        session_id=state.session_id,
                        field="status",
                        expected="COMPLETED|FAILED",
                        actual="UNKNOWN",
                        basis="snapshot contains an in-flight session; rerun is required",
                    )
                continue

            comparison_fields = [
                ("outcome", state.business_code, result["code"]),
                ("payload_bytes", state.payload_bytes, result["payload_bytes"]),
                ("forwarded_bytes", state.forwarded_bytes, result["forwarded_bytes"]),
                (
                    "encapsulation_digest",
                    state.encapsulation_digest,
                    result["encapsulation_digest"],
                ),
                ("fragment_count", state.fragment_count, result["fragment_count"]),
            ]
            for field, expected, actual in comparison_fields:
                if expected != actual and mismatch is None:
                    mismatch = ReplayMismatch(
                        tunnel_id=state.route_id,
                        session_id=state.session_id,
                        field=field,
                        expected=expected,
                        actual=actual,
                        basis=result["basis"],
                    )
            if mismatch is None:
                saved_record = self.restored.ledger.get(state.session_id)
                if saved_record is None:
                    mismatch = ReplayMismatch(
                        tunnel_id=state.route_id,
                        session_id=state.session_id,
                        field="ledger.record",
                        expected="present",
                        actual="missing",
                        basis="replay lookup",
                    )
                else:
                    ledger_pairs = [
                        ("ledger.business_code", saved_record.business_code, result["code"]),
                        (
                            "ledger.payload_bytes",
                            saved_record.payload_bytes,
                            result["payload_bytes"],
                        ),
                        (
                            "ledger.forwarded_bytes",
                            saved_record.forwarded_bytes,
                            result["forwarded_bytes"],
                        ),
                        (
                            "ledger.encapsulation_digest",
                            saved_record.encapsulation_digest,
                            result["encapsulation_digest"],
                        ),
                        (
                            "ledger.wire_bytes",
                            saved_record.wire_bytes,
                            result["wire_bytes"],
                        ),
                        (
                            "ledger.peer_payload_digest",
                            saved_record.peer_payload_digest,
                            result["peer_payload_digest"],
                        ),
                    ]
                    for field, expected, actual in ledger_pairs:
                        if expected != actual and mismatch is None:
                            mismatch = ReplayMismatch(
                                tunnel_id=state.route_id,
                                session_id=state.session_id,
                                field=field,
                                expected=expected,
                                actual=actual,
                                basis=result["basis"],
                            )
            records.append(self._replay_record(state, result))

        replay_ledger = Ledger(records)
        replay_statistics = replay_ledger.statistics()
        expected_statistics = self.restored.statistics
        if mismatch is None:
            stats_mismatch = compare_statistics(expected_statistics, replay_statistics)
            if stats_mismatch is not None:
                field, expected, actual = stats_mismatch
                mismatch = ReplayMismatch(
                    tunnel_id=None,
                    session_id="*",
                    field=field,
                    expected=expected,
                    actual=actual,
                    basis="ledger statistics rebuilt from replayed sessions",
                )
        if mismatch is None:
            expected_digest = expected_statistics.get("ledger_digest")
            actual_digest = replay_statistics.get("ledger_digest")
            if expected_digest != actual_digest:
                mismatch = ReplayMismatch(
                    tunnel_id=None,
                    session_id="*",
                    field="statistics.ledger_digest",
                    expected=expected_digest,
                    actual=actual_digest,
                    basis="canonical replay ledger digest",
                )
        return ReplayReport(
            code=(ErrorCode.OK.value if mismatch is None else ErrorCode.E_REPLAY.value),
            ok=mismatch is None,
            first_mismatch=mismatch,
            sessions_replayed=len(sessions),
            session_results=tuple(session_results),
            expected_statistics=expected_statistics,
            replay_statistics=replay_statistics,
            restored_report=self.restored.report.to_dict(),
        )

    def _replay_session(
        self,
        state: SessionState,
        config: TunnelConfig,
        routes: dict[str, Any],
        peers: dict[str, PeerDefinition],
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "session_id": state.session_id,
            "route_id": state.route_id,
            "sequence": state.sequence,
            "outcome": "REPLAY_ERROR",
            "code": "REPLAY_ERROR",
            "payload_bytes": 0,
            "forwarded_bytes": 0,
            "wire_bytes": 0,
            "encapsulation_digest": "",
            "fragment_count": 0,
            "peer_behavior": None,
            "peer_payload_digest": None,
            "basis": "",
        }
        route = routes.get(state.route_id)
        if route is None:
            base["basis"] = "route missing from snapshot configuration"
            return base
        peer = peers.get(route.peer_id)
        if peer is None:
            base["basis"] = "peer missing from snapshot configuration"
            return base
        base["peer_behavior"] = peer.behavior
        if not state.payload_b64:
            base["basis"] = "payload_b64 missing"
            return base
        try:
            payload = base64.b64decode(state.payload_b64, validate=True)
        except (ValueError, binascii.Error):
            base["basis"] = "payload_b64 invalid"
            return base
        encoded = encode_message(
            tunnel_id=route.id,
            sequence=state.sequence,
            cipher=route.cipher,
            mtu=route.mtu,
            payload=payload,
        )
        decoder = FrameDecoder()
        reassembler = Reassembler(timeout_ms=config.global_limits.fragment_timeout_ms)
        reassembled = []
        try:
            for frame in encoded.frames:
                decoded = decoder.decode(frame, expected_mtu=route.mtu)
                message = reassembler.add(decoded)
                if message is not None:
                    decoder.mark_complete(message.tunnel_id, message.sequence)
                    reassembled.append(message)
        except TunnelError as exc:
            base["basis"] = f"local codec replay failed: {exc}"
            return base
        if len(reassembled) != 1 or reassembled[0].payload != payload:
            base["basis"] = "local reassembly differs from saved payload"
            return base

        code = self._peer_outcome(peer, route.id, state.sequence, config)
        wire_bytes = (
            0
            if code == ErrorCode.E_PEER.value
            else encoded.wire_bytes + (4 * len(encoded.frames))
        )
        peer_digest = None
        if code in {ErrorCode.OK.value, ErrorCode.E_DIGEST.value}:
            peer_digest = reassembled[0].payload_digest
            if code == ErrorCode.E_DIGEST.value:
                peer_digest = _flip_digest(peer_digest)
        base.update(
            {
                "outcome": "COMPLETED" if code == ErrorCode.OK.value else "FAILED",
                "code": code,
                "payload_bytes": len(payload),
                "forwarded_bytes": (
                    0 if code == ErrorCode.E_PEER.value else len(payload)
                ),
                "wire_bytes": wire_bytes,
                "encapsulation_digest": reassembled[0].payload_digest,
                "peer_payload_digest": peer_digest,
                "fragment_count": len(encoded.frames),
                "basis": "re-encoded, framed, decoded and reassembled locally; peer behavior applied from snapshot config",
            }
        )
        return base

    def _peer_outcome(
        self,
        peer: PeerDefinition,
        tunnel_id: str,
        sequence: int,
        config: TunnelConfig,
    ) -> str:
        if peer.behavior == "unreachable":
            return ErrorCode.E_PEER.value
        if peer.behavior == "digest-error":
            return ErrorCode.E_DIGEST.value
        if peer.behavior == "delay-script":
            delay_ms = _scripted_delay_ms(peer, tunnel_id, sequence)
            if delay_ms > config.global_limits.peer_timeout_ms:
                return ErrorCode.E_TIMEOUT.value
        return ErrorCode.OK.value

    def _replay_record(
        self, state: SessionState, result: dict[str, Any]
    ) -> SessionRecord:
        saved_record = self.restored.ledger.get(state.session_id)
        return SessionRecord(
            session_id=state.session_id,
            route_id=state.route_id,
            peer_id=state.peer_id,
            sequence=state.sequence,
            start_ms=state.start_ms,
            end_ms=state.end_ms if state.end_ms is not None else state.start_ms,
            payload_bytes=result["payload_bytes"],
            forwarded_bytes=result["forwarded_bytes"],
            wire_bytes=result["wire_bytes"],
            queue_events=tuple(state.queue_events),
            status=(
                "COMPLETED" if result["code"] == ErrorCode.OK.value else "FAILED"
            ),
            business_code=result["code"],
            encapsulation_digest=result["encapsulation_digest"],
            fragment_count=result["fragment_count"],
            peer_payload_digest=result["peer_payload_digest"],
            replay_of=state.replay_of,
            error_detail=(saved_record.error_detail if saved_record is not None else None),
        )


def _scripted_delay_ms(peer: PeerDefinition, tunnel_id: str, sequence: int) -> int:
    delay = peer.delay_ms
    if not peer.script:
        return delay
    try:
        with Path(peer.script).open("r", encoding="utf-8") as handle:
            script = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return delay
    if not isinstance(script, dict):
        return delay
    by_sequence = script.get("by_sequence")
    if isinstance(by_sequence, dict):
        value = by_sequence.get(f"{tunnel_id}:{sequence}")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            delay = value
    delays = script.get("delays_ms")
    if isinstance(delays, dict):
        values = delays.get(tunnel_id)
        if isinstance(values, list) and 0 <= sequence < len(values):
            value = values[sequence]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                delay = value
    return delay


def _flip_digest(digest: str) -> str:
    first = digest[:2]
    replacement = "00" if first != "00" else "01"
    return replacement + digest[2:]
