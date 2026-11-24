from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .canonical import canonical_text, digest_hex, first_difference
from .errors import DomainError
from .ledger import Ledger, first_replay_difference
from .models import ScriptMessage, Spec
from .service_client import LoopbackService
from .session import SessionResult, build_full_session
from .termination import StopController, hard_terminate_self, termination_support


REPORT_SCHEMA_VERSION = 1


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _ledger_trace(session: SessionResult) -> dict[str, Any]:
    return {
        "certificates": session.cert_invalidation,
        "messages": session.message_invalidation,
    }


def _ledger_record(session: SessionResult, seq: int) -> dict[str, Any]:
    record = copy.deepcopy(session.message_records[seq])
    record["invalidation_trace"] = _ledger_trace(session)
    return record


def _validate_replies(
    request: ScriptMessage,
    expected_replies: list[ScriptMessage],
    actual_replies: list[dict[str, Any]],
) -> None:
    if len(actual_replies) != len(expected_replies):
        seq = expected_replies[0].seq if expected_replies else request.seq
        raise DomainError(
            "E_REPLY",
            f"receipt count mismatch: expected {len(expected_replies)}, got {len(actual_replies)}",
            cert_id=request.cert_ids[0] if request.cert_ids else None,
            field_path="reply.replies",
            message_seq=seq,
            details={"client_seq": request.client_seq},
        )
    required = [
        "seq",
        "sender",
        "client_seq",
        "server_seq",
        "arrival_ms",
        "type",
        "cert_ids",
        "expected",
    ]
    for expected, actual in zip(expected_replies, actual_replies, strict=True):
        if not isinstance(actual, dict):
            raise DomainError(
                "E_REPLY",
                "receipt must be an object",
                cert_id=expected.cert_ids[0] if expected.cert_ids else None,
                field_path="reply.replies[]",
                message_seq=expected.seq,
            )
        for field in required:
            if field not in actual:
                raise DomainError(
                    "E_REPLY",
                    f"receipt is missing field: {field}",
                    cert_id=expected.cert_ids[0] if expected.cert_ids else None,
                    field_path=f"reply.{field}",
                    message_seq=expected.seq,
                )
        expected_dict = expected.as_dict()
        if canonical_text(actual) != canonical_text(expected_dict):
            difference = first_difference(expected_dict, actual, "$.reply") or {}
            raise DomainError(
                "E_REPLY",
                "receipt does not match the declared server script",
                cert_id=expected.cert_ids[0] if expected.cert_ids else None,
                field_path=str(difference.get("path", "reply")),
                message_seq=expected.seq,
                details={"expected": difference.get("left"), "actual": difference.get("right")},
            )


def _batch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"PASS": 0, "FAIL": 0, "INDETERMINATE": 0}
    for record in records:
        counts[record["decision"]] += 1
    return {
        "seqs": [record["seq"] for record in records],
        "decisions": {str(record["seq"]): record["decision"] for record in records},
        "counts": counts,
        "affected_by_error": [
            record["seq"] for record in records if record["affected_by_error"]
        ],
    }


def _replay_committed(
    spec: Spec,
    session: SessionResult,
    ledger: Ledger,
) -> dict[str, Any]:
    expected = {message.seq: _ledger_record(session, message.seq) for message in spec.messages}
    actual = ledger.committed_records()
    expected_prefix = [message.seq for message in spec.messages[: len(actual)]]
    if sorted(actual) != expected_prefix:
        missing = sorted(set(expected_prefix) - set(actual))
        seq = missing[0] if missing else expected_prefix[0]
        raise DomainError(
            "E_REPLAY",
            "committed ledger records are not a contiguous message prefix",
            cert_id=None,
            field_path="message.presence",
            message_seq=seq,
        )
    expected_committed = {seq: expected[seq] for seq in actual if seq in expected}
    missing_expected = sorted(set(actual) - set(expected))
    if missing_expected:
        raise DomainError(
            "E_REPLAY",
            f"ledger contains message {missing_expected[0]} that is absent from the current script",
            cert_id=None,
            field_path="message.presence",
            message_seq=missing_expected[0],
        )
    difference = first_replay_difference(expected_committed, actual)
    if difference is not None:
        raise DomainError(
            "E_REPLAY",
            f"replay mismatch at message {difference['seq']}",
            cert_id=difference.get("cert_id"),
            field_path=difference.get("field"),
            message_seq=difference.get("seq"),
            details=difference,
        )
    for batch in ledger.read_batch_payloads():
        expected_trace = _ledger_trace(session)
        actual_trace = batch.get("invalidation_trace")
        if canonical_text(expected_trace) != canonical_text(actual_trace):
            first_seq = None
            messages = batch.get("messages", [])
            if messages:
                first_seq = messages[0].get("seq")
            raise DomainError(
                "E_REPLAY",
                "invalidation trace differs from ledger batch",
                cert_id=session.spec.messages[0].cert_ids[0] if session.spec.messages else None,
                field_path="invalidation_trace",
                message_seq=first_seq,
            )
    return {
        "verified_seqs": sorted(actual),
        "verified_batches": len(ledger.list_batches()),
        "content_digest": ledger.content_digest(),
    }


def execute(
    cert_path: str | Path,
    handshake_path: str | Path,
    spec: Spec,
    ledger_path: str | Path,
    port: int,
    resume: bool = False,
    terminate_after: int | None = None,
    termination: str = "graceful",
    drop_client_seq: int | None = None,
    omit_reply_field: str | None = None,
    host: str = "127.0.0.1",
) -> dict[str, Any]:
    ledger_dir = Path(ledger_path).resolve().parent
    run_report_path = ledger_dir / "run-report.json"
    exit_report_path = ledger_dir / "exit-report.json"
    session = build_full_session(spec)
    controller = StopController()
    controller.install()
    with Ledger(ledger_path) as ledger:
        snapshot_result = ledger.sync_snapshot(spec)
        committed_before = ledger.progress_cursor()
        replay_result: dict[str, Any] | None = None
        final_replay_result: dict[str, Any] | None = None
        if resume:
            replay_result = _replay_committed(spec, session, ledger)
        initial_unknown = ledger.unknown_sequences(spec) if resume else []
        groups = spec.message_groups()
        service = LoopbackService(
            spec,
            cert_path,
            handshake_path,
            port,
            host,
            drop_client_seq,
            omit_reply_field,
        )
        try:
            service.start()
            for group in groups:
                request = group["request"]
                replies = service.exchange(request)
                _validate_replies(request, group["replies"], replies)
                group_messages: list[ScriptMessage] = [request, *group["replies"]]
                records = [_ledger_record(session, message.seq) for message in group_messages]
                through_seq = max(record["seq"] for record in records)
                if through_seq <= committed_before:
                    continue
                if request.seq <= committed_before < through_seq:
                    raise DomainError(
                        "E_REPLAY",
                        "ledger cursor splits a declared handshake batch",
                        cert_id=request.cert_ids[0] if request.cert_ids else None,
                        field_path="progress_cursor",
                        message_seq=through_seq,
                    )
                batch_index = ledger.next_batch_index()
                ledger.commit_batch(
                    batch_index=batch_index,
                    records=records,
                    spec_digest=digest_hex(spec.spec_digest_input()),
                    progress_cursor=through_seq,
                    invalidation_trace=_ledger_trace(session),
                )
                if terminate_after is not None and through_seq >= terminate_after:
                    if termination == "hard":
                        hard_terminate_self()
                    controller.request_graceful_signal()
                if controller.requested:
                    break
                if initial_unknown and through_seq >= max(initial_unknown):
                    break
        finally:
            service.stop()
        cursor = ledger.progress_cursor()
        if resume and cursor >= spec.messages[-1].seq:
            final_replay_result = _replay_committed(spec, session, ledger)
        persisted = ledger.committed_records()
        all_records = [persisted[seq] for seq in sorted(persisted)]
        counts = {"PASS": 0, "FAIL": 0, "INDETERMINATE": 0}
        affected_count = 0
        expected_mismatches: list[int] = []
        for record in all_records:
            counts[record["decision"]] += 1
            if record["affected_by_error"]:
                affected_count += 1
            if not record["expected_match"]:
                expected_mismatches.append(record["seq"])
        remaining = [message.seq for message in spec.messages if message.seq > cursor]
        common = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "spec_digest": digest_hex(spec.spec_digest_input()),
            "cert_source_digest": spec.cert_source_digest,
            "handshake_source_digest": spec.handshake_source_digest,
            "ledger_path": str(Path(ledger_path).resolve()),
            "host": host,
            "port": port,
            "socket_timeout_ms": spec.socket_timeout_ms,
            "snapshot_sync": snapshot_result,
            "ledger_content_digest": ledger.content_digest(),
            "progress_cursor": cursor,
            "committed_batches": ledger.list_batches(),
            "batch_summaries": [
                _batch_summary(batch["messages"])
                for batch in ledger.read_batch_payloads()
            ],
            "decision_counts": counts,
            "affected_by_error_count": affected_count,
            "expected_mismatch_seqs": expected_mismatches,
            "remaining_message_seqs": remaining,
            "initial_unknown_seqs": initial_unknown,
            "resume": resume,
            "replay": replay_result,
            "final_replay": final_replay_result,
            "termination_support": termination_support(),
        }
        if controller.requested or remaining:
            report = {
                **common,
                "status": "GRACEFUL_EXIT" if controller.requested else "INCOMPLETE",
                "exit_kind": "graceful-signal" if controller.requested else "not-finished",
                "signal": controller.signal_name,
                "complete_exit_report": True,
                "committed_message_count": len(all_records),
                "unprocessed_message_seqs": remaining,
                "ledger_only_recovery_required": controller.requested,
            }
            if controller.requested:
                _atomic_json(exit_report_path, report)
            return report
        report = {
            **common,
            "status": "COMPLETE",
            "exit_kind": "complete",
            "complete_exit_report": True,
            "committed_message_count": len(all_records),
            "unprocessed_message_seqs": [],
            "replayed_message_seqs": replay_result["verified_seqs"] if replay_result else [],
            "rerun_message_seqs": [seq for seq in initial_unknown],
            "ledger_only_recovery_required": False,
        }
        _atomic_json(run_report_path, report)
        return report


def reset_ledger_files(ledger_path: str | Path) -> None:
    path = Path(ledger_path)
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    report_dir = path.resolve().parent
    for name in ("run-report.json", "exit-report.json"):
        try:
            (report_dir / name).unlink()
        except FileNotFoundError:
            pass
