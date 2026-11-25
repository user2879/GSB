from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import canonical_text, sha256_hex
from .diagnosis import diagnose
from .errors import VerifyError
from .invalidation import derive_invalidation, recompute_incremental
from .ledger import Ledger
from .model import STAGES, bundle_digest, load_bundle


LEDGER_STAGE_FIELDS = (
    "shaper_id",
    "evaluated",
    "verdict",
    "decision",
    "required_q",
    "balance_before_refill_q",
    "refill_q",
    "balance_before_decision_q",
    "balance_after_decision_q",
    "state_clock_ms",
)


def _verify_error(message: str, **locator: Any) -> VerifyError:
    return VerifyError(message, **locator)


def open_readonly_ledger(path: str | Path) -> Ledger:
    target = Path(path)
    if not target.exists():
        raise VerifyError("ledger file does not exist", object_id=str(target), field="ledger")
    uri = target.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=30)
    except sqlite3.DatabaseError as exc:
        raise VerifyError(f"cannot read ledger: {exc}", object_id=str(target), field="ledger") from exc
    conn.row_factory = sqlite3.Row
    ledger = Ledger(target, conn)
    stored = ledger.get_meta("format_version")
    if stored != "1":
        raise VerifyError(f"unsupported ledger format_version {stored}", field="format_version", actual=stored, expected="1")
    digest_stored = ledger.get_meta("content_digest")
    digest_computed = ledger.compute_content_digest()
    if digest_stored != digest_computed:
        raise VerifyError(
            "ledger content digest does not match rows",
            field="content_digest",
            expected=digest_computed,
            actual=digest_stored,
        )
    return ledger


def _compare_stage_fields(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    event_seq: int,
    pipeline_id: str,
    stage: str,
) -> None:
    expected_allowed = expected.get("verdict") == "allow"
    actual_allowed = bool(actual.get("allowed"))
    if actual_allowed != expected_allowed:
        raise VerifyError(
            "ledger allowed column differs from replay verdict",
            event_seq=event_seq,
            pipeline_id=pipeline_id,
            stage=stage,
            field="allowed",
            expected=expected_allowed,
            actual=actual_allowed,
        )
    for field in LEDGER_STAGE_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if field == "evaluated":
            actual_value = bool(actual_value)
            actual_for_compare = actual_value
        else:
            actual_for_compare = actual_value
        if expected_value != actual_for_compare:
            raise VerifyError(
                "ledger stage field differs from deterministic replay",
                event_seq=event_seq,
                pipeline_id=pipeline_id,
                stage=stage,
                field=field,
                expected=expected_value,
                actual=actual_value,
            )
    expected_record_text = canonical_text(expected)
    actual_record_text = actual.get("record_json")
    if actual_record_text != expected_record_text:
        raise VerifyError(
            "ledger canonical stage record is not byte-identical",
            event_seq=event_seq,
            pipeline_id=pipeline_id,
            stage=stage,
            field="record_json",
            expected=expected_record_text,
            actual=actual_record_text,
        )
    if actual.get("receipt_digest") and actual["receipt_digest"] != sha256_hex(expected):
        raise VerifyError(
            "receipt digest does not match replayed stage record",
            event_seq=event_seq,
            pipeline_id=pipeline_id,
            stage=stage,
            field="receipt_digest",
            expected=sha256_hex(expected),
            actual=actual.get("receipt_digest"),
        )
    if actual.get("wait_ms") is not None:
        if int(actual["wait_ms"]) < 0:
            raise VerifyError(
                "wait_ms must be non-negative",
                event_seq=event_seq,
                pipeline_id=pipeline_id,
                stage=stage,
                field="wait_ms",
                actual=actual["wait_ms"],
            )
        if int(actual["wait_end_ms"]) - int(actual["wait_start_ms"]) != int(actual["wait_ms"]):
            raise VerifyError(
                "wait interval is inconsistent",
                event_seq=event_seq,
                pipeline_id=pipeline_id,
                stage=stage,
                field="wait_ms",
                expected=int(actual["wait_end_ms"]) - int(actual["wait_start_ms"]),
                actual=actual["wait_ms"],
            )


def verify_ledger(
    spec_dir: str | Path,
    ledger_path: str | Path,
    *,
    revision: int | None = None,
) -> dict[str, Any]:
    bundle = load_bundle(spec_dir)
    digest = bundle_digest(bundle)
    bundle["spec_digest"] = digest
    ledger = open_readonly_ledger(ledger_path)
    try:
        revision = int(revision or ledger.current_revision())
        revision_row = ledger.conn.execute(
            "SELECT * FROM revisions WHERE revision=?", (revision,)
        ).fetchone()
        if revision_row is None:
            raise VerifyError("ledger has no revision record", field="revision", actual=revision)
        if revision_row["spec_digest"] != digest:
            raise VerifyError(
                "raw spec digest does not match latest ledger revision",
                field="spec_digest",
                expected=digest,
                actual=revision_row["spec_digest"],
            )

        artifacts = ledger.load_artifacts(revision)
        requested_path = artifacts.get("path_selection", {}).get("requested_path", "auto")
        diagnosis = diagnose(bundle, forced_path=requested_path)
        expected_results = diagnosis["events"]
        stored_results = ledger.load_event_results(revision)
        if len(stored_results) != len(expected_results):
            raise VerifyError(
                "ledger event count differs from raw event stream",
                field="event_seq",
                expected=len(expected_results),
                actual=len(stored_results),
            )

        judgment_rows = ledger.load_judgment_rows(revision)
        rows_by_key = {(int(row["event_seq"]), row["stage"]): row for row in judgment_rows}
        stored_by_seq = {int(item["event_seq"]): item for item in stored_results}

        for expected_event in expected_results:
            seq = int(expected_event["event_seq"])
            stored_event = stored_by_seq.get(seq)
            if stored_event is None:
                raise VerifyError(
                    "ledger is missing event",
                    event_seq=seq,
                    pipeline_id=expected_event["pipeline_id"],
                    field="event_seq",
                )
            for field in ("event_id", "pipeline_id", "arrival_ms", "packet_length", "window_start_ms", "verdict", "decision"):
                if stored_event.get(field) != expected_event.get(field):
                    raise VerifyError(
                        "event field differs from replay",
                        event_seq=seq,
                        pipeline_id=expected_event["pipeline_id"],
                        field=field,
                        expected=expected_event.get(field),
                        actual=stored_event.get(field),
                    )
            if canonical_text(stored_event) != canonical_text(expected_event):
                raise VerifyError(
                    "stored event JSON is not byte-identical to replay",
                    event_seq=seq,
                    pipeline_id=expected_event["pipeline_id"],
                    field="event_json",
                )
            for stage_index, stage in enumerate(STAGES):
                expected_stage = expected_event["stages"][stage_index]
                row = rows_by_key.get((seq, stage))
                if row is None:
                    raise VerifyError(
                        "ledger is missing stage judgment",
                        event_seq=seq,
                        pipeline_id=expected_event["pipeline_id"],
                        stage=stage,
                        field="stage",
                    )
                _compare_stage_fields(
                    expected_stage,
                    row,
                    event_seq=seq,
                    pipeline_id=expected_event["pipeline_id"],
                    stage=stage,
                )

        for artifact_key, diagnosis_key in (
            ("path_selection", "path_selection"),
            ("stage_stats", "stage_stats"),
            ("error_bounds", "error_bounds"),
            ("windows", "windows"),
        ):
            stored_artifact = artifacts.get(artifact_key)
            expected_artifact = diagnosis.get(diagnosis_key)
            if canonical_text(stored_artifact) != canonical_text(expected_artifact):
                raise VerifyError(
                    "ledger fixed-point/path artifact differs from replay",
                    field=artifact_key,
                    expected=sha256_hex(expected_artifact),
                    actual=sha256_hex(stored_artifact) if stored_artifact is not None else None,
                )

        _verify_revision_propagation(ledger)

        cursor = ledger.cursor()
        if int(cursor["sealed_seq"]) != len(expected_results) - 1:
            raise VerifyError(
                "recovery cursor does not cover final event",
                field="sealed_seq",
                expected=len(expected_results) - 1,
                actual=cursor["sealed_seq"],
            )
        return {
            "ok": True,
            "code": "E_VERIFY_OK",
            "revision": revision,
            "verified_events": len(expected_results),
            "verified_judgments": len(judgment_rows),
            "spec_digest": digest,
            "content_digest": ledger.get_meta("content_digest"),
            "unknown_batch_history": len(ledger.unknown_batches()),
        }
    finally:
        ledger.close()


def _verify_revision_propagation(ledger: Ledger) -> None:
    rows = ledger.conn.execute("SELECT * FROM revisions ORDER BY revision").fetchall()
    prior_results: list[dict[str, Any]] | None = None
    for row in rows:
        revision = int(row["revision"])
        change = json.loads(row["change_json"] or "{}")
        propagation = json.loads(row["propagation_json"] or "{}")
        if not change.get("edit"):
            prior_results = ledger.load_event_results(revision)
            continue
        old_spec = change.get("old_spec")
        new_spec = change.get("new_spec")
        if not old_spec or not new_spec:
            raise VerifyError("revision edit is missing replayable spec snapshots", field="change_json")
        expected_closure = derive_invalidation(old_spec, new_spec)
        if canonical_text(propagation) != canonical_text(expected_closure):
            raise VerifyError(
                "stored invalidation propagation differs from independent replay",
                field="propagation_json",
                expected=sha256_hex(expected_closure),
                actual=sha256_hex(propagation),
            )
        parent = int(row["parent_revision"] or revision - 1)
        if prior_results is None:
            prior_results = ledger.load_event_results(parent)
        incremental = recompute_incremental(old_spec, new_spec, prior_results, expected_closure)
        current_results = ledger.load_event_results(revision)
        if canonical_text(incremental["results"]) != canonical_text(current_results):
            first = incremental["unchanged_check"].get("first_mismatch") or {}
            raise VerifyError(
                "incremental invalidation replay differs from ledger",
                event_seq=first.get("event_seq"),
                stage=first.get("stage"),
                field=first.get("field", "incremental_results"),
            )
        prior_results = current_results
