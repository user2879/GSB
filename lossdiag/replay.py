"""Ledger replay with first-difference localization."""

from __future__ import annotations

from typing import Any, Iterable

from .errors import replay_error
from .pipeline import build_session
from .store import LedgerRecord, WorkbenchStore
from .util import content_digest


def _first_mapping_difference(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    flow_id: str | None,
    gap_interval: dict[str, int] | None,
    prefix: str = "",
) -> tuple[str, Any, Any] | None:
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            return f"{prefix}{key}", None, actual[key]
        if key not in actual:
            return f"{prefix}{key}", expected[key], None
        if expected[key] != actual[key]:
            return f"{prefix}{key}", expected[key], actual[key]
    return None


def _gap_map(flow: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result = {}
    for gap in flow.get("gaps", []):
        key = (
            int(gap["gap_interval"]["start_seq"]),
            int(gap["gap_interval"]["end_seq"]),
        )
        result[key] = gap
    return result


def find_first_report_difference(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    version_diff = _first_mapping_difference(
        expected.get("version_combo", {}),
        actual.get("version_combo", {}),
        flow_id=None,
        gap_interval=None,
        prefix="version_combo.",
    )
    if version_diff:
        field, left, right = version_diff
        raise replay_error(
            "ledger replay version combination differs",
            flow_id=None,
            gap_interval=None,
            field=field,
            expected=left,
            actual=right,
        )

    alignment_diff = _first_mapping_difference(
        expected.get("alignment", {}),
        actual.get("alignment", {}),
        flow_id=None,
        gap_interval=None,
        prefix="alignment.",
    )
    if alignment_diff:
        field, left, right = alignment_diff
        raise replay_error(
            "ledger replay alignment differs",
            flow_id=None,
            gap_interval=None,
            field=field,
            expected=left,
            actual=right,
        )

    expected_flows = {
        flow["flow_id"]: flow for flow in expected.get("diagnosis", {}).get("flows", [])
    }
    actual_flows = {
        flow["flow_id"]: flow for flow in actual.get("diagnosis", {}).get("flows", [])
    }
    for flow_id in sorted(set(expected_flows) | set(actual_flows)):
        if flow_id not in expected_flows:
            raise replay_error(
                "ledger replay produced an unexpected flow",
                flow_id=flow_id,
                gap_interval=None,
                field="flow_presence",
                expected=None,
                actual=actual_flows[flow_id],
            )
        if flow_id not in actual_flows:
            raise replay_error(
                "ledger replay is missing a flow",
                flow_id=flow_id,
                gap_interval=None,
                field="flow_presence",
                expected=expected_flows[flow_id],
                actual=None,
            )
        expected_flow = expected_flows[flow_id]
        actual_flow = actual_flows[flow_id]
        for field_name in (
            "five_tuple",
            "sequence_envelope",
            "observed_unique_count",
            "observed_frame_count",
        ):
            if expected_flow.get(field_name) != actual_flow.get(field_name):
                raise replay_error(
                    "ledger replay flow metadata differs",
                    flow_id=flow_id,
                    gap_interval=None,
                    field=field_name,
                    expected=expected_flow.get(field_name),
                    actual=actual_flow.get(field_name),
                )

        expected_gaps = _gap_map(expected_flow)
        actual_gaps = _gap_map(actual_flow)
        for key in sorted(set(expected_gaps) | set(actual_gaps)):
            gap_interval = {"start_seq": key[0], "end_seq": key[1]}
            if key not in expected_gaps:
                raise replay_error(
                    "ledger replay produced an unexpected gap",
                    flow_id=flow_id,
                    gap_interval=gap_interval,
                    field="gap_presence",
                    expected=None,
                    actual=actual_gaps[key],
                )
            if key not in actual_gaps:
                raise replay_error(
                    "ledger replay is missing a gap",
                    flow_id=flow_id,
                    gap_interval=gap_interval,
                    field="gap_presence",
                    expected=expected_gaps[key],
                    actual=None,
                )
            expected_gap = expected_gaps[key]
            actual_gap = actual_gaps[key]
            for field_name in (
                "classification",
                "count",
                "evidence",
                "source_versions",
                "time_window_ms",
                "time_source",
            ):
                if expected_gap.get(field_name) != actual_gap.get(field_name):
                    raise replay_error(
                        "ledger replay gap field differs",
                        flow_id=flow_id,
                        gap_interval=gap_interval,
                        field=field_name,
                        expected=expected_gap.get(field_name),
                        actual=actual_gap.get(field_name),
                    )

        counts_diff = _first_mapping_difference(
            expected_flow.get("counts", {}),
            actual_flow.get("counts", {}),
            flow_id=flow_id,
            gap_interval=None,
            prefix="counts.",
        )
        if counts_diff:
            field, left, right = counts_diff
            raise replay_error(
                "ledger replay flow counts differ",
                flow_id=flow_id,
                gap_interval=None,
                field=field,
                expected=left,
                actual=right,
            )
        if expected_flow.get("gap_digest") != actual_flow.get("gap_digest"):
            raise replay_error(
                "ledger replay gap digest differs",
                flow_id=flow_id,
                gap_interval=None,
                field="gap_digest",
                expected=expected_flow.get("gap_digest"),
                actual=actual_flow.get("gap_digest"),
            )

    expected_rates = {flow["flow_id"]: flow for flow in expected.get("rates", {}).get("flows", [])}
    actual_rates = {flow["flow_id"]: flow for flow in actual.get("rates", {}).get("flows", [])}
    for flow_id in sorted(set(expected_rates) | set(actual_rates)):
        if flow_id not in expected_rates or flow_id not in actual_rates:
            raise replay_error(
                "ledger replay rate flow presence differs",
                flow_id=flow_id,
                gap_interval=None,
                field="rates.flow_presence",
                expected=expected_rates.get(flow_id),
                actual=actual_rates.get(flow_id),
            )
        expected_rate = expected_rates[flow_id]
        actual_rate = actual_rates[flow_id]
        for field_name in (
            "true_loss",
            "expected",
            "rate_ppm",
            "exact_fraction",
            "unwindowed_true_loss",
            "windows",
        ):
            if expected_rate.get(field_name) != actual_rate.get(field_name):
                raise replay_error(
                    "ledger replay rate field differs",
                    flow_id=flow_id,
                    gap_interval=None,
                    field=f"rates.{field_name}",
                    expected=expected_rate.get(field_name),
                    actual=actual_rate.get(field_name),
                )

    for field_name in ("ranking", "rate_comparison"):
        if expected.get("rates", {}).get(field_name) != actual.get("rates", {}).get(field_name):
            raise replay_error(
                "ledger replay rate ordering differs",
                flow_id=None,
                gap_interval=None,
                field=f"rates.{field_name}",
                expected=expected.get("rates", {}).get(field_name),
                actual=actual.get("rates", {}).get(field_name),
            )
    if expected.get("report_digest") != actual.get("report_digest"):
        raise replay_error(
            "ledger replay report digest differs",
            flow_id=None,
            gap_interval=None,
            field="report_digest",
            expected=expected.get("report_digest"),
            actual=actual.get("report_digest"),
        )
    if expected.get("inputs", {}).get("input_digest") != actual.get("inputs", {}).get("input_digest"):
        raise replay_error(
            "ledger replay input digest differs while conclusions happened to match",
            flow_id=None,
            gap_interval=None,
            field="inputs.input_digest",
            expected=expected.get("inputs", {}).get("input_digest"),
            actual=actual.get("inputs", {}).get("input_digest"),
        )
    return {
        "first_difference": None,
        "matches": True,
    }


def verify_ledger(
    store: WorkbenchStore,
    ledger_uid: str,
    *,
    wire_path: str | None = None,
    wire_manifest_path: str | None = None,
    event_paths: list[str] | None = None,
    window_ms: int | None = None,
) -> dict[str, Any]:
    ledger: LedgerRecord = store.load_ledger(ledger_uid)
    stored_inputs = ledger.payload.get("inputs", {})
    replay_wire = wire_path or stored_inputs.get("wire_path")
    replay_manifest = wire_manifest_path or stored_inputs.get("wire_manifest_path")
    replay_events = event_paths or stored_inputs.get("event_paths")
    replay_window = window_ms if window_ms is not None else ledger.payload.get("window_ms")
    if not replay_wire or not replay_manifest or not replay_events or replay_window is None:
        raise replay_error(
            "ledger record lacks replayable input declarations",
            flow_id=None,
            gap_interval=None,
            field="ledger.inputs",
            expected=stored_inputs,
            actual={
                "wire_path": replay_wire,
                "wire_manifest_path": replay_manifest,
                "event_paths": replay_events,
                "window_ms": replay_window,
            },
        )
    session = build_session(
        wire_path=replay_wire,
        wire_manifest_path=replay_manifest,
        event_paths=list(replay_events),
        window_ms=int(replay_window),
    )
    try:
        comparison = find_first_report_difference(ledger.report, session.report)
    except Exception as exc:
        if getattr(exc, "code", None) == "E_REPLAY":
            exc.details["ledger_uid"] = ledger_uid
            exc.details["replay_input_digest"] = session.input_digest
            exc.details["stored_input_digest"] = ledger.input_digest
        raise
    return {
        "ok": True,
        "ledger_uid": ledger_uid,
        "matches": comparison["matches"],
        "first_difference": comparison["first_difference"],
        "replay_report_digest": session.report_digest,
        "ledger_report_digest": ledger.report_digest,
        "replay_input_digest": session.input_digest,
        "ledger_input_digest": ledger.input_digest,
        "verified_at": session.report["determinism"]["serialization"],
    }


def compare_payload_digests(payloads: Iterable[dict[str, Any]]) -> list[str]:
    return [content_digest(payload) for payload in payloads]
