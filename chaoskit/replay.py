"""Replay verification and first-difference localization.

After recovery the committed timeline is regenerated from the recorded seed and
replayed entry by entry. Structural fields (start time, target, parameters)
must match exactly; observation summaries are compared within their declared
quantization bounds plus scheduling jitter. The first mismatch is localized to
injection index, field name and both values, then reported as ``E_REPLAY``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .errors import ReplayError
from .ledger import Ledger
from .rng import mix_u64
from .spec import ExperimentSpec
from .timeline import TimelineEntry, build_timeline, timeline_digest

_STRUCTURAL_FIELDS = (
    "start_offset_ms",
    "target_id",
    "injection_id",
    "injection_type",
    "duration_ms",
    "params",
)


def verify_identity(spec: ExperimentSpec, ledger: Ledger) -> None:
    row = ledger.experiment_row()
    if row is None:
        raise ReplayError(
            "ledger has no experiment snapshot",
            basis="ledger_missing",
            injection_index=-1,
            field="experiment",
            ledger_value=None,
            replay_value=None,
        )
    if row["seed"] != spec.seed:
        raise ReplayError(
            "replay seed differs from ledger",
            basis="seed",
            injection_index=-1,
            field="seed",
            ledger_value=row["seed"],
            replay_value=spec.seed,
        )
    if row["spec_digest"] != spec.spec_digest:
        raise ReplayError(
            "target service configuration differs from ledger snapshot",
            basis="spec_digest",
            injection_index=-1,
            field="spec_digest",
            ledger_value=row["spec_digest"],
            replay_value=spec.spec_digest,
        )
    if row["timeline_digest"] != timeline_digest(build_timeline(spec)):
        raise ReplayError(
            "regenerated timeline digest differs from ledger",
            basis="timeline_digest",
            injection_index=-1,
            field="timeline_digest",
            ledger_value=row["timeline_digest"],
            replay_value=timeline_digest(build_timeline(spec)),
        )


def verify_timeline_entries(spec: ExperimentSpec, ledger: Ledger) -> None:
    regenerated = build_timeline(spec)
    for entry in regenerated:
        row = ledger.injection_row(entry.index)
        if row is None:
            raise ReplayError(
                f"injection {entry.index} missing from ledger",
                basis="missing_injection",
                injection_index=entry.index,
                field="existence",
                ledger_value=None,
                replay_value=entry.to_dict(),
            )
        actual = {
            "start_offset_ms": row["start_offset_ms"],
            "target_id": row["target_id"],
            "injection_id": row["injection_id"],
            "injection_type": row["injection_type"],
            "duration_ms": row["duration_ms"],
            "params": json.loads(row["params_json"]),
        }
        for field_name in _STRUCTURAL_FIELDS:
            if getattr(entry, field_name) != actual[field_name]:
                raise ReplayError(
                    f"structural mismatch at injection {entry.index} field {field_name}",
                    basis="structural",
                    injection_index=entry.index,
                    field=field_name,
                    ledger_value=actual[field_name],
                    replay_value=getattr(entry, field_name),
                )


def _observation_tolerance(field: str, ledger_summary: Dict[str, Any],
                           replay_summary: Dict[str, Any], interval_ms: int) -> int:
    if field == "sample_count":
        return 1
    if field == "success_rate_bps":
        return max(
            ledger_summary.get("rate_quant_error_bps", 0),
            replay_summary.get("rate_quant_error_bps", 0),
        ) + 1
    if field in ("p50_latency_ms", "p95_latency_ms", "max_latency_ms"):
        # Whole-ms latency quantization (1ms) plus thread/socket scheduling
        # jitter bounded by one sampling interval; structural changes such as a
        # modified fixed cost remain far outside this envelope.
        return 2 + max(1, interval_ms)
    return 0


def verify_observations(ledger: Ledger, replay_summaries: Dict[int, List[Dict[str, Any]]],
                        interval_ms: int) -> None:
    for idx, replay_list in replay_summaries.items():
        stored_rows = ledger.observation_rows(idx)
        stored_map = {
            row["phase"]: json.loads(row["summary_json"])
            for row in stored_rows
        }
        for replay_summary in replay_list:
            phase = replay_summary["phase"]
            ledger_summary = stored_map.get(phase)
            if ledger_summary is None:
                raise ReplayError(
                    f"phase {phase} missing for injection {idx}",
                    basis="observation_phase",
                    injection_index=idx,
                    field=f"{phase}.existence",
                    ledger_value=None,
                    replay_value=replay_summary,
                )
            for field_name in (
                "sample_count",
                "success_count",
                "failure_count",
                "success_rate_bps",
                "p50_latency_ms",
                "p95_latency_ms",
                "max_latency_ms",
                "insufficient",
            ):
                ledger_value = ledger_summary.get(field_name)
                replay_value = replay_summary.get(field_name)
                if ledger_value is None or replay_value is None:
                    if ledger_value != replay_value:
                        raise ReplayError(
                            f"observation mismatch at injection {idx} {phase}.{field_name}",
                            basis="observation",
                            injection_index=idx,
                            field=f"{phase}.{field_name}",
                            ledger_value=ledger_value,
                            replay_value=replay_value,
                        )
                    continue
                tolerance = _observation_tolerance(field_name, ledger_summary, replay_summary, interval_ms)
                if abs(int(ledger_value) - int(replay_value)) > tolerance:
                    raise ReplayError(
                        f"observation mismatch at injection {idx} {phase}.{field_name}",
                        basis="observation",
                        injection_index=idx,
                        field=f"{phase}.{field_name}",
                        ledger_value=ledger_value,
                        replay_value=replay_value,
                    )


def replay_seed_for(entry: TimelineEntry, spec: ExperimentSpec) -> int:
    return mix_u64([spec.seed, entry.index, entry.injection_id])
