#!/usr/bin/env python3
"""Run the local acceptance scenario and persist a machine-readable report."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_pipeline.compare import compare_states  # noqa: E402
from feature_pipeline.engine import PipelineEngine  # noqa: E402
from feature_pipeline.state import SnapshotStore  # noqa: E402
from feature_pipeline.validation import (  # noqa: E402
    E_EVENT_INVALID,
    E_FEATURE_CYCLE,
    E_SCHEMA_UNKNOWN_FIELD,
    E_TIMESTAMP_INVALID,
    load_event_file,
    validate_feature_table,
)


def run_engine(
    engine: PipelineEngine,
    *,
    events: Path,
    features: Path,
    labels: Path,
    report: Path,
    reset: bool = False,
    edits: Path | None = None,
    injections: Path | None = None,
    config: Path | None = None,
    stop_after: str | None = None,
) -> tuple[dict[str, Any], int]:
    result, code = engine.run(
        config_path=config or ROOT / "samples/config.json",
        events_path=events,
        features_path=features,
        labels_path=labels,
        edits_path=edits,
        injections_path=injections,
        reset=reset,
        stop_after=stop_after,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result, code


def validation_checks(work: Path) -> dict[str, Any]:
    config_types = {"view", "click", "add_to_cart", "purchase"}
    bad_jsonl = work / "bad_events.jsonl"
    bad_jsonl.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps(
                    {
                        "event_id": "bad-unknown",
                        "user_id": "u1",
                        "item_id": "i1",
                        "event_type": "view",
                        "timestamp": 1,
                        "context": {"price": 1},
                        "unexpected": True,
                    }
                ),
                json.dumps(
                    {
                        "event_id": "bad-ts",
                        "user_id": "u1",
                        "item_id": "i1",
                        "event_type": "view",
                        "timestamp": -1,
                        "context": {"price": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _, event_failures = load_event_file(bad_jsonl, config_types)
    cycle_raw = {
        "features": [
            {
                "name": "a",
                "entity": "user",
                "window_ms": 1,
                "aggregation": "count",
                "upstream_features": ["b"],
            },
            {
                "name": "b",
                "entity": "user",
                "window_ms": 1,
                "aggregation": "count",
                "upstream_features": ["a"],
            },
        ]
    }
    _, cycle_failures = validate_feature_table(cycle_raw)
    codes = sorted({item["code"] for item in event_failures + cycle_failures})
    checks = {
        "invalid_json_code": any(
            item["code"] == E_EVENT_INVALID and item["error_type"] == "JSONDecodeError"
            for item in event_failures
        ),
        "unknown_field_code": any(
            item["code"] == E_SCHEMA_UNKNOWN_FIELD and item.get("field") == "unexpected"
            for item in event_failures
        ),
        "invalid_timestamp_code": any(
            item["code"] == E_TIMESTAMP_INVALID for item in event_failures
        ),
        "cycle_code": any(item["code"] == E_FEATURE_CYCLE for item in cycle_failures),
    }
    return {"passed": all(checks.values()), "checks": checks, "observed_codes": codes}


def main() -> int:
    samples = ROOT / "samples"
    output = ROOT / "artifacts/acceptance"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    work = output / "work"
    work.mkdir()

    incremental_state = work / "runtime-incremental"
    full_state = work / "runtime-full"
    serial_state = work / "runtime-serial"
    recovery_state = work / "runtime-recovery"
    injection_state = work / "runtime-injection"
    engine = PipelineEngine(incremental_state)

    reports: dict[str, dict[str, Any]] = {}
    reports["base"], _ = run_engine(
        engine,
        events=samples / "events_base_10.jsonl",
        features=samples / "features_base.json",
        labels=samples / "labels_base.json",
        report=work / "inc_00_base.json",
        reset=True,
    )
    reports["append"], _ = run_engine(
        engine,
        events=samples / "events_pre_backfill.jsonl",
        features=samples / "features_base.json",
        labels=samples / "labels_base.json",
        report=work / "inc_01_append.json",
    )
    reports["window"], _ = run_engine(
        engine,
        events=samples / "events_pre_backfill.jsonl",
        features=samples / "features_window_changed.json",
        labels=samples / "labels_base.json",
        report=work / "inc_02_window.json",
    )
    reports["aggregation_delay"], _ = run_engine(
        engine,
        events=samples / "events_pre_backfill.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        report=work / "inc_03_agg_delay.json",
    )
    reports["backfill"], _ = run_engine(
        engine,
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        edits=samples / "edits.json",
        report=work / "inc_04_backfill.json",
    )
    reports["full"], _ = run_engine(
        PipelineEngine(full_state),
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        report=work / "full.json",
        reset=True,
    )
    inc_state, _ = SnapshotStore(incremental_state).recover()
    full_rebuilt, _ = SnapshotStore(full_state).recover()
    equivalence = compare_states(inc_state, full_rebuilt)
    feature_sample_stage_versions_equal = all(
        inc_state.manifest.get("stage_versions", {}).get(stage)
        == full_rebuilt.manifest.get("stage_versions", {}).get(stage)
        for stage in ("features", "samples")
    )
    definition_hash_equal = (
        inc_state.manifest.get("definition_hash")
        == full_rebuilt.manifest.get("definition_hash")
    )

    reports["serial"], _ = run_engine(
        PipelineEngine(serial_state),
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        config=samples / "config_serial.json",
        report=work / "serial_full.json",
        reset=True,
    )
    serial_state_value, _ = SnapshotStore(serial_state).recover()
    parallel_vs_serial = compare_states(inc_state, serial_state_value)

    reports["stop_events"], stop_exit = run_engine(
        PipelineEngine(recovery_state),
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        report=work / "recovery_stop_events.json",
        reset=True,
        stop_after="events",
    )
    reports["resume"], resume_exit = run_engine(
        PipelineEngine(recovery_state),
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        report=work / "recovery_resume.json",
    )
    resumed_state, _ = SnapshotStore(recovery_state).recover()
    recovery_equivalence = compare_states(resumed_state, full_rebuilt)

    injection_engine = PipelineEngine(injection_state)
    run_engine(
        injection_engine,
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        report=work / "injection_seed.json",
        reset=True,
    )
    reports["injection"], injection_exit = run_engine(
        injection_engine,
        events=samples / "events.jsonl",
        features=samples / "features.json",
        labels=samples / "labels.json",
        edits=samples / "edits_injection.json",
        injections=samples / "injections.json",
        report=work / "injection_hit.json",
    )
    injection_passed = all(item["hit"] for item in reports["injection"]["injected_items"])

    validations = validation_checks(work)
    ledger_checks = {
        "window_only_invalidates_changed_feature": reports["window"]["invalidation_ledger"][
            "invalidated_features"
        ]
        == ["item_view_count_5s"],
        "append_reuses_all_feature_names": reports["append"]["invalidation_ledger"][
            "reused_features"
        ]
        == sorted(reports["append"]["invalidation_ledger"]["reused_features"]),
        "backfill_preserves_unrelated_features": {
            "item_view_count_5s",
            "user_purchase_count_10s",
            "user_view_count_5s",
        }.issubset(reports["backfill"]["invalidation_ledger"]["reused_features"]),
        "empty_windows_marked": reports["full"]["empty_window_feature_count"] > 0,
    }
    checks = {
        "incremental_full_equivalence": equivalence["equal"],
        "feature_sample_stage_versions_equal": feature_sample_stage_versions_equal,
        "definition_hash_equal": definition_hash_equal,
        "parallel_serial_equivalence": parallel_vs_serial["equal"],
        "graceful_stop_exit_zero": stop_exit == 0,
        "resume_exit_zero": resume_exit == 0,
        "resume_equivalence": recovery_equivalence["equal"],
        "injection_all_hit": injection_passed and injection_exit == 0,
        "business_codes": validations["passed"],
        "invalidation_ledger": all(ledger_checks.values()),
    }
    result = {
        "all_passed": all(checks.values()),
        "checks": checks,
        "layer_checks": checks,
        "equivalence": equivalence,
        "feature_sample_stage_versions_equal": feature_sample_stage_versions_equal,
        "definition_hash_equal": definition_hash_equal,
        "incremental_stage_versions": inc_state.manifest.get("stage_versions"),
        "full_stage_versions": full_rebuilt.manifest.get("stage_versions"),
        "parallel_vs_serial": parallel_vs_serial,
        "recovery_equivalence": recovery_equivalence,
        "invalidation_checks": ledger_checks,
        "validation": validations,
        "injected_items": reports["injection"]["injected_items"],
        "injection_failures": reports["injection"]["failures"],
        "step_ledgers": {
            name: reports[name]["invalidation_ledger"]
            for name in ("append", "window", "aggregation_delay", "backfill")
        },
        "empty_window_feature_count": reports["full"]["empty_window_feature_count"],
        "feature_snapshot_version_incremental": reports["backfill"]["feature_snapshot_version"],
        "feature_snapshot_version_full": reports["full"]["feature_snapshot_version"],
    }
    report_path = ROOT / "artifacts/acceptance_report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_passed": result["all_passed"], "checks": checks}, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
