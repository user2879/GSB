"""Reproducible acceptance suite matching the task's requested scenarios."""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .calibration import downgrade_v2_to_v1, reverse_downgrade, write_downgrade_artifacts
from .codes import E_SIDECAR_MISSING
from .edits import load_edit_sequence, replacement_weights
from .engine import (
    ForcedInterrupt,
    QcalEngine,
    RunPaths,
    prepare_inputs,
    snapshot_summary,
)
from .inputs import operation_apply
from .io_utils import atomic_write_json, load_json


def _clean(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _missing_sidecar_check(paths: RunPaths) -> tuple[str, list[str]]:
    prepared = prepare_inputs(paths)
    v1_records, sidecar, thousandths = downgrade_v2_to_v1(prepared.parsed_v2.records)
    with tempfile.TemporaryDirectory(prefix="qcal-sidecar-check-") as temp_dir:
        temp_root = Path(temp_dir)
        write_downgrade_artifacts(temp_root, v1_records, sidecar, thousandths)
        _, failures, status = reverse_downgrade(
            v1_records, temp_root / "deliberately-absent.sidecar.json"
        )
    codes = sorted(set(failure.code for failure in failures))
    return status, codes


def _injection_hit_map(injection_config: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for injection in injection_config.get("injections", []):
        injection_id = str(injection.get("id", "INJ-UNKNOWN"))
        candidates = [failure for failure in failures if failure.get("injection_id") == injection_id]
        if not candidates:
            candidates = [
                failure
                for failure in failures
                if failure.get("layer") == injection.get("layer")
                and failure.get("code") == injection.get("expected_code")
            ]
        mapped_code = candidates[0]["code"] if candidates else None
        hits.append(
            {
                "id": injection_id,
                "type": injection.get("type"),
                "mapped_code": mapped_code,
                "hit": bool(candidates),
            }
        )
    hits.sort(key=lambda item: item["id"])
    return hits


def run_acceptance(samples_dir: str | Path, artifacts_dir: str | Path) -> dict[str, Any]:
    paths = RunPaths.discover(samples_dir)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle_manifest = load_json(Path(samples_dir) / "manifest.json")
    seed = int(bundle_manifest.get("seed", 20261121))

    full_out = artifacts / "full"
    full_state = artifacts / "state_full"
    _clean(full_out)
    _clean(full_state)
    full_report = QcalEngine(
        paths,
        output_dir=full_out,
        state_dir=full_state,
        fresh=True,
    ).run()
    full_digest = full_report["result_digest"]

    missing_status, missing_codes = _missing_sidecar_check(paths)

    base_prepared = prepare_inputs(paths)
    current_weights = copy.deepcopy(base_prepared.weights)
    current_config = copy.deepcopy(base_prepared.config)
    edit_root = artifacts / "edit_sequence"
    _clean(edit_root)
    incremental_state = edit_root / "incremental_state"
    _clean(incremental_state)
    steps: list[dict[str, Any]] = []
    sequence = [{"id": "step-00-baseline", "operation": {}}]
    for operation in load_edit_sequence(paths.directory / "edit_sequence.yaml"):
        sequence.append({"id": str(operation["id"]), "operation": operation})

    final_incremental_report: dict[str, Any] | None = None
    final_prepared = None
    for index, step in enumerate(sequence):
        operation = step["operation"]
        if operation:
            current_weights, current_config, _ = operation_apply(
                current_weights,
                current_config,
                operation,
                replacement_weights,
            )
        prepared = prepare_inputs(
            paths,
            config_override=current_config,
            weights_override=current_weights,
        )
        incremental_report = QcalEngine(
            paths,
            output_dir=edit_root / "incremental" / step["id"],
            state_dir=incremental_state,
            prepared=prepared,
            fresh=(index == 0),
        ).run()
        full_state_step = edit_root / "full_state" / step["id"]
        _clean(full_state_step)
        rebuild_report = QcalEngine(
            paths,
            output_dir=edit_root / "rebuild" / step["id"],
            state_dir=full_state_step,
            prepared=prepared,
            fresh=True,
        ).run()
        match = incremental_report["result_digest"] == rebuild_report["result_digest"]
        steps.append(
            {
                "id": step["id"],
                "digest": incremental_report["result_digest"],
                "match": match,
                "recomputed": incremental_report["reuse"]["recomputed"],
            }
        )
        final_incremental_report = incremental_report
        final_prepared = prepared

    assert final_prepared is not None and final_incremental_report is not None
    repeat_report = QcalEngine(
        paths,
        output_dir=edit_root / "incremental_repeat",
        state_dir=incremental_state,
        prepared=final_prepared,
    ).run()
    incremental_reuse = {
        "digest_equal": repeat_report["result_digest"] == final_incremental_report["result_digest"],
        "recomputed": repeat_report["reuse"]["recomputed"],
    }

    interrupt_out = artifacts / "forced_interrupt"
    interrupt_state = artifacts / "state_forced_interrupt"
    _clean(interrupt_out)
    _clean(interrupt_state)
    forced_exit = 0
    interrupted_snapshot: dict[str, Any] = {}
    try:
        QcalEngine(
            paths,
            output_dir=interrupt_out,
            state_dir=interrupt_state,
            fresh=True,
            force_interrupt_after_stage="quantile_calibration",
        ).run()
    except ForcedInterrupt as exc:
        forced_exit = exc.exit_code
        interrupted_snapshot = snapshot_summary(interrupt_state)
    resumed_report = QcalEngine(
        paths,
        output_dir=interrupt_out,
        state_dir=interrupt_state,
    ).run()
    forced = {
        "exit_code": forced_exit,
        "recovered_stages": interrupted_snapshot.get("recovered_stages", {}),
        "recovered_digest_equal_full": resumed_report["result_digest"] == full_digest,
    }

    budget_state = artifacts / "state_budget"
    _clean(budget_state)
    budget_report = QcalEngine(
        paths,
        output_dir=artifacts / "budget_limited",
        state_dir=budget_state,
        fresh=True,
        budget_override=96,
    ).run()
    budget_resumed = QcalEngine(
        paths,
        output_dir=artifacts / "budget_resumed",
        state_dir=budget_state,
        budget_override=0,
    ).run()
    budget = {
        "pending": budget_report["pending"],
        "resume_digest_equal_full": budget_resumed["result_digest"] == full_digest,
    }

    injection_config = load_json(paths.directory / "injections.yaml")
    injection_state = artifacts / "state_injection"
    _clean(injection_state)
    injection_report = QcalEngine(
        paths,
        output_dir=artifacts / "injection",
        state_dir=injection_state,
        fresh=True,
        injection_path=paths.directory / "injections.yaml",
        budget_override=32,
    ).run()
    hits = _injection_hit_map(injection_config, injection_report["failures"])
    injection = {
        "hits": hits,
        "completed_quantization_layers": sorted(injection_report["quantization_errors"]),
        "all_hit": all(item["hit"] for item in hits),
    }

    report = {
        "schema_version": "qcal-acceptance/1.0",
        "seed": seed,
        "full_run": {
            "result_digest": full_digest,
            "failures": full_report["failure_count"],
            "pending": full_report["pending_count"],
            "quantization_errors": full_report["quantization_errors"],
            "distillation_record_count": full_report["distillation_record_count"],
        },
        "conversion": {
            "residuals": full_report["conversion"]["residuals"],
            "round_trip_restored": full_report["conversion"]["round_trip_restored"],
            "missing_sidecar_status": missing_status,
            "missing_sidecar_codes": missing_codes,
        },
        "edit_sequence": {
            "all_match": all(step["match"] for step in steps),
            "steps": steps,
        },
        "incremental_reuse": incremental_reuse,
        "forced_interrupt": forced,
        "budget": budget,
        "injection": injection,
        "verification": {
            "full_failures_zero": full_report["failure_count"] == 0,
            "full_pending_zero": full_report["pending_count"] == 0,
            "distillation_expected_24": full_report["distillation_record_count"] == 24,
            "sidecar_missing_detected": E_SIDECAR_MISSING in missing_codes,
            "all_edits_match": all(step["match"] for step in steps),
            "incremental_reuse_zero_recompute": all(
                not layers for layers in incremental_reuse["recomputed"].values()
            ),
            "forced_recovery_matches": forced["recovered_digest_equal_full"],
            "budget_recovery_matches": budget["resume_digest_equal_full"],
            "all_injections_hit": injection["all_hit"],
        },
    }
    atomic_write_json(artifacts / "acceptance_report.json", report)
    return report
