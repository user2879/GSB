from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, fingerprint
from .config import LabConfig
from .conversion import convert_v1_to_v2, convert_v2_to_v1
from .engine import ExperimentLab
from .formats import parse_v1_file, parse_v2_file
from .sample import generate_dataset


def _copy_inputs(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "preference_pairs.jsonl",
        "exchange_v1.txt",
        "exchange_v2.json",
        "config.json",
        "injections.json",
    ):
        source_path = source / name
        if source_path.exists():
            shutil.copy2(source_path, target / name)


def _new_lab(root: Path) -> ExperimentLab:
    return ExperimentLab(
        root,
        state_dir=root / "state",
        output_dir=root / "output",
    )


def _roundtrip_report(config: LabConfig, v1_path: Path, v2_path: Path) -> dict[str, Any]:
    v1_records = sorted(parse_v1_file(v1_path).records.values(), key=lambda item: item.pair_id)
    v2_records = sorted(parse_v2_file(v2_path, config).records.values(), key=lambda item: item.pair_id)
    up = convert_v1_to_v2(v1_records, config)
    down = convert_v2_to_v1(v2_records, config)
    restored = convert_v1_to_v2(down.records, config, sidecar=down.sidecar)
    original = {record.pair_id: record for record in v2_records}
    mismatches = []
    for record in restored.records:
        expected = original[record.pair_id]
        if (
            record.reward_score != expected.reward_score
            or record.reward_delta != expected.reward_delta
            or record.reward_label != expected.reward_label
            or record.precision_label != expected.precision_label
        ):
            mismatches.append(record.pair_id)
    sidecar_complete = all(
        {
            "score_residual_milli",
            "delta_residual_milli",
            "reward_label",
            "precision_label",
            "v1_content_fingerprint",
            "v2_content_fingerprint",
        }.issubset(entry)
        for entry in down.sidecar["records"].values()
    )
    return {
        "up_record_count": len(up.records),
        "down_record_count": len(down.records),
        "restored_after_downgrade_count": len(restored.restored_pair_ids),
        "restore_mismatch_pair_ids": sorted(mismatches),
        "sidecar_all_discarded_fields_recorded": sidecar_complete,
        "down_losses": {
            key: value
            for key, value in down.losses.items()
            if key != "rows"
        },
        "up_losses": {
            key: value
            for key, value in up.losses.items()
            if key != "rows"
        },
    }


def run_verification(
    work_root: str | Path,
    *,
    pair_count: int = 120,
    seed: int = 20260101,
) -> dict[str, Any]:
    root = Path(work_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    sample_dir = root / "sample_data"
    generate_dataset(sample_dir, pair_count=pair_count, seed=seed)
    incremental_dir = root / "incremental"
    _copy_inputs(sample_dir, incremental_dir)
    incremental_lab = _new_lab(incremental_dir)
    initial_report = incremental_lab.run(resume=True)
    sample_report_path = root / "sample_injection_report.json"
    atomic_write_json(sample_report_path, initial_report)

    edits = [
        {"op": "force_strategy_tier", "tier": "large"},
        {"op": "switch_version", "version": "v1"},
        {
            "op": "replace_pair",
            "pair_id": "pair-000010",
            "changes": {
                "prompt": "amber anchor atlas",
                "chosen": "cedar circle cobalt",
                "rejected": "delta drift ember",
                "annotator_id": "ann-04",
                "timestamp": "2026-02-01T12:00:00Z",
                "v1_score_milli": 777,
                "v1_delta_milli": 42,
                "v2_score": 0.7774,
                "v2_delta": 0.0422,
                "reward_label": "preference",
                "precision_label": "float64",
            },
        },
        {"op": "delete_pair", "pair_id": "pair-000011"},
        {
            "op": "set_weights",
            "weights": {
                "prompt": 0.21,
                "chosen": 1.13,
                "rejected": 0.97,
                "length": 0.04,
                "bias": -0.01,
            },
        },
        {"op": "set_threshold", "value": 0.07},
        {"op": "force_strategy_tier", "tier": "small"},
        {"op": "switch_version", "version": "v2"},
        {"op": "force_strategy_tier", "tier": None},
    ]
    steps: list[dict[str, Any]] = []
    all_match = True
    for index, edit in enumerate(edits, 1):
        incremental_lab.apply_edit(edit)
        incremental_report = incremental_lab.run()
        full_dir = root / f"full_after_{index:02d}"
        _copy_inputs(incremental_dir, full_dir)
        full_report = _new_lab(full_dir).run()
        matched = (
            incremental_report["deterministic_core_fingerprint"]
            == full_report["deterministic_core_fingerprint"]
        )
        all_match = all_match and matched
        steps.append(
            {
                "step": index,
                "edit": edit,
                "matched": matched,
                "incremental_core_fingerprint": incremental_report[
                    "deterministic_core_fingerprint"
                ],
                "full_core_fingerprint": full_report["deterministic_core_fingerprint"],
                "invalidation": incremental_report["invalidation"],
                "snapshot_global_version": incremental_report["snapshot"]["global_version"],
            }
        )

    roundtrip = _roundtrip_report(
        LabConfig.load(incremental_dir / "config.json"),
        incremental_dir / "exchange_v1.txt",
        incremental_dir / "exchange_v2.json",
    )

    serial_dir = root / "concurrency_serial"
    parallel_dir = root / "concurrency_parallel"
    _copy_inputs(sample_dir, serial_dir)
    _copy_inputs(sample_dir, parallel_dir)
    serial_lab = _new_lab(serial_dir)
    parallel_lab = _new_lab(parallel_dir)
    serial_lab.update_config({"concurrency": 1})
    parallel_lab.update_config({"concurrency": 8})
    serial_report = serial_lab.run()
    parallel_report = parallel_lab.run()
    concurrency_match = (
        serial_report["deterministic_core_fingerprint"]
        == parallel_report["deterministic_core_fingerprint"]
    )

    crash_dir = root / "crash_recovery"
    _copy_inputs(sample_dir, crash_dir)
    crash_lab = _new_lab(crash_dir)
    before_crash = crash_lab.run()
    crash_detected = False
    try:
        crash_lab.run(crash_after_stage="features")
    except Exception as exc:
        crash_detected = type(exc).__name__ == "SimulatedTermination"
    recovered = crash_lab.run(resume=True)
    recovery_ok = (
        recovered["recovery_point"] is not None
        and recovered["recovery_point"]["completed_round_cursor"]
        == before_crash["snapshot"]["completed_round_cursor"]
    )

    report = {
        "schema_version": 1,
        "all_incremental_full_matches": all_match,
        "steps": steps,
        "conversion_roundtrip": roundtrip,
        "concurrency_serial_parallel_match": concurrency_match,
        "crash_simulation": {
            "termination_observed": crash_detected,
            "recovered_from_complete_snapshot": recovery_ok,
            "recovery_point": recovered["recovery_point"],
        },
        "sample_injection_report": str(sample_report_path),
        "sample_initial_fingerprint": initial_report["deterministic_core_fingerprint"],
        "final_summary": {
            "records": len(incremental_report["records"]),
            "boundary_samples": len(incremental_report["boundary_samples"]),
            "failure_business_codes": incremental_report["failure_business_codes"],
            "injection_business_codes": incremental_report["injection_business_codes"],
            "conversion_v1_to_v2_losses": {
                key: value
                for key, value in incremental_report["conversion"]["v1_to_v2"]["losses"].items()
                if key != "rows"
            },
            "conversion_v2_to_v1_losses": {
                key: value
                for key, value in incremental_report["conversion"]["v2_to_v1"]["losses"].items()
                if key != "rows"
            },
        },
        "verification_fingerprint": "",
    }
    report["verification_fingerprint"] = fingerprint(
        {key: value for key, value in report.items() if key != "verification_fingerprint"},
        prefix="verification:",
    )
    atomic_write_json(root / "verification_report.json", report)
    return report
