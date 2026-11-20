from __future__ import annotations

import json
import shutil
from pathlib import Path

from aedwb.pipeline import run_workbench
from aedwb.samplegen import generate_sample


def _run(sample: Path, state: Path) -> dict:
    return run_workbench(sample, state, state / "report.json")


def _copy_sample(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _set_workers(sample: Path, workers: int) -> None:
    config_path = sample / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["thread_pool_workers"] = workers
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_serial_and_parallel_results_are_identical(tmp_path: Path):
    serial_sample = generate_sample(tmp_path / "serial", clean=False)
    parallel_sample = _copy_sample(serial_sample, tmp_path / "parallel")
    _set_workers(serial_sample, 1)
    _set_workers(parallel_sample, 8)

    serial = _run(serial_sample, tmp_path / "serial-state")
    parallel = _run(parallel_sample, tmp_path / "parallel-state")

    assert serial["run"]["max_workers"] == 1
    assert parallel["run"]["max_workers"] == 8
    assert serial["consistency_signature"] == parallel["consistency_signature"]
    assert serial["global_frames"] == parallel["global_frames"]
    assert serial["drift_alerts"] == parallel["drift_alerts"]
    assert serial["failures"] == parallel["failures"]
    assert [
        (entry["packet_index"], entry["packet_id"], entry["stage_ledger"])
        for entry in serial["ledger"]["entries"]
    ] == [
        (entry["packet_index"], entry["packet_id"], entry["stage_ledger"])
        for entry in parallel["ledger"]["entries"]
    ]


def test_fault_injections_are_isolated_and_reported(tmp_path: Path):
    sample = tmp_path / "injected"
    state = tmp_path / "state"
    generate_sample(sample, injected=True)
    report = _run(sample, state)

    assert len(report["injection_hits"]) == 5
    assert all(item["applied"] and item["hit"] for item in report["injection_hits"])
    assert all(item["expected_code_observed"] for item in report["injection_hits"])
    assert [item["business_code"] for item in report["injection_hits"]] == [
        "E_PACKET_INVALID",
        "E_PACKET_INVALID",
        "E_PACKET_INVALID",
        "E_PACKET_INVALID",
        "E_CLASS_UNMAPPED",
    ]
    failure_codes = [item["code"] for item in report["failures"]]
    assert failure_codes.count("E_PACKET_INVALID") == 4
    assert failure_codes.count("E_CLASS_UNMAPPED") == 1
    assert failure_codes.count("E_CALIB_RESIDUAL") == 1
    assert sum(packet["valid"] for packet in report["packets"]) == 22
    assert len(report["global_frames"]) == 750
    assert report["global_frames"]

    unmapped = next(packet for packet in report["packets"] if packet["packet_id"] == "a0_probe_unmapped")
    assert unmapped["mapping"]["unmapped_frame_count"] == 25
    assert all(frame["calibrated_probability"] == [0.0, 0.0, 0.0, 0.0] for frame in unmapped["calibration"]["frames"])
    assert all(
        frame["original_codes"] == [0, 1, 2, 3]
        for frame in unmapped["mapping"]["unmapped_frames"]
    )

    signature = report["consistency_signature"]
    failure_order = [item["sort_key"] for item in report["failures"]]
    second = _run(sample, state)
    assert second["consistency_signature"] == signature
    assert [item["sort_key"] for item in second["failures"]] == failure_order
    assert second["ledger"]["cache_summary"]["computed_stage_count"] == 0
