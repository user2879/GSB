from __future__ import annotations

import hashlib
from collections import Counter

from aedwb.samplegen import generate_sample


def test_full_run_calibrates_maps_realigns_and_diagnoses(base_run):
    report = base_run["report"]

    assert report["failures"] == []
    assert len(report["packets"]) == 21
    assert all(packet["valid"] for packet in report["packets"])
    assert report["global_frames"]
    assert report["boundary_samples"]
    assert report["drift_alerts"]
    assert report["truth_evaluation"]["frame_count"] == len(report["global_frames"])

    tolerance = report["calibration"]["parameters"]["tolerance"]
    for packet in report["packets"]:
        for frame in packet["calibration"]["frames"]:
            if frame["unmapped"]:
                assert frame["calibrated_probability"] == [0.0, 0.0, 0.0, 0.0]
            else:
                assert abs(frame["probability_sum"] - 1.0) <= tolerance

    legacy = next(packet for packet in report["packets"] if packet["packet_id"] == "a0_base_legacy")
    assert legacy["raw_score_type"] == "logit"
    assert legacy["raw_scores"]
    assert all(isinstance(value, int) for row in legacy["raw_scores"] for value in row)
    assert legacy["mapping"]["mapping_by_name"] is True
    assert {entry["code"]: entry["name"] for entry in legacy["mapping"]["entries"]} == {
        0: "music",
        1: "noise",
        2: "alarm",
        3: "speech",
    }

    fine = next(packet for packet in report["packets"] if packet["packet_id"] == "a0_fine_legacy")
    assert fine["alignment"]["resampled"] is True
    assert fine["alignment"]["original_frame_count"] == 150
    assert fine["alignment"]["target_frame_count"] == 75
    for frame in fine["calibration"]["frames"]:
        assert "global_frame_index" in frame
        assert frame["original_frame_indices"]
        assert max(frame["original_frame_indices"]) < fine["alignment"]["original_frame_count"]

    coarse = next(packet for packet in report["packets"] if packet["packet_id"] == "a0_coarse_new")
    assert coarse["alignment"]["resampled"] is True
    assert coarse["alignment"]["max_residual"] > 0.0

    sources = Counter(alert["source"] for alert in report["drift_alerts"])
    assert sources["codebook_mapping"] > 0
    assert sources["frame_resampling"] > 0
    assert sources["logit_conversion"] > 0
    assert sources["confidence_shift"] > 0

    logit_alert = next(alert for alert in report["drift_alerts"] if alert["source"] == "logit_conversion")
    assert logit_alert["source_scores"]["logit_conversion"] > 0.0
    assert "confidence_shift" in logit_alert["source_scores"]


def test_stage_snapshots_and_manifest_are_complete(base_run):
    state = base_run["state"]
    for stage in ("parse", "map", "align", "calibrate", "decide", "drift"):
        stage_dir = state / "snapshots" / stage
        assert (stage_dir / "manifest.json").is_file()
        assert list(stage_dir.glob("chunk-*.json"))

    import json

    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["global_version"]
    assert manifest["config_hash"]
    assert manifest["calibration_fingerprint"]
    assert manifest["packet_header_versions"]
    assert len(manifest["completed_packet_ids"]) == 21
    assert set(manifest["stage_snapshots"]) == {
        "parse",
        "map",
        "align",
        "calibrate",
        "decide",
        "drift",
    }


def test_sample_generation_is_byte_identical(tmp_path):
    first = generate_sample(tmp_path / "first", injected=True, clean=False)
    second = generate_sample(tmp_path / "second", injected=True, clean=False)
    first_files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    for relative in first_files:
        assert hashlib.sha256((first / relative).read_bytes()).digest() == hashlib.sha256(
            (second / relative).read_bytes()
        ).digest()
