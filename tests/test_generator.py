from __future__ import annotations

import json
import wave
from pathlib import Path

from fingerprint_workbench.generator import GeneratorConfig, generate_materials


def test_generator_is_byte_deterministic(generated: dict, tmp_path: Path) -> None:
    second = generate_materials(tmp_path / "second", GeneratorConfig(seed=20261119, track_count=7))
    first_root = Path(generated["root"])
    second_root = Path(second["root"])
    first_manifest = (first_root / "truth_manifest.json").read_bytes()
    second_manifest = (second_root / "truth_manifest.json").read_bytes()
    assert first_manifest == second_manifest
    for record in generated["manifest"]["tracks"]:
        assert (first_root / record["file"]).read_bytes() == (second_root / record["file"]).read_bytes()
    for record in generated["manifest"]["queries"]:
        assert (first_root / record["file"]).read_bytes() == (second_root / record["file"]).read_bytes()


def test_ground_truth_ranges_and_wav_format(generated: dict) -> None:
    manifest = generated["manifest"]
    root = Path(generated["root"])
    assert manifest["sample_rate"] == 44_100
    assert manifest["channels"] == 1
    assert manifest["bit_depth"] == 16
    lengths = {}
    for record in manifest["tracks"]:
        path = root / record["file"]
        with wave.open(str(path), "rb") as reader:
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            assert reader.getframerate() == 44_100
            assert reader.getnframes() == record["duration_samples"]
        lengths[record["track_id"]] = record["duration_samples"]
        for start, end in record["silence_intervals_samples"]:
            assert 0 <= start < end <= record["duration_samples"]
    kinds = set()
    for variant in manifest["variants"]:
        kinds.add(variant["kind"])
        assert 0 <= variant["source_start_sample"] < variant["source_end_sample"]
        assert variant["source_end_sample"] <= lengths[variant["source_track"]]
        assert 0 <= variant["destination_start_sample"] < variant["destination_end_sample"]
        assert variant["destination_end_sample"] <= lengths[variant["destination_track"]]
    assert {"exact_repeat", "pitch_shift", "time_stretch"} <= kinds
    query = manifest["queries"][0]
    assert query["expected_end_sample"] <= lengths[query["expected_track_id"]]
    parsed = json.loads((root / "truth_manifest.json").read_text(encoding="utf-8"))
    assert parsed["sample_range_convention"] == "half_open_[start,end)"
