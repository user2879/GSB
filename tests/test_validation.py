from __future__ import annotations

import json
from pathlib import Path

from motion_pipeline.config import parse_config

from .helpers import sample_config_dict


def test_config_validation_reports_all_business_errors() -> None:
    source = sample_config_dict()
    source["frame_width"] = 0
    source["frame_height"] = -1
    source["prediction_steps"] = 0
    source["motion_model"]["linear_gain"] = float("nan")
    source["noise_layer"]["amplitude"] = float("inf")
    source["fixed_point_bits"] = 0
    source["quantization_levels"] = [3]
    source["precision_levels"] = [7]
    source["scale_tiers"] = [4, 2]
    source["latency_budget_ms"] = 0
    source["thread_pool_workers"] = 0
    source["snapshot_commit_interval_batches"] = 0

    result = parse_config(source)

    assert not result.ok
    assert result.config is None
    assert result.failures
    assert {entry["business_code"] for entry in result.failures} == {"E_CONFIG_INVALID"}
    fields = {entry["field"] for entry in result.failures}
    assert {
        "frame_width",
        "frame_height",
        "prediction_steps",
        "linear_gain",
        "amplitude",
        "fixed_point_bits",
        "quantization_levels[0]",
        "precision_levels[0]",
        "scale_tiers",
        "latency_budget_ms",
        "thread_pool_workers",
        "snapshot_commit_interval_batches",
    } <= fields


def test_sequence_validation_keeps_valid_rows_and_sorts_failures(tmp_path: Path) -> None:
    from motion_pipeline.samples import load_sequences

    valid_a = {
        "sequence_id": "a",
        "frames": [
            {
                "trajectories": [
                    {"shape": "circle", "x": 1, "y": 1, "radius": 1, "color": [1, 2, 3]}
                ],
                "color_blocks": [],
                "noise_seed": 1,
            }
        ],
    }
    valid_b = {
        "sequence_id": "b",
        "frames": [
            {
                "trajectories": [
                    {
                        "shape": "rect",
                        "x": 1,
                        "y": 1,
                        "width": 2,
                        "height": 2,
                        "color": [3, 2, 1],
                    }
                ],
                "color_blocks": [],
                "noise_seed": 2,
            }
        ],
    }
    invalid_z = {"sequence_id": "z", "frames": []}
    missing_id = {"frames": valid_a["frames"]}
    lines = [
        "not valid json",
        json.dumps(missing_id),
        json.dumps(valid_b),
        json.dumps(invalid_z),
        json.dumps(valid_a),
        json.dumps(valid_a),
    ]
    path = tmp_path / "sequences.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = load_sequences(path)

    assert [sequence.sequence_id for sequence in result.sequences] == ["a", "b"]
    assert all(entry["business_code"] == "E_SEQ_INVALID" for entry in result.failures)
    assert [entry["sequence_id"] for entry in result.failures] == [
        "@line:00000001",
        "@line:00000002",
        "a",
        "z",
    ]
    assert result.failures == sorted(
        result.failures,
        key=lambda entry: (
            entry["sequence_id"],
            entry.get("line", 0),
            entry["business_code"],
            entry.get("field", ""),
        ),
    )

