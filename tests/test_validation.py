from __future__ import annotations

from pathlib import Path

from layout_workbench.commands import load_command_stream
from layout_workbench.config import RecognitionConfig
from layout_workbench.validation import validate_command_sequence


def test_command_sequence_incremental_equals_full(sample_data: Path, tmp_path: Path) -> None:
    config = RecognitionConfig(snapshot_commit_interval_commands=1)
    commands = [
        {
            "id": "threshold",
            "type": "set_threshold",
            "payload": {"value": 168},
        },
        {
            "id": "metric",
            "type": "set_match_metric",
            "payload": {"value": "ink_balance"},
        },
        {
            "id": "replace",
            "type": "replace_image",
            "payload": {
                "name": "sample_b",
                "source_png": str(sample_data / "replacements" / "sample_b_v2.png"),
                "source_truth": str(sample_data / "replacements" / "sample_b_v2.truth.json"),
            },
        },
        {
            "id": "delete",
            "type": "delete_image",
            "payload": {"name": "sample_c"},
        },
    ]
    report = validate_command_sequence(
        data_dir=sample_data,
        config=config,
        commands=commands,
        base_dir=sample_data.parent,
        work_dir=tmp_path / "validation",
    )
    assert report["passed"] is True
    assert report["first_overall_divergence"] is None
    assert report["semantic_equivalence"]["all_steps_equal"] is True
    assert report["invalidation"]["all_steps_match_dependency_plan"] is True
    assert report["snapshot_verification"]["valid"] is True
    assert report["snapshot_restore"]["semantic_equal"] is True
    assert report["concurrency_equivalence"]["semantic_equal"] is True
    assert report["final_snapshot_version"] == len(commands) + 1


def test_injected_fault_reports_hit_and_first_divergence(sample_data: Path, tmp_path: Path) -> None:
    config = RecognitionConfig(
        fault_injections=(
            {
                "id": "F-ORDER-01",
                "type": "reverse_character_order",
                "image": "sample_a",
            },
        )
    )
    report = validate_command_sequence(
        data_dir=sample_data,
        config=config,
        commands=[],
        base_dir=sample_data.parent,
        work_dir=tmp_path / "fault",
    )
    assert report["passed"] is False
    divergence = report["first_overall_divergence"]
    assert divergence is not None
    assert divergence["image"] == "sample_a"
    assert divergence["stage"] in {"recognition", "line_aggregation", "segmentation"}
    incremental_path = Path(report["steps"][0]["incremental_report"])
    import json

    incremental = json.loads(incremental_path.read_text(encoding="utf-8"))
    assert any(hit["id"] == "F-ORDER-01" for hit in incremental["fault_hits"])

