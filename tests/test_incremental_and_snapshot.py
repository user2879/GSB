from __future__ import annotations

from pathlib import Path
import json

import pytest

from pref_reward_lab.engine import ExperimentLab
from pref_reward_lab.errors import SnapshotError
from pref_reward_lab.sample import generate_dataset


def make_lab(tmp_path: Path, pairs: int = 30) -> ExperimentLab:
    data = tmp_path / "data"
    generate_dataset(data, pair_count=pairs, seed=99, include_invalid_rows=False)
    injections = data / "injections.json"
    if injections.exists():
        injections.unlink()
    return ExperimentLab(
        data,
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "output",
    )


def test_weight_change_reuses_features_and_recomputes_scores(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.update_config({"forced_strategy_tier": "medium"})
    first = lab.run()
    assert first["invalidation"]["reused_counts"]["features"] == 0

    lab.apply_edit(
        {
            "op": "set_weights",
            "weights": {
                "prompt": 0.2,
                "chosen": 1.2,
                "rejected": 0.8,
                "length": 0.03,
                "bias": 0.01,
            },
        }
    )
    second = lab.run()
    assert second["invalidation"]["reused_counts"]["features"] == 30
    assert second["invalidation"]["recomputed_counts"]["scoring"] == 30
    assert second["invalidation"]["reused_counts"]["conversion"] == 60
    assert second["snapshot"]["global_version"] == first["snapshot"]["global_version"] + 1


def test_replace_pair_only_invalidates_affected_feature_and_score(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.update_config({"forced_strategy_tier": "medium"})
    lab.run()
    lab.apply_edit(
        {
            "op": "replace_pair",
            "pair_id": "pair-000010",
            "changes": {
                "prompt": "prism quiet raven",
                "chosen": "silver sky solstice",
                "rejected": "stone summit thistle",
                "v1_score_milli": 711,
                "v1_delta_milli": 44,
                "v2_score": 0.7111,
                "v2_delta": 0.0439,
            },
        }
    )
    report = lab.run()
    assert report["invalidation"]["recomputed_counts"]["features"] == 1
    assert report["invalidation"]["recomputed_counts"]["scoring"] >= 1
    assert report["invalidation"]["reused_counts"]["features"] == 29
    assert "pair-000010" in report["invalidation"]["changed_pair_ids"]


def test_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.run()
    lab.update_config({"reward_delta_threshold": 0.08})
    with pytest.raises(SnapshotError) as excinfo:
        lab.run(resume=True)
    assert excinfo.value.code == "E_CONFIG_MISMATCH"
    assert "config_hash" in excinfo.value.details["reasons"][0]


def test_crash_temp_is_ignored_and_previous_snapshot_is_recovered(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    first = lab.run()
    with pytest.raises(Exception) as excinfo:
        lab.run(crash_after_stage="scoring")
    assert type(excinfo.value).__name__ == "SimulatedTermination"
    assert any(
        path.name.startswith(".tmp-crash-")
        for path in (tmp_path / "state" / "generations").iterdir()
    )
    recovered = lab.run(resume=True)
    assert recovered["recovery_point"]["completed_round_cursor"] == first["snapshot"][
        "completed_round_cursor"
    ]
    assert recovered["snapshot"]["global_version"] == first["snapshot"]["global_version"] + 1


def test_resume_rejects_tampered_stage_file(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    lab.run()
    manifest = json.loads((tmp_path / "state" / "manifest.json").read_text())
    features_path = (
        tmp_path / "state" / "generations" / manifest["generation"] / "features.json"
    )
    features_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(SnapshotError) as excinfo:
        lab.run(resume=True)
    assert excinfo.value.code == "E_SNAPSHOT_INCOMPLETE"


def test_orphan_generation_from_crash_before_manifest_is_not_reused(tmp_path: Path) -> None:
    lab = make_lab(tmp_path)
    first = lab.run()
    manifest = json.loads((tmp_path / "state" / "manifest.json").read_text())
    next_name = f"gen-{first['snapshot']['global_version'] + 1:08d}-{manifest['config_hash'][:12]}"
    orphan = tmp_path / "state" / "generations" / next_name
    orphan.mkdir()
    (orphan / "features.json").write_text('{"orphan":true}\n', encoding="utf-8")
    second = lab.run(resume=True)
    assert second["snapshot"]["global_version"] == first["snapshot"]["global_version"] + 1
    assert second["snapshot"]["generation"].endswith("-r01")
