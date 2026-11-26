from vertical_workbench.clock import ManualClock
from vertical_workbench.config import WorkbenchConfig
from vertical_workbench.pipeline import assign_reading_and_conflicts
from vertical_workbench.studio import Studio


def test_cross_column_wrong_order_and_missing_conflicts_are_recorded():
    truth = {
        "keypoints": [
            {"keypoint_id": "t1", "x": 440, "y": 200, "column_rank": 0, "reading_order": 1},
            {"keypoint_id": "t2", "x": 440, "y": 100, "column_rank": 0, "reading_order": 2},
            {"keypoint_id": "t3", "x": 365, "y": 100, "column_rank": 1, "reading_order": 3},
            {"keypoint_id": "t4", "x": 365, "y": 170, "column_rank": 1, "reading_order": 4},
        ]
    }
    keypoints = [
        {"keypoint_id": "p-cross", "x": 365, "y": 100, "column_rank": 0, "confidence_bp": 9000},
        {"keypoint_id": "p-order", "x": 440, "y": 110, "column_rank": 0, "confidence_bp": 9000},
        {"keypoint_id": "p-late", "x": 440, "y": 205, "column_rank": 0, "confidence_bp": 9000},
    ]
    result = assign_reading_and_conflicts("synthetic", keypoints, truth)
    types = {conflict["type"] for conflict in result["conflicts"]}
    assert "跨栏" in types
    assert "错序" in types
    assert "缺号" in types
    assert all(conflict["conflict_id"] for conflict in result["conflicts"])


def test_annotation_replacement_invalidates_only_keypoint_and_reading(tmp_path):
    config = WorkbenchConfig(sample_count=2, stride=64, global_point_budget=160, max_active_leases=16)
    studio = Studio.create(tmp_path, config, ManualClock(909))
    studio.engine.run(1)
    studio.apply({"op": "replace_annotation_version", "version_id": "kv-relabel"})
    relabel_stages = {}
    for item in studio.engine.last_result["reuse"]:
        relabel_stages[item["stage"]] = relabel_stages.get(item["stage"], 0) + 1
    assert relabel_stages["window"] > 0
    assert relabel_stages["connected_component"] > 0
    assert relabel_stages["keypoint"] > 0
    assert relabel_stages["reading_order"] == 2

    result = studio.apply(
        {
            "op": "replace_annotation_version",
            "version_id": "kv-content",
            "jitter_by_image": {"sample_001": 1},
        }
    )
    recompute_targets = {
        item["target"]
        for item in result["recompute"]
        if item["stage"] in {"keypoint", "reading_order"}
    }
    assert "sample_001" in recompute_targets
    assert any(target.startswith("sample_001:") for target in recompute_targets)
    assert "sample_002" not in recompute_targets
    assert not any(item["stage"] == "window" for item in result["recompute"])


def test_lease_expiry_generation_change_is_deterministic_in_replay(tmp_path):
    config = WorkbenchConfig(sample_count=1, stride=64, lease_ttl_ms=100, max_active_leases=8)
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    studio_a = Studio.create(root_a, config, ManualClock(1000))
    studio_b = Studio.create(root_b, config, ManualClock(1000))
    result_a = studio_a.engine.run(1)
    result_b = studio_b.engine.run(1)
    assert result_a["lease_snapshot"] == result_b["lease_snapshot"]

    studio_a.clock.advance(150)
    studio_b.clock.advance(150)
    command = {"op": "update_config", "patch": {"stride": 32}}
    changed_a = studio_a.apply(command)
    changed_b = studio_b.apply(command)
    from vertical_workbench.hashing import canonical_json

    assert canonical_json(changed_a) == canonical_json(changed_b)
    assert changed_a["business_code_counts"]["E_LEASE_EXPIRED"] >= 3
    assert any(item["basis"] == "LEASE_GENERATION_CHANGED" for item in changed_a["invalidations"])
