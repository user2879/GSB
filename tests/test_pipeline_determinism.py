import json
import random


from vertical_workbench.clock import ManualClock
from vertical_workbench.config import WorkbenchConfig
from vertical_workbench.engine import WorkbenchEngine
from vertical_workbench.hashing import canonical_json
from vertical_workbench.studio import Studio
from vertical_workbench.vision import compute_window


def small_config(**patch):
    values = dict(sample_count=2, stride=64, global_point_budget=160, max_active_leases=16)
    values.update(patch)
    return WorkbenchConfig(**values)


def test_baseline_detects_all_truth_points(tmp_path):
    Studio.create(tmp_path, small_config(), ManualClock(1000))
    engine = WorkbenchEngine(tmp_path, ManualClock(2000), write_snapshots=False)
    result = engine.run(1)
    assert not result["failures"]
    truth_count = 0
    for truth_path in engine.root.joinpath("truth").glob("*.json"):
        truth_count += len(json.loads(truth_path.read_text(encoding="utf-8"))["keypoints"])
    assert len(result["keypoints"]) == truth_count
    assert not result["conflicts"]

    second = engine.run(2)
    assert len(second["reuse"]) > len(second["recompute"])
    assert any(item["stage"] == "window" for item in second["reuse"])


def test_two_blank_replays_match_after_edits(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    studio_a = Studio.create(root_a, small_config(), ManualClock(1000))
    studio_b = Studio.create(root_b, small_config(), ManualClock(1000))
    studio_a.engine.run(1)
    studio_b.engine.run(1)
    commands = [
        {"op": "update_config", "patch": {"stride": 32}},
        {"op": "replace_annotation_version", "version_id": "kv-relabel"},
        {"op": "replace_annotation_version", "version_id": "kv-edit", "jitter_by_image": {"sample_001": 1}},
        {"op": "set_budget", "patch": {"global_point_budget": 45}},
        {"op": "set_budget", "patch": {"global_point_budget": 160}},
    ]
    for command in commands:
        result_a = studio_a.apply(command)
        result_b = studio_b.apply(command)
        assert canonical_json(result_a) == canonical_json(result_b)


def test_worker_counts_produce_same_core_fields(tmp_path):
    Studio.create(tmp_path, small_config(max_workers=1))
    signatures = []
    for workers in (1, 2, 4):
        raw = Studio(tmp_path).read_config_raw()
        raw["max_workers"] = workers
        Studio(tmp_path).write_config_raw(raw)
        engine = WorkbenchEngine(tmp_path, ManualClock(3000 + workers), write_snapshots=False)
        result = engine.run(workers)
        signatures.append(
            canonical_json(
                {
                    "kp": [(p["keypoint_id"], p["x"], p["y"], p["reading_order"]) for p in result["keypoints"]],
                    "conflicts": result["conflicts"],
                    "keys": [p["cache_key"] for p in result["keypoints"]],
                }
            )
        )
    assert signatures[0] == signatures[1] == signatures[2]


def test_score_and_area_equality_are_flagged(tmp_path):
    config = small_config()
    Studio.create(tmp_path, config, ManualClock(1))
    engine = WorkbenchEngine(tmp_path, ManualClock(2), write_snapshots=False)
    loaded = __import__("vertical_workbench.loader", fromlist=["load_workspace"]).load_workspace(tmp_path)
    sample = loaded.samples[0]
    rect = {"x": 0, "y": 0, "w": 64, "h": 64}
    window = compute_window(sample["image_id"], sample["pixels"], rect, config)
    equal_config = small_config(candidate_score_threshold=window["score"])
    equal_window = compute_window(sample["image_id"], sample["pixels"], rect, equal_config)
    assert equal_window["score_boundary_equal"]
    assert equal_window["candidate"] is (window["ink_count"] >= config.min_component_area)

    from vertical_workbench.vision import merge_components

    component = {
        "component_id": "raw-001",
        "coords": [(0, 0)] * 12,
        "area": 12,
        "x_min": 0,
        "x_max": 4,
        "y_min": 0,
        "y_max": 4,
        "sum_x": 0,
        "sum_y": 0,
    }
    domains, _records, comparisons = merge_components([component], [], small_config(area_boundary_threshold=12))
    assert comparisons == 0
    assert domains[0]["area_boundary_equal"] is True


def test_random_command_sequences_replay_identically(tmp_path):
    for seed in (101, 202, 303):
        root_a = tmp_path / f"a-{seed}"
        root_b = tmp_path / f"b-{seed}"
        config = small_config(sample_count=2, stride=64, max_active_leases=32)
        studio_a = Studio.create(root_a, config, ManualClock(1000 + seed))
        studio_b = Studio.create(root_b, config, ManualClock(1000 + seed))
        studio_a.engine.run(1)
        studio_b.engine.run(1)
        rng = random.Random(seed)
        image_ids = ["sample_001", "sample_002"]

        for step in range(7):
            choice = rng.randrange(8)
            if choice == 0:
                size = rng.choice([32, 64])
                command = {
                    "op": "update_config",
                    "patch": {
                        "window_size": size,
                        "stride": rng.choice([16, 32] if size == 32 else [32, 64]),
                    },
                }
            elif choice == 1:
                command = {
                    "op": "update_config",
                    "patch": {
                        "quant_gear": rng.choice(["coarse", "normal", "fine"]),
                        "fixed_frac_bits": rng.choice([4, 8, 16]),
                    },
                }
            elif choice == 2:
                command = {
                    "op": "replace_annotation_version",
                    "version_id": f"kv-{seed}-{step}",
                    "jitter_by_image": {rng.choice(image_ids): 1} if step % 2 else None,
                }
            elif choice == 3:
                image_id = rng.choice(image_ids)
                columns = rng.randint(3, 4)
                command = {
                    "op": "replace_image",
                    "image_id": image_id,
                    "columns": columns,
                    "counts": [rng.randint(4, 7) for _ in range(columns)],
                    "seed": rng.randint(1000, 9000),
                }
            elif choice == 4 and len(image_ids) < 4:
                image_id = f"sample_{len(image_ids) + 1:03d}"
                image_ids.append(image_id)
                command = {
                    "op": "add_image",
                    "image_id": image_id,
                    "columns": 3,
                    "counts": [5, 6, 7],
                    "seed": rng.randint(1000, 9000),
                }
            elif choice == 5 and len(image_ids) > 1:
                image_id = image_ids.pop()
                command = {"op": "delete_image", "image_id": image_id}
            elif choice == 6:
                command = {
                    "op": "set_budget",
                    "patch": {
                        "global_point_budget": rng.choice([40, 90, 220]),
                        "column_point_quota": rng.choice([5, 8, 12]),
                    },
                }
            else:
                command = {"op": "set_lease_limit", "max_active_leases": rng.choice([6, 16, 32])}

            result_a = studio_a.apply(command)
            result_b = studio_b.apply(command)
            assert canonical_json(result_a) == canonical_json(result_b), (seed, step, command)
