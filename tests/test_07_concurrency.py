from lora_bench.pipeline import run_pipeline
from lora_bench.utils import read_json
import os


def test_concurrency_levels_produce_identical_outputs(
    input_dir, tmp_path, base_config
):
    serial_config = dict(base_config, concurrency=1)
    parallel_config = dict(base_config, concurrency=4)
    serial = run_pipeline(
        input_dir, str(tmp_path / "serial"), serial_config
    )
    parallel = run_pipeline(
        input_dir, str(tmp_path / "parallel"), parallel_config
    )
    assert [item["task"] for item in serial["validation_summary"]["entries"]] == [
        item["task"] for item in parallel["validation_summary"]["entries"]
    ]
    for task in serial["manifest"]["completed_tasks"]:
        left = read_json(
            os.path.join(
                str(tmp_path / "serial"), "current", "train", f"{task}.json"
            )
        )
        right = read_json(
            os.path.join(
                str(tmp_path / "parallel"),
                "current",
                "train",
                f"{task}.json",
            )
        )
        assert left == right
    assert serial["invalidation"] == parallel["invalidation"]


def test_stable_sort_keys_govern_output_order(input_dir, tmp_path, base_config):
    config = dict(base_config, target_layers=[3, 1, 2], concurrency=3)
    outcome = run_pipeline(input_dir, str(tmp_path / "ws"), config)
    tasks = [entry["task"] for entry in outcome["validation_summary"]["entries"]]
    assert tasks == sorted(tasks)
    sort_keys = [
        entry["sort_key"] for entry in outcome["validation_summary"]["entries"]
    ]
    assert sort_keys == sorted(sort_keys)
    # Whitelist is normalised into ascending layer order inside the sort key.
    assert all("layers=1,2,3" in key for key in sort_keys)


def test_repeated_parallel_runs_are_deterministic(input_dir, tmp_path, base_config):
    config = dict(base_config, concurrency=4)
    first = run_pipeline(input_dir, str(tmp_path / "a"), config)
    second = run_pipeline(input_dir, str(tmp_path / "b"), config)
    left = {
        item["task"]: item
        for item in first["validation_summary"]["entries"]
    }
    right = {
        item["task"]: item
        for item in second["validation_summary"]["entries"]
    }
    assert left == right
