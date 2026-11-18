from lora_bench.pipeline import run_pipeline


def test_sample_edit_invalidates_only_owner_task(
    input_dir, workspace, base_config, edit_tools
):
    first = run_pipeline(input_dir, workspace, base_config)
    tasks = first["manifest"]["completed_tasks"]
    victim = "cls_mod4"
    sample_id = f"{victim}-000"
    edit_tools["update_sample"](
        sample_id, lambda sample: sample["input_tokens"].append(3)
    )
    second = run_pipeline(input_dir, workspace, base_config)
    invalidated = [e["task"] for e in second["invalidation"]["invalidated"]]
    assert invalidated == [victim]
    assert second["trained_tasks"] == [victim]
    assert sorted(second["reused_tasks"]) == sorted(t for t in tasks if t != victim)
    reasons = {
        e["task"]: e["reasons"]
        for e in second["invalidation"]["invalidated"]
    }
    assert reasons[victim] == ["task_samples"]


def test_rank_change_invalidates_single_task(
    input_dir, workspace, base_config, edit_tools
):
    run_pipeline(input_dir, workspace, base_config)
    edit_tools["update_manifest"](
        lambda manifest: None
    )
    config = dict(base_config)
    config["task_overrides"] = {"cls_parity": {"rank": 4, "alpha": 4.0}}
    outcome = run_pipeline(input_dir, workspace, config)
    invalidated = {
        e["task"]: e["reasons"]
        for e in outcome["invalidation"]["invalidated"]
    }
    assert list(invalidated) == ["cls_parity"]
    assert invalidated["cls_parity"] == ["rank"]
    assert outcome["trained_tasks"] == ["cls_parity"]


def test_target_layer_change_invalidates_every_task(
    input_dir, workspace, base_config
):
    run_pipeline(input_dir, workspace, base_config)
    config = dict(base_config, target_layers=[1, 3])
    outcome = run_pipeline(input_dir, workspace, config)
    assert len(outcome["invalidation"]["invalidated"]) == 4
    assert all(
        e["reasons"] == ["target_layers"]
        for e in outcome["invalidation"]["invalidated"]
    )
    assert sorted(outcome["trained_tasks"]) == sorted(
        outcome["manifest"]["completed_tasks"]
    )


def test_base_replacement_invalidates_every_task(
    input_dir, workspace, base_config, edit_tools
):
    run_pipeline(input_dir, workspace, base_config)
    # Promote base_v2 to an active pack and point the run at it.
    edit_tools["replace_base"]("v2")
    outcome = run_pipeline(
        input_dir, workspace, base_config, base_subdir="base_active"
    )
    reasons = {
        e["task"]: e["reasons"]
        for e in outcome["invalidation"]["invalidated"]
    }
    assert sorted(reasons) == sorted(outcome["manifest"]["completed_tasks"])
    assert all(value == ["base_version"] for value in reasons.values())


def test_combined_edit_reasons_are_priority_ordered(
    input_dir, workspace, base_config, edit_tools
):
    run_pipeline(input_dir, workspace, base_config)
    edit_tools["update_sample"](
        "reg_sum-000", lambda sample: sample["input_tokens"].append(7)
    )
    edit_tools["replace_base"]("v2")
    config = dict(base_config, target_layers=[0, 1])
    outcome = run_pipeline(
        input_dir, workspace, config, base_subdir="base_active"
    )
    target = next(
        e
        for e in outcome["invalidation"]["invalidated"]
        if e["task"] == "reg_sum"
    )
    assert target["reasons"][0] == "base_version"
    assert "target_layers" in target["reasons"]
    assert "task_samples" in target["reasons"]
