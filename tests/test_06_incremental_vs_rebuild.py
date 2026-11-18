import os

from lora_bench.pipeline import run_pipeline
from lora_bench.rebuild import compare_outcomes, full_rebuild


def _assert_equal(incremental, rebuild, baseline):
    # Recompute verdicts against the same baseline so lists match field by field.
    from lora_bench.deps import plan_invalidation

    rebuild["invalidation"] = plan_invalidation(
        baseline, rebuild["manifest"]["task_identities"]
    )
    comparison = compare_outcomes(incremental, rebuild)
    assert comparison["equal"], comparison["mismatches"]


def _verdict_baseline(incremental, previous_identities):
    """Identities used for the verdict of the current step.

    Each incremental step compares against the identities stored in the
    previous committed snapshot; the from-scratch rebuild must use the same
    predecessor baseline for its verdict to match field by field.
    """
    return previous_identities


def test_edit_sequence_matches_full_rebuild(
    input_dir, workspace, tmp_path, base_config, edit_tools
):
    # Baseline run establishes the starting dependency identities.
    baseline_run = run_pipeline(input_dir, workspace, base_config)
    baseline_identities = baseline_run["manifest"]["task_identities"]

    # Edit 1: modify one sample of one task.
    edit_tools["update_sample"](
        "cls_mod4-001", lambda sample: sample["input_tokens"].append(5)
    )
    config_step2 = dict(base_config)
    incremental = run_pipeline(input_dir, workspace, config_step2)
    step1_baseline = dict(baseline_identities)
    rebuild = full_rebuild(
        input_dir,
        str(tmp_path / "rebuild1"),
        config_step2,
        baseline_identities=step1_baseline,
    )
    _assert_equal(incremental, rebuild, step1_baseline)
    step2_identities = dict(incremental["manifest"]["task_identities"])

    # Edit 2: rank/alpha override for one task.
    config_step3 = dict(
        config_step2,
        task_overrides={"reg_cycle": {"rank": 3, "alpha": 7.0}},
    )
    incremental = run_pipeline(input_dir, workspace, config_step3)
    rebuild = full_rebuild(
        input_dir,
        str(tmp_path / "rebuild2"),
        config_step3,
        baseline_identities=step2_identities,
    )
    _assert_equal(incremental, rebuild, step2_identities)
    step3_identities = dict(incremental["manifest"]["task_identities"])

    # Edit 3: target layer whitelist changes (base-level fan-out).
    config_step4 = dict(config_step3, target_layers=[1, 2])
    incremental = run_pipeline(input_dir, workspace, config_step4)
    rebuild = full_rebuild(
        input_dir,
        str(tmp_path / "rebuild3"),
        config_step4,
        baseline_identities=step3_identities,
    )
    _assert_equal(incremental, rebuild, step3_identities)
    step4_identities = dict(incremental["manifest"]["task_identities"])

    # Edit 4: base version replacement (base-level fan-out).
    edit_tools["replace_base"]("v2")
    incremental = run_pipeline(
        input_dir, workspace, config_step4, base_subdir="base_active"
    )
    rebuild = full_rebuild(
        input_dir,
        str(tmp_path / "rebuild4"),
        config_step4,
        baseline_identities=step4_identities,
        base_subdir="base_active",
    )
    _assert_equal(incremental, rebuild, step4_identities)
    step5_identities = dict(incremental["manifest"]["task_identities"])

    # Edit 5: add then delete a sample, ending with a task-scoped change.
    record = {
        "sample_id": "cls_parity-900",
        "task": "cls_parity",
        "input_tokens": [1, 2, 3, 4],
        "target_tokens": [0],
        "label": 1,
    }
    edit_tools["add_sample"](record)
    edit_tools["update_manifest"](
        lambda manifest: manifest["tasks"][1]["train_sample_ids"].append(
            "cls_parity-900"
        )
    )
    edit_tools["delete_sample"]("cls_parity-900")
    incremental = run_pipeline(
        input_dir, workspace, config_step4, base_subdir="base_active"
    )
    rebuild = full_rebuild(
        input_dir,
        str(tmp_path / "rebuild5"),
        config_step4,
        baseline_identities=step5_identities,
        base_subdir="base_active",
    )
    _assert_equal(incremental, rebuild, step5_identities)


def test_failure_list_order_matches_rebuild(
    input_dir, workspace, tmp_path, base_config, edit_tools
):
    run_pipeline(input_dir, workspace, base_config)
    baseline = read_manifest_identities(workspace)
    edit_tools["update_manifest"](
        lambda manifest: manifest["tasks"][0].__setitem__(
            "val_sample_ids", []
        )
    )
    incremental = run_pipeline(input_dir, workspace, base_config)
    rebuild = full_rebuild(
        input_dir,
        str(tmp_path / "rebuild-empty"),
        base_config,
        baseline_identities=baseline,
    )
    assert incremental["failures"] == rebuild["failures"]
    assert incremental["failures"]


def read_manifest_identities(workspace):
    return read_json_safe(
        os.path.join(workspace, "current", "manifest.json")
    )["task_identities"]


def read_json_safe(path):
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
