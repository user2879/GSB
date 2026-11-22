import json
import os
import shutil

import numpy as np
import yaml

from peft_fp import errors
from peft_fp.pipeline import apply_upgrade_step, run_pipeline
from tests.conftest import strip_version_fields


def test_single_layer_upgrade_invalidates_only_referencing_tasks(sample_dir, tmp_path):
    work = str(tmp_path / "samples")
    shutil.copytree(sample_dir, work)
    store = str(tmp_path / "store")

    r1 = run_pipeline(work, store)
    assert all(t["disposition"] == "trained" for t in r1["tasks"])

    applied = apply_upgrade_step(work, 0)
    assert applied == ["encoder.layer0.mlp"]

    r2 = run_pipeline(work, store)
    disp = {t["name"]: t["disposition"] for t in r2["tasks"]}
    # task_alpha / task_beta target the replaced layer -> retrained
    assert disp["task_alpha"] == "retrained"
    assert disp["task_beta"] == "retrained"
    # the rest reuse their exported adapters without retraining
    assert disp["task_gamma"] == "reused"
    assert disp["task_delta"] == "reused"
    mismatch = [f for f in r2["failures"] if f["code"] == errors.E_BASE_VERSION_MISMATCH]
    assert {f["task"] for f in mismatch} == {"task_alpha", "task_beta"}
    # reused adapters keep bit-identical weights across the upgrade
    for t in r2["tasks"]:
        if t["disposition"] == "reused":
            assert t["adapter_key"] in {x["adapter_key"] for x in r1["tasks"]}


def test_incremental_edits_match_full_rebuild(sample_dir, tmp_path):
    """Edit sequence: remove a task, change rank/alpha of another, replace a
    base layer, switch quant tier.  The incremental store must match a fresh
    full rebuild field by field."""
    work = str(tmp_path / "work")
    shutil.copytree(sample_dir, work)
    store = str(tmp_path / "store_inc")

    run_pipeline(work, store)  # v1: initial

    # edit 1: drop task_delta, change task_beta rank and alpha
    tasks_path = os.path.join(work, "tasks.json")
    tasks = json.load(open(tasks_path))
    tasks["tasks"] = [t for t in tasks["tasks"] if t["name"] != "task_delta"]
    for t in tasks["tasks"]:
        if t["name"] == "task_beta":
            t["rank"] = 2
            t["alpha"] = 3.0
    json.dump(tasks, open(tasks_path, "w"), indent=2, sort_keys=True)
    run_pipeline(work, store)  # v2

    # edit 2: base single-layer replacement
    apply_upgrade_step(work, 0)
    run_pipeline(work, store)  # v3

    # edit 3: quant tier switch 12 -> 16 bits
    cfg_path = os.path.join(work, "peft_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    cfg["frac_bits"] = 16
    cfg["quant_tier"] = 2
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=True)
    r_inc = run_pipeline(work, store)  # v4

    # fresh full rebuild from the final inputs
    full = str(tmp_path / "full")
    shutil.copytree(work, full)
    r_full = run_pipeline(full, str(tmp_path / "store_full"))

    a, b = strip_version_fields(r_inc), strip_version_fields(r_full)
    assert a == b
    # disposition ordering is deterministic (task-name order) in both runs
    assert [t["name"] for t in r_inc["tasks"]] == [t["name"] for t in r_full["tasks"]]
    # adapter and merged weights identical element-wise
    for stage in ("train", "merge"):
        d_inc = os.path.join(store, "stages", stage)
        d_full = os.path.join(tmp_path, "store_full", "stages", stage)
        assert sorted(os.listdir(d_inc)) == sorted(os.listdir(d_full))
        for rel in os.listdir(d_inc):
            if rel.endswith(".npy"):
                assert np.array_equal(
                    np.load(os.path.join(d_inc, rel)), np.load(os.path.join(d_full, rel))
                ), f"{stage}/{rel}"
