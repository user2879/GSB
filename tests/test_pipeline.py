import json
import os
import shutil

import numpy as np
import yaml

from peft_fp import errors
from peft_fp.pipeline import run_pipeline
from tests.conftest import strip_version_fields


def test_determinism_repeated_runs(sample_dir, tmp_path):
    r1 = run_pipeline(sample_dir, str(tmp_path / "s1"))
    r2 = run_pipeline(sample_dir, str(tmp_path / "s2"))
    assert strip_version_fields(r1) == strip_version_fields(r2)
    # adapter binaries identical element-wise
    for rel in os.listdir(os.path.join(tmp_path, "s1", "stages", "train")):
        if rel.endswith(".npy"):
            a = np.load(os.path.join(tmp_path, "s1", "stages", "train", rel))
            b = np.load(os.path.join(tmp_path, "s2", "stages", "train", rel))
            assert np.array_equal(a, b), rel


def test_concurrent_matches_serial(sample_dir, tmp_path):
    cfg_path = os.path.join(sample_dir, "peft_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    serial_dir = str(tmp_path / "samples_serial")
    shutil.copytree(sample_dir, serial_dir)
    cfg["max_workers"] = 1
    yaml.safe_dump(cfg, open(os.path.join(serial_dir, "peft_config.yaml"), "w"), sort_keys=True)

    par_dir = str(tmp_path / "samples_par")
    shutil.copytree(sample_dir, par_dir)
    cfg["max_workers"] = 4
    yaml.safe_dump(cfg, open(os.path.join(par_dir, "peft_config.yaml"), "w"), sort_keys=True)

    r_serial = run_pipeline(serial_dir, str(tmp_path / "store_serial"))
    r_par = run_pipeline(par_dir, str(tmp_path / "store_par"))
    s, p = strip_version_fields(r_serial), strip_version_fields(r_par)
    # config hash differs (max_workers is a config field); compare the rest
    s.pop("config_hash"), p.pop("config_hash")
    assert s == p
    assert [t["name"] for t in r_par["tasks"]] == sorted(t["name"] for t in r_par["tasks"])


def test_merge_budget_exhaustion_and_pending(sample_dir, tmp_path):
    mod_dir = str(tmp_path / "samples_budget")
    shutil.copytree(sample_dir, mod_dir)
    cfg_path = os.path.join(mod_dir, "peft_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    # quota = budget // 4 tasks; layer numel up to 64, rank up to 3 -> cost up
    # to 192.  budget 300 -> quota 75: tasks with rank*numel > 75 abort, and
    # the global budget runs out before the last tasks.
    cfg["merge_budget"] = 300
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=True)

    r1 = run_pipeline(mod_dir, str(tmp_path / "b1"))
    r2 = run_pipeline(mod_dir, str(tmp_path / "b2"))
    codes = [f["code"] for f in r1["failures"]]
    assert errors.E_MERGE_BUDGET_EXHAUSTED in codes or errors.E_MERGE_PENDING in codes
    # reproducible deduction sequence and pending order
    assert r1["merge"]["consumption"] == r2["merge"]["consumption"]
    assert r1["merge"]["pending"] == r2["merge"]["pending"]
    assert r1["merge"]["consumed"] <= 300
    # pending tasks keep a stable, sorted order
    assert r1["merge"]["pending"] == sorted(r1["merge"]["pending"])


def test_snapshot_and_manifest(sample_dir, tmp_path):
    store = str(tmp_path / "store")
    r1 = run_pipeline(sample_dir, store)
    r2 = run_pipeline(sample_dir, store)
    assert r2["global_version"] == r1["global_version"] + 1
    assert r2["resume"]["resumed"] is True
    manifest = json.load(open(os.path.join(store, "manifest.json")))
    assert manifest["global_version"] == r2["global_version"]
    assert manifest["base_fingerprint"] == r2["base_fingerprint"]
    assert set(manifest["stages"]) == {"validate", "train", "merge", "export"}
    # snapshot interval = 1 -> one snapshot per version
    snaps = os.listdir(os.path.join(store, "snapshots"))
    assert f"v{r1['global_version']}" in snaps and f"v{r2['global_version']}" in snaps
    # second run reuses adapters (no retraining)
    assert all(t["disposition"] == "reused" for t in r2["tasks"])
