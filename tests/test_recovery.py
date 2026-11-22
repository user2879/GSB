"""快照恢复与版本变化识别 (业务要求 8)."""

from __future__ import annotations

import json
from pathlib import Path

from cqdiag.pipeline import Pipeline
from cqdiag.store import SnapshotStore

from conftest import make_config, make_samples


def test_staging_leftover_discarded(tmp_path):
    config = make_config()
    samples = make_samples(config)
    Pipeline(config, samples, SnapshotStore(tmp_path)).run()
    # 模拟提交中断: 残留 staging 目录
    junk = tmp_path / "staging" / "calib"
    junk.mkdir(parents=True)
    (junk / "t__0.json").write_text("{}")
    store = SnapshotStore(tmp_path)
    assert any("staging" in m for m in store.recovery_log)
    assert not (tmp_path / "staging").exists()
    # 已提交产物不受影响
    assert store.get("calib", "t:0") is not None


def test_corrupt_artifact_dropped_and_recomputed(tmp_path):
    config = make_config()
    samples = make_samples(config)
    Pipeline(config, samples, SnapshotStore(tmp_path)).run()
    victim = tmp_path / "snapshots" / "quant" / "t__1.json"
    victim.write_text("{corrupt")
    store = SnapshotStore(tmp_path)
    assert any("quant/t:1" in m for m in store.recovery_log)
    result = Pipeline(config, samples, store).run()
    # 损坏产物的列被重算, 其余列复用
    assert result.stats["quant"]["recomputed"] == ["t:1"]
    assert sorted(result.stats["quant"]["reused"]) == ["t:0", "t:2", "t:3"]


def test_version_change_detection(tmp_path):
    config = make_config()
    samples = make_samples(config)
    Pipeline(config, samples, SnapshotStore(tmp_path)).run()
    store = SnapshotStore(tmp_path)
    from cqdiag.fingerprints import config_fp
    assert store.detect_version_change(config_fp(config)) == "unchanged"
    config2 = make_config(quant_bits=4)
    assert store.detect_version_change(config_fp(config2)) == "config_changed"
    assert SnapshotStore(tmp_path / "fresh").detect_version_change("x") == "fresh_workspace"


def test_manifest_records_versions(tmp_path):
    config = make_config()
    samples = make_samples(config)
    Pipeline(config, samples, SnapshotStore(tmp_path)).run()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["global_version"] == 1
    assert manifest["config_hash"]
    assert manifest["completed_columns"] == ["t:0", "t:1", "t:2", "t:3"]
    assert set(manifest["artifacts"]["calib"].keys()) == {"t:0", "t:1", "t:2", "t:3"}
    for entry in manifest["artifacts"]["calib"].values():
        assert entry["key"] and entry["version"] == 1
