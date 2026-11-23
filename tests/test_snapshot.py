import json
import os

from fablock.errors import Code
from fablock.pipeline import Pipeline
from fablock.snapshot import SnapshotManager


def _snapshot(tmp_path, cfg, dataset):
    return SnapshotManager(str(tmp_path / "snap"), cfg.config_hash(),
                           "packfp", cfg.quantization_level_bits)


def test_commit_and_resume(tmp_path, cfg, dataset):
    snap = _snapshot(tmp_path, cfg, dataset)
    p1 = Pipeline(cfg, dataset, snapshot_mgr=snap)
    p1.run(2)
    manifest = snap.load_manifest()
    assert manifest["completed_batch_cursor"] == 2
    assert manifest["global_version"] == 1
    assert manifest["config_hash"] == cfg.config_hash()
    assert manifest["quant_level"] == cfg.quantization_level_bits
    assert set(manifest["artifact_versions"]) == {
        "load_validate", "block_score", "softmax_reduce", "output_summary"}
    # 阶段产物已整体替换提交
    for stage in ("load_validate", "block_score", "softmax_reduce",
                  "output_summary"):
        assert os.path.exists(os.path.join(
            str(tmp_path / "snap"), "current", f"{stage}.bin"))
    # 新实例按游标续跑：只执行剩余批次
    snap2 = _snapshot(tmp_path, cfg, dataset)
    p2 = Pipeline(cfg, dataset, snapshot_mgr=snap2)
    reports = p2.run(3)
    ran = [r["batch"] for r in reports if r.get("batch") is not None]
    assert ran == [2]
    assert snap2.load_manifest()["completed_batch_cursor"] == 3


def test_version_mismatch(tmp_path, cfg, dataset):
    snap = _snapshot(tmp_path, cfg, dataset)
    Pipeline(cfg, dataset, snapshot_mgr=snap).run(1)
    manifest = snap.load_manifest()
    manifest["global_version"] = 999
    with open(snap.manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    cursor, issues = snap.check_resume()
    assert cursor == 0
    assert issues[0]["code"] == Code.E_SNAPSHOT_VERSION_MISMATCH


def test_config_hash_mismatch(tmp_path, cfg, dataset):
    snap = _snapshot(tmp_path, cfg, dataset)
    Pipeline(cfg, dataset, snapshot_mgr=snap).run(1)
    other = cfg.with_overrides(fixed_point_frac_bits=10)
    snap2 = SnapshotManager(str(tmp_path / "snap"), other.config_hash(),
                            "packfp", other.quantization_level_bits)
    cursor, issues = snap2.check_resume()
    assert cursor == 0
    assert issues[0]["code"] == Code.E_SNAPSHOT_VERSION_MISMATCH


def test_graceful_stop_after_batch_commit(tmp_path, cfg, dataset):
    snap = _snapshot(tmp_path, cfg, dataset)
    p = Pipeline(cfg, dataset, snapshot_mgr=snap)

    def stop_after_first(batch, report):
        if batch == 0:
            p.request_stop()  # 模拟退出信号：完成当前批次落盘后停止

    reports = p.run(5, on_batch_complete=stop_after_first)
    ran = [r["batch"] for r in reports if r.get("batch") is not None]
    assert ran == [0]  # 当前批次完成后才停止
    assert snap.load_manifest()["completed_batch_cursor"] == 1


def test_commit_interval(tmp_path, cfg, dataset):
    snap = _snapshot(tmp_path, cfg, dataset)
    every2 = cfg.with_overrides(snapshot_commit_interval_batches=2)
    snap = SnapshotManager(str(tmp_path / "snap"), every2.config_hash(),
                           "packfp", every2.quantization_level_bits)
    Pipeline(every2, dataset, snapshot_mgr=snap).run(3)
    # 间隔为 2：批次 1 提交一次，批次 2 未达间隔不提交
    assert snap.load_manifest()["completed_batch_cursor"] == 2
