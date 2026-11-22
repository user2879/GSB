"""端到端：失效判定、预算待办、增量vs全量、恢复、并发一致性。"""

import json
import os
import shutil

import pytest

from tlc import datagen
from tlc.errors import BizError, E_RETRAIN_BUDGET_PENDING, E_SNAPSHOT_STALE
from tlc.pipeline import CrashError, rebuild, run_batch


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return datagen.generate(str(tmp_path_factory.mktemp("data")))


def _run_chain(data_dir, batches, state_dir):
    report = None
    for k in range(0, batches + 1):
        report = run_batch(data_dir, k, state_dir)
    return report


def test_deterministic_repeat(data_dir, tmp_path):
    r1 = _run_chain(data_dir, 1, str(tmp_path / "s1"))
    r2 = _run_chain(data_dir, 1, str(tmp_path / "s2"))
    assert r1 == r2


def test_shift_driven_invalidation(data_dir, tmp_path):
    run_batch(data_dir, 0, str(tmp_path / "s"))
    report = run_batch(data_dir, 1, str(tmp_path / "s"))
    by_name = {t["task"]: t for t in report["tasks"]}
    # layer2 被替换：task_alpha 目标层未被替换且偏移低于阈值 -> 复用
    alpha = by_name["task_alpha"]
    assert alpha["decision"] == "reuse"
    assert alpha["invalidation"]["reason"] == "base_shift"
    assert alpha["invalidation"]["max_shift"] <= alpha["invalidation"]["threshold"]
    compared = alpha["invalidation"]["compared"]
    assert {c["layer"] for c in compared} == {"layer1", "layer2", "layer3"}
    assert all("shift" in c for c in compared)
    # task_beta 上游 layer2 被替换且偏移超阈值 -> 重训（非引用即失效，而是偏移判定）
    beta = by_name["task_beta"]
    assert beta["decision"] == "retrain"
    assert beta["invalidation"]["max_shift"] > beta["invalidation"]["threshold"]


def test_budget_pending_and_retry(data_dir, tmp_path):
    state = str(tmp_path / "s")
    _run_chain(data_dir, 1, state)
    r2 = run_batch(data_dir, 2, state)
    assert E_RETRAIN_BUDGET_PENDING in r2["codes"]
    assert r2["summary"]["pending"] == ["task_gamma"]
    gamma = next(t for t in r2["tasks"] if t["task"] == "task_gamma")
    # 待办任务不写入验证汇总，但贡献偏移仍写入诊断
    assert gamma["metrics"] is None
    assert gamma["invalidation"]["max_shift"] > gamma["invalidation"]["threshold"]
    assert gamma["pending"]["code"] == E_RETRAIN_BUDGET_PENDING
    # 下一批预算恢复后优先处理并完成训练
    r3 = run_batch(data_dir, 3, state)
    assert r3["summary"]["pending"] == []
    gamma3 = next(t for t in r3["tasks"] if t["task"] == "task_gamma")
    assert gamma3["metrics"] is not None
    assert gamma3["train"] is not None


def test_incremental_equals_full_rebuild(data_dir, tmp_path):
    inc = _run_chain(data_dir, 3, str(tmp_path / "inc"))
    full = rebuild(data_dir, 3, str(tmp_path / "full"))
    assert inc == full
    snap_inc = json.load(open(os.path.join(tmp_path / "inc", "snapshot.json")))
    snap_full = json.load(open(os.path.join(tmp_path / "full", "snapshot.json")))
    assert snap_inc == snap_full


def test_crash_resume_field_identical(data_dir, tmp_path, monkeypatch):
    ref = _run_chain(data_dir, 2, str(tmp_path / "ref"))
    state = str(tmp_path / "crashy")
    _run_chain(data_dir, 1, state)
    monkeypatch.setenv("TLC_CRASH_AFTER_STAGE", "C_train")
    monkeypatch.setenv("TLC_CRASH_MODE", "raise")
    with pytest.raises(CrashError):
        run_batch(data_dir, 2, state)
    monkeypatch.delenv("TLC_CRASH_AFTER_STAGE")
    monkeypatch.delenv("TLC_CRASH_MODE")
    resumed = run_batch(data_dir, 2, state)
    assert resumed == ref


def test_snapshot_stale(data_dir, tmp_path, monkeypatch):
    state = str(tmp_path / "s")
    _run_chain(data_dir, 1, state)
    monkeypatch.setenv("TLC_CRASH_AFTER_STAGE", "A_forward")
    monkeypatch.setenv("TLC_CRASH_MODE", "raise")
    with pytest.raises(CrashError):
        run_batch(data_dir, 2, state)
    monkeypatch.delenv("TLC_CRASH_AFTER_STAGE")
    monkeypatch.delenv("TLC_CRASH_MODE")
    # 修改配置使 run_key 变化 -> 暂存区过期
    cfg_path = os.path.join(data_dir, "config.json")
    cfg = json.load(open(cfg_path))
    old = cfg["learning_rate"]
    cfg["learning_rate"] = 0.05
    json.dump(cfg, open(cfg_path, "w"))
    try:
        with pytest.raises(BizError) as exc:
            run_batch(data_dir, 2, state)
        assert exc.value.code == E_SNAPSHOT_STALE
    finally:
        cfg["learning_rate"] = old
        json.dump(cfg, open(cfg_path, "w"))


def test_concurrency_invariance(data_dir, tmp_path):
    par = _run_chain(data_dir, 3, str(tmp_path / "par"))
    cfg_path = os.path.join(data_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["max_concurrency"] = 1
    json.dump(cfg, open(cfg_path, "w"))
    try:
        ser = _run_chain(data_dir, 3, str(tmp_path / "ser"))
    finally:
        cfg["max_concurrency"] = 3
        json.dump(cfg, open(cfg_path, "w"))
    for r in (par, ser):
        r.pop("run_key")
        r.pop("config_hash")
        r["config"].pop("max_concurrency")
    assert par == ser


def test_closure_tolerance(data_dir, tmp_path):
    report = _run_chain(data_dir, 3, str(tmp_path / "s"))
    tol = report["config"]["closure_rel_tol"]
    for task in report["tasks"]:
        if task["closure"] is not None:
            assert task["closure"]["rel_err"] <= tol


def test_task_add_remove_and_content_change(data_dir, tmp_path):
    r2 = _run_chain(data_dir, 2, str(tmp_path / "s"))
    names = [t["task"] for t in r2["tasks"]]
    assert "task_delta" in names  # 批次 2 新增任务
    beta = next(t for t in r2["tasks"] if t["task"] == "task_beta")
    assert beta["decision"] == "retrain"
    assert beta["invalidation"]["reason"] == "content_changed"  # 秩 2->3
    assert beta["adapter"]["rank"] == 3
