import json
import os

import pytest

from hdr_sdr_workbench.engine import WorkbenchEngine
from hdr_sdr_workbench.errors import (E_CLOSURE_DIVERGE, E_HASH_MISMATCH,
                                      E_RECOVER_ORPHAN, E_SNAPSHOT_STALE,
                                      E_STATE_VERSION, WorkbenchError)
from hdr_sdr_workbench.persistence import (DIR_OUTPUTS, WorkspaceStore,
                                           output_name, sha256_file)
from hdr_sdr_workbench.synthesis import SequenceConfig, generate_sequence


@pytest.fixture()
def engine_factory(tmp_path):
    def make(name="ws", cfg=None):
        ws = str(tmp_path / name)
        cfg = cfg or SequenceConfig(width=48, height=27, frame_count=8, seed=5)
        src = generate_sequence(ws, cfg)
        return WorkbenchEngine(ws, src), src, ws
    return make


def test_incremental_closure_keeps_unaffected_hash(engine_factory):
    engine, _, _ = engine_factory()
    engine.apply_operation([0, 1, 2], {"exposure_stops": 1.0})
    untouched_before = {i: engine.frame_entries[i]["output_hash"] for i in (3, 4, 5, 6, 7)}
    engine.apply_operation([0, 1], {"tonemap": "reinhard", "tonemap_param": 2.0})
    for i, digest in untouched_before.items():
        assert engine.frame_entries[i]["output_hash"] == digest
    engine.verify_all()


def test_range_and_frame_operations_byte_identical(engine_factory, tmp_path):
    cfg = SequenceConfig(width=40, height=24, frame_count=6, seed=11)
    src_a = generate_sequence(str(tmp_path / "a"), cfg)
    src_b = generate_sequence(str(tmp_path / "b"), cfg)
    eng_a = WorkbenchEngine(str(tmp_path / "a"), src_a)
    eng_b = WorkbenchEngine(str(tmp_path / "b"), src_b)
    def output_map(engine):
        return {row["frame"]: (row["state_id"], row["output_sha256"])
                for row in engine.hash_listing()}
    assert output_map(eng_a) == output_map(eng_b)  # 基线即逐字节一致
    eng_a.apply_operation([0, 1, 2, 3], {"exposure_stops": 0.5})
    for i in range(4):
        eng_b.apply_operation([i], {"exposure_stops": 0.5})
    # 区间操作与逐帧操作到达同一状态：state_id 与 8 位输出哈希必须一致
    assert output_map(eng_a) == output_map(eng_b)


def test_historical_rebuild_matches_recorded_hash(engine_factory):
    engine, _, _ = engine_factory()
    engine.apply_operation([0, 1], {"exposure_stops": 1.0})  # v1
    engine.apply_operation([0], {"colorspace": "bt709"})     # v2
    rebuilt = engine.rebuild_historical(1, [0, 1])
    recorded_path = os.path.join(engine.store.paths[DIR_OUTPUTS], output_name(0, 1))
    assert os.path.exists(recorded_path)
    assert rebuilt[0]["output_hash"] == sha256_file(recorded_path)


def test_hash_mismatch_detected(engine_factory):
    engine, _, ws = engine_factory()
    engine.apply_operation([0, 1], {"exposure_stops": 1.0})
    entry = engine.frame_entries[0]
    path = os.path.join(engine.store.paths[DIR_OUTPUTS], entry["output_file"])
    with open(path, "r+b") as fh:
        fh.seek(0)
        fh.write(b"\xff\xff\xff")
    with pytest.raises(WorkbenchError) as info:
        engine.verify_all()
    assert info.value.code == E_HASH_MISMATCH


def test_orphan_recovery_and_resume(engine_factory):
    engine, src, ws = engine_factory()
    engine.apply_operation([0, 1], {"exposure_stops": 1.0})
    engine.apply_operation([2], {"tonemap": "reinhard", "tonemap_param": 4.0})
    committed = engine.hash_listing()
    # 制造未提交孤儿 + tmp 残留，模拟强制终止
    orphan = os.path.join(engine.store.paths[DIR_OUTPUTS], "orphan_uncommitted.u8")
    open(orphan, "wb").write(b"x" * 32)
    open(os.path.join(engine.store.paths["tmp"], ".partial"), "wb").write(b"p")
    assert not engine.store.shutdown_report_exists()

    store = WorkspaceStore(ws)
    referenced = {e["output_file"] for e in engine.frame_entries.values()}
    referenced |= {e["metric_file"] for e in engine.frame_entries.values()}
    with pytest.raises(WorkbenchError) as info:
        store.recover()
    assert info.value.code == E_RECOVER_ORPHAN
    orphans = set(info.value.details["orphans"])
    # 注入的两个未提交孤儿必须被回收
    assert "outputs/orphan_uncommitted.u8" in orphans
    assert "tmp/.partial" in orphans
    # manifest 当前引用的最新产物不得被当作孤儿
    assert not any(name in orphans for name in referenced)

    restarted = WorkbenchEngine(ws, src)
    report = restarted.recover()
    assert report["recovered"] is True
    assert not os.path.exists(orphan)
    assert restarted.hash_listing() == committed
    # 恢复后继续操作，结果与未中断时一致
    restarted.apply_operation([3], {"exposure_stops": -0.5})
    assert restarted.frame_entries[3]["version"] == 3
    restarted.verify_all()


def test_stale_snapshot_rejected(engine_factory):
    engine, _, ws = engine_factory()
    engine.apply_operation([0], {"exposure_stops": 1.0})
    engine.apply_operation([1], {"exposure_stops": 1.0})
    old = engine.store.read_manifest()
    stale = dict(old)
    stale["committed_at_seq"] = int(old["committed_at_seq"]) - 1
    with pytest.raises(WorkbenchError) as info:
        engine.store.reject_if_stale(stale)
    assert info.value.code == E_SNAPSHOT_STALE


def test_state_version_conflict(tmp_path):
    ws = str(tmp_path / "ws")
    cfg = SequenceConfig(width=32, height=16, frame_count=2, seed=1)
    src = generate_sequence(ws, cfg)
    engine = WorkbenchEngine(ws, src)
    manifest = engine.store.read_manifest()
    manifest["schema_version"] = 999
    engine.store.write_manifest  # 仅引用，下面用底层写
    from hdr_sdr_workbench.persistence import atomic_write_json
    atomic_write_json(os.path.join(ws, "manifest.json"), manifest)
    with pytest.raises(WorkbenchError) as info:
        WorkbenchEngine(ws, src)
    assert info.value.code == E_STATE_VERSION


def test_frame_out_of_range(engine_factory):
    engine, _, _ = engine_factory()
    with pytest.raises(WorkbenchError) as info:
        engine.apply_operation([99], {"exposure_stops": 1.0})
    assert info.value.code == E_STATE_VERSION


def test_closure_divergence_guard(engine_factory, monkeypatch):
    engine, _, _ = engine_factory()
    engine.apply_operation([0, 1], {"exposure_stops": 1.0})
    # 人为让闭包外帧状态发生变化，验证自检会报 E_CLOSURE_DIVERGE
    real_apply = engine.state_table.apply

    def patched(frames, changes, **kw):
        op = real_apply(frames, changes, **kw)
        engine.state_table.states[5] = engine.state_table.states[0]
        return op
    monkeypatch.setattr(engine.state_table, "apply", patched)
    with pytest.raises(WorkbenchError) as info:
        engine.apply_operation([0], {"exposure_stops": 2.0})
    assert info.value.code == E_CLOSURE_DIVERGE
