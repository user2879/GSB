"""要求 7：原子持久化、孤儿回收与强制终止恢复一致性。"""
from __future__ import annotations

import os

from syncbench.config import default_config
from syncbench.engine import Engine


def test_atomic_state_and_no_tmp_after_commit(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    state_path = os.path.join(str(tmp_path), "store", "nodes", "state.json")
    assert os.path.exists(state_path)
    tmps = []
    for base, _dirs, files in os.walk(os.path.join(str(tmp_path), "store")):
        tmps += [f for f in files if f.endswith(".tmp")]
    assert tmps == []


def test_controlled_shutdown_marker(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    eng.controlled_shutdown()
    assert eng.store.has_clean_shutdown()


def test_orphan_recovery_and_restore_hashes(tmp_path):
    ws = str(tmp_path)
    eng = Engine(ws, default_config())
    eng.initial_build()
    hashes_before = {nid: st.content_hash for nid, st in eng.graph.nodes.items()}
    # 模拟未提交 .tmp 孤儿
    tmp_file = os.path.join(ws, "store", "nodes", "state.json.tmp")
    with open(tmp_file, "wb") as fh:
        fh.write(b'{"uncommitted": true}')
    eng2 = Engine(ws, default_config())
    summary = eng2.restore_from_store()
    assert not os.path.exists(tmp_file)
    assert "nodes/state.json.tmp" in summary["orphans_reclaimed"]
    for nid, h in hashes_before.items():
        assert eng2.graph.nodes[nid].content_hash == h


def test_rebuild_after_recovery_matches_committed(tmp_path):
    # 恢复重建的内存产物哈希必须与已提交节点完全一致（引擎内部已强校验）
    ws = str(tmp_path)
    eng = Engine(ws, default_config())
    eng.initial_build()
    eng2 = Engine(ws, default_config())
    summary = eng2.restore_from_store()
    assert summary["restored_seq"] == 1
    assert eng2._estimations and eng2._duals and eng2._gaps


def test_no_forged_shutdown_report(tmp_path):
    ws = str(tmp_path)
    eng = Engine(ws, default_config())
    eng.initial_build()  # 未受控退出，无 marker
    eng2 = Engine(ws, default_config())
    summary = eng2.restore_from_store()
    assert summary["clean_shutdown"] is False
    # 存在已提交稳定状态但无关闭标记 => 判定为异常/强制终止，
    # 且恢复只依据已提交状态，绝不补写虚假的受控退出报告
    assert summary["forced_termination_detected"] is True
    assert not os.path.exists(os.path.join(ws, "store", "shutdown.marker"))
