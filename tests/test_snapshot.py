"""要求 6：不可变快照、版本/哈希识别与预算降级。"""
from __future__ import annotations

import pytest

from syncbench.config import default_config
from syncbench.engine import Engine
from syncbench.errors import E_SNAPSHOT_STALE, BusinessError
def test_snapshot_immutable_and_versioned(engine):
    snap = engine.snapshots.current
    assert snap is not None
    assert snap.content_hash
    # 同版本重复渲染守卫通过
    assert engine.snapshots.render_guard(snap.version) is snap


def test_stale_snapshot_rejected(engine):
    current = engine.snapshots.current.version
    with pytest.raises(BusinessError) as exc:
        engine.snapshots.render_guard(current + 99)
    assert exc.value.code == E_SNAPSHOT_STALE


def test_snapshot_monotonic_version(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    v0 = eng.snapshots.current.version
    from syncbench.config import PARAM_GAP_POLICY, GAP_MARK
    eng.apply_change(PARAM_GAP_POLICY, 1, GAP_MARK)
    assert eng.snapshots.current.version == v0 + 1


def test_budget_degradation_does_not_change_state(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    before = {nid: st.content_hash for nid, st in eng.graph.nodes.items()}
    # 把预算压到 0，强制每次瓦片渲染超预算降级
    eng.cfg.tile_budget_ms = 0.0
    from syncbench.config import PARAM_GAP_POLICY, GAP_MARK
    eng.apply_change(PARAM_GAP_POLICY, 2, GAP_MARK)
    # 测量/对齐节点哈希不因降级而改变
    after = {nid: st.content_hash for nid, st in eng.graph.nodes.items()}
    changed = [nid for nid in before if before[nid] != after[nid]]
    assert all(nid.endswith(("gapfill:2", "tile:2")) for nid in changed)
