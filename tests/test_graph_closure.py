"""要求 4：依赖图失效闭包精确，闭包外复用且哈希不变。"""
from __future__ import annotations

from syncbench.config import (
    GAP_DRIFT,
    PARAM_DELAY_COMP,
    PARAM_DRIFT_MODEL,
    PARAM_GAP_POLICY,
    PARAM_REFERENCE,
    DRIFT_THERMAL,
    default_config,
)
from syncbench.engine import Engine
from syncbench.graph import (
    NODE_AALIGN,
    NODE_DRIFT,
    NODE_GAPFILL,
    NODE_MAP,
    NODE_TILE,
    NODE_VALIGN,
    node_id,
)


def _snapshot_hashes(eng: Engine) -> dict[str, str]:
    return {nid: st.content_hash for nid, st in eng.graph.nodes.items()}


def test_drift_model_closure(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    before = _snapshot_hashes(eng)
    r = eng.apply_change(PARAM_DRIFT_MODEL, 2, DRIFT_THERMAL)
    closure = set(r["closure"])
    # 仅通道 2 全链失效
    assert closure == {node_id(k, 2) for k in
                       (NODE_DRIFT, NODE_GAPFILL, NODE_MAP, NODE_AALIGN,
                        NODE_VALIGN, NODE_TILE)}
    # 其它通道与通道 2 之外节点哈希不变
    for nid, h in before.items():
        if nid not in closure:
            assert eng.graph.nodes[nid].content_hash == h


def test_delay_comp_closure(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    before = _snapshot_hashes(eng)
    r = eng.apply_change(PARAM_DELAY_COMP, 3, False)
    closure = set(r["closure"])
    assert node_id(NODE_DRIFT, 3) not in closure
    assert node_id(NODE_GAPFILL, 3) not in closure
    assert node_id(NODE_MAP, 3) in closure
    for nid, h in before.items():
        if nid not in closure:
            assert eng.graph.nodes[nid].content_hash == h


def test_gap_policy_closure(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    r = eng.apply_change(PARAM_GAP_POLICY, 0, GAP_DRIFT)
    closure = set(r["closure"])
    # 只影响 gapfill 与 tile，漂移/映射/对齐不动
    assert closure == {node_id(NODE_GAPFILL, 0), node_id(NODE_TILE, 0)}


def test_reference_switch_preserves_drift_gapfill(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    before = _snapshot_hashes(eng)
    r = eng.apply_change(PARAM_REFERENCE, None, 1)
    closure = set(r["closure"])
    # 所有 map/aalign/valign/tile 失效
    for ch in range(4):
        assert node_id(NODE_MAP, ch) in closure
        assert node_id(NODE_AALIGN, ch) in closure
    # drift / gapfill 全部复用且哈希不变
    for ch in range(4):
        assert node_id(NODE_DRIFT, ch) not in closure
        assert node_id(NODE_GAPFILL, ch) not in closure
        assert eng.graph.nodes[node_id(NODE_DRIFT, ch)].content_hash == before[node_id(NODE_DRIFT, ch)]
        assert eng.graph.nodes[node_id(NODE_GAPFILL, ch)].content_hash == before[node_id(NODE_GAPFILL, ch)]


def test_closure_equals_actual_recompute(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    # 引擎内部 assert_closure_matches 已强校验；多次修改不抛即通过
    eng.apply_change(PARAM_DRIFT_MODEL, 1, DRIFT_THERMAL)
    eng.apply_change(PARAM_DELAY_COMP, 0, False)
    eng.apply_change(PARAM_GAP_POLICY, 2, GAP_DRIFT)
    eng.apply_change(PARAM_REFERENCE, None, 3)


def test_reuse_source_recorded(tmp_path):
    eng = Engine(str(tmp_path), default_config())
    eng.initial_build()
    eng.apply_change(PARAM_GAP_POLICY, 0, GAP_DRIFT)
    # 通道 1 的节点全部复用，应记录 reuse_source
    st = eng.graph.nodes[node_id(NODE_DRIFT, 1)]
    assert st.reuse_source is not None and st.reuse_source.startswith("drift:1@v")
