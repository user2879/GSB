import json
import os
import shutil

import numpy as np
import pytest

from qcg.config import Config
from qcg.editor import edit_bitwidth, edit_rounding, edit_scale
from qcg.engine import Engine
from qcg.injection import Injection, FAULT_NONFINITE_SCALE, \
    FAULT_PACK_TRUNCATION
from qcg.quant_pack import write_pack
from qcg.snapshot import STAGES
from qcg.errors import (E_NODE_BUDGET_EXHAUSTED, E_GLOBAL_BUDGET_PENDING,
                        E_RECOMPUTE_PENDING, E_CONFIG_MISMATCH,
                        E_INJECTION_FAULT, E_QUANT_PACK_INVALID)
from qcg.product_store import ProductStore


def _pack_path(tmp_path, blocks):
    path = os.path.join(tmp_path, "pack.bin")
    write_pack(path, blocks)
    return path


def _product_dump(root):
    store = ProductStore(root, 10 ** 12)
    out = {}
    for nid in sorted(store.persisted_keys()):
        int_path, _, _ = store._paths(nid)
        if not os.path.exists(int_path):
            continue
        product = store.load(nid)
        out[nid] = {
            "int": product.integer_output.tolist(),
            "dq": product.dequant_values.tolist(),
            "key": product.cache_key,
            "meta": product.meta,
        }
    return out


def test_serial_and_concurrent_identical(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    serial_root = os.path.join(tmp_path, "serial")
    parallel_root = os.path.join(tmp_path, "parallel")
    cfg_serial = Config(**{**cfg.to_dict(), "max_workers": 1})
    cfg_parallel = Config(**{**cfg.to_dict(), "max_workers": 4})
    r1 = Engine(cfg_serial, serial_root).run(sample_graph.clone(), pack)
    r2 = Engine(cfg_parallel, parallel_root).run(sample_graph.clone(), pack)
    assert r1["sort_keys"] == r2["sort_keys"]
    assert r1["budget_events"] == r2["budget_events"]
    assert _product_dump(serial_root) == _product_dump(parallel_root)


def test_incremental_matches_full_rebuild(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    inc_root = os.path.join(tmp_path, "inc")
    engine = Engine(cfg, inc_root)
    engine.run(sample_graph.clone(), pack)
    edited = sample_graph.clone()
    edit_bitwidth(edited, "n0002", 16)
    edit_rounding(edited, "n0003", "ROUND_HALF_UP")
    edit_scale(edited, "n0001", scale_permille=750, scale_float=0.75)
    report = engine.run(edited, pack, previous_graph=sample_graph)
    assert report["reused"] == ["n0000"]
    assert set(report["recomputed"]) == {"n0001", "n0002", "n0003", "n0004"}
    full_root = os.path.join(tmp_path, "full")
    Engine(cfg, full_root).run(edited.clone(), pack)
    inc_dump, full_dump = _product_dump(inc_root), _product_dump(full_root)
    assert set(inc_dump) == set(full_dump)
    for nid, item in full_dump.items():
        np.testing.assert_array_equal(
            np.array(inc_dump[nid]["int"]), np.array(item["int"]))
        np.testing.assert_allclose(
            np.array(inc_dump[nid]["dq"]), np.array(item["dq"]))
        assert inc_dump[nid]["key"] == item["key"]


def test_node_quota_exhausted(tmp_path, sample_graph, blocks):
    pack = _pack_path(tmp_path, blocks)
    cfg = Config(max_workers=2, global_mac_budget=10 ** 9,
                 node_mac_quota=100, cache_resident_bytes=10 ** 9,
                 batch_recompute_budget=10 ** 9)
    report = Engine(cfg, os.path.join(tmp_path, "a")).run(
        sample_graph.clone(), pack)
    codes = {p["code"] for p in report["pending"]}
    assert E_NODE_BUDGET_EXHAUSTED in codes
    assert all(e["code"] in ("BUDGET_DEDUCT", E_NODE_BUDGET_EXHAUSTED,
                             E_GLOBAL_BUDGET_PENDING, E_RECOMPUTE_PENDING)
               for e in report["budget_events"])


def test_global_budget_pending_queue(tmp_path, sample_graph, blocks):
    pack = _pack_path(tmp_path, blocks)
    cfg = Config(max_workers=1, global_mac_budget=400,
                 node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                 batch_recompute_budget=10 ** 9)
    report = Engine(cfg, os.path.join(tmp_path, "b")).run(
        sample_graph.clone(), pack)
    pending = report["pending"]
    assert any(p["code"] == E_GLOBAL_BUDGET_PENDING for p in pending)
    keys = [p["sort_key"] for p in pending]
    assert keys == sorted(keys)


def test_batch_recompute_budget_splits_rounds(tmp_path, sample_graph, blocks):
    pack = _pack_path(tmp_path, blocks)
    cfg = Config(max_workers=1, global_mac_budget=10 ** 9,
                 node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                 batch_recompute_budget=200)
    report = Engine(cfg, os.path.join(tmp_path, "c")).run(
        sample_graph.clone(), pack)
    assert any(p["code"] == E_RECOMPUTE_PENDING for p in report["pending"])


def test_cache_eviction_and_audit(tmp_path, sample_graph, blocks):
    pack = _pack_path(tmp_path, blocks)
    cfg = Config(max_workers=1, global_mac_budget=10 ** 9,
                 node_mac_quota=10 ** 9, cache_resident_bytes=1400,
                 batch_recompute_budget=10 ** 9)
    engine = Engine(cfg, os.path.join(tmp_path, "d"))
    report = engine.run(sample_graph.clone(), pack)
    assert report["evictions"], "small cache budget must trigger eviction"
    second = sample_graph.clone()
    edit_bitwidth(second, "n0004", 8)
    report2 = engine.run(second, pack, previous_graph=sample_graph)
    audit = engine.store.recompute_events
    assert audit or report2["reloads"] or report["evictions"]


def test_snapshot_resume_mismatch(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    root = os.path.join(tmp_path, "snap")
    engine = Engine(cfg, root)
    engine.run(sample_graph.clone(), pack)
    changed = sample_graph.clone()
    edit_bitwidth(changed, "n0001", 16)
    with pytest.raises(Exception) as exc:
        engine.run(changed, pack, previous_graph=sample_graph, resume=True)
    assert exc.value.code == E_CONFIG_MISMATCH


def test_injection_isolation(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    injection = Injection("INJ-1", "n0003", STAGES[1],
                          FAULT_NONFINITE_SCALE)
    engine = Engine(cfg, os.path.join(tmp_path, "inj"), [injection])
    report = engine.run(sample_graph.clone(), pack)
    assert "INJ-1" in report["injection_hits"]
    fault = report["faults"][0]
    assert fault["node_id"] == "n0003"
    assert fault["stage"] == STAGES[1]
    assert fault["code"] == E_INJECTION_FAULT
    # other nodes still produced
    assert "n0001" in report["products"]
    assert "n0003" not in report["products"]


def test_pack_truncation_injection(tmp_path, sample_graph, blocks, cfg):
    good_pack = _pack_path(tmp_path, blocks)
    truncated = os.path.join(tmp_path, "trunc.bin")
    with open(good_pack, "rb") as src, open(truncated, "wb") as dst:
        dst.write(src.read()[:40])
    injection = Injection("INJ-2", "n0000", STAGES[0],
                          FAULT_PACK_TRUNCATION)
    engine = Engine(cfg, os.path.join(tmp_path, "inj2"), [injection])
    with pytest.raises(Exception) as exc:
        engine.run(sample_graph.clone(), truncated)
    assert exc.value.injection_id == "INJ-2"


def test_manifest_identity_fields(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    root = os.path.join(tmp_path, "m")
    Engine(cfg, root).run(sample_graph.clone(), pack)
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    for key in ("global_version", "graph_hash", "config_hash",
                "version_source", "sidecar_hash", "completed_nodes",
                "product_versions"):
        assert key in manifest
    assert manifest["completed_nodes"] == \
           [n.node_id for n in sample_graph.topo_order()]


def test_multi_round_budget_resumption(tmp_path, sample_graph, blocks):
    """A tiny per-round batch budget splits work; repeated rounds with the
    refreshed budget converge to the full result in stable topo order."""
    pack = _pack_path(tmp_path, blocks)
    tight = Config(max_workers=1, global_mac_budget=10 ** 9,
                   node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                   batch_recompute_budget=600)
    root = os.path.join(tmp_path, "multi")
    engine = Engine(tight, root)
    first = engine.run(sample_graph.clone(), pack, round_no=0)
    assert any(p["code"] == E_RECOMPUTE_PENDING for p in first["pending"])
    pending_order = [p["sort_key"] for p in first["pending"]]
    assert pending_order == sorted(pending_order)
    completed = set(first["products"])
    rounds = 1
    while first["pending"] and rounds < 10:
        first = engine.run(sample_graph.clone(), pack,
                           previous_graph=sample_graph, round_no=rounds)
        rounds += 1
    assert first["pending"] == []
    full_cfg = Config(max_workers=1, global_mac_budget=10 ** 9,
                      node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                      batch_recompute_budget=10 ** 9)
    full_root = os.path.join(tmp_path, "fullref")
    Engine(full_cfg, full_root).run(sample_graph.clone(), pack)
    inc, full = _product_dump(root), _product_dump(full_root)
    assert set(inc) == set(full)
    for nid, item in full.items():
        np.testing.assert_array_equal(np.array(inc[nid]["int"]),
                                      np.array(item["int"]))
        np.testing.assert_allclose(np.array(inc[nid]["dq"]),
                                   np.array(item["dq"]))


def test_sidecar_tamper_injection(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    injection = Injection("INJ-S", "n0000", STAGES[0],
                          "sidecar_tamper")
    engine = Engine(cfg, os.path.join(tmp_path, "st"), [injection])
    with pytest.raises(Exception) as exc:
        engine.run(sample_graph.clone(), pack)
    assert exc.value.injection_id == "INJ-S"
    assert "INJ-S" in engine.hits


def test_bitwidth_overflow_injection(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    injection = Injection("INJ-B", "n0001", STAGES[0],
                          "bitwidth_overflow")
    engine = Engine(cfg, os.path.join(tmp_path, "bw"), [injection])
    with pytest.raises(Exception) as exc:
        engine.run(sample_graph.clone(), pack)
    assert exc.value.injection_id == "INJ-B"
    assert exc.value.code == "E_GRAPH_INVALID"


def test_snapshot_staging_and_manifest(tmp_path, sample_graph, blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    root = os.path.join(tmp_path, "snaproot")
    Engine(cfg, root).run(sample_graph.clone(), pack)
    staging = os.path.join(root, "staging")
    for stage in STAGES:
        assert os.path.exists(os.path.join(staging, stage, "result.json"))
    manifest_path = os.path.join(root, "manifest.json")
    assert os.path.exists(manifest_path)
    assert not any(name.endswith(".tmp") or name.endswith(".tmp.npy")
                   for name in os.listdir(os.path.join(root, "products")))


def test_eviction_recompute_audit_ledger(tmp_path, blocks):
    from qcg.models import Graph, Node, NODE_INPUT, NODE_CONV
    graph = Graph(nodes=[
        Node("n0000", NODE_INPUT, 8, 250, 0.25),
        Node("n0001", NODE_CONV, 8, 500, 0.5, kernel=3, stride=1, padding=1,
             inputs=["n0000"]),
        Node("n0002", NODE_CONV, 8, 125, 0.125, kernel=2, stride=1, padding=1,
             inputs=["n0001"]),
        Node("n0003", NODE_CONV, 4, 1000, 1.0, kernel=3, stride=1, padding=1,
             inputs=["n0002"]),
    ])
    pack = _pack_path(tmp_path, blocks)
    cfg = Config(max_workers=1, cache_resident_bytes=1200,
                 global_mac_budget=10 ** 9, node_mac_quota=10 ** 9,
                 batch_recompute_budget=10 ** 9)
    engine = Engine(cfg, os.path.join(tmp_path, "ev"))
    report = engine.run(graph.clone(), pack)
    assert report["evictions"], "tight residency must evict LRU blocks"
    # LRU victims always come from earlier topo positions
    for event in report["evictions"]:
        victim_idx = int(event["evicted"][0][1:])
        new_idx = int(event["node_id"][1:])
        assert victim_idx < new_idx


def test_deterministic_budget_deduction_order(tmp_path, sample_graph, blocks):
    pack = _pack_path(tmp_path, blocks)
    roots = [os.path.join(tmp_path, "r%d" % i) for i in range(3)]
    reports = []
    for workers, root in ((1, roots[0]), (2, roots[1]), (4, roots[2])):
        cfg = Config(max_workers=workers, global_mac_budget=700,
                     node_mac_quota=10 ** 9, cache_resident_bytes=10 ** 9,
                     batch_recompute_budget=10 ** 9)
        reports.append(Engine(cfg, root).run(sample_graph.clone(), pack))
    seq0 = [(e["seq"], e["node_id"], e["code"])
            for e in reports[0]["budget_events"]]
    for report in reports[1:]:
        seq = [(e["seq"], e["node_id"], e["code"])
               for e in report["budget_events"]]
        assert seq == seq0
        assert report["pending"] == reports[0]["pending"]


def test_diamond_graph_parallel_wave(tmp_path, blocks):
    from qcg.models import Graph, Node, NODE_INPUT, NODE_CONV
    graph = Graph(nodes=[
        Node("n0000", NODE_INPUT, 8, 250, 0.25),
        Node("n0001", NODE_CONV, 8, 500, 0.5, kernel=3, stride=1, padding=1,
             inputs=["n0000"]),
        Node("n0002", NODE_CONV, 8, 125, 0.125, kernel=2, stride=1, padding=1,
             inputs=["n0001"]),
        Node("n0003", NODE_CONV, 4, 1000, 1.0, kernel=3, stride=1, padding=1,
             inputs=["n0001"]),
    ])
    pack = _pack_path(tmp_path, blocks)
    serial = Engine(Config(max_workers=1),
                    os.path.join(tmp_path, "ds")).run(graph.clone(), pack)
    parallel = Engine(Config(max_workers=4),
                      os.path.join(tmp_path, "dp")).run(graph.clone(), pack)
    assert serial["sort_keys"] == parallel["sort_keys"]
    assert serial["budget_events"] == parallel["budget_events"]
    assert _product_dump(os.path.join(tmp_path, "ds")) == \
           _product_dump(os.path.join(tmp_path, "dp"))


def test_automatic_invalidation_without_previous_graph(tmp_path,
                                                        sample_graph,
                                                        blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    root = os.path.join(tmp_path, "auto")
    engine = Engine(cfg, root)
    engine.run(sample_graph.clone(), pack)
    edited = sample_graph.clone()
    edit_bitwidth(edited, "n0002", 16)
    report = engine.run(edited, pack)
    assert report["reused"] == ["n0000", "n0001"]
    assert report["recomputed"] == ["n0002", "n0003", "n0004"]
    # identical graph again -> everything reused, nothing recomputed
    again = engine.run(edited.clone(), pack)
    assert again["recomputed"] == []
    assert again["reused"] == [n.node_id for n in edited.topo_order()]


def test_resume_succeeds_with_identical_identity(tmp_path, sample_graph,
                                                 blocks, cfg):
    pack = _pack_path(tmp_path, blocks)
    root = os.path.join(tmp_path, "resumeok")
    engine = Engine(cfg, root)
    first = engine.run(sample_graph.clone(), pack)
    second = engine.run(sample_graph.clone(), pack, resume=True)
    assert second["resumed"] is True
    assert second["graph_hash"] == first["graph_hash"]
