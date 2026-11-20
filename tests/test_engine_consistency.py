import dataclasses
import shutil
from pathlib import Path

import pytest

from rerank_core.clock import DeterministicClock
from rerank_core.commands import Command
from rerank_core.engine import RerankEngine
from rerank_core.snapshot import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]


def _make_engine(config, loaded, tmp_path, *, concurrency=None, budget=1000.0):
    candidates, failures = loaded
    cfg = config
    if concurrency is not None:
        cfg = dataclasses.replace(config, concurrency=concurrency)
    cfg = dataclasses.replace(cfg, latency_budget_ms=budget)
    store = SnapshotStore(tmp_path / "snap")
    return RerankEngine(cfg, candidates, failures, store, DeterministicClock())


def _business_payload(engine):
    report = engine.build_report(high_tier=False)
    return [
        {
            "request_id": q["request_id"],
            "fusion_order": q["fusion_order"],
            "fusion_scores": q["fusion_scores"],
            "contributions": q["contributions"],
            "rerank": q["rerank"],
            "fusion_cache_keys": q["fusion_cache_keys"],
            "rerank_cache_key": q["rerank_cache_key"],
            "normalize_cache_key": q["normalize_cache_key"],
            "discount_cache_keys": q["discount_cache_keys"],
        }
        for q in report["per_request"]
    ]


def test_concurrency_matches_single_thread(config, loaded, tmp_path):
    single = _make_engine(config, loaded, tmp_path / "single", concurrency=1)
    multi = _make_engine(config, loaded, tmp_path / "multi", concurrency=4)
    single.handle_command(Command(0, "poll", {}))
    multi.handle_command(Command(0, "poll", {}))
    assert _business_payload(single) == _business_payload(multi)


def test_window_switch_changes_rerank_scope(config, loaded, tmp_path):
    engine = _make_engine(config, loaded, tmp_path)
    engine.handle_command(Command(0, "poll", {}))
    before = {q["request_id"]: q["rerank"] for q in
              engine.build_report(high_tier=False)["per_request"]}

    engine.handle_command(Command(1, "set_rerank_window", {"window": 2}))
    after = {q["request_id"]: q["rerank"] for q in
             engine.build_report(high_tier=False)["per_request"]}
    for request_id, rerank in after.items():
        assert rerank["window_size"] == 2
        assert len(rerank["final_order"]) == 2
        assert len(rerank["outside_order"]) == len(before[request_id]["final_order"]) + \
            len(before[request_id]["outside_order"]) - 2
    # 窗口外候选不参与重排，保持融合序
    fused = engine.build_report(high_tier=False)["per_request"][0]
    assert fused["rerank"]["outside_order"] == fused["fusion_order"][2:]


def test_frac_bits_change_alters_quantization_and_scores(config, loaded, tmp_path):
    engine = _make_engine(config, loaded, tmp_path)
    engine.handle_command(Command(0, "poll", {}))
    low = engine.build_report(high_tier=False)["per_request"][0]
    low_score = low["fusion_scores"][low["fusion_order"][0]]
    low_norm = low["contributions"][low["fusion_order"][0]][0]["normalized_fixed"]

    engine.handle_command(
        Command(1, "set_quant", {"frac_bits": 8, "quant_levels": 256})
    )
    report = engine.build_report(high_tier=False)
    assert report["frac_bits"] == 8
    high = report["per_request"][0]
    high_norm = high["contributions"][high["fusion_order"][0]][0]["normalized_fixed"]
    assert high_norm != low_norm
    # 8 档相对 4 档放大 10000 倍，舍入误差在一个低档位宽内
    assert abs(high_norm - low_norm * 10000) <= 10000
    assert high["fusion_scores"][high["fusion_order"][0]] != low_score


def test_weight_change_reuses_normalization(config, loaded, tmp_path):
    engine = _make_engine(config, loaded, tmp_path)
    engine.handle_command(Command(0, "poll", {}))
    response = engine.handle_command(
        Command(1, "set_weight", {"feature_index": 0, "weight": 0.75})
    )
    for item in response.recomputed:
        assert "normalize" not in item["stages"]
        assert "fusion" in item["stages"]
    assert response.reused == []


def test_candidate_add_only_affects_one_request(config, loaded, tmp_path):
    engine = _make_engine(config, loaded, tmp_path)
    engine.handle_command(Command(0, "poll", {}))
    record = {
        "request_id": "req-004",
        "item_id": "item-004-added",
        "recall_scores": {ch: 0.3 for ch in config.recall_channels},
        "dense": [0.1, 0.2, 0.3, 0.4],
        "sparse": [1, 2, 3],
    }
    response = engine.handle_command(
        Command(1, "add_candidate", {"request_id": "req-004", "record": record})
    )
    recomputed = {item["request_id"] for item in response.recomputed}
    assert recomputed == {"req-004"}
    reused = {item["request_id"] for item in response.reused}
    assert "req-000" in reused


def test_budget_pending_then_drain(config, loaded, tmp_path):
    # 预算为 0：首条命令必然挂起，返回中间状态而非阻塞
    engine = _make_engine(config, loaded, tmp_path, budget=0.0)
    first = engine.handle_command(Command(0, "poll", {}))
    assert first.code == "E_INTERACT_BUDGET_PENDING"
    assert first.pending is True
    assert engine.pending is not None

    # 放大预算后 poll 补交，最终完成
    engine.config = dataclasses.replace(engine.config, latency_budget_ms=5000.0)
    drained = engine.drain_pending(Command(1, "poll", {}))
    assert drained.code == "OK"
    assert drained.pending is False
    assert engine.pending is None
    assert drained.completed_requests == drained.total_requests


def test_snapshot_resume_reproduces_state(config, loaded, tmp_path):
    snap = tmp_path / "snap"
    engine = _make_engine(config, loaded, snap)
    engine.handle_command(Command(0, "poll", {}))
    payload_before = _business_payload(engine)

    candidates, failures = loaded
    new_engine = RerankEngine(
        config, candidates, failures, SnapshotStore(snap),
        DeterministicClock(), resume=True,
    )
    new_engine.handle_command(Command(1, "poll", {}))
    assert _business_payload(new_engine) == payload_before


def test_pending_then_commit_snapshot_is_recoverable(config, loaded, tmp_path):
    from rerank_core.clock import MonotonicClock
    snap = tmp_path / "snap_pending"
    candidates, failures = loaded
    cfg = dataclasses.replace(config, latency_budget_ms=0.0)
    engine = RerankEngine(
        cfg, candidates, failures, SnapshotStore(snap), MonotonicClock()
    )
    first = engine.handle_command(Command(0, "poll", {}))
    assert first.code == "E_INTERACT_BUDGET_PENDING"

    engine.config = dataclasses.replace(engine.config, latency_budget_ms=5000.0)
    drained = engine.drain_pending(Command(1, "poll", {}))
    assert drained.code == "OK"
    report = engine.build_report(high_tier=False)

    # 磁盘快照必须包含所有生效缓存键，且能被新引擎恢复
    store = SnapshotStore(snap)
    manifest = store.load_manifest()
    for request_id, key_map in manifest["stage_cache_keys"].items():
        assert store.has_artifact(key_map["normalize"])
        assert store.has_artifact(key_map["rerank"])
        for key in key_map["fusion"].values():
            assert store.has_artifact(key)

    candidates, failures = loaded
    resumed = RerankEngine(
        config, candidates, failures, SnapshotStore(snap),
        DeterministicClock(), resume=True,
    )
    resumed.handle_command(Command(2, "poll", {}))
    assert resumed.build_report(high_tier=False)["per_request"][0]["fusion_order"] == \
        report["per_request"][0]["fusion_order"]
