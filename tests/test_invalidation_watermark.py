from pathlib import Path

from dispatch_tokens.engine import SchedulerEngine
from dispatch_tokens.invalidation import (propagate_invalidation,
                                          stable_topo_order)
from dispatch_tokens.spec import load_spec
from dispatch_tokens.watermark import (choose_candidate,
                                       comparison_error_bound,
                                       global_quota_error_bound,
                                       node_water_level)

SPEC_PATH = Path(__file__).resolve().parents[1] / "samples" / "demo_spec.json"


def test_stable_topo_order_respects_dependencies():
    spec = load_spec(SPEC_PATH)
    order = stable_topo_order(spec.shards)
    position = {sid: idx for idx, sid in enumerate(order)}
    for shard in spec.shards:
        for upstream in shard.upstream:
            assert position[upstream] < position[shard.shard_id]


def test_node_quota_change_propagates_downstream():
    spec = load_spec(SPEC_PATH)
    rev1 = spec.revisions[0]
    rev2 = spec.revisions[1]
    result = propagate_invalidation(spec, rev1, rev2,
                                    shard_nodes={"sh-0": "node-a"})
    assert result.directly_invalidated == ("sh-0",)
    # sh-0 的全部直接与间接下游都失效。
    assert set(result.invalidated) == {"sh-0", "sh-1", "sh-2", "sh-3",
                                       "sh-4"}
    downstream_steps = [s for s in result.steps if s.caused_by_shard]
    assert downstream_steps
    assert all(s.reason.startswith("downstream_of:")
               for s in downstream_steps)


def test_scale_change_invalidates_all_shards():
    spec = load_spec(SPEC_PATH)
    result = SchedulerEngine(spec).run()
    v3 = next(item for item in result.invalidations
              if item["revision_version"] == 3)
    assert set(v3["invalidated"]) == {s.shard_id for s in spec.shards}


def test_water_level_integer_only_and_error_bounds():
    water = node_water_level("n", balance=350, cap=1000, rate=10, scale=100)
    assert water.ratio_scaled == 35
    assert water.quantization_error == 1
    assert water.truncation == "TRUNC_TOWARD_ZERO"
    assert comparison_error_bound(100) == 2
    assert global_quota_error_bound([100, 100, 100]) == 3


def test_choose_candidate_uses_then_shard_id_tie_break():
    topo = {"s1": 0, "s2": 1}
    candidates = [
        {"shard_id": "s2", "node_id": "n1", "demand_scaled": 100,
         "priority": 0},
        {"shard_id": "s1", "node_id": "n2", "demand_scaled": 100,
         "priority": 5},
    ]
    # 两候选需求差落在误差界内：拓扑序优先。
    decision = choose_candidate(
        candidates, balances={"n1": 101, "n2": 100},
        caps={"n1": 500, "n2": 500}, scale=100, topo_index=topo)
    assert decision["winner_shard_id"] == "s1"
    assert decision["error_influenced"] is True
    assert decision["ordered_shard_ids"][0] == "s1"


def test_affordable_candidate_beats_unaffordable_regardless_of_gap():
    topo = {"s1": 0, "s2": 1}
    candidates = [
        {"shard_id": "s1", "node_id": "n1", "demand_scaled": 100,
         "priority": 0},
        {"shard_id": "s2", "node_id": "n2", "demand_scaled": 100,
         "priority": 0},
    ]
    decision = choose_candidate(
        candidates, balances={"n1": 5, "n2": 900},
        caps={"n1": 1000, "n2": 1000}, scale=100, topo_index=topo)
    # s1 虽拓扑更靠前但额度不足，可负担的 s2 胜出。
    assert decision["winner_shard_id"] == "s2"
    assert decision["error_influenced"] is False


def test_choose_candidate_clear_gap_not_error_influenced():
    topo = {"s1": 0, "s2": 1}
    candidates = [
        {"shard_id": "s1", "node_id": "n1", "demand_scaled": 10,
         "priority": 0},
        {"shard_id": "s2", "node_id": "n2", "demand_scaled": 10,
         "priority": 0},
    ]
    decision = choose_candidate(
        candidates, balances={"n1": 900, "n2": 20},
        caps={"n1": 1000, "n2": 1000}, scale=100, topo_index=topo)
    assert decision["winner_shard_id"] == "s1"
    assert decision["error_influenced"] is False
