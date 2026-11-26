from __future__ import annotations

import copy

from workbench.engine import WorkbenchEngine
from workbench.invalidation import plan_evolution
from workbench.timeutil import ScriptedClock
from workbench.water import ErrorAttribution, WaterModel, choose_by_water_and_order
from workbench.testkit import write_test_spec


def test_water_quantization_error_bounds_and_tie_rule():
    model = WaterModel(2)
    attribution = ErrorAttribution(
        version_edges=1,
        version_error_fp=1,
        dependency_uncertain=1,
        dependency_error_fp=1,
        lease_boundaries=1,
        lease_error_fp=1,
    )
    record = model.shard_error_record(
        shard_id="s1",
        resource_units=2,
        global_budget=6,
        attribution=attribution,
    )
    assert record["scale_factor"] == 100
    assert record["quantization_truncation_error_fp"] == 1
    assert record["version_error_fp"] == 1
    assert record["dependency_error_fp"] == 1
    assert record["lease_error_fp"] == 1
    assert record["synthetic_error_bound_fp"] >= 4

    node = model.node_water(
        active_shards=1,
        concurrency_quota=2,
        active_leases=1,
        lease_limit=2,
        node_resource_units=2,
        global_budget=6,
        attribution=attribution,
    )
    assert node["comparison_error_bound_fp"] >= 5
    summary = model.global_error_summary([record, record])
    assert summary["shard_count"] == 2
    assert summary["global_aggregate_error_bound_fp"] >= 2

    candidates = [
        {"shard_id": "z", "water_fp": 10, "comparison_error_bound_fp": 5},
        {"shard_id": "a", "water_fp": 11, "comparison_error_bound_fp": 5},
    ]
    winner = choose_by_water_and_order(
        candidates, topo_index={"a": 0, "z": 1}, water_model=model
    )
    assert winner["shard_id"] == "a"
    assert winner["error_affected"] is True


def test_frozen_leaf_is_never_admitted(tmp_path):
    raw = write_test_spec(tmp_path / "base.json").raw
    next(s for s in raw["shards"] if s["shard_id"] == "s6")["frozen"] = True
    spec = write_test_spec(tmp_path / "frozen.json", raw)
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock(), parallel=False)
    try:
        result = engine.run_until_complete()
        assert set(result["completed"]) == {"s1", "s2", "s3", "s4", "s5"}
        assert engine.ledger.get_lease("lease-s6") is None
        assert engine.admission_block_reason("s6") == "frozen"
        assert all(d["shard_id"] != "s6" or not d["admitted"] for d in engine.decisions)
        assert engine.ledger.active_resource_sum() == 0
    finally:
        engine.close()


def test_quota_evolution_propagates_node_references(tmp_path):
    old_spec = write_test_spec(tmp_path / "old.json")
    raw = copy.deepcopy(old_spec.raw)
    for node in raw["nodes"]:
        if node["node_id"] == "n2":
            node["concurrency_quota"] = 1
    new_spec = write_test_spec(tmp_path / "new.json", raw)
    plan = plan_evolution(old_spec, new_spec)
    assert plan["invalid_shards"] == ["s2", "s4", "s5", "s6"]
    assert plan["unchanged_shards"] == ["s1", "s3"]
    quota_step = next(
        step
        for step in plan["propagation_steps"]
        if step["shard_id"] == "s2" and step["stage"] == "direct"
    )
    assert any("concurrency_quota" in reason["basis"] for reason in quota_step["reasons"])
