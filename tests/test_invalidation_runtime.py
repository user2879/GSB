from __future__ import annotations

import copy

from gsb.invalidation import plan_invalidation
from gsb.ledger import compare_record_sets
from gsb.runtime import ClusterRuntime
from gsb.spec import parse_spec
from helpers import base_raw_spec


def test_quota_change_invalidates_dependency_closure_only() -> None:
    original = parse_spec(base_raw_spec())
    raw = base_raw_spec()
    raw["nodes"][0]["quota"] = 9
    evolved = parse_spec(raw)
    plan = plan_invalidation(original, evolved)
    assert plan.invalid_shards == ("raw-a", "join-ab", "report")
    assert plan.preserved_shards == ("raw-b", "raw-d")
    assert plan.steps[0].direct_reasons == ("node_quota_changed",)
    assert plan.steps[1].propagation_basis == ("upstream_invalid:raw-a",)
    assert plan.steps[2].propagation_basis == ("upstream_invalid:join-ab",)


def test_capacity_version_change_invalidates_everything() -> None:
    original = parse_spec(base_raw_spec())
    raw = base_raw_spec()
    raw["capacity_version"] = "cap-next"
    evolved = parse_spec(raw)
    plan = plan_invalidation(original, evolved)
    assert plan.preserved_shards == ()
    assert plan.invalid_shards == evolved.topological_order()
    assert set(plan.global_changes) == {"capacity_version"}
    assert all(step.kind == "direct" for step in plan.steps)


def test_incremental_rerun_matches_full_rebuild_and_preserves_bytes(tmp_path) -> None:
    raw = base_raw_spec(global_budget=6)
    original = parse_spec(raw)
    evolved_raw = copy.deepcopy(raw)
    evolved_raw["nodes"][0]["quota"] = 9
    evolved = parse_spec(evolved_raw)

    from gsb.ledger import Ledger

    initial_ledger = Ledger(tmp_path / "initial.db", session_version=1)
    with ClusterRuntime(original, ledger=initial_ledger, session_version=1) as runtime:
        initial = runtime.execute()
    initial_ledger.close()

    plan = plan_invalidation(original, evolved)
    incremental_ledger = Ledger(tmp_path / "incremental.db", session_version=2)
    with ClusterRuntime(evolved, ledger=incremental_ledger, session_version=2) as runtime:
        incremental = runtime.recompute_subset(initial.records, plan)
    incremental_ledger.close()
    assert incremental.preserved["preserved"] is True

    full_ledger = Ledger(tmp_path / "full.db", session_version=2)
    with ClusterRuntime(evolved, ledger=full_ledger, session_version=2) as runtime:
        full = runtime.execute()
    full_ledger.close()
    comparison = compare_record_sets(incremental.records, full.records)
    assert comparison["consistent"] is True
    assert comparison["shards_compared"] == len(evolved.shards)

