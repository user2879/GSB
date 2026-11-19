import pytest

from probegi.budget import plan_budget
from probegi.errors import BudgetDeniedError
from probegi.hashing import sha256_json

KW = dict(grid_n=32, order=3, n_directions=256, n_queries=100000,
          field_scale=1.0)


def test_degradation_deterministic():
    a = plan_budget(memory_budget_bytes=1024 * 1024, time_budget_ms=100.0, **KW)
    b = plan_budget(memory_budget_bytes=1024 * 1024, time_budget_ms=100.0, **KW)
    assert sha256_json(a) == sha256_json(b)
    assert a["queries_dropped"] == 0


def test_degradation_priority_order():
    plan = plan_budget(memory_budget_bytes=1024 * 1024, time_budget_ms=100.0, **KW)
    actions = [s["action"] for s in plan["degradation_steps"]]
    # Priority: reduce_order, then cluster_representatives, then directions.
    rank = {"reduce_order": 0, "cluster_representatives": 1,
            "reduce_directions": 2}
    assert [rank[a] for a in actions] == sorted(rank[a] for a in actions)
    assert "reduce_order" in actions and "cluster_representatives" in actions
    assert plan["resident_memory_bytes"] <= 1024 * 1024
    assert plan["est_batch_time_ms"] <= 100.0
    for step in plan["degradation_steps"]:
        assert step["error_bound_after"] >= step["error_bound_before"]


def test_generous_budget_no_degradation():
    plan = plan_budget(memory_budget_bytes=10 ** 9, time_budget_ms=1e9, **KW)
    assert plan["degradation_steps"] == []
    assert plan["final"] == {"order": 3, "clustered": False,
                             "n_directions": 256}


def test_impossible_budget_denied():
    with pytest.raises(BudgetDeniedError) as exc:
        plan_budget(memory_budget_bytes=1, time_budget_ms=1e9, **KW)
    assert exc.value.code == "E_BUDGET_DENIED"
