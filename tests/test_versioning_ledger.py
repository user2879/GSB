import json

import pytest

from shardbench.engine import WorkbenchEngine

CFG_V1 = "samples/sharding_config.json"
CFG_V2 = "samples/sharding_config_v2.json"
PRED = {"eq": ["user_id", 42]}


@pytest.fixture
def engine(tmp_path):
    eng = WorkbenchEngine(str(tmp_path), global_limit=2, per_db_limit=1,
                          lease_ttl_ms=5_000)
    yield eng
    eng.shutdown()


def test_probe_commits_batch_with_lease_and_reasons(engine):
    engine.load_config(CFG_V1)
    plan = engine.compile("orders_by_user", PRED)
    batch_id, records = engine.probe([plan])
    assert batch_id == 1
    tasks = engine.ledger.current_tasks()
    assert len(tasks) == len(plan.targets)
    for row in tasks:
        assert row["rule_version"] == 1 and row["ring_version"] == 1
        assert row["lease"]["lease_id"].startswith("lease-")
        assert row["lease"]["granted_ms"] <= row["lease"]["expires_ms"]
        assert row["reasons"], "each hit must carry its derivation"
        assert row["plan_digest"] == plan.plan_digest
    summary = engine.ledger.summary()
    assert summary["tasks"] == len(plan.targets)


def test_rule_evolution_marks_old_rows_stale_and_replaces_view(engine):
    engine.load_config(CFG_V1)
    plan_v1 = engine.compile("orders_by_user", PRED)
    engine.probe([plan_v1])
    assert engine.ledger.summary()["tasks"] == len(plan_v1.targets)

    engine.load_config(CFG_V2)  # rule version 2, mod salt changed
    assert engine.versions.current().rule_version == 2
    assert engine.ledger.summary()["tasks"] == 0  # old conclusions excluded
    assert engine.ledger.current_tasks() == []
    assert engine.last_diff["upgraded"] == ["orders_by_user"]
    assert engine.last_diff["added"] == [] and engine.last_diff["removed"] == []
    # invalidated set must contain the changed unit and its downstream dependent
    assert "orders_by_user" in engine.last_invalidation["invalidated"]
    assert "order_user_join" in engine.last_invalidation["invalidated"]
    assert "orders_by_amount" not in engine.last_invalidation["invalidated"]

    plan_v2 = engine.compile("orders_by_user", PRED)
    assert plan_v2.rule_version == 2
    engine.probe([plan_v2])
    rows = engine.ledger.current_tasks()
    assert {r["rule_version"] for r in rows} == {2}
    assert engine.ledger.summary()["tasks"] == len(plan_v2.targets)


def test_plan_cache_cleared_wholesale_on_version_bump(engine):
    engine.load_config(CFG_V1)
    engine.compile("orders_by_user", PRED)
    assert engine.plan_cache
    engine.load_config(CFG_V2)
    assert engine.plan_cache == {}
