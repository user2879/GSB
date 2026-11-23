from shardbench.algorithms import (HashRing, route_consistent_hash, route_mod,
                                   route_range, slot_to_db_table)
from shardbench.config_loader import load_config
from shardbench.planner import compile_plan

CFG = "samples/sharding_config.json"


def test_mod_routing_deterministic_and_in_range():
    cfg = load_config(CFG)
    unit = cfg.units["orders_by_user"]
    hits1 = route_mod(unit, [(42,)])
    hits2 = route_mod(unit, [(42,)])
    assert hits1 == hits2
    for h in hits1:
        assert 0 <= h.db < unit.physical_dbs
        assert 0 <= h.table < unit.tables_per_db
        assert "mod:" in h.reason


def test_range_boundary_ownership_left_closed():
    cfg = load_config(CFG)
    unit = cfg.units["orders_by_amount"]
    # boundaries [100, 500, 1000], left_closed: 500 belongs to bucket [500,1000)
    low = route_range(unit, [(499,)], {})[0]
    boundary = route_range(unit, [(500,)], {})[0]
    assert (boundary.db, boundary.table) != (low.db, low.table)
    assert slot_to_db_table(2, unit.tables_per_db) == (boundary.db, boundary.table)


def test_range_interval_covers_multiple_buckets():
    cfg = load_config(CFG)
    unit = cfg.units["orders_by_amount"]
    hits = route_range(unit, [], {"amount": (150, 800)})
    slots = sorted(h.db * unit.tables_per_db + h.table for h in hits)
    assert slots == [1, 2]  # buckets [100,500) and [500,1000)


def test_consistent_hash_deterministic_and_wraps():
    cfg = load_config(CFG)
    unit = cfg.units["users_by_id"]
    ring = HashRing(unit, ring_version=1)
    assert ring.node_count() == unit.total_slots * 8
    s1 = ring.lookup_slot((9,))
    assert ring.lookup_slot((9,)) == s1
    assert 0 <= s1 < unit.total_slots
    hits = route_consistent_hash(unit, ring, [(3,), (9,), (27,)])
    assert all("hash_ring:" in h.reason for h in hits)


def test_plan_identical_regardless_of_input_order():
    cfg = load_config(CFG)
    p1 = {"and": [{"in": ["user_id", [7, 42, 1001]]}, {"eq": ["status", "paid"]}]}
    p2 = {"and": [{"eq": ["status", "PAID"]}, {"in": ["user_id", [1001, 42, 7]]}]}
    plan1 = compile_plan(cfg, "orders_by_user", p1)
    plan2 = compile_plan(cfg, "orders_by_user", p2)
    assert plan1.plan_digest == plan2.plan_digest
    assert plan1.targets == plan2.targets
    assert plan1.reasons == plan2.reasons
    assert plan1.rule_version == 1 and plan1.ring_version == 1


def test_full_scan_when_shard_key_not_covered():
    cfg = load_config(CFG)
    plan = compile_plan(cfg, "orders_by_user", {"eq": ["status", "paid"]})
    assert len(plan.targets) == cfg.units["orders_by_user"].total_slots
    assert all(r.startswith("full_scan:") for r in plan.reasons)
