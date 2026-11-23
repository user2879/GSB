import pytest

from shardbench.config_loader import load_config
from shardbench.engine import WorkbenchEngine
from shardbench.invalidation import invalidated_subset, topo_order

CFG = "samples/sharding_config.json"


def test_topo_order_is_stable_and_respects_dependencies():
    cfg = load_config(CFG)
    order = topo_order(cfg.units)
    assert order == sorted(order, key=lambda u: (
        0 if u != "order_user_join" else 1, u))
    assert order.index("order_user_join") > order.index("orders_by_user")
    assert order.index("order_user_join") > order.index("users_by_id")


def test_invalidation_propagates_to_indirect_downstream():
    cfg = load_config(CFG)
    invalidated, steps = invalidated_subset(cfg, ["orders_by_user"])
    assert invalidated == ["orders_by_user", "order_user_join"]
    assert steps == [{
        "from": "orders_by_user",
        "to": "order_user_join",
        "reason": "order_user_join depends_on orders_by_user; upstream result changed",
    }]


def test_subset_recompute_keeps_unaffected_byte_identical(tmp_path):
    eng = WorkbenchEngine(str(tmp_path))
    try:
        eng.load_config(CFG)
        preds = {
            "orders_by_user": [{"eq": ["user_id", 42]}],
            "orders_by_amount": [{"eq": ["amount", 500]}],
            "users_by_id": [{"eq": ["user_id", 9]}],
            "order_user_join": [{"eq": ["user_id", 42]}],
        }
        before = {}
        for unit, plist in preds.items():
            for p in plist:
                before.setdefault(unit, []).append(eng.compile(unit, p).plan_digest)

        result = eng.invalidate(["users_by_id"], preds)
        assert result["invalidated"] == ["users_by_id", "order_user_join"]
        assert result["unaffected_byte_identical"] is True
        after_amount = [pl.plan_digest for pl in
                        eng.plan_cache["orders_by_amount"].values()]
        assert after_amount == before["orders_by_amount"]
    finally:
        eng.shutdown()
