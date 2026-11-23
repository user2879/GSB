import random
import unittest

from cfgpipe.graph import (compute_invalidation, node_subscription_map,
                           topo_order)
from cfgpipe.spec import (apply_changeset, initial_state,
                          validate_changeset, validate_declaration)

from tests.test_spec import base_decl


def make_state(decl):
    return initial_state(decl)


class TestInvalidation(unittest.TestCase):
    def setUp(self):
        self.decl = validate_declaration(base_decl())

    def test_closure_and_topo_order(self):
        cs = validate_changeset(
            {"version": 9, "changes": [{"key": "alpha", "value": 2.0}]},
            self.decl, 3)
        state = make_state(self.decl)
        nxt = apply_changeset(state, cs)
        inv = compute_invalidation(state, cs, node_subscription_map(nxt))
        self.assertEqual(list(inv.items), ["alpha", "beta"])
        self.assertEqual(list(inv.nodes), ["n1"])
        reasons = {s.name: s.reason for s in inv.steps}
        self.assertIn("value", reasons["alpha"])
        self.assertIn("alpha", reasons["beta"])
        self.assertIn("beta", reasons["n1"])

    def test_subscription_change_affects_nodes_not_items(self):
        cs = validate_changeset(
            {"version": 9,
             "changes": [{"key": "gamma", "subscribers_add": ["n1"]}]},
            self.decl, 3)
        state = make_state(self.decl)
        nxt = apply_changeset(state, cs)
        inv = compute_invalidation(state, cs, node_subscription_map(nxt))
        self.assertEqual(list(inv.items), [])
        self.assertEqual(list(inv.nodes), ["n1"])

    def test_determinism_under_declaration_permutation(self):
        cs_raw = {"version": 9, "changes": [{"key": "alpha", "value": 2.0}]}
        outputs = []
        for seed in range(8):
            raw = base_decl()
            rng = random.Random(seed)
            rng.shuffle(raw["items"])
            # 版本号单调约束按声明顺序，重排后需重排版本以通过校验
            for idx, item in enumerate(raw["items"], start=1):
                item["version"] = idx
            decl = validate_declaration(raw)
            cs = validate_changeset(cs_raw, decl, 3)
            state = make_state(decl)
            nxt = apply_changeset(state, cs)
            inv = compute_invalidation(state, cs, node_subscription_map(nxt))
            outputs.append(inv.canonical())
        self.assertEqual(len(set(outputs)), 1)

    def test_topo_order_stable(self):
        derived = {"a": "", "b": "a", "c": "a", "d": "b"}
        self.assertEqual(topo_order(["d", "c", "b", "a"], derived),
                         ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
