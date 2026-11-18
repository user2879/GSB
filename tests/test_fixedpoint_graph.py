"""要求3/4：定点量化、误差传播、DAG、失效传播、E_CYCLE/E_RANGE/E_REF。"""
import unittest

from ccdplatform.errors import CycleError, RangeError, ReferenceError
from ccdplatform.fixedpoint import (
    GridSpec,
    derive_quantized,
    propagated_error_bound,
    trunc_zero,
)
from ccdplatform.graph import DependencyGraph
from ccdplatform.model import ConfigItem, Derivation


def item(key, value, refs=None, targets=("n1",)):
    derivation = Derivation(tuple(refs)) if refs else None
    return ConfigItem(key=key, value=value, targets=tuple(targets), derivation=derivation)


class FixedPointTest(unittest.TestCase):
    def test_trunc_toward_zero_negative(self):
        self.assertEqual(trunc_zero(7, 2), 3)
        self.assertEqual(trunc_zero(-7, 2), -3)  # 向零而非向下取整(-4)

    def test_quantize_and_range(self):
        grid = GridSpec(exponent=3, int_bits=63)
        self.assertEqual(grid.quantize(1.5), 1500)
        small = GridSpec(exponent=3, int_bits=8)  # qmax=127
        with self.assertRaises(RangeError) as ctx:
            small.quantize(1.5, key="big")  # 1500 > 127
        self.assertEqual(ctx.exception.code, "E_RANGE")
        self.assertEqual(ctx.exception.details["key"], "big")

    def test_derive_dimension_and_truncation(self):
        grid = GridSpec(exponent=3)
        # q(alpha)=1500, coef 0.5 -> 500; q(y)=trunc(500*1500/1000)=750
        q = derive_quantized([("alpha", 500)], {"alpha": 1500}, grid.scale)
        self.assertEqual(q, 750)

    def test_propagated_error_bound_integer_grid(self):
        grid = GridSpec(exponent=3)
        # |coef|=500, upstream B=1, q(x)=1500, scale=1000
        bound, contrib = propagated_error_bound(
            [("alpha", 500)], {"alpha": 1}, {"alpha": 1500}, grid.scale
        )
        # 1 (rescale) + ceil(500*1/1000)=1 + ceil(1500/1000)=2 -> 4
        self.assertEqual(contrib["alpha"], 3)
        self.assertEqual(bound, 4)

    def test_scale_changes_interpretation(self):
        coarse = GridSpec(exponent=2)
        fine = GridSpec(exponent=4)
        self.assertEqual(coarse.quantize(1.5), 150)
        self.assertEqual(fine.quantize(1.5), 15000)
        # 同一物理值在更细格度下格点整数等比放大
        self.assertEqual(fine.quantize(1.5) / fine.scale, coarse.quantize(1.5) / coarse.scale)


class GraphTest(unittest.TestCase):
    def _items(self):
        return {
            "a": item("a", 1),
            "b": item("b", 2, refs=[("a", 1.0)]),
            "c": item("c", 3, refs=[("b", 1.0)]),
        }

    def test_topological_stable_order(self):
        g = DependencyGraph.from_items(self._items())
        order = g.topological_order()
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("b"), order.index("c"))

    def test_downstream_closure_indirect(self):
        g = DependencyGraph.from_items(self._items())
        affected = g.downstream_closure(["a"])
        self.assertEqual(affected, ["b", "c"])  # 稳定拓扑序，含间接下游 c

    def test_cycle_detection(self):
        cyclic = {
            "a": item("a", 1, refs=[("b", 1.0)]),
            "b": item("b", 2, refs=[("a", 1.0)]),
        }
        with self.assertRaises(CycleError) as ctx:
            DependencyGraph.from_items(cyclic)
        self.assertEqual(ctx.exception.code, "E_CYCLE")

    def test_missing_reference(self):
        bad = {"a": item("a", 1, refs=[("ghost", 1.0)])}
        with self.assertRaises(ReferenceError):
            DependencyGraph.from_items(bad)

    def test_delete_referenced_blocked(self):
        g = DependencyGraph.from_items(self._items())
        with self.assertRaises(ReferenceError) as ctx:
            g.assert_deletable(["a"], self._items())
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertIn("b", ctx.exception.details["blocked"]["a"])

    def test_reachability_matrix(self):
        g = DependencyGraph.from_items(self._items())
        keys, reach = g.reachability_matrix()
        ia, ic = keys.index("a"), keys.index("c")
        self.assertTrue(bool(reach[ic, ia]))  # c 间接可达 a


if __name__ == "__main__":
    unittest.main()
