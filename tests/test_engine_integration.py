import copy
import shutil
import tempfile
import unittest

import numpy as np

from opdiag.engine import Engine
from opdiag.errors import (
    E_INJ_IMPL_MISSING, E_INJ_NONFINITE, E_PERM_INVALID,
    E_RECOMPUTE_BUDGET_PENDING, E_UPSTREAM_FAILED,
)


def simple_chain():
    return {
        "nodes": [
            {"id": "A", "op": "gelu", "input_ref": "in0", "params": {}},
            {"id": "T", "op": "transpose", "perm": [1, 0]},
            {"id": "B", "op": "relu", "params": {"slope": 0.1}},
            {"id": "S", "op": "sigmoid", "input_ref": "in1",
             "params": {"scale": 1.0}},
        ],
        "edges": [["A", "T"], ["T", "B"]],
    }


def inputs():
    return {
        "in0": {"shape": [3, 4], "distribution": "uniform", "low": -2,
                "high": 2, "seed": 7},
        "in1": {"shape": [3, 4], "distribution": "normal", "seed": 8},
    }


def cfg(**kw):
    base = {"precision": "fp32", "bit_width": 16, "quant_mode": "uniform",
            "tolerance": 1e-5, "max_workers": 2, "element_budget": 1000,
            "snapshot_interval": 2, "enable_fusion": True,
            "output_dir": "out"}
    base.update(kw)
    return base


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="opdiag_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, chain=None, config=None, injections=None):
        eng = Engine(chain or simple_chain(), inputs(),
                     config or cfg(), injections or [],
                     base_dir=self.tmp)
        return eng, eng.run()

    def test_deterministic_twice_identical(self):
        e1, a1 = self._run()
        shutil.rmtree(self.tmp + "/out", ignore_errors=True)
        e2, a2 = self._run()
        for nid in a1.products:
            np.testing.assert_array_equal(
                a1.products[nid]["output"], a2.products[nid]["output"])
            np.testing.assert_array_equal(
                a1.products[nid]["quant"]["q"],
                a2.products[nid]["quant"]["q"])

    def test_precision_and_bitwidth_change_outputs(self):
        _, a32 = self._run(config=cfg(bit_width=16))
        _, a64 = self._run(config=cfg(precision="fp64", bit_width=8,
                                      quant_mode="pow2"))
        q32 = a32.products["A"]["quant"]["q"]
        q64 = a64.products["A"]["quant"]["q"]
        self.assertEqual(q64.dtype, np.int8)
        self.assertFalse(np.array_equal(q32.astype(np.int32),
                                        q64.astype(np.int32)))

    def test_cache_reuse_after_unchanged_rerun(self):
        e1, a1 = self._run()
        e2, a2 = self._run()
        self.assertEqual(set(a2.report["reuse_list"]), set(a1.products))
        self.assertEqual(a2.report["recompute_list"], [])
        # 缓存键稳定
        self.assertEqual(a1.products["A"]["cache_key"],
                         a2.products["A"]["cache_key"])

    def test_invalidation_propagates_only_downstream(self):
        e1, a1 = self._run()
        chain = simple_chain()
        next(n for n in chain["nodes"] if n["id"] == "A")["params"] = {}
        # 修改 A 无关参数无效，改为修改实际影响：换输入 seed 不在此测，
        # 这里修改 B 的 slope（B 是 A/T 的下游，S 独立）
        next(n for n in chain["nodes"] if n["id"] == "B")["params"
                                                           ]["slope"] = 0.3
        e2, a2 = self._run(chain=chain)
        recomputed = set(a2.report["recompute_list"])
        reused = set(a2.report["reuse_list"])
        self.assertIn("B", recomputed)
        # 独立节点 S 与上游 A/T 不应被 B 的参数编辑影响
        self.assertIn("S", reused)
        self.assertIn("A", reused)
        self.assertIn("T", reused)

    def test_budget_pending_then_drain(self):
        _, a1 = self._run(config=cfg(element_budget=20))
        self.assertTrue(a1.report["pending_list"])
        self.assertIn(E_RECOMPUTE_BUDGET_PENDING, a1.report["failure_codes"])
        # 续跑直到清空
        report = a1.report
        for _ in range(20):
            eng = Engine(simple_chain(), inputs(), cfg(element_budget=20),
                         [], base_dir=self.tmp)
            report = eng.run().report
            if not report["pending_list"]:
                break
        self.assertEqual(report["pending_list"], [])
        self.assertNotIn(E_RECOMPUTE_BUDGET_PENDING, report["failure_codes"])

    def test_snapshot_resume_matches_full_rebuild(self):
        # 小预算多次续跑
        report = None
        for _ in range(20):
            eng = Engine(simple_chain(), inputs(), cfg(element_budget=15),
                         [], base_dir=self.tmp)
            art = eng.run()
            report = art.report
            if not report["pending_list"]:
                break
        inc_outputs = {n: np.load(
            eng.store.product_path(n))["output"]
            for n in report["nodes"] and
            [x["node_id"] for x in report["nodes"]]}
        # 全量重建到独立目录
        full_dir = tempfile.mkdtemp(prefix="opdiag_full_")
        try:
            feng = Engine(simple_chain(), inputs(),
                          cfg(element_budget=10 ** 9), [],
                          base_dir=full_dir)
            fart = feng.run()
            for nid, p in fart.products.items():
                np.testing.assert_array_equal(
                    inc_outputs[nid], p["output"])
        finally:
            shutil.rmtree(full_dir, ignore_errors=True)

    def test_injection_nonfinite_isolation(self):
        inj = [{"id": "I1", "node_id": "A", "stage": "compute",
                "fault": "nonfinite"}]
        _, art = self._run(injections=inj)
        codes = {h["code"] for h in art.report["injection_hits"]}
        self.assertIn(E_INJ_NONFINITE, codes)
        # A 失败 -> T、B 级联失败；独立 S 仍成功
        status = {n["node_id"]: n["status"] for n in art.report["nodes"]}
        self.assertEqual(status["S"], "recomputed")
        self.assertIn(E_UPSTREAM_FAILED, art.report["failure_codes"])

    def test_injection_impl_missing(self):
        inj = [{"id": "I2", "node_id": "B", "stage": "compute",
                "fault": "impl_version_missing"}]
        _, art = self._run(injections=inj)
        self.assertIn(E_INJ_IMPL_MISSING,
                      {h["code"] for h in art.report["injection_hits"]})

    def test_injection_duplicate_perm(self):
        inj = [{"id": "I3", "node_id": "T", "stage": "validate",
                "fault": "duplicate_perm"}]
        _, art = self._run(injections=inj)
        hit = next(h for h in art.report["injection_hits"]
                   if h["injection_id"] == "I3")
        self.assertEqual(hit["code"], E_PERM_INVALID)

    def test_fusion_diagnostic_recorded(self):
        _, art = self._run()
        diag = art.report["fusion_diagnostics"]
        self.assertEqual(len(diag), 1)
        d = diag[0]
        self.assertEqual(d["activation"], "A")
        self.assertEqual(d["transpose"], "T")
        self.assertIn("unfused_path", d)
        self.assertIn("fused_path", d)
        self.assertFalse(d["over_tolerance"])

    def test_inplace_safe_leaf_does_not_corrupt_inputs(self):
        chain = simple_chain()
        # 独立叶子 S 标记原地写回（无消费者、非外部输入引用场景）
        next(n for n in chain["nodes"] if n["id"] == "S")["inplace"] = True
        eng = Engine(chain, inputs(), cfg(), [], base_dir=self.tmp)
        # 先记录外部输入快照
        eng._materialize_inputs()
        before = eng.inputs["in1"].copy()
        art = eng.run()
        np.testing.assert_array_equal(eng.inputs["in1"], before)
        self.assertNotIn("E_INPLACE_CONFLICT", art.report["failure_codes"])

    def test_inplace_uses_same_buffer_and_matches_outofplace(self):
        import numpy as np
        from opdiag import ops
        x = np.linspace(-2, 2, 12, dtype=np.float32).reshape(3, 4)
        buf = x.copy()
        y_ip, r_ip = ops.activate_inplace("relu", buf, {"slope": 0.25},
                                          "fp32")
        y_ref, r_ref = ops.run_activation("relu", x, {"slope": 0.25},
                                          "fp32")
        self.assertTrue(np.shares_memory(y_ip, buf))
        np.testing.assert_array_equal(y_ip, y_ref)
        np.testing.assert_array_equal(r_ip, r_ref)

    def test_precision_switch_invalidates_all_nodes(self):
        _, a1 = self._run(config=cfg(precision="fp32", bit_width=16))
        _, a2 = self._run(config=cfg(precision="fp64", bit_width=8,
                                    quant_mode="pow2"))
        # 缓存键含精度档位/位宽/量化档位，全部节点必须重算、零复用
        self.assertEqual(set(a2.report["recompute_list"]),
                         set(a1.products.keys()))
        self.assertEqual(a2.report["reuse_list"], [])
        # 整数 dtype 真实改变
        self.assertEqual(a2.products["A"]["quant"]["q"].dtype, np.int8)

    def test_order_id_and_round_counts_persisted(self):
        _, art = self._run()
        p_t = art.products["T"]
        self.assertEqual(p_t["order_ids"].shape, p_t["output"].shape)
        # 顺序标识为输出字典序 0..N-1
        n = p_t["order_ids"].size
        np.testing.assert_array_equal(
            p_t["order_ids"].reshape(-1), np.arange(n))
        self.assertTrue(np.all(p_t["round_counts"] >= 0))

    def test_max_workers_not_exceeded(self):
        _, art = self._run(config=cfg(max_workers=2))
        self.assertLessEqual(art.report["observed_max_concurrency"], 2)


if __name__ == "__main__":
    unittest.main()
