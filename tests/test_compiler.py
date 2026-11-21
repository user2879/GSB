import unittest

from configbench.aggregate import aggregate_slice, slice_digest
from configbench.compiler import build_node_slices, compile_version, propagate
from configbench.manifest import load_all, load_manifest, load_registry


class CompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.samples = load_all(
            "samples/registry.json", "samples/manifest.json", "samples/samples.json")

    def test_unchanged_artifact_byte_identical(self):
        v1, v2, v3 = self.manifest.versions
        a1, _ = compile_version(None, v1)
        a2, prop12 = compile_version(v1, v2)
        a3, prop23 = compile_version(v2, v3)
        # v1->v2 只有 base-rate 变化；threshold/composite 产物必须逐字节不变。
        self.assertEqual(prop12["direct_changed"], ["base-rate"])
        self.assertEqual(a1["threshold"].artifact_id, a2["threshold"].artifact_id)
        self.assertEqual(a1["threshold"].content, a2["threshold"].content)
        self.assertNotEqual(a1["base-rate"].artifact_id, a2["base-rate"].artifact_id)
        # composite 派生自 base-rate，必须被间接失效。
        self.assertIn("composite", prop12["invalidated"])
        # v2->v3 threshold 与 composite(系数/目标) 变化。
        self.assertEqual(prop23["direct_changed"], ["composite", "threshold"])
        self.assertEqual(a2["base-rate"].artifact_id, a3["base-rate"].artifact_id)
        # 传播步骤稳定且带触发源/受影响节点。
        self.assertTrue(all("trigger_source" in s and "affected_nodes" in s
                            for s in prop23["steps"]))

    def test_propagation_order_deterministic(self):
        v1, v2 = self.manifest.versions[0], self.manifest.versions[1]
        runs = [propagate(v1, v2)["invalidated"] for _ in range(5)]
        self.assertTrue(all(r == runs[0] for r in runs))
        # 拓扑序：被依赖项排在派生项之前。
        order = runs[0]
        self.assertLess(order.index("base-rate"), order.index("composite"))

    def test_slice_digest_stable(self):
        v1 = self.manifest.versions[0]
        slices = build_node_slices(v1, self.registry)
        d1 = slice_digest(1, slices["node-alpha"])
        d2 = slice_digest(1, tuple(reversed(slices["node-alpha"])))
        self.assertEqual(d1, d2)

    def test_aggregate_byte_identical_across_runs(self):
        v1 = self.manifest.versions[0]
        slices = build_node_slices(v1, self.registry)
        node = self.registry.nodes["node-alpha"]
        s1, h1 = aggregate_slice(node, 1, slices["node-alpha"], self.samples)
        s2, h2 = aggregate_slice(node, 1, slices["node-alpha"], self.samples)
        self.assertEqual(h1, h2)
        self.assertEqual(s1, s2)
        # 不同节点口径 scale 不同 -> 摘要不同但同节点可复现。
        node_b = self.registry.nodes["node-beta"]
        sb, hb = aggregate_slice(node_b, 1, slices["node-beta"], self.samples)
        self.assertNotEqual(h1, hb)
        self.assertEqual(sb["node_scale"], 4)

    def test_changed_value_changes_digest(self):
        v1, v2 = self.manifest.versions[0], self.manifest.versions[1]
        sl1 = build_node_slices(v1, self.registry)
        sl2 = build_node_slices(v2, self.registry)
        node = self.registry.nodes["node-alpha"]
        _, h1 = aggregate_slice(node, 1, sl1["node-alpha"], self.samples)
        _, h2 = aggregate_slice(node, 2, sl2["node-alpha"], self.samples)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
