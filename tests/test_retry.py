import json
import os
import tempfile
import unittest

from configbench.manifest import load_manifest, load_registry
from configbench.platform import Platform
from configbench.samples import generate_samples  # noqa: F401 确认依赖可用
import itertools

_PORT = itertools.count(48300)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    return next(_PORT)


class RetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.samples_path = os.path.join(ROOT, "samples/samples.json")

    def _registry_manifest(self):
        port = _free_port()
        reg_path = os.path.join(self.tmp, "registry.json")
        with open(reg_path, "w") as f:
            json.dump({"nodes": [
                {"node_id": "node-alpha", "host": "127.0.0.1", "port": port,
                 "group_key": "region", "scale": 3, "fail_inject": "flaky"}]}, f)
        registry = load_registry(reg_path)
        m_path = os.path.join(self.tmp, "manifest.json")
        with open(m_path, "w") as f:
            json.dump({
                "latency_budget_ms": 500,
                "default_dispatch": {"timeout_ms": 2000, "retries": 2,
                                     "strategy": "retry_then_fail"},
                "versions": [{"version": 1, "items": {
                    "base-rate": {"value_type": "int", "value": 7,
                                  "targets": ["node-alpha"]}}}]}, f)
        from configbench.manifest import load_sample_decl
        manifest = load_manifest(m_path, registry)
        samples = load_sample_decl(self.samples_path)
        return registry, manifest, samples

    def test_retry_then_success_semantics_unchanged(self):
        registry, manifest, samples = self._registry_manifest()
        p = Platform(registry, manifest, samples,
                     os.path.join(self.tmp, "retry.db"), max_global=1,
                     per_node_quota=1)
        p.start_nodes()
        self.addCleanup(p.close)
        pub = p.publish_version(1)
        result = pub["results"][0]
        self.assertEqual(result.outcome, "SUCCESS")
        self.assertGreaterEqual(len(result.attempts), 2)
        self.assertEqual(result.attempts[0]["outcome"], "CONN_REFUSED")
        self.assertEqual(result.attempts[-1]["outcome"], "SUCCESS")
        # 尝试序号从 1 单调递增。
        self.assertEqual([a["attempt_no"] for a in result.attempts],
                         list(range(1, len(result.attempts) + 1)))
        self.assertEqual(pub["receipts"]["success"], 1)
        # 重试不改语义：权威重算与回执逐字节一致。
        self.assertTrue(p.compare_paths(1)["match"])


if __name__ == "__main__":
    unittest.main()
