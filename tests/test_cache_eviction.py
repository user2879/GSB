import os
import shutil
import tempfile
import unittest

from layerbench.app import Workbench
from layerbench.cache import CacheIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "samples", "workspace")
TINY = os.path.join(ROOT, "samples", "manifests", "tiny_cache_manifest.json")


class CacheEvictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lb-cache-")
        self.workdir = os.path.join(self.tmp, "wd")
        self.bench = Workbench(self.workdir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_capacity_enforced_with_stable_eviction(self):
        result = self.bench.build(TINY, source_root=SRC)
        index = CacheIndex.load(self.bench.cache_dir, capacity=2,
                                clock_ms=lambda: 0, now_ms=lambda: 0)
        self.assertLessEqual(index.size(), 2)
        evictions = result["report"]["evictions"]
        self.assertTrue(evictions)
        for ev in evictions:
            self.assertEqual(ev["reason"], "capacity_lru_evict")
            self.assertIn("min(last_used_seq", ev["rule"])

    def test_eviction_does_not_depend_on_dict_order(self):
        # 连续两次独立构建，淘汰出的层摘要集合应一致（同输入同容量）
        wd1 = os.path.join(self.tmp, "a")
        wd2 = os.path.join(self.tmp, "b")
        r1 = Workbench(wd1).build(TINY, source_root=SRC)
        r2 = Workbench(wd2).build(TINY, source_root=SRC)
        e1 = sorted(e["layer_digest"] for e in r1["report"]["evictions"])
        e2 = sorted(e["layer_digest"] for e in r2["report"]["evictions"])
        self.assertEqual(e1, e2)


if __name__ == "__main__":
    unittest.main()
