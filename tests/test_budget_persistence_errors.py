import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tactical_pathbench.budget import Budget, LRUCache, Scheduler
from tactical_pathbench.errors import BudgetDeniedError, DependencyStaleError
from tactical_pathbench.persistence import PersistenceStore
from tactical_pathbench.workbench import PathfindingWorkbench, build_workbench


class BudgetTests(unittest.TestCase):
    def test_invalid_budget_rejected(self):
        with self.assertRaises(BudgetDeniedError):
            Scheduler(Budget(thread_budget=0))

    def test_lru_eviction_and_degradation(self):
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        self.assertEqual(cache.evictions, ["a"])
        scheduler = Scheduler(Budget(thread_budget=1, memory_budget=1))
        evicted = scheduler.enforce_memory(cache)
        self.assertTrue(evicted)
        self.assertTrue(any(d.reason == "memory_budget_exceeded" for d in scheduler.degradation_log))

    def test_parallel_and_serial_results_identical(self):
        parallel = build_workbench(seed=3, budget=Budget(thread_budget=4))
        serial = build_workbench(seed=3, budget=Budget(thread_budget=1))
        rp = parallel.solve_batch()
        rs = serial.solve_batch()
        self.assertEqual(set(rp), set(rs))
        for key in rp:
            self.assertEqual(rp[key].path_hash(), rs[key].path_hash())


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.store = PersistenceStore(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_commit_recover_orphan_and_restore(self):
        wb = build_workbench(seed=13, budget=Budget(thread_budget=2, memory_budget=8))
        wb.solve_batch()
        wb.modify_terrain((0, 5), 8)
        wb.solve_batch()
        wb.persist(self.store)
        self.store.write_loose_cache("loose.json", {"x": 1})
        self.store.remove_exit_report()
        self.store.tmp_path.write_text("partial", encoding="utf-8")
        recovery = self.store.recover()
        self.assertTrue(recovery.recovered)
        self.assertFalse(recovery.had_clean_exit_report)
        self.assertEqual(len(recovery.orphan_files), 2)
        restored = PathfindingWorkbench.restore(self.store)
        before = wb.solve_batch()
        after = restored.solve_batch()
        for key in before:
            self.assertEqual(before[key].path_hash(), after[key].path_hash())

    def test_tampered_manifest_detected(self):
        wb = build_workbench(seed=14)
        wb.persist(self.store)
        path = self.store.manifest_path
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["payload"]["map_version"] = 99999
        path.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        with self.assertRaises(DependencyStaleError):
            self.store.load_manifest()

    def test_clean_exit_report_requires_manifest(self):
        fresh = PersistenceStore(Path(tempfile.mkdtemp()))
        with self.assertRaises(DependencyStaleError):
            fresh.write_exit_report({"status": "clean"})
        shutil.rmtree(fresh.workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
