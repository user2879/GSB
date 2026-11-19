import threading
import unittest

from cache_invalidation.engine import Engine
from cache_invalidation.model import EntryID
from cache_invalidation.store import Store
from tests.support import sample_graph


class ConcurrencyTest(unittest.TestCase):
    def test_concurrent_explicit_invalidation_each_entry_once(self):
        engine = Engine(sample_graph(), store=Store())
        targets = [
            EntryID("catalog", 100),
            EntryID("catalog", 101),
            EntryID("pricing", 200),
            EntryID("inventory", 300),
        ]
        barrier = threading.Barrier(len(targets) * 4)

        def worker(entry_id: EntryID, clock_ms: int) -> None:
            barrier.wait()
            engine.explicit_invalidate(entry_id, 1000)

        threads = []
        for repeat in range(4):
            for index, entry_id in enumerate(targets):
                thread = threading.Thread(
                    target=worker,
                    args=(entry_id, 1000))
                threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for entry_id in targets:
            self.assertLessEqual(engine.versions[entry_id], 1)
        self.assertEqual(
            engine.versions[EntryID("storefront", 400)], 1)
        self.assertEqual(engine.versions[EntryID("search", 500)], 1)
        total_invalidations = sum(engine.versions.values())
        self.assertEqual(total_invalidations, 6)

    def test_concurrent_batch_never_double_counts_same_entry(self):
        engine = Engine(sample_graph(), store=Store())
        barrier = threading.Barrier(3)

        def batch(clock_ms: int) -> None:
            barrier.wait()
            engine.submit_batch(
                explicit=[EntryID("catalog", 100)],
                reap_now_ms=1000,
                refresh_namespaces=["catalog"],
                now_ms=12000)

        threads = [
            threading.Thread(target=batch, args=(1000,))
            for index in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for entry_id, version in engine.versions.items():
            self.assertLessEqual(
                version, 1,
                msg=f"{entry_id} 在多轮中被重复计入: {version}")
        self.assertEqual(sum(engine.versions.values()), 6)
        for record in engine.records:
            self.assertEqual(
                len(record.affected),
                len(set(record.affected)))

    def test_clock_errors_under_concurrency_leave_state_unchanged(self):
        engine = Engine(sample_graph(), store=Store())
        engine.explicit_invalidate(EntryID("catalog", 101), 500)
        versions_before = dict(engine.versions)

        errors = []

        def rollback() -> None:
            try:
                engine.explicit_invalidate(EntryID("catalog", 100), 10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=rollback) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(errors), 8)
        self.assertEqual(engine.versions, versions_before)


if __name__ == "__main__":
    unittest.main()
