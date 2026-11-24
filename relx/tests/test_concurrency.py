import unittest

from relx.pipeline import compare_artifacts, full_rebuild


class ConcurrencyEquivalenceTest(unittest.TestCase):
    def test_serial_and_pool_artifacts_identical(self):
        _engine_serial, serial_report = full_rebuild("data", thread_count=1)
        _engine_pool, pool_report = full_rebuild("data", thread_count=8)
        identical, left, right = compare_artifacts(
            serial_report.to_dict(), pool_report.to_dict()
        )
        self.assertTrue(identical)

    def test_ledger_deduction_order_identical_across_workers(self):
        _s, serial = full_rebuild("data", thread_count=1)
        _p, pool = full_rebuild("data", thread_count=8)
        serial_rows = [(row.sentence_id, row.stage, row.points, row.batch_no)
                       for row in serial.ledger]
        pool_rows = [(row.sentence_id, row.stage, row.points, row.batch_no)
                     for row in pool.ledger]
        self.assertEqual(serial_rows, pool_rows)
        self.assertTrue(serial.ledger_consistent)
        self.assertTrue(pool.ledger_consistent)

    def test_pool_size_never_exceeds_configured_cap(self):
        import relx.engine as engine_mod
        observed = []
        original = engine_mod.ThreadPoolExecutor

        class SpyPool(original):
            def __init__(self, max_workers=None, *args, **kwargs):
                observed.append(max_workers)
                super().__init__(max_workers, *args, **kwargs)

        engine_mod.ThreadPoolExecutor = SpyPool
        try:
            from relx.pipeline import build_runtime
            state, _ = build_runtime("data")
            object.__setattr__(state.config, "thread_count", 4)
            engine_mod.ExtractionEngine(state).run()
        finally:
            engine_mod.ThreadPoolExecutor = original
        self.assertTrue(observed)
        self.assertTrue(all(size <= 4 for size in observed))

    def test_failure_order_stable(self):
        _s, serial = full_rebuild("data", thread_count=1)
        _p, pool = full_rebuild("data", thread_count=8)
        key = lambda items: [(f.code, f.sentence_id, f.stage) for f in items]
        self.assertEqual(key(serial.failures), key(pool.failures))


if __name__ == "__main__":
    unittest.main()
