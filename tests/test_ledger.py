import os
import shutil
import tempfile
import unittest

from cfgpipe.ledger import FORMAT_VERSION, Ledger


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ledger.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_commit_and_recovery_scan(self):
        led = Ledger(self.path, "digest-a")
        led.begin_batch("b1", ["n1", "n2"], 100)
        led.commit_batch("b1", 0, 0, 200,
                         [{"node_id": "n1", "source": "new", "digest": "d1",
                           "config_version": 1, "basis_version": None,
                           "elapsed_ms": 1, "attempts": 1, "wait_ms": 0}],
                         [], {"alpha": (1, 0, 0, 1)})
        led.close()
        led2 = Ledger(self.path, "digest-a")
        info = led2.recovery_scan()
        self.assertFalse(info.replaced)
        self.assertEqual(info.cursor, 0)
        self.assertEqual(info.unknown_nodes, [])
        self.assertEqual(led2.error_bounds_of("b1"), {"alpha": (1, 0, 0, 1)})
        led2.close()

    def test_uncommitted_batch_marks_unknown(self):
        led = Ledger(self.path, "digest-a")
        led.begin_batch("b_crash", ["n2", "n3"], 100)
        led.close()  # 模拟强制终止后留下的在途标记
        led2 = Ledger(self.path, "digest-a")
        info = led2.recovery_scan()
        self.assertEqual(info.unknown_nodes, ["n2", "n3"])
        self.assertEqual(info.inflight_batches, ["b_crash"])
        self.assertEqual(info.cursor, -1)
        led2.close()

    def test_digest_mismatch_replaces_wholesale(self):
        led = Ledger(self.path, "digest-a")
        led.commit_batch("b1", 0, 0, 1, [], [], {})
        led.close()
        led2 = Ledger(self.path, "digest-b")
        self.assertTrue(led2.replaced)
        info = led2.recovery_scan()
        self.assertEqual(info.cursor, -1)
        self.assertEqual(info.batches, [])
        led2.close()

    def test_format_version_mismatch_replaces(self):
        led = Ledger(self.path, "digest-a")
        led.db.execute("UPDATE meta SET value = ? WHERE key = 'format_version'",
                       (str(FORMAT_VERSION + 1),))
        led.db.commit()
        led.close()
        led2 = Ledger(self.path, "digest-a")
        self.assertTrue(led2.replaced)
        led2.close()


if __name__ == "__main__":
    unittest.main()
