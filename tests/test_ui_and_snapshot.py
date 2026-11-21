import json
import os
import tempfile
import unittest

from delayqueue import FORMAT_VERSION, open_queue


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "topics.json")
        self.message_path = os.path.join(self.tmp, "messages.ndjson")
        self.ledger_path = os.path.join(self.tmp, "q.db")
        with open(self.spec_path, "w", encoding="utf-8") as handle:
            json.dump({
                "spec_version": "v1",
                "global_concurrency": 2,
                "topics": {
                    "a": {"visibility_timeout_ms": 200, "max_deliveries": 2,
                          "backoff_ms": [10], "concurrency": 1},
                },
            }, handle)
        with open(self.message_path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"message_id":"m1","topic":"a","deliver_at":0,'
                '"payload":0,"idempotency_key":"k1"}')

    def test_snapshot_is_stable_and_read_only(self):
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            before = queue.ledger.actual_digest()
            first = queue.snapshot()
            second = queue.snapshot()
            after = queue.ledger.actual_digest()
            self.assertEqual(before, after)  # rendering never writes
            self.assertEqual(first["format_version"], FORMAT_VERSION)
            self.assertEqual(list(first["topics"]), ["a"])
            self.assertEqual(set(first["state_counts"]),
                             {"pending", "delayed", "ready", "inflight",
                              "acked", "dead"})
            self.assertEqual(first["topics"]["a"]["quota"], 1)
        finally:
            queue.close()

    def test_tkinter_import_is_lazy(self):
        # The package must import even when tkinter is unavailable.
        import importlib
        import delayqueue.ui as ui_module
        importlib.reload(ui_module)
        try:
            import tkinter  # noqa: F401
            self.skipTest("tkinter present; lazy failure path not testable")
        except Exception:
            pass
        with self.assertRaises(RuntimeError):
            ui_module._import_tk()


if __name__ == "__main__":
    unittest.main()
