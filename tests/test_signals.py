import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests._fixtures import write_workspace, make_pipeline


class SignalExitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = write_workspace(self.root / "ws", workers=2, interval=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_graceful_stop_commits_snapshot_and_is_resumable(self):
        snap = self.root / "snap"
        pipe = make_pipeline(self.paths, snap)

        original = pipe._process_sentence

        def slow_process(sentence_id, text):
            time.sleep(0.05)
            return original(sentence_id, text)

        pipe._process_sentence = slow_process

        def trigger_stop():
            time.sleep(0.08)
            pipe.request_stop()

        thread = threading.Thread(target=trigger_stop)
        thread.start()
        outcome = pipe.run()
        thread.join()

        self.assertTrue(outcome.stopped_early)
        manifest_path = snap / "manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["global_version"], 1)
        self.assertLess(len(manifest["completed_sentences"]), 5)

        resumed = make_pipeline(self.paths, self.root / "resume_snap")
        resumed.snapshot = pipe.snapshot
        resumed._process_sentence = original
        final_outcome = resumed.run()
        self.assertFalse(final_outcome.stopped_early)
        self.assertEqual(
            sorted(resumed.results_by_id()), ["S1", "S2", "S3", "S4", "S5"]
        )
        self.assertGreater(len(final_outcome.reused), 0)


if __name__ == "__main__":
    unittest.main()
