import tempfile
import unittest
from pathlib import Path

from segpos.fingerprints import stable_sort_key
from tests._fixtures import make_pipeline, write_workspace


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_worker_counts_produce_identical_outputs(self):
        outputs = {}
        for workers in (1, 2, 4, 8):
            paths = write_workspace(
                self.root / f"ws{workers}", workers=workers, interval=3
            )
            pipe = make_pipeline(paths, self.root / f"snap{workers}")
            pipe.run()
            ordered = [
                pipe.results_by_id()[sid]
                for sid in sorted(pipe.results_by_id())
            ]
            outputs[workers] = ordered
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(outputs[2], outputs[4])
        self.assertEqual(outputs[4], outputs[8])

    def test_stable_sort_key_derives_from_id_and_stage(self):
        self.assertEqual(
            stable_sort_key("S2", "tagging"),
            stable_sort_key("S2", "tagging"),
        )
        self.assertNotEqual(
            stable_sort_key("S1", "tagging"),
            stable_sort_key("S2", "tagging"),
        )


class ExceptionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = write_workspace(self.root / "ws", workers=3, interval=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stage_exception_isolates_sentence_and_keeps_others(self):
        pipe = make_pipeline(self.paths, self.root / "snap")
        pipe.fault_injection = {"S3": "tagging"}
        outcome = pipe.run()
        failed_ids = [item["sentence_id"] for item in outcome.report["failures"]]
        self.assertIn("S3", failed_ids)
        failure = next(f for f in outcome.report["failures"] if f["sentence_id"] == "S3")
        self.assertEqual(failure["stage"], "tagging")
        self.assertEqual(failure["code"], "E_STAGE_FAILED")
        self.assertEqual(failure["exception_type"], "InjectedStageError")
        self.assertNotIn("S3", pipe.results_by_id())
        self.assertEqual(
            sorted(pipe.results_by_id()), ["S1", "S2", "S4", "S5"]
        )

        pipe.fault_injection = {}
        outcome2 = pipe.run()
        self.assertNotIn("S3", pipe.results_by_id())
        self.assertTrue(
            any(f["sentence_id"] == "S3" for f in outcome2.report["failures"])
        )


if __name__ == "__main__":
    unittest.main()
