import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import engine
from dhj import join as jm
from dhj import snapshot as sm
from dhj.errors import SnapshotError
from dhj.shuffle import generation_fingerprint
from tools.experiment_lib import load_experiment_spec, stats_calculator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SnapshotFloatCrashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = os.path.join(ROOT, "data", "sample.json")
        cls.float_query = os.path.join(ROOT, "data", "query_float.json")
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _d(self, name):
        path = os.path.join(self.tmp, name)
        os.makedirs(path, exist_ok=True)
        return path

    def test_snapshot_match_and_replacement(self):
        spec = load_experiment_spec(self.sample)
        buckets = self._d("snap-buckets")
        results = self._d("snap-results")
        snaps = self._d("snap-dir")
        engine.generate_version(buckets, spec, 3, 4, 4)
        calc = stats_calculator(spec, 4)
        jm.run_join(buckets, results, spec, 4, 4, calc)
        snapshot, state = engine.persist_snapshot(
            spec, snaps, buckets, results, 4, 4
        )
        loaded = sm.load_snapshot(snaps)
        self.assertEqual(loaded["content_digest"], snapshot["content_digest"])
        outcome = engine.open_or_replace_session(
            spec, snaps, buckets, results, 4, 4
        )
        self.assertEqual(outcome["action"], "resumed")
        tampered = self._d("snap-tampered")
        engine.generate_version(tampered, spec, 3, 4, 3)
        replaced = engine.open_or_replace_session(
            spec, snaps, tampered, results, 4, 4
        )
        self.assertEqual(replaced["action"], "whole_state_replaced")
        self.assertTrue(os.path.isdir(tampered + ".replaced"))

    def test_snapshot_digest_tamper_detected(self):
        spec = load_experiment_spec(self.sample)
        buckets = self._d("digest-buckets")
        results = self._d("digest-results")
        snaps = self._d("digest-snaps")
        engine.generate_version(buckets, spec, 3, 4, 4)
        calc = stats_calculator(spec, 4)
        jm.run_join(buckets, results, spec, 4, 4, calc)
        engine.persist_snapshot(spec, snaps, buckets, results, 4, 4)
        path = sm.snapshot_path(snaps)
        with open(path) as handle:
            doc = json.load(handle)
        doc["content_digest"] = "0" * 64
        with open(path, "w") as handle:
            json.dump(doc, handle)
        with self.assertRaises(SnapshotError):
            sm.load_snapshot(snaps)

    def test_float_key_attribution_partition_invariant(self):
        spec = load_experiment_spec(self.sample, self.float_query)
        calc = stats_calculator(spec, 4)
        outputs = {}
        row_sets = {}
        for n in (3, 5, 8):
            buckets = self._d(f"float-b-{n}")
            results = self._d(f"float-r-{n}")
            engine.generate_version(buckets, spec, 3, n, 4)
            joined = jm.run_join(buckets, results, spec, n, 4, calc)
            attribution = {}
            ids = set()
            for doc in joined["documents"].values():
                for row in doc["rows"]:
                    key = jm.output_row_id(row).split(":", 1)[1]
                    ids.add(key)
                    attribution[key] = {
                        "source": row["primary_error_source"],
                        "match": row["key_quantization_match"],
                        "floats": {
                            ref: (
                                detail["fixed_point"],
                                detail["abs_error"],
                                detail["error_source"],
                            )
                            for ref, detail in row["projection"].items()
                            if "float_ref" in detail
                        },
                    }
            outputs[n] = attribution
            row_sets[n] = ids
        self.assertEqual(row_sets[3], row_sets[5])
        self.assertEqual(row_sets[5], row_sets[8])
        for key in row_sets[3]:
            self.assertEqual(outputs[3][key], outputs[5][key])
            self.assertEqual(outputs[5][key], outputs[8][key])

    def test_subprocess_sigkill_then_resume(self):
        spec = load_experiment_spec(self.sample)
        kill_dir = self._d("subprocess-kill")
        worker = os.path.join(ROOT, "tools", "crash_worker.py")
        cmd = [
            sys.executable,
            worker,
            "--sample",
            self.sample,
            "--bucket-dir",
            kill_dir,
            "--partitions",
            "4",
            "--fraction-bits",
            "4",
            "--kill-after",
            "3",
        ]
        killed = subprocess.run(cmd, cwd=ROOT)
        self.assertNotEqual(killed.returncode, 0)
        states = engine.scan_bucket_states(kill_dir, 4)
        self.assertIn("unknown", states.values())
        engine.recover_bucket_directory(kill_dir, spec, 3, 4, 4)
        engine.generate_version(kill_dir, spec, 3, 4, 4)
        clean_dir = self._d("subprocess-clean")
        engine.generate_version(clean_dir, spec, 3, 4, 4)
        self.assertEqual(
            generation_fingerprint(kill_dir, 4),
            generation_fingerprint(clean_dir, 4),
        )


if __name__ == "__main__":
    unittest.main()
