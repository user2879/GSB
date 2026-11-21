import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from hashlot.digest import content_digest
from hashlot.samplegen import generate

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "skew_sample.json")


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="hashlot_e2e_")
        if not os.path.exists(SAMPLE):
            os.makedirs(os.path.dirname(SAMPLE), exist_ok=True)
            import hashlib.samplegen as gen
            with open(SAMPLE, "w", encoding="utf-8") as handle:
                json.dump(generate(), handle, sort_keys=True)

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def _run_in(self, out_dir, *extra, command="run"):
        args = [sys.executable, "-m", "hashlot.cli",
                "--sample", SAMPLE, "--out-dir", out_dir,
                "--virtual-clock", *extra, command]
        env = dict(os.environ, PYTHONHASHSEED="42")
        return subprocess.run(args, capture_output=True, text=True, env=env)

    def _run(self, *extra, command="run"):
        tag = command + ("_multi" if "--workers" in extra else "")
        out_dir = os.path.join(self.out, tag)
        args = [sys.executable, "-m", "hashlot.cli",
                "--sample", SAMPLE, "--out-dir", out_dir,
                "--virtual-clock", *extra, command]
        env = dict(os.environ, PYTHONHASHSEED="42")
        proc = subprocess.run(args, capture_output=True, text=True, env=env)
        return proc, out_dir

    def test_deterministic_across_hash_seed(self):
        digests = set()
        for seed in ("1", "42", "999"):
            out_dir = os.path.join(self.out, f"seed{seed}")
            args = [sys.executable, "-m", "hashlot.cli", "--sample", SAMPLE,
                    "--out-dir", out_dir, "--virtual-clock", "run"]
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(args, capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = open(os.path.join(out_dir, "report.json"), "rb").read()
            digests.add(content_digest(json.loads(report)))
        self.assertEqual(len(digests), 1)

    def test_workers_match_single_thread(self):
        single, d1 = self._run(command="run")
        multi, d2 = self._run("--workers", "4", command="run")
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertEqual(multi.returncode, 0, multi.stderr)
        r1 = json.load(open(os.path.join(d1, "report.json")))["result_digest"]
        r2 = json.load(open(os.path.join(d2, "report.json")))["result_digest"]
        self.assertEqual(r1, r2)

    def test_hotspot_and_borrow_present(self):
        proc, out_dir = self._run(command="run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        mapping = json.load(open(os.path.join(out_dir, "mapping_summary.json")))
        watermark = json.load(open(os.path.join(out_dir, "watermark.json")))
        self.assertTrue(mapping["hot_partitions"])
        borrowed = [c for c in watermark["connections"] if c["is_borrowed"]]
        self.assertTrue(borrowed)
        self.assertLessEqual(
            watermark["peak_active_connections"],
            watermark["global_max_connections"])
        for conn in watermark["connections"]:
            self.assertLessEqual(conn["peak_keys"], conn["key_cap"])

    def test_replay_ok(self):
        out_dir = os.path.join(self.out, "shared_replay")
        run_proc = self._run_in(out_dir, command="run")
        self.assertEqual(run_proc.returncode, 0, run_proc.stderr)
        proc = self._run_in(out_dir, command="replay")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        replay = json.load(open(os.path.join(out_dir, "replay_report.json")))
        self.assertGreater(replay["batches_replayed"], 0)

    def test_crash_recover_matches_normal(self):
        normal_dir = os.path.join(self.out, "normal")
        env = dict(os.environ, PYTHONHASHSEED="42")
        proc = subprocess.run(
            [sys.executable, "-m", "hashlot.cli", "--sample", SAMPLE,
             "--out-dir", normal_dir, "--virtual-clock", "run"],
            capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        crash_dir = os.path.join(self.out, "crash")
        crashed = subprocess.run(
            [sys.executable, "-m", "hashlot.cli", "--sample", SAMPLE,
             "--out-dir", crash_dir, "--virtual-clock",
             "--crash-after-batch", "5", "run"],
            capture_output=True, text=True, env=env)
        self.assertEqual(crashed.returncode, 7)
        recovered = subprocess.run(
            [sys.executable, "-m", "hashlot.cli", "--sample", SAMPLE,
             "--out-dir", crash_dir, "--virtual-clock", "recover"],
            capture_output=True, text=True, env=env)
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        normal = json.load(open(os.path.join(normal_dir, "report.json")))["result_digest"]
        after = json.load(open(os.path.join(crash_dir, "report.json")))["result_digest"]
        self.assertEqual(normal, after)
        replay = json.load(open(os.path.join(crash_dir, "replay_report.json")))
        self.assertTrue(replay["unknown_connections"])

    def test_inject_borrow_cycle(self):
        proc, _ = self._run("--inject-borrow-cycle", command="run")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("E_BORROW", proc.stdout)

    def test_inject_early_deadline(self):
        proc, _ = self._run("--inject-early-deadline", command="run")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("E_DEADLINE", proc.stdout)

    def test_inject_digest_mismatch(self):
        out_dir = os.path.join(self.out, "shared_digest")
        run_proc = self._run_in(out_dir, "--inject-digest-mismatch", command="run")
        self.assertEqual(run_proc.returncode, 0, run_proc.stderr)
        proc = self._run_in(out_dir, "--inject-digest-mismatch", command="replay")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("E_DIGEST", proc.stdout)

    def test_snapshot_replacement_listed(self):
        from hashlot.spec import load_spec
        from hashlot.ledger import Ledger
        proc, out_dir = self._run(command="run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        ledger = Ledger(os.path.join(out_dir, "ledger.sqlite"))
        ledger.write_snapshot("stale", 999, {"old": 1}, [])
        ledger.close()
        spec = load_spec(os.path.join(out_dir, "effective_spec.json"))
        fresh = Ledger(os.path.join(out_dir, "ledger.sqlite"))
        replaced = fresh.initialize(
            "changed-table-digest",
            spec.slot_params_version + 1,
            {"hotspot_row_threshold": spec.hotspot.row_threshold + 1,
             "subpartition_count": spec.hotspot.subpartition_count,
             "partition_count": spec.topology.partition_count,
             "hash_rule": "FNV1A32_V1"})
        self.assertTrue(any(item["table"] == "snapshot" for item in replaced))
        self.assertTrue(any(item["table"] == "batches" for item in replaced))
        fresh.close()


if __name__ == "__main__":
    unittest.main()
