import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import bucket as bm
from dhj import engine
from dhj import join as jm
from dhj import verify as vm
from dhj.errors import BucketError, ResumeError, VerifyError
from dhj.shuffle import bucket_path, compare_buckets
from tools.experiment_lib import load_experiment_spec, stats_calculator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BucketAndJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_experiment_spec(os.path.join(ROOT, "data", "sample.json"))
        cls.tmp = tempfile.mkdtemp()
        cls.n = 4
        cls.fb = 4

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _dir(self, name):
        path = os.path.join(self.tmp, name)
        os.makedirs(path, exist_ok=True)
        return path

    def test_v1_v2_v3_round_trip(self):
        for version in (1, 2, 3):
            directory = self._dir(f"gen{version}")
            engine.generate_version(
                directory, self.spec, version, self.n, self.fb
            )
            for side in ("left", "right"):
                for partition in range(self.n):
                    bkt = bm.read_bucket_file(
                        bucket_path(directory, side, partition)
                    )
                    self.assertEqual(bkt.version, version)
                    if version >= 2:
                        self.assertTrue(bkt.columns)
                    if version == 3:
                        self.assertTrue(all(
                            r.q_key != bm.fp.NULL_Q for r in bkt.rows
                        ))

    def test_upgrade_byte_identical_to_direct_v3(self):
        v1 = self._dir("up-v1")
        direct = self._dir("up-direct-v3")
        upgraded = self._dir("up-v3")
        engine.generate_version(v1, self.spec, 1, self.n, self.fb)
        engine.generate_version(direct, self.spec, 3, self.n, self.fb)
        report, checkpoint = engine.upgrade_generation(
            v1, upgraded, self.spec, self.n, self.fb
        )
        self.assertEqual(len(report), 2 * self.n)
        for side in ("left", "right"):
            for partition in range(self.n):
                a = bm.read_bucket_file(bucket_path(upgraded, side, partition))
                b = bm.read_bucket_file(bucket_path(direct, side, partition))
                self.assertIsNone(compare_buckets(a, b))
                with open(bucket_path(upgraded, side, partition), "rb") as fh:
                    raw_a = fh.read()
                with open(bucket_path(direct, side, partition), "rb") as fh:
                    raw_b = fh.read()
                self.assertEqual(raw_a, raw_b)
                backup = bm.backup_path_for(
                    bucket_path(upgraded, side, partition), 3
                )
                self.assertTrue(os.path.exists(backup))
                self.assertEqual(
                    bm.read_bucket_file(backup).version, 1
                )


    def test_v2_to_v3_upgrade_fields(self):
        v2 = self._dir("v2-source")
        upgraded = self._dir("v2-upgraded")
        engine.generate_version(v2, self.spec, 2, self.n, self.fb)
        report, _ = engine.upgrade_generation(
            v2, upgraded, self.spec, self.n, self.fb
        )
        self.assertEqual(len(report), 2 * self.n)
        for item in report:
            self.assertEqual(item["source_version"], 2)
            self.assertEqual(item["target_version"], 3)
            self.assertEqual(
                item["added_fields"],
                ["quantized_join_keys", "row_order_table"],
            )
        direct = self._dir("v2-direct-v3")
        engine.generate_version(direct, self.spec, 3, self.n, self.fb)
        for side in ("left", "right"):
            for partition in range(self.n):
                a = open(bucket_path(upgraded, side, partition), "rb").read()
                b = open(bucket_path(direct, side, partition), "rb").read()
                self.assertEqual(a, b)

    def test_unsupported_version_is_e_bucket(self):
        directory = self._dir("bad-version")
        engine.generate_version(directory, self.spec, 3, self.n, self.fb)
        path = bucket_path(directory, "left", 0)
        raw = bytearray(open(path, "rb").read())
        raw[8:10] = struct.pack(">H", 9)
        open(path, "wb").write(bytes(raw))
        with self.assertRaises(BucketError) as ctx:
            bm.read_bucket_file(path)
        self.assertEqual(ctx.exception.code, "E_BUCKET")
        self.assertEqual(ctx.exception.details["version"], 9)

    def test_truncated_bucket_is_unknown_and_e_bucket(self):
        directory = self._dir("truncated")
        engine.generate_version(directory, self.spec, 3, self.n, self.fb)
        path = bucket_path(directory, "left", 1)
        raw = open(path, "rb").read()
        open(path, "wb").write(raw[: len(raw) // 2])
        with self.assertRaises(BucketError):
            bm.read_bucket_file(path)
        status, bucket = bm.bucket_status(path)
        self.assertEqual(status, "unknown")
        self.assertIsNone(bucket)

    def test_digest_tamper_detected(self):
        directory = self._dir("digest-tamper")
        engine.generate_version(directory, self.spec, 2, self.n, self.fb)
        path = bucket_path(directory, "left", 0)
        bkt = bm.read_bucket_file(path)
        bkt.rows[0].digest ^= 0xFF
        bm.write_bucket_file(path, bkt)
        with self.assertRaises(BucketError) as ctx:
            bm.read_bucket_file(path)
        self.assertEqual(ctx.exception.code, "E_BUCKET")

    def test_full_join_and_resume_byte_identical(self):
        directory = self._dir("join-v3")
        results = self._dir("join-results")
        engine.generate_version(directory, self.spec, 3, self.n, self.fb)
        calc = stats_calculator(self.spec, self.fb)
        full = jm.run_join(
            directory, results, self.spec, self.n, self.fb, calc
        )
        resume_dir = self._dir("join-results-resume")
        reuse = dict(full["documents"])
        resumed = jm.run_join(
            directory, resume_dir, self.spec, self.n, self.fb, calc,
            reuse=reuse,
        )
        jm.cross_check_resume(full["documents"], resumed["documents"])
        for partition in full["documents"]:
            a = open(os.path.join(
                results, f"join-p{partition:05d}.json"), "rb").read()
            b = open(os.path.join(
                resume_dir, f"join-p{partition:05d}.json"), "rb").read()
            self.assertEqual(a, b)

    def test_resume_mismatch_raises_e_resume(self):
        directory = self._dir("resume-bad-v3")
        results = self._dir("resume-bad-results")
        engine.generate_version(directory, self.spec, 3, self.n, self.fb)
        calc = stats_calculator(self.spec, self.fb)
        full = jm.run_join(
            directory, results, self.spec, self.n, self.fb, calc
        )
        bad = {p: dict(d) for p, d in full["documents"].items()}
        target = bad[0]
        target["result_digest"] = "0" * 64
        with self.assertRaises(ResumeError) as ctx:
            jm.cross_check_resume(full["documents"], bad)
        self.assertEqual(ctx.exception.code, "E_RESUME")
        self.assertEqual(ctx.exception.details["partition"], 0)

    def test_independent_verifier_e_verify(self):
        directory = self._dir("verify-v3")
        results = self._dir("verify-results")
        engine.generate_version(directory, self.spec, 3, self.n, self.fb)
        calc = stats_calculator(self.spec, self.fb)
        good = jm.run_join(
            directory, results, self.spec, self.n, self.fb, calc
        )
        jm.build_ledger  # noqa: B018
        from tools.experiment_lib import write_ledger

        write_ledger(results, good, self.spec, self.n, self.fb)
        outcome = vm.verify_all(
            self.spec, directory, results, self.n, self.fb, calc
        )
        self.assertEqual(outcome["status"], "ok")
        bad_doc = dict(good["documents"][0])
        bad_doc["result_digest"] = "f" * 64
        bad_doc["stats"]["row_count"] = -1
        bad_path = os.path.join(results, "join-p00000.json")
        from dhj.jsonio import write_json

        write_json(bad_path, bad_doc)
        with self.assertRaises(VerifyError) as ctx:
            vm.verify_all(
                self.spec, directory, results, self.n, self.fb, calc
            )
        self.assertEqual(ctx.exception.code, "E_VERIFY")
        self.assertEqual(ctx.exception.details["partition"], 0)

    def test_partition_count_change_recomputes_and_matches_global_set(self):
        d4 = self._dir("pc4")
        r4 = self._dir("pc4r")
        d7 = self._dir("pc7")
        r7 = self._dir("pc7r")
        engine.generate_version(d4, self.spec, 3, 4, self.fb)
        engine.generate_version(d7, self.spec, 3, 7, self.fb)
        calc = stats_calculator(self.spec, self.fb)
        join4 = jm.run_join(d4, r4, self.spec, 4, self.fb, calc)
        join7 = jm.run_join(d7, r7, self.spec, 7, self.fb, calc)
        ids4 = sorted(
            jm.output_row_id(r).split(":", 1)[1]
            for doc in join4["documents"].values() for r in doc["rows"]
        )
        ids7 = sorted(
            jm.output_row_id(r).split(":", 1)[1]
            for doc in join7["documents"].values() for r in doc["rows"]
        )
        self.assertEqual(ids4, ids7)


if __name__ == "__main__":
    unittest.main()
