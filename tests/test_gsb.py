from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from gsb.cli import main as cli_main
from gsb.errors import GSBError
from gsb.filtering import filtered_arrays, predicate_digest
from gsb.generation import TableSample, corrupt_sample_file, ensure_table_sample
from gsb.logging_store import StructuredLogger
from gsb.model import load_scenario
from gsb.sampling import draw_sample
from gsb.session import Session, restore_snapshot_into
from gsb.snapshot import load_snapshot_payload, restore_snapshot, save_snapshot
from gsb.util import atomic_write_text, canonical_json, logical_content_digest, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_skewed.json"
FLOAT_DEMO = ROOT / "examples" / "demo_float.json"


class GSBEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.workdir = cls.base / "demo-work"
        cls.logger = StructuredLogger(cls.base / "logs" / "events.sqlite3", cls.base / "logs" / "events.jsonl")
        cls.session = Session.from_spec_file(DEMO, cls.workdir, logger=cls.logger)
        cls.result = cls.session.run()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_report_is_json_serializable_and_contains_required_sections(self) -> None:
        report = self.session.current_report()
        text = canonical_json(report, pretty=True)
        loaded = json.loads(text)
        for key in (
            "parameters",
            "data",
            "histograms",
            "sampling",
            "estimates",
            "attribution",
            "truth",
            "decision",
        ):
            self.assertIn(key, loaded)
        self.assertEqual(loaded["status"], "OK")
        self.assertEqual(loaded["parameters"]["parameter_version"], 1)
        self.assertEqual(loaded["estimates"]["fixed_point_scale"], 1_000_000)
        self.assertTrue(loaded["data"]["tables"])
        self.assertIn("quantile_scope", loaded["histograms"]["left"])

    def test_histogram_and_estimation_invariants(self) -> None:
        result = self.result
        for side in ("left", "right"):
            histogram = result.histograms[side]
            self.assertEqual(sum(bucket.row_count for bucket in histogram.buckets), histogram.row_count)
            self.assertEqual(histogram.bucket_count, 16)
            self.assertEqual(histogram.quantile_scope, "full_filtered")
            self.assertEqual(len(histogram.edges), 17)
            self.assertEqual(len({bucket.digest for bucket in histogram.buckets}), len(histogram.buckets))
        hist = result.histogram_estimate
        sample = result.sampling_estimate
        self.assertGreaterEqual(hist["estimate_fixed"], 0)
        self.assertLessEqual(hist["estimate_fixed"], int(hist["max_possible"] * 1_000_000))
        self.assertGreaterEqual(sample["estimate_fixed"], 0)
        self.assertLessEqual(sample["estimate_fixed"], int(sample["max_possible"] * 1_000_000))
        self.assertLessEqual(hist["quantization_error_upper"], 1.0)
        self.assertLessEqual(sample["quantization_error_upper"], 1.0)
        self.assertGreater(hist["overlap_pair_count"], 0)
        self.assertTruthyRanges(hist["overlap_row_ratio"])
        self.assertTrue(result.report["estimates"]["compatibility"]["histogram_within_bounds"])
        self.assertTrue(result.report["estimates"]["compatibility"]["sampling_within_bounds"])
        self.assertTrue(result.report["estimates"]["compatibility"]["truth_within_bounds"])

    def assertTruthyRanges(self, ratios: dict) -> None:
        self.assertGreaterEqual(ratios["left"], 0.0)
        self.assertLessEqual(ratios["left"], 1.0)
        self.assertGreaterEqual(ratios["right"], 0.0)
        self.assertLessEqual(ratios["right"], 1.0)

    def test_truth_join_and_decision(self) -> None:
        result = self.result
        left_arrays, _ = filtered_arrays(
            self.session.spec.left,
            self.session.samples["orders"].columns,
            self.session.spec.join.left_predicates,
        )
        right_arrays, _ = filtered_arrays(
            self.session.spec.right,
            self.session.samples["customers"].columns,
            self.session.spec.join.right_predicates,
        )
        left_counts = {int(key): int(count) for key, count in zip(*np.unique(left_arrays["customer_id"], return_counts=True))}
        right_counts = {int(key): int(count) for key, count in zip(*np.unique(right_arrays["id"], return_counts=True))}
        expected = sum(count * right_counts.get(key, 0) for key, count in left_counts.items())
        self.assertEqual(result.truth["row_count"], expected)
        self.assertEqual(result.decision["outcome"], "OK")
        interval = result.decision["interval"]
        self.assertLessEqual(interval["lower"], result.truth["row_count"])
        self.assertGreaterEqual(interval["upper"], result.truth["row_count"])
        self.assertIn(result.decision["selected_path"], {"histogram", "sampling"})

    def test_attribution_sorting_and_categories(self) -> None:
        top = self.result.attribution["top_k"]
        self.assertGreater(len(top), 0)
        self.assertEqual(self.result.attribution["pair_count"], len(self.result.attribution["pairs"]))
        for record in top:
            self.assertIn(record["source"], {"boundary_membership", "sample_sparsity", "skew_key_enrichment"})
            self.assertTrue(record["basis"])
            self.assertEqual(
                record["difference_fixed"],
                record["histogram_fixed"] - record["sampling_fixed"],
            )
        keys = [
            (
                -record["absolute_difference"],
                record["left_lower"],
                record["left_bucket"],
                record["right_lower"],
                record["right_bucket"],
            )
            for record in top
        ]
        self.assertEqual(keys, sorted(keys))

    def test_attribution_is_physically_order_independent(self) -> None:
        reversed_samples = {}
        for name, sample in self.session.samples.items():
            arrays = {}
            for column_name, array in sample.columns.items():
                arrays[column_name] = np.ascontiguousarray(array[::-1])
            reversed_samples[name] = TableSample(
                table=sample.table,
                columns=arrays,
                content_digest=logical_content_digest(
                    arrays,
                    names=[column.name for column in self.session.spec.table(name).columns],
                ),
                file_digest=sample.file_digest,
                path=sample.path,
                manifest_path=sample.manifest_path,
            )
        alternate = self.session._analyze(self.session.spec, reversed_samples)
        self.assertEqual(alternate.histograms["left"].digest, self.result.histograms["left"].digest)
        self.assertEqual(alternate.ledgers["left"].digest, self.result.ledgers["left"].digest)
        self.assertEqual(alternate.attribution["digest"], self.result.attribution["digest"])
        self.assertEqual(alternate.decision["digest"], self.result.decision["digest"])
        self.assertEqual(alternate.analysis_digest, self.result.analysis_digest)

    def test_float_scenario_and_boundary_declarations(self) -> None:
        workdir = self.base / "float-work"
        session = Session.from_spec_file(FLOAT_DEMO, workdir)
        result = session.run()
        self.assertEqual(result.decision["outcome"], "OK")
        left_buckets = result.histograms["left"].buckets
        right_buckets = result.histograms["right"].buckets
        self.assertTrue(left_buckets[0].lower_inclusive)
        self.assertTrue(all(bucket.upper_inclusive for bucket in left_buckets))
        self.assertFalse(any(bucket.lower_inclusive for bucket in right_buckets[1:]))
        self.assertGreater(result.truth["row_count"], 0)
        self.assertGreater(result.sampling_estimate["sample_hit_pairs"], 0)

    def test_parameter_evolution_replaces_whole_view(self) -> None:
        workdir = self.base / "evolution-work"
        session = Session.from_spec_file(DEMO, workdir)
        first = session.run()
        evolution = session.evolve(buckets=20, sample_seed=987654)
        self.assertEqual(evolution.transition["old_parameter_version"], 1)
        self.assertEqual(evolution.transition["new_parameter_version"], 2)
        self.assertTrue(evolution.transition["stale_artifacts_excluded"])
        self.assertEqual(session.parameter_version, 2)
        self.assertEqual(session.current.parameter_version, 2)
        self.assertEqual(len(session.stale_versions), 1)
        self.assertEqual(session.stale_versions[0]["parameter_version"], 1)
        self.assertNotEqual(first.analysis_digest, session.current.analysis_digest)
        report = session.current_report()
        self.assertEqual(report["last_transition"]["new_parameter_version"], 2)
        affected = report["last_transition"]["affected_overlap_pairs"]
        self.assertEqual(affected, sorted(affected, key=lambda item: (item["left_lower"], item["left_bucket"], item["right_lower"], item["right_bucket"])))
        second = session.evolve(sample_size=450)
        self.assertEqual(second.transition["new_parameter_version"], 3)
        self.assertEqual(session.parameter_version, 3)
        no_op = session.evolve(buckets=20, sample_size=450, sample_seed=987654)
        self.assertTrue(no_op.transition.get("no_op"))
        boundary_change = session.evolve(boundary_rules={"int": "left_open_right_closed"})
        self.assertEqual(boundary_change.transition["new_parameter_version"], 4)
        self.assertEqual(
            boundary_change.transition["parameter_diff"]["boundary_rules"]["new"]["int"],
            "left_open_right_closed",
        )
        with self.assertRaises(GSBError) as context:
            session.evolve(buckets=0)
        self.assertEqual(context.exception.code, "E_BUCKET")

    def test_snapshot_replay_and_state_replacement(self) -> None:
        workdir = self.base / "snapshot-work"
        session = Session.from_spec_file(DEMO, workdir)
        original = session.run()
        snapshot_path = self.base / "snapshots" / "state.json"
        session.save_snapshot(snapshot_path)
        self.assertTrue(snapshot_path.exists())
        session.evolve(buckets=32)
        self.assertEqual(session.parameter_version, 2)
        restored, transition = restore_snapshot_into(
            snapshot_path,
            self.base / "snapshot-restore",
            current=session,
        )
        self.assertTrue(transition["version_changed"])
        self.assertEqual(restored.parameter_version, 1)
        self.assertEqual(restored.current.analysis_digest, original.analysis_digest)
        self.assertEqual(transition["replay"]["status"], "OK")
        self.assertTrue(transition["stale_artifacts_excluded"])
        self.assertGreaterEqual(len(restored.stale_versions), 1)
        after_restore = restored.evolve(buckets=20)
        self.assertEqual(after_restore.transition["new_parameter_version"], 3)

    def test_snapshot_replay_detects_first_bucket_field_mismatch(self) -> None:
        workdir = self.base / "snapshot-tamper-work"
        session = Session.from_spec_file(DEMO, workdir)
        session.run()
        path = self.base / "snapshots" / "tampered.json"
        save_snapshot(session, path)
        payload = load_snapshot_payload(path)
        payload["histograms"]["left"]["buckets"][2]["row_count"] += 1
        atomic_write_text(path, canonical_json(payload, pretty=True))
        with self.assertRaises(GSBError) as context:
            restore_snapshot(path, self.base / "snapshot-tamper-restore")
        error = context.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.details["field"], "row_count")
        self.assertEqual(error.details["bucket_lower"], payload["histograms"]["left"]["buckets"][2]["lower"])

    def test_sample_corruption_and_structured_logging(self) -> None:
        workdir = self.base / "corrupt-work"
        database = self.base / "corrupt-logs" / "events.sqlite3"
        logger = StructuredLogger(database, self.base / "corrupt-logs" / "events.jsonl")
        session = Session.from_spec_file(DEMO, workdir, logger=logger)
        target = session.samples[sorted(session.samples)[0]].path
        corrupt_sample_file(target)
        with self.assertRaises(GSBError) as context:
            session.verify_sample_files()
        self.assertEqual(context.exception.code, "E_SAMPLE")
        logger.error(context.exception, "corrupt_test")
        events = logger.read_events(20)
        self.assertTrue(any(event["code"] == "E_SAMPLE" for event in events))

    def test_deterministic_npz_files_are_byte_identical(self) -> None:
        spec = load_scenario(DEMO)
        first = ensure_table_sample(spec, "orders", self.base / "det-a")
        second = ensure_table_sample(spec, "orders", self.base / "det-b")
        self.assertEqual(first.file_digest, second.file_digest)
        self.assertEqual(first.content_digest, second.content_digest)
        self.assertEqual(Path(first.path).read_bytes(), Path(second.path).read_bytes())
        self.assertEqual(sha256_file(first.path), first.file_digest)

    def test_physical_row_order_changes_are_sample_invariant(self) -> None:
        spec = load_scenario(DEMO)
        table = spec.left
        arrays = self.session.samples["orders"].columns
        names = [column.name for column in table.columns]
        reversed_arrays = {name: np.ascontiguousarray(array[::-1]) for name, array in arrays.items()}
        digest = logical_content_digest(arrays, names=names)
        reversed_digest = logical_content_digest(reversed_arrays, names=names)
        self.assertEqual(digest, reversed_digest)
        filtered_a, _ = filtered_arrays(table, arrays, spec.join.left_predicates)
        filtered_b, _ = filtered_arrays(table, reversed_arrays, spec.join.left_predicates)
        predicate_hash = predicate_digest(spec.join.left_predicates)
        ledger_a = draw_sample(
            filtered_a,
            table=table.name,
            key_column=spec.join.left_key,
            table_rows=table.declared_rows,
            sample_size=300,
            sample_seed=12345,
            data_digest=digest,
            predicate_digest=predicate_hash,
            names=names,
        )
        ledger_b = draw_sample(
            filtered_b,
            table=table.name,
            key_column=spec.join.left_key,
            table_rows=table.declared_rows,
            sample_size=300,
            sample_seed=12345,
            data_digest=reversed_digest,
            predicate_digest=predicate_hash,
            names=names,
        )
        self.assertEqual(ledger_a.digest, ledger_b.digest)
        self.assertEqual(
            [entry.to_dict() for entry in ledger_a.entries],
            [entry.to_dict() for entry in ledger_b.entries],
        )


    def test_estimate_miss_reports_out_of_bounds_and_strict_exit(self) -> None:
        workdir = self.base / "miss-work"
        session = Session.from_spec_file(ROOT / "examples" / "demo_estimate_miss.json", workdir)
        result = session.run()
        self.assertEqual(result.decision["outcome"], "E_EST")
        self.assertFalse(result.decision["within_interval"])
        self.assertTrue(
            result.decision["histogram"]["out_of_bounds_ratio"] > 0
            or result.decision["sampling"]["out_of_bounds_ratio"] > 0
        )
        self.assertIsNotNone(result.decision["dominant_deviation_bucket"])
        self.assertIn("pair_id", result.decision["dominant_deviation_bucket"])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = cli_main(
                [
                    "run",
                    "--spec",
                    str(ROOT / "examples" / "demo_estimate_miss.json"),
                    "--workdir",
                    str(self.base / "miss-cli"),
                    "--report",
                    str(self.base / "miss-report.json"),
                    "--strict-estimate",
                    "--quiet",
                ]
            )
        self.assertEqual(exit_code, 7)
        report = json.loads((self.base / "miss-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["decision"]["outcome"], "E_EST")

    def test_spec_validation_special_codes(self) -> None:
        with self.assertRaises(GSBError) as invalid:
            load_scenario(ROOT / "examples" / "anomalies" / "invalid_spec.json")
        self.assertEqual(invalid.exception.code, "E_SPEC")
        self.assertGreaterEqual(len(invalid.exception.details["errors"]), 2)
        with self.assertRaises(GSBError) as version:
            load_scenario(ROOT / "examples" / "anomalies" / "missing_parameter_version.json")
        self.assertEqual(version.exception.code, "E_VERSION")
        with self.assertRaises(GSBError) as bucket:
            load_scenario(ROOT / "examples" / "anomalies" / "zero_buckets.json")
        self.assertEqual(bucket.exception.code, "E_BUCKET")
        with self.assertRaises(GSBError) as sample:
            Session.from_spec_file(
                ROOT / "examples" / "anomalies" / "sample_size_exceeded.json",
                self.base / "sample-exceeded",
            ).run()
        self.assertEqual(sample.exception.code, "E_SAMPLE")
        self.assertIn("upper_limit", sample.exception.details)

    def test_cli_injection_exit_codes(self) -> None:
        cases = [
            (["--inject-missing-version"], 4),
            (["--inject-buckets-zero"], 3),
            (["--inject-sample-corruption"], 5),
        ]
        for index, (extra, expected) in enumerate(cases):
            with self.subTest(extra=extra):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = cli_main(
                        [
                            "run",
                            "--spec",
                            str(DEMO),
                            "--workdir",
                            str(self.base / f"cli-inject-{index}"),
                            *extra,
                        ]
                    )
                self.assertEqual(exit_code, expected)
                payload = json.loads(stderr.getvalue().strip().splitlines()[-1])
                self.assertEqual(payload["code"], {4: "E_VERSION", 3: "E_BUCKET", 5: "E_SAMPLE"}[expected])


if __name__ == "__main__":
    unittest.main()
