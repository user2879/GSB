from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from joinlab.anomaly import inject_skew, inject_snapshot_missing_version, inject_type_mismatch
from joinlab.canonical import digest_json
from joinlab.errors import JoinLabError
from joinlab.estimation import _classify_difference, estimate_join
from joinlab.histogram import Bucket, build_histogram, histogram_bytes
from joinlab.logging_utils import StructuredLogger
from joinlab.params import AnalysisParams, normalise_params
from joinlab.sampling import build_sample, sample_bytes
from joinlab.session import WorkbenchSession
from joinlab.snapshot import load_snapshot
from joinlab.spec import ingest_workspace, load_workspace
from joinlab.truth import execute_truth_join, validate_estimates

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WORKSPACE = ROOT / "samples" / "base_tables.json"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def workspace_document(left_rows, right_rows, *, left_type="integer", right_type="integer", duplicate_name=False):
    left_table = {
        "name": "left_table",
        "row_count": len(left_rows),
        "skew": False,
        "columns": [{"name": "id", "type": left_type}, {"name": "v", "type": "integer"}],
        "rows": left_rows,
    }
    right_table = {
        "name": "left_table" if duplicate_name else "right_table",
        "row_count": len(right_rows),
        "skew": False,
        "columns": [{"name": "id", "type": right_type}, {"name": "v", "type": "integer"}],
        "rows": right_rows,
    }
    return {
        "workspace_id": "test-workspace",
        "tables": [left_table, right_table],
        "join": {
            "left_table": "left_table",
            "left_key": "id",
            "right_table": "right_table",
            "right_key": "id",
        },
        "filters": {},
        "attribution_k": 3,
    }


def make_workspace(tmp: Path, left_rows=None, right_rows=None, **kwargs) -> Path:
    left_rows = left_rows or [[index % 10, index] for index in range(1, 25)]
    right_rows = right_rows or [[index % 10, index + 100] for index in range(1, 31)]
    directory = tmp / "workspace"
    write_json(directory / "base.json", workspace_document(left_rows, right_rows, **kwargs))
    return directory / "base.json"


def make_session(tmp: Path, bucket_count=4, sample_size=8, seed=7) -> WorkbenchSession:
    workspace = make_workspace(tmp)
    spec = load_workspace(workspace)
    params = normalise_params(
        {
            "bucket_count": bucket_count,
            "sample_size": sample_size,
            "seed": seed,
            "quantile_policy": "nearest_rank",
            "attribution_k": 3,
        },
        spec.default_params,
        spec.attribution_k,
    )
    return WorkbenchSession(
        spec,
        tmp / "data.sqlite",
        params,
        StructuredLogger(None),
        tmp / "version_state.json",
    )


class LoaderTests(unittest.TestCase):
    def test_digest_is_permutation_independent_and_sqlite_readback_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            left = [[i, i * 2] for i in range(1, 13)]
            right = [[i % 7, i] for i in range(1, 17)]
            path_a = make_workspace(tmp / "a", left, right)
            spec_a = load_workspace(path_a)
            first = ingest_workspace(spec_a, tmp / "a.sqlite")
            second = ingest_workspace(spec_a, tmp / "a.sqlite")
            self.assertEqual(first["tables"], second["tables"])
            self.assertEqual(first["data_digest"], second["data_digest"])

            path_b = make_workspace(tmp / "b", list(reversed(left)), list(reversed(right)))
            spec_b = load_workspace(path_b)
            self.assertEqual(spec_a.tables["left_table"].data_digest, spec_b.tables["left_table"].data_digest)
            self.assertEqual(spec_a.tables["right_table"].data_digest, spec_b.tables["right_table"].data_digest)

    def test_duplicate_table_rejected_with_field_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            path = make_workspace(tmp, duplicate_name=True)
            with self.assertRaises(JoinLabError) as context:
                load_workspace(path)
            self.assertEqual(context.exception.code, "E_TABLE")
            self.assertEqual(context.exception.field_path, "tables[1].name")

    def test_missing_join_key_rejected_with_field_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            document = workspace_document([[1, 1]], [[1, 1]])
            document["join"]["left_key"] = "missing"
            path = tmp / "bad.json"
            write_json(path, document)
            with self.assertRaises(JoinLabError) as context:
                load_workspace(path)
            self.assertEqual(context.exception.code, "E_TABLE")
            self.assertEqual(context.exception.field_path, "join.left_key")

    def test_row_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            document = workspace_document([[1, 1]], [[1, 1]])
            document["tables"][0]["row_count"] = 2
            path = tmp / "bad.json"
            write_json(path, document)
            with self.assertRaises(JoinLabError) as context:
                load_workspace(path)
            self.assertEqual(context.exception.code, "E_TABLE")
            self.assertEqual(context.exception.field_path, "tables[0].row_count")


class HistogramTests(unittest.TestCase):
    def test_integer_left_closed_boundaries_and_determinism(self):
        rows = [{"id": value} for value in range(1, 9)]
        first = build_histogram("t", "id", "integer", rows, 4, "nearest_rank")
        second = build_histogram("t", "id", "integer", rows, 4, "nearest_rank")
        self.assertEqual(histogram_bytes(first), histogram_bytes(second))
        self.assertEqual([bucket.row_count for bucket in first.buckets], [2, 2, 2, 2])
        self.assertEqual(first.bucket_index(3), 1)
        self.assertEqual(first.bucket_index(4), 1)
        self.assertEqual(first.buckets[0].interval if hasattr(first.buckets[0], "interval") else first.buckets[0].as_dict()["interval"], "[1, 3)")
        self.assertTrue(first.buckets[0].lower_inclusive)
        self.assertFalse(first.buckets[0].upper_inclusive)

    def test_float_right_closed_boundaries(self):
        rows = [{"id": value} for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
        histogram = build_histogram("t", "id", "float", rows, 2, "nearest_rank")
        self.assertEqual([bucket.row_count for bucket in histogram.buckets], [3, 3])
        self.assertEqual(histogram.bucket_index(3.0), 0)
        self.assertEqual(histogram.bucket_index(3.0001), 1)
        self.assertTrue(histogram.buckets[0].upper_inclusive)
        self.assertFalse(histogram.buckets[1].lower_inclusive)

    def test_null_count_and_e_hist(self):
        rows = [{"id": None}, {"id": 1}, {"id": 2}]
        histogram = build_histogram("t", "id", "integer", rows, 2)
        self.assertEqual(histogram.null_count, 1)
        self.assertEqual(sum(bucket.null_count for bucket in histogram.buckets), 0)
        with self.assertRaises(JoinLabError) as zero:
            build_histogram("t", "id", "integer", rows, 0)
        self.assertEqual(zero.exception.code, "E_HIST")
        with self.assertRaises(JoinLabError) as excess:
            build_histogram("t", "id", "integer", rows, 10)
        self.assertEqual(excess.exception.code, "E_HIST")


class SamplingTests(unittest.TestCase):
    def test_sampling_is_order_independent_and_keeps_frequencies(self):
        rows = [
            {"__rid": f"r{index:02d}", "id": index % 4}
            for index in range(30)
        ]
        first = build_sample("t", "id", "integer", rows, 10, 123, "digest")
        second = build_sample("t", "id", "integer", list(reversed(rows)), 10, 123, "digest")
        self.assertEqual(first.sample_digest, second.sample_digest)
        self.assertEqual(first.frequencies, second.frequencies)
        self.assertEqual(sum(first.frequencies.values()) + first.null_sample_count, 10)
        self.assertEqual(sample_bytes(first), sample_bytes(second))

    def test_sample_size_excess_is_e_sample(self):
        rows = [{"__rid": "r1", "id": 1}]
        with self.assertRaises(JoinLabError) as context:
            build_sample("t", "id", "integer", rows, 2, 1, "digest")
        self.assertEqual(context.exception.code, "E_SAMPLE")


class EstimationTests(unittest.TestCase):
    def test_attribution_classifier_covers_all_three_sources(self):
        dense = Bucket(0, 1, 10, True, False, 10, 0, 8, 1, 1)
        skew = Bucket(0, 1, 10, True, False, 10, 0, 4, 1, 5)
        boundary = Bucket(0, 1, 10, True, False, 10, 0, 8, 1, 1)
        self.assertEqual(
            _classify_difference(skew, skew, 6, 6, 5),
            ("skew_enrichment", "倾斜键富集"),
        )
        self.assertEqual(
            _classify_difference(boundary, boundary, 2, 6, 0),
            ("sample_sparsity", "样本稀疏"),
        )
        self.assertEqual(
            _classify_difference(dense, dense, 6, 6, 5),
            ("boundary_ownership", "边界归属"),
        )

    def test_sample_workspace_estimates_pass_and_attribution_is_stable(self):
        spec = load_workspace(SAMPLE_WORKSPACE)
        params = AnalysisParams(10, 320, 20261120, "nearest_rank", 5)
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            session = WorkbenchSession(spec, tmp / "data.sqlite", params, StructuredLogger(None), tmp / "state.json")
            report = session.build_current()
            self.assertEqual(report["validation"]["status"], "PASS")
            self.assertIsInstance(report["estimate"]["histogram_path"]["estimate_fixed"], int)
            self.assertIsInstance(report["estimate"]["sample_path"]["estimate_fixed"], int)
            keys = [
                (-abs(item["difference_fixed"]), item["key_interval"]["lower"], item["left_bucket_index"], item["right_bucket_index"])
                for item in report["attribution"]
            ]
            self.assertEqual(keys, sorted(keys))

            view = session.current
            reversed_left = dataclasses.replace(view.histograms["accounts"], buckets=tuple(reversed(view.histograms["accounts"].buckets)))
            reversed_right = dataclasses.replace(view.histograms["events"], buckets=tuple(reversed(view.histograms["events"].buckets)))
            estimate_reordered = estimate_join(
                reversed_left,
                reversed_right,
                view.samples["accounts"],
                view.samples["events"],
                len(view.filtered_rows["accounts"]),
                len(view.filtered_rows["events"]),
                5,
            )
            self.assertEqual(
                estimate_reordered["histogram_path"]["estimate_fixed"],
                view.estimate["histogram_path"]["estimate_fixed"],
            )
            self.assertEqual(estimate_reordered["attribution"], view.estimate["attribution"])

    def test_truth_join_sequence_is_input_order_independent(self):
        left = [{"__rid": "l2", "id": 2}, {"__rid": "l1", "id": 1}]
        right = [{"__rid": "r3", "id": 2}, {"__rid": "r1", "id": 1}, {"__rid": "r2", "id": 2}]
        first = execute_truth_join(left, right, "id", "id", "integer", "integer")
        second = execute_truth_join(list(reversed(left)), list(reversed(right)), "id", "id", "integer", "integer")
        self.assertEqual(first["true_row_count"], 3)
        self.assertEqual(first["output_sequence_digest"], second["output_sequence_digest"])
        self.assertEqual([row["key"] for row in first["output_preview"]], [1, 2, 2])

    def test_estimate_out_of_tolerance_is_e_estimate(self):
        truth = {
            "true_row_count": 100,
            "output_sequence_digest": "digest",
        }
        histogram = {"estimate_fixed": 10_000_000_000, "estimate_rows": "10000.000000"}
        sample = {"estimate_fixed": 20_000_000_000, "estimate_rows": "20000.000000"}
        with self.assertRaises(JoinLabError) as context:
            validate_estimates(truth, histogram, sample, 10, 100, [], 1_000_000)
        self.assertEqual(context.exception.code, "E_ESTIMATE")
        self.assertIn("histogram_estimate_fixed", context.exception.details)
        self.assertIn("sample_estimate_fixed", context.exception.details)


class VersionAndSnapshotTests(unittest.TestCase):
    def test_parameter_change_replaces_whole_view_and_marks_old_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            session = make_session(tmp, bucket_count=4, sample_size=8)
            first = session.build_current()
            old_hist_digest = first["histograms"]["left_table"]["digest"]
            changed = AnalysisParams(6, 10, 8, "nearest_rank", 3)
            result = session.replace_parameters(changed)
            self.assertTrue(result["changed"])
            self.assertEqual(result["replacement"]["from_version"], 1)
            self.assertEqual(result["replacement"]["to_version"], 2)
            self.assertEqual(result["report"]["params_version"], 2)
            self.assertTrue(result["report"]["stale_epochs"][0]["stale"])
            self.assertNotEqual(result["report"]["histograms"]["left_table"]["digest"], old_hist_digest)
            changed_tables = {item["table"]: item for item in result["replacement"]["changed_tables"]}
            self.assertEqual(changed_tables["left_table"]["old_bucket_count"], 4)
            self.assertEqual(changed_tables["left_table"]["new_bucket_count"], 6)
            self.assertEqual(result["replacement"]["affected_conclusions"], [
                "histogram_estimate",
                "sample_estimate",
                "bucket_attribution",
                "truth_validation",
            ])

    def test_snapshot_restore_replays_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            session = make_session(tmp, bucket_count=4, sample_size=8)
            baseline = session.build_current()
            snapshot_path = tmp / "snapshot.json"
            session.save_snapshot(snapshot_path)
            session.replace_parameters(AnalysisParams(5, 9, 9, "lower_rank", 3))

            reopened = make_session(tmp, bucket_count=5, sample_size=9, seed=9)
            reopened.build_current()
            restored = reopened.restore_snapshot(snapshot_path)
            self.assertTrue(restored["replay"]["verified"])
            self.assertEqual(restored["report"]["params_version"], baseline["params_version"])

            document = json.loads(snapshot_path.read_text(encoding="utf-8"))
            table_name = next(iter(document["histograms"]))
            document["histograms"][table_name]["digest"] = "tampered"
            tampered = tmp / "tampered.json"
            write_json(tampered, document)
            verifier = make_session(tmp / "tamper", bucket_count=5, sample_size=9, seed=9)
            verifier.build_current()
            with self.assertRaises(JoinLabError) as context:
                verifier.restore_snapshot(tampered)
            self.assertEqual(context.exception.code, "E_REPLAY")
            self.assertIn("first_mismatch", context.exception.details)

    def test_snapshot_missing_version_is_e_snap(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            session = make_session(tmp)
            session.build_current()
            snapshot_path = tmp / "snapshot.json"
            session.save_snapshot(snapshot_path)
            missing = inject_snapshot_missing_version(snapshot_path, tmp / "missing.json")
            with self.assertRaises(JoinLabError) as context:
                load_snapshot(missing)
            self.assertEqual(context.exception.code, "E_SNAP")
            self.assertEqual(context.exception.field_path, "snapshot.params_version")


class AnomalyTests(unittest.TestCase):
    def test_skew_and_type_injections_raise_expected_codes(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            skew_workspace = inject_skew(SAMPLE_WORKSPACE, tmp / "skew")
            with self.assertRaises(JoinLabError) as skew:
                load_workspace(skew_workspace)
            self.assertEqual(skew.exception.code, "E_SKEW")

            type_workspace = inject_type_mismatch(SAMPLE_WORKSPACE, tmp / "type")
            with self.assertRaises(JoinLabError) as type_error:
                load_workspace(type_workspace)
            self.assertEqual(type_error.exception.code, "E_TYPE")
            self.assertTrue(type_error.exception.field_path.endswith(".type"))


if __name__ == "__main__":
    unittest.main()
