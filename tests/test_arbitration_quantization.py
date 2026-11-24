import os
import sys
import unittest
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowquery_diag.aggregation import Evidence  # noqa: E402
from slowquery_diag.arbitration import merge_evidence  # noqa: E402
from slowquery_diag.diagnosis import build_report  # noqa: E402
from slowquery_diag.models import (  # noqa: E402
    CandidateRule,
    SourceDeclaration,
    SOURCE_APP,
    SOURCE_DB,
    SOURCE_SLOW,
    canonical_json,
)
from slowquery_diag.quantization import (  # noqa: E402
    SATURATION,
    Quantizer,
    build_error_bound,
    indistinguishable,
)


def _evidence(source_id, source_type, version, query_id=1,
              latency=1000, rows=10):
    return Evidence(
        source_id=source_id, source_type=source_type, format_version=version,
        query_id=query_id, latency_us=latency, rows_scanned=rows,
        sample_count=1, weight_total=1, latency_caliber="p95",
        sample_period_ms=1000, filled_fields=(),
    )


class ArbitrationTests(unittest.TestCase):
    def test_bias_then_version_then_id_order(self):
        db_old = _evidence("z_db", SOURCE_DB, 1)
        db_new = _evidence("a_db", SOURCE_DB, 2)
        app = _evidence("a_app", SOURCE_APP, 3, latency=999)
        slow = _evidence("a_slow", SOURCE_SLOW, 2)
        verdicts = merge_evidence({
            "app": {1: app}, "slow": {1: slow},
            "db_old": {1: db_old}, "db_new": {1: db_new},
        })
        verdict = verdicts[1]
        self.assertEqual(verdict.latency.winner_source, "a_db")
        order = [view.source_id for view in verdict.latency.views]
        self.assertEqual(order, ["a_db", "z_db", "a_slow", "a_app"])
        rejected = [view for view in verdict.latency.views if not view.adopted]
        self.assertTrue(all(view.reject_reason for view in rejected))

    def test_byte_identical_under_shuffled_input(self):
        evidences = [
            _evidence("s1", SOURCE_DB, 2, query_id=q, latency=q * 100)
            for q in (1, 2)
        ] + [
            _evidence("s2", SOURCE_APP, 1, query_id=q, latency=q * 200)
            for q in (1, 2)
        ]
        map_a = {
            f"s{i}": {e.query_id: e for e in evidences if e.source_id == f"s{i}"}
            for i in (1, 2)
        }
        map_b = {"s2": map_a["s2"], "s1": map_a["s1"]}
        verdict_a = canonical_json(
            [asdict(v) for v in merge_evidence(map_a).values()]
        )
        verdict_b = canonical_json(
            [asdict(v) for v in merge_evidence(map_b).values()]
        )
        self.assertEqual(verdict_a, verdict_b)


class QuantizationTests(unittest.TestCase):
    def test_floor_truncation_and_saturation(self):
        quantizer = Quantizer(2)  # tick = 100 units
        self.assertEqual(quantizer.quantize(299), 2)
        self.assertEqual(quantizer.quantize(300), 3)
        huge = (SATURATION + 5) * quantizer.scale
        self.assertEqual(quantizer.quantize(huge), SATURATION)
        self.assertEqual(quantizer.saturated_count, 1)

    def test_three_segment_bound_and_dominant(self):
        quantizer = Quantizer(1)
        evidences = [
            _evidence("db", SOURCE_DB, 2, latency=10_000),
            _evidence("app", SOURCE_APP, 3, latency=20_000),
        ]
        evidences[1] = Evidence(
            **{**evidences[1].__dict__, "sample_period_ms": 5000}
        )
        source_a = SourceDeclaration(
            "db", SOURCE_DB, 2, 1, "p95", 1000, {}, "x"
        )
        source_b = SourceDeclaration(
            "app", SOURCE_APP, 3, 1, "avg", 5000, {}, "y"
        )
        from slowquery_diag.models import NormalizedRecord, SourceBatch
        batches = [
            SourceBatch(source_a, (NormalizedRecord(
                "db", SOURCE_DB, 2, 1, 1, 10_000, 5, 1),)),
            SourceBatch(source_b, (NormalizedRecord(
                "app", SOURCE_APP, 3, 1, 1, 20_000, 0, 1),)),
        ]
        bound = build_error_bound(
            quantizer, "latency_us", 1, evidences, batches
        )
        names = {c.name: c.raw_units for c in bound.contributions}
        self.assertEqual(names["quantization"], 18)
        self.assertEqual(names["sampling_period"], 4_000_000)
        self.assertGreater(names["caliber"], 0)
        self.assertEqual(bound.dominant_segment, "sampling_period")
        self.assertEqual(
            bound.total_raw,
            names["quantization"] + names["caliber"] + names["sampling_period"],
        )

    def test_indistinguishable_bound(self):
        self.assertTrue(indistinguishable(100, 101, bound_ticks=2))
        self.assertFalse(indistinguishable(100, 103, bound_ticks=2))


class DiagnosisTests(unittest.TestCase):
    def _candidate(self, cid, gain):
        return CandidateRule(
            cid, "query 1 summary", "SELECT 1", ("db",), gain
        )

    def test_stable_tiebreak_marks_error_affected(self):
        candidates = [
            self._candidate("cand_b", 100), self._candidate("cand_a", 100)
        ]
        winner = _evidence("db", SOURCE_DB, 2, query_id=1, latency=5000)
        verdicts = merge_evidence({"db": {1: winner}})
        quantizer = Quantizer(1)
        bounds = {
            ("latency_us", 1): build_error_bound(
                quantizer, "latency_us", 1, [winner], []
            )
        }
        report = build_report(
            candidates, verdicts, quantizer, bounds, {"db": SOURCE_DB}
        )
        ids = [item.candidate_id for item in report.evaluations]
        self.assertEqual(ids, ["cand_a", "cand_b"])
        self.assertTrue(all(item.error_affected for item in report.evaluations))
        self.assertGreater(
            report.session_ratios["error_indistinguishable"], 0
        )


if __name__ == "__main__":
    unittest.main()
