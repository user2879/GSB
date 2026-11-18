from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quota_app.codec import parse_document
from quota_app.derive import resolve_effective
from quota_app.engine import JudgmentEngine
from quota_app.errors import E_LATENCY, E_PREVIEW
from quota_app.logging_utils import StructuredLogger, read_log
from quota_app.models import AccessEvent, Judgment
from quota_app.preview import PreviewResult, PreviewService

from .helpers import document, event_records


class PreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp.name) / "app.jsonl"
        self.logger = StructuredLogger(self.log_path)
        self.events = [AccessEvent(**record) for record in event_records()]
        self.doc = document()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fast_and_authoritative_match_with_zero_bound(self) -> None:
        service = PreviewService(self.events, logger=self.logger, budget_ms=1000)
        preview, authoritative, comparison = service.run(self.doc)
        self.assertTrue(comparison.ok)
        self.assertEqual(preview.error_bound_single, 0)
        self.assertEqual(preview.error_bound_aggregate, 0)
        self.assertEqual(
            [row.to_json() for row in preview.judgments],
            [row.to_json() for row in authoritative.judgments],
        )

    def test_latency_budget_emits_e_latency_without_raising(self) -> None:
        service = PreviewService(self.events, logger=self.logger, budget_ms=0)
        with patch("quota_app.preview.monotonic_ns", side_effect=[0, 5_000_000]):
            result = service.fast_preview(self.doc)
        self.assertTrue(result.latency_exceeded)
        logs = read_log(self.log_path)
        self.assertEqual(logs[-1]["code"], E_LATENCY)
        self.assertEqual(logs[-1]["details"]["budget_ms"], 0)
        self.assertTrue(logs[-1]["details"]["non_blocking"])

    def test_preview_mismatch_reports_volume_seq_and_fields(self) -> None:
        service = PreviewService(self.events, logger=self.logger, budget_ms=1000)
        preview = service.fast_preview(self.doc)
        authoritative = service.authoritative_replay(self.doc)
        altered = list(authoritative.judgments)
        first = altered[0]
        altered[0] = Judgment(
            seq=first.seq,
            timestamp_ms=first.timestamp_ms,
            volume_id=first.volume_id,
            decision="OVER_QUOTA",
            reason=first.reason,
            operation_count=first.operation_count,
            write_bytes=first.write_bytes,
            cumulative_write_bytes=first.cumulative_write_bytes,
            tokens_scaled=first.tokens_scaled,
            tokens_capacity_scaled=first.tokens_capacity_scaled,
            last_refill_ms=first.last_refill_ms,
            single_error_upper=0,
            aggregate_error_upper=0,
        )
        altered_result = PreviewResult(
            judgments=altered,
            usage=authoritative.usage,
            elapsed_ms=authoritative.elapsed_ms,
            budget_ms=authoritative.budget_ms,
            decision_counts=authoritative.decision_counts,
        )
        comparison = service.compare(preview, altered_result)
        self.assertFalse(comparison.ok)
        self.assertEqual(comparison.mismatches[0]["volume_id"], "root")
        self.assertEqual(comparison.mismatches[0]["seq"], 1)
        self.assertEqual(comparison.mismatches[0]["difference_fields"], ["decision"])
        logs = read_log(self.log_path)
        self.assertEqual(logs[-1]["code"], E_PREVIEW)
        self.assertEqual(logs[-1]["volume_id"], "root")
        self.assertEqual(logs[-1]["field_path"], "event:1")

    def test_requested_volume_differs(self) -> None:
        parsed = parse_document(document_raw_with_small_capacity())
        service = PreviewService(self.events, logger=self.logger, budget_ms=1000)
        preview = service.fast_preview(parsed)
        authoritative = service.authoritative_replay(self.doc)
        comparison = service.compare(preview, authoritative)
        self.assertFalse(comparison.ok)
        self.assertTrue(any(item["seq"] == 2 for item in comparison.mismatches))


def document_raw_with_small_capacity() -> dict:
    from .helpers import document_raw

    raw = document_raw(5)
    raw["volumes"][0]["capacity_bytes"] = 150
    return raw


if __name__ == "__main__":
    unittest.main()
