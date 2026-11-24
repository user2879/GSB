from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mmledger.conversion import SidecarStore, convert_receipt
from mmledger.errors import (
    E_DUPLICATE_BATCH_ID,
    E_DURATION_NONFINITE,
    E_RECEIPT_LAYOUT,
    E_TRUNCATED_RECEIPT,
    E_UNKNOWN_PARAMETER_VERSION,
)
from mmledger.jsonutil import MISSING, is_missing
from mmledger.models import MergeRules, R1, R2, R3, Receipt, RequestSpec
from mmledger.normalization import batch_input_hash, normalize_batch, resolve_ownership
from mmledger.receipts import parse_receipt_paths

from tests.helpers import write_json


def receipt(
    *,
    batch_id: str = "b1",
    request_id: str = "q1",
    output_digest: str = "d1",
    worker_id: str | None = "w1",
    parameter_version: int | None = 3,
    duration_ms: int | None = 10,
    stable_rank: int | None = 1,
    overwritten_ref: str | None = None,
    parameter_digest: str | None = "pd3",
    version: str = R3,
    uid: str = "source#0",
) -> Receipt:
    return Receipt(
        batch_id=batch_id,
        request_id=request_id,
        output_digest=output_digest,
        parameter_version=MISSING if parameter_version is None else parameter_version,
        duration_ms=MISSING if duration_ms is None else duration_ms,
        worker_id=MISSING if worker_id is None else worker_id,
        stable_rank=MISSING if stable_rank is None else stable_rank,
        overwritten_ref=MISSING if version in (R1, R2) else overwritten_ref,
        parameter_digest=MISSING if parameter_digest is None else parameter_digest,
        format_version=version,
        source_version=version,
        uid=uid,
        source_file="source.json",
    )


class ReceiptParsingTests(unittest.TestCase):
    def test_three_versions_share_one_internal_field_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(
                root / "r1.json",
                {"version": "r1", "receipts": [{"batch_id": "b1", "request_id": "q1", "output_digest": "d1"}]},
            )
            write_json(
                root / "r2.json",
                {
                    "version": "r2",
                    "receipts": [
                        {
                            "batch_id": "b1",
                            "request_id": "q2",
                            "output_digest": "d2",
                            "parameter_version": 3,
                            "duration_ms": 7,
                            "worker_id": "w1",
                        }
                    ],
                },
            )
            write_json(
                root / "r3.json",
                {
                    "version": "r3",
                    "receipts": [
                        {
                            "batch_id": "b1",
                            "request_id": "q3",
                            "output_digest": "d3",
                            "parameter_version": 3,
                            "duration_ms": 8,
                            "worker_id": "w1",
                            "stable_rank": 3,
                            "overwritten_ref": "b1:q2:w1:2",
                            "parameter_digest": "pd3",
                        }
                    ],
                },
            )
            parsed = parse_receipt_paths(
                [root / "r1.json", root / "r2.json", root / "r3.json"],
                base_dir=root,
                known_versions=frozenset({3, 4}),
            )
            self.assertEqual(parsed.failures, ())
            self.assertEqual(len(parsed.receipts), 3)
            by_version = {item.source_version: item for item in parsed.receipts}
            self.assertEqual(by_version[R1].missing_fields, ("parameter_version", "duration_ms", "worker_id", "stable_rank", "overwritten_ref", "parameter_digest"))
            self.assertEqual(by_version[R2].missing_fields, ("stable_rank", "overwritten_ref", "parameter_digest"))
            self.assertEqual(by_version[R3].missing_fields, ())
            self.assertIs(by_version[R1].duration_ms, MISSING)
            self.assertNotEqual(by_version[R1].as_fields()["duration_ms"], 0)
            self.assertEqual(set(by_version[R1].as_fields()), set(by_version[R3].as_fields()))
            self.assertEqual(by_version[R1].completeness, "partial")
            self.assertEqual(by_version[R3].completeness, "complete")

    def test_bad_records_are_isolated_and_codes_are_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(
                root / "good.json",
                {
                    "version": "r3",
                    "receipts": [
                        {
                            "batch_id": "b1",
                            "request_id": "q1",
                            "output_digest": "d1",
                            "parameter_version": 3,
                            "duration_ms": 4,
                            "worker_id": "w1",
                            "stable_rank": 1,
                            "overwritten_ref": None,
                            "parameter_digest": "pd3",
                        }
                    ],
                },
            )
            write_json(
                root / "bad_layout.json",
                {"version": "r1", "receipts": [{"batch_id": "b2", "request_id": "q2"}]},
            )
            write_json(
                root / "unknown_param.json",
                {
                    "version": "r2",
                    "receipts": [
                        {
                            "batch_id": "b3",
                            "request_id": "q3",
                            "output_digest": "d3",
                            "parameter_version": 99,
                            "duration_ms": 4,
                            "worker_id": "w1",
                        }
                    ],
                },
            )
            (root / "truncated.json").write_text('{"version":"r3","receipts":[', encoding="utf-8")
            nonfinite = {
                "version": "r2",
                "receipts": [
                    {
                        "batch_id": "b4",
                        "request_id": "q4",
                        "output_digest": "d4",
                        "parameter_version": 3,
                        "duration_ms": 5,
                        "worker_id": "w1",
                    }
                ],
            }
            text = json.dumps(nonfinite, indent=2).replace('"duration_ms": 5', '"duration_ms": NaN')
            (root / "nonfinite.json").write_text(text, encoding="utf-8")
            write_json(
                root / "duplicate.json",
                {
                    "version": "r2",
                    "receipts": [
                        {
                            "batch_id": "b5",
                            "request_id": "q5",
                            "output_digest": "d5",
                            "parameter_version": 3,
                            "duration_ms": 5,
                            "worker_id": "w1",
                        },
                        {
                            "batch_id": "b5",
                            "request_id": "q5",
                            "output_digest": "d5",
                            "parameter_version": 3,
                            "duration_ms": 5,
                            "worker_id": "w1",
                        },
                    ],
                },
            )
            parsed = parse_receipt_paths(
                sorted(root.glob("*.json")),
                base_dir=root,
                known_versions=frozenset({3, 4}),
            )
            self.assertEqual(len(parsed.receipts), 2)  # good r3 + first duplicate record
            codes = {failure.code for failure in parsed.failures}
            self.assertIn(E_RECEIPT_LAYOUT, codes)
            self.assertIn(E_UNKNOWN_PARAMETER_VERSION, codes)
            self.assertIn(E_TRUNCATED_RECEIPT, codes)
            self.assertIn(E_DURATION_NONFINITE, codes)
            self.assertIn(E_DUPLICATE_BATCH_ID, codes)
            self.assertEqual([failure.code for failure in parsed.failures], [failure.code for failure in sorted(parsed.failures, key=lambda item: item.sort_key())])


class ConversionTests(unittest.TestCase):
    def test_r1_to_r3_marks_missing_and_r3_to_r1_roundtrips(self) -> None:
        source = receipt(uid="r3#0")
        sidecar = SidecarStore()
        down, down_event, failure = convert_receipt(source, R1, sidecar, sequence=1)
        self.assertIsNone(failure)
        assert down is not None and down_event is not None
        self.assertEqual(down.format_version, R1)
        self.assertTrue(is_missing(down.duration_ms))
        self.assertEqual(down.completeness, "partial")
        self.assertIn("duration_ms", down_event.downgraded_fields)
        restored, up_event, failure2 = convert_receipt(down, R3, sidecar, sequence=2)
        self.assertIsNone(failure2)
        assert restored is not None and up_event is not None
        for name in (
            "batch_id",
            "request_id",
            "output_digest",
            "parameter_version",
            "duration_ms",
            "worker_id",
            "stable_rank",
            "overwritten_ref",
            "parameter_digest",
        ):
            self.assertEqual(getattr(source, name), getattr(restored, name), name)
        self.assertEqual(restored.completeness, "complete")
        self.assertEqual(up_event.irreversible_fields, ())

    def test_r1_upgrade_without_sidecar_is_explicitly_partial(self) -> None:
        source = receipt(version=R1, uid="r1#0", parameter_version=None, duration_ms=None, worker_id=None, stable_rank=None, parameter_digest=None)
        converted, event, failure = convert_receipt(source, R3, SidecarStore(), sequence=1)
        self.assertIsNone(failure)
        assert converted is not None and event is not None
        self.assertTrue(is_missing(converted.duration_ms))
        self.assertTrue(is_missing(converted.parameter_version))
        self.assertEqual(converted.completeness, "partial")
        self.assertIn("duration_ms", event.irreversible_fields)
        self.assertIn("parameter_digest", event.irreversible_fields)

    def test_pruned_sidecar_reports_irreversible_field(self) -> None:
        source = receipt(uid="r3#0")
        sidecar = SidecarStore()
        down, _, _ = convert_receipt(source, R1, sidecar, sequence=1)
        assert down is not None
        restored, event, failure = convert_receipt(
            down,
            R3,
            sidecar,
            sequence=2,
            prune_fields=("stable_rank",),
        )
        self.assertIsNone(failure)
        assert restored is not None and event is not None
        self.assertTrue(is_missing(restored.stable_rank))
        self.assertIn("stable_rank", event.irreversible_fields)

    def test_prune_during_downgrade_is_marked_irreversible(self) -> None:
        source = receipt(uid="r3#0")
        sidecar = SidecarStore()
        down, event, failure = convert_receipt(
            source,
            R1,
            sidecar,
            sequence=1,
            prune_fields=("parameter_digest",),
        )
        self.assertIsNone(failure)
        assert down is not None and event is not None
        self.assertIn("parameter_digest", event.irreversible_fields)

    def test_format_roundtrip_does_not_change_ownership_decision(self) -> None:
        merge_rules = MergeRules(
            worker_sets={"active": ("w1",)},
            parameter_version_sets={"production": (3, 4)},
            active_worker_set="active",
            active_parameter_version_set="production",
            worker_priority=("w1",),
        )
        source = receipt(batch_id="batch-a", parameter_version=3, duration_ms=20, uid="a#0", output_digest="da")
        competitor = receipt(batch_id="batch-b", parameter_version=4, duration_ms=30, uid="b#0", output_digest="db")
        request_a = RequestSpec(1, "q1", "image_text", 1, "batch-a", "w1", 11)
        request_b = RequestSpec(2, "q1", "image_text", 2, "batch-b", "w1", 12)

        def decide(source_receipt: Receipt):
            batches = []
            for request, item in ((request_a, source_receipt), (request_b, competitor)):
                batch, failures = normalize_batch(
                    request.batch_id,
                    [request],
                    [item],
                    merge_rules=merge_rules,
                    input_hash=batch_input_hash([request], [item]),
                )
                self.assertEqual(failures, [])
                batches.append(batch)
            result = resolve_ownership(batches, [request_a, request_b], merge_rules)
            self.assertFalse(result.failures)
            return result.decisions[0]

        before = decide(source)
        sidecar = SidecarStore()
        down, _, _ = convert_receipt(source, R1, sidecar, sequence=1)
        assert down is not None
        restored, _, _ = convert_receipt(down, R3, sidecar, sequence=2)
        assert restored is not None
        after = decide(restored)
        self.assertEqual(before.as_dict(), after.as_dict())
        self.assertEqual(after.batch_id, "batch-b")


if __name__ == "__main__":
    unittest.main()
