import unittest

from receipt_ledger.attribution import attribute
from receipt_ledger.convert import receipt_ref
from receipt_ledger.errors import E_ATTRIBUTION_UNRESOLVED
from receipt_ledger.models import LedgerEntry, Worker
from receipt_ledger.receipts import parse_receipt_line
from receipt_ledger.resources import build_batch_resources, propagate_invalidation
from receipt_ledger.util import FIXED_SCALE


def receipt(text, line=1):
    parsed, failure = parse_receipt_line(text, "receipt_r3.jsonl", line, "r3")
    assert failure is None, failure
    return parsed


class AttributionTests(unittest.TestCase):
    def test_override_then_ranking(self):
        old = receipt(
            '{"version":"r3","batch_id":"b-old","request_id":"q","output_digest":"d-old",'
            '"param_version":1,"duration_ms":90,"worker_id":"w","rank_in_batch":0,'
            '"supersedes":[],"param_summary":"p1"}'
        )
        new = receipt(
            '{"version":"r3","batch_id":"b-new","request_id":"q","output_digest":"d-new",'
            '"param_version":1,"duration_ms":40,"worker_id":"w","rank_in_batch":0,'
            f'"supersedes":["{receipt_ref(old)}"],"param_summary":"p1"}}'
        )
        entries, failures, debug = attribute(
            [old, new], set(), {"q": 1}
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].batch_id, "b-new")
        self.assertIn("override:", entries[0].basis)
        self.assertEqual(entries[0].superseded_refs, (receipt_ref(old),))
        self.assertIn(receipt_ref(old), debug["superseded_refs"])

    def test_no_override_uses_version_desc_then_batch_asc(self):
        first = receipt(
            '{"version":"r3","batch_id":"b-a","request_id":"q","output_digest":"d1",'
            '"param_version":3,"duration_ms":100,"worker_id":"w","rank_in_batch":0,'
            '"supersedes":[],"param_summary":"p"}'
        )
        second = receipt(
            '{"version":"r3","batch_id":"b-b","request_id":"q","output_digest":"d2",'
            '"param_version":4,"duration_ms":120,"worker_id":"w","rank_in_batch":0,'
            '"supersedes":[],"param_summary":"p"}'
        )
        entries, _, _ = attribute([first, second], set(), {"q": 1})
        self.assertEqual(entries[0].batch_id, "b-b")

    def test_all_unavailable_is_unresolved_but_others_kept(self):
        bad = receipt(
            '{"version":"r3","batch_id":"b","request_id":"q-bad","output_digest":"d",'
            '"param_version":9,"duration_ms":10,"worker_id":"w","rank_in_batch":0,'
            '"supersedes":[],"param_summary":"p"}'
        )
        good = receipt(
            '{"version":"r3","batch_id":"b2","request_id":"q-good","output_digest":"d2",'
            '"param_version":1,"duration_ms":10,"worker_id":"w","rank_in_batch":0,'
            '"supersedes":[],"param_summary":"p"}'
        )
        entries, failures, _ = attribute(
            [bad, good], {("w", 9)}, {"q-bad": 1, "q-good": 2}
        )
        self.assertEqual([entry.request_id for entry in entries], ["q-good"])
        self.assertEqual(failures[0].code, E_ATTRIBUTION_UNRESOLVED)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.workers = (
            Worker("w", 9000, 4, 8, 3, "int8"),
        )

    def test_fixed_point_metrics_and_invalidation_path(self):
        rows = [
            receipt(
                f'{{"version":"r3","batch_id":"b","request_id":"q{i}","output_digest":"d{i}",'
                '"param_version":1,"duration_ms":10,"worker_id":"w","rank_in_batch":'
                f'{i},"supersedes":[],"param_summary":"p"}}',
                line=i + 1,
            )
            for i in range(2)
        ]
        resources = build_batch_resources(rows, self.workers, set(), {"q0": 3, "q1": 3})
        self.assertEqual(len(resources), 1)
        resource = resources[0]
        self.assertTrue(resource.invalid)
        self.assertEqual(
            list(resource.propagation_path),
            ["worker:w", "param_version:v1->v3", "batch:b"],
        )
        self.assertEqual(resource.utilization_fixed, FIXED_SCALE)
        self.assertGreater(resource.macs_fixed, 0)

        entries = [
            LedgerEntry(
                request_id="q0", seq=1, batch_id="b", worker_id="w", param_version=1,
                source_version="r3", duration_ms=10, output_digest="d0",
                rank_in_batch=0, integrity_tag="t", basis="x", candidate_count=1,
                superseded_refs=(), receipt_ref="r", invalid=False, invalid_reason=None,
                propagation_path=(),
            )
        ]
        updated, reuse = propagate_invalidation(entries, resources)
        self.assertTrue(updated[0].invalid)
        self.assertEqual(
            list(updated[0].propagation_path[-2:]), ["request:q0", "attribution"]
        )
        self.assertEqual(reuse[0].action, "invalidate")


if __name__ == "__main__":
    unittest.main()
