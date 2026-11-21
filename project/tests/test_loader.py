from __future__ import annotations

import unittest

from arbitration_workbench.loader import load_evidence_file, load_evidence_text
from arbitration_workbench.numeric import read_json
from arbitration_workbench.rules import parse_rules

from .helpers import ROOT


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = parse_rules(read_json(ROOT / "samples" / "rules.json"))

    def test_valid_sample_loads_48_tickets(self):
        loaded = load_evidence_file(ROOT / "samples" / "evidence.jsonl", self.rules)
        self.assertEqual(48, len(loaded.evidence_by_ticket))
        self.assertEqual([], loaded.failures)
        self.assertEqual(48, len(loaded.ticket_ids))

    def test_invalid_sample_emits_all_required_codes_and_excludes_tickets(self):
        loaded = load_evidence_file(ROOT / "samples" / "evidence_invalid.jsonl", self.rules)
        codes = {failure["code"] for failure in loaded.failures}
        self.assertIn("E_JSON_INVALID", codes)
        self.assertIn("E_EVIDENCE_TYPE_UNKNOWN", codes)
        self.assertIn("E_ROLE_UNKNOWN", codes)
        self.assertIn("E_TIMESTAMP_INVALID", codes)
        self.assertIn("E_DUPLICATE_EVIDENCE", codes)
        for ticket_id in ("T9001", "T9002", "T9003", "T9004", "T9005"):
            self.assertNotIn(ticket_id, loaded.evidence_by_ticket)
        self.assertIn("T9006", loaded.evidence_by_ticket)
        self.assertEqual(2, len(loaded.evidence_by_ticket["T9006"]))
        ordered = loaded.failures
        self.assertEqual(ordered, sorted(ordered, key=lambda item: (item["ticket_id"], item["line_no"], item["stage"], item["code"])))

    def test_all_timestamp_equal_ticket_is_isolated(self):
        text = "\n".join(
            [
                '{"ticket_id":"TX","evidence_seq":1,"evidence_type":"failed_login","source_role":"external","timestamp":10,"terms":[]}',
                '{"ticket_id":"TX","evidence_seq":2,"evidence_type":"mfa_push","source_role":"backend","timestamp":10,"terms":[]}',
                '{"ticket_id":"TY","evidence_seq":1,"evidence_type":"failed_login","source_role":"external","timestamp":10,"terms":[]}',
                '{"ticket_id":"TY","evidence_seq":2,"evidence_type":"mfa_push","source_role":"backend","timestamp":11,"terms":[]}',
            ]
        )
        loaded = load_evidence_text(text, self.rules)
        self.assertNotIn("TX", loaded.evidence_by_ticket)
        self.assertIn("TY", loaded.evidence_by_ticket)
        self.assertTrue(any(f["ticket_id"] == "TX" and f["code"] == "E_TIMESTAMP_INVALID" for f in loaded.failures))


if __name__ == "__main__":
    unittest.main()
