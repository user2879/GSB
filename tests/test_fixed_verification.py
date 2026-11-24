from __future__ import annotations

import unittest

from tlschain.session import build_full_session
from tlschain.spec import load_spec

from tests.helpers import CERT_PATH, HANDSHAKE_PATH, load_json, temporary_directory, write_json


class FixedVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec(CERT_PATH, HANDSHAKE_PATH)
        self.session = build_full_session(self.spec)

    def test_decision_counts_and_expected_results(self) -> None:
        self.assertEqual(self.session.decision_counts(), {"PASS": 5, "FAIL": 1, "INDETERMINATE": 2})
        self.assertEqual(self.session.expected_mismatch_seqs(), [])
        self.assertEqual(self.session.affected_count(), 2)

    def test_error_decomposition_and_cumulative_bounds(self) -> None:
        record = self.session.message_records[1]
        bound = record["selected_chain"]["error_bound"]
        self.assertEqual(bound["modulus_quantization"], 6)
        self.assertEqual(bound["time_window"], 6)
        self.assertEqual(bound["path_length"], 1)
        self.assertEqual(bound["total"], 13)
        certificates = record["selected_chain"]["certificates"]
        self.assertEqual([item["step_error_bound"]["cumulative"] for item in certificates], [4, 8, 12])
        self.assertTrue(all(item["signature_state"] == "PASS" for item in certificates))
        self.assertTrue(all(item["window_state"] == "PASS" for item in certificates))

    def test_boundary_window_is_indeterminate_and_affected(self) -> None:
        record = self.session.message_records[5]
        self.assertEqual(record["decision"], "INDETERMINATE")
        self.assertTrue(record["affected_by_error"])
        leaf = record["selected_chain"]["certificates"][0]
        self.assertEqual(leaf["cert_id"], "leaf-edge")
        self.assertEqual(leaf["window_state"], "INDETERMINATE")
        self.assertEqual(leaf["min_time_margin_q"], 0)

    def test_equal_candidate_scores_use_stable_tie_break(self) -> None:
        record = self.session.message_records[6]
        self.assertEqual(record["decision"], "INDETERMINATE")
        self.assertTrue(record["affected_by_error"])
        self.assertEqual(record["selected_cert_ids"][0], "leaf-b-1")
        self.assertEqual(record["selection_reason"], "score-difference-within-composite-error-bound")
        self.assertEqual(record["pairwise_indistinguishability_bound"], 26)
        self.assertEqual(record["candidates"][0]["score_vector"], record["candidates"][1]["score_vector"])

    def test_expired_chain_fails_on_window(self) -> None:
        record = self.session.message_records[4]
        self.assertEqual(record["decision"], "FAIL")
        self.assertEqual(record["selected_chain"]["certificates"][0]["cert_id"], "leaf-expired")
        self.assertEqual(record["selected_chain"]["certificates"][0]["window_state"], "FAIL")

    def test_tampered_signature_changes_decision(self) -> None:
        directory = temporary_directory("tlschain-signature-")
        document = load_json(CERT_PATH)
        for cert in document["certificates"]:
            if cert["id"] == "leaf-server":
                cert["signature"] += 1
                break
        path = directory / "certificates.json"
        write_json(path, document)
        spec = load_spec(path, HANDSHAKE_PATH)
        session = build_full_session(spec)
        first = session.message_records[1]
        self.assertEqual(first["decision"], "FAIL")
        self.assertEqual(first["selected_chain"]["certificates"][0]["signature_state"], "FAIL")

    def test_trust_anchor_mismatch_fails(self) -> None:
        directory = temporary_directory("tlschain-anchor-")
        handshake = load_json(HANDSHAKE_PATH)
        for message in handshake["messages"]:
            if message["seq"] in (1, 2):
                message["cert_ids"] = ["leaf-server", "intermediate-ca"]
        path = directory / "handshake.json"
        write_json(path, handshake)
        session = build_full_session(load_spec(CERT_PATH, path))
        self.assertEqual(session.message_records[1]["decision"], "FAIL")
        self.assertFalse(session.message_records[1]["selected_chain"]["anchor_match"])

    def test_saturation_is_explicit_and_forces_indeterminate(self) -> None:
        directory = temporary_directory("tlschain-saturation-")
        document = load_json(CERT_PATH)
        document["fixed"]["max_bits"] = 16
        document["fixed"]["scale"] = 65535
        path = directory / "certificates.json"
        write_json(path, document)
        session = build_full_session(load_spec(path, HANDSHAKE_PATH))
        record = session.message_records[1]
        self.assertEqual(record["decision"], "INDETERMINATE")
        self.assertTrue(record["affected_by_error"])
        self.assertTrue(record["selected_chain"]["score_saturated"])
        self.assertTrue(
            all(cert["saturation_applied"] for cert in record["selected_chain"]["certificates"])
        )


if __name__ == "__main__":
    unittest.main()
