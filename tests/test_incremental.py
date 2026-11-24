from __future__ import annotations

import unittest

from tlschain.canonical import canonical_text
from tlschain.session import build_full_session, compare_incremental_with_full, recompute_session
from tlschain.spec import load_spec

from tests.helpers import (
    CERT_PATH,
    HANDSHAKE_PATH,
    MODIFIED_CERT_PATH,
    load_json,
    temporary_directory,
    write_json,
)


class IncrementalTests(unittest.TestCase):
    def test_incremental_equals_full_rebuild(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        changed_spec = load_spec(MODIFIED_CERT_PATH, HANDSHAKE_PATH)
        incremental = recompute_session(base, changed_spec)
        full = build_full_session(changed_spec)
        comparison = compare_incremental_with_full(incremental, full)
        self.assertTrue(comparison["equal"])
        self.assertEqual(incremental.decision_counts(), full.decision_counts())
        self.assertEqual(incremental.certificate_hashes(), full.certificate_hashes())
        self.assertEqual(incremental.message_hashes(), full.message_hashes())

    def test_invalidation_propagates_certificate_to_messages(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        incremental = recompute_session(base, load_spec(MODIFIED_CERT_PATH, HANDSHAKE_PATH))
        self.assertEqual(incremental.invalidated_cert_ids, ["leaf-expired"])
        self.assertEqual(incremental.invalidated_message_seqs, [4, 5, 6, 7, 8])
        self.assertEqual(incremental.cert_invalidation[0]["reason"], "direct_change:not_after_ms,signature")
        self.assertEqual(
            incremental.message_invalidation[0]["reason"],
            "carries_affected_certificate:leaf-expired",
        )
        self.assertEqual(
            incremental.message_invalidation[1]["reason"],
            "follows_affected_message:4",
        )

    def test_retained_bytes_are_identical(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        incremental = recompute_session(base, load_spec(MODIFIED_CERT_PATH, HANDSHAKE_PATH))
        retained_certificates = sorted(set(base.cert_bases) - set(incremental.invalidated_cert_ids))
        retained_messages = sorted(set(base.message_records) - set(incremental.invalidated_message_seqs))
        for cert_id in retained_certificates:
            self.assertEqual(
                canonical_text(base.cert_bases[cert_id]),
                canonical_text(incremental.cert_bases[cert_id]),
            )
        for seq in retained_messages:
            self.assertEqual(
                canonical_text(base.message_records[seq]),
                canonical_text(incremental.message_records[seq]),
            )
        self.assertEqual(retained_messages, [1, 2, 3])

    def test_fixed_parameter_change_invalidates_everything(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        directory = temporary_directory("tlschain-scale-")
        document = load_json(CERT_PATH)
        document["fixed"]["scale"] = 1_000_001
        path = directory / "certificates.json"
        write_json(path, document)
        changed_spec = load_spec(path, HANDSHAKE_PATH)
        incremental = recompute_session(base, changed_spec)
        full = build_full_session(changed_spec)
        self.assertEqual(set(incremental.invalidated_cert_ids), {cert.cert_id for cert in changed_spec.certificates})
        self.assertEqual(incremental.invalidated_message_seqs, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(incremental.cert_invalidation[0]["reason"], "fixed_parameters_changed")
        self.assertTrue(compare_incremental_with_full(incremental, full)["equal"])

    def test_issuer_change_propagates_through_new_reference_graph(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        directory = temporary_directory("tlschain-issuer-")
        document = load_json(CERT_PATH)
        for cert in document["certificates"]:
            if cert["id"] == "intermediate-ca":
                cert["issuer"] = "intermediate-b-1"
                break
        path = directory / "certificates.json"
        write_json(path, document)
        changed = load_spec(path, HANDSHAKE_PATH)
        incremental = recompute_session(base, changed)
        self.assertIn("intermediate-ca", incremental.invalidated_cert_ids)
        self.assertIn("leaf-server", incremental.invalidated_cert_ids)
        self.assertIn("leaf-expired", incremental.invalidated_cert_ids)
        reasons = {entry["id"]: entry["reason"] for entry in incremental.cert_invalidation}
        self.assertIn("direct_change:issuer", reasons["intermediate-ca"])
        self.assertEqual(reasons["leaf-server"], "issuer_descendant:intermediate-ca")
        self.assertTrue(compare_incremental_with_full(incremental, build_full_session(changed))["equal"])

    def test_modulus_change_invalidates_own_and_cascading_messages(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        directory = temporary_directory("tlschain-modulus-")
        document = load_json(CERT_PATH)
        for cert in document["certificates"]:
            if cert["id"] == "leaf-server":
                cert["modulus"] += 2
                break
        path = directory / "certificates.json"
        write_json(path, document)
        changed = load_spec(path, HANDSHAKE_PATH)
        incremental = recompute_session(base, changed)
        self.assertIn("leaf-server", incremental.invalidated_cert_ids)
        self.assertEqual(incremental.invalidated_message_seqs, [1, 2, 3, 4, 5, 6, 7, 8])
        reasons = {entry["id"]: entry["reason"] for entry in incremental.cert_invalidation}
        self.assertEqual(reasons["leaf-server"], "direct_change:modulus")
        self.assertTrue(compare_incremental_with_full(incremental, build_full_session(changed))["equal"])

    def test_topological_order_starts_at_root_for_global_change(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        directory = temporary_directory("tlschain-root-")
        document = load_json(CERT_PATH)
        document["certificates"][0]["not_before_ms"] -= 1
        path = directory / "certificates.json"
        write_json(path, document)
        changed = load_spec(path, HANDSHAKE_PATH)
        incremental = recompute_session(base, changed)
        self.assertEqual(incremental.cert_invalidation[0]["id"], "root-ca")
        self.assertEqual(sorted(incremental.invalidated_cert_ids), sorted(cert.cert_id for cert in changed.certificates))

    def test_message_edit_recomputes_only_that_message_and_successors(self) -> None:
        base = build_full_session(load_spec(CERT_PATH, HANDSHAKE_PATH))
        directory = temporary_directory("tlschain-message-edit-")
        handshake = load_json(HANDSHAKE_PATH)
        handshake["messages"][5]["arrival_ms"] = handshake["messages"][5]["arrival_ms"] + 5
        path = directory / "handshake.json"
        write_json(path, handshake)
        changed = load_spec(CERT_PATH, path)
        incremental = recompute_session(base, changed)
        self.assertEqual(incremental.invalidated_cert_ids, [])
        self.assertEqual(incremental.invalidated_message_seqs, [6, 7, 8])
        self.assertEqual(incremental.message_invalidation[0]["reason"], "direct_message_change")
        self.assertTrue(compare_incremental_with_full(incremental, build_full_session(changed))["equal"])

    def test_trust_anchor_change_invalidates_all_messages(self) -> None:
        directory = temporary_directory("tlschain-anchor-edit-")
        document = load_json(CERT_PATH)
        extra_anchor = dict(document["certificates"][4])
        extra_anchor["id"] = "leaf-extra-anchor"
        extra_anchor["issuer"] = "root-ca"
        extra_anchor["serial"] = 3999
        document["certificates"].append(extra_anchor)
        base_path = directory / "certificates_base.json"
        write_json(base_path, document)
        changed_document = load_json(base_path)
        changed_document["trust_anchors"] = ["root-ca", "leaf-extra-anchor"]
        changed_path = directory / "certificates_changed.json"
        write_json(changed_path, changed_document)
        base = build_full_session(load_spec(base_path, HANDSHAKE_PATH))
        changed = load_spec(changed_path, HANDSHAKE_PATH)
        incremental = recompute_session(base, changed)
        self.assertEqual(incremental.invalidated_cert_ids, [])
        self.assertEqual(incremental.invalidated_message_seqs, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(incremental.message_invalidation[0]["reason"], "trust_anchors_changed")
        self.assertTrue(compare_incremental_with_full(incremental, build_full_session(changed))["equal"])


if __name__ == "__main__":
    unittest.main()
