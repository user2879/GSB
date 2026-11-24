from __future__ import annotations

import unittest

from tlschain.errors import DomainError
from tlschain.spec import load_spec

from tests.helpers import (
    CERT_PATH,
    HANDSHAKE_PATH,
    SAMPLES,
    load_json,
    temporary_directory,
    write_json,
)


class SpecValidationTests(unittest.TestCase):
    def _assert_mutation_error(
        self,
        cert_mutator,
        expected_cert_id: str | None,
        field_fragment: str,
        handshake_mutator=None,
    ) -> DomainError:
        directory = temporary_directory("tlschain-spec-")
        cert_doc = load_json(CERT_PATH)
        handshake_doc = load_json(HANDSHAKE_PATH)
        cert_mutator(cert_doc)
        if handshake_mutator is not None:
            handshake_mutator(handshake_doc)
        cert_path = directory / "certificates.json"
        handshake_path = directory / "handshake.json"
        write_json(cert_path, cert_doc)
        write_json(handshake_path, handshake_doc)
        with self.assertRaises(DomainError) as raised:
            load_spec(cert_path, handshake_path)
        error = raised.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.cert_id, expected_cert_id)
        self.assertIn(field_fragment, error.field_path or "")
        return error

    def test_valid_sample_loads_and_matches_expected_results(self) -> None:
        spec = load_spec(CERT_PATH, HANDSHAKE_PATH)
        self.assertEqual(len(spec.certificates), 9)
        self.assertEqual(len(spec.messages), 8)
        self.assertEqual(spec.fixed.scale, 1_000_000)
        self.assertEqual(spec.trust_anchors, ("root-ca",))

    def test_duplicate_certificate_id(self) -> None:
        def mutate(document):
            document["certificates"].append(dict(document["certificates"][0]))

        self._assert_mutation_error(mutate, "root-ca", "certificates.id")

    def test_missing_issuer_reference(self) -> None:
        def mutate(document):
            for cert in document["certificates"]:
                if cert["id"] == "leaf-server":
                    cert["issuer"] = "missing-ca"

        self._assert_mutation_error(mutate, "leaf-server", "issuer")

    def test_non_positive_serial_and_modulus(self) -> None:
        def zero_serial(document):
            document["certificates"][1]["serial"] = 0

        self._assert_mutation_error(zero_serial, "intermediate-ca", "serial")

        def negative_modulus(document):
            document["certificates"][1]["modulus"] = -7

        self._assert_mutation_error(negative_modulus, "intermediate-ca", "modulus")

    def test_invalid_validity_window(self) -> None:
        def mutate(document):
            document["certificates"][4]["not_before_ms"] = document["certificates"][4]["not_after_ms"]

        self._assert_mutation_error(mutate, "leaf-server", "not_before_ms")

    def test_unsupported_algorithm(self) -> None:
        def mutate(document):
            document["certificates"][4]["algorithm"] = "REAL_RSA"

        self._assert_mutation_error(mutate, "leaf-server", "algorithm")

    def test_self_cycle_and_mutual_cycle(self) -> None:
        def self_cycle(document):
            document["certificates"][1]["issuer"] = "intermediate-ca"

        self._assert_mutation_error(self_cycle, "intermediate-ca", "issuer")

        def mutual_cycle(document):
            document["certificates"][0]["issuer"] = "intermediate-ca"

        self._assert_mutation_error(mutual_cycle, "root-ca", "issuer")

    def test_handshake_sequence_and_arrival_validation(self) -> None:
        def bad_sequence(document):
            document["messages"][1]["seq"] = 7

        self._assert_mutation_error(
            lambda document: None,
            "leaf-server",
            "messages[1].seq",
            bad_sequence,
        )

        def decreasing_arrival(document):
            document["messages"][2]["arrival_ms"] = document["messages"][0]["arrival_ms"] - 1

        self._assert_mutation_error(
            lambda document: None,
            "leaf-server",
            "arrival_ms",
            decreasing_arrival,
        )

        def bad_client_seq(document):
            document["messages"][2]["client_seq"] = 9

        self._assert_mutation_error(
            lambda document: None,
            "leaf-server",
            "client_seq",
            bad_client_seq,
        )

    def test_duplicate_json_key_is_rejected(self) -> None:
        directory = temporary_directory("tlschain-duplicate-")
        raw = CERT_PATH.read_text(encoding="utf-8")
        raw = raw.replace('"format_version": 1,', '"format_version": 1, "format_version": 1,', 1)
        cert_path = directory / "certificates.json"
        cert_path.write_text(raw, encoding="utf-8")
        with self.assertRaises(DomainError) as raised:
            load_spec(cert_path, HANDSHAKE_PATH)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertIn("format_version", raised.exception.field_path or "")

    def test_corrupted_and_self_cycle_command_anomalies(self) -> None:
        with self.assertRaises(DomainError) as corrupted:
            load_spec(CERT_PATH, HANDSHAKE_PATH, "corrupted-json")
        self.assertEqual(corrupted.exception.code, "E_SPEC")
        self.assertIn("corrupted", corrupted.exception.message)
        with self.assertRaises(DomainError) as cycle:
            load_spec(CERT_PATH, HANDSHAKE_PATH, "self-cycle")
        self.assertEqual(cycle.exception.code, "E_SPEC")
        self.assertEqual(cycle.exception.cert_id, "intermediate-ca")

    def test_shipped_invalid_samples(self) -> None:
        with self.assertRaises(DomainError) as bad_algorithm:
            load_spec(SAMPLES / "certificates_bad_algorithm.json", HANDSHAKE_PATH)
        self.assertEqual(bad_algorithm.exception.cert_id, "leaf-server")
        with self.assertRaises(DomainError) as bad_sequence:
            load_spec(CERT_PATH, SAMPLES / "handshake_bad_seq.json")
        self.assertEqual(bad_sequence.exception.code, "E_SPEC")
        self.assertIsNotNone(bad_sequence.exception.message_seq)


if __name__ == "__main__":
    unittest.main()
