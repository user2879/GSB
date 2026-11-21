from __future__ import annotations

import hashlib
import unittest

from txn_mw.canonical import body_digest
from txn_mw.codec import HEADER, decode_record, encode_record, validate_sequence
from txn_mw.errors import ProtocolError


def payload(**overrides):
    body = overrides.pop("body", {"order_id": "A-1"})
    data = overrides.pop("data", {"body": body})
    value = {
        "txn_id": "txn-1",
        "version": 1,
        "state": "half",
        "event_seq": 1,
        "body_digest": body_digest(body),
        "type": "half",
        "data": data,
    }
    value.update(overrides)
    return value


class CodecTests(unittest.TestCase):
    def test_round_trip(self):
        encoded = encode_record(payload())
        decoded = decode_record(encoded)
        self.assertEqual(decoded["txn_id"], "txn-1")
        self.assertEqual(decoded["event_seq"], 1)
        self.assertEqual(decoded["state"], "half")

    def test_missing_mandatory_field_reports_path(self):
        value = payload()
        value.pop("version")
        with self.assertRaises(ProtocolError) as caught:
            encode_record(value)
        self.assertEqual(caught.exception.path, "$.record.version")
        self.assertEqual(caught.exception.code, "E_TXN")

    def test_unknown_state_is_rejected(self):
        with self.assertRaises(ProtocolError) as caught:
            encode_record(payload(state="delivered"))
        self.assertEqual(caught.exception.path, "$.record.state")

    def test_body_digest_mismatch_is_rejected(self):
        value = payload()
        value["body_digest"] = "0" * 64
        with self.assertRaises(ProtocolError) as caught:
            encode_record(value)
        self.assertEqual(caught.exception.path, "$.record.body_digest")

    def test_length_field_self_consistency(self):
        encoded = bytearray(encode_record(payload()))
        magic, length = HEADER.unpack(encoded[: HEADER.size])
        encoded[4:8] = (length + 1).to_bytes(4, "big")
        with self.assertRaises(ProtocolError) as caught:
            decode_record(bytes(encoded))
        self.assertEqual(caught.exception.path, "$.record.length")

    def test_checksum_mismatch_is_rejected(self):
        encoded = bytearray(encode_record(payload()))
        encoded[-1] ^= 0xFF
        with self.assertRaises(ProtocolError) as caught:
            decode_record(bytes(encoded))
        self.assertEqual(caught.exception.path, "$.record.digest")

    def test_sequence_must_strictly_increase_within_transaction(self):
        first = payload(event_seq=1)
        second = payload(event_seq=1, type="half_ack", data={"ack_sent": True})
        with self.assertRaises(ProtocolError) as caught:
            validate_sequence([first, second])
        self.assertEqual(caught.exception.path, "$.record.event_seq")


if __name__ == "__main__":
    unittest.main()

