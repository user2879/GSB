import json
import unittest
from pathlib import Path

from lease_middleware.codec import decode_record, decode_records, encode_record, upgrade_record
from lease_middleware.errors import E_LEASE, MiddlewareError
from lease_middleware.spec import load_spec

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class CodecTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(SAMPLES / "config.json")

    def test_same_lease_normalizes_identically_across_formats(self):
        raws = json.loads((SAMPLES / "leases_same_lease.json").read_text())
        records = decode_records(raws, self.spec)
        views = [{k: v for k, v in r.to_dict().items() if k != "format"} for r in records]
        self.assertEqual(views[0], views[1])
        self.assertEqual(views[1], views[2])
        self.assertEqual(views[0]["lease_version"], 2)

    def test_r3_digest_mismatch_rejected(self):
        raws = json.loads((SAMPLES / "leases_r3.json").read_text())
        raws[0]["value_digest"] = "deadbeef"
        with self.assertRaises(MiddlewareError) as ctx:
            decode_records(raws, self.spec)
        self.assertEqual(ctx.exception.code, E_LEASE)
        self.assertEqual(ctx.exception.field_path, "value_digest")

    def test_dangling_parent_rejected(self):
        raws = json.loads((SAMPLES / "leases_r3.json").read_text())
        raws[1]["parent_lease_id"] = "L:missing:x:0000"
        with self.assertRaises(MiddlewareError) as ctx:
            decode_records(raws, self.spec)
        self.assertEqual(ctx.exception.code, E_LEASE)
        self.assertEqual(ctx.exception.field_path, "parent_lease_id")

    def test_missing_required_field(self):
        raw = {"format": "r2", "lease_id": "L:c:k:0001", "config_key": "cache.policy",
               "expires_at_ms": 123}
        with self.assertRaises(MiddlewareError) as ctx:
            decode_record(raw, self.spec)
        self.assertEqual(ctx.exception.code, E_LEASE)

    def test_encode_roundtrip_and_upgrade_preserves_version(self):
        raws = json.loads((SAMPLES / "leases_r1.json").read_text())
        record = decode_records(raws, self.spec)[0]
        encoded = encode_record(record, "r1")
        self.assertEqual(set(encoded), {"format", "lease_id", "config_key", "expires_at_ms"})
        upgraded = upgrade_record(record)
        self.assertEqual(upgraded.format, "r3")
        self.assertEqual(upgraded.lease_version, record.lease_version)
        self.assertEqual(upgraded.renewal_count, 0)
        self.assertIsNone(upgraded.parent_lease_id)


if __name__ == "__main__":
    unittest.main()
