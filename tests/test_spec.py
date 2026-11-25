import unittest

from batch_receipt_ledger.errors import SpecError
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class SpecValidationTest(unittest.TestCase):
    def test_valid_spec_loads(self):
        spec = parse_spec(make_spec())
        self.assertEqual(len(spec.workers), 2)
        self.assertEqual(spec.effective_quota("worker-a"), 2)

    def test_duplicate_worker_id(self):
        raw = make_spec()
        raw["workers"][1]["worker_id"] = raw["workers"][0]["worker_id"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "worker_id")
        self.assertIn("/workers/1", ctx.exception.location)

    def test_host_must_be_loopback_literal(self):
        raw = make_spec()
        raw["workers"][0]["host"] = "0.0.0.0"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "host")

    def test_duplicate_port(self):
        raw = make_spec()
        raw["workers"][1]["port"] = raw["workers"][0]["port"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "port")

    def test_dimensions_positive_int(self):
        raw = make_spec()
        raw["model"]["hidden_dim"] = 0
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "hidden_dim")

    def test_lease_and_concurrency_positive(self):
        raw = make_spec()
        raw["limits"]["lease_ms"] = 0
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "lease_ms")

        raw2 = make_spec()
        raw2["workers"][0]["concurrency"] = -2
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw2)
        self.assertEqual(ctx.exception.field, "concurrency")

    def test_duplicate_request_id(self):
        raw = make_spec()
        raw["deliveries"][2]["request_id"] = raw["deliveries"][0]["request_id"]
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.field, "request_id")

    def test_supersede_merge_excludes_old(self):
        raw = make_spec(deliveries=4)
        raw["deliveries"].append({
            "request_id": "req-new", "arrival_seq": 5, "value": 42,
            "replaces": "req-001"})
        spec = parse_spec(raw)
        active = spec.effective_deliveries()
        self.assertNotIn("req-001", [d.request_id for d in active])
        self.assertIn("req-new", [d.request_id for d in active])
        self.assertEqual(spec.supersession_map()["req-new"], "req-001")

    def test_first_error_location_reported(self):
        raw = make_spec()
        raw["model"]["input_dim"] = -1
        raw["workers"][0]["port"] = 1
        with self.assertRaises(SpecError) as ctx:
            parse_spec(raw)
        self.assertEqual(ctx.exception.location, "/model")
        self.assertEqual(ctx.exception.field, "input_dim")


if __name__ == "__main__":
    unittest.main()
