import json
import unittest
from pathlib import Path

from rlcb.codes import E_SPEC, SpecError
from rlcb.spec import load_bundle

SERVICES = Path("samples/acceptance/services.json")
REQUESTS = Path("samples/acceptance/requests.json")


class SpecValidationTest(unittest.TestCase):
    def test_loads_valid_bundle(self):
        bundle = load_bundle(SERVICES, REQUESTS)
        self.assertEqual([s.service_id for s in bundle.services],
                         ["svcA", "svcB", "svcC"])
        self.assertEqual(len(bundle.requests), 37)

    def _write(self, payload):
        path = Path("/tmp/rlcb_bad.json")
        path.write_text(json.dumps(payload, indent=2), "utf-8")
        return path

    def test_threshold_out_of_range_reports_line(self):
        data = json.loads(SERVICES.read_text())
        data["services"][1]["failure_threshold_bp"] = 12000
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self._write(data), REQUESTS)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertIn("failure_threshold_bp", ctx.exception.field)
        self.assertIsInstance(ctx.exception.line, int)

    def test_non_loopback_host_rejected(self):
        data = json.loads(SERVICES.read_text())
        data["services"][0]["host"] = "0.0.0.0"
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self._write(data), REQUESTS)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertIn("host", ctx.exception.field)

    def test_duplicate_port_rejected(self):
        data = json.loads(SERVICES.read_text())
        data["services"][2]["port"] = data["services"][0]["port"]
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self._write(data), REQUESTS)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertIn("port", ctx.exception.field)

    def test_duplicate_service_id_rejected(self):
        data = json.loads(SERVICES.read_text())
        data["services"][1]["service_id"] = data["services"][0]["service_id"]
        with self.assertRaises(SpecError):
            load_bundle(self._write(data), REQUESTS)

    def test_negative_rps_rejected(self):
        data = json.loads(SERVICES.read_text())
        data["services"][0]["rps_limit"] = -1
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self._write(data), REQUESTS)
        self.assertIn("rps_limit", ctx.exception.field)

    def test_unsupported_degrade_level(self):
        data = json.loads(SERVICES.read_text())
        data["services"][0]["degrade_level"] = 9
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self._write(data), REQUESTS)
        self.assertIn("degrade_level", ctx.exception.field)

    def test_arrival_seq_decrease_rejected_with_line(self):
        data = json.loads(REQUESTS.read_text())
        data["requests"][6]["arrival_seq"] = 0
        data["requests"][7]["arrival_seq"] = 0
        path = Path("/tmp/rlcb_bad_req.json")
        path.write_text(json.dumps(data, indent=2), "utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_bundle(SERVICES, path)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertIn("arrival_seq", ctx.exception.field)

    def test_unknown_service_in_request(self):
        data = json.loads(REQUESTS.read_text())
        data["requests"][0]["service_id"] = "ghost"
        path = Path("/tmp/rlcb_bad_req2.json")
        path.write_text(json.dumps(data), "utf-8")
        with self.assertRaises(SpecError):
            load_bundle(SERVICES, path)


if __name__ == "__main__":
    unittest.main()
