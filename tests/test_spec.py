"""E_SPEC / E_QUOTA validation tests."""

import os
import tempfile
import unittest

from helpers import free_ports, make_requests, make_spec_dict, write_json, write_ndjson

from quota_workbench.errors import QuotaError, SpecError
from quota_workbench.spec import load_requests, load_spec


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qwb-spec-")
        self.ports = free_ports(2)
        self.spec_path = os.path.join(self.tmp, "spec.json")
        write_json(self.spec_path, make_spec_dict(self.ports))

    def _load_raw(self, raw):
        return load_spec(self.spec_path, _raw_override=raw)

    def test_valid_spec_loads(self):
        spec = load_spec(self.spec_path)
        self.assertEqual(sorted(spec.tenants), ["alpha", "beta"])
        self.assertEqual(spec.tenant_instance["alpha"], "svc-a")
        self.assertEqual(spec.tenant_instance["beta"], "svc-b")
        self.assertEqual(len(spec.digest()), 64)

    def test_duplicate_tenant_id(self):
        raw = make_spec_dict(self.ports)
        raw["tenants"].append(dict(raw["tenants"][0]))
        with self.assertRaises(SpecError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.field, "tenants[2].tenant_id")

    def test_host_must_be_loopback_literal(self):
        raw = make_spec_dict(self.ports)
        raw["instances"][0]["host"] = "localhost"
        with self.assertRaises(SpecError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.field, "instances[0].host")

    def test_duplicate_port_rejected(self):
        raw = make_spec_dict(self.ports)
        raw["instances"][1]["port"] = self.ports[0]
        with self.assertRaises(SpecError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.field, "instances[1].port")

    def test_negative_share_rejected(self):
        raw = make_spec_dict(self.ports)
        raw["tenants"][0]["share"] = -1
        with self.assertRaises(SpecError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.field, "tenants[0].share")

    def test_bool_share_rejected(self):
        raw = make_spec_dict(self.ports)
        raw["tenants"][0]["share"] = True
        with self.assertRaises(SpecError):
            self._load_raw(raw)

    def test_share_sum_overflow_is_e_quota(self):
        raw = make_spec_dict(self.ports)
        raw["tenants"][0]["share"] = 95
        with self.assertRaises(QuotaError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.code, "E_QUOTA")
        self.assertEqual(ctx.exception.field, "tenants[].share")

    def test_tenant_must_be_hosted(self):
        raw = make_spec_dict(self.ports)
        raw["instances"][0]["hosts"] = []
        with self.assertRaises(SpecError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.field, "instances[].hosts")

    def test_unsupported_rule_version(self):
        raw = make_spec_dict(self.ports)
        raw["rule_version"] = 99
        with self.assertRaises(SpecError) as ctx:
            self._load_raw(raw)
        self.assertEqual(ctx.exception.field, "rule_version")


class RequestStreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qwb-req-")
        self.ports = free_ports(2)
        self.spec_path = os.path.join(self.tmp, "spec.json")
        write_json(self.spec_path, make_spec_dict(self.ports))
        self.spec = load_spec(self.spec_path)
        self.req_path = os.path.join(self.tmp, "requests.ndjson")

    def _load(self, rows):
        write_ndjson(self.req_path, rows)
        return load_requests(self.req_path, self.spec)

    def test_valid_stream(self):
        requests = self._load(make_requests(6))
        self.assertEqual(len(requests), 6)
        self.assertEqual(requests[0].request_id, "r000")

    def test_non_monotonic_arrival_reports_line(self):
        rows = make_requests(4)
        rows[3]["arrival_ms"] = 0
        with self.assertRaises(SpecError) as ctx:
            self._load(rows)
        self.assertEqual(ctx.exception.field, "arrival_ms")
        self.assertEqual(ctx.exception.line, 4)

    def test_unknown_tenant_reports_line(self):
        rows = make_requests(3)
        rows[1]["tenant_id"] = "ghost"
        with self.assertRaises(SpecError) as ctx:
            self._load(rows)
        self.assertEqual(ctx.exception.field, "tenant_id")
        self.assertEqual(ctx.exception.line, 2)

    def test_demand_must_be_positive(self):
        rows = make_requests(2)
        rows[0]["demand"] = 0
        with self.assertRaises(SpecError) as ctx:
            self._load(rows)
        self.assertEqual(ctx.exception.field, "demand")
        self.assertEqual(ctx.exception.line, 1)

    def test_duplicate_request_id(self):
        rows = make_requests(2)
        rows[1]["request_id"] = rows[0]["request_id"]
        with self.assertRaises(SpecError) as ctx:
            self._load(rows)
        self.assertEqual(ctx.exception.field, "request_id")
        self.assertEqual(ctx.exception.line, 2)


if __name__ == "__main__":
    unittest.main()
