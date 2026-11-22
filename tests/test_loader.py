import json
import os
import tempfile
import unittest

from workbench.errors import SchemaError
from workbench.loader import load_declaration
from workbench.sample_data import build_cluster, build_corrupt_cluster


class LoaderTest(unittest.TestCase):
    def _write(self, payload) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_valid_deterministic_load(self):
        path = self._write(build_cluster())
        spec = load_declaration(path)
        self.assertEqual(len(spec.services), 4)
        self.assertEqual(spec.minutes, 24)
        self.assertEqual([svc.service_id for svc in spec.services],
                         [item['service_id'] for item in build_cluster()['services']])
        self.assertEqual(spec.digest(), load_declaration(path).digest())

    def test_duplicate_service_id_rejected(self):
        payload = build_cluster()
        payload["services"][1]["service_id"] = payload["services"][0]["service_id"]
        with self.assertRaises(SchemaError) as caught:
            load_declaration(self._write(payload))
        self.assertEqual(caught.exception.code, "E_SCHEMA")
        self.assertEqual(caught.exception.details["field"], "service_id")
        self.assertEqual(caught.exception.details["service_index"], 1)

    def test_negative_budget_rejected_first(self):
        path = self._write(build_corrupt_cluster())
        with self.assertRaises(SchemaError) as caught:
            load_declaration(path)
        self.assertEqual(caught.exception.details["field"], "global_replica_budget")

    def test_min_greater_than_max_rejected(self):
        payload = build_cluster()
        payload["services"][2]["min_replicas"] = 99
        with self.assertRaises(SchemaError) as caught:
            load_declaration(self._write(payload))
        self.assertEqual(caught.exception.details["field"], "min_replicas")
        self.assertEqual(caught.exception.details["service_id"],
                         payload["services"][2]["service_id"])

    def test_negative_latency_and_bool_integer_rejected(self):
        payload = build_cluster()
        payload["services"][0]["latency_ms"] = -5
        with self.assertRaises(SchemaError) as caught:
            load_declaration(self._write(payload))
        self.assertEqual(caught.exception.details["field"], "latency_ms")

        payload = build_cluster()
        payload["global_replica_budget"] = True
        with self.assertRaises(SchemaError):
            load_declaration(self._write(payload))

    def test_series_length_mismatch_rejected_as_schema_at_load(self):
        payload = build_cluster()
        payload["services"][1]["requests"].append(7)
        with self.assertRaises(SchemaError) as caught:
            load_declaration(self._write(payload))
        self.assertEqual(caught.exception.code, "E_SCHEMA")
        self.assertEqual(caught.exception.details["field"], "requests")


if __name__ == "__main__":
    unittest.main()
