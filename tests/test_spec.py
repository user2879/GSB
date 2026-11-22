from __future__ import annotations

import unittest

from quota_workbench.errors import WorkbenchError
from quota_workbench.spec import (
    SHARE_SCALE,
    load_declaration,
    load_events,
    parse_declaration,
    parse_events,
)
from tests.helpers import SAMPLES, sample_events, sample_spec


class SpecValidationTests(unittest.TestCase):
    def test_sample_loads_and_versions_are_order_independent(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        reversed_data = sample_spec()
        reversed_data["tenants"].reverse()
        reversed_data["workloads"].reverse()
        reversed_declaration = parse_declaration(reversed_data)
        self.assertEqual(declaration.spec_version(), reversed_declaration.spec_version())
        self.assertEqual(declaration.versions(), reversed_declaration.versions())
        self.assertEqual(sum(item.quota_bp for item in declaration.tenants), SHARE_SCALE)

    def test_quota_overflow_has_specialized_code(self) -> None:
        data = sample_spec()
        data["tenants"][0]["quota_bp"] += 1
        with self.assertRaises(WorkbenchError) as caught:
            parse_declaration(data)
        self.assertEqual(caught.exception.code, "E_QUOTA")
        self.assertEqual(caught.exception.field, "quota_bp_sum")

    def test_reserve_overflow_has_specialized_code(self) -> None:
        data = sample_spec()
        data["tenants"][0]["reserve_units"] = data["cluster_total_units"] + 1
        with self.assertRaises(WorkbenchError) as caught:
            parse_declaration(data)
        self.assertEqual(caught.exception.code, "E_RESERVE")
        self.assertEqual(caught.exception.field, "reserve_units_sum")

    def test_first_invalid_tenant_field_is_reported(self) -> None:
        data = sample_spec()
        data["tenants"][0]["quota_bp"] = True
        data["tenants"][1]["reserve_units"] = -1
        with self.assertRaises(WorkbenchError) as caught:
            parse_declaration(data)
        error = caught.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.object_id, "tenant-alpha")
        self.assertEqual(error.field, "quota_bp")

    def test_duplicate_ids_and_startup_orders_are_rejected(self) -> None:
        duplicate_tenant = sample_spec()
        duplicate_tenant["tenants"][1]["tenant_id"] = duplicate_tenant["tenants"][0]["tenant_id"]
        with self.assertRaises(WorkbenchError) as caught:
            parse_declaration(duplicate_tenant)
        self.assertEqual(caught.exception.field, "tenant_id")

        duplicate_start = sample_spec()
        duplicate_start["workloads"][1]["startup_order"] = duplicate_start["workloads"][0]["startup_order"]
        with self.assertRaises(WorkbenchError) as caught:
            parse_declaration(duplicate_start)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "startup_order")

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        data = sample_spec()
        data["workloads"][0]["requested_replicas"] = True
        with self.assertRaises(WorkbenchError) as caught:
            parse_declaration(data)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "requested_replicas")

    def test_event_sequence_time_and_target_validation(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        broken_sequence = list(sample_events())
        broken_sequence[1] = broken_sequence[1].replace('"event_id":2', '"event_id":3')
        with self.assertRaises(WorkbenchError) as caught:
            parse_events(broken_sequence, declaration)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "event_id")

        broken_time = list(sample_events())
        broken_time[1] = broken_time[1].replace('"time_ms":2000', '"time_ms":999')
        with self.assertRaises(WorkbenchError) as caught:
            parse_events(broken_time, declaration)
        self.assertEqual(caught.exception.field, "time_ms")

        broken_target = list(sample_events())
        broken_target[0] = broken_target[0].replace("tenant-alpha", "tenant-missing")
        with self.assertRaises(WorkbenchError) as caught:
            parse_events(broken_target, declaration)
        self.assertEqual(caught.exception.field, "target_tenant")

    def test_event_type_and_pressure_validation(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        broken_type = list(sample_events())
        broken_type[0] = broken_type[0].replace("TENANT_PRESSURE", "UNKNOWN")
        with self.assertRaises(WorkbenchError) as caught:
            parse_events(broken_type, declaration)
        self.assertEqual(caught.exception.field, "type")

        broken_pressure = list(sample_events())
        broken_pressure[0] = broken_pressure[0].replace('"pressure_units":35', '"pressure_units":-1')
        with self.assertRaises(WorkbenchError) as caught:
            parse_events(broken_pressure, declaration)
        self.assertEqual(caught.exception.field, "pressure_units")

    def test_sample_event_loader(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        events = load_events(SAMPLES / "normal.ndjson", declaration)
        self.assertEqual([item.event_id for item in events], [1, 2, 3, 4])
        self.assertEqual(events[1].target_workload, "wl-beta-batch-b")


if __name__ == "__main__":
    unittest.main()
