from __future__ import annotations

import copy
import json
import unittest

from quota_workbench.engine import RuntimeState, process_event
from quota_workbench.spec import load_declaration, load_events, parse_declaration, parse_events
from tests.helpers import SAMPLES, sample_events, sample_spec


class EngineTests(unittest.TestCase):
    def test_normal_scenario_preempts_expected_replicas(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        events = load_events(SAMPLES / "normal.ndjson", declaration)
        state = RuntimeState.initial(declaration)
        preemptions = []
        for event in events:
            result = process_event(declaration, state, event)
            self.assertEqual(result.terminal_code, "OK")
            preemptions.extend(
                (item["tenant_id"], item["workload_id"], item["reclaimed_units"])
                for item in result.record["preemptions"]
            )
            state = result.state
        self.assertEqual(
            preemptions,
            [
                ("tenant-alpha", "wl-alpha-batch", 7),
                ("tenant-beta", "wl-beta-batch-b", 6),
                ("tenant-gamma", "wl-gamma-batch", 5),
                ("tenant-alpha", "wl-alpha-batch", 7),
            ],
        )
        self.assertEqual(
            state.to_dict(declaration)["running_replicas"]["wl-alpha-batch"], 0
        )
        self.assertEqual(
            state.to_dict(declaration)["running_replicas"]["wl-gamma-batch"], 2
        )

    def test_workload_declaration_order_does_not_change_result(self) -> None:
        data = sample_spec()
        declaration_a = parse_declaration(data)
        reversed_data = copy.deepcopy(data)
        reversed_data["workloads"].reverse()
        reversed_data["tenants"].reverse()
        declaration_b = parse_declaration(reversed_data)
        events_a = load_events(SAMPLES / "normal.ndjson", declaration_a)
        events_b = parse_events(sample_events(), declaration_b)
        state_a = RuntimeState.initial(declaration_a)
        state_b = RuntimeState.initial(declaration_b)
        for event_a, event_b in zip(events_a, events_b):
            result_a = process_event(declaration_a, state_a, event_a)
            result_b = process_event(declaration_b, state_b, event_b)
            self.assertEqual(result_a.record_json, result_b.record_json)
            self.assertEqual(result_a.record_digest, result_b.record_digest)
            state_a = result_a.state
            state_b = result_b.state

    def test_highest_startup_order_wins_within_tenant(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        event = load_events(SAMPLES / "normal.ndjson", declaration)[1]
        result = process_event(declaration, RuntimeState.initial(declaration), event)
        self.assertEqual(result.record["preemptions"][0]["workload_id"], "wl-beta-batch-b")
        basis = result.record["preemptions"][0]["selection_basis"]
        self.assertEqual(basis["tenant_priority"], 20)
        self.assertEqual(basis["startup_order"], 50)
        self.assertTrue(basis["preemptible"])

    def test_tenant_priority_orders_multiple_overage_tenants(self) -> None:
        data = sample_spec()
        for workload in data["workloads"]:
            if workload["workload_id"] == "wl-alpha-batch":
                workload["requested_replicas"] = 10
            if workload["workload_id"] == "wl-gamma-batch":
                workload["requested_replicas"] = 5
        declaration = parse_declaration(data)
        zero_event = parse_events(
            [
                '{"event_id":1,"time_ms":0,"type":"TENANT_PRESSURE",'
                '"target_tenant":"tenant-alpha","pressure_units":0}'
            ],
            declaration,
        )[0]
        result = process_event(declaration, RuntimeState.initial(declaration), zero_event)
        first = result.record["preemptions"][0]
        self.assertEqual(first["tenant_id"], "tenant-alpha")
        self.assertEqual(first["workload_id"], "wl-alpha-batch")
        self.assertGreater(len(result.record["preemptions"]), 1)
        self.assertEqual(result.terminal_code, "OK")

    def test_reserve_floor_blocks_preemption(self) -> None:
        data = {
            "cluster_total_units": 10,
            "tenants": [
                {"tenant_id": "alpha", "quota_bp": 10000, "reserve_units": 9, "priority": 1}
            ],
            "workloads": [
                {
                    "workload_id": "wl",
                    "tenant_id": "alpha",
                    "requested_replicas": 2,
                    "units_per_replica": 5,
                    "preemptible": True,
                    "startup_order": 1,
                }
            ],
        }
        declaration = parse_declaration(data)
        event = parse_events(
            [
                '{"event_id":1,"time_ms":0,"type":"TENANT_PRESSURE",'
                '"target_tenant":"alpha","pressure_units":3}'
            ],
            declaration,
        )[0]
        result = process_event(declaration, RuntimeState.initial(declaration), event)
        self.assertEqual(result.terminal_code, "E_UNSATISFIED")
        self.assertEqual(result.record["gap"]["remaining_gap_units"], 3)
        self.assertEqual(
            result.record["gap"]["blocked_workloads"][0]["reason"], "RESERVE_FLOOR"
        )

    def test_unsatisfied_gap_lists_tenant_and_non_preemptible_blocker(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        event = load_events(SAMPLES / "unsat.ndjson", declaration)[0]
        result = process_event(declaration, RuntimeState.initial(declaration), event)
        self.assertTrue(result.unsatisfied)
        self.assertEqual(result.record["terminal"]["reason"], "NO_PREEMPTIBLE_REPLICA")
        self.assertEqual(result.record["gap"]["remaining_gap_units"], 12)
        self.assertEqual(result.record["gap"]["affected_tenant_ids"], ["tenant-gamma"])
        self.assertEqual(result.record["terminal"]["preemption_count"], 3)
        blocked = result.record["gap"]["rejected_preemption_requests"]
        self.assertEqual(blocked[0]["workload_id"], "wl-gamma-core")
        self.assertEqual(blocked[0]["reason"], "NON_PREEMPTIBLE")

    def test_each_preemption_links_before_and_after_matrices(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        event = load_events(SAMPLES / "normal.ndjson", declaration)[0]
        result = process_event(declaration, RuntimeState.initial(declaration), event)
        digest_to_iteration = {
            item["matrix_digest"]: item["iteration"] for item in result.record["iterations"]
        }
        preemption = result.record["preemptions"][0]
        self.assertIn(preemption["matrix_before_digest"], digest_to_iteration)
        self.assertIn(preemption["matrix_after_digest"], digest_to_iteration)
        self.assertEqual(
            digest_to_iteration[preemption["matrix_after_digest"]] - digest_to_iteration[preemption["matrix_before_digest"]],
            1,
        )
        self.assertEqual(result.record["iterations"][0]["terminal_reason"], None)
        self.assertEqual(result.record["iterations"][-1]["terminal_reason"], "NO_EXCESS")

    def test_pressure_is_attributed_through_workload_target(self) -> None:
        declaration = load_declaration(SAMPLES / "spec.json")
        event = load_events(SAMPLES / "normal.ndjson", declaration)[1]
        result = process_event(declaration, RuntimeState.initial(declaration), event)
        pressure = result.record["workload_state"]["pressure_by_tenant"]
        self.assertEqual(pressure["tenant-beta"], 15)
        self.assertEqual(pressure["tenant-alpha"], 0)


if __name__ == "__main__":
    unittest.main()
