import unittest

from dr_bench.canonical import canonical_bytes
from dr_bench.engine import (
    PATH_DIFF,
    PATH_FULL,
    PATH_INCREMENTAL,
    run_diff_synthesis,
    run_full,
    run_incremental,
)
from dr_bench.loader import parse_scene
from dr_bench.models import CODE_BUSY, CODE_COOLDOWN, CODE_MANUAL, CODE_SWITCHED
from dr_bench.samplegen import build_demo_scene, build_scale_scene
from dr_bench.session import assert_paths_identical


class PathEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.scene = parse_scene(build_demo_scene())

    def test_three_paths_byte_identical(self):
        full = run_full(self.scene)
        incremental = run_incremental(self.scene)
        diff, _ = run_diff_synthesis(self.scene)
        for index in range(len(full)):
            self.assertEqual(canonical_bytes(full[index]), canonical_bytes(incremental[index]))
            self.assertEqual(canonical_bytes(full[index]), canonical_bytes(diff[index]))

    def test_scale_scene_byte_identical(self):
        scene = parse_scene(build_scale_scene())
        report = assert_paths_identical(scene)
        self.assertEqual(report["rounds"], len(scene.rounds))
        self.assertIn(report["planner"]["selected_path"], (PATH_FULL, PATH_INCREMENTAL, PATH_DIFF))

    def test_no_observation_round_flagged(self):
        conclusions = run_full(self.scene)
        round3 = next(item for item in conclusions if item["round_index"] == 3)
        self.assertTrue(round3["instance_states"]["inv-2"]["no_observation"])
        self.assertFalse(round3["instance_states"]["inv-2"]["observed_in_round"])

    def test_unknown_never_available(self):
        conclusions = run_full(self.scene)
        for conclusion in conclusions:
            for instance_id, record in conclusion["instance_states"].items():
                if record["state"] == "unknown":
                    self.assertNotIn(instance_id, conclusion["available_instances"][record["service_id"]])


class SwitchSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.scene = parse_scene(build_demo_scene())
        self.conclusions = run_full(self.scene)

    def test_manual_strategy_refuses(self):
        for conclusion in self.conclusions:
            decision = conclusion["service_decisions"]["payment-service"]
            if decision["trigger_reasons"]:
                self.assertEqual(decision["code"], CODE_MANUAL)
                self.assertIsNone(decision["switch"])

    def test_switch_transaction_updates_roles_and_versions_atomically(self):
        switched = next(c for c in self.conclusions if c["business_codes"]["order-service"] == CODE_SWITCHED)
        record = switched["service_decisions"]["order-service"]["switch"]
        self.assertEqual(record["old_primary"], "ord-1")
        self.assertEqual(record["new_primary"], "ord-3")
        self.assertEqual(record["new_roles"]["ord-1"], "replica")
        self.assertEqual(record["new_roles"]["ord-3"], "primary")
        self.assertEqual(record["promoted_version"], max(record["old_versions"].values()) + 1)
        self.assertEqual(switched["runtime_versions"]["ord-3"], record["promoted_version"])

    def test_cooldown_suppresses_further_switch(self):
        codes = [c["business_codes"]["order-service"] for c in self.conclusions]
        self.assertIn(CODE_SWITCHED, codes)
        # A second unhealthy window inside 1000ms cannot trigger another switch.
        self.assertNotIn(CODE_COOLDOWN, codes[: codes.index(CODE_SWITCHED) + 1])

    def test_competing_requests_arbitrated_to_single_transaction(self):
        scene = parse_scene(build_demo_scene())
        requests = [
            {"service_id": "order-service", "round_index": 2, "request_id": "req-z"},
            {"service_id": "order-service", "round_index": 2, "request_id": "req-a"},
        ]
        conclusions = run_incremental(scene, extra_requests=requests)
        decision = conclusions[2]["service_decisions"]["order-service"]
        self.assertEqual(len([c for c in decision["busy"] if c["code"] == CODE_BUSY]), 2)
        busy_ids = [item["request_id"] for item in decision["busy"]]
        self.assertEqual(busy_ids, ["req-a", "req-z"])
        self.assertEqual(decision["switch"]["request_id"], "auto:order-service")


if __name__ == "__main__":
    unittest.main()


class CooldownTests(unittest.TestCase):
    def _cooldown_scene(self):
        return {
            "scene_id": "cooldown-check",
            "scene_version": 1,
            "services": [
                {
                    "service_id": "svc",
                    "strategy": "auto",
                    "cooldown_ms": 1000,
                    "min_healthy_replicas": 1,
                    "success_threshold": 1,
                    "failure_threshold": 1,
                }
            ],
            "instances": [
                {"service_id": "svc", "instance_id": "a", "role": "primary", "version": 1},
                {"service_id": "svc", "instance_id": "b", "role": "replica", "version": 1},
                {"service_id": "svc", "instance_id": "c", "role": "replica", "version": 1},
            ],
            "observations": [
                {"service_id": "svc", "instance_id": "a", "round_index": 0, "observed_at_ms": 0, "result": "success"},
                {"service_id": "svc", "instance_id": "b", "round_index": 0, "observed_at_ms": 0, "result": "success"},
                {"service_id": "svc", "instance_id": "c", "round_index": 0, "observed_at_ms": 0, "result": "success"},
                {"service_id": "svc", "instance_id": "a", "round_index": 1, "observed_at_ms": 100, "result": "timeout"},
                {"service_id": "svc", "instance_id": "b", "round_index": 1, "observed_at_ms": 100, "result": "success"},
                {"service_id": "svc", "instance_id": "c", "round_index": 1, "observed_at_ms": 100, "result": "success"},
                {"service_id": "svc", "instance_id": "b", "round_index": 2, "observed_at_ms": 500, "result": "timeout"},
                {"service_id": "svc", "instance_id": "c", "round_index": 2, "observed_at_ms": 500, "result": "success"},
                {"service_id": "svc", "instance_id": "b", "round_index": 3, "observed_at_ms": 1200, "result": "timeout"},
                {"service_id": "svc", "instance_id": "c", "round_index": 3, "observed_at_ms": 1200, "result": "success"},
            ],
        }

    def test_cooldown_suppresses_then_allows(self):
        from dr_bench.models import CODE_COOLDOWN
        scene = parse_scene(self._cooldown_scene())
        conclusions = run_full(scene)
        codes = [c["business_codes"]["svc"] for c in conclusions]
        self.assertEqual(codes, ["OK", "SWITCHED", CODE_COOLDOWN, "SWITCHED"])
        self.assertEqual(conclusions[2]["service_decisions"]["svc"]["cooldown_remaining_ms"], 600)
        self.assertEqual(conclusions[3]["switch_targets"]["svc"], "c")

    def test_cooldown_identical_across_paths(self):
        from dr_bench.canonical import canonical_bytes as cb
        scene = parse_scene(self._cooldown_scene())
        full = run_full(scene)
        incremental = run_incremental(scene)
        diff, _ = run_diff_synthesis(scene)
        for index in range(len(full)):
            self.assertEqual(cb(full[index]), cb(incremental[index]))
            self.assertEqual(cb(full[index]), cb(diff[index]))
