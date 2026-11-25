import unittest

from rlcb.invalidate import apply_changes, param_digest, propagate
from rlcb.planner import choose_plan
from rlcb.spec import load_bundle
from pathlib import Path

SERVICES = Path("samples/acceptance/services.json")
REQUESTS = Path("samples/acceptance/requests.json")


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(SERVICES, REQUESTS)

    def test_version_bumps_monotonically_and_digest_changes(self):
        spec = next(s for s in self.bundle.services
                    if s.service_id == "svcA")
        old_version = spec.config_version
        old_digest = param_digest(spec)
        new_services, changes = apply_changes(
            self.bundle.services, {"svcA": {"rps_limit": 2}})
        updated = next(s for s in new_services if s.service_id == "svcA")
        self.assertEqual(updated.config_version, old_version + 1)
        self.assertNotEqual(param_digest(updated), old_digest)
        self.assertEqual(changes[0].from_version, old_version)
        self.assertEqual(changes[0].to_version, old_version + 1)

    def test_propagation_only_touches_affected_service(self):
        new_services, changes = apply_changes(
            self.bundle.services, {"svcB": {"failure_threshold_bp": 2000}})
        trace = propagate(new_services, changes, self.bundle.requests,
                          lease_records=[])
        affected_services = {rid.split("-")[0]
                             for rid in trace.affected_requests}
        self.assertEqual(affected_services, {"svcB"})
        self.assertTrue(all(w.startswith("svcB:")
                            for w in trace.invalid_windows))
        self.assertTrue(all(b.startswith("svcB:")
                            for b in trace.invalid_batches))
        # svcA and svcC requests must remain untouched.
        self.assertNotIn("svcA-R000", trace.affected_requests)

    def test_window_length_change_invalidates_windows(self):
        new_services, changes = apply_changes(
            self.bundle.services, {"svcA": {"window_length": 2}})
        trace = propagate(new_services, changes, self.bundle.requests, [])
        svc_a_count = sum(1 for r in self.bundle.requests
                          if r.service_id == "svcA")
        self.assertEqual(len(trace.affected_requests), svc_a_count)
        self.assertTrue(trace.invalid_windows)


class PlannerTest(unittest.TestCase):
    def test_full_when_no_changes(self):
        plan = choose_plan(service_count=3, batch_count=4, window_count=5,
                           changed_services=[], changed_batches=[],
                           all_batches=["a", "b"])
        self.assertEqual(plan.mode, "full")

    def test_incremental_when_scale_and_fraction_ok(self):
        plan = choose_plan(
            service_count=6, batch_count=10, window_count=12,
            changed_services=["s1"],
            changed_batches=["s1:B1"],
            all_batches=[f"s{i}:B" for i in range(6)])
        self.assertEqual(plan.mode, "incremental")
        self.assertIn("1/2", plan.reason)
        self.assertEqual(plan.boundaries["incremental_batch_limit"], 6)

    def test_full_when_changed_fraction_too_large(self):
        # 3 of 4 work units changed (> 1/2) even though scale is large.
        plan = choose_plan(
            service_count=2, batch_count=2, window_count=20,
            changed_services=["s1", "s2"],
            changed_batches=["s1:B0", "s1:B1", "s2:B0"],
            all_batches=["s1:B0", "s1:B1", "s2:B0", "s2:B1"])
        self.assertEqual(plan.mode, "full")
        self.assertIn("1/2", plan.reason)

    def test_force_mode_overrides_rules(self):
        plan = choose_plan(service_count=10, batch_count=10, window_count=10,
                           changed_services=["s"], changed_batches=["s:B"],
                           all_batches=["s:B"], force_mode="full")
        self.assertEqual(plan.mode, "full")


if __name__ == "__main__":
    unittest.main()
