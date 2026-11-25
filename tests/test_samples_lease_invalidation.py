import unittest

from shardflow.errors import LeaseError
from shardflow.invalidation import propagate_invalidation
from shardflow.lease import Lease, check_renew, grant_lease
from shardflow.samples import (
    fixed_point_aggregate,
    generate_samples,
    sample_digest,
)
from shardflow.spec import load_spec
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "demo", "demo.spec.json")
EVENTS_PATH = os.path.join(ROOT, "demo", "demo.events.ndjson")


class SampleTests(unittest.TestCase):
    def test_samples_are_deterministic(self):
        first_groups, first_values = generate_samples(42, 64)
        second_groups, second_values = generate_samples(42, 64)
        self.assertTrue((first_groups == second_groups).all())
        self.assertTrue((first_values == second_values).all())
        self.assertEqual(sample_digest(first_groups, first_values),
                         sample_digest(second_groups, second_values))

    def test_fixed_point_aggregate_is_deterministic(self):
        groups, values = generate_samples(7, 50)
        config = {"alpha": 3, "beta": 5}
        first = fixed_point_aggregate(groups, values, 4, config, 10)
        second = fixed_point_aggregate(groups, values, 4, config, 10)
        self.assertEqual(first, second)
        self.assertTrue(all(row["sum"] >= 0 for row in first["rows"]))


class LeaseTests(unittest.TestCase):
    def test_grant_and_renew_optimistic_version(self):
        lease = grant_lease("s1", "node-a", 1, 1000, 500)
        self.assertEqual(lease.lease_version, 1)
        check_renew(lease, 1, 2, 1300, 500)
        self.assertEqual(lease.lease_version, 2)
        self.assertEqual(lease.expires_ms, 1800)
        self.assertEqual(lease.renewals, 1)

    def test_version_mismatch_rejects_without_extending(self):
        lease = grant_lease("s1", "node-a", 1, 1000, 500)
        before = lease.expires_ms
        with self.assertRaises(LeaseError) as ctx:
            check_renew(lease, 99, 2, 1300, 500)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        self.assertEqual(lease.expires_ms, before)
        self.assertEqual(lease.lease_version, 1)

    def test_renewal_limit_enforced(self):
        lease = grant_lease("s1", "node-a", 1, 1000, 500)
        check_renew(lease, 1, 1, 1300, 500)
        with self.assertRaises(LeaseError) as ctx:
            check_renew(lease, 2, 1, 1600, 500)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        self.assertEqual(lease.renewals, 1)

    def test_real_monotonic_expiry(self):
        lease = grant_lease("s1", "node-a", 1, 0, 1)
        self.assertTrue(lease.is_expired(1))
        self.assertFalse(lease.is_expired(0))


class InvalidationTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(SPEC_PATH, EVENTS_PATH)

    def test_revision_two_invalidates_alpha_consumers_and_dependents(self):
        ordered, reasons = propagate_invalidation(self.spec, 1, 2)
        self.assertEqual(ordered, ["s1", "s3", "s5", "s6", "s7", "s8"])
        self.assertEqual(
            reasons["s6"]["reasons"][-1]["reason"],
            "dependency_invalidation",
        )

    def test_revision_three_only_invalidates_gamma_subgraph(self):
        ordered, _ = propagate_invalidation(self.spec, 2, 3)
        self.assertEqual(ordered, ["s3", "s6", "s8"])

    def test_propagation_order_is_topological(self):
        ordered, _ = propagate_invalidation(self.spec, 1, 2)
        positions = {sid: ordered.index(sid) for sid in ordered}
        self.assertLess(positions["s3"], positions["s6"])
        self.assertLess(positions["s6"], positions["s8"])


if __name__ == "__main__":
    unittest.main()
