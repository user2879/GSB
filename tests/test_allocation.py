from __future__ import annotations

import random
import unittest

from quota_workbench.allocation import derive_entitlements
from quota_workbench.spec import Declaration, Tenant, parse_declaration
from tests.helpers import sample_spec


def declaration(cluster: int, tenants: list[tuple[str, int, int, int]]) -> Declaration:
    return Declaration(
        cluster,
        tuple(Tenant(tenant_id, quota, reserve, priority) for tenant_id, quota, reserve, priority in tenants),
        (),
    )


class AllocationTests(unittest.TestCase):
    def test_sample_entitlements(self) -> None:
        matrix = derive_entitlements(parse_declaration(sample_spec()))
        rows = {item["tenant_id"]: item for item in matrix["tenants"]}
        self.assertEqual(rows["tenant-alpha"]["share_entitlement_units"], 60)
        self.assertEqual(rows["tenant-beta"]["share_entitlement_units"], 36)
        self.assertEqual(rows["tenant-gamma"]["share_entitlement_units"], 24)
        self.assertEqual(matrix["entitlement_total_units"], 120)
        self.assertEqual(matrix["unassigned_units"], 0)

    def test_share_sum_below_scale_leaves_unassigned_capacity(self) -> None:
        matrix = derive_entitlements(
            declaration(100, [("alpha", 5000, 0, 1), ("beta", 3000, 0, 2)])
        )
        self.assertEqual(matrix["shared_budget_units"], 80)
        self.assertEqual(matrix["entitlement_total_units"], 80)
        self.assertEqual(matrix["unassigned_units"], 20)

    def test_largest_remainder_uses_explicit_tie_order(self) -> None:
        matrix = derive_entitlements(
            declaration(
                10,
                [
                    ("tenant-b", 3333, 0, 2),
                    ("tenant-a", 3333, 0, 1),
                    ("tenant-c", 3334, 0, 3),
                ],
            )
        )
        self.assertEqual(matrix["entitlement_total_units"], 10)
        rows = {item["tenant_id"]: item for item in matrix["tenants"]}
        self.assertEqual(rows["tenant-c"]["share_rounding_bonus_units"], 1)
        self.assertAlmostEqual(
            rows["tenant-a"]["share_quantization_error_scaled_1e4"],
            rows["tenant-a"]["share_entitlement_units"] * 10000 - 10 * 3333,
        )

    def test_reserve_conflict_reduces_only_above_floor(self) -> None:
        matrix = derive_entitlements(
            declaration(100, [("alpha", 5000, 80, 1), ("beta", 5000, 0, 2)])
        )
        rows = {item["tenant_id"]: item for item in matrix["tenants"]}
        self.assertEqual(rows["alpha"]["share_entitlement_units"], 50)
        self.assertEqual(rows["alpha"]["reserve_uplift_units"], 30)
        self.assertEqual(rows["alpha"]["entitlement_units"], 80)
        self.assertEqual(rows["beta"]["capacity_reduction_units"], 30)
        self.assertEqual(rows["beta"]["entitlement_units"], 20)
        self.assertEqual(matrix["entitlement_total_units"], 100)

    def test_random_invariants(self) -> None:
        random.seed(20261123)
        for _ in range(2000):
            cluster = random.randint(1, 500)
            count = random.randint(1, 7)
            shares = [random.randint(1, 5000) for _ in range(count)]
            if sum(shares) > 10000:
                continue
            reserves = [random.randint(0, cluster) for _ in range(count)]
            if sum(reserves) > cluster:
                continue
            decl = declaration(
                cluster,
                [(f"tenant-{index}", shares[index], reserves[index], index) for index in range(count)],
            )
            matrix = derive_entitlements(decl)
            self.assertLessEqual(matrix["entitlement_total_units"], cluster)
            self.assertGreaterEqual(matrix["unassigned_units"], 0)
            for row in matrix["tenants"]:
                self.assertGreaterEqual(row["entitlement_units"], row["reserve_units"])
                self.assertEqual(
                    row["excess_units"],
                    max(0, row["actual_units"] - row["entitlement_units"]),
                )


if __name__ == "__main__":
    unittest.main()
