from __future__ import annotations

import unittest
from dataclasses import replace

from bucketpack.engine import run_full
from bucketpack.errors import E_WIDTH, BusinessError

from .common import sample_spec


class PackingTests(unittest.TestCase):
    def test_pack_ranges_are_contiguous_and_budgeted(self) -> None:
        spec = sample_spec()
        state = run_full(spec)
        grouped = {}
        for pack in state.packs:
            grouped.setdefault(pack.source_id, []).append(pack)
        for source_id, packs in grouped.items():
            ordered = sorted(packs, key=lambda item: item.index)
            cursor = 0
            for pack in ordered:
                self.assertEqual(pack.row_start, cursor)
                cursor = pack.row_end
                self.assertLessEqual(pack.row_count, spec.packing.rows_per_pack)
                self.assertLessEqual(pack.width_used, spec.packing.pack_width_budget)
                self.assertEqual(
                    pack.width_used,
                    pack.row_count * state.expansions[source_id].row_width,
                )
            self.assertEqual(cursor, state.expansions[source_id].rows)

    def test_single_row_over_budget_is_width_error(self) -> None:
        spec = sample_spec()
        too_small = spec.replace(
            packing=replace(spec.packing, pack_width_budget=1)
        )
        with self.assertRaises(BusinessError) as caught:
            run_full(too_small)
        error = caught.exception
        self.assertEqual(error.code, E_WIDTH)
        self.assertEqual(error.path, "$.packing.pack_width_budget")
        self.assertIn("primary_key", error.details)
        self.assertIn("row_width", error.details)

    def test_global_budget_below_pack_width_is_width_error(self) -> None:
        spec = sample_spec()
        too_small = spec.replace(
            packing=replace(spec.packing, global_width_budget=50)
        )
        with self.assertRaises(BusinessError) as caught:
            run_full(too_small)
        self.assertEqual(caught.exception.code, E_WIDTH)
        self.assertEqual(caught.exception.path, "$.packing.global_width_budget")


if __name__ == "__main__":
    unittest.main()
