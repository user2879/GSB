import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_pipeline.config import parse_merge_config
from feature_pipeline.errors import E_CONFIG_INVALID, ConfigInvalidError
from feature_pipeline.models import V1, V2, V3, Cell, EntityTable, MergeConfig
from feature_pipeline.preprocess import (
    apply_scaling,
    fill_missing,
    percentile_rank,
    scaling_statistics,
)


BASE_CONFIG = {
    "feature_whitelist": ["f_a"],
    "scaling_method": "zscore",
    "missing_fill": {"strategy": "mean"},
    "conflict_priority": "source_version",
    "numeric_tolerance": 0.01,
    "max_workers": 2,
    "batch_size": 4,
}


class ConfigTests(unittest.TestCase):
    def test_missing_field_uses_business_code(self) -> None:
        broken = dict(BASE_CONFIG)
        broken.pop("numeric_tolerance")
        with self.assertRaises(ConfigInvalidError) as ctx:
            parse_merge_config(broken)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_invalid_enums_and_ranges(self) -> None:
        for key, value in (
            ("scaling_method", "bogus"),
            ("conflict_priority", "bogus"),
            ("numeric_tolerance", -1.0),
            ("max_workers", 0),
            ("batch_size", 0),
        ):
            broken = dict(BASE_CONFIG)
            broken[key] = value
            with self.assertRaises(ConfigInvalidError) as ctx:
                parse_merge_config(broken)
            self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)


class PreprocessTests(unittest.TestCase):
    def _tables(self):
        tables = []
        for version, values, missing_entity in (
            (V1, [1.0, 2.0, 3.0], "e3"),
            (V2, [1.5, 2.5, 3.5], "e2"),
        ):
            table = EntityTable(source_version=version, feature_order=["f_a"])
            for index, value in enumerate(values, start=1):
                entity = f"e{index}"
                table.cells.append(
                    Cell(entity, "f_a", version, value, False, "float64")
                )
            table.cells.append(
                Cell(missing_entity, "f_a", version, float("nan"), True, None)
            )
            tables.append(table)
        return tables

    def test_fill_uses_nonmissing_only(self) -> None:
        config = parse_merge_config(BASE_CONFIG)
        tables = self._tables()
        audit = fill_missing(tables, config)
        # Mean of the six present values.
        self.assertAlmostEqual(audit["f_a"]["fill_value"],
                               float(np.mean([1, 2, 3, 1.5, 2.5, 3.5])))
        self.assertEqual(audit["f_a"]["nonmissing_sample_size"], 6)
        self.assertEqual(audit["f_a"]["filled_cell_count"], 2)

    def test_zscore_zero_std_maps_to_zero(self) -> None:
        config = parse_merge_config(BASE_CONFIG)
        table = EntityTable(source_version=V1, feature_order=["f_a"])
        for entity in ("e1", "e2", "e3"):
            table.cells.append(Cell(entity, "f_a", V1, 4.0, False, "per_mille"))
        stats = scaling_statistics([table], config)
        self.assertEqual(stats["f_a"]["decision"], "zero_std_maps_to_zero")
        apply_scaling([table], stats, config)
        self.assertTrue(all(cell.value == 0.0 for cell in table.cells))

    def test_percentile_tie_rules_are_deterministic(self) -> None:
        samples = np.array([1.0, 1.0, 2.0, 3.0])
        avg = percentile_rank(1.0, samples, "average_rank")
        minimum = percentile_rank(1.0, samples, "min_rank")
        maximum = percentile_rank(1.0, samples, "max_rank")
        self.assertLess(minimum, avg)
        self.assertLess(avg, maximum)
        self.assertEqual(avg, 100.0 * 1.5 / 4)

    def test_percentile_boundary_equal_gets_50(self) -> None:
        config = parse_merge_config({**BASE_CONFIG, "scaling_method": "percentile"})
        table = EntityTable(source_version=V3, feature_order=["f_a"])
        for entity in ("e1", "e2"):
            table.cells.append(Cell(entity, "f_a", V3, 7.0, False, "float32"))
        stats = scaling_statistics([table], config)
        self.assertEqual(
            stats["f_a"]["decision"],
            "boundary_equal_all_values_get_50th_percentile",
        )
        apply_scaling([table], stats, config)
        self.assertTrue(all(cell.value == 50.0 for cell in table.cells))


if __name__ == "__main__":
    unittest.main()
