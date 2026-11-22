from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from slowdiag.config import default_config_dict, load_config_data
from slowdiag.errors import E_SOURCE, SlowDiagError


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.raw = default_config_dict()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_source_error(self, raw: dict, path: str) -> None:
        with self.assertRaises(SlowDiagError) as ctx:
            load_config_data(raw, self.workspace)
        self.assertEqual(ctx.exception.code, E_SOURCE)
        self.assertEqual(ctx.exception.path, path)

    def test_default_config_loads_and_normalizes_types(self) -> None:
        config = load_config_data(self.raw, self.workspace)
        self.assertEqual(len(config.sources), 4)
        self.assertEqual(config.dataset.query_count, 8)
        self.assertEqual(config.migration.reducer, "median")
        self.assertEqual(len(config.migration.pairs), 12)
        self.assertEqual(config.sources[0].resolved_data_path.name, "records.sqlite3")

    def test_first_invalid_source_field_is_reported(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][0]["sample_rows"] = 0
        raw["sources"][1]["type"] = "bad"
        self.assert_source_error(raw, "sources[0].sample_rows")

    def test_boolean_is_not_an_integer(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][0]["sample_period_ms"] = True
        self.assert_source_error(raw, "sources[0].sample_period_ms")

    def test_duplicate_source_id(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][1]["id"] = raw["sources"][0]["id"]
        self.assert_source_error(raw, "sources[1].id")

    def test_unsupported_source_type(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][0]["type"] = "raw_socket"
        self.assert_source_error(raw, "sources[0].type")

    def test_illegal_semantics_combination(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][1]["scan_semantics"] = "sampled"
        self.assert_source_error(raw, "sources[1].scan_semantics")

    def test_colliding_data_paths(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][1]["data_path"] = raw["sources"][0]["data_path"]
        self.assert_source_error(raw, "sources[1].data_path")

    def test_unknown_declared_format_version(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["sources"][0]["format_version"] = 99
        self.assert_source_error(raw, "sources[0].format_version")

    def test_migration_replay_count_must_be_positive_integer(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["migration"]["replay_count"] = 0
        self.assert_source_error(raw, "migration.replay_count")

    def test_invalid_pair_source_path(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["migration"]["pairs"] = [{"from": "app-prod", "to": "missing"}]
        self.assert_source_error(raw, "migration.pairs[0].to")


if __name__ == "__main__":
    unittest.main()
