import json
import os
import tempfile
import unittest

from shardbench.config import load_config
from shardbench.errors import ConfigError, CycleError, RingError


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def sample_paths():
    return (
        os.path.join(SAMPLE_DIR, "config.json"),
        os.path.join(SAMPLE_DIR, "units.json"),
    )


class ValidationTests(unittest.TestCase):
    def test_valid_sample_loads(self):
        config_path, units_path = sample_paths()
        config = load_config(config_path, units_path)
        self.assertEqual(len(config.tables), 4)
        self.assertEqual(len(config.units), 11)
        self.assertIn("u_user_eq", config.topo_order)

    def test_duplicate_table_rejected(self):
        config_path, units_path = sample_paths()
        payload = _read_json(config_path)
        payload["tables"].append(dict(payload["tables"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            _write_json(path, payload)
            with self.assertRaises(ConfigError) as ctx:
                load_config(path, units_path)
        self.assertEqual(ctx.exception.code, "E_CONFIG")
        self.assertTrue(any("name" in part for part in ctx.exception.paths))

    def test_shard_key_missing_column_rejected(self):
        config_path, units_path = sample_paths()
        payload = _read_json(config_path)
        payload["tables"][0]["shard_key_columns"] = ["not_a_column"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            _write_json(path, payload)
            with self.assertRaises(ConfigError) as ctx:
                load_config(path, units_path)
        self.assertEqual(ctx.exception.code, "E_CONFIG")
        self.assertIn("shard_key_columns", ctx.exception.paths[0])

    def test_non_positive_counts_rejected(self):
        config_path, units_path = sample_paths()
        payload = _read_json(config_path)
        payload["tables"][0]["tables_per_db"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            _write_json(path, payload)
            with self.assertRaises(ConfigError) as ctx:
                load_config(path, units_path)
        self.assertIn("tables_per_db", ctx.exception.paths[0])

    def test_modulo_base_must_be_positive(self):
        bad = os.path.join(SAMPLE_DIR, "variants", "config_corrupt.json")
        with self.assertRaises(ConfigError) as ctx:
            load_config(bad, os.path.join(SAMPLE_DIR, "units.json"))
        self.assertEqual(ctx.exception.code, "E_CONFIG")
        self.assertIn("base", ctx.exception.paths[0])

    def test_range_boundaries_strictly_increasing(self):
        config_path, units_path = sample_paths()
        payload = _read_json(config_path)
        range_table = next(t for t in payload["tables"] if t["algorithm"] == "range")
        range_table["params"]["boundaries"] = [100, 100, 300]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            _write_json(path, payload)
            with self.assertRaises(ConfigError) as ctx:
                load_config(path, units_path)
        self.assertIn("boundaries", ctx.exception.paths[0])

    def test_ring_weight_must_be_positive_int(self):
        config_path, units_path = sample_paths()
        payload = _read_json(config_path)
        ring_table = next(
            t for t in payload["tables"] if t["algorithm"] == "consistent_hash"
        )
        ring_table["params"]["nodes"][0]["weight"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            _write_json(path, payload)
            with self.assertRaises(ConfigError) as ctx:
                load_config(path, units_path)
        self.assertIn("weight", ctx.exception.paths[0])

    def test_ring_same_weight_conflict_is_e_ring(self):
        bad = os.path.join(SAMPLE_DIR, "variants", "config_ring_conflict.json")
        with self.assertRaises(RingError) as ctx:
            load_config(bad, os.path.join(SAMPLE_DIR, "units.json"))
        self.assertEqual(ctx.exception.code, "E_RING")

    def test_missing_dependency_reference_rejected(self):
        config_path, units_path = sample_paths()
        payload = _read_json(units_path)
        payload["routing_units"][0]["depends_on"] = ["ghost_unit"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "units.json")
            _write_json(path, payload)
            with self.assertRaises(ConfigError) as ctx:
                load_config(config_path, path)
        self.assertEqual(ctx.exception.code, "E_CONFIG")

    def test_dependency_cycle_is_e_cycle(self):
        config_path, units_path = sample_paths()
        payload = _read_json(units_path)
        for unit in payload["routing_units"]:
            if unit["id"] == "u_user_eq":
                unit["depends_on"] = ["u_user_in"]
            if unit["id"] == "u_user_in":
                unit["depends_on"] = ["u_user_eq"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "units.json")
            _write_json(path, payload)
            with self.assertRaises(CycleError) as ctx:
                load_config(config_path, path)
        self.assertEqual(ctx.exception.code, "E_CYCLE")
        self.assertIn("u_user_eq", ctx.exception.cycle)
        self.assertIn("u_user_in", ctx.exception.cycle)


if __name__ == "__main__":
    unittest.main()
