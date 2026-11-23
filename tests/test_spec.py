import json
import unittest
from pathlib import Path

from arcore.errors import SpecError
from arcore.spec import load_declaration

SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "declaration.json"


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(SAMPLE.read_text(encoding="utf-8"))

    def _load(self, payload):
        path = Path(self._testMethodName + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            return load_declaration(path)
        finally:
            path.unlink()

    def test_valid_declaration_loads(self):
        declaration = load_declaration(SAMPLE)
        self.assertEqual(len(declaration.sources), 3)
        self.assertIn("svc-db", declaration.topology.nodes)

    def test_duplicate_source_id_rejected(self):
        self.base["sources"][1]["source_id"] = self.base["sources"][0]["source_id"]
        with self.assertRaises(SpecError) as ctx:
            self._load(self.base)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "sources[1].source_id")

    def test_negative_seed_rejected(self):
        self.base["sources"][0]["seed"] = -1
        with self.assertRaises(SpecError) as ctx:
            self._load(self.base)
        self.assertEqual(ctx.exception.source_id, "src-gateway")
        self.assertEqual(ctx.exception.field, "sources[0].seed")

    def test_boolean_is_not_an_integer(self):
        self.base["parameters"]["correlation_window_ms"] = True
        with self.assertRaises(SpecError):
            self._load(self.base)

    def test_topology_dangling_reference_rejected(self):
        self.base["topology"]["edges"].append(["svc-ghost", "svc-db"])
        with self.assertRaises(SpecError):
            self._load(self.base)

    def test_topology_cycle_rejected(self):
        self.base["topology"]["edges"].append(["svc-db", "svc-gateway"])
        with self.assertRaises(SpecError) as ctx:
            self._load(self.base)
        self.assertEqual(ctx.exception.field, "topology.edges")

    def test_non_positive_window_rejected(self):
        self.base["parameters"]["silence_window_ms"] = 0
        with self.assertRaises(SpecError) as ctx:
            self._load(self.base)
        self.assertEqual(ctx.exception.field,
                         "parameters.silence_window_ms")

    def test_layer_weights_must_be_integers(self):
        self.base["parameters"]["layer_weights"] = [1, 0.5]
        with self.assertRaises(SpecError):
            self._load(self.base)

    def test_service_used_but_not_in_topology_rejected(self):
        self.base["sources"][0]["services"] = ["svc-missing", "svc-auth"]
        with self.assertRaises(SpecError) as ctx:
            self._load(self.base)
        self.assertEqual(ctx.exception.field, "topology.edges")

    def test_scale_must_be_power_of_ten(self):
        self.base["parameters"]["fixed_point_scale"] = 500
        with self.assertRaises(SpecError):
            self._load(self.base)


if __name__ == "__main__":
    unittest.main()
