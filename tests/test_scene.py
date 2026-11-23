import json
import os
import tempfile
import unittest

from convd.errors import SceneError
from convd.scene import (ExecutorSet, Scenario, load_executor_set,
                         load_scenario)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_scenario():
    return {
        "scenario_id": "sc",
        "max_rounds": 3,
        "window_ms": 300,
        "pulse_interval_ms": 50,
        "lease_ms": 250,
        "receipt_timeout_ms": 100,
        "budget_version": 1,
        "max_in_flight": 2,
        "max_in_flight_per_executor": 1,
        "max_retries": 1,
        "executor_set_version": 1,
    }


def base_executors():
    return {"executors": [
        {"executor_id": "a", "port": 40001, "target_digest": "t1",
         "seed": 1},
        {"executor_id": "b", "port": 40002, "target_digest": "t2",
         "seed": 2},
    ]}


class TestSceneValidation(unittest.TestCase):
    def test_valid_scenario_loads(self):
        sc = Scenario(base_scenario())
        self.assertEqual(sc.scenario_id, "sc")
        self.assertTrue(sc.digest)

    def test_window_smaller_than_lease_rejected(self):
        data = base_scenario()
        data["window_ms"] = 100
        with self.assertRaises(SceneError) as ctx:
            Scenario(data)
        self.assertEqual(ctx.exception.code, "E_SCENE")
        self.assertEqual(ctx.exception.field, "window_ms")

    def test_non_positive_interval_rejected(self):
        data = base_scenario()
        data["pulse_interval_ms"] = 0
        with self.assertRaises(SceneError) as ctx:
            Scenario(data)
        self.assertEqual(ctx.exception.field, "pulse_interval_ms")

    def test_first_invalid_field_is_stable(self):
        data = base_scenario()
        data["max_rounds"] = 0
        data["lease_ms"] = -5
        with self.assertRaises(SceneError) as ctx:
            Scenario(data)
        self.assertEqual(ctx.exception.field, "max_rounds")

    def test_executor_port_range(self):
        data = base_executors()
        data["executors"][0]["port"] = 70000
        with self.assertRaises(SceneError) as ctx:
            ExecutorSet(data)
        self.assertEqual(ctx.exception.field, "executors[0].port")

    def test_duplicate_executor_id_rejected(self):
        data = base_executors()
        data["executors"][1]["executor_id"] = "a"
        with self.assertRaises(SceneError) as ctx:
            ExecutorSet(data)
        self.assertEqual(ctx.exception.field, "executors[1].executor_id")

    def test_duplicate_port_rejected(self):
        data = base_executors()
        data["executors"][1]["port"] = 40001
        with self.assertRaises(SceneError) as ctx:
            ExecutorSet(data)
        self.assertEqual(ctx.exception.field, "executors[1].port")

    def test_canonical_order_independent_of_declaration(self):
        data = base_executors()
        reversed_set = ExecutorSet({"executors": data["executors"][::-1]})
        normal_set = ExecutorSet(data)
        self.assertEqual(reversed_set.digest, normal_set.digest)

    def test_load_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            sc_path = os.path.join(tmp, "s.json")
            ex_path = os.path.join(tmp, "e.json")
            with open(sc_path, "w") as fh:
                json.dump(base_scenario(), fh)
            with open(ex_path, "w") as fh:
                json.dump(base_executors(), fh)
            self.assertEqual(load_scenario(sc_path).scenario_id, "sc")
            self.assertEqual(len(load_executor_set(ex_path).specs), 2)


if __name__ == "__main__":
    unittest.main()
