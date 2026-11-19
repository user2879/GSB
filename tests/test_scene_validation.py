import json
import tempfile
import unittest
from pathlib import Path

from fanout_ledger.config import load_scene
from fanout_ledger.errors import SceneError

VALID = {
    "scene_version": "t-v1",
    "nodes": [
        {"node_id": "a", "host": "127.0.0.1", "port": 48720, "process_ms": 10,
         "failure_rate": 0, "slow_rate": 0, "slow_extra_ms": 0, "seed": 1},
        {"node_id": "b", "host": "127.0.0.1", "port": 48721, "process_ms": 10,
         "failure_rate": 0, "slow_rate": 0, "slow_extra_ms": 0, "seed": 2},
    ],
    "task": {"task_id": "t", "payload_bytes": 1024, "max_attempts": 3,
             "attempt_timeout_ms": 100, "backoff_ms": [10, 20, 30]},
    "scheduling": {"global_concurrency": 2, "node_concurrency": 1,
                   "session_budget_ms": 1000},
    "slo": {"success_rate_percent": 90, "p95_latency_ms": 500},
}


class SceneValidationTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_scene(path)

    def test_valid_scene_loads(self):
        scene = self._load(VALID)
        self.assertEqual(scene.scene_version, "t-v1")
        self.assertEqual(len(scene.nodes), 2)
        self.assertEqual(scene.task.backoff_ms, (10, 20, 30))

    def test_duplicate_node_id_rejected_with_field_path(self):
        payload = json.loads(json.dumps(VALID))
        payload["nodes"][1]["node_id"] = "a"
        with self.assertRaises(SceneError) as ctx:
            self._load(payload)
        self.assertEqual(ctx.exception.code, "E_SCENE")
        fields = [e["field"] for e in ctx.exception.details["errors"]]
        self.assertTrue(any("node_id" in f for f in fields))

    def test_duplicate_port_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["nodes"][1]["port"] = payload["nodes"][0]["port"]
        with self.assertRaises(SceneError) as ctx:
            self._load(payload)
        self.assertEqual(ctx.exception.code, "E_SCENE")
        self.assertTrue(any("port" in e["field"] for e in ctx.exception.details["errors"]))

    def test_non_loopback_host_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["nodes"][0]["host"] = "10.0.0.1"
        with self.assertRaises(SceneError) as ctx:
            self._load(payload)
        self.assertTrue(any(".host" in e["field"] for e in ctx.exception.details["errors"]))

    def test_port_range_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["nodes"][0]["port"] = 70000
        with self.assertRaises(SceneError):
            self._load(payload)

    def test_payload_must_be_positive_int(self):
        payload = json.loads(json.dumps(VALID))
        payload["task"]["payload_bytes"] = 0
        with self.assertRaises(SceneError) as ctx:
            self._load(payload)
        self.assertTrue(any("payload_bytes" in e["field"]
                            for e in ctx.exception.details["errors"]))

    def test_backoff_must_be_monotone_non_negative(self):
        payload = json.loads(json.dumps(VALID))
        payload["task"]["backoff_ms"] = [30, 10]
        with self.assertRaises(SceneError) as ctx:
            self._load(payload)
        self.assertTrue(any("backoff_ms" in e["field"]
                            for e in ctx.exception.details["errors"]))

    def test_slo_ranges_enforced(self):
        payload = json.loads(json.dumps(VALID))
        payload["slo"]["success_rate_percent"] = 101
        with self.assertRaises(SceneError):
            self._load(payload)
        payload = json.loads(json.dumps(VALID))
        payload["slo"]["p95_latency_ms"] = 0
        with self.assertRaises(SceneError):
            self._load(payload)

    def test_corrupt_json_reports_e_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(SceneError) as ctx:
                load_scene(path)
            self.assertEqual(ctx.exception.code, "E_SCENE")

    def test_multiple_errors_collected_together(self):
        payload = json.loads(json.dumps(VALID))
        payload["task"]["max_attempts"] = 0
        payload["slo"]["success_rate_percent"] = -1
        with self.assertRaises(SceneError) as ctx:
            self._load(payload)
        self.assertGreaterEqual(len(ctx.exception.details["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
