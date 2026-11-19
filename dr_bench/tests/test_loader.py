import json
import unittest
from pathlib import Path

from dr_bench.errors import SceneError
from dr_bench.loader import parse_scene
from dr_bench.samplegen import build_demo_scene


class LoaderValidationTests(unittest.TestCase):
    def test_valid_scene_loads(self):
        scene = parse_scene(build_demo_scene())
        self.assertEqual(scene.scene_id, "dr-demo-scene")
        self.assertIn("order-service", scene.services)
        self.assertEqual(len(scene.rounds), 8)

    def test_duplicate_service_rejected_with_path(self):
        payload = build_demo_scene()
        payload["services"].append(dict(payload["services"][0]))
        with self.assertRaises(SceneError) as ctx:
            parse_scene(payload)
        self.assertEqual(ctx.exception.code, "E_SCENE")
        self.assertTrue(any("duplicate service id" in e["message"] for e in ctx.exception.detail["errors"]))
        self.assertTrue(any(e["path"].startswith("services[") for e in ctx.exception.detail["errors"]))

    def test_two_primaries_rejected(self):
        payload = build_demo_scene()
        for instance in payload["instances"]:
            if instance["instance_id"] == "ord-2":
                instance["role"] = "primary"
        with self.assertRaises(SceneError) as ctx:
            parse_scene(payload)
        self.assertTrue(any("at most one primary" in e["message"] for e in ctx.exception.detail["errors"]))

    def test_duplicate_round_rejected(self):
        payload = build_demo_scene()
        payload["observations"].append(dict(payload["observations"][0]))
        with self.assertRaises(SceneError) as ctx:
            parse_scene(payload)
        paths = [e["path"] for e in ctx.exception.detail["errors"]]
        self.assertTrue(any("duplicate round" in e["message"] for e in ctx.exception.detail["errors"]))
        self.assertTrue(any(path.startswith("observations[") for path in paths))

    def test_non_monotonic_timestamp_rejected(self):
        payload = build_demo_scene()
        target = next(
            item
            for item in payload["observations"]
            if item["instance_id"] == "ord-1" and item["round_index"] == 1
        )
        target["observed_at_ms"] = 50
        with self.assertRaises(SceneError) as ctx:
            parse_scene(payload)
        self.assertTrue(any("non-decreasing" in e["message"] for e in ctx.exception.detail["errors"]))

    def test_unsupported_result_rejected(self):
        payload = build_demo_scene()
        payload["observations"][0]["result"] = "bogus"
        with self.assertRaises(SceneError) as ctx:
            parse_scene(payload)
        self.assertTrue(any(e["path"].endswith(".result") for e in ctx.exception.detail["errors"]))

    def test_negative_and_bool_rejected(self):
        payload = build_demo_scene()
        payload["services"][0]["cooldown_ms"] = -1
        payload["services"][0]["failure_threshold"] = True
        with self.assertRaises(SceneError) as ctx:
            parse_scene(payload)
        messages = ctx.exception.detail["errors"]
        self.assertTrue(any(e["path"].endswith("cooldown_ms") for e in messages))
        self.assertTrue(any(e["path"].endswith("failure_threshold") for e in messages))

    def test_unknown_service_reference_rejected(self):
        payload = build_demo_scene()
        payload["observations"][0]["service_id"] = "missing"
        with self.assertRaises(SceneError):
            parse_scene(payload)


if __name__ == "__main__":
    unittest.main()
