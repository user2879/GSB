import json
import tempfile
import unittest
from pathlib import Path

from _helpers import base_manifest, mutate

from haptic_engine.errors import (
    E_CURVE,
    E_DEVICE,
    E_EVENT,
    E_FORMAT,
    E_GLOBAL,
    E_JSON,
)
from haptic_engine.parser import load_manifest


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _expect(self, payload, code):
        path = self.root / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            load_manifest(path)
        self.assertEqual(ctx.exception.code, code)
        # Validation must never create state/output artifacts.
        self.assertFalse((self.root / "state").exists())

    def test_valid_manifest_parses(self):
        path = self.root / "ok.json"
        path.write_text(json.dumps(base_manifest(str(self.root))), encoding="utf-8")
        manifest = load_manifest(path)
        self.assertEqual(len(manifest.devices), 2)

    def test_invalid_json(self):
        path = self.root / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            load_manifest(path)
        self.assertEqual(ctx.exception.code, E_JSON)

    def test_duplicate_device_id(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][1]["device_id"] = "d-a"
        self._expect(payload, E_DEVICE)

    def test_empty_device_id(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][0]["device_id"] = "   "
        self._expect(payload, E_DEVICE)

    def test_non_positive_sample_rate(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][0]["sample_rate_hz"] = 0
        self._expect(payload, E_DEVICE)

    def test_levels_below_two(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][0]["levels"] = 1
        self._expect(payload, E_DEVICE)

    def test_inverted_intensity_range(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][0]["min_intensity"] = 1.0
        payload["devices"][0]["max_intensity"] = 1.0
        self._expect(payload, E_DEVICE)

    def test_non_finite_intensity(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][0]["max_intensity"] = float("inf")
        self._expect(payload, E_DEVICE)

    def test_negative_start_ms(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["events"][0]["start_ms"] = -1
        self._expect(payload, E_EVENT)

    def test_zero_duration(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["events"][0]["duration_ms"] = 0
        self._expect(payload, E_EVENT)

    def test_unknown_target_device(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["events"][0]["target_device"] = "ghost"
        self._expect(payload, E_EVENT)

    def test_curve_not_strictly_increasing(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["events"][0]["curve"][1]["time_ms"] = 0
        self._expect(payload, E_CURVE)

    def test_curve_intensity_out_of_range(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["events"][0]["curve"][1]["intensity"] = 5.0
        self._expect(payload, E_CURVE)

    def test_curve_does_not_cover_event(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["events"][0]["curve"][-1]["time_ms"] = 90
        self._expect(payload, E_CURVE)

    def test_missing_state_dir(self):
        payload = mutate(base_manifest(str(self.root)))
        del payload["state_dir"]
        self._expect(payload, E_FORMAT)

    def test_inflight_below_frame_budget(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["global"]["inflight_limit"] = 1
        self._expect(payload, E_GLOBAL)

    def test_boolean_rejected_as_integer(self):
        payload = mutate(base_manifest(str(self.root)))
        payload["devices"][0]["sample_rate_hz"] = True
        self._expect(payload, E_DEVICE)


if __name__ == "__main__":
    unittest.main()
