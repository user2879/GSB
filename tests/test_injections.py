from __future__ import annotations

import unittest

from renderdiag.resources import load_injections
from tests.support import SampleWorkspace


class InjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = SampleWorkspace()
        cls.injections = {
            item.injection_id: item
            for item in load_injections(cls.sample.paths["injections"])
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sample.close()

    def test_fusion_dimension_mismatch_is_isolated_to_one_frame(self) -> None:
        injection = self.injections["INJ-DIM"]
        engine = self.sample.engine("inject-dim", injections=[injection])
        response = engine.drain_current()
        self.assertEqual(response["code"], "E_PARTIAL_FAILURE")
        self.assertIn("INJ-DIM", [item["id"] for item in response["injection_hits"]])
        self.assertEqual(response["current_state"]["completed_frames"], [0, 2])
        self.assertTrue(response["current_state"]["block_rows_completed"]["0"])
        self.assertTrue(response["current_state"]["block_rows_completed"]["2"])
        self.assertFalse(response["current_state"]["block_rows_completed"]["1"])

    def test_nonfinite_latent_is_isolated_to_one_frame(self) -> None:
        injection = self.injections["INJ-NONFINITE"]
        engine = self.sample.engine("inject-nonfinite", injections=[injection])
        response = engine.drain_current()
        self.assertEqual(response["code"], "E_PARTIAL_FAILURE")
        self.assertIn("INJ-NONFINITE", [item["id"] for item in response["injection_hits"]])
        self.assertEqual(response["current_state"]["completed_frames"], [0, 2])
        self.assertIn("E_NONFINITE_LATENT", [item["code"] for item in response["failures"]])

    def test_block_row_out_of_range_preserves_other_rows_and_frames(self) -> None:
        injection = self.injections["INJ-ROW-OOB"]
        engine = self.sample.engine("inject-row", injections=[injection])
        response = engine.drain_current()
        self.assertEqual(response["code"], "E_PARTIAL_FAILURE")
        self.assertIn("INJ-ROW-OOB", [item["id"] for item in response["injection_hits"]])
        self.assertEqual(response["current_state"]["completed_frames"], [1, 2])
        self.assertEqual(response["current_state"]["block_rows_completed"]["0"], [0, 1, 3])
        self.assertEqual(response["current_state"]["block_rows_completed"]["1"], [0, 1, 2, 3])
        self.assertIn("E_BLOCK_ROW_OUT_OF_RANGE", [item["code"] for item in response["failures"]])

    def test_quantization_out_of_range_is_reported_with_injection_id(self) -> None:
        injection = self.injections["INJ-QUANT-OOB"]
        engine = self.sample.engine("inject-quant", injections=[injection])
        response = engine.drain_current()
        self.assertEqual(response["code"], "E_PARTIAL_FAILURE")
        self.assertIn("INJ-QUANT-OOB", [item["id"] for item in response["injection_hits"]])
        failures = [item for item in response["failures"] if item.get("injection_id") == "INJ-QUANT-OOB"]
        self.assertTrue(failures)
        self.assertEqual(failures[0]["code"], "E_CONFIG_INVALID")
        self.assertEqual(response["current_state"]["completed_frames"], [])

    def test_stale_snapshot_version_is_a_real_restore_failure(self) -> None:
        injection = self.injections["INJ-STALE-SNAP"]
        warm = self.sample.engine("inject-snapshot")
        self.assertEqual(warm.drain_current()["code"], "OK")
        self.assertEqual(warm.snapshot_store.current_version(), 5)
        restarted = self.sample.engine("inject-snapshot", injections=[injection])
        hits = restarted.injection_report()
        self.assertEqual([item["id"] for item in hits], ["INJ-STALE-SNAP"])
        self.assertEqual(hits[0]["code"], "E_STALE_SNAPSHOT_VERSION")
        self.assertIn(
            "E_STALE_SNAPSHOT_VERSION",
            [item["code"] for item in restarted.failures],
        )


if __name__ == "__main__":
    unittest.main()
