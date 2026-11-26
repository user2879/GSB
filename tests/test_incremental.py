from __future__ import annotations

import unittest

from renderdiag.resources import load_command_stream
from tests.support import SampleWorkspace


def _comparable(results):
    return {
        "frames": results["frames"],
        "denoise_steps": results["denoise_steps"],
        "fusion": results["fusion"],
        "rows": results["rows"],
        "text_fingerprint": results["text_fingerprint"],
        "image_fingerprint": results["image_fingerprint"],
        "motion_fingerprints": results["motion_fingerprints"],
    }


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = SampleWorkspace()
        cls.commands = load_command_stream(cls.sample.paths["commands"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sample.close()

    def test_incremental_sequence_matches_blank_full_rebuild(self) -> None:
        incremental = self.sample.engine("incremental")
        self.assertEqual(incremental.drain_current()["code"], "OK")
        for command in self.commands:
            response = incremental.run_command(command)
            self.assertEqual(response["code"], "OK", response["failures"])

        full = self.sample.engine("full-final")
        for command in self.commands:
            full.submit(command)
        full_response = full.drain_current()
        self.assertEqual(full_response["code"], "OK")
        self.assertEqual(
            _comparable(incremental.results()),
            _comparable(full.results()),
        )

    def test_targeted_invalidation_reuses_unaffected_encodings_and_denoise(self) -> None:
        engine = self.sample.engine("reuse-targets")
        self.assertEqual(engine.drain_current()["code"], "OK")

        weighted = engine.run_command(
            {"op": "set_weights", "weights": {"text": 1.5, "image": 0.25, "motion": 0.5}}
        )
        self.assertEqual(weighted["code"], "OK")
        status = {item["stage"]: item["status"] for item in weighted["ledger"]}
        self.assertEqual(status["encode_text"], "reused")
        self.assertEqual(status["encode_image"], "reused")
        self.assertEqual(status["encode_motion"], "reused")
        self.assertEqual(status["fusion"], "recomputed")
        self.assertEqual(status["denoise"], "recomputed")

        quantized = engine.run_command({"op": "set_quant_level", "level": 4})
        self.assertEqual(quantized["code"], "OK")
        quant_status = {item["stage"]: item["status"] for item in quantized["ledger"]}
        self.assertEqual(quant_status["encode_text"], "reused")
        self.assertEqual(quant_status["encode_image"], "reused")
        self.assertEqual(quant_status["fusion"], "recomputed")

        reblocked = engine.run_command({"op": "set_block_rows", "rows": 2})
        self.assertEqual(reblocked["code"], "OK")
        block_status = {item["stage"]: item["status"] for item in reblocked["ledger"]}
        self.assertEqual(block_status["denoise"], "reused")
        self.assertEqual(block_status["decode_row"], "recomputed")

    def test_stale_and_latest_summaries_are_strictly_distinguished(self) -> None:
        engine = self.sample.engine("stale")
        self.assertEqual(engine.drain_current()["code"], "OK")
        previous_cursor = engine.cursor
        engine.submit({"op": "set_guidance", "guidance": 2.25})
        stale = engine.advance()
        self.assertEqual(stale["code"], "E_STALE_SUMMARY")
        self.assertTrue(stale["stale"])
        self.assertFalse(stale["is_latest"])
        self.assertEqual(stale["requested_cursor"], previous_cursor + 1)
        self.assertEqual(stale["based_on_cursor"], previous_cursor)
        self.assertNotEqual(stale["requested_config_fp"], stale["based_on_config_fp"])
        self.assertTrue(stale["pending"])
        self.assertTrue(all(item["code"] == "E_BUDGET_PENDING" for item in stale["pending"]))
        self.assertLessEqual(stale["summary"]["progress"], 1.0)

        latest = engine.drain_current()
        self.assertEqual(latest["code"], "OK")
        self.assertFalse(latest["stale"])
        self.assertTrue(latest["is_latest"])
        self.assertEqual(latest["based_on_cursor"], latest["requested_cursor"])
        self.assertEqual(latest["based_on_config_fp"], latest["requested_config_fp"])
        self.assertEqual(latest["summary"]["completed_frames"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
