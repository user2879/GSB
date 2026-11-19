from __future__ import annotations

import copy
import unittest

from tests.helpers import SAMPLES
from voicecal.attribution import attribute_turn
from voicecal.calibration import calibrate_turn
from voicecal.config import load_config
from voicecal.util import decimal_to_fixed


def turn(turn_id: str, index: int, speaker: str, candidates: list[str], position: int, gap_ms: float = 0.0) -> dict:
    return {
        "turn_id": turn_id,
        "turn_index": index,
        "conversation_id": "conv",
        "conversation_position": position,
        "audio": f"audio/{turn_id}.wav",
        "speaker_id": speaker,
        "speaker_candidates": sorted(candidates),
        "gap_ms": gap_ms,
        "previous_turn_id": None,
        "link_type": "continuation",
    }


class CalibrationAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(SAMPLES / "calibration.json")

    def initial(self, values: dict[str, int], key: str = "initial") -> dict:
        return {"cache_key": key, "status": "ok", "intensities_fixed": values}

    def test_switch_penalty_only_applies_on_speaker_change(self) -> None:
        emotions = self.config["evidence"]["emotions"]
        values = {emotion: 250000 for emotion in emotions}
        first_turn = turn("t1", 1, "a", ["a", "b"], 1)
        second_turn = turn("t2", 2, "b", ["a", "b"], 2)
        _, first = calibrate_turn(self.initial(values), first_turn, None, None, {e: 0 for e in emotions}, 0.4, self.config)
        _, second = calibrate_turn(
            self.initial(values, "initial-2"),
            second_turn,
            first_turn,
            first,
            first["chain_sum_fixed"],
            0.4,
            self.config,
        )
        self.assertFalse(first["coefficients"]["speaker_changed"])
        self.assertFalse(first["coefficients"]["switch_penalty_applied"])
        self.assertTrue(second["coefficients"]["speaker_changed"])
        self.assertTrue(second["coefficients"]["switch_penalty_applied"])
        dp = self.config["fixed_point"]["decimal_places"]
        self.assertEqual(second["coefficients"]["transition_factor_fixed"], decimal_to_fixed(0.73, dp))
        for emotion in emotions:
            terms = second["terms_fixed"][emotion]
            self.assertGreater(terms["switch_penalty_applied"], 0)
            self.assertEqual(terms["final_quantized"], second["calibrated_evidence_fixed"][emotion])

    def test_attribution_tie_breaks_by_lexicographic_speaker_id(self) -> None:
        config = copy.deepcopy(self.config)
        config["attribution"]["score_weights"] = {"emotion_affinity": 0.5, "pitch_match": 0.5, "script_prior": 0.0}
        profile = {
            "pitch_center_hz": 150.0,
            "pitch_tolerance_hz": 100.0,
            "emotion_affinity": {emotion: 0.5 for emotion in config["evidence"]["emotions"]},
        }
        config["attribution"]["speaker_profiles"] = {"a": copy.deepcopy(profile), "z": copy.deepcopy(profile)}
        current = turn("tie", 1, "z", ["z", "a"], 1)
        calibration = {
            "cache_key": "cal",
            "status": "ok",
            "calibrated_evidence_fixed": {emotion: 250000 for emotion in config["evidence"]["emotions"]},
        }
        extract = {
            "cache_key": "extract",
            "status": "ok",
            "aggregate_features": {"duration_ms": 1000.0, "f0_mean_hz": 150.0},
        }
        _, record = attribute_turn(current, calibration, extract, None, None, config, [current])
        self.assertEqual(record["speaker_id"], "a")
        self.assertEqual(record["score_tie_break"], "speaker_id_lexicographic")

    def test_min_segment_constraint_blocks_attribution(self) -> None:
        config = copy.deepcopy(self.config)
        config["attribution"]["min_segment_ms"] = 99999
        current = turn("short", 1, "a", ["a", "b"], 1)
        calibration = {
            "cache_key": "cal",
            "status": "ok",
            "calibrated_evidence_fixed": {emotion: 250000 for emotion in config["evidence"]["emotions"]},
        }
        extract = {
            "cache_key": "extract",
            "status": "ok",
            "aggregate_features": {"duration_ms": 10.0, "f0_mean_hz": 150.0},
        }
        _, record = attribute_turn(current, calibration, extract, None, None, config, [current])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["code"], "E_SEGMENT_TOO_SHORT")


if __name__ == "__main__":
    unittest.main()
