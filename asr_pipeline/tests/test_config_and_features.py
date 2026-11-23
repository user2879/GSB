import json
import os
import tempfile
import unittest

import numpy as np

from asr_pipeline import config as C
from asr_pipeline import features as F
from asr_pipeline import audio as A
from asr_pipeline.errors import E_CONFIG_INVALID, PipelineError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples")


class ConfigValidationTests(unittest.TestCase):
    def test_loads_valid_configs(self):
        sp = C.load_score_params(os.path.join(SAMPLE, "score_params.json"))
        sch = C.load_schedule_config(os.path.join(SAMPLE, "schedule.json"))
        self.assertEqual(sp["mel_bands"], 6)
        self.assertIn(sch["quant_mode"], C.ALLOWED_QUANT_MODES)

    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        return handle.name

    @staticmethod
    def _read(name):
        with open(os.path.join(SAMPLE, name)) as handle:
            return json.load(handle)

    def test_rejects_bad_mel_bands(self):
        sp = self._read("score_params.json")
        sp["mel_bands"] = 0
        path = self._write(sp)
        with self.assertRaises(PipelineError) as ctx:
            C.load_score_params(path)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_rejects_non_finite_bigram(self):
        sp = self._read("score_params.json")
        first = next(iter(sp["bigram_probs"]))
        sp["bigram_probs"][first][next(iter(sp["bigram_probs"][first]))] = float("inf")
        path = self._write(sp)
        with self.assertRaises(PipelineError) as ctx:
            C.load_score_params(path)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_rejects_bad_concurrency_and_quant(self):
        sch = self._read("schedule.json")
        sch["max_concurrency"] = C.MAX_CONCURRENCY + 1
        path = self._write(sch)
        with self.assertRaises(PipelineError) as ctx:
            C.load_schedule_config(path)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)
        sch["max_concurrency"] = 4
        sch["quant_mode"] = "fp99"
        path = self._write(sch)
        with self.assertRaises(PipelineError):
            C.load_schedule_config(path)

    def test_rejects_emission_dimension_mismatch(self):
        sp = self._read("score_params.json")
        sp["emission_table"][0].pop()
        path = self._write(sp)
        with self.assertRaises(PipelineError) as ctx:
            C.load_score_params(path)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)


class FeatureDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.sp = C.load_score_params(os.path.join(SAMPLE, "score_params.json"))
        self.sch = C.load_schedule_config(os.path.join(SAMPLE, "schedule.json"))
        self.item = A.validate_wav(os.path.join(SAMPLE, "wav", "utter_001.wav"))

    def test_frame_padding_marks_last_frame(self):
        frames, last_padded = F.frame_signal(
            self.item["samples"], self.sp["frame_length_ms"], self.sp["frame_shift_ms"]
        )
        self.assertEqual(frames.shape[0], last_padded.shape[0])
        self.assertTrue(last_padded[-1] or not last_padded.any())

    def test_emissions_are_deterministic_and_float64(self):
        first = F.compute_emissions(self.item["samples"], self.sp, self.sch)
        second = F.compute_emissions(self.item["samples"], self.sp, self.sch)
        self.assertEqual(first["emissions"].dtype, np.float64)
        np.testing.assert_array_equal(first["emissions"], second["emissions"])
        np.testing.assert_array_equal(first["point_costs"], second["point_costs"])

    def test_quantization_changes_values_but_stays_deterministic(self):
        q_sch = dict(self.sch, quant_mode="q16", quant_frac_bits=8)
        quantized = F.compute_emissions(self.item["samples"], self.sp, q_sch)
        again = F.compute_emissions(self.item["samples"], self.sp, q_sch)
        np.testing.assert_array_equal(quantized["emissions"], again["emissions"])
        baseline = F.compute_emissions(self.item["samples"], self.sp, self.sch)
        self.assertTrue(np.any(quantized["emissions"] != baseline["emissions"]))


if __name__ == "__main__":
    unittest.main()
