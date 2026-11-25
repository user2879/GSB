from __future__ import annotations

import unittest

from speaker_diag.engine import infer_all_local
from speaker_diag.spec import load_workload_data
from speaker_diag.synthetic import FEATURE_SCALE, synthesize

from tests.helpers import tiny_spec, tiny_workload


class DeterminismTests(unittest.TestCase):
    def test_worker_and_request_order_do_not_change_audio_windows_or_features(self) -> None:
        spec = tiny_spec()
        bundle = synthesize(spec)
        raw = tiny_workload()
        raw["workers"] = list(reversed(raw["workers"]))
        raw["requests"] = list(reversed(raw["requests"]))
        reordered = load_workload_data(raw, source_name="<reordered>")
        bundle2 = synthesize(reordered)
        self.assertEqual(spec.window_starts_samples, reordered.window_starts_samples)
        self.assertEqual(bundle.window_digest, bundle2.window_digest)
        self.assertEqual(bundle.truth_digest, bundle2.truth_digest)
        self.assertEqual(bundle.feature_digest, bundle2.feature_digest)
        self.assertEqual(bundle.pcm_digest, bundle2.pcm_digest)

    def test_repeat_synthesis_is_byte_identical(self) -> None:
        spec = tiny_spec()
        first = synthesize(spec)
        second = synthesize(spec)
        self.assertEqual(first.pcm.tobytes(), second.pcm.tobytes())
        self.assertEqual(first.features.tobytes(), second.features.tobytes())
        self.assertEqual(first.truth_manifest(), second.truth_manifest())

    def test_features_are_integer_fixed_point_and_truth_has_explanations(self) -> None:
        spec = tiny_spec()
        bundle = synthesize(spec)
        self.assertEqual(bundle.features.dtype.kind, "i")
        self.assertGreaterEqual(int(bundle.features.min()), 0)
        self.assertLessEqual(int(bundle.features.max()), FEATURE_SCALE)
        for window in bundle.windows:
            self.assertIn(window.truth_source, {"single_active", "overlap_dominant", "silence"})
            self.assertTrue(window.explanation)
            self.assertEqual(window.end_sample - window.start_sample, spec.window_samples)

    def test_batch_execution_order_does_not_change_local_attribution(self) -> None:
        spec = tiny_spec()
        bundle = synthesize(spec)
        all_results = infer_all_local(spec, bundle)
        reversed_results = infer_all_local(spec, bundle, batch_ids=list(reversed(list(all_results))))
        for batch_id in all_results:
            self.assertEqual(
                all_results[batch_id].receipt["windows"],
                reversed_results[batch_id].receipt["windows"],
            )


if __name__ == "__main__":
    unittest.main()

