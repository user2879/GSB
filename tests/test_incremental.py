import json
import os

from tests._base import BaseCase
from vdu.fingerprint import label_fingerprint
from vdu.pipeline import Pipeline
from vdu.synth import relabel_transcript


def run_pipeline(case, tier="coarse", max_batches=None):
    pipeline = Pipeline(case.samples, case.config_path, case.snapshots,
                        tier_name=tier, max_batches=max_batches)
    return pipeline.run()


class IncrementalTests(BaseCase):
    def test_full_run_then_all_reused(self):
        first = run_pipeline(self)
        self.assertEqual(first["mode"], "rebuild")
        second = run_pipeline(self)
        self.assertTrue(second["audit"]["reused"])
        self.assertEqual(second["audit"]["computed"], [])
        self.assertEqual(first["segments"], second["segments"])

    def test_relabel_invalidates_only_affected_semantic_paths(self):
        first = run_pipeline(self)
        window_keys_first = {w["window_id"]: w["cache_key"]
                             for w in first["windows"]}
        new_transcript = relabel_transcript(
            self.transcript, 4, ["seg02"])
        self.write_transcript(new_transcript)
        self.transcript = new_transcript
        second = run_pipeline(self)
        self.assertNotEqual(label_fingerprint(self.transcript),
                            first["manifest"]["label_fp"])
        # Window features depend on audio only: every window key is reused.
        window_keys_second = {w["window_id"]: w["cache_key"]
                              for w in second["windows"]}
        self.assertEqual(window_keys_first, window_keys_second)
        self.assertEqual(
            [k for k in second["audit"]["reused"] if k.startswith("window:")],
            [w["cache_key"] for w in second["windows"]])
        # At least one word and one segment changed; unaffected stay reused.
        self.assertTrue(second["audit"]["computed"])
        self.assertTrue(second["audit"]["reused"])

    def test_global_relabel_changes_neighborhood_evidence(self):
        first = run_pipeline(self)
        new_transcript = relabel_transcript(
            self.transcript, 5,
            [s["segment_id"] for s in self.transcript["segments"]])
        self.write_transcript(new_transcript)
        self.transcript = new_transcript
        second = run_pipeline(self)
        # Even untouched-prefix segments can change via co-occurrence weights.
        self.assertTrue(second["audit"]["computed"])

    def test_full_rebuild_field_identical(self):
        baseline = run_pipeline(self)
        import shutil
        shutil.rmtree(self.snapshots, ignore_errors=True)
        os.makedirs(self.snapshots)
        rebuilt = run_pipeline(self)
        self.assertEqual(
            [w["features"] for w in baseline["windows"]],
            [w["features"] for w in rebuilt["windows"]])
        self.assertEqual(
            [w["strength"] for w in baseline["words"]],
            [w["strength"] for w in rebuilt["words"]])
        self.assertEqual(
            [c["calibration"] for c in baseline["calibrations"]],
            [c["calibration"] for c in rebuilt["calibrations"]])
        self.assertEqual(baseline["segments"], rebuilt["segments"])

    def test_single_segment_audio_change_scoped_invalidation(self):
        first = run_pipeline(self)
        seg = self.transcript["segments"][1]
        # Perturb one word's audio span by moving its boundary and re-rendering.
        seg["words"][0]["end"] += 40
        seg["words"][1]["start"] += 40
        self.write_transcript(self.transcript)
        self.rewrite_audio(self.transcript)
        second = run_pipeline(self)
        self.assertTrue(second["audit"]["computed"])
        # Segment summaries all carry word/window reference sets.
        for summary in second["segments"]:
            self.assertTrue(summary["word_keys"])
            self.assertTrue(summary["window_ids"])

    def test_ledger_audit_records_reasons(self):
        run_pipeline(self)
        current = os.path.join(self.snapshots, "current", "ledger.json")
        with open(current, encoding="utf-8") as handle:
            ledger = json.load(handle)
        statuses = {entry["status"] for entry in ledger["audit"]}
        self.assertIn("valid", statuses)
