import tempfile
import unittest
from pathlib import Path

from score_pipeline.engine import Engine
from score_pipeline.errors import (
    E_BEAT_RANGE,
    E_PEDAL_OVERLAP,
    E_VELOCITY_MONO,
    PipelineError,
)
from score_pipeline.generator import generate_score, write_score_bundle
from score_pipeline.injection import (
    FAULT_PEDAL_OVERLAP,
    FAULT_TEMPO_NON_MONOTONIC,
    InjectionConfig,
    build_faulty_payload,
)
from score_pipeline.model import Note, read_json, write_json
from score_pipeline.persistence import Store
from score_pipeline.validator import validate_workspace


class ErrorChainTests(unittest.TestCase):
    def setUp(self):
        self.score = generate_score(seed=77)
        self.engine = Engine(self.score)

    def test_tempo_non_monotonic(self):
        _, tempo, _, _ = build_faulty_payload(
            FAULT_TEMPO_NON_MONOTONIC, self.score,
            self.score.tempo, self.score.dynamics, self.score.pedals,
        )
        with self.assertRaises(PipelineError) as ctx:
            self.engine.full_rebuild(tempo=tempo)
        self.assertEqual(ctx.exception.code, E_VELOCITY_MONO)

    def test_note_zero_duration(self):
        score = generate_score(seed=77)
        first_part = sorted(score.parts)[0]
        notes = list(score.parts[first_part])
        notes[0] = Note(
            note_id=notes[0].note_id, pitch=notes[0].pitch,
            start_beat=notes[0].start_beat, duration_beats=0.0,
            velocity=notes[0].velocity, articulation=notes[0].articulation,
        )
        score.parts[first_part] = notes
        with self.assertRaises(PipelineError) as ctx:
            Engine(score).full_rebuild()
        self.assertEqual(ctx.exception.code, E_BEAT_RANGE)

    def test_pedal_overlap(self):
        _, _, _, pedals = build_faulty_payload(
            FAULT_PEDAL_OVERLAP, self.score,
            self.score.tempo, self.score.dynamics, self.score.pedals,
        )
        with self.assertRaises(PipelineError) as ctx:
            self.engine.full_rebuild(pedals=pedals)
        self.assertEqual(ctx.exception.code, E_PEDAL_OVERLAP)

    def test_injection_config_serializable_and_stable(self):
        config = InjectionConfig(
            param_faults={"tempo_non_monotonic": 1},
            crash_at_version=2,
            crash_point="commit_crash_after_blob",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "injection.json"
            config.save(str(path))
            loaded = InjectionConfig.load(str(path))
            self.assertEqual(loaded.to_dict(), config.to_dict())


class ValidatorTests(unittest.TestCase):
    def test_validator_passes_on_clean_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            write_score_bundle(generate_score(seed=88), str(workspace / "sample"))
            score = generate_score(seed=88)
            store = Store(workspace / "state")
            store.commit_initial(Engine(score).full_rebuild(version=0))
            report = validate_workspace(str(workspace))
            self.assertTrue(report["ok"], report["first_failure"])
            self.assertEqual(
                report["checked_note_total"], report["score_note_total"]
            )

    def test_validator_locates_first_bad_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            write_score_bundle(generate_score(seed=88), str(workspace / "sample"))
            score = generate_score(seed=88)
            store = Store(workspace / "state")
            state = Engine(score).full_rebuild(version=0)
            store.commit_initial(state)
            manifest = store.read_manifest()
            digest = sorted(manifest["shard_blobs"].values())[0]
            blob_path = store.blobs / f"{digest}.json"
            blob = read_json(blob_path)
            target_note = blob["notes"][0]["note_id"]
            blob["notes"][0]["start_ms"] = blob["notes"][0]["start_ms"] + 50.0
            write_json(blob_path, blob)
            report = validate_workspace(str(workspace))
            self.assertFalse(report["ok"])
            self.assertEqual(report["first_failure"]["first_bad_note"], target_note)


if __name__ == "__main__":
    unittest.main()
