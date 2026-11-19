import copy
import tempfile
import unittest

from score_pipeline.engine import (
    Engine,
    compute_closure,
    make_operation,
)
from score_pipeline.errors import PipelineError
from score_pipeline.generator import generate_score
from score_pipeline.tempo import TempoMap


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.score = generate_score(seed=123)
        self.engine = Engine(self.score)

    def test_full_rebuild_deterministic_hash(self):
        first = self.engine.full_rebuild(version=0)
        second = self.engine.full_rebuild(version=0)
        self.assertEqual(first.timeline_hash, second.timeline_hash)
        self.assertEqual(first.rendered_notes, second.rendered_notes)

    def test_incremental_equals_full_rebuild_all_versions(self):
        state = self.engine.full_rebuild(version=0)
        expr = {
            "tempo": self.score.tempo,
            "dynamics": self.score.dynamics,
            "pedals": self.score.pedals,
        }
        specs = [
            ("tempo", self._mutated_tempo()),
            ("dynamics", self._mutated_dynamics()),
            ("pedals", self._mutated_pedals()),
        ]
        for index, (kind, payload) in enumerate(specs, start=1):
            kwargs = {kind: payload}
            op = make_operation(
                index, state, kind, total_beats=self.score.total_beats, **kwargs
            )
            state, closure = self.engine.incremental_apply(state, op)
            expr[kind] = payload
            rebuilt = self.engine.full_rebuild(
                tempo=expr["tempo"],
                dynamics=expr["dynamics"],
                pedals=expr["pedals"],
                version=index,
            )
            self.assertEqual(state.timeline_hash, rebuilt.timeline_hash)
            self.assertEqual(state.rendered_notes, rebuilt.rendered_notes)
            self.assertEqual(state.rendered_pedals, rebuilt.rendered_pedals)

    def test_closure_tempo_shifts_outside_and_recomputes_inside(self):
        baseline = self.engine.full_rebuild(version=0)
        new_tempo = self._mutated_tempo()
        op = make_operation(1, baseline, "tempo", tempo=new_tempo,
                            total_beats=self.score.total_beats)
        part_id = sorted(self.score.parts)[0]
        old_map = TempoMap(baseline.tempo)
        new_map = TempoMap(new_tempo)
        closure = compute_closure(
            "tempo", op.changed_range, True, new_map, old_map,
            self.score.parts[part_id],
        )
        union = set(closure["recompute_note_ids"]) | set(closure["shift_note_ids"])
        self.assertEqual(union, {n.note_id for n in self.score.parts[part_id]})
        self.assertTrue(set(closure["recompute_note_ids"]).isdisjoint(closure["shift_note_ids"]))

    def test_closure_dynamics_keeps_outside_notes_unchanged(self):
        baseline = self.engine.full_rebuild(version=0)
        new_dynamics = self._mutated_dynamics()
        op = make_operation(1, baseline, "dynamics", dynamics=new_dynamics,
                            total_beats=self.score.total_beats)
        state, closure = self.engine.incremental_apply(baseline, op)
        part_id = sorted(self.score.parts)[0]
        outside = set(closure["parts"][part_id]["shift_note_ids"])
        before = {row["note_id"]: row for row in baseline.rendered_notes[part_id]}
        after = {row["note_id"]: row for row in state.rendered_notes[part_id]}
        for note_id in outside:
            self.assertEqual(before[note_id], after[note_id])

    def test_touched_shards_match_closure(self):
        baseline = self.engine.full_rebuild(version=0)
        op = make_operation(1, baseline, "dynamics", dynamics=self._mutated_dynamics(),
                            total_beats=self.score.total_beats)
        _, closure = self.engine.incremental_apply(baseline, op)
        expected_shards = set()
        from score_pipeline.engine import shard_index

        for part_id, part_closure in closure["parts"].items():
            for note_id in part_closure["recompute_note_ids"]:
                note = next(n for n in self.score.parts[part_id] if n.note_id == note_id)
                expected_shards.add(f"{part_id}:{shard_index(note.start_beat)}")
        self.assertEqual(set(closure["touched_shard_ids"]), expected_shards)

    def test_replay_fragment_matches_historical_hash(self):
        state = self.engine.full_rebuild(version=0)
        operations = []
        for index, (kind, payload) in enumerate(
            [("tempo", self._mutated_tempo()), ("dynamics", self._mutated_dynamics())], start=1
        ):
            op = make_operation(
                index, state, kind, total_beats=self.score.total_beats, **{kind: payload}
            )
            operations.append(op)
            state, _ = self.engine.incremental_apply(state, op)
        shard_id = sorted(state.shards)[0]
        fragment = self.engine.replay_fragment(operations, version=1, shard_id=shard_id)
        historical = self.engine.full_rebuild(
            tempo=operations[0].tempo, version=1
        )
        self.assertEqual(fragment["content_hash"], historical.shards[shard_id]["content_hash"])

    def _mutated_tempo(self):
        tempo = [dict(point) for point in self.score.tempo]
        tempo[1]["bpm"] = round(float(tempo[1]["bpm"]) + 20.0, 6)
        return tempo

    def _mutated_dynamics(self):
        dynamics = [dict(item) for item in self.score.dynamics]
        dynamics[0]["end_gain"] = round(float(dynamics[0]["end_gain"]) + 0.15, 6)
        return dynamics

    def _mutated_pedals(self):
        pedals = [dict(item) for item in self.score.pedals]
        pedals[0]["end_beat"] = round(float(pedals[0]["end_beat"]) - 0.5, 6)
        return pedals


if __name__ == "__main__":
    unittest.main()
