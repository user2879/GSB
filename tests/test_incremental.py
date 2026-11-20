import tempfile
import unittest

from animpipe.budget import Budget
from animpipe.demo import _mutate_keyframe
from animpipe.errors import E_CLOSURE_MISMATCH
from animpipe.incremental import (
    baseline_state,
    plan_closure,
    run_incremental,
    verify_closure,
)
from animpipe.pipeline import run_pipeline
from animpipe.samples import generate_all
from animpipe.skeleton import generate_skeleton
from animpipe.verify import compare_animations


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sample = generate_all(cls.tmp)
        cls.budget = Budget(4, 1 << 24, 1 << 20)

    def _serial(self, source, target_skel, fps):
        return run_pipeline(
            source,
            self.sample.source_skel,
            target_skel,
            fps,
            budget=self.budget,
            workers=1,
            block_size=1,
        ).output

    def test_baseline_matches_serial(self):
        state = baseline_state(
            self.sample.source_anim,
            self.sample.source_skel,
            self.sample.target_skel,
            self.sample.target_fps,
            self.budget,
            workers=4,
            block_size=4,
        )
        serial = self._serial(
            self.sample.source_anim, self.sample.target_skel, self.sample.target_fps
        )
        result = compare_animations(serial, state.last_output)
        self.assertTrue(result["equal"], result)

    def test_bone_length_closure_only_touches_root(self):
        state = baseline_state(
            self.sample.source_anim,
            self.sample.source_skel,
            self.sample.target_skel,
            self.sample.target_fps,
            self.budget,
            workers=4,
            block_size=4,
        )
        grown = generate_skeleton("target", 20261121, length_scale=1.6)
        manifest = plan_closure(
            state, reason="target_bone_length_change", bone_length_changed=True
        )
        run_incremental(state, manifest, updated_target_skel=grown)
        verify_closure(state, manifest)
        for entry in state.recomputed:
            if entry["stage"] == "retarget":
                self.assertEqual(entry["channels"], ["root"])
        serial = self._serial(self.sample.source_anim, grown, self.sample.target_fps)
        self.assertTrue(compare_animations(serial, state.last_output)["equal"])

    def test_keyframe_closure_is_a_strict_subset(self):
        state = baseline_state(
            self.sample.source_anim,
            self.sample.source_skel,
            self.sample.target_skel,
            self.sample.target_fps,
            self.budget,
            workers=4,
            block_size=4,
        )
        before = list(state.last_output.content_hash())
        modified = _mutate_keyframe(self.sample.source_anim, joint=5, frame=10)
        manifest = plan_closure(
            state,
            reason="keyframe_rotation_change",
            affected_source_frames=list(range(10, modified.frame_count)),
            affected_joints=[self.sample.source_anim.joint_names[5]],
        )
        run_incremental(state, manifest, updated_source=modified)
        verify_closure(state, manifest)
        rt_blocks = sorted(
            e["block"] for e in state.recomputed if e["stage"] == "retarget"
        )
        all_blocks = list(range(0, self.sample.source_anim.frame_count, 4))
        self.assertLess(len(rt_blocks), len(all_blocks))
        serial = self._serial(
            modified, self.sample.target_skel, self.sample.target_fps
        )
        self.assertTrue(compare_animations(serial, state.last_output)["equal"])

    def test_closure_mismatch_is_detectable(self):
        state = baseline_state(
            self.sample.source_anim,
            self.sample.source_skel,
            self.sample.target_skel,
            self.sample.target_fps,
            self.budget,
            workers=4,
            block_size=4,
        )
        manifest = plan_closure(
            state,
            reason="keyframe_rotation_change",
            affected_source_frames=[2],
        )
        fake = type(state)(
            source=state.source,
            source_skel=state.source_skel,
            target_skel=state.target_skel,
            fps=state.fps,
            budget=state.budget,
            workers=4,
            block_size=4,
            rt_cache={},
            rs_cache={},
            recomputed=[
                {"stage": "retarget", "block": 999, "channels": ["rotation"], "reason": "x"},
                {"stage": "resample", "block": 999, "channels": ["rotation"], "reason": "x"},
            ],
            manifest=manifest,
            last_output=None,
        )
        with self.assertRaises(E_CLOSURE_MISMATCH):
            verify_closure(fake, manifest)


if __name__ == "__main__":
    unittest.main()
