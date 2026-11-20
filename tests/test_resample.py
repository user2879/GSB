import unittest

import numpy as np

from animpipe.animation import Animation
from animpipe.resample import output_timestamps, resample
from animpipe.skeleton import generate_skeleton
from animpipe.samples import generate_source_animation


class ResampleTests(unittest.TestCase):
    def test_grid_never_extrapolates(self):
        times = np.linspace(0.0, 1.0, 11)
        grid = output_timestamps(times, fps=7.0)
        self.assertGreaterEqual(grid[0], times[0] - 1e-12)
        self.assertLessEqual(grid[-1], times[-1] + 1e-12)
        self.assertAlmostEqual(grid[-1], times[-1])

    def test_interp_modes_land_on_keyframes(self):
        skel = generate_skeleton("source", 1)
        base = generate_source_animation(skel, frame_count=8, fps=4.0)
        for mode in ("slerp", "nlerp"):
            anim = Animation(
                joint_names=base.joint_names,
                times=base.times.copy(),
                rotations=base.rotations.copy(),
                root_translations=base.root_translations.copy(),
                interp=mode,
            )
            out = resample(anim, fps=4.0)
            np.testing.assert_allclose(out.times, base.times, atol=1e-12)
            np.testing.assert_allclose(
                out.rotations, base.rotations, atol=1e-9
            )
        # step interpolation holds the previous keyframe; the first
        # sample and timestamps still match exactly.
        step_anim = Animation(
            joint_names=base.joint_names,
            times=base.times.copy(),
            rotations=base.rotations.copy(),
            root_translations=base.root_translations.copy(),
            interp="step",
        )
        step_out = resample(step_anim, fps=4.0)
        np.testing.assert_allclose(step_out.times, base.times, atol=1e-12)
        np.testing.assert_allclose(
            step_out.rotations[0], base.rotations[0], atol=1e-12
        )
        np.testing.assert_allclose(
            step_out.rotations[1], base.rotations[0], atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
