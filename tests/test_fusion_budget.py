import unittest

from fixedpoint_render.budget import allocate
from fixedpoint_render.config import CONDITION_TYPES
from fixedpoint_render.encoding import encode_motion, encode_text
from fixedpoint_render.fixedpoint import q_scale
from fixedpoint_render.fusion import fuse_conditions, normalize_weights
from fixedpoint_render.samplegen import default_config


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.config = default_config()

    def _encoded(self):
        from PIL import Image
        from fixedpoint_render.encoding import encode_image
        img = Image.new("RGB", (self.config.width, self.config.height),
                        (10, 20, 30))
        motion = {"tiles": [{"tile": t, "vx": t, "vy": -t}
                            for t in range(self.config.tile_rows)]}
        return {
            "text": encode_text(0, ["city", "rain"], self.config),
            "image": encode_image(0, img, self.config),
            "motion": encode_motion(0, motion, self.config),
        }

    def test_normalized_weights_sum_to_one(self):
        weights, _ = normalize_weights(self.config)
        self.assertEqual(sum(weights.values()), q_scale(self.config.frac_bits))

    def test_fusion_order_and_traces(self):
        result = fuse_conditions(0, self._encoded(), self.config)
        self.assertEqual(len(result.vector), self.config.condition_dim)
        ops = [t.op for t in result.traces]
        # image < motion < text, dimension 0 must precede dimension 1.
        self.assertLess(ops.index("fuse:image[0]"),
                        ops.index("fuse:motion[0]"))
        self.assertLess(ops.index("fuse:motion[0]"),
                        ops.index("fuse:text[0]"))
        self.assertLess(ops.index("fuse:image[0]"),
                        ops.index("fuse:image[1]"))

    def test_weight_change_alters_integers_and_residuals(self):
        encoded = self._encoded()
        base = fuse_conditions(0, encoded, self.config)
        cfg2 = self.config.__class__(**{
            **self.config.to_dict(),
            "weights": {"text": "0.10", "image": "0.80", "motion": "0.10"}})
        changed = fuse_conditions(0, encoded, cfg2)
        self.assertNotEqual(base.vector, changed.vector)
        self.assertNotEqual(
            [t.residual for t in base.traces],
            [t.residual for t in changed.traces])

    def test_qmode_change_alters_quantization(self):
        encoded = self._encoded()
        trunc = fuse_conditions(0, encoded,
                                self.config.__class__(**{
                                    **self.config.to_dict(), "qmode": "trunc"}))
        near = fuse_conditions(0, encoded,
                               self.config.__class__(**{
                                   **self.config.to_dict(),
                                   "qmode": "nearest"}))
        self.assertNotEqual(
            [t.value for t in trunc.traces],
            [t.value for t in near.traces])


class BudgetTests(unittest.TestCase):
    def test_budget_never_exceeds_caps_and_marks_degraded(self):
        config = default_config()  # global 20 < 5 frames * 6 = 30 demand
        from PIL import Image
        from fixedpoint_render.encoding import encode_image
        encoded_motion, fusions = {}, {}
        for f in range(5):
            img = Image.new("RGB", (config.width, config.height),
                            (f * 5, 10, 20))
            motion = {"tiles": [{"tile": t, "vx": f + t, "vy": f}
                                for t in range(config.tile_rows)],
                      "frame_cover": 50 + f * 5}
            enc = {
                "text": encode_text(f, ["city"], config),
                "image": encode_image(f, img, config),
                "motion": encode_motion(f, motion, config),
            }
            encoded_motion[f] = enc["motion"]
            fusions[f] = fuse_conditions(f, enc, config)
        plan = allocate(5, encoded_motion, fusions, config)
        self.assertLessEqual(plan.total_allocated, plan.global_budget)
        for entry in plan.entries:
            self.assertLessEqual(entry.allocated, entry.demand)
            self.assertEqual(entry.truncated, entry.demand - entry.allocated)
            self.assertEqual(entry.degraded, entry.truncated > 0)
        self.assertTrue(any(e.degraded for e in plan.entries))

    def test_allocation_determinism(self):
        config = default_config()
        from PIL import Image
        from fixedpoint_render.encoding import encode_image
        encoded_motion, fusions = {}, {}
        for f in range(4):
            img = Image.new("RGB", (config.width, config.height))
            motion = {"tiles": [{"tile": t, "vx": 1, "vy": 1}
                                for t in range(config.tile_rows)]}
            enc = {
                "text": encode_text(f, ["fog"], config),
                "image": encode_image(f, img, config),
                "motion": encode_motion(f, motion, config),
            }
            encoded_motion[f] = enc["motion"]
            fusions[f] = fuse_conditions(f, enc, config)
        first = allocate(4, encoded_motion, fusions, config).to_dict()
        second = allocate(4, encoded_motion, fusions, config).to_dict()
        self.assertEqual(first, second)

    def test_per_frame_override(self):
        config = default_config()
        from PIL import Image
        from fixedpoint_render.encoding import encode_image
        encoded_motion, fusions = {}, {}
        for f in range(3):
            img = Image.new("RGB", (config.width, config.height))
            motion = {"tiles": [{"tile": t, "vx": 2, "vy": 2}
                                for t in range(config.tile_rows)]}
            enc = {
                "text": encode_text(f, ["neon"], config),
                "image": encode_image(f, img, config),
                "motion": encode_motion(f, motion, config),
            }
            encoded_motion[f] = enc["motion"]
            fusions[f] = fuse_conditions(f, enc, config)
        plan = allocate(3, encoded_motion, fusions, config, {0: 2})
        self.assertEqual(plan.entries[0].demand, 2)


if __name__ == "__main__":
    unittest.main()
