import unittest

import numpy as np

from offscreen_render.geometry import (
    build_interval_chain,
    interval_at,
    ray_aabb,
    validate_chain_integrity,
)
from offscreen_render.scene import VolumeBox


def box(index, lo, hi, sigma=0.5, ior=1.3):
    return VolumeBox(index, f"b{index}", np.array(lo, float), np.array(hi, float),
                     sigma, 0.5, ior)


class TestRayAABB(unittest.TestCase):
    def test_hit_and_miss(self):
        lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        hit = ray_aabb(np.array([0.5, 0.5, -1.0]), np.array([0.0, 0.0, 1.0]), lo, hi)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], 1.0)
        self.assertAlmostEqual(hit[1], 2.0)
        miss = ray_aabb(np.array([5.0, 5.0, -1.0]), np.array([0.0, 0.0, 1.0]), lo, hi)
        self.assertIsNone(miss)

    def test_parallel_axis_inside(self):
        lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        hit = ray_aabb(np.array([0.5, 0.5, 0.5]), np.array([1.0, 0.0, 0.0]), lo, hi)
        self.assertIsNotNone(hit)


class TestIntervalChain(unittest.TestCase):
    def setUp(self):
        # 两个重叠盒：b0 先声明，重叠区应归 b0。
        self.b0 = box(0, [0, 0, 0], [10, 10, 10], sigma=0.2)
        self.b1 = box(1, [2, 2, 2], [8, 8, 8], sigma=0.9)

    def test_overlap_priority_by_declaration(self):
        o = np.array([-1.0, 5.0, 5.0])
        d = np.array([1.0, 0.0, 0.0])
        chain = build_interval_chain(o, d, (self.b0, self.b1))
        validate_chain_integrity(chain)
        owners = [(round(s.t_enter, 3), round(s.t_exit, 3), s.volume.index) for s in chain]
        # [1,2) 仅 b0；[2,8) 重叠归 b0；[8,11) 仅 b0 -> 整段都属于 b0
        self.assertTrue(all(owner == 0 for _, _, owner in owners))
        self.assertAlmostEqual(chain[0].t_enter, 1.0)
        self.assertAlmostEqual(chain[-1].t_exit, 11.0)

    def test_chain_no_gap_no_double_count(self):
        b2 = box(2, [10, 0, 0], [20, 10, 10], sigma=0.4)
        o = np.array([-1.0, 5.0, 5.0])
        d = np.array([1.0, 0.0, 0.0])
        chain = build_interval_chain(o, d, (self.b0, b2))
        validate_chain_integrity(chain)
        # 首尾相接：前一段退出点 == 后一段进入点
        for a, b in zip(chain[:-1], chain[1:]):
            self.assertAlmostEqual(a.t_exit, b.t_enter)

    def test_midpoint_attribution_at_boundary(self):
        thin = box(0, [0, 0, 4.0], [10, 10, 6.0], sigma=0.3)
        o = np.array([5.0, 5.0, 0.0])
        d = np.array([0.0, 0.0, 1.0])
        chain = build_interval_chain(o, d, (thin,))
        self.assertEqual(len(chain), 1)
        seg = chain[0]
        # 边界点显式归属：进入边界属该段，退出边界由 interval_at 处理
        self.assertIsNotNone(interval_at(chain, seg.t_enter))
        self.assertIsNone(interval_at(chain, seg.t_exit + 1.0))

    def test_declaration_order_deterministic(self):
        o = np.array([-1.0, 5.0, 5.0])
        d = np.array([1.0, 0.0, 0.0])
        chain_a = build_interval_chain(o, d, (self.b0, self.b1))
        chain_b = build_interval_chain(o, d, (self.b0, self.b1))
        self.assertEqual([s.volume.index for s in chain_a],
                         [s.volume.index for s in chain_b])


if __name__ == "__main__":
    unittest.main()
