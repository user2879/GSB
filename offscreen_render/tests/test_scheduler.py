import threading
import unittest

from offscreen_render.errors import RenderError
from offscreen_render.samplegen import build_deterministic_scene
from offscreen_render.scheduler import TileScheduler, build_tile_specs, tile_ray_memory
from offscreen_render.assembly import assemble_supersampled
import tempfile


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        self.scene = build_deterministic_scene(self.state, self.out)

    def test_specs_cover_grid(self):
        specs = build_tile_specs(self.scene)
        cfg = self.scene.config
        gw, gh = cfg.width * cfg.supersample, cfg.height * cfg.supersample
        covered = sum(s.width * s.height for s in specs)
        self.assertEqual(covered, gw * gh)
        # 索引按行序连续
        self.assertEqual([s.index for s in specs], list(range(len(specs))))

    def test_order_independent_results(self):
        # 分两次以不同并发上限渲染，拼装结果必须逐元素一致。
        from dataclasses import replace
        scene_a = self.scene
        tiles_a = TileScheduler(scene_a).run()
        grid_a, _, _, _ = assemble_supersampled(tiles_a, scene_a.config)

        scene_b = replace(scene_a, config=replace(scene_a.config, concurrency=1))
        tiles_b = TileScheduler(scene_b).run()
        grid_b, _, _, _ = assemble_supersampled(tiles_b, scene_b.config)

        import numpy as np
        np.testing.assert_array_equal(grid_a, grid_b)

    def test_memory_limit_single_tile_rejected(self):
        from dataclasses import replace
        specs = build_tile_specs(self.scene)
        need = max(tile_ray_memory(s.width, s.height, self.scene.config.max_steps)
                   for s in specs)
        cfg = replace(self.scene.config, memory_limit_bytes=need - 1)
        scene = replace(self.scene, config=cfg)
        with self.assertRaises(RenderError) as ctx:
            TileScheduler(scene)
        self.assertEqual(ctx.exception.code, "E_2003")

    def test_interrupt_at_tile_boundary(self):
        event = threading.Event()
        event.set()  # 预置中断：所有块都不应启动
        scheduler = TileScheduler(self.scene, event)
        with self.assertRaises(RenderError) as ctx:
            scheduler.run()
        self.assertEqual(ctx.exception.code, "E_4001")

    def test_skip_completed_tiles(self):
        all_specs = build_tile_specs(self.scene)
        # 先手动渲染第 0 块并视为已完成，调度其余块。
        skip = {0}
        results = TileScheduler(self.scene).run(skip=skip)
        self.assertNotIn(0, {t.tile_index for t in results})
        self.assertEqual(len(results), len(all_specs) - 1)


if __name__ == "__main__":
    unittest.main()
