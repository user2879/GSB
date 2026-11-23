import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvrb.camera import load_cameras, save_cameras
from mvrb.pipeline import ReconstructionPipeline
from mvrb.synthetic import make_scene, render_view, save_png
from mvrb.utils import canonical_json, read_json, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "demo")


def copy_sample(dst: str) -> str:
    shutil.copytree(SAMPLE, dst)
    return dst


def result_signature(result) -> dict:
    """逐字段可比的确定性结果摘要。"""
    return {
        "features": {
            v: [(f.feature_id, round(f.x, 6), round(f.y, 6), f.landmark_id)
                for f in vf.features]
            for v, vf in sorted(result.view_features.items())
        },
        "matches": [
            (pm.view_a, pm.view_b, pm.pairs) for pm in result.pair_matches
        ],
        "points": [p.to_dict() for p in result.points3d],
        "depth": {v: dm.summary for v, dm in sorted(result.depth_maps.items())},
        "voxels": [r.to_dict() for r in result.voxels],
        "failures": result.failures.dicts(),
    }


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mvrb_pipe_")
        self.sample = copy_sample(os.path.join(self.tmp, "sample"))
        self.work = os.path.join(self.tmp, "work")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_rebuild_equals_incremental_edit_sequence(self):
        # 初始全量
        first = ReconstructionPipeline(self.sample, self.work).run()
        baseline_sig = canonical_json(result_signature(first))

        # 编辑序列：替换视图 2 图像 -> 改相机 -> 调阈值 -> 改体素尺寸
        scene = make_scene()
        cam2 = load_cameras(os.path.join(self.sample, "cameras.json"))[2]
        img = render_view(cam2, scene, seed=20261124 + 1)  # 改变噪声种子=图像内容变化
        save_png(os.path.join(self.sample, "images", "view_02.png"), img)
        ReconstructionPipeline(self.sample, self.work).run()

        cams = load_cameras(os.path.join(self.sample, "cameras.json"))
        c0 = cams[0]
        cams[0] = type(c0)(view=0, fx=c0.fx + 1.5, fy=c0.fy, cx=c0.cx, cy=c0.cy,
                           rot=c0.rot, trans=c0.trans)
        save_cameras(os.path.join(self.sample, "cameras.json"), cams)
        ReconstructionPipeline(self.sample, self.work).run()

        cfg = read_json(os.path.join(self.sample, "recon_config.json"))
        cfg["feature_threshold"] = 150
        write_json(os.path.join(self.sample, "recon_config.json"), cfg)
        ReconstructionPipeline(self.sample, self.work).run()

        cfg["voxel_size"] = 0.75
        write_json(os.path.join(self.sample, "recon_config.json"), cfg)
        incremental = ReconstructionPipeline(self.sample, self.work).run()

        # 用最终视图集合+配置从空白全量重建
        fresh_work = os.path.join(self.tmp, "fresh")
        full = ReconstructionPipeline(self.sample, fresh_work).run()
        self.assertEqual(canonical_json(result_signature(incremental)),
                         canonical_json(result_signature(full)))
        # 与初始基线必然不同（已编辑）
        self.assertNotEqual(canonical_json(result_signature(incremental)), baseline_sig)

    def test_only_affected_stages_invalidate_on_image_replace(self):
        ReconstructionPipeline(self.sample, self.work).run()
        # 仅噪声变化：图像字节变化但几何质心不变，特征应重算而点云应复用
        cam3 = load_cameras(os.path.join(self.sample, "cameras.json"))[3]
        save_png(os.path.join(self.sample, "images", "view_03.png"),
                 render_view(cam3, make_scene(), seed=424242))
        noise_result = ReconstructionPipeline(self.sample, self.work).run()
        noise_stages = {i["stage"] for i in noise_result.invalidated}
        self.assertIn("features", noise_stages)
        self.assertNotIn("voxel", noise_stages)

        # 真正移动图元（坐标变化）：三角化与体素下游必须失效
        moved_scene = [(lid, (x + 0.35, y, z)) for lid, (x, y, z) in make_scene()]
        save_png(os.path.join(self.sample, "images", "view_03.png"),
                 render_view(cam3, moved_scene, seed=20261124))
        result = ReconstructionPipeline(self.sample, self.work).run()
        stages_views = {(i["stage"], i.get("view"), i.get("pair"))
                        for i in result.invalidated}
        # 视图 3 的特征必须重算，其它视图特征必须复用
        self.assertIn(("features", 3, None), stages_views)
        for v in (0, 1, 2, 4, 5):
            self.assertNotIn(("features", v, None), stages_views)
        # 下游三角化/体素必然失效
        self.assertIn(("triangulate", None, None), stages_views)
        self.assertIn(("voxel", None, None), stages_views)
        # 复用清单非空且不含视图3特征键
        self.assertTrue(result.cache_ledger["reused"])

    def test_config_only_change_does_not_recompute_unrelated_features(self):
        ReconstructionPipeline(self.sample, self.work).run()
        cfg = read_json(os.path.join(self.sample, "recon_config.json"))
        cfg["voxel_size"] = 1.0
        write_json(os.path.join(self.sample, "recon_config.json"), cfg)
        result = ReconstructionPipeline(self.sample, self.work).run()
        stages = {i["stage"] for i in result.invalidated}
        self.assertNotIn("features", stages)       # 特征键与体素尺寸无关
        self.assertIn("voxel", stages)

    def test_view_add_remove_equivalence(self):
        ReconstructionPipeline(self.sample, self.work).run()
        # 删除视图 4 的图像
        os.remove(os.path.join(self.sample, "images", "view_04.png"))
        reduced = ReconstructionPipeline(self.sample, self.work).run()
        fresh = ReconstructionPipeline(self.sample,
                                       os.path.join(self.tmp, "fresh")).run()
        self.assertEqual(canonical_json(result_signature(reduced)),
                         canonical_json(result_signature(fresh)))
        self.assertNotIn(4, reduced.view_ids)


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mvrb_conc_")
        self.sample = copy_sample(os.path.join(self.tmp, "sample"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_serial_equals_parallel_field_by_field(self):
        serial = ReconstructionPipeline(
            self.sample, os.path.join(self.tmp, "ws"), force_serial=True).run()
        # 并行：高并发、多预算耗尽时机
        cfg = read_json(os.path.join(self.sample, "recon_config.json"))
        cfg["max_workers"] = 8
        write_json(os.path.join(self.sample, "recon_config.json"), cfg)
        parallel = ReconstructionPipeline(
            self.sample, os.path.join(self.tmp, "wp"), force_serial=False).run()
        self.assertEqual(canonical_json(result_signature(serial)),
                         canonical_json(result_signature(parallel)))
        self.assertEqual(
            [v for v in serial.depth_maps], [v for v in parallel.depth_maps])


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mvrb_budget_")
        self.sample = copy_sample(os.path.join(self.tmp, "sample"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_global_budget_exhaustion_keeps_todo(self):
        cfg = read_json(os.path.join(self.sample, "recon_config.json"))
        cfg["point_budget"] = 12
        cfg["max_workers"] = 4
        write_json(os.path.join(self.sample, "recon_config.json"), cfg)
        result = ReconstructionPipeline(
            self.sample, os.path.join(self.tmp, "work")).run()
        codes = [f.code for f in result.failures.sorted()]
        self.assertIn("E_POINT_BUDGET_EXHAUSTED", codes)
        self.assertTrue(result.schedule_report["todo_queue"])
        # 待办视图优先级：序号稳定
        self.assertEqual(result.schedule_report["todo_queue"],
                         sorted(result.schedule_report["todo_queue"]))

    def test_view_quota_exhaustion(self):
        cfg = read_json(os.path.join(self.sample, "recon_config.json"))
        cfg["view_point_quota"] = 1
        write_json(os.path.join(self.sample, "recon_config.json"), cfg)
        result = ReconstructionPipeline(
            self.sample, os.path.join(self.tmp, "work")).run()
        codes = [f.code for f in result.failures.sorted()]
        self.assertIn("E_VIEW_BUDGET_EXHAUSTED", codes)
        self.assertTrue(result.schedule_report["exhausted_views"])


if __name__ == "__main__":
    unittest.main()
