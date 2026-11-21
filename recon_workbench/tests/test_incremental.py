"""增量重算：内容/参数指纹精确失效，未失效产物直接复用。"""
from __future__ import annotations

import unittest


from recon_workbench.artifact_store import ArtifactStore
from recon_workbench.config import RebuildConfig
from recon_workbench.editing import apply_edit_sequence
from recon_workbench.engine import WorkbenchEngine
from recon_workbench.loader import load_sample
from recon_workbench.utils import read_json, atomic_write_json

from ._helpers import SampleHarness


class IncrementalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()
        self.config = RebuildConfig()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def _engine(self, sample_dir: str, work_dir: str) -> WorkbenchEngine:
        load = load_sample(sample_dir)
        store = ArtifactStore(work_dir)
        return WorkbenchEngine(load, self.config, store, concurrency=1)

    def test_full_then_reuse(self) -> None:
        engine = self._engine(
            self.harness.sample_dir, self.harness.root + "/w1"
        )
        engine.run("float64", version=1)
        decision = engine.last_context.decision
        self.assertTrue(decision.recomputed)
        self.assertEqual(decision.reused, [])

        engine2 = self._engine(
            self.harness.sample_dir, self.harness.root + "/w1"
        )
        engine2.run("float64", version=2)
        decision2 = engine2.last_context.decision
        self.assertTrue(decision2.reused)
        self.assertEqual(decision2.recomputed, [])

    def test_image_edit_invalidates_dependent_chain(self) -> None:
        work = self.harness.root + "/w2"
        engine = self._engine(self.harness.sample_dir, work)
        engine.run("float64", version=1)
        keys_before = set(engine.last_context.decision.recomputed)

        edited = self.harness.root + "/edited"
        apply_edit_sequence(
            self.harness.sample_dir, edited,
            [{"command": "replace_image", "view": 2,
              "x": 40, "y": 40, "size": 18,
              "color": [255, 0, 0]}],
        )
        engine2 = self._engine(edited, work)
        engine2.run("float64", version=2)
        reused = set(engine2.last_context.decision.reused)
        recomputed = set(engine2.last_context.decision.recomputed)

        # 未触及的视图 0、4 特征缓存必须复用
        feat_keys = engine2._feature_cache_keys()
        self.assertIn(feat_keys[0], reused)
        self.assertIn(feat_keys[4], reused)
        # 被替换视图的特征键必须重算
        self.assertIn(feat_keys[2], recomputed)
        # 点云依赖全部深度，必然重算
        self.assertTrue(
            any(k.startswith("cloud:") for k in recomputed)
        )

    def test_config_edit_invalidates_all(self) -> None:
        work = self.harness.root + "/w3"
        engine = self._engine(self.harness.sample_dir, work)
        engine.run("float64", version=1)

        data = read_json(self.harness.config_path)
        data["corner_threshold"] = data["corner_threshold"] * 1.5
        atomic_write_json(self.harness.config_path, data)
        new_config = RebuildConfig.from_dict(data)
        self.config = new_config
        engine2 = self._engine(self.harness.sample_dir, work)
        engine2.run("float64", version=2)
        # 参数指纹变化 => 全部键重算
        self.assertEqual(engine2.last_context.decision.reused, [])
        self.assertTrue(engine2.last_context.decision.recomputed)

    def test_camera_edit_keeps_features(self) -> None:
        work = self.harness.root + "/w4"
        engine = self._engine(self.harness.sample_dir, work)
        engine.run("float64", version=1)

        edited = self.harness.root + "/edited_cam"
        apply_edit_sequence(
            self.harness.sample_dir, edited,
            [{"command": "edit_camera", "view": 1,
              "field": "fov_deg", "value": 61.0}],
        )
        engine2 = self._engine(edited, work)
        engine2.run("float64", version=2)
        reused = set(engine2.last_context.decision.reused)
        recomputed = set(engine2.last_context.decision.recomputed)
        feat_keys = engine2._feature_cache_keys()
        # 相机参数不影响角点检测：图像未变则特征键保持不变（可复用）
        self.assertIn(feat_keys[1], reused)
        # 但深度键依赖相机指纹，涉及视图 1 的深度必须重算
        pair_keys = engine2._pair_cache_keys(feat_keys)
        self.assertIn(pair_keys[1]["depth_b"], recomputed)


if __name__ == "__main__":
    unittest.main()
