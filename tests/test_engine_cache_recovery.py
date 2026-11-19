import json
import os
import tempfile
import unittest

import numpy as np

from voice_repair_workbench import audio_io, synthesis
from voice_repair_workbench.cache import compute_invalidation_closure
from voice_repair_workbench.engine import RepairEngine
from voice_repair_workbench.errors import (
    E_CACHE_HASH,
    E_SNAPSHOT_STALE,
    RepairError,
)
from voice_repair_workbench.metrics import compare_conclusions
from voice_repair_workbench.persistence import Workspace
from voice_repair_workbench.pipeline import STAGE_ORDER
from voice_repair_workbench.project import (
    default_v2_project,
    legacy_v1_project,
)


class TestEngineCacheRecovery(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.material = os.path.join(self.work, "materials")
        self.paths = synthesis.generate_material(self.material)
        with open(self.paths["segments"], encoding="utf-8") as fh:
            self.segments = json.load(fh)["segments"]
        self.v1 = legacy_v1_project()
        self.v2 = default_v2_project()

    def _engine(self, name):
        return RepairEngine(Workspace(os.path.join(self.work, name)))

    def test_closure_computation(self):
        self.assertEqual(compute_invalidation_closure(["gate"]),
                         ["gate", "normalize", "limit"])
        self.assertEqual(compute_invalidation_closure(["denoise"]), list(STAGE_ORDER))
        self.assertEqual(compute_invalidation_closure([]), [])

    def test_deterministic_pcm_and_full_reuse_second_run(self):
        eng = self._engine("ws_det")
        snap1 = eng.execute(self.paths["mono16"], self.v1, self.segments, expected_channels=1)
        snap2 = eng.execute(self.paths["mono16"], self.v1, self.segments, expected_channels=1)
        self.assertEqual(snap1["output"]["pcm16_sha256"], snap2["output"]["pcm16_sha256"])
        self.assertEqual(snap2["recomputed_closure"], [])
        self.assertTrue(all(t["action"] == "reused" for t in snap2["execution_trace"]))

    def test_recomputed_set_equals_closure(self):
        eng = self._engine("ws_closure")
        eng.execute(self.paths["mono16"], self.v2, self.segments, expected_channels=1)
        changed = json.loads(json.dumps(self.v2))
        changed["gate"]["threshold_db"] = -30.0
        snap = eng.execute(self.paths["mono16"], changed, self.segments, expected_channels=1)
        self.assertEqual(snap["recomputed_closure"],
                         ["gate", "normalize", "limit"])
        actions = {t["stage"]: t["action"] for t in snap["execution_trace"]}
        self.assertEqual(actions["denoise"], "reused")
        self.assertEqual(set(k for k, v in actions.items() if v == "recomputed"),
                         {"gate", "normalize", "limit"})

    def test_cross_format_conclusions_within_tolerance(self):
        reports = {}
        for label, key, ch in (("mono16", "mono16", 1), ("stereo24", "stereo24", 2),
                               ("lossy", "lossy", 1)):
            if key == "lossy":
                from voice_repair_workbench import lossy_codec
                parsed = audio_io.parse_audio(self.paths["mono16"], expected_channels=1)
                path = lossy_codec.encode_file(
                    parsed.mono_analysis, 48000,
                    os.path.join(self.material, "x.lrx"))
            else:
                path = self.paths[key]
            eng = self._engine(f"ws_{label}")
            snap = eng.execute(path, self.v1, self.segments, expected_channels=ch)
            reports[label] = snap["metrics"]
        self.assertTrue(compare_conclusions(reports["mono16"], reports["stereo24"])["consistent"])
        self.assertTrue(compare_conclusions(reports["mono16"], reports["lossy"])["consistent"])

    def test_channel_order_only_affects_dependent_stages(self):
        # 构造非对称双声道素材：两声道电平不同，使两种通道顺序产生可观测差异
        from voice_repair_workbench import pipeline
        parsed = audio_io.parse_audio(self.paths["stereo24"], expected_channels=2)
        raw = parsed.raw_float.copy()
        raw[:, 1] *= 0.5
        order_a = json.loads(json.dumps(self.v2))
        order_b = json.loads(json.dumps(self.v2))
        order_b["channel_order"] = "per_channel"
        res_a = pipeline.run_full_pipeline(raw, 2, order_a, 48000)
        res_b = pipeline.run_full_pipeline(raw, 2, order_b, 48000)
        # denoise/gate 逐声道结果与通道顺序无关 -> 哈希不变
        self.assertTrue(np.array_equal(res_a["denoise"], res_b["denoise"]))
        self.assertTrue(np.array_equal(res_a["gate"], res_b["gate"]))
        # normalize 依赖通道顺序 -> 结果改变，limit 跟随改变
        self.assertFalse(np.array_equal(res_a["normalize"], res_b["normalize"]))
        self.assertFalse(np.array_equal(res_a["limit"], res_b["limit"]))

    def test_tampered_cache_raises_cache_hash(self):
        eng = self._engine("ws_tamper")
        snap = eng.execute(self.paths["mono16"], self.v2, self.segments, expected_channels=1)
        target = os.path.join(eng.ws.cache_dir, snap["stages"][0]["key"] + ".npy")
        data = np.load(target).copy()
        data[:] = 0.0
        np.save(target, data)
        with self.assertRaises(RepairError) as ctx:
            eng.ws.verify_stage_hashes(snap)
        self.assertEqual(ctx.exception.code, E_CACHE_HASH)

    def test_stale_snapshot_rejected(self):
        eng = self._engine("ws_stale")
        eng.execute(self.paths["mono16"], self.v2, self.segments, expected_channels=1)
        other = json.loads(json.dumps(self.v2))
        other["gate"]["threshold_db"] = -60.0
        with self.assertRaises(RepairError) as ctx:
            eng.restore(other)
        self.assertEqual(ctx.exception.code, E_SNAPSHOT_STALE)

    def test_force_termination_recovery_matches_last_commit(self):
        eng = self._engine("ws_crash")
        snap = eng.execute(self.paths["mono16"], self.v2, self.segments, expected_channels=1)
        import soundfile as sf
        pcm_before, _ = sf.read(os.path.join(eng.ws.root, "repaired.wav"), dtype="int16")
        # 模拟强制终止：遗留未提交 .tmp，且无关闭标记/退出报告
        with open(os.path.join(eng.ws.cache_dir, "denoise_ORPHAN.tmp"), "wb") as fh:
            fh.write(b"uncommitted")
        self.assertFalse(os.path.exists(eng.ws.closed_flag))
        restored = eng.restore(self.v2)
        self.assertIn("denoise_ORPHAN.tmp", restored["recovery_audit"]["uncommitted_tmp_reclaimed"])
        self.assertFalse(restored["recovery_audit"]["exit_report_present"])
        pcm_after, _ = sf.read(os.path.join(eng.ws.root, "repaired.wav"), dtype="int16")
        self.assertTrue(np.array_equal(pcm_before, pcm_after))
        self.assertEqual(restored["snapshot"]["metrics"], snap["metrics"])

    def test_graceful_shutdown_writes_report_and_marker(self):
        eng = self._engine("ws_grace")
        snap = eng.execute(self.paths["mono16"], self.v2, self.segments, expected_channels=1)
        report = eng.graceful_shutdown(snap)
        self.assertEqual(report["exit_kind"], "graceful")
        self.assertTrue(os.path.exists(eng.ws.closed_flag))
        self.assertTrue(os.path.exists(eng.ws.exit_report_path))


if __name__ == "__main__":
    unittest.main()
