import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ancient_ocr.config import Config, write_default_config
from ancient_ocr.pipeline import Pipeline, bootstrap_demo_workspace
from ancient_ocr.sample_gen import generate_sample


def _prepare(budget_overrides=None, workers=4):
    root = Path(tempfile.mkdtemp())
    (root / "config").mkdir()
    path = root / "config" / "config.json"
    overrides = dict(workers and {"max_workers": workers} or {})
    if budget_overrides:
        overrides.update(budget_overrides)
    write_default_config(path, overrides)
    cfg = Config.load(path)
    build_root = root / "samples"
    build_root.mkdir()
    from ancient_ocr.sample_gen import build_default_corpus
    build_default_corpus(build_root, cfg.frac_bits)
    return root, cfg


def _comparable(state):
    """剥去时间戳等易变字段，保留验收要求逐字段一致的内容。"""
    keep = {
        "samples": state["samples"],
        "issues": state["issues"],
        "invalidation": state["invalidation"],
        "cache_state_digest": state["cache_state_digest"],
        "todos": state["scheduler"]["todos"],
        "failures": state["scheduler"]["failures"],
        "budget": state["scheduler"]["budget"],
    }
    return keep


class PipelineDeterminismTest(unittest.TestCase):
    def test_full_rebuild_matches_incremental_command_sequence(self):
        root, cfg = _prepare()
        pipeline = Pipeline(root / "samples", cfg, root / "snapshots")
        first = pipeline.execute()
        # 增量命令：关键点版本替换 -> 单图替换 -> 增删样本
        pipeline.apply_command({"op": "replace_keypoint_version",
                                "sample_id": "page_a", "version": "annot-v2"})
        pipeline.apply_command({"op": "replace_image",
                                "sample_id": "page_b", "seed": 8088})
        pipeline.apply_command({"op": "add_sample",
                                "sample_id": "page_d", "seed": 2002})
        incremental = pipeline.execute()

        # 从空白全量重建同一终态
        fresh_root = Path(tempfile.mkdtemp())
        (fresh_root / "config").mkdir()
        write_default_config(fresh_root / "config" / "config.json", cfg.to_dict())
        fresh_samples = fresh_root / "samples"
        fresh_samples.mkdir()
        generate_sample(fresh_samples, "page_a", 1000, cfg.frac_bits)
        generate_sample(fresh_samples, "page_b", 8088, cfg.frac_bits)
        generate_sample(fresh_samples, "page_c", 1074, cfg.frac_bits)
        generate_sample(fresh_samples, "page_d", 2002, cfg.frac_bits)
        # page_a 关键点切到 v2
        kp_path = fresh_samples / "page_a" / "keypoint_versions.json"
        kp = json.loads(kp_path.read_text(encoding="utf-8"))
        kp["active_version"] = "annot-v2"
        kp_path.write_text(json.dumps(kp, ensure_ascii=False, indent=2), encoding="utf-8")
        fresh_cfg = Config.load(fresh_root / "config" / "config.json")
        rebuilt = Pipeline(fresh_samples, fresh_cfg,
                           fresh_root / "snapshots").execute()

        self.assertEqual(
            _comparable(incremental)["cache_state_digest"],
            _comparable(rebuilt)["cache_state_digest"])
        self.assertEqual(
            _comparable(incremental)["samples"].keys(),
            _comparable(rebuilt)["samples"].keys())
        for sample_id in incremental["samples"]:
            self.assertEqual(
                incremental["samples"][sample_id]["artifacts"],
                rebuilt["samples"][sample_id]["artifacts"])
        shutil.rmtree(root)
        shutil.rmtree(fresh_root)

    def test_concurrency_level_does_not_change_results(self):
        root_a, cfg_a = _prepare(workers=1)
        root_b, cfg_b = _prepare(workers=8)
        state_a = Pipeline(root_a / "samples", cfg_a,
                           root_a / "snapshots").execute()
        state_b = Pipeline(root_b / "samples", cfg_b,
                           root_b / "snapshots").execute()
        self.assertEqual(state_a["cache_state_digest"], state_b["cache_state_digest"])
        self.assertEqual(
            [s["artifacts"]["fine"]["regressions"]
             for s in state_a["samples"].values()],
            [s["artifacts"]["fine"]["regressions"]
             for s in state_b["samples"].values()])
        shutil.rmtree(root_a)
        shutil.rmtree(root_b)


if __name__ == "__main__":
    unittest.main()
