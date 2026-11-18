"""测试公共夹具：临时目录中生成确定性样本与标准配置。"""

import json
import os
import shutil
import tempfile

from xmodal.samplegen import generate_dataset

DEFAULT_CONFIG = {
    "image_size": [64, 64],
    "block_size": [16, 16],
    "quant_levels": 16,
    "text_dim": 32,
    "orientation_bins": 4,
    "align_weights": {"mean": 0.35, "gradient": 0.25,
                      "orientation": 0.25, "text": 0.15},
    "retrieval_threshold": 0.05,
    "top_k": 3,
    "max_workers": 2,
    "snapshot_commit_interval": 2,
    "format_version": 2,
}

SPEC = {
    "spec_version": "test",
    "formats": {"v1": {"format_version": 1}, "v2": {"format_version": 2}},
}


class Workspace:
    def __init__(self, count=6):
        self.root = tempfile.mkdtemp(prefix="xmodal-test-")
        self.input_dir = os.path.join(self.root, "input")
        self.work_dir = os.path.join(self.root, "work")
        self.config_path = os.path.join(self.root, "config.json")
        self.spec_path = os.path.join(self.root, "spec.json")
        generate_dataset(self.input_dir, count, DEFAULT_CONFIG["image_size"])
        self.write_config(DEFAULT_CONFIG)
        with open(self.spec_path, "w", encoding="utf-8") as fh:
            json.dump(SPEC, fh)

    def write_config(self, cfg):
        with open(self.config_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, sort_keys=True)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)
