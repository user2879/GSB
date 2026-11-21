"""测试公共夹具：临时目录生成确定性样本与默认配置。"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

from recon_workbench.config import RebuildConfig
from recon_workbench.generate_sample import generate
from recon_workbench.loader import LoadResult, load_sample


class SampleHarness:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="rwb_test_")
        self.sample_dir = os.path.join(self.root, "sample")
        self.work_dir = os.path.join(self.root, "work")
        self.viz_dir = os.path.join(self.root, "viz")
        self.report_path = os.path.join(self.root, "report.json")
        self.config_path = os.path.join(self.sample_dir, "rebuild_config.json")
        generate(self.sample_dir)
        self._write_config(RebuildConfig().to_dict())

    def _write_config(self, data: Dict[str, Any]) -> None:
        from recon_workbench.utils import atomic_write_json

        atomic_write_json(self.config_path, data)

    def config(self, **overrides: Any) -> RebuildConfig:
        data = RebuildConfig().to_dict()
        data.update(overrides)
        self._write_config(data)
        return RebuildConfig.from_dict(data)

    def load(self) -> LoadResult:
        return load_sample(self.sample_dir)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
