"""pytest 公共夹具：小尺寸确定性配置与临时工作区（测试提速，不影响 256 主示例）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from photometric_altopt.config import PipelineConfig
from photometric_altopt.forward import build_view_table
from photometric_altopt.generator import generate_observations, load_observations


@pytest.fixture
def small_config() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.observation.size = 24
    cfg.opt.max_iterations = 4
    return cfg


@pytest.fixture
def observation_bundle(tmp_path: Path, small_config: PipelineConfig):
    obs_dir = generate_observations(small_config.observation, tmp_path / "observations")
    observations = load_observations(obs_dir)["observations"]
    views = build_view_table(small_config.observation)
    return {
        "dir": obs_dir,
        "observations": observations,
        "views": views,
        "config": small_config,
    }
