"""业务要求 1：确定性观测集生成与逐字节复现。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from photometric_altopt.config import PipelineConfig
from photometric_altopt.generator import (
    MANIFEST_FILENAME,
    generate_observations,
    load_observations,
    load_truth,
)
from photometric_altopt.persistence import directory_manifest, read_json, sha256_file


def test_observation_shapes_and_dtype(tmp_path: Path, small_config: PipelineConfig):
    obs_dir = generate_observations(small_config.observation, tmp_path / "obs")
    data = load_observations(obs_dir)
    observations = data["observations"]
    size = small_config.observation.size
    assert observations.shape == (12, size, size)
    assert observations.dtype == np.dtype("<f4")
    assert np.all(np.isfinite(observations))
    assert np.all(observations >= 0.0)
    truth = load_truth(obs_dir)
    assert truth["normals"].shape == (size, size, 3)
    assert np.allclose(np.linalg.norm(truth["normals"], axis=-1), 1.0)


def test_generation_byte_identical(tmp_path: Path, small_config: PipelineConfig):
    dir_a = generate_observations(small_config.observation, tmp_path / "a")
    dir_b = generate_observations(small_config.observation, tmp_path / "b")
    manifest_a = directory_manifest(dir_a)
    manifest_b = directory_manifest(dir_b)
    assert manifest_a == manifest_b
    for rel in manifest_a:
        assert sha256_file(dir_a / rel) == sha256_file(dir_b / rel)


def test_manifest_records_numeric_conventions(tmp_path: Path, small_config: PipelineConfig):
    obs_dir = generate_observations(small_config.observation, tmp_path / "obs")
    manifest = read_json(obs_dir / MANIFEST_FILENAME)
    cfg_block = manifest["config"]
    assert cfg_block["storage_dtype"] == "float32"
    assert cfg_block["compute_dtype"] == "float64"
    assert "L2" in cfg_block["light_normalization"]
    # 单位化光源范数严格为 1
    for vec in cfg_block["light_directions_unit_world"]:
        assert np.isclose(np.linalg.norm(vec), 1.0)
    # 6 光源 × 2 相机 = 12 视点
    assert cfg_block["num_views"] == 12


def test_artifacts_are_committed(tmp_path: Path, small_config: PipelineConfig):
    obs_dir = generate_observations(small_config.observation, tmp_path / "obs")
    for item in obs_dir.iterdir():
        if item.is_file() and not item.name.endswith(".committed"):
            assert (obs_dir / (item.name + ".committed")).exists()
