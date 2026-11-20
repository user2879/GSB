"""加载层与配置合法性校验测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mvrwb.config import ReconstructionConfig
from mvrwb.errors import E_VIEW_INVALID, StageError
from mvrwb.util import read_json, write_json_atomic


def base_config() -> dict:
    return {
        "feature_threshold": 150.0,
        "match_min_inliers": 6,
        "triang_min_parallax_deg": 1.5,
        "depth_resolution": [60, 80],
        "depth_quant_levels": 32,
        "depth_smooth_passes": 2,
        "voxel_size": 0.35,
        "fixed_point_frac_bits": 12,
        "thread_workers": 4,
        "global_point_budget": 200,
        "per_view_quota": 40,
        "snapshot_commit_interval_views": 2,
        "view_pairs": [[1, 2]],
    }


@pytest.mark.parametrize("field,bad_value", [
    ("feature_threshold", 0.0),
    ("feature_threshold", -3.0),
    ("match_min_inliers", 0),
    ("triang_min_parallax_deg", 0.0),
    ("triang_min_parallax_deg", 95.0),
    ("voxel_size", 0.0),
    ("fixed_point_frac_bits", 40),
    ("thread_workers", 0),
    ("global_point_budget", 0),
    ("per_view_quota", -1),
    ("depth_quant_levels", 1),
    ("depth_smooth_passes", -1),
    ("snapshot_commit_interval_views", 0),
])
def test_invalid_config_rejected(field, bad_value):
    raw = base_config()
    raw[field] = bad_value
    with pytest.raises(StageError) as info:
        ReconstructionConfig.from_dict(raw)
    assert info.value.code == E_VIEW_INVALID


def test_bad_depth_resolution_rejected():
    raw = base_config()
    raw["depth_resolution"] = [1, 80]
    with pytest.raises(StageError) as info:
        ReconstructionConfig.from_dict(raw)
    assert info.value.code == E_VIEW_INVALID


def test_valid_config_accepted_and_fingerprinted():
    cfg = ReconstructionConfig.from_dict(base_config())
    assert len(cfg.fingerprint()) == 64
    # 参数分区指纹随具体字段变化
    fp_a = cfg.section_fingerprint("voxel_size")
    raw = base_config(); raw["voxel_size"] = 0.1
    fp_b = ReconstructionConfig.from_dict(raw).section_fingerprint("voxel_size")
    assert fp_a != fp_b


def test_singular_intrinsics_rejected(sample_dir, tmp_path):
    from mvrwb.camera import load_camera
    work = tmp_path / "views"
    work.mkdir()
    cam_path = str(work / "view_0001.cam.json")
    payload = read_json(os.path.join(sample_dir, "views", "view_0001.cam.json"))
    payload["intrinsics"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    write_json_atomic(cam_path, payload)
    with pytest.raises(StageError) as info:
        load_camera(cam_path, 1)
    assert info.value.code == E_VIEW_INVALID


def test_non_orthogonal_rotation_rejected(sample_dir, tmp_path):
    from mvrwb.camera import load_camera
    cam_path = str(tmp_path / "bad.cam.json")
    payload = read_json(os.path.join(sample_dir, "views", "view_0001.cam.json"))
    payload["rotation"] = np.eye(3).tolist()
    payload["rotation"][0][1] = 0.5  # 破坏正交性
    write_json_atomic(cam_path, payload)
    with pytest.raises(StageError) as info:
        load_camera(cam_path, 1)
    assert info.value.code == E_VIEW_INVALID


def test_png_header_validation(tmp_path):
    from mvrwb.camera import load_gray_png
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"NOT-A-PNG" + b"\x00" * 32)
    with pytest.raises(StageError) as info:
        load_gray_png(str(fake), 9)
    assert info.value.code == E_VIEW_INVALID
