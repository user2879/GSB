"""确定性观测集生成器（业务要求 1）。

产物目录布局::

    obs_<seed>/
      manifest.json            配置口径、文件清单与全部内容哈希
      observations.npy         (M, H, W) float32 小端 线性光观测
      truth_normals.npy        (H, W, 3) float64 逐像素真值法向
      truth_albedo.npy         (H, W) float64 逐像素真值反照率
      height.npy               (H, W) float64 高度场
      view_index.npy           (M,) 全局视点编号

真值清单与求解输入同目录但文件名以 ``truth_`` 开头，求解器路径上不会读取它们，
仅由核对/测试代码显式使用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ObservationConfig
from .forward import build_view_table, render_all
from .geometry import build_surface
from .persistence import (
    atomic_commit,
    atomic_save_npy,
    atomic_write_json,
    sha256_array,
)

OBS_FILENAME = "observations.npy"
TRUTH_NORMALS_FILENAME = "truth_normals.npy"
TRUTH_ALBEDO_FILENAME = "truth_albedo.npy"
HEIGHT_FILENAME = "height.npy"
VIEW_INDEX_FILENAME = "view_index.npy"
MANIFEST_FILENAME = "manifest.json"


def generate_observations(cfg: ObservationConfig, out_root: Path) -> Path:
    """生成整套观测集并原子落盘，返回观测目录路径。"""
    out_root = Path(out_root)
    obs_dir = out_root / f"obs_{cfg.seed}_{cfg.size}"
    obs_dir.mkdir(parents=True, exist_ok=True)

    surface = build_surface(cfg.size, cfg.seed)
    views = build_view_table(cfg)
    images_f64 = render_all(surface.normals, surface.albedo, views, cfg)

    # 线性光浮点表示口径：计算 float64，落盘转 float32 小端
    images_f32 = images_f64.astype("<f4", copy=False)
    normals_f64 = surface.normals.astype(np.float64, copy=False)
    albedo_f64 = surface.albedo.astype(np.float64, copy=False)

    files: dict[str, str] = {}
    paths = {
        OBS_FILENAME: images_f32,
        TRUTH_NORMALS_FILENAME: normals_f64,
        TRUTH_ALBEDO_FILENAME: albedo_f64,
        HEIGHT_FILENAME: surface.height.astype(np.float64, copy=False),
        VIEW_INDEX_FILENAME: views.view_index.astype(np.int64, copy=False),
    }
    for name, array in paths.items():
        target = obs_dir / name
        files[name] = atomic_save_npy(target, array)
        atomic_commit(target, {"sha256": files[name], "dtype": str(array.dtype), "shape": list(array.shape)})

    manifest = {
        "schema_version": 1,
        "config": {
            "seed": cfg.seed,
            "size": cfg.size,
            "num_lights": cfg.num_lights,
            "num_cameras": len(cfg.camera_poses),
            "num_views": views.size,
            "light_directions_raw": cfg.light_directions,
            "light_normalization": cfg.light_normalization,
            "light_directions_unit_world": [
                (np.asarray(v, dtype=np.float64) / np.linalg.norm(v)).tolist()
                for v in cfg.light_directions
            ],
            "camera_poses": cfg.camera_poses,
            "specular_gain": cfg.specular_gain,
            "shininess": cfg.shininess,
            "storage_dtype": cfg.storage_dtype,
            "compute_dtype": cfg.compute_dtype,
            "linear_light_note": cfg.linear_light_note,
            "forward_model": "I=max(rho*max(n.l,0)+ks*max(n.h,0)^sh,0); Blinn-Phong, orthographic",
        },
        "files": files,
        "observation_array_hash": sha256_array(images_f32),
        "view_index": views.view_index.tolist(),
        "camera_index": views.camera_index.tolist(),
        "light_index": views.light_index.tolist(),
    }
    manifest_path = obs_dir / MANIFEST_FILENAME
    atomic_write_json(manifest_path, manifest)
    atomic_commit(manifest_path)
    return obs_dir


def load_observations(obs_dir: Path) -> dict[str, np.ndarray]:
    """读取观测集（不含 truth_* 文件，供求解器使用）。"""
    obs_dir = Path(obs_dir)
    return {
        "observations": np.load(obs_dir / OBS_FILENAME),
    }


def load_truth(obs_dir: Path) -> dict[str, np.ndarray]:
    """读取仅用于核对的真值清单（求解器路径禁止调用）。"""
    obs_dir = Path(obs_dir)
    return {
        "normals": np.load(obs_dir / TRUTH_NORMALS_FILENAME),
        "albedo": np.load(obs_dir / TRUTH_ALBEDO_FILENAME),
    }
