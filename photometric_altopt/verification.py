"""仅用于核对的工具：把求解/优化结果与真值清单比对。

本模块显式读取 ``truth_*`` 文件，求解与优化路径不允许导入它。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conflict import normal_angle_deg
from .generator import load_truth


def compare_with_truth(
    normals: np.ndarray, albedo: np.ndarray, obs_dir: Path
) -> dict[str, float]:
    truth = load_truth(Path(obs_dir))
    angle = normal_angle_deg(normals, truth["normals"])
    rho_diff = np.abs(albedo - truth["albedo"])
    return {
        "normal_angle_deg_mean": float(np.mean(angle)),
        "normal_angle_deg_max": float(np.max(angle)),
        "albedo_abs_mean": float(np.mean(rho_diff)),
        "albedo_abs_max": float(np.max(rho_diff)),
    }
