"""前向模型：兰伯特漫反射 + Blinn-Phong 镜面反射的线性光渲染。

观测生成器与可微渲染阶段共用本模块，保证两条独立流程的求解语义严格对齐：

    I_v = max( rho * max(n.l, 0) + ks * max(n.h, 0)^sh , 0 )

其中所有向量均在相机局部坐标系，``rho`` 为标量反照率，``ks``/``sh``
为配置中固定的镜面增益与光泽度（不参与优化）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ObservationConfig
from .errors import E_ZERO_LIGHT, BusinessError


@dataclass
class ViewTable:
    """全部视点的预计算量（M = 相机数 × 光源数）。"""

    view_index: np.ndarray  # (M,) 全局视点编号
    camera_index: np.ndarray  # (M,)
    light_index: np.ndarray  # (M,)
    rotations: np.ndarray  # (M, 3, 3)
    lights: np.ndarray  # (M, 3) 相机系单位光源方向（指向光源）
    half_vectors: np.ndarray  # (M, 3) 相机系 Blinn 半程向量
    view_dirs: np.ndarray  # (M, 3) 相机系视线方向，恒为 +Z

    @property
    def size(self) -> int:
        return int(self.view_index.shape[0])


def normalize_light(raw: np.ndarray) -> np.ndarray:
    """按配置声明的 L2 口径单位化光源方向，零向量抛 ``E_ZERO_LIGHT``。"""
    vec = np.asarray(raw, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        zero_positions = np.argwhere((norm[..., 0] <= 0.0))
        raise BusinessError(
            E_ZERO_LIGHT,
            "检测到零向量光源，无法单位化",
            details={"light_index": zero_positions.reshape(-1).tolist()},
        )
    return vec / norm


def build_view_table(cfg: ObservationConfig, skip_views: set[int] | None = None) -> ViewTable:
    """枚举 相机×光源 组合，构造相机系光源与半程向量。

    Args:
        cfg: 观测配置。
        skip_views: 需要跳过的全局视点编号集合（管线层异常注入用）。
    """
    skip = set(skip_views or ())
    lights_world = normalize_light(np.asarray(cfg.light_directions, dtype=np.float64))
    view_dir_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    vi, ci, li, rots, lc, hv, vd = [], [], [], [], [], [], []
    global_id = 0
    for cam_id, pose in enumerate(cfg.camera_poses):
        rot = np.asarray(pose["rotation"], dtype=np.float64)
        for light_id in range(cfg.num_lights):
            if global_id in skip:
                global_id += 1
                continue
            light_cam = rot @ lights_world[light_id]
            half = light_cam + view_dir_local
            half = half / np.linalg.norm(half)
            vi.append(global_id)
            ci.append(cam_id)
            li.append(light_id)
            rots.append(rot)
            lc.append(light_cam)
            hv.append(half)
            vd.append(view_dir_local.copy())
            global_id += 1

    return ViewTable(
        view_index=np.asarray(vi, dtype=np.int64),
        camera_index=np.asarray(ci, dtype=np.int64),
        light_index=np.asarray(li, dtype=np.int64),
        rotations=np.stack(rots).astype(np.float64),
        lights=np.stack(lc).astype(np.float64),
        half_vectors=np.stack(hv).astype(np.float64),
        view_dirs=np.stack(vd).astype(np.float64),
    )


def render_view(
    normals_cam: np.ndarray,
    albedo: np.ndarray,
    light: np.ndarray,
    half: np.ndarray,
    specular_gain: float,
    shininess: float,
) -> np.ndarray:
    """单视点线性光渲染，返回 (H, W) 观测图。

    法向以相机系 (H, W, 3) 传入；本函数只含逐元素运算，可被向量化广播复用。
    """
    n_dot_l = np.clip(normals_cam @ light, 0.0, None)
    n_dot_h = np.clip(normals_cam @ half, 0.0, None)
    diffuse = albedo * n_dot_l
    specular = specular_gain * np.power(n_dot_h, shininess)
    return np.clip(diffuse + specular, 0.0, None)


def render_all(
    normals_world: np.ndarray,
    albedo: np.ndarray,
    views: ViewTable,
    cfg: ObservationConfig,
) -> np.ndarray:
    """全视点渲染，返回 (M, H, W) 线性光观测栈。"""
    h, w = normals_world.shape[:2]
    out = np.empty((views.size, h, w), dtype=np.float64)
    for m in range(views.size):
        normals_cam = normals_world @ views.rotations[m].T
        out[m] = render_view(
            normals_cam,
            albedo,
            views.lights[m],
            views.half_vectors[m],
            cfg.specular_gain,
            cfg.shininess,
        )
    return out


def render_and_grad(
    normals_world: np.ndarray,
    albedo: np.ndarray,
    views: ViewTable,
    cfg: ObservationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全视点渲染并返回观测对世界系法向/反照率的解析导数。

    返回:
        images: (M, H, W)
        d_image_d_normal_world: (M, H, W, 3)，逐像素世界系法向梯度（已含旋转链导）
        d_image_d_albedo: (M, H, W)
    """
    h, w = normals_world.shape[:2]
    m_total = views.size
    images = np.empty((m_total, h, w), dtype=np.float64)
    grad_n = np.empty((m_total, h, w, 3), dtype=np.float64)
    grad_rho = np.empty((m_total, h, w), dtype=np.float64)

    for m in range(m_total):
        rot = views.rotations[m]
        n_cam = normals_world @ rot.T
        light = views.lights[m]
        half = views.half_vectors[m]

        raw_nl = n_cam @ light
        raw_nh = n_cam @ half
        nl = np.clip(raw_nl, 0.0, None)
        nh = np.clip(raw_nh, 0.0, None)
        diffuse = albedo * nl
        specular = cfg.specular_gain * np.power(nh, cfg.shininess)
        raw = diffuse + specular
        images[m] = np.clip(raw, 0.0, None)

        active = raw > 0.0
        # dI/d n_cam
        d_nl = (raw_nl > 0.0)[..., None]
        d_nh = (raw_nh > 0.0)[..., None]
        d_spec = (
            cfg.specular_gain
            * cfg.shininess
            * np.power(nh, max(cfg.shininess - 1.0, 0.0))[..., None]
        )
        grad_n_cam = active[..., None] * (
            albedo[..., None] * light[None, None, :] * d_nl
            + d_spec * half[None, None, :] * d_nh
        )
        # 相机系 -> 世界系链导：n_cam = R n_world => d n_world = R^T d n_cam
        grad_n[m] = grad_n_cam @ rot
        grad_rho[m] = active * nl

    return images, grad_n, grad_rho
