"""确定性本地样本生成（固定种子，可重复，无外部资源）。

生成内容
--------
1. ``scene.json``：介质盒（雾/玻璃板/有色水体 + 一个重叠归属测试盒）、
   点光源、相机与全部渲染/调度参数；
2. ``medium_grid.npz``：由固定种子 PCG 派生的介质标量栅格（消光/反照率/
   折射率），用于复核盒参数与栅格派生规则一致；
3. ``expected_transmittance.json``：沿若干确定性探针光线按 *解析* 指数
   积分得到的期望透射率，供验证脚本与渲染器步进结果比对。
"""
from __future__ import annotations

import json
import os

import numpy as np

from .geometry import build_interval_chain
from .scene import Camera, PointLight, RenderConfig, Scene, VolumeBox

SEED = 20261119
GRID_RESOLUTION = (24, 24, 24)


def _pcg_generator(seed: int) -> np.random.Generator:
    # PCG64 是 NumPy 稳定可用的确定性 BitGenerator，跨 1.x/2.x 行为一致。
    return np.random.Generator(np.random.PCG64(seed))


def build_deterministic_scene(state_dir: str, output_dir: str,
                              width: int = 32, height: int = 24,
                              supersample: int = 2) -> Scene:
    rng = _pcg_generator(SEED)

    # 场景包围盒 [0,10]^3；三个典型介质 + 一个重叠归属测试盒。
    volumes = (
        VolumeBox(0, "fog",
                  np.array([0.0, 0.0, 2.0]), np.array([10.0, 10.0, 5.5]),
                  sigma_t=0.18, albedo=0.85, ior=1.0003),
        VolumeBox(1, "glass_pane",
                  np.array([1.5, 1.0, 5.5]), np.array([8.5, 9.0, 6.0]),
                  sigma_t=0.05, albedo=0.10, ior=1.52),
        VolumeBox(2, "tinted_water",
                  np.array([2.0, 2.0, 6.0]), np.array([8.0, 8.0, 9.0]),
                  sigma_t=0.60, albedo=0.45, ior=1.333),
        # 与 fog 重叠：声明顺序靠后，重叠区应归 fog（index 更小）。
        VolumeBox(3, "overlap_probe",
                  np.array([4.0, 4.0, 3.0]), np.array([6.0, 6.0, 7.0]),
                  sigma_t=0.35, albedo=0.60, ior=1.20),
        # 高密度吸收板，置于所有介质前方且不与既有盒的 z 区间重叠，
        # 因而按声明/几何规则它能真实生效：sigma*厚度 = 9.5*1.2 ≈ 11.4，
        # 中心光线穿过时透射率降到 ~1e-5，可被截断阈值可靠拦截。
        VolumeBox(4, "dense_slab",
                  np.array([3.0, 3.0, 0.6]), np.array([7.0, 7.0, 1.8]),
                  sigma_t=9.5, albedo=0.25, ior=1.40),
    )

    bbox_min = np.zeros(3)
    bbox_max = np.full(3, 10.0)

    def _inside() -> np.ndarray:
        # 严格限制在所有介质合并后的包围盒 [0,10]x[0,10]x[2,9] 内。
        return np.array([
            rng.uniform(0.5, 9.5),
            rng.uniform(0.5, 9.5),
            rng.uniform(2.2, 8.8),
        ])

    lights = tuple(
        PointLight(i, _inside(), intensity=float(rng.uniform(40.0, 90.0)))
        for i in range(3)
    )

    camera = Camera(
        position=np.array([5.0, 5.0, -2.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        fov_v_deg=60.0,
        aspect=width / height,
    )

    grid_w = width * supersample
    grid_h = height * supersample
    # 分块取不超过网格的最大 2 的幂，保证任意默认分辨率下都能整除覆盖。
    def _largest_pow2(n: int, cap: int) -> int:
        t = 1
        while t * 2 <= min(n, cap):
            t *= 2
        return t
    tile_w = _largest_pow2(grid_w, 16)
    tile_h = _largest_pow2(grid_h, 12)
    config = RenderConfig(
        width=width, height=height, supersample=supersample,
        step_length=0.0625, max_steps=512,
        transmittance_cutoff=1e-3, fp32_mantissa_bits=23,
        transmittance_tolerance=1e-9,
        tile_width=tile_w, tile_height=tile_h,
        concurrency=4, memory_limit_bytes=64 * 1024 * 1024,
        state_dir=os.path.abspath(state_dir),
        output_dir=os.path.abspath(output_dir),
        step_jitter=(0.125, 0.25, 0.5),
        cutoff_jitter=(1e-5, 1e-4, 1e-3, 5e-3, 2e-2),
        light_radius=3.0,
        background=0.02,
    )
    return Scene(volumes=volumes, lights=lights, camera=camera, config=config,
                 bbox_min=bbox_min, bbox_max=bbox_max)


def scene_to_json(scene: Scene) -> dict:
    cfg = scene.config
    return {
        "volumes": [
            {"name": v.name, "min_corner": v.min_corner.tolist(),
             "max_corner": v.max_corner.tolist(), "sigma_t": v.sigma_t,
             "albedo": v.albedo, "ior": v.ior}
            for v in scene.volumes
        ],
        "lights": [
            {"position": l.position.tolist(), "intensity": l.intensity}
            for l in scene.lights
        ],
        "camera": {
            "position": scene.camera.position.tolist(),
            "forward": scene.camera.forward.tolist(),
            "fov_v_deg": scene.camera.fov_v_deg,
            "aspect": scene.camera.aspect,
        },
        "render": {
            "width": cfg.width, "height": cfg.height,
            "supersample": cfg.supersample, "step_length": cfg.step_length,
            "max_steps": cfg.max_steps,
            "transmittance_cutoff": cfg.transmittance_cutoff,
            "fp32_mantissa_bits": cfg.fp32_mantissa_bits,
            "transmittance_tolerance": cfg.transmittance_tolerance,
            "tile_width": cfg.tile_width, "tile_height": cfg.tile_height,
            "concurrency": cfg.concurrency,
            "memory_limit_bytes": cfg.memory_limit_bytes,
            "state_dir": cfg.state_dir, "output_dir": cfg.output_dir,
            "step_jitter": list(cfg.step_jitter),
            "cutoff_jitter": list(cfg.cutoff_jitter),
            "light_radius": cfg.light_radius, "background": cfg.background,
        },
        "sample": {"seed": SEED, "grid_resolution": list(GRID_RESOLUTION)},
    }


def build_medium_grid(scene: Scene) -> dict[str, np.ndarray]:
    """按固定种子在包围盒上派生三维介质标量栅格。

    栅格值在对应盒参数附近做确定性扰动，仅用于复核派生可重复性；
    渲染本身仍以盒的标量参数为准。
    """
    rng = _pcg_generator(SEED + 1)
    nx, ny, nz = GRID_RESOLUTION
    sigma_grid = np.zeros(GRID_RESOLUTION, dtype=np.float64)
    albedo_grid = np.zeros(GRID_RESOLUTION, dtype=np.float64)
    ior_grid = np.ones(GRID_RESOLUTION, dtype=np.float64)
    xs = np.linspace(scene.bbox_min[0], scene.bbox_max[0], nx)
    ys = np.linspace(scene.bbox_min[1], scene.bbox_max[1], ny)
    zs = np.linspace(scene.bbox_min[2], scene.bbox_max[2], nz)

    # 声明顺序赋值，后写盒不覆盖先写盒 -> 与渲染归属规则一致。
    for vol in scene.volumes:
        ix = np.flatnonzero((xs >= vol.min_corner[0]) & (xs < vol.max_corner[0]))
        iy = np.flatnonzero((ys >= vol.min_corner[1]) & (ys < vol.max_corner[1]))
        iz = np.flatnonzero((zs >= vol.min_corner[2]) & (zs < vol.max_corner[2]))
        if not (ix.size and iy.size and iz.size):
            continue
        sub_sigma = sigma_grid[np.ix_(ix, iy, iz)]
        mask = sub_sigma == 0.0
        noise = rng.normal(0.0, 0.01, size=sub_sigma.shape)
        sigma_vals = np.maximum(1e-4, vol.sigma_t * (1.0 + noise))
        sub_sigma[mask] = sigma_vals[mask]
        sigma_grid[np.ix_(ix, iy, iz)] = sub_sigma
        sub_albedo = albedo_grid[np.ix_(ix, iy, iz)]
        mask_a = sub_albedo == 0.0
        sub_albedo[mask_a] = vol.albedo
        albedo_grid[np.ix_(ix, iy, iz)] = sub_albedo
        sub_ior = ior_grid[np.ix_(ix, iy, iz)]
        mask_i = sub_ior == 1.0
        sub_ior[mask_i] = vol.ior
        ior_grid[np.ix_(ix, iy, iz)] = sub_ior

    return {"sigma_t": sigma_grid, "albedo": albedo_grid, "ior": ior_grid,
            "x": xs, "y": ys, "z": zs}


def expected_probe_transmittance(scene: Scene) -> dict:
    """沿固定探针光线解析积分期望透射率（无截断、无步长离散误差）。"""
    probes = [
        {"name": "center_axis",
         "origin": [5.0, 5.0, 0.0], "direction": [0.0, 0.0, 1.0]},
        {"name": "off_center",
         "origin": [3.0, 7.0, 0.0], "direction": [0.0, 0.0, 1.0]},
        {"name": "diagonal",
         "origin": [0.0, 0.0, 0.0],
         "direction": list(np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))},
        {"name": "miss",
         "origin": [-5.0, -5.0, 0.0], "direction": [0.0, 0.0, 1.0]},
    ]
    out = []
    for probe in probes:
        origin = np.array(probe["origin"], dtype=np.float64)
        direction = np.array(probe["direction"], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        chain = build_interval_chain(origin, direction, scene.volumes)
        tau = 0.0
        segments = []
        for seg in chain:
            tau += seg.volume.sigma_t * seg.length
            segments.append({
                "t_enter": seg.t_enter, "t_exit": seg.t_exit,
                "box_index": seg.volume.index, "box_name": seg.volume.name,
                "optical_depth": seg.volume.sigma_t * seg.length,
            })
        out.append({"name": probe["name"], "origin": probe["origin"],
                    "direction": direction.tolist(),
                    "optical_depth": tau,
                    "expected_transmittance": float(np.exp(-tau)),
                    "segments": segments})
    return {"seed": SEED, "probes": out}


def generate_sample(sample_dir: str, state_dir: str, output_dir: str,
                    width: int = 32, height: int = 24,
                    supersample: int = 2) -> dict[str, str]:
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    scene = build_deterministic_scene(state_dir, output_dir, width, height, supersample)

    scene_path = os.path.join(sample_dir, "scene.json")
    with open(scene_path, "w", encoding="utf-8") as fh:
        json.dump(scene_to_json(scene), fh, ensure_ascii=False, indent=2)

    grid = build_medium_grid(scene)
    grid_path = os.path.join(sample_dir, "medium_grid.npz")
    np.savez_compressed(grid_path, **grid)

    expected = expected_probe_transmittance(scene)
    expected_path = os.path.join(sample_dir, "expected_transmittance.json")
    with open(expected_path, "w", encoding="utf-8") as fh:
        json.dump(expected, fh, ensure_ascii=False, indent=2)

    return {"scene": os.path.abspath(scene_path),
            "grid": os.path.abspath(grid_path),
            "expected": os.path.abspath(expected_path)}
