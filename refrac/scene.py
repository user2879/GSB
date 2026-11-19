"""场景数据结构、确定性生成器与 JSON 序列化。

场景组成：
- 若干透明球体与轴对齐平板玻璃层（折射率 + RGB 吸收系数）；
- 一个不透明背景平面；
- 一个方向光；
- 固定针孔相机参数。
同一生成配置两次产出的 JSON 逐字节一致（键序固定、分隔符固定）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


from .rng import DeterministicRng

SCENE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Camera:
    eye: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_y_deg: float
    width: int
    height: int


@dataclass(frozen=True)
class DirectionalLight:
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    intensity: float


@dataclass(frozen=True)
class BackgroundPlane:
    body_id: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    base_color: tuple[float, float, float]
    checker_color: tuple[float, float, float]
    checker_size: float


@dataclass(frozen=True)
class TransparentBody:
    body_id: str
    kind: str  # "sphere" | "slab"
    ior: float
    absorption: tuple[float, float, float]
    # sphere:
    center: Optional[tuple[float, float, float]] = None
    radius: Optional[float] = None
    # slab:
    axis: Optional[int] = None
    thickness: Optional[float] = None
    half_extent: Optional[float] = None
    slab_normal: Optional[tuple[float, float, float]] = None


@dataclass(frozen=True)
class Scene:
    format_version: int
    seed: int
    camera: Camera
    light: DirectionalLight
    background: BackgroundPlane
    bodies: tuple[TransparentBody, ...]
    sample_count: int
    max_depth: int
    generator_tag: str
    max_tir_bounces: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "seed": self.seed,
            "generator_tag": self.generator_tag,
            "sample_count": self.sample_count,
            "max_depth": self.max_depth,
            "max_tir_bounces": self.max_tir_bounces,
            "camera": asdict(self.camera),
            "light": asdict(self.light),
            "background": asdict(self.background),
            "bodies": [asdict(body) for body in self.bodies],
        }

    def to_json(self) -> str:
        return dumps_canonical(self.to_dict())


def dumps_canonical(payload: Any) -> str:
    """规范 JSON：固定键序、固定分隔符、固定换行，保证逐字节一致。"""
    return json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def load_scene(path: str) -> Scene:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return scene_from_dict(data)


def scene_from_dict(data: dict[str, Any]) -> Scene:
    return Scene(
        format_version=int(data["format_version"]),
        seed=int(data["seed"]),
        generator_tag=str(data.get("generator_tag", "unknown")),
        sample_count=int(data["sample_count"]),
        max_depth=int(data["max_depth"]),
        max_tir_bounces=int(data.get("max_tir_bounces", 24)),
        camera=Camera(**data["camera"]),
        light=DirectionalLight(**data["light"]),
        background=BackgroundPlane(**data["background"]),
        bodies=tuple(TransparentBody(**body) for body in data["bodies"]),
    )


def generate_scene(seed: int = 20261119, *, width: int = 120, height: int = 90,
                   sample_count: int = 8, max_depth: int = 10,
                   max_tir_bounces: int = 24) -> Scene:
    """按固定种子程序化生成确定性场景。

    RNG 只驱动有限个取值（位置、折射率、吸收系数等），全部经确定性
    舍入为固定小数位写入 JSON，使输出与 numpy/平台无关。
    """
    rng = DeterministicRng(seed)

    def r3(low: float, high: float, digits: int = 5) -> tuple[float, float, float]:
        return tuple(round(rng.uniform(low, high), digits) for _ in range(3))

    bodies: list[TransparentBody] = []

    # 三个透明球体：折射率 1.33/1.5/1.75 附近，位置由 RNG 决定。
    sphere_specs = [
        ("sphere_water", 1.22, (0.08, 0.04, 0.02)),
        ("sphere_glass", 1.30, (0.03, 0.06, 0.09)),
        ("sphere_dense", 1.38, (0.10, 0.05, 0.05)),
    ]
    for index, (body_id, base_ior, base_absorb) in enumerate(sphere_specs):
        center = (
            round(-1.35 + index * 1.35 + rng.uniform(-0.08, 0.08), 5),
            round(rng.uniform(-0.35, 0.35), 5),
            round(rng.uniform(-0.6, 0.6), 5),
        )
        radius = round(rng.uniform(0.55, 0.72), 5)
        ior = round(base_ior + rng.uniform(-0.01, 0.01), 5)
        absorption = tuple(round(channel * rng.uniform(0.85, 1.15), 6) for channel in base_absorb)
        bodies.append(TransparentBody(
            body_id=body_id, kind="sphere", ior=ior, absorption=absorption,
            center=center, radius=radius,
        ))

    # 两块有限半宽的轴对齐平板玻璃层（z 轴厚度），分别偏置在左右两侧，
    # 只覆盖部分视口，便于核对覆盖像素与深度上限隔离的局部性。
    # 小尺寸有限平板：保持真实 TIR 弹跳的同时让光线在少量界面内自然逸出。
    slab_specs = [
        ("slab_back", 1.35, -1.0, -1.05, 0.15, 0.05),
        ("slab_front", 1.40, 0.9, 1.05, 0.15, 0.05),
    ]
    for body_id, base_ior, z_center, x_center, half_extent, thickness in slab_specs:
        ior = round(base_ior + rng.uniform(-0.005, 0.005), 5)
        absorption = r3(0.02, 0.06, digits=6)
        bodies.append(TransparentBody(
            body_id=body_id, kind="slab", ior=ior, absorption=absorption,
            center=(x_center, 0.0, z_center), axis=2, thickness=thickness,
            half_extent=half_extent,
        ))

    camera = Camera(
        eye=(0.0, 0.0, 4.2),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_y_deg=45.0,
        width=width,
        height=height,
    )
    light = DirectionalLight(
        direction=(0.4, -0.7, -0.58),
        color=(1.0, 0.97, 0.92),
        intensity=1.0,
    )
    background = BackgroundPlane(
        body_id="background_plane",
        point=(0.0, 0.0, -6.0),
        normal=(0.0, 0.0, 1.0),
        base_color=(0.72, 0.76, 0.82),
        checker_color=(0.28, 0.31, 0.36),
        checker_size=1.0,
    )
    return Scene(
        format_version=SCENE_FORMAT_VERSION,
        seed=seed,
        camera=camera,
        light=light,
        background=background,
        bodies=tuple(bodies),
        sample_count=sample_count,
        max_depth=max_depth,
        generator_tag="deterministic-scene-v1",
        max_tir_bounces=24,
    )
