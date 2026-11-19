"""Deterministic procedural scene generation and serialization."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from .models import (
    Camera,
    Ground,
    LightCone,
    Occluder,
    Scene,
    TransparentSurface,
    Vec3,
    VolumeSettings,
    q,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 20261119
DEFAULT_WIDTH = 128
DEFAULT_HEIGHT = 80


def _normalize(vector: Vec3) -> Vec3:
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (q(x / length), q(y / length), q(z / length))


def canonical_json_bytes(payload: Any) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def scene_bytes(scene: Scene) -> bytes:
    return canonical_json_bytes(scene.to_dict())


def scene_sha256(scene: Scene) -> str:
    return hashlib.sha256(scene_bytes(scene)).hexdigest()


def generate_scene(
    seed: int = DEFAULT_SEED,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Scene:
    """Generate the complete scene from a stable, local random seed."""

    if width < 16 or height < 16:
        raise ValueError("width and height must both be at least 16")
    rng = random.Random(int(seed))

    camera = Camera(
        position=(0.0, 2.15, 7.2),
        target=(0.0, 1.05, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_y_deg=50.0,
        width=int(width),
        height=int(height),
    )

    surfaces: list[TransparentSurface] = []
    lens_specs = [
        ("lens_left", (-1.20, 1.72, 2.65), (0.10, 0.96, 0.99), 0.70, -0.20, 1.46),
        ("lens_right", (1.12, 1.48, 1.80), (-0.12, 0.93, 0.99), 0.82, 0.24, 1.52),
    ]
    for index, (name, center, normal, radius, curvature, ior) in enumerate(lens_specs):
        jitter_r = rng.uniform(-0.035, 0.035)
        jitter_c = rng.uniform(-0.025, 0.025)
        surfaces.append(
            TransparentSurface(
                surface_id=name,
                kind="curved_disk",
                center=tuple(q(v + rng.uniform(-0.03, 0.03)) for v in center),  # type: ignore[arg-type]
                normal=_normalize(normal),
                radius=q(radius + jitter_r),
                half_width=0.0,
                half_height=0.0,
                half_depth=0.0,
                curvature=q(curvature + jitter_c),
                ior=q(ior + rng.uniform(-0.015, 0.015)),
                transmission=q(0.90 - index * 0.035),
                tint=(q(0.84 + index * 0.03), q(0.94), q(1.0)),
                sort_index=index,
            )
        )

    slab_specs = [
        ("glass_front", (-0.34, 1.16, 3.72), (0.0, 0.12, 0.99), 1.05, 0.74, 0.075, 1.49),
        ("glass_back", (1.74, 0.94, 0.62), (-0.15, 0.08, 0.99), 0.72, 0.64, 0.065, 1.57),
    ]
    for index, (name, center, normal, hw, hh, hd, ior) in enumerate(slab_specs):
        surfaces.append(
            TransparentSurface(
                surface_id=name,
                kind="glass_slab",
                center=tuple(q(v + rng.uniform(-0.025, 0.025)) for v in center),  # type: ignore[arg-type]
                normal=_normalize(normal),
                radius=0.0,
                half_width=q(hw + rng.uniform(-0.02, 0.02)),
                half_height=q(hh + rng.uniform(-0.02, 0.02)),
                half_depth=q(hd),
                curvature=0.0,
                ior=q(ior + rng.uniform(-0.012, 0.012)),
                transmission=q(0.86 - index * 0.02),
                tint=(q(0.88), q(0.95), q(1.0)),
                sort_index=len(lens_specs) + index,
            )
        )

    surfaces = sorted(surfaces, key=lambda surface: surface.sort_index)

    occluder_specs = [
        ("block_left", (-1.92, 0.50, 0.42), (0.48, 0.50, 0.52), (0.52, 0.34, 0.27)),
        ("block_right", (1.62, 0.38, -0.64), (0.60, 0.38, 0.54), (0.29, 0.36, 0.48)),
        ("block_far", (0.24, 0.72, -2.55), (0.92, 0.72, 0.42), (0.38, 0.42, 0.32)),
    ]
    occluders = []
    for name, center, extents, albedo in occluder_specs:
        jitter = tuple(rng.uniform(-0.025, 0.025) for _ in range(3))
        occluders.append(
            Occluder(
                occluder_id=name,
                center=tuple(q(center[i] + jitter[i]) for i in range(3)),  # type: ignore[arg-type]
                half_extents=tuple(q(extents[i] + rng.uniform(-0.015, 0.015)) for i in range(3)),  # type: ignore[arg-type]
                albedo=albedo,
            )
        )

    light_cone = LightCone(
        cone_id="key_volume_cone",
        apex=(-2.55, 5.25, 2.35),
        direction=_normalize((0.16, -1.0, -0.09)),
        length=7.2,
        radius_start=0.22,
        radius_end=3.25,
        scatter=0.56,
        absorption=0.055,
        color=(1.0, 0.74, 0.42),
        shadow_samples=4,
        jitter_seed=int(seed) ^ 0x5EED5EED,
    )

    return Scene(
        schema_version=SCHEMA_VERSION,
        seed=int(seed),
        camera=camera,
        surfaces=tuple(surfaces),
        light_cone=light_cone,
        occluders=tuple(occluders),
        ground=Ground(
            plane_y=0.0,
            albedo_a=(0.22, 0.26, 0.24),
            albedo_b=(0.31, 0.33, 0.29),
            checker_scale=0.82,
        ),
        volume=VolumeSettings(
            base_step=0.42,
            max_steps=14,
            near_plane=0.10,
            near_clip=0.75,
            far_clip=12.0,
            phase_isotropic=1.0 / (4.0 * math.pi),
        ),
        metadata={
            "generator": "volumetric_editor.scene",
            "material_model": "thin-refractive-layers",
            "sorting": "back-to-front depth,camera-distance,surface-sort-index",
        },
    )


def save_scene(scene: Scene, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = scene_bytes(scene)
    target.write_bytes(payload + b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_scene(path: str | Path) -> Scene:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_data = payload["camera"]
    camera = Camera(
        position=tuple(camera_data["position"]),
        target=tuple(camera_data["target"]),
        up=tuple(camera_data["up"]),
        fov_y_deg=float(camera_data["fov_y_deg"]),
        width=int(camera_data["width"]),
        height=int(camera_data["height"]),
    )
    surfaces = []
    for item in payload["surfaces"]:
        surfaces.append(
            TransparentSurface(
                surface_id=item["surface_id"],
                kind=item["kind"],
                center=tuple(item["center"]),
                normal=tuple(item["normal"]),
                radius=float(item["radius"]),
                half_width=float(item["half_width"]),
                half_height=float(item["half_height"]),
                half_depth=float(item["half_depth"]),
                curvature=float(item["curvature"]),
                ior=float(item["ior"]),
                transmission=float(item["transmission"]),
                tint=tuple(item["tint"]),
                sort_index=int(item["sort_index"]),
            )
        )
    cone_data = payload["light_cone"]
    cone = LightCone(
        cone_id=cone_data["cone_id"],
        apex=tuple(cone_data["apex"]),
        direction=tuple(cone_data["direction"]),
        length=float(cone_data["length"]),
        radius_start=float(cone_data["radius_start"]),
        radius_end=float(cone_data["radius_end"]),
        scatter=float(cone_data["scatter"]),
        absorption=float(cone_data["absorption"]),
        color=tuple(cone_data["color"]),
        shadow_samples=int(cone_data["shadow_samples"]),
        jitter_seed=int(cone_data["jitter_seed"]),
    )
    occluders = tuple(
        Occluder(
            occluder_id=item["occluder_id"],
            center=tuple(item["center"]),
            half_extents=tuple(item["half_extents"]),
            albedo=tuple(item["albedo"]),
        )
        for item in payload["occluders"]
    )
    ground_data = payload["ground"]
    ground = Ground(
        plane_y=float(ground_data["plane_y"]),
        albedo_a=tuple(ground_data["albedo_a"]),
        albedo_b=tuple(ground_data["albedo_b"]),
        checker_scale=float(ground_data["checker_scale"]),
    )
    volume_data = payload["volume"]
    volume = VolumeSettings(
        base_step=float(volume_data["base_step"]),
        max_steps=int(volume_data["max_steps"]),
        near_plane=float(volume_data["near_plane"]),
        near_clip=float(volume_data["near_clip"]),
        far_clip=float(volume_data["far_clip"]),
        phase_isotropic=float(volume_data["phase_isotropic"]),
    )
    return Scene(
        schema_version=int(payload["schema_version"]),
        seed=int(payload["seed"]),
        camera=camera,
        surfaces=tuple(surfaces),
        light_cone=cone,
        occluders=occluders,
        ground=ground,
        volume=volume,
        metadata=dict(payload.get("metadata", {})),
    )
