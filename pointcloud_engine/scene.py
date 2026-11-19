"""Scene description loading, validation and canonical versioning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .camera import Camera
from .errors import EngineError, is_finite_number
from .io_utils import hash_canonical
from .pointcloud import PointCloud, load_point_cloud, validate_point_cloud

BYTES_PER_SAMPLE = 8  # float32 grayscale + float32 depth per supersample cell
KNOWN_INJECTIONS = {
    "zero_radius",
    "non_finite_normal",
    "out_of_bounds_tile",
    "tile_runtime_error",
}


@dataclass
class DecayModel:
    kind: str = "exp"
    coefficient: float = 0.0

    def factor(self, depth: np.ndarray) -> np.ndarray:
        if self.kind == "exp":
            return np.exp(-self.coefficient * depth)
        if self.kind == "linear":
            return np.clip(1.0 - self.coefficient * depth, 0.0, 1.0)
        raise EngineError("E_DECAY", f"unknown decay kind {self.kind!r}")


@dataclass
class Scene:
    raw: dict[str, Any]
    base_dir: str
    point_cloud_path: str
    world_scale: float
    origin_offset: np.ndarray
    camera_params: dict[str, Any]
    width: int
    height: int
    supersample: int
    point_radius: float
    decay: DecayModel
    tile_width: int
    tile_height: int
    concurrency: int
    memory_budget_bytes: int
    state_dir: str
    output_dir: str
    injections: list[dict[str, Any]] = field(default_factory=list)
    tile_delay_seconds: float = 0.0
    near: float = 1e-6

    cloud: PointCloud | None = None
    camera: Camera | None = None

    @property
    def grid_width(self) -> int:
        return self.width * self.supersample

    @property
    def grid_height(self) -> int:
        return self.height * self.supersample

    @property
    def tiles_x(self) -> int:
        return self.grid_width // self.tile_width

    @property
    def tiles_y(self) -> int:
        return self.grid_height // self.tile_height

    @property
    def tile_count(self) -> int:
        return self.tiles_x * self.tiles_y

    def tile_bytes(self) -> int:
        return self.tile_width * self.tile_height * BYTES_PER_SAMPLE

    def world_points(self) -> np.ndarray:
        assert self.cloud is not None
        return self.cloud.positions * self.world_scale + self.origin_offset

    def version_id(self, cloud: PointCloud) -> str:
        # The version identity covers render *configuration* and the structural
        # skeleton of the cloud. Concrete point *values* (a local gray/position
        # edit) are intentionally excluded: they keep the version stable so that
        # recovery can perform per-block invalidation via the block digests.
        # Structural changes (point count, source file) or any camera/parameter
        # change alter the version and force a full archival re-render.
        # Use the scene-relative cloud reference (not an absolute path) so a
        # workspace copy/move keeps the same version identity.
        canonical = {
            "cloud_source": os.path.relpath(cloud.source_path, self.base_dir)
            if os.path.isabs(cloud.source_path) else cloud.source_path,
            "cloud_count": cloud.count,
            "cloud_shapes": [list(cloud.positions.shape), list(cloud.grays.shape),
                             list(cloud.normals.shape)],
            "world_scale": self.world_scale,
            "origin_offset": [float(v) for v in self.origin_offset],
            "camera": {
                "position": [float(v) for v in self.camera_params["position"]],
                "target": [float(v) for v in self.camera_params["target"]],
                "up": [float(v) for v in self.camera_params["up"]],
                "vfov_deg": float(self.camera_params["vfov_deg"]),
                "aspect": float(self.camera_params["aspect"]),
                "near": self.near,
            },
            "width": self.width,
            "height": self.height,
            "supersample": self.supersample,
            "point_radius": self.point_radius,
            "decay": {"kind": self.decay.kind, "coefficient": self.decay.coefficient},
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
        }
        return hash_canonical(canonical)


def _require_keys(raw: dict[str, Any], keys: list[str], context: str) -> None:
    for key in keys:
        if key not in raw:
            raise EngineError("E_SCHEMA", f"{context}: missing required field {key!r}", field=key)


def _as_positive_int(value: Any, name: str, code: str = "E_DIM") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EngineError(code, f"{name} must be a positive integer, got {value!r}", field=name)
    return value


def _as_vector3(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise EngineError("E_SCHEMA", f"{name} must be a 3-element array", field=name)
    out = []
    for component in value:
        if not is_finite_number(component):
            raise EngineError("E_FINITE", f"{name} contains a non-finite component", field=name)
        out.append(float(component))
    return np.asarray(out, dtype=np.float64)


def _resolve(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def load_scene(scene_path: str) -> Scene:
    if not os.path.isfile(scene_path):
        raise EngineError("E_SCENE", f"scene file not found: {scene_path}", path=scene_path)
    try:
        with open(scene_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineError("E_SCENE", f"scene JSON unreadable or malformed: {exc}")
    if not isinstance(raw, dict):
        raise EngineError("E_SCHEMA", "scene root must be a JSON object")

    base_dir = os.path.dirname(os.path.abspath(scene_path))
    _require_keys(
        raw,
        [
            "point_cloud", "world_scale", "origin_offset", "camera",
            "width", "height", "supersample", "point_radius", "depth_decay",
            "tile_width", "tile_height", "concurrency", "memory_budget_bytes",
            "state_dir", "output_dir",
        ],
        "scene",
    )

    injections = raw.get("fault_injections", [])
    if not isinstance(injections, list):
        raise EngineError("E_INJECT", "fault_injections must be a list")
    for directive in injections:
        if not isinstance(directive, dict) or "type" not in directive:
            raise EngineError("E_INJECT", "each injection must be an object with a 'type'")
        if directive["type"] not in KNOWN_INJECTIONS:
            raise EngineError("E_INJECT", f"unknown injection type {directive['type']!r}")

    if not is_finite_number(raw["world_scale"]) or float(raw["world_scale"]) <= 0.0:
        raise EngineError("E_SCALE", "world_scale must be a finite positive number")
    world_scale = float(raw["world_scale"])
    origin_offset = _as_vector3(raw["origin_offset"], "origin_offset")

    width = _as_positive_int(raw["width"], "width")
    height = _as_positive_int(raw["height"], "height")
    supersample = _as_positive_int(raw["supersample"], "supersample")
    tile_width = _as_positive_int(raw["tile_width"], "tile_width", code="E_TILE")
    tile_height = _as_positive_int(raw["tile_height"], "tile_height", code="E_TILE")
    concurrency = _as_positive_int(raw["concurrency"], "concurrency")
    memory_budget_bytes = _as_positive_int(raw["memory_budget_bytes"], "memory_budget_bytes")

    if not is_finite_number(raw["point_radius"]) or float(raw["point_radius"]) < 0.0:
        raise EngineError("E_RADIUS", "point_radius must be a finite non-negative number")
    point_radius = float(raw["point_radius"])

    decay_raw = raw["depth_decay"]
    if not isinstance(decay_raw, dict) or "kind" not in decay_raw or "coefficient" not in decay_raw:
        raise EngineError("E_DECAY", "depth_decay must be an object with 'kind' and 'coefficient'")
    if decay_raw["kind"] not in ("exp", "linear") or not is_finite_number(decay_raw["coefficient"]):
        raise EngineError("E_DECAY", "depth_decay has an unknown kind or non-finite coefficient")
    coefficient = float(decay_raw["coefficient"])
    if coefficient < 0.0:
        raise EngineError("E_DECAY", "depth_decay coefficient must be non-negative")
    decay = DecayModel(kind=str(decay_raw["kind"]), coefficient=coefficient)

    tile_delay_seconds = raw.get("tile_delay_seconds", 0.0)
    if not is_finite_number(tile_delay_seconds) or float(tile_delay_seconds) < 0.0:
        raise EngineError("E_SCHEMA", "tile_delay_seconds must be a finite non-negative number")
    tile_delay_seconds = float(tile_delay_seconds)

    cam_raw = raw["camera"]
    if not isinstance(cam_raw, dict):
        raise EngineError("E_SCHEMA", "camera must be an object")
    _require_keys(cam_raw, ["position", "target", "up", "vfov_deg", "aspect"], "camera")
    position = _as_vector3(cam_raw["position"], "camera.position")
    target = _as_vector3(cam_raw["target"], "camera.target")
    up = _as_vector3(cam_raw["up"], "camera.up")
    vfov = cam_raw["vfov_deg"]
    if not is_finite_number(vfov) or not (0.0 < float(vfov) < 180.0):
        raise EngineError("E_FOV", "vfov_deg must be inside the open interval (0, 180) degrees")
    aspect = cam_raw["aspect"]
    if not is_finite_number(aspect) or float(aspect) <= 0.0:
        raise EngineError("E_SCHEMA", "camera.aspect must be a finite positive number")
    near = float(raw.get("near", 1e-6))
    if not is_finite_number(near) or near <= 0.0:
        raise EngineError("E_SCHEMA", "near must be a finite positive number")
    camera_params = {
        "position": position,
        "target": target,
        "up": up,
        "vfov_deg": float(vfov),
        "aspect": float(aspect),
    }
    camera = Camera.build(position, target, up, float(vfov), float(aspect), near)

    # Parameter-level fault injections (data-driven, no source modification).
    for directive in injections:
        if directive["type"] == "out_of_bounds_tile":
            tile_width = int(directive.get("tile_width", tile_width + 1))
            tile_height = int(directive.get("tile_height", tile_height + 1))
        elif directive["type"] == "zero_radius":
            point_radius = 0.0

    grid_w = width * supersample
    grid_h = height * supersample
    if tile_width <= 0 or tile_height <= 0 or grid_w % tile_width != 0 or grid_h % tile_height != 0:
        raise EngineError(
            "E_TILE",
            "tile dimensions must be positive integers that divide the supersampled grid exactly",
            grid=[grid_w, grid_h],
            tile=[tile_width, tile_height],
        )
    if point_radius <= 0.0:
        raise EngineError("E_RADIUS", "point_radius of zero produces no sample footprint")

    state_dir = _resolve(base_dir, str(raw["state_dir"]))
    output_dir = _resolve(base_dir, str(raw["output_dir"]))
    point_cloud_path = _resolve(base_dir, str(raw["point_cloud"]))

    scene = Scene(
        raw=raw,
        base_dir=base_dir,
        point_cloud_path=point_cloud_path,
        world_scale=world_scale,
        origin_offset=origin_offset,
        camera_params=camera_params,
        width=width,
        height=height,
        supersample=supersample,
        point_radius=point_radius,
        decay=decay,
        tile_width=tile_width,
        tile_height=tile_height,
        concurrency=concurrency,
        memory_budget_bytes=memory_budget_bytes,
        state_dir=state_dir,
        output_dir=output_dir,
        injections=injections,
        tile_delay_seconds=tile_delay_seconds,
        near=near,
        camera=camera,
    )

    if scene.tile_bytes() > memory_budget_bytes:
        raise EngineError(
            "E_MEMORY",
            "memory budget cannot hold even one tile's sample buffer",
            tile_bytes=scene.tile_bytes(),
            budget=memory_budget_bytes,
        )

    cloud = load_point_cloud(point_cloud_path)
    for directive in injections:
        if directive["type"] == "non_finite_normal":
            idx = int(directive.get("point_index", 0))
            if not 0 <= idx < cloud.count:
                raise EngineError("E_INJECT", "non_finite_normal point_index out of range")
            cloud.normals[idx] = (float("nan"), 0.0, 0.0)
    validate_point_cloud(cloud)
    scene.cloud = cloud
    return scene
