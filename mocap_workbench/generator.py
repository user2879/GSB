"""Deterministic procedural multi-camera motion-capture scene generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constraints import ConstraintSet, default_constraints
from .errors import BusinessError, ErrorCode, ensure
from .serialization import (
    array_digest,
    canonical_json_bytes,
    read_json,
    read_npy,
    sha256_bytes,
    write_json,
    write_npy,
)
from .skeleton import (
    JOINT_NAMES,
    PARENTS,
    Skeleton,
    default_source_skeleton,
    default_target_skeleton,
    forward_kinematics,
    rotation_between,
    rotations_about_axes,
)


@dataclass(frozen=True)
class GenerationConfig:
    frames: int
    seed: int = 20261119
    fps: int = 60
    camera_count: int = 3

    def __post_init__(self) -> None:
        if int(self.frames) <= 0:
            raise BusinessError(ErrorCode.E_INPUT, "frames must be positive")
        if int(self.fps) != 60:
            raise BusinessError(ErrorCode.E_INPUT, "the workbench fixes frame rate at 60 fps")
        if int(self.camera_count) != 3:
            raise BusinessError(ErrorCode.E_INPUT, "this scene format requires exactly three cameras")

    def to_manifest(self) -> dict[str, Any]:
        return {"frames": self.frames, "seed": self.seed, "fps": self.fps, "camera_count": self.camera_count}


@dataclass(frozen=True)
class Camera:
    name: str
    position: tuple[float, float, float]
    world_to_camera: np.ndarray
    focal_px: float
    principal_px: tuple[float, float]
    image_size: tuple[int, int]
    near: float = 0.1
    far: float = 20.0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": list(self.position),
            "world_to_camera": self.world_to_camera,
            "focal_px": self.focal_px,
            "principal_px": list(self.principal_px),
            "image_size": list(self.image_size),
            "near": self.near,
            "far": self.far,
        }

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "Camera":
        return cls(
            name=str(payload["name"]),
            position=tuple(float(x) for x in payload["position"]),
            world_to_camera=np.asarray(payload["world_to_camera"], dtype=np.float64),
            focal_px=float(payload["focal_px"]),
            principal_px=tuple(float(x) for x in payload["principal_px"]),
            image_size=tuple(int(x) for x in payload["image_size"]),
            near=float(payload["near"]),
            far=float(payload["far"]),
        )


@dataclass
class Scene:
    config: GenerationConfig
    source: Skeleton
    target: Skeleton
    cameras: tuple[Camera, ...]
    constraints: ConstraintSet
    observations: np.ndarray
    visibility: np.ndarray
    sigma: np.ndarray
    truth: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.observations.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.observations.shape[1])

    def visibility_bits(self) -> np.ndarray:
        bits = np.zeros(self.visibility.shape[:2], dtype=np.uint8)
        for camera in range(self.visibility.shape[2]):
            bits |= self.visibility[:, :, camera].astype(np.uint8) << camera
        return bits

    def array_manifest(self) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "visibility": self.visibility,
            "sigma": self.sigma,
            "truth": self.truth,
        }

    def digest(self) -> str:
        parts = [canonical_json_bytes(self.structure_manifest())]
        for name in sorted(self.array_manifest()):
            parts.append(array_digest(self.array_manifest()[name]).encode("ascii"))
        return sha256_bytes(b"".join(parts))

    def structure_manifest(self) -> dict[str, Any]:
        return {
            "format": "mocap-workbench-scene-v1",
            "generation": self.config.to_manifest(),
            "source_skeleton": self.source.to_manifest(),
            "target_skeleton": self.target.to_manifest(),
            "cameras": [camera.to_manifest() for camera in self.cameras],
            "constraints": self.constraints.to_manifest(),
            "fps": 60,
            "truth_source": {
                "kind": "procedural_fixed_seed",
                "seed": self.config.seed,
                "fps": 60,
                "derived_from": "generator_only",
                "not_inferred_from_observations": True,
            },
        }


def _look_at_camera(name: str, position: tuple[float, float, float], target=(0.0, 0.0, 1.05)) -> Camera:
    pos = np.asarray(position, dtype=np.float64)
    look = np.asarray(target, dtype=np.float64) - pos
    forward = look / np.linalg.norm(look)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world_to_camera = np.stack((right, up, forward), axis=0)
    return Camera(
        name=name,
        position=tuple(float(x) for x in position),
        world_to_camera=world_to_camera,
        focal_px=1180.0,
        principal_px=(640.0, 360.0),
        image_size=(1280, 720),
    )


def default_cameras() -> tuple[Camera, Camera, Camera]:
    return (
        _look_at_camera("cam0", (-2.65, -2.25, 2.05)),
        _look_at_camera("cam1", (2.75, -1.85, 1.75)),
        _look_at_camera("cam2", (0.20, 3.00, 2.20)),
    )


def _rodrigues(axis: tuple[float, float, float], angles: np.ndarray) -> np.ndarray:
    return rotations_about_axes(np.asarray(axis, dtype=np.float64), angles)


def _procedural_local_rotations(frames: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    lookup = {name: i for i, name in enumerate(JOINT_NAMES)}
    rots = np.tile(np.eye(3), (frames, len(JOINT_NAMES), 1, 1))
    t = np.arange(frames, dtype=np.float64) / 60.0
    omega = 2.0 * np.pi * 0.80
    phase_jitter = ((seed % 997) / 997.0) * 0.15

    def compose(joint: str, axis: tuple[float, float, float], angles: np.ndarray) -> None:
        rots[:, lookup[joint]] = rots[:, lookup[joint]] @ _rodrigues(axis, angles)

    compose("spine", (1.0, 0.0, 0.0), 0.045 * np.sin(omega * t + 0.2))
    compose("chest", (1.0, 0.0, 0.0), 0.075 * np.sin(omega * t + 0.55 + phase_jitter))
    compose("neck", (1.0, 0.0, 0.0), 0.035 * np.sin(omega * t + 1.0))
    compose("head", (1.0, 0.0, 0.0), 0.025 * np.sin(omega * t + 1.3))

    arm_swing = 0.46 * np.sin(omega * t)
    shoulder_lift = 0.13 * np.sin(2.0 * omega * t + 0.4)
    compose("l_shoulder", (0.0, 0.0, 1.0), arm_swing)
    compose("l_shoulder", (0.0, 1.0, 0.0), shoulder_lift)
    compose("r_shoulder", (0.0, 0.0, 1.0), -arm_swing)
    compose("r_shoulder", (0.0, 1.0, 0.0), -shoulder_lift)

    elbow_l = 0.18 + 0.62 * (0.5 + 0.5 * np.sin(1.35 * omega * t + 0.25))
    elbow_r = 0.18 + 0.62 * (0.5 + 0.5 * np.sin(1.35 * omega * t + 0.40))
    compose("l_elbow", (0.0, 0.0, 1.0), elbow_l)
    compose("r_elbow", (0.0, 0.0, 1.0), -elbow_r)

    hip_swing = 0.42 * np.sin(omega * t + 0.1)
    compose("l_hip", (1.0, 0.0, 0.0), hip_swing)
    compose("r_hip", (1.0, 0.0, 0.0), -hip_swing)
    knee_l = 0.16 + 0.64 * (0.5 + 0.5 * np.sin(omega * t + 1.1))
    knee_r = 0.16 + 0.64 * (0.5 + 0.5 * np.sin(omega * t + 1.35))
    compose("l_knee", (1.0, 0.0, 0.0), -knee_l)
    compose("r_knee", (1.0, 0.0, 0.0), -knee_r)

    root_positions = np.column_stack(
        (
            0.065 * np.sin(0.55 * omega * t),
            0.115 * np.sin(omega * t),
            1.00 + 0.026 * np.sin(2.0 * omega * t),
        )
    )
    return rots, root_positions


def _occlusion_score(frames: int, joints: int, cameras: int, seed: int) -> np.ndarray:
    frame = np.arange(frames, dtype=np.uint64)[:, None, None]
    joint = np.arange(joints, dtype=np.uint64)[None, :, None]
    camera = np.arange(cameras, dtype=np.uint64)[None, None, :]
    value = (
        frame * np.uint64(73856093)
        ^ joint * np.uint64(19349663)
        ^ camera * np.uint64(83492791)
        ^ np.uint64(seed & 0xFFFFFFFF) * np.uint64(2654435761)
    )
    value ^= value >> np.uint64(13)
    value *= np.uint64(1274126177)
    value ^= value >> np.uint64(16)
    return (value & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)


def _project_visibility(
    truth: np.ndarray, cameras: tuple[Camera, ...], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    frames, joints, _ = truth.shape
    cameras_n = len(cameras)
    frustum = np.zeros((frames, joints, cameras_n), dtype=bool)
    projected = np.full((frames, joints, cameras_n, 3), np.nan, dtype=np.float64)
    lookup = {name: i for i, name in enumerate(JOINT_NAMES)}
    for camera_index, camera in enumerate(cameras):
        local = np.einsum("ij,fkj->fki", camera.world_to_camera, truth - np.asarray(camera.position)[None, None, :])
        z = local[:, :, 2]
        u = camera.focal_px * local[:, :, 0] / np.maximum(z, 1e-12) + camera.principal_px[0]
        v = camera.focal_px * local[:, :, 1] / np.maximum(z, 1e-12) + camera.principal_px[1]
        margin = 12.0
        visible = (
            (z > camera.near)
            & (z < camera.far)
            & (u >= -margin)
            & (u <= camera.image_size[0] + margin)
            & (v >= -margin)
            & (v <= camera.image_size[1] + margin)
        )
        frustum[:, :, camera_index] = visible
        projected[:, :, camera_index, 0] = local[:, :, 0]
        projected[:, :, camera_index, 1] = local[:, :, 1]
        projected[:, :, camera_index, 2] = local[:, :, 2]

    score = _occlusion_score(frames, joints, cameras_n, seed)
    hidden_probability = np.full((joints,), 0.075, dtype=np.float64)
    for name in ("l_wrist", "r_wrist", "l_ankle", "r_ankle"):
        hidden_probability[lookup[name]] = 0.19
    for name in ("l_elbow", "r_elbow", "l_knee", "r_knee"):
        hidden_probability[lookup[name]] = 0.12
    visible = frustum & (score >= hidden_probability[None, :, None])
    # The root is always observed by camera zero so the hierarchy always has an anchor.
    visible[:, 0, 0] = True
    return visible, projected


def _generate_observations(
    truth: np.ndarray, cameras: tuple[Camera, ...], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, joints, _ = truth.shape
    visibility, _ = _project_visibility(truth, cameras, seed)
    rng = np.random.Generator(np.random.PCG64(np.uint64(seed + 71)))
    base_sigma = 0.0055
    joint_multiplier = np.ones((joints,), dtype=np.float64)
    lookup = {name: i for i, name in enumerate(JOINT_NAMES)}
    for name in ("l_wrist", "r_wrist", "l_ankle", "r_ankle"):
        joint_multiplier[lookup[name]] = 1.35
    sigma = base_sigma * joint_multiplier[None, :, None] * (0.82 + 0.36 * rng.random((frames, joints, len(cameras))))
    normal = rng.standard_normal((frames, joints, len(cameras), 3))
    normal[:, :, :, 2] *= 1.45
    observations = np.empty_like(truth)[:, :, None, :] + np.zeros((1, 1, len(cameras), 3))
    for camera_index, camera in enumerate(cameras):
        local_noise = normal[:, :, camera_index, :] * sigma[:, :, camera_index, None]
        world_noise = np.einsum("ji,fkj->fki", camera.world_to_camera, local_noise)
        observations[:, :, camera_index, :] = truth + world_noise
    observations = np.where(visibility[:, :, :, None], observations, np.nan)
    return observations, visibility, sigma


def generate_scene(config: GenerationConfig) -> Scene:
    source = default_source_skeleton()
    target = default_target_skeleton()
    local_rotations, root_positions = _procedural_local_rotations(config.frames, config.seed)
    truth = forward_kinematics(source, local_rotations, root_positions)
    cameras = default_cameras()
    observations, visibility, sigma = _generate_observations(truth, cameras, config.seed)
    constraints = default_constraints(target)
    scene = Scene(
        config=config,
        source=source,
        target=target,
        cameras=cameras,
        constraints=constraints,
        observations=observations,
        visibility=visibility,
        sigma=sigma,
        truth=truth,
    )
    ensure(np.any(visibility), ErrorCode.E_OBS_MISSING, "generator produced no observations")
    return scene


def save_scene(scene: Scene, output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays_dir = root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    array_hashes: dict[str, str] = {}
    array_paths: dict[str, str] = {}
    for name, array in scene.array_manifest().items():
        path = arrays_dir / f"{name}.npy"
        array_hashes[name] = write_npy(path, array)
        array_paths[name] = f"arrays/{name}.npy"

    bits = scene.visibility_bits()
    visibility_payload = {
        "encoding": "bit i is set when camera i is visible; invisible camera observations are NaN",
        "camera_order": [camera.name for camera in scene.cameras],
        "shape": list(bits.shape),
        "frames": bits.astype(int).tolist(),
    }
    write_json(root / "visibility_manifest.json", visibility_payload)

    manifest = scene.structure_manifest()
    manifest["observation_manifest"] = {
        "arrays": array_paths,
        "array_sha256": array_hashes,
        "observation_shape": list(scene.observations.shape),
        "observation_dtype": str(scene.observations.dtype),
        "visibility_shape": list(scene.visibility.shape),
        "visibility_encoding": "boolean per frame/joint/camera; bitset mirror is visibility_manifest.json",
        "noise_scale_units": "metres in each camera-local axis",
        "truth_usage": "evaluation_only",
        "truth_reference_path": "arrays/truth.npy",
        "visibility_manifest_path": "visibility_manifest.json",
    }
    manifest["scene_digest"] = scene.digest()
    write_json(root / "scene.json", manifest)
    return manifest


def load_scene(scene_dir: str | Path) -> Scene:
    root = Path(scene_dir)
    manifest = read_json(root / "scene.json")
    try:
        arrays = manifest["observation_manifest"]["arrays"]
        observations = read_npy(root / arrays["observations"])
        visibility = read_npy(root / arrays["visibility"]).astype(bool)
        sigma = read_npy(root / arrays["sigma"])
        truth = read_npy(root / arrays["truth"])
        generation = manifest["generation"]
        scene = Scene(
            config=GenerationConfig(
                frames=int(generation["frames"]),
                seed=int(generation["seed"]),
                fps=int(generation["fps"]),
                camera_count=int(generation["camera_count"]),
            ),
            source=Skeleton.from_manifest(manifest["source_skeleton"]),
            target=Skeleton.from_manifest(manifest["target_skeleton"]),
            cameras=tuple(Camera.from_manifest(item) for item in manifest["cameras"]),
            constraints=ConstraintSet.from_manifest(manifest["constraints"]),
            observations=observations,
            visibility=visibility,
            sigma=sigma,
            truth=truth,
        )
    except (KeyError, ValueError, OSError) as exc:
        raise BusinessError(ErrorCode.E_INPUT, "scene directory is incomplete or invalid", path=str(root)) from exc
    return scene


def scene_file_hashes(scene_dir: str | Path) -> dict[str, str]:
    root = Path(scene_dir)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            from .serialization import sha256_file

            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result
