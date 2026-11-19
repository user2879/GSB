"""Deterministic software offscreen renderer for transparent media and volumes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import Camera, Scene, TileResult, TransparentSurface

VOLUME_FIELDS = (
    "segment_start",
    "segment_step",
    "cumulative_transmittance",
    "density",
    "shadow_visibility",
    "contribution_luminance",
)

_EPS = 1e-9
_HASH_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_HASH_A = np.uint64(0x9E3779B185EBCA87)
_HASH_B = np.uint64(0xC2B2AE3D27D4EB4F)
_HASH_C = np.uint64(0x165667B19E3779F9)


@dataclass(frozen=True)
class TileSpec:
    key: str
    tx: int
    ty: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def pixels(self) -> int:
        return self.width * self.height


def plan_tiles(width: int, height: int, tile_size: int) -> list[TileSpec]:
    tiles: list[TileSpec] = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            tx = x0 // tile_size
            ty = y0 // tile_size
            tiles.append(
                TileSpec(
                    key=f"tile_{ty:04d}_{tx:04d}",
                    tx=tx,
                    ty=ty,
                    x0=x0,
                    y0=y0,
                    x1=min(x0 + tile_size, width),
                    y1=min(y0 + tile_size, height),
                )
            )
    return tiles


def _normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    length = np.sqrt(np.sum(vector * vector, axis=axis, keepdims=True))
    return vector / np.maximum(length, _EPS)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def camera_basis(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    return position, forward, right, true_up


def make_rays(camera: Camera, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, forward, right, true_up = camera_basis(camera)
    aspect = camera.width / camera.height
    tan_half = math.tan(math.radians(camera.fov_y_deg) * 0.5)
    xx, yy = np.meshgrid(xs.astype(np.float64), ys.astype(np.float64))
    u = ((xx + 0.5) / camera.width) * 2.0 - 1.0
    v = 1.0 - ((yy + 0.5) / camera.height) * 2.0
    directions = (
        forward[None, None, :]
        + (u * aspect * tan_half)[:, :, None] * right[None, None, :]
        + (v * tan_half)[:, :, None] * true_up[None, None, :]
    )
    directions = _normalize(directions)
    origins = np.broadcast_to(position[None, None, :], directions.shape).copy()
    return origins, directions


def _surface_basis(normal: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), _EPS)
    reference = np.asarray((0.0, 1.0, 0.0))
    if abs(float(np.dot(reference, n))) > 0.94:
        reference = np.asarray((1.0, 0.0, 0.0))
    u = np.cross(reference, n)
    u = u / max(np.linalg.norm(u), _EPS)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), _EPS)
    return n, u, v


def _intersect_aabb(
    origins: np.ndarray,
    directions: np.ndarray,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    shape = directions.shape[:2]
    t_near = np.full(shape, -np.inf, dtype=np.float64)
    t_far = np.full(shape, np.inf, dtype=np.float64)
    normal = np.zeros_like(directions)
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    for axis in range(3):
        origin_axis = origins[:, :, axis] - c[axis]
        direction_axis = directions[:, :, axis]
        safe = np.where(np.abs(direction_axis) < _EPS, np.where(direction_axis >= 0.0, _EPS, -_EPS), direction_axis)
        t0 = (-h[axis] - origin_axis) / safe
        t1 = (h[axis] - origin_axis) / safe
        t0, t1 = np.minimum(t0, t1), np.maximum(t0, t1)
        axis_normal = np.where(t0 == (-h[axis] - origin_axis) / safe, -1.0, 1.0)
        better = t0 > t_near
        t_near = np.where(better, t0, t_near)
        for component in range(3):
            candidate = np.where(component == axis, axis_normal, 0.0)
            normal[:, :, component] = np.where(better, candidate, normal[:, :, component])
        t_far = np.minimum(t_far, t1)
    valid = (t_far >= t_near) & (t_far > 0.0) & np.isfinite(t_near)
    t_hit = np.where(t_near > _EPS, t_near, t_far)
    return np.where(valid, t_hit, np.inf), normal


def _point_in_boxes(
    points: np.ndarray,
    occluders: Iterable,
    padding: float = 0.0,
) -> np.ndarray:
    result = np.zeros(points.shape[:2], dtype=bool)
    for occluder in occluders:
        center = np.asarray(occluder.center, dtype=np.float64)
        half = np.asarray(occluder.half_extents, dtype=np.float64) + padding
        inside = np.all(np.abs(points - center[None, None, :]) <= half[None, None, :], axis=-1)
        result |= inside
    return result


def _sample_base_scene(
    scene: Scene,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear RGB and the nearest opaque hit distance for each ray."""

    shape = directions.shape[:2]
    rgb = np.zeros((*shape, 3), dtype=np.float64)
    background = np.clip(0.5 + 0.5 * directions[:, :, 1:2], 0.0, 1.0)
    rgb[:, :, 0] = 0.018 + 0.035 * background[:, :, 0]
    rgb[:, :, 1] = 0.028 + 0.055 * background[:, :, 0]
    rgb[:, :, 2] = 0.045 + 0.085 * background[:, :, 0]
    hit_t = np.full(shape, np.inf, dtype=np.float64)
    normal = _normalize(directions.copy())
    albedo = np.zeros_like(rgb)

    ground = scene.ground
    dy = directions[:, :, 1]
    ground_t = (ground.plane_y - origins[:, :, 1]) / np.where(np.abs(dy) < _EPS, np.where(dy >= 0.0, _EPS, -_EPS), dy)
    ground_valid = (dy < -_EPS) & (ground_t > scene.volume.near_plane) & (ground_t < hit_t)
    if np.any(ground_valid):
        points = origins + directions * ground_t[:, :, None]
        checker = (
            np.floor(points[:, :, 0] / ground.checker_scale)
            + np.floor(points[:, :, 2] / ground.checker_scale)
        ).astype(np.int64)
        odd = np.mod(checker, 2) != 0
        ground_albedo = np.where(odd[:, :, None], np.asarray(ground.albedo_b), np.asarray(ground.albedo_a))
        albedo = np.where(ground_valid[:, :, None], ground_albedo, albedo)
        normal = np.where(ground_valid[:, :, None], np.asarray((0.0, 1.0, 0.0)), normal)
        hit_t = np.where(ground_valid, ground_t, hit_t)

    for occluder in scene.occluders:
        box_t, box_normal = _intersect_aabb(origins, directions, occluder.center, occluder.half_extents)
        valid = (box_t > scene.volume.near_plane) & (box_t < hit_t) & np.isfinite(box_t)
        # _intersect_aabb only records the axis with the largest near plane. Flip
        # the outward normal to face the incoming ray.
        face = np.where(_dot(box_normal, directions)[:, :, None] > 0.0, -box_normal, box_normal)
        albedo = np.where(valid[:, :, None], np.asarray(occluder.albedo), albedo)
        normal = np.where(valid[:, :, None], face, normal)
        hit_t = np.where(valid, box_t, hit_t)

    light = np.asarray(scene.light_cone.apex, dtype=np.float64)
    hit_points = origins + directions * np.where(np.isfinite(hit_t), hit_t, 0.0)[:, :, None]
    light_vectors = light[None, None, :] - hit_points
    light_vectors = _normalize(light_vectors)
    diffuse = np.clip(_dot(normal, light_vectors), 0.0, 1.0)
    warm = np.asarray(scene.light_cone.color, dtype=np.float64)
    lit = albedo * (0.16 + 0.84 * diffuse[:, :, None]) * (0.72 + 0.28 * warm[None, None, :])
    has_hit = np.isfinite(hit_t)
    rgb = np.where(has_hit[:, :, None], lit, rgb)
    return rgb, hit_t


def _refract(directions: np.ndarray, normals: np.ndarray, ior: float) -> tuple[np.ndarray, np.ndarray]:
    n = _normalize(normals)
    facing = _dot(directions, n) > 0.0
    n = np.where(facing[:, :, None], -n, n)
    cos_i = np.clip(-_dot(directions, n), 0.0, 1.0)
    eta = 1.0 / ior
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    valid = k >= 0.0
    root = np.sqrt(np.maximum(k, 0.0))
    refracted = eta * directions + (eta * cos_i - root)[:, :, None] * n
    refracted = _normalize(refracted)
    reflected = directions - 2.0 * _dot(directions, n)[:, :, None] * n
    reflected = _normalize(reflected)
    refracted = np.where(valid[:, :, None], refracted, reflected)
    normal_facing = n
    return refracted, normal_facing


def _transparent_hits(
    scene: Scene,
    origins: np.ndarray,
    directions: np.ndarray,
    opaque_t: np.ndarray,
) -> list[dict[str, np.ndarray | TransparentSurface]]:
    hits: list[dict[str, np.ndarray | TransparentSurface]] = []
    shape = directions.shape[:2]
    for surface in scene.surfaces:
        n, u, v = _surface_basis(surface.normal)
        center = np.asarray(surface.center, dtype=np.float64)
        t = np.full(shape, np.inf, dtype=np.float64)
        normal = np.broadcast_to(n[None, None, :], directions.shape).copy()
        valid = np.zeros(shape, dtype=bool)

        if surface.kind == "curved_disk":
            denominator = _dot(directions, np.broadcast_to(n, directions.shape))
            plane_t = _dot(center[None, None, :] - origins, np.broadcast_to(n, directions.shape)) / np.where(
                np.abs(denominator) < _EPS,
                np.where(denominator >= 0.0, _EPS, -_EPS),
                denominator,
            )
            point = origins + directions * plane_t[:, :, None]
            qvec = point - center[None, None, :]
            local_u = _dot(qvec, np.broadcast_to(u, directions.shape))
            local_v = _dot(qvec, np.broadcast_to(v, directions.shape))
            valid = (
                (np.abs(denominator) > _EPS)
                & (plane_t > scene.volume.near_plane)
                & (plane_t < opaque_t)
                & (local_u * local_u + local_v * local_v <= surface.radius * surface.radius)
            )
            bent = (
                n[None, None, :]
                + surface.curvature
                * (local_u[:, :, None] * u[None, None, :] + local_v[:, :, None] * v[None, None, :])
            )
            normal = _normalize(bent)
            t = np.where(valid, plane_t, t)
        elif surface.kind == "glass_slab":
            facing = _dot(directions, np.broadcast_to(n, directions.shape)) < 0.0
            side = np.where(facing, 1.0, -1.0)
            plane_center = center[None, None, :] + side[:, :, None] * surface.half_depth * n[None, None, :]
            plane_normal = n[None, None, :] * side[:, :, None]
            denominator = _dot(directions, plane_normal)
            plane_t = _dot(plane_center - origins, plane_normal) / np.where(
                np.abs(denominator) < _EPS,
                np.where(denominator >= 0.0, _EPS, -_EPS),
                denominator,
            )
            point = origins + directions * plane_t[:, :, None]
            qvec = point - plane_center
            local_u = _dot(qvec, np.broadcast_to(u, directions.shape))
            local_v = _dot(qvec, np.broadcast_to(v, directions.shape))
            valid = (
                (np.abs(denominator) > _EPS)
                & (plane_t > scene.volume.near_plane)
                & (plane_t < opaque_t)
                & (np.abs(local_u) <= surface.half_width)
                & (np.abs(local_v) <= surface.half_height)
            )
            normal = _normalize(plane_normal)
            t = np.where(valid, plane_t, t)
        else:
            continue

        point = origins + directions * np.where(valid, t, 0.0)[:, :, None]
        distance = np.linalg.norm(point - origins, axis=-1)
        hits.append(
            {
                "surface": surface,
                "valid": valid,
                "t": t,
                "distance": distance,
                "point": point,
                "normal": normal,
            }
        )
    return hits


def _compose_transparent(
    scene: Scene,
    base_rgb: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    opaque_t: np.ndarray,
    hits: list[dict[str, np.ndarray | TransparentSurface]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = directions.shape[:2]
    image = base_rgb.copy()
    visibility = np.zeros((h, w), dtype=np.uint32)
    if not hits:
        return image, visibility

    layer_colors: list[np.ndarray] = []
    for hit in hits:
        surface = hit["surface"]
        assert isinstance(surface, TransparentSurface)
        valid = hit["valid"]
        assert isinstance(valid, np.ndarray)
        normal = hit["normal"]
        point = hit["point"]
        assert isinstance(normal, np.ndarray) and isinstance(point, np.ndarray)
        refracted, facing_normal = _refract(directions, normal, surface.ior)
        ref_origin = point + facing_normal * 2.0e-4
        ref_origin = np.where(valid[:, :, None], ref_origin, origins)
        refracted = np.where(valid[:, :, None], refracted, directions)
        sampled, _ = _sample_base_scene(scene, ref_origin, refracted)
        tint = np.asarray(surface.tint, dtype=np.float64)
        transmission = float(surface.transmission)
        specular = np.asarray((0.78, 0.88, 1.0), dtype=np.float64)
        color = sampled * tint[None, None, :] * transmission + specular * (1.0 - transmission)
        layer_colors.append(color)
        visibility |= np.where(valid, np.uint32(1 << surface.sort_index), np.uint32(0)).astype(np.uint32)

    depths = np.stack([np.where(hit["valid"], hit["t"], np.inf) for hit in hits], axis=0)
    distances = np.stack([np.where(hit["valid"], hit["distance"], np.inf) for hit in hits], axis=0)
    valids = np.stack([hit["valid"] for hit in hits], axis=0)
    colors = np.stack(layer_colors, axis=0)

    for y in range(h):
        for x in range(w):
            indices = [index for index in range(len(hits)) if bool(valids[index, y, x])]
            if not indices:
                continue
            indices.sort(
                key=lambda index: (
                    -float(depths[index, y, x]),
                    -float(distances[index, y, x]),
                    int(scene.surfaces[index].sort_index),
                )
            )
            current = image[y, x].copy()
            for index in indices:
                surface = scene.surfaces[index]
                transmission = float(surface.transmission)
                current = (
                    current * transmission
                    + colors[index, y, x] * transmission
                    + np.asarray((0.72, 0.84, 1.0)) * (1.0 - transmission)
                )
            image[y, x] = current
    return image, visibility


def _hash01(
    seed: int,
    xs: np.ndarray,
    ys: np.ndarray,
    step_index: int,
    shadow_index: int,
) -> np.ndarray:
    x = xs.astype(np.uint64)
    y = ys.astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (
            np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
            ^ ((x + np.uint64(1)) * _HASH_A)
            ^ ((y + np.uint64(1)) * _HASH_B)
            ^ (np.uint64(step_index + 1) * _HASH_C)
            ^ (np.uint64(shadow_index + 1) * np.uint64(0xD6E8FEB86659FD93))
        )
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
        h &= _HASH_MASK
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _volume_trace(
    scene: Scene,
    origins: np.ndarray,
    directions: np.ndarray,
    opaque_t: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = directions.shape[:2]
    volume = scene.volume
    cone = scene.light_cone
    steps = volume.max_steps
    step = float(volume.base_step)
    start0 = float(volume.near_clip)
    xx, yy = np.meshgrid(xs, ys)
    apex = np.asarray(cone.apex, dtype=np.float64)
    axis = np.asarray(cone.direction, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), _EPS)
    color = np.asarray(cone.color, dtype=np.float64)

    transmittance = np.ones((h, w), dtype=np.float64)
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    trace = np.zeros((h, w, steps, len(VOLUME_FIELDS)), dtype=np.float32)
    inside = np.zeros((h, w, steps), dtype=np.uint8)

    for step_index in range(steps):
        segment_start = start0 + step_index * step
        segment_mid = segment_start + 0.5 * step
        points = origins + directions * segment_mid
        relative = points - apex[None, None, :]
        axial = _dot(relative, np.broadcast_to(axis, relative.shape))
        radial_vec = relative - axial[:, :, None] * axis[None, None, :]
        radial = np.linalg.norm(radial_vec, axis=-1)
        radius_limit = cone.radius_start + (cone.radius_end - cone.radius_start) * np.clip(
            axial / max(cone.length, _EPS), 0.0, 1.0
        )
        geometric = (
            (axial >= 0.0)
            & (axial <= cone.length)
            & (radial <= radius_limit)
            & (segment_mid <= opaque_t)
        )
        edge_width = np.maximum(radius_limit * 0.075, 0.035)
        edge = np.clip((radius_limit - radial) / edge_width, 0.0, 1.0)
        tip_fade = np.clip((cone.length - axial) / 0.22, 0.0, 1.0)
        density = np.where(geometric, edge * tip_fade, 0.0)
        inside[:, :, step_index] = geometric.astype(np.uint8)

        visibility = np.zeros((h, w), dtype=np.float64)
        if cone.shadow_samples > 0:
            to_light = apex[None, None, :] - points
            for shadow_index in range(cone.shadow_samples):
                jitter = _hash01(cone.jitter_seed, xx, yy, step_index, shadow_index)
                fraction = (shadow_index + jitter) / cone.shadow_samples
                shadow_points = points + to_light * fraction[:, :, None]
                blocked = _point_in_boxes(shadow_points, scene.occluders, padding=0.002)
                blocked |= shadow_points[:, :, 1] <= scene.ground.plane_y + 1.0e-5
                visibility += np.where(blocked, 0.0, 1.0)
            visibility /= cone.shadow_samples
            visibility = np.where(geometric, visibility, 0.0)

        sigma_s = cone.scatter * density
        sigma_t = (cone.scatter + cone.absorption) * density
        contribution = (
            transmittance[:, :, None]
            * sigma_s[:, :, None]
            * visibility[:, :, None]
            * volume.phase_isotropic
            * step
            * color[None, None, :]
        )
        rgb += contribution
        transmittance *= np.exp(-sigma_t * step)

        trace[:, :, step_index, 0] = segment_start
        trace[:, :, step_index, 1] = step
        trace[:, :, step_index, 2] = transmittance.astype(np.float32)
        trace[:, :, step_index, 3] = density.astype(np.float32)
        trace[:, :, step_index, 4] = visibility.astype(np.float32)
        trace[:, :, step_index, 5] = np.mean(contribution, axis=-1).astype(np.float32)

    return rgb, trace, inside, transmittance


def render_tile(scene: Scene, spec: TileSpec) -> TileResult:
    xs = np.arange(spec.x0, spec.x1, dtype=np.int64)
    ys = np.arange(spec.y0, spec.y1, dtype=np.int64)
    origins, directions = make_rays(scene.camera, xs, ys)
    base_rgb, opaque_t = _sample_base_scene(scene, origins, directions)
    hits = _transparent_hits(scene, origins, directions, opaque_t)
    surface_rgb, visibility = _compose_transparent(scene, base_rgb, origins, directions, opaque_t, hits)
    volume_rgb, trace, inside, transmittance = _volume_trace(scene, origins, directions, opaque_t, xs, ys)
    volume_alpha = 1.0 - transmittance
    rgb = surface_rgb + volume_rgb

    image = np.empty((spec.height, spec.width, 4), dtype=np.float32)
    image[:, :, :3] = np.clip(rgb, 0.0, 4.0).astype(np.float32)
    image[:, :, 3] = np.clip(volume_alpha, 0.0, 1.0).astype(np.float32)
    cloud_mask = np.any(inside, axis=-1)
    if np.any(cloud_mask):
        visibility |= np.uint32(1 << 31)
    return TileResult(
        key=spec.key,
        tx=spec.tx,
        ty=spec.ty,
        x0=spec.x0,
        y0=spec.y0,
        x1=spec.x1,
        y1=spec.y1,
        image=image,
        visibility_mask=visibility.astype(np.uint32, copy=False),
        volume_trace=trace,
        volume_inside=inside,
    )


def image_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image, dtype="<f4").tobytes(order="C")).hexdigest()


def trace_sha256(trace: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(trace, dtype="<f4").tobytes(order="C")).hexdigest()


def tile_hashes(result: TileResult) -> dict[str, str]:
    image_hash = image_sha256(result.image)
    trace_hash = trace_sha256(result.volume_trace)
    inside_hash = hashlib.sha256(np.ascontiguousarray(result.volume_inside, dtype=np.uint8).tobytes(order="C")).hexdigest()
    combined = hashlib.sha256(
        image_hash.encode("ascii") + trace_hash.encode("ascii") + inside_hash.encode("ascii")
    ).hexdigest()
    return {
        "image_sha256": image_hash,
        "volume_trace_sha256": trace_hash,
        "volume_inside_sha256": inside_hash,
        "combined_sha256": combined,
    }


def assemble_image(tiles: dict[str, TileResult], width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.float32)
    written = np.zeros((height, width), dtype=np.uint8)
    for key in sorted(tiles):
        tile = tiles[key]
        image[tile.y0 : tile.y1, tile.x0 : tile.x1, :] = tile.image
        written[tile.y0 : tile.y1, tile.x0 : tile.x1] += 1
    if not np.all(written == 1):
        raise RuntimeError("tile merge produced missing or duplicate pixels")
    return image


def assemble_trace(tiles: dict[str, TileResult], width: int, height: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    trace = np.zeros((height, width, steps, len(VOLUME_FIELDS)), dtype=np.float32)
    inside = np.zeros((height, width, steps), dtype=np.uint8)
    for key in sorted(tiles):
        tile = tiles[key]
        trace[tile.y0 : tile.y1, tile.x0 : tile.x1, :, :] = tile.volume_trace
        inside[tile.y0 : tile.y1, tile.x0 : tile.x1, :] = tile.volume_inside
    return trace, inside
