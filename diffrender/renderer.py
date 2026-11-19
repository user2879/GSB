"""Deterministic soft rasterization, shading, and analytic pixel gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from .errors import DiffRenderError, E_GRAD_MISMATCH, E_SCENE_INVALID, E_TILE_HASH
from .hashing import (
    array_max_abs_diff,
    digest_sequence,
    pack_floats,
    pack_ints,
    sha256_bytes,
    tolerant_float_hash,
)
from .math3d import camera_basis, clamp, component_basis, dot, normalize, vsub
from .scene import Scene


DEPTH_FAR = 1.0e300
ALL_GEOMETRY_FIELDS = ("cx", "cy", "cz", "radius")
SPHERE_FIELDS = ("cx", "cy", "cz", "radius", "albedo", "roughness")
LIGHT_FIELDS = ("dx", "dy", "dz", "ambient")
CAMERA_FIELDS = ("ex", "ey", "ez", "yaw", "pitch", "roll", "focal")


@dataclass(frozen=True)
class Projection:
    index: int
    qx: float
    qy: float
    qz: float
    cx: float
    cy: float
    radius_px: float
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass(frozen=True)
class TileResult:
    ty: int
    tx: int
    y0: int
    y1: int
    x0: int
    x1: int
    ids: tuple[int, ...]
    bary: tuple[float, ...]
    depth: tuple[float, ...]
    coverage: tuple[float, ...]
    normals: tuple[float, ...]
    ambient_channel: tuple[float, ...]
    diffuse_channel: tuple[float, ...]
    intensity: tuple[float, ...]
    spheres: tuple[int, ...]

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def key(self) -> tuple[int, int]:
        return (self.ty, self.tx)

    @property
    def key_string(self) -> str:
        return tile_key(self.ty, self.tx)

    def raster_hash(self) -> str:
        return digest_sequence(
            (
                pack_ints((self.ty, self.tx, self.y0, self.y1, self.x0, self.x1)),
                pack_ints(self.ids),
                pack_floats(self.bary),
                pack_floats(self.depth),
                pack_floats(self.coverage),
                pack_floats(self.normals),
                pack_ints(self.spheres),
            )
        )

    def shade_hash(self) -> str:
        return digest_sequence(
            (
                pack_floats(self.ambient_channel),
                pack_floats(self.diffuse_channel),
                pack_floats(self.intensity),
            )
        )

    def content_hash(self) -> str:
        return digest_sequence((self.raster_hash().encode("ascii"), self.shade_hash().encode("ascii")))

    def image_pixels(self) -> tuple[float, ...]:
        return self.intensity


@dataclass(frozen=True)
class RenderResult:
    scene: Scene
    tiles: dict[tuple[int, int], TileResult]
    image: tuple[float, ...]
    raster_hash: str
    shade_hash: str
    image_hash: str
    render_hash: str
    parameter_hash: str

    def tile(self, ty: int, tx: int) -> TileResult:
        return self.tiles[(ty, tx)]


@dataclass(frozen=True)
class GradientResult:
    loss: float
    image_hash: str
    pixel_gradients: dict[str, tuple[float, ...]]
    scalar_gradients: dict[str, float]
    chunks: dict[str, dict[tuple[int, int], tuple[float, ...]]]
    gradient_hash: str
    parameter_hash: str

    def scalar(self, key: str) -> float:
        return self.scalar_gradients[key]


def tile_key(ty: int, tx: int) -> str:
    return f"tile:{ty}:{tx}"


def raster_node_key(ty: int, tx: int) -> str:
    return f"raster:{tile_key(ty, tx)}"


def shade_node_key(ty: int, tx: int) -> str:
    return f"shade:{tile_key(ty, tx)}"


def grad_chunk_node_key(ty: int, tx: int, param_key: str) -> str:
    return f"gradchunk:{tile_key(ty, tx)}:{param_key}"


def grad_aggregate_node_key(param_key: str) -> str:
    return f"grad:{param_key}"


def param_node_key(param_key: str) -> str:
    return f"param:{param_key}"


def tile_layout(scene: Scene) -> tuple[tuple[int, int, int, int, int, int], ...]:
    out = []
    for ty, y0 in enumerate(range(0, scene.height, scene.tile_size)):
        for tx, x0 in enumerate(range(0, scene.width, scene.tile_size)):
            out.append((ty, tx, y0, y0 + scene.tile_size, x0, x0 + scene.tile_size))
    return tuple(out)


def project_sphere(scene: Scene, index: int) -> Projection:
    sphere = scene.spheres[index]
    right, up, forward = camera_basis(scene.camera.yaw, scene.camera.pitch, scene.camera.roll)
    delta = vsub(sphere.center, scene.camera.eye)
    qx, qy, qz = dot(delta, right), dot(delta, up), dot(delta, forward)
    if qz <= 0.0:
        raise DiffRenderError(E_SCENE_INVALID, f"sphere {index} projects behind the camera")
    focal = scene.camera.focal
    cx = scene.width * 0.5 + focal * qx / qz
    cy = scene.height * 0.5 - focal * qy / qz
    radius_px = focal * sphere.radius / qz
    x0 = max(0, int(math.floor(cx - radius_px - 1.0)))
    x1 = min(scene.width, int(math.ceil(cx + radius_px + 1.0)))
    y0 = max(0, int(math.floor(cy - radius_px - 1.0)))
    y1 = min(scene.height, int(math.ceil(cy + radius_px + 1.0)))
    return Projection(index, qx, qy, qz, cx, cy, radius_px, x0, x1, y0, y1)


def _coverage(radius_px: float, dx: float, dy: float) -> tuple[float, float, float]:
    dist = math.hypot(dx, dy)
    t = radius_px + 0.5 - dist
    return (clamp(t, 0.0, 1.0), dist, t)


def _sphere_pixel(
    scene: Scene, projection: Projection, x: float, y: float
) -> tuple[float, float, float, float, float, tuple[float, float, float], int]:
    dx = x - projection.cx
    dy = y - projection.cy
    cov, dist, t = _coverage(projection.radius_px, dx, dy)
    if projection.radius_px <= 1.0e-15:
        return cov, dx, dy, dist, 0.0, (0.0, 0.0, 0.0), 0
    u = dx / projection.radius_px
    v = -dy / projection.radius_px
    rho2 = u * u + v * v
    w = math.sqrt(max(0.0, 1.0 - rho2))
    right, up, forward = camera_basis(scene.camera.yaw, scene.camera.pitch, scene.camera.roll)
    normal = (
        right[0] * u + up[0] * v - forward[0] * w,
        right[1] * u + up[1] * v - forward[1] * w,
        right[2] * u + up[2] * v - forward[2] * w,
    )
    return cov, dx, dy, dist, t, normal, 0


def _barycentric(projection: Projection, x: float, y: float) -> tuple[float, float, float]:
    r = max(projection.radius_px, 1.0e-15)
    a = (x - (projection.cx - r)) / (2.0 * r)
    b = (y - (projection.cy - r)) / (2.0 * r)
    if a >= b:
        # Triangle (p00, p10, p11).
        return (1.0 - a, a - b, b)
    # Triangle (p00, p01, p11).
    return (1.0 - b, b - a, a)


def render_tile(
    scene: Scene,
    ty: int,
    tx: int,
    projections: Sequence[Projection] | None = None,
) -> TileResult:
    y0, y1 = ty * scene.tile_size, (ty + 1) * scene.tile_size
    x0, x1 = tx * scene.tile_size, (tx + 1) * scene.tile_size
    if projections is None:
        projections = tuple(project_sphere(scene, i) for i in range(len(scene.spheres)))
    w, h = x1 - x0, y1 - y0
    count = w * h
    ids = [-1] * count
    bary = [0.0] * (count * 3)
    depth = [DEPTH_FAR] * count
    coverage = [0.0] * count
    normals = [0.0] * (count * 3)
    spheres_used: set[int] = set()

    for projection in projections:
        if projection.x1 <= x0 or projection.x0 >= x1 or projection.y1 <= y0 or projection.y0 >= y1:
            continue
        for py in range(max(y0, projection.y0), min(y1, projection.y1)):
            for px in range(max(x0, projection.x0), min(x1, projection.x1)):
                cov, dx, dy, dist, t, normal, _ = _sphere_pixel(
                    scene, projection, px + 0.5, py + 0.5
                )
                if cov <= 0.0:
                    continue
                spheres_used.add(projection.index)
                sphere = scene.spheres[projection.index]
                rho2 = 0.0
                if projection.radius_px > 1.0e-15:
                    u, v = dx / projection.radius_px, -dy / projection.radius_px
                    rho2 = min(1.0, u * u + v * v)
                pixel_depth = projection.qz - sphere.radius * math.sqrt(max(0.0, 1.0 - rho2))
                offset = (py - y0) * w + (px - x0)
                if pixel_depth < depth[offset]:
                    ids[offset] = projection.index
                    depth[offset] = pixel_depth
                    coverage[offset] = cov
                    bary[offset * 3 : offset * 3 + 3] = _barycentric(
                        projection, px + 0.5, py + 0.5
                    )
                    normals[offset * 3 : offset * 3 + 3] = normal

    raster = TileResult(
        ty=ty,
        tx=tx,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        ids=tuple(ids),
        bary=tuple(bary),
        depth=tuple(depth),
        coverage=tuple(coverage),
        normals=tuple(normals),
        ambient_channel=(),
        diffuse_channel=(),
        intensity=(),
        spheres=tuple(sorted(spheres_used)),
    )
    return shade_tile(scene, raster)


def shade_tile(scene: Scene, raster: TileResult) -> TileResult:
    """Recompute only the shading channels for an existing raster tile."""

    count = raster.width * raster.height
    ambient = [0.0] * count
    diffuse = [0.0] * count
    intensity = [scene.background] * count
    light = normalize(scene.light.direction)
    for py in range(raster.y0, raster.y1):
        for px in range(raster.x0, raster.x1):
            offset = (py - raster.y0) * raster.width + (px - raster.x0)
            sphere_id = raster.ids[offset]
            if sphere_id < 0:
                continue
            sphere = scene.spheres[sphere_id]
            normal = raster.normals[offset * 3 : offset * 3 + 3]
            lambert = max(0.0, dot(normal, light))
            roughness_gain = 1.0 / (1.0 + sphere.roughness)
            ambient_value = sphere.albedo * scene.light.ambient * raster.coverage[offset]
            diffuse_value = (
                sphere.albedo
                * (1.0 - scene.light.ambient)
                * lambert
                * roughness_gain
                * raster.coverage[offset]
            )
            ambient[offset] = ambient_value
            diffuse[offset] = diffuse_value
            intensity[offset] = (
                ambient_value
                + diffuse_value
                + scene.background * (1.0 - raster.coverage[offset])
            )
    return replace(
        raster,
        ambient_channel=tuple(ambient),
        diffuse_channel=tuple(diffuse),
        intensity=tuple(intensity),
    )


def compose_image(scene: Scene, tiles: dict[tuple[int, int], TileResult]) -> tuple[float, ...]:
    image = [0.0] * (scene.width * scene.height)
    for ty, tx, _y0, _y1, _x0, _x1 in tile_layout(scene):
        tile = tiles[(ty, tx)]
        for local_y in range(tile.height):
            start = (tile.y0 + local_y) * scene.width + tile.x0
            src = local_y * tile.width
            image[start : start + tile.width] = tile.intensity[src : src + tile.width]
    return tuple(image)


def image_hash(image: Sequence[float]) -> str:
    return sha256_bytes(pack_floats(image))


def raster_manifest_hash(scene: Scene, tiles: dict[tuple[int, int], TileResult]) -> str:
    chunks: list[bytes] = []
    for ty, tx, *_ in tile_layout(scene):
        tile = tiles[(ty, tx)]
        chunks.extend(
            (tile_key(ty, tx).encode("ascii"), tile.raster_hash().encode("ascii"))
        )
    return digest_sequence(chunks)


def shade_manifest_hash(scene: Scene, tiles: dict[tuple[int, int], TileResult]) -> str:
    chunks: list[bytes] = []
    for ty, tx, *_ in tile_layout(scene):
        tile = tiles[(ty, tx)]
        chunks.extend(
            (tile_key(ty, tx).encode("ascii"), tile.shade_hash().encode("ascii"))
        )
    return digest_sequence(chunks)


def render_full(scene: Scene) -> RenderResult:
    projections = tuple(project_sphere(scene, i) for i in range(len(scene.spheres)))
    tiles: dict[tuple[int, int], TileResult] = {}
    for ty, tx, *_ in tile_layout(scene):
        tiles[(ty, tx)] = render_tile(scene, ty, tx, projections)
    image = compose_image(scene, tiles)
    raster_hash = raster_manifest_hash(scene, tiles)
    shade_hash = shade_manifest_hash(scene, tiles)
    img_hash = image_hash(image)
    render_hash = digest_sequence(
        (
            raster_hash.encode("ascii"),
            shade_hash.encode("ascii"),
            img_hash.encode("ascii"),
        )
    )
    return RenderResult(
        scene=scene,
        tiles=tiles,
        image=image,
        raster_hash=raster_hash,
        shade_hash=shade_hash,
        image_hash=img_hash,
        render_hash=render_hash,
        parameter_hash=scene.parameter_hash(),
    )


def mean_squared_loss(image: Sequence[float], target: Sequence[float]) -> float:
    if len(image) != len(target):
        raise DiffRenderError(E_SCENE_INVALID, "image and target dimensions differ")
    return sum((a - b) * (a - b) for a, b in zip(image, target)) / float(len(image))


@dataclass(frozen=True)
class _ProjectionBundle:
    projection: Projection
    focal: float
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]
    basis_derivatives: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        ...,
    ]


def _projection_bundles(scene: Scene) -> tuple[_ProjectionBundle, ...]:
    right, up, forward, derivatives = camera_basis(
        scene.camera.yaw, scene.camera.pitch, scene.camera.roll, with_derivatives=True
    )
    return tuple(
        _ProjectionBundle(project_sphere(scene, i), scene.camera.focal, right, up, forward, derivatives)
        for i in range(len(scene.spheres))
    )


def _pixel_state(
    bundle: _ProjectionBundle, x: float, y: float
) -> tuple[float, float, float, float, float, float, float]:
    p = bundle.projection
    dx, dy = x - p.cx, y - p.cy
    cov, dist, t = _coverage(p.radius_px, dx, dy)
    if p.radius_px <= 1.0e-15:
        return cov, dx, dy, dist, t, 0.0, 0.0
    u = dx / p.radius_px
    v = -dy / p.radius_px
    return cov, dx, dy, dist, t, u, v


def _normal(bundle: _ProjectionBundle, u: float, v: float) -> tuple[float, float, float]:
    w = math.sqrt(max(0.0, 1.0 - min(1.0, u * u + v * v)))
    return (
        bundle.right[0] * u + bundle.up[0] * v - bundle.forward[0] * w,
        bundle.right[1] * u + bundle.up[1] * v - bundle.forward[1] * w,
        bundle.right[2] * u + bundle.up[2] * v - bundle.forward[2] * w,
    )


def _screen_derivatives(
    bundle: _ProjectionBundle,
    dx: float,
    dy: float,
    dist: float,
    u: float,
    v: float,
    t: float,
    dqx: float,
    dqy: float,
    dqz: float,
    dr: float,
    dfocal: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return d(cx), d(cy), d(radius_px), d(coverage), du, dv, dw, dt."""

    p = bundle.projection
    qz2 = p.qz * p.qz
    if qz2 <= 1.0e-30:
        return (0.0,) * 8
    f = bundle.focal
    sphere_radius = p.radius_px * p.qz / f
    dcx = dfocal * p.qx / p.qz + f * (p.qz * dqx - p.qx * dqz) / qz2
    dcy = -dfocal * p.qy / p.qz - f * (p.qz * dqy - p.qy * dqz) / qz2
    drpx = dfocal * sphere_radius / p.qz + f * (p.qz * dr - sphere_radius * dqz) / qz2
    ddx, ddy = -dcx, -dcy
    ddist = 0.0 if dist <= 1.0e-15 else (dx * ddx + dy * ddy) / dist
    dt = drpx - ddist
    dcov = dt if 0.0 < t < 1.0 else 0.0
    if p.radius_px <= 1.0e-15:
        return dcx, dcy, drpx, dcov, 0.0, 0.0, 0.0, dt
    rp2 = p.radius_px * p.radius_px
    du = ddx / p.radius_px - dx * drpx / rp2
    dv = -ddy / p.radius_px + dy * drpx / rp2
    rho2 = u * u + v * v
    w = math.sqrt(max(0.0, 1.0 - min(1.0, rho2)))
    if w > 1.0e-12 and rho2 < 1.0:
        dw = -(u * du + v * dv) / w
    else:
        dw = 0.0
    return dcx, dcy, drpx, dcov, du, dv, dw, dt


def _derivative_intensity(
    scene: Scene,
    visible_index: int,
    bundle: _ProjectionBundle,
    x: float,
    y: float,
    key: str,
) -> float:
    sphere = scene.spheres[visible_index]
    light = normalize(scene.light.direction)
    cov, dx, dy, dist, t, u, v = _pixel_state(bundle, x, y)
    normal = _normal(bundle, u, v)
    lambert = max(0.0, dot(normal, light))
    roughness_gain = 1.0 / (1.0 + sphere.roughness)
    ambient = scene.light.ambient
    base = ambient + (1.0 - ambient) * lambert * roughness_gain
    parts = key.split(".")
    dqx = dqy = dqz = dr = dfocal = 0.0
    dn = (0.0, 0.0, 0.0)
    dlight = (0.0, 0.0, 0.0)

    if parts[0] == "sphere":
        index = int(parts[1])
        field = parts[2]
        if index != visible_index:
            return 0.0
        if field in ALL_GEOMETRY_FIELDS:
            if field in ("cx", "cy", "cz"):
                axis = component_basis(SPHERE_FIELDS.index(field))
                dqx = dot(axis, bundle.right)
                dqy = dot(axis, bundle.up)
                dqz = dot(axis, bundle.forward)
            elif field == "radius":
                dr = 1.0
        elif field == "albedo":
            return base * cov
        elif field == "roughness":
            dk = -1.0 / ((1.0 + sphere.roughness) ** 2)
            return sphere.albedo * (1.0 - ambient) * lambert * cov * dk
        else:
            return 0.0
    elif parts[0] == "light":
        field = parts[1]
        if field == "ambient":
            return sphere.albedo * (1.0 - lambert * roughness_gain) * cov
        component = LIGHT_FIELDS.index(field)
        light_norm = math.sqrt(dot(scene.light.direction, scene.light.direction))
        dlight = tuple(
            ((1.0 if i == component else 0.0) - light[i] * light[component]) / light_norm
            for i in range(3)
        )
        if lambert <= 0.0:
            return 0.0
        dlam = dot(normal, dlight)
        return sphere.albedo * (1.0 - ambient) * roughness_gain * cov * dlam
    elif parts[0] == "camera":
        field = parts[1]
        if field in ("ex", "ey", "ez"):
            axis = component_basis(CAMERA_FIELDS.index(field))
            dqx, dqy, dqz = -dot(axis, bundle.right), -dot(axis, bundle.up), -dot(axis, bundle.forward)
        elif field in ("yaw", "pitch", "roll"):
            k = {"yaw": 0, "pitch": 1, "roll": 2}[field]
            dright, dup, dforward = bundle.basis_derivatives[k]
            delta = vsub(sphere.center, scene.camera.eye)
            dqx = dot(delta, dright)
            dqy = dot(delta, dup)
            dqz = dot(delta, dforward)
        elif field == "focal":
            dfocal = 1.0
        else:
            return 0.0
    else:
        return 0.0

    _dcx, _dcy, _drpx, dcov, du, dv, dw, _dt = _screen_derivatives(
        bundle, dx, dy, dist, u, v, t, dqx, dqy, dqz, dr, dfocal
    )
    if parts[0] == "camera" and parts[1] in ("yaw", "pitch", "roll"):
        k = {"yaw": 0, "pitch": 1, "roll": 2}[parts[1]]
        dright, dup, dforward = bundle.basis_derivatives[k]
        w = math.sqrt(max(0.0, 1.0 - min(1.0, u * u + v * v)))
        dn = (
            dright[0] * u + dup[0] * v - dforward[0] * w
            + bundle.right[0] * du + bundle.up[0] * dv - bundle.forward[0] * dw,
            dright[1] * u + dup[1] * v - dforward[1] * w
            + bundle.right[1] * du + bundle.up[1] * dv - bundle.forward[1] * dw,
            dright[2] * u + dup[2] * v - dforward[2] * w
            + bundle.right[2] * du + bundle.up[2] * dv - bundle.forward[2] * dw,
        )
    else:
        dn = (
            bundle.right[0] * du + bundle.up[0] * dv - bundle.forward[0] * dw,
            bundle.right[1] * du + bundle.up[1] * dv - bundle.forward[1] * dw,
            bundle.right[2] * du + bundle.up[2] * dv - bundle.forward[2] * dw,
        )
    dlam = dot(dn, light) if lambert > 0.0 else 0.0
    d_base_direct = sphere.albedo * (1.0 - ambient) * roughness_gain
    coefficient = sphere.albedo * base - scene.background
    return coefficient * dcov + d_base_direct * cov * dlam


def gradient_tile(
    scene: Scene,
    target: Sequence[float],
    tile: TileResult,
    bundles: Sequence[_ProjectionBundle] | None = None,
) -> dict[str, tuple[float, ...]]:
    if len(target) != scene.width * scene.height:
        raise DiffRenderError(E_SCENE_INVALID, "target image dimension mismatch")
    if bundles is None:
        bundles = _projection_bundles(scene)
    keys = scene.param_keys()
    count = tile.width * tile.height
    result = {key: [0.0] * count for key in keys}
    inv_pixels = 1.0 / float(scene.width * scene.height)
    for local_y in range(tile.height):
        y = tile.y0 + local_y
        for local_x in range(tile.width):
            x = tile.x0 + local_x
            offset = local_y * tile.width + local_x
            sphere_id = tile.ids[offset]
            if sphere_id < 0:
                continue
            intensity = tile.intensity[offset]
            target_value = target[y * scene.width + x]
            dl_di = 2.0 * (intensity - target_value) * inv_pixels
            if dl_di == 0.0:
                continue
            bundle = bundles[sphere_id]
            px, py = x + 0.5, y + 0.5
            for key in keys:
                result[key][offset] += dl_di * _derivative_intensity(
                    scene, sphere_id, bundle, px, py, key
                )
    return {key: tuple(values) for key, values in result.items()}


def gradient_hash(gradients: dict[str, Sequence[float]], keys: Sequence[str] | None = None) -> str:
    order = tuple(keys) if keys is not None else tuple(sorted(gradients))
    chunks: list[bytes] = []
    for key in order:
        chunks.extend((key.encode("utf-8"), pack_floats(gradients[key])))
    return digest_sequence(chunks)


def gradient_full(scene: Scene, target: Sequence[float]) -> GradientResult:
    render_result = render_full(scene)
    bundles = _projection_bundles(scene)
    keys = scene.param_keys()
    chunks: dict[str, dict[tuple[int, int], tuple[float, ...]]] = {key: {} for key in keys}
    for ty, tx, *_ in tile_layout(scene):
        chunk_map = gradient_tile(scene, target, render_result.tiles[(ty, tx)], bundles)
        for key in keys:
            chunks[key][(ty, tx)] = chunk_map[key]
    pixel_gradients: dict[str, tuple[float, ...]] = {}
    scalar_gradients: dict[str, float] = {}
    for key in keys:
        pixels = [0.0] * (scene.width * scene.height)
        for ty, tx, *_ in tile_layout(scene):
            values = chunks[key][(ty, tx)]
            for local_y in range(scene.tile_size):
                row = (ty * scene.tile_size + local_y) * scene.width + tx * scene.tile_size
                src = local_y * scene.tile_size
                pixels[row : row + scene.tile_size] = values[src : src + scene.tile_size]
        pixel_gradients[key] = tuple(pixels)
        scalar_gradients[key] = sum(pixels)
    hash_value = gradient_hash(pixel_gradients, keys)
    return GradientResult(
        loss=mean_squared_loss(render_result.image, target),
        image_hash=render_result.image_hash,
        pixel_gradients=pixel_gradients,
        scalar_gradients=scalar_gradients,
        chunks=chunks,
        gradient_hash=hash_value,
        parameter_hash=scene.parameter_hash(),
    )


def finite_difference_loss_gradient(
    scene: Scene, target: Sequence[float], key: str, step: float = 1.0e-8
) -> float:
    if step <= 0.0:
        raise DiffRenderError(E_SCENE_INVALID, "finite difference step must be positive")
    plus = scene.with_value(key, scene.value(key) + step)
    minus = scene.with_value(key, scene.value(key) - step)
    plus_loss = mean_squared_loss(render_full(plus).image, target)
    minus_loss = mean_squared_loss(render_full(minus).image, target)
    return (plus_loss - minus_loss) / (2.0 * step)


def compare_gradient_to_finite_difference(
    scene: Scene,
    target: Sequence[float],
    keys: Iterable[str] | None = None,
    step: float = 1.0e-8,
    tolerance: float = 2.0e-4,
) -> dict[str, dict[str, float]]:
    result = gradient_full(scene, target)
    selected = tuple(keys) if keys is not None else scene.param_keys()
    report: dict[str, dict[str, float]] = {}
    for key in selected:
        fd = finite_difference_loss_gradient(scene, target, key, step)
        analytic = result.scalar_gradients[key]
        error = abs(fd - analytic)
        report[key] = {"analytic": analytic, "finite_difference": fd, "abs_error": error}
        if error > tolerance:
            raise DiffRenderError(
                E_GRAD_MISMATCH,
                f"analytic gradient mismatch for {key}",
                {"param_key": key, "abs_error": error, "tolerance": tolerance, **report[key]},
            )
    return report


def compare_render_results(
    incremental: RenderResult,
    full: RenderResult,
    tolerance: float = 1.0e-12,
) -> dict[str, object]:
    if incremental.image_hash != full.image_hash:
        max_error = array_max_abs_diff(incremental.image, full.image)
        first_tile = None
        for ty, tx, *_ in tile_layout(incremental.scene):
            if incremental.tiles[(ty, tx)].content_hash() != full.tiles[(ty, tx)].content_hash():
                first_tile = tile_key(ty, tx)
                break
        if first_tile is None:
            for index, (left, right) in enumerate(zip(incremental.image, full.image)):
                if left != right:
                    y, x = divmod(index, incremental.scene.width)
                    first_tile = tile_key(y // incremental.scene.tile_size, x // incremental.scene.tile_size)
                    break
        if max_error > tolerance or first_tile is not None:
            raise DiffRenderError(
                E_TILE_HASH,
                "incremental image hash differs from full render",
                {
                    "first_divergent_tile": first_tile,
                    "max_abs_error": max_error,
                    "tolerance": tolerance,
                },
            )
    return {
        "image_hash": incremental.image_hash,
        "raster_hash": incremental.raster_hash,
        "shade_hash": incremental.shade_hash,
    }


def compare_gradient_results(
    incremental: GradientResult,
    full: GradientResult,
    tolerance: float = 1.0e-11,
) -> dict[str, object]:
    keys = tuple(sorted(incremental.pixel_gradients))
    mismatch_tile = None
    mismatch_key = None
    max_error = 0.0
    tile_positions = tuple(sorted(next(iter(full.chunks.values())).keys()))
    for key in keys:
        inc = incremental.pixel_gradients[key]
        ref = full.pixel_gradients[key]
        err = array_max_abs_diff(inc, ref)
        if err > max_error:
            max_error = err
        if err > tolerance and mismatch_key is None:
            mismatch_key = key
            for ty, tx in tile_positions:
                a = incremental.chunks[key][(ty, tx)]
                b = full.chunks[key][(ty, tx)]
                if array_max_abs_diff(a, b) > tolerance:
                    mismatch_tile = tile_key(ty, tx)
                    break
    semantic_inc = tolerant_float_hash(
        [v for key in keys for v in incremental.pixel_gradients[key]], decimals=10
    )
    semantic_full = tolerant_float_hash(
        [v for key in keys for v in full.pixel_gradients[key]], decimals=10
    )
    if mismatch_key is not None or semantic_inc != semantic_full:
        raise DiffRenderError(
            E_GRAD_MISMATCH,
            "incremental gradient differs from full recomputation",
            {
                "first_divergent_tile": mismatch_tile,
                "param_key": mismatch_key,
                "max_abs_error": max_error,
                "tolerance": tolerance,
                "incremental_semantic_hash": semantic_inc,
                "full_semantic_hash": semantic_full,
            },
        )
    return {
        "gradient_hash": incremental.gradient_hash,
        "semantic_hash": semantic_inc,
        "max_abs_error": max_error,
    }
