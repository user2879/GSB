"""Deterministic NPR rasterizer with mode-independent shared geometry."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .errors import NprError, require
from .geometry import GeometryEngine
from .model import Color, Project, Shape, Style
from .png import encode_png, image_hashes

RenderMode = Literal["lineart", "sketch", "toon"]
RENDER_MODES: tuple[RenderMode, ...] = ("lineart", "sketch", "toon")


@dataclass(frozen=True)
class RenderConfig:
    width: int = 240
    height: int = 160
    supersample: int = 2

    def __post_init__(self) -> None:
        require(
            isinstance(self.width, int) and 16 <= self.width <= 4096,
            "E_RENDER_MISMATCH",
            "render width must be an integer in [16, 4096]",
        )
        require(
            isinstance(self.height, int) and 16 <= self.height <= 4096,
            "E_RENDER_MISMATCH",
            "render height must be an integer in [16, 4096]",
        )
        require(
            self.supersample in (1, 2, 4),
            "E_RENDER_MISMATCH",
            "supersample must be 1, 2 or 4",
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "supersample": self.supersample,
        }


@dataclass
class RenderResult:
    mode: str
    image: np.ndarray
    shape_hashes: dict[str, str]
    geometry_hashes: dict[str, str]
    pixel_sha256: str
    png_sha256: str
    flatten_builds: int
    skeleton_builds: int
    render_builds: int
    render_cache_hits: int

    def report(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "full_pixel_sha256": self.pixel_sha256,
            "full_png_sha256": self.png_sha256,
            "shape_render_hashes": dict(sorted(self.shape_hashes.items())),
            "geometry_hashes": dict(sorted(self.geometry_hashes.items())),
            "flatten_builds": self.flatten_builds,
            "skeleton_builds": self.skeleton_builds,
            "render_builds": self.render_builds,
            "render_cache_hits": self.render_cache_hits,
        }


def _mode_color(style: Style, shape: Shape) -> Color:
    for tag in shape.style_tags:
        if tag in style.toon_palette:
            return style.toon_palette[tag]
    return style.toon_default_color


def _fill_mask(vertices: np.ndarray, width: int, height: int, sample: int) -> np.ndarray:
    """Scanline polygon fill using the exact flattened contour coordinates."""

    if vertices.shape[0] < 3:
        return np.zeros((height * sample, width * sample), dtype=bool)
    canvas_w = width * sample
    canvas_h = height * sample
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    points = np.asarray(vertices, dtype=np.float64) * sample
    for row in range(canvas_h):
        scan_y = row + 0.5
        intersections: list[float] = []
        for index in range(points.shape[0]):
            x0, y0 = points[index]
            x1, y1 = points[(index + 1) % points.shape[0]]
            if (y0 <= scan_y < y1) or (y1 <= scan_y < y0):
                factor = (scan_y - y0) / (y1 - y0)
                intersections.append(x0 + factor * (x1 - x0))
        if len(intersections) < 2:
            continue
        intersections.sort()
        for start, stop in zip(intersections[0::2], intersections[1::2]):
            left = max(0, int(math.ceil(start - 0.5)))
            right = min(canvas_w, int(math.ceil(stop - 0.5)))
            if right > left:
                mask[row, left:right] = True
    return mask


def _stroke_mask(
    paths: list[np.ndarray], half_width: float, width: int, height: int, sample: int
) -> np.ndarray:
    canvas_w = width * sample
    canvas_h = height * sample
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    if half_width <= 0.0:
        return mask
    grid_x = (np.arange(canvas_w, dtype=np.float64) + 0.5) / sample
    grid_y = (np.arange(canvas_h, dtype=np.float64) + 0.5) / sample
    radius = max(half_width, 0.35)
    for path in paths:
        for first, second in zip(path[:-1], path[1:]):
            x0, y0 = float(first[0]), float(first[1])
            x1, y1 = float(second[0]), float(second[1])
            vx = x1 - x0
            vy = y1 - y0
            length_sq = vx * vx + vy * vy
            x_min = max(0, int(math.floor((min(x0, x1) - radius) * sample)))
            x_max = min(canvas_w, int(math.ceil((max(x0, x1) + radius) * sample)) + 1)
            y_min = max(0, int(math.floor((min(y0, y1) - radius) * sample)))
            y_max = min(canvas_h, int(math.ceil((max(y0, y1) + radius) * sample)) + 1)
            if x_max <= x_min or y_max <= y_min:
                continue
            xs = grid_x[x_min:x_max][None, :]
            ys = grid_y[y_min:y_max][:, None]
            if length_sq <= 1e-20:
                distance_sq = (xs - x0) ** 2 + (ys - y0) ** 2
            else:
                projection = ((xs - x0) * vx + (ys - y0) * vy) / length_sq
                projection = np.clip(projection, 0.0, 1.0)
                nearest_x = x0 + projection * vx
                nearest_y = y0 + projection * vy
                distance_sq = (xs - nearest_x) ** 2 + (ys - nearest_y) ** 2
            mask[y_min:y_max, x_min:x_max] |= distance_sq <= radius * radius
    return mask


def _downsample_rgba(rgba: np.ndarray, sample: int) -> np.ndarray:
    if sample == 1:
        return np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width, _ = rgba.shape
    blocks = rgba.reshape(height // sample, sample, width // sample, sample, 4).astype(np.float64)
    alpha = blocks[..., 3] / 255.0
    alpha_mean = alpha.mean(axis=(1, 3))
    premultiplied = blocks[..., :3] * alpha[..., None]
    rgb_mean = premultiplied.mean(axis=(1, 3))
    out = np.zeros((height // sample, width // sample, 4), dtype=np.float64)
    nonzero = alpha_mean > 1e-9
    out[..., :3][nonzero] = rgb_mean[nonzero] / alpha_mean[nonzero, None]
    out[..., 3] = alpha_mean * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _shape_style_fingerprint(style: Style, mode: str, shape: Shape | None = None) -> str:
    common = {
        "line_color": style.line_color,
        "line_opacity": style.line_opacity,
        "width_scale": style.width_scale,
    }
    if mode == "lineart":
        payload = {"mode": mode, **common}
    elif mode == "sketch":
        payload = {
            "mode": mode,
            **common,
            "sketch_ink": style.sketch_ink,
            "sketch_hatch_spacing": style.sketch_hatch_spacing,
            "sketch_hatch_angle": style.sketch_hatch_angle,
            "sketch_shade_strength": style.sketch_shade_strength,
        }
    elif mode == "toon":
        fill_color = _mode_color(style, shape) if shape is not None else style.toon_default_color
        payload = {
            "mode": mode,
            **common,
            "toon_fill_color": fill_color,
            "toon_shadow_color": style.toon_shadow_color,
            "toon_light": style.toon_light,
            "toon_light_strength": style.toon_light_strength,
        }
    else:
        raise NprError("E_RENDER_MISMATCH", "unknown render mode", mode=mode)
    from .model import canonical_hash

    return canonical_hash(payload)


class RenderEngine:
    def __init__(
        self,
        project: Project,
        config: RenderConfig | None = None,
        geometry: GeometryEngine | None = None,
    ):
        self.project = project
        self.config = config or RenderConfig()
        self.geometry = geometry or GeometryEngine(project)
        self.geometry.set_project(project)
        self._render_cache: dict[tuple[object, ...], np.ndarray] = {}
        self.render_builds = 0
        self.render_cache_hits = 0

    def set_project(self, project: Project) -> None:
        self.project = project
        self.geometry.set_project(project)

    def reset_stats(self) -> None:
        self.render_builds = 0
        self.render_cache_hits = 0

    def effective_width(self, shape: Shape) -> float:
        return float(
            shape.stroke_width
            * self.project.style.width_scale
            * self.project.stroke_sample.width_scale
        )

    def style_fingerprint(self, mode: str, shape: Shape | None = None) -> str:
        return _shape_style_fingerprint(self.project.style, mode, shape)

    def render_key(self, shape_id: str, mode: str | None = None) -> tuple[object, ...]:
        mode = mode or "lineart"
        skeleton = self.geometry.ensure(shape_id)
        shape = self.project.shape_by_id(shape_id)
        shape_fingerprint = hashlib.sha256(
            repr(tuple(shape.style_tags)).encode("utf-8")
        ).hexdigest()
        return (
            shape_id,
            mode,
            skeleton.geometry_hash(),
            skeleton.signature,
            self.style_fingerprint(mode, shape),
            round(self.effective_width(shape), 9),
            shape_fingerprint,
            self.config.width,
            self.config.height,
            self.config.supersample,
        )

    def _render_shape(self, shape: Shape, mode: str) -> np.ndarray:
        skeleton = self.geometry.ensure(shape.id)
        flattened = self.geometry.flattened(shape.id)
        sample = self.config.supersample
        width = self.config.width * sample
        height = self.config.height * sample
        paths = skeleton.paths()
        half_width = self.effective_width(shape) * 0.5
        outline = _stroke_mask(
            paths, half_width, self.config.width, self.config.height, sample
        )
        style = self.project.style
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        alpha = np.zeros((height, width), dtype=np.uint8)

        if mode in ("sketch", "toon"):
            fill = np.zeros((height, width), dtype=bool)
            for contour in flattened.contours:
                if contour.closed:
                    fill |= _fill_mask(
                        contour.vertices, self.config.width, self.config.height, sample
                    )
        else:
            fill = np.zeros((height, width), dtype=bool)

        if mode == "toon":
            color = _mode_color(style, shape)
            rgb[fill] = np.asarray(color, dtype=np.uint8)
            alpha[fill] = 255
            if fill.any():
                grid_x = (np.arange(width, dtype=np.float64) + 0.5) / sample
                grid_y = (np.arange(height, dtype=np.float64) + 0.5) / sample
                xx, yy = np.meshgrid(grid_x, grid_y)
                light_x, light_y = style.toon_light
                projection = xx * light_x + yy * light_y
                if projection[fill].size:
                    threshold = float(np.quantile(projection[fill], 0.52))
                    shadow = fill & (projection >= threshold)
                    strength = style.toon_light_strength
                    base = np.asarray(color, dtype=np.float64)
                    shade = np.asarray(style.toon_shadow_color, dtype=np.float64)
                    blended = np.clip(
                        np.rint(base * (1.0 - strength) + shade * strength), 0, 255
                    ).astype(np.uint8)
                    rgb[shadow] = blended
        elif mode == "sketch":
            if fill.any():
                grid_x = (np.arange(width, dtype=np.float64) + 0.5) / sample
                grid_y = (np.arange(height, dtype=np.float64) + 0.5) / sample
                xx, yy = np.meshgrid(grid_x, grid_y)
                angle = math.radians(style.sketch_hatch_angle)
                coordinate = xx * math.cos(angle) + yy * math.sin(angle)
                spacing = max(style.sketch_hatch_spacing, 0.5)
                stripe_width = max(0.5, spacing * 0.16)
                hatch = np.mod(coordinate, spacing) <= stripe_width
                hatch &= fill
                rgb[hatch] = np.asarray(style.sketch_ink, dtype=np.uint8)
                alpha[hatch] = int(round(style.sketch_shade_strength * 255.0))

        outline_color = style.sketch_ink if mode == "sketch" else style.line_color
        opacity = style.line_opacity * (0.92 if mode == "sketch" else 1.0)
        rgb[outline] = np.asarray(outline_color, dtype=np.uint8)
        alpha[outline] = np.maximum(alpha[outline], int(round(opacity * 255.0)))
        rgba = np.dstack((rgb, alpha))
        return _downsample_rgba(rgba, sample)

    def _get_shape_image(self, shape: Shape, mode: str) -> tuple[np.ndarray, str]:
        key = self.render_key(shape.id, mode)
        cached = self._render_cache.get(key)
        if cached is not None:
            self.render_cache_hits += 1
            return cached, hashlib.sha256(cached.tobytes(order="C")).hexdigest()
        image = self._render_shape(shape, mode)
        image.setflags(write=False)
        self._render_cache[key] = image
        self.render_builds += 1
        return image, hashlib.sha256(image.tobytes(order="C")).hexdigest()

    def export_cache(self, mode: str) -> dict[str, dict[str, object]]:
        """Export reusable raw RGBA shape images and their exact cache keys."""

        exported: dict[str, dict[str, object]] = {}
        for shape in self.project.ordered_shapes():
            key = self.render_key(shape.id, mode)
            image = self._render_cache.get(key)
            if image is None:
                continue
            exported[shape.id] = {
                "cache_key": list(key),
                "rgba": image.tobytes(order="C"),
                "render_hash": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
            }
        return exported

    def render(self, mode: RenderMode = "lineart") -> RenderResult:
        require(mode in RENDER_MODES, "E_RENDER_MISMATCH", "unknown render mode", mode=mode)
        base = np.zeros((self.config.height, self.config.width, 4), dtype=np.uint8)
        paper = self.project.style.paper_color
        base[..., :3] = np.asarray(paper, dtype=np.uint8)
        base[..., 3] = 255
        shape_hashes: dict[str, str] = {}
        geometry_hashes: dict[str, str] = {}
        for shape in self.project.ordered_shapes():
            image, digest = self._get_shape_image(shape, mode)
            shape_hashes[shape.id] = digest
            geometry_hashes[shape.id] = self.geometry.ensure(shape.id).geometry_hash()
            source_alpha = image[..., 3:4].astype(np.float64) / 255.0
            dest_alpha = base[..., 3:4].astype(np.float64) / 255.0
            out_alpha = source_alpha + dest_alpha * (1.0 - source_alpha)
            numerator = (
                image[..., :3].astype(np.float64) * source_alpha
                + base[..., :3].astype(np.float64)
                * dest_alpha
                * (1.0 - source_alpha)
            )
            base[..., :3] = np.clip(
                np.rint(numerator / np.maximum(out_alpha, 1e-12)), 0, 255
            ).astype(np.uint8)
            base[..., 3:4] = np.clip(np.rint(out_alpha * 255.0), 0, 255).astype(np.uint8)
        png_bytes = encode_png(base)
        hashes = image_hashes(base, png_bytes)
        return RenderResult(
            mode=mode,
            image=base,
            shape_hashes=shape_hashes,
            geometry_hashes=geometry_hashes,
            pixel_sha256=hashes["pixel_sha256"],
            png_sha256=hashes["png_sha256"],
            flatten_builds=self.geometry.flatten_builds,
            skeleton_builds=self.geometry.skeleton_builds,
            render_builds=self.render_builds,
            render_cache_hits=self.render_cache_hits,
        )


def verify_render_hashes(expected: dict[str, str], actual: dict[str, str]) -> None:
    if expected != actual:
        changed = sorted(
            key
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        )
        raise NprError(
            "E_RENDER_MISMATCH",
            "render hashes do not match the committed manifest",
            changed_shapes=changed,
        )


def restore_render_cache(
    engine: RenderEngine,
    mode: str,
    keys: dict[str, list[object]],
    images: dict[str, bytes],
    expected_hashes: dict[str, str],
) -> list[str]:
    restored: list[str] = []
    for shape_id, raw in images.items():
        if shape_id not in keys:
            continue
        expected_key = engine.render_key(shape_id, mode)
        stored_key = tuple(keys[shape_id])
        if stored_key != expected_key:
            continue
        array = np.frombuffer(raw, dtype=np.uint8)
        expected_size = engine.config.width * engine.config.height * 4
        if array.size != expected_size:
            raise NprError(
                "E_RENDER_MISMATCH",
                "cached shape image size does not match render config",
                shape=shape_id,
            )
        image = array.reshape(engine.config.height, engine.config.width, 4).copy()
        digest = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        if expected_hashes.get(shape_id) != digest:
            raise NprError(
                "E_RENDER_MISMATCH",
                "cached shape image hash mismatch",
                shape=shape_id,
            )
        image.setflags(write=False)
        engine._render_cache[expected_key] = image
        restored.append(shape_id)
    return restored
