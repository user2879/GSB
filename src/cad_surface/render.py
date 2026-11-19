"""Small deterministic software rasterizer used as the offline display path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .canonical import atomic_write_bytes, sha256_bytes
from .geometry import PatchGeometry
from .model import Camera, NURBSModel


DISPLAY_MODES = ("control_net", "shaded_surface", "isoparametric")


@dataclass(frozen=True)
class RenderBuffer:
    mode: str
    width: int
    height: int
    rgb: np.ndarray
    depth: np.ndarray
    content_hash: str

    @classmethod
    def create(cls, mode: str, rgb: np.ndarray, depth: np.ndarray) -> "RenderBuffer":
        rgb_u8 = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_f8 = np.ascontiguousarray(depth, dtype="<f8")
        content_hash = sha256_bytes(
            b"render-buffer.v1\0"
            + mode.encode("ascii")
            + np.asarray(rgb_u8.shape, dtype="<i8").tobytes()
            + rgb_u8.tobytes(order="C")
            + depth_f8.tobytes(order="C")
        )
        return cls(mode=mode, width=int(rgb_u8.shape[1]), height=int(rgb_u8.shape[0]), rgb=rgb_u8, depth=depth_f8, content_hash=content_hash)

    def to_ppm_bytes(self) -> bytes:
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        return header + self.rgb.tobytes(order="C")


class SoftwareRenderer:
    def __init__(self, model: NURBSModel, geometries: Mapping[str, PatchGeometry]) -> None:
        self.model = model
        self.geometries = geometries
        self.camera = model.camera
        self.width, self.height = self.camera.viewport
        self._setup_camera()
        self._material_colors = {
            patch.key: np.asarray(patch.material.base_color, dtype="<f8") for patch in model.patches
        }
        self._materials = {patch.key: patch.material for patch in model.patches}

    def _setup_camera(self) -> None:
        eye = np.asarray(self.camera.eye, dtype="<f8")
        target = np.asarray(self.camera.target, dtype="<f8")
        up = np.asarray(self.camera.up, dtype="<f8")
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        self.eye = eye
        self.forward = forward
        self.right = right
        self.true_up = true_up
        self.focal = 1.0 / np.tan(np.deg2rad(self.camera.fov_y_degrees) * 0.5)
        light = np.asarray(self.camera.light_direction, dtype="<f8")
        self.light = light / np.linalg.norm(light)

    def _background(self) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[:] = np.asarray(self.camera.background, dtype=np.uint8)
        depth = np.full((self.height, self.width), np.inf, dtype="<f8")
        return rgb, depth

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative = points - self.eye
        camera_x = relative @ self.right
        camera_y = relative @ self.true_up
        camera_z = relative @ self.forward
        safe_z = np.maximum(camera_z, 1e-12)
        aspect = self.width / float(self.height)
        ndc_x = camera_x * self.focal / (safe_z * aspect)
        ndc_y = camera_y * self.focal / safe_z
        screen_x = (ndc_x + 1.0) * 0.5 * (self.width - 1)
        screen_y = (1.0 - ndc_y) * 0.5 * (self.height - 1)
        return screen_x, screen_y, camera_z

    def _shade(self, normals: np.ndarray, material_key: str) -> np.ndarray:
        material = self._materials[material_key]
        base = self._material_colors[material_key]
        n = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-15)
        lambert = np.abs(n @ self.light)
        half = self.light + np.array([0.0, 0.0, 1.0], dtype="<f8")
        half = half / np.linalg.norm(half)
        specular = np.power(np.clip(np.abs(n @ half), 0.0, 1.0), material.shininess)
        intensity = material.ambient + material.diffuse * lambert + material.specular * specular
        color = base[None, None, :] * intensity[..., None]
        return np.clip(color * 255.0, 0.0, 255.0).astype(np.uint8)

    def render_shaded_surface(self) -> RenderBuffer:
        rgb, depth = self._background()
        for patch_key in sorted(self.geometries):
            geometry = self.geometries[patch_key]
            vertices = geometry.points.reshape(geometry.vertex_count, 3)
            normals = geometry.normals.reshape(geometry.vertex_count, 3)
            sx, sy, z = self._project(vertices)
            for triangle in geometry.triangles:
                indices = triangle.astype(np.int64)
                z_values = z[indices]
                if np.any(z_values <= self.camera.near) or np.any(z_values >= self.camera.far):
                    continue
                x_values = sx[indices]
                y_values = sy[indices]
                min_x = max(0, int(np.floor(np.min(x_values))))
                max_x = min(self.width - 1, int(np.ceil(np.max(x_values))))
                min_y = max(0, int(np.floor(np.min(y_values))))
                max_y = min(self.height - 1, int(np.ceil(np.max(y_values))))
                if min_x > max_x or min_y > max_y:
                    continue
                x0, x1, x2 = x_values
                y0, y1, y2 = y_values
                denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(float(denominator)) < 1e-15:
                    continue
                xs = np.arange(min_x, max_x + 1, dtype="<f8") + 0.5
                ys = np.arange(min_y, max_y + 1, dtype="<f8") + 0.5
                grid_x, grid_y = np.meshgrid(xs, ys)
                b0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
                b1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
                b2 = 1.0 - b0 - b1
                inside = (b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
                if not np.any(inside):
                    continue
                tri_depth = b0 * z_values[0] + b1 * z_values[1] + b2 * z_values[2]
                depth_window = depth[min_y : max_y + 1, min_x : max_x + 1]
                update = inside & (tri_depth > self.camera.near) & (tri_depth < depth_window)
                if not np.any(update):
                    continue
                tri_normals = normals[indices]
                interpolated = (
                    b0[..., None] * tri_normals[0]
                    + b1[..., None] * tri_normals[1]
                    + b2[..., None] * tri_normals[2]
                )
                shaded = self._shade(interpolated, patch_key)
                rgb_window = rgb[min_y : max_y + 1, min_x : max_x + 1]
                depth_window[update] = tri_depth[update]
                rgb_window[update] = shaded[update]
        return RenderBuffer.create("shaded_surface", rgb, depth)

    def _draw_line_segments(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        vertices: np.ndarray,
        segments: list[tuple[int, int]],
        colors: np.ndarray,
    ) -> None:
        sx, sy, z = self._project(vertices)
        for segment_index, (a, b) in enumerate(segments):
            if z[a] <= self.camera.near or z[b] <= self.camera.near:
                continue
            xa, ya, xb, yb = float(sx[a]), float(sy[a]), float(sx[b]), float(sy[b])
            steps = max(1, int(np.ceil(max(abs(xb - xa), abs(yb - ya)))) + 1)
            values = np.linspace(0.0, 1.0, steps + 1, dtype="<f8")
            xs = np.rint(xa + (xb - xa) * values).astype(np.int64)
            ys = np.rint(ya + (yb - ya) * values).astype(np.int64)
            zs = z[a] + (z[b] - z[a]) * values
            color = colors[segment_index % len(colors)]
            for x, y, value_z in zip(xs, ys, zs):
                if 0 <= x < self.width and 0 <= y < self.height and value_z < depth[y, x]:
                    depth[y, x] = value_z
                    rgb[y, x] = color

    def render_control_net(self) -> RenderBuffer:
        rgb, depth = self._background()
        for patch in self.model.patches:
            points = patch.control_points.reshape(-1, 3)
            height, width = patch.control_shape
            segments: list[tuple[int, int]] = []
            for i in range(height):
                for j in range(width - 1):
                    segments.append((i * width + j, i * width + j + 1))
            for j in range(width):
                for i in range(height - 1):
                    segments.append((i * width + j, (i + 1) * width + j))
            material_color = np.clip(
                np.asarray(patch.material.base_color, dtype="<f8") * 0.55 + 0.45,
                0.0,
                1.0,
            )
            self._draw_line_segments(
                rgb,
                depth,
                points,
                segments,
                np.rint(material_color * 255.0).astype(np.uint8)[None, :],
            )
        return RenderBuffer.create("control_net", rgb, depth)

    def render_isoparametric(self) -> RenderBuffer:
        rgb, depth = self._background()
        for patch_key in sorted(self.geometries):
            geometry = self.geometries[patch_key]
            vertices = geometry.points.reshape(geometry.vertex_count, 3)
            u_segments, v_segments = geometry.iso_segments()
            base = self._material_colors[patch_key]
            u_color = np.clip(base * 0.45 + 0.55, 0.0, 1.0)
            v_color = np.array([0.92, 0.97, 1.0], dtype="<f8")
            colors = np.stack(
                [
                    np.rint(u_color * 255.0).astype(np.uint8),
                    np.rint(v_color * 255.0).astype(np.uint8),
                ],
                axis=0,
            )
            self._draw_line_segments(rgb, depth, vertices, u_segments, colors[:1])
            self._draw_line_segments(rgb, depth, vertices, v_segments, colors[1:])
        return RenderBuffer.create("isoparametric", rgb, depth)

    def render(self, mode: str) -> RenderBuffer:
        if mode == "control_net":
            return self.render_control_net()
        if mode == "shaded_surface":
            return self.render_shaded_surface()
        if mode == "isoparametric":
            return self.render_isoparametric()
        raise ValueError(f"unknown display mode: {mode}")

    def render_all(self) -> dict[str, RenderBuffer]:
        return {mode: self.render(mode) for mode in DISPLAY_MODES}


def display_hashes(buffers: Mapping[str, RenderBuffer]) -> dict[str, str]:
    return {mode: buffers[mode].content_hash for mode in DISPLAY_MODES if mode in buffers}


def write_ppm(path: Path, buffer: RenderBuffer) -> None:
    atomic_write_bytes(path, buffer.to_ppm_bytes())


def render_report(buffers: Mapping[str, RenderBuffer]) -> dict[str, Any]:
    return {
        mode: {
            "hash": buffer.content_hash,
            "width": buffer.width,
            "height": buffer.height,
        }
        for mode, buffer in buffers.items()
    }
