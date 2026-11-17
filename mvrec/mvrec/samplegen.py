"""Fixed-seed synthetic multi-view sample generation.

The scene is a set of bright square texture markers placed at deterministic
3D positions.  Each view renders the markers through a pinhole camera so
that feature extraction, matching and triangulation all have real geometry
to work on while remaining fully reproducible and CPU-only.
"""

import argparse
import json
import os

import numpy as np

from .pngio import write_png

SEED = 20261118
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
FOCAL = 360.0


def build_scene(marker_count=24):
    rng = np.random.default_rng(SEED)
    markers = []
    grid = int(np.ceil(np.sqrt(marker_count)))
    for index in range(marker_count):
        layer = index // 8
        col = index % 8
        x = (col - 3.5) * 0.18
        y = ((index % grid) - (grid - 1) / 2.0) * 0.16
        z = 0.45 + layer * 0.22
        markers.append(np.array([x, y, z], dtype=np.float64))
    intensities = rng.integers(160, 255, size=marker_count).astype(np.uint8)
    return markers, intensities


def camera_pose(view_id, view_count):
    angle = 2.0 * np.pi * view_id / view_count
    radius = 1.2
    center = np.array([0.0, 0.0, 0.8])
    position = np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        0.8 + 0.15 * np.sin(2.0 * angle),
    ])
    forward = center - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # OpenCV-style camera frame: x=right, y=down, z=forward (into the scene).
    R = np.stack([right, -up, forward])
    t = -R @ position
    return R, t, position


def render_view(view_id, view_count, markers, intensities, rng):
    R, t, position = camera_pose(view_id, view_count)
    K = np.array([
        [FOCAL, 0.0, IMAGE_WIDTH / 2.0],
        [0.0, FOCAL, IMAGE_HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ])
    gray = rng.integers(8, 24, size=(IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.uint8)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    for marker_index, (marker, intensity) in enumerate(zip(markers, intensities)):
        projected = P @ np.append(marker, 1.0)
        if projected[2] <= 0.05:
            continue
        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        col, row = int(round(u)), int(round(v))
        half = 4
        r0, r1 = max(0, row - half), min(IMAGE_HEIGHT, row + half + 1)
        c0, c1 = max(0, col - half), min(IMAGE_WIDTH, col + half + 1)
        # Bright outer border keeps the marker one connected component; the
        # inner 5x5 pattern encodes the marker id so descriptors are unique.
        pattern_rng = np.random.default_rng(SEED + marker_index + 1)
        inner_pattern = pattern_rng.integers(0, 2, size=(7, 7))
        dark = max(40, int(intensity) // 3)
        for ry in range(r0, r1):
            for rx in range(c0, c1):
                ly, lx = ry - r0, rx - c0
                border = ly in (0, 8) or lx in (0, 8)
                if border:
                    gray[ry, rx] = intensity
                else:
                    bit = inner_pattern[ly - 1, lx - 1]
                    gray[ry, rx] = intensity if bit else dark
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    camera = {
        "view_id": view_id,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "K": K.tolist(),
        "R": R.tolist(),
        "t": [round(float(v), 9) for v in t],
        "position": [round(float(v), 9) for v in position],
    }
    return gray, rgb, camera


def generate_sample(output_dir, view_count=8, corrupt_views=None, missing_camera_views=None):
    corrupt_views = set(corrupt_views or [])
    missing_camera_views = set(missing_camera_views or [])
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    markers, intensities = build_scene()
    rng = np.random.default_rng(SEED)
    cameras = []
    for view_id in range(view_count):
        gray, rgb, camera = render_view(view_id, view_count, markers, intensities, rng)
        image_path = os.path.join(image_dir, "view_%03d.png" % view_id)
        write_png(image_path, rgb)
        if view_id in corrupt_views:
            with open(image_path, "r+b") as handle:
                handle.seek(-20, os.SEEK_END)
                handle.write(b"\x00" * 8)
        if view_id not in missing_camera_views:
            cameras.append(camera)
    with open(os.path.join(output_dir, "cameras.json"), "w", encoding="utf-8") as handle:
        json.dump(cameras, handle, indent=2, sort_keys=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="generate deterministic synthetic views")
    parser.add_argument("output_dir")
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    generate_sample(args.output_dir, args.views)
    print("generated %d views under %s" % (args.views, args.output_dir))


if __name__ == "__main__":
    main()
