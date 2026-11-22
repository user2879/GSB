"""Deterministic synthetic multi-label image generator.

Each image is produced by an isolated ``numpy.random.Generator`` seeded from a
global base seed plus the sample index, so outputs are reproducible byte-for-byte
without touching NumPy's global random state. Geometric primitives map one to
one onto labels; co-occurrence, occlusion and pixel noise are all covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# Fixed primitive palette and shapes; label i always uses SHAPES[i % len].
SHAPES = ("rectangle", "ellipse", "triangle")
PALETTE = (
    (214, 69, 65),
    (46, 134, 222),
    (39, 174, 96),
    (241, 196, 15),
    (142, 68, 173),
    (230, 126, 34),
    (26, 188, 156),
    (192, 57, 43),
)
BACKGROUND = (245, 245, 242)


def _rng(base_seed: int, index: int) -> np.random.Generator:
    seed_seq = np.random.SeedSequence([int(base_seed), int(index)])
    return np.random.default_rng(seed_seq)


def _draw_rectangle(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color):
    canvas[y0:y1, x0:x1] = color


def _draw_ellipse(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color):
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = (x0 + x1 - 1) / 2.0
    cy = (y0 + y1 - 1) / 2.0
    rx = max((x1 - x0) / 2.0, 1e-9)
    ry = max((y1 - y0) / 2.0, 1e-9)
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    canvas[y0:y1, x0:x1][mask] = color


def _draw_triangle(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color):
    # Bounding-box rasterization with a deterministic half-plane test.
    points = np.asarray(
        [(x0, y1 - 1), (x1 - 1, y1 - 1), ((x0 + x1) // 2, y0)], dtype=np.int64
    )
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = xx.astype(np.int64)
    py = yy.astype(np.int64)
    d1 = (px - points[1, 0]) * (points[0, 1] - points[1, 1]) - (
        points[0, 0] - points[1, 0]
    ) * (py - points[1, 1])
    d2 = (px - points[2, 0]) * (points[1, 1] - points[2, 1]) - (
        points[1, 0] - points[2, 0]
    ) * (py - points[2, 1])
    d3 = (px - points[0, 0]) * (points[2, 1] - points[0, 1]) - (
        points[2, 0] - points[0, 0]
    ) * (py - points[0, 1])
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    mask = ~(has_neg & has_pos)
    canvas[y0:y1, x0:x1][mask] = color


_DRAWERS = {
    "rectangle": _draw_rectangle,
    "ellipse": _draw_ellipse,
    "triangle": _draw_triangle,
}


def generate_sample(
    sample_name: str,
    index: int,
    labels: tuple[str, ...],
    positive_ratios: dict[str, float],
    *,
    width: int = 160,
    height: int = 120,
    base_seed: int = 20261123,
    variant: int = 0,
) -> tuple[bytes, dict]:
    """Generate one deterministic (png_bytes, annotation) pair."""
    rng = _rng(base_seed + variant, index)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = BACKGROUND

    present: list[str] = []
    boxes: dict[str, list[int]] = {}
    # A separate per-label Bernoulli draw realizes the configured positive ratio.
    for label_idx, label in enumerate(labels):
        u = float(rng.random())
        if u >= float(positive_ratios.get(label, 0.5)):
            continue
        shape = SHAPES[label_idx % len(SHAPES)]
        color = PALETTE[label_idx % len(PALETTE)]
        primitive_w = int(rng.integers(18, max(19, width // 3)))
        primitive_h = int(rng.integers(18, max(19, height // 3)))
        x0 = int(rng.integers(0, width - primitive_w))
        y0 = int(rng.integers(0, height - primitive_h))
        x1 = x0 + primitive_w
        y1 = y0 + primitive_h
        # Later primitives overwrite earlier ones, producing deterministic
        # occlusion when bounding boxes overlap.
        _DRAWERS[shape](canvas, x0, y0, x1, y1, color)
        present.append(label)
        boxes[label] = [x0, y0, x1, y1]

    # Deterministic salt-and-pepper speckle noise on a fixed pixel subset.
    noise_count = int(0.006 * width * height)
    flat = canvas.reshape(-1, 3)
    noise_idx = rng.integers(0, flat.shape[0], size=noise_count)
    speckle = rng.integers(0, 256, size=(noise_count, 3), dtype=np.int64)
    flat[noise_idx] = speckle.astype(np.uint8)

    image = Image.fromarray(canvas, mode="RGB")
    png_path = Path(f"{sample_name}.png")
    # Default Pillow PNG encoder is deterministic for identical pixel input.
    png_bytes = _encode_png(image)
    annotation = {
        "image": png_path.name,
        "labels": present,
        "boxes": {label: boxes[label] for label in present},
        "width": width,
        "height": height,
        "seed": int(base_seed + variant),
        "index": int(index),
    }
    return png_bytes, annotation


def _encode_png(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def write_sample(
    output_dir: str | Path,
    sample_name: str,
    png_bytes: bytes,
    annotation: dict,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{sample_name}.png"
    annotation_path = output_dir / f"{sample_name}.json"
    image_path.write_bytes(png_bytes)
    annotation_path.write_text(
        json.dumps(annotation, sort_keys=True, indent=2), encoding="utf-8"
    )
    return image_path, annotation_path


def generate_dataset(
    output_dir: str | Path,
    config,
    sample_count: int,
    *,
    name_prefix: str = "sample",
    variant: int = 0,
) -> list[str]:
    """Generate a complete dataset directory; return ordered sample names."""
    names: list[str] = []
    width, height = config.image_size
    for index in range(sample_count):
        name = f"{name_prefix}_{index:03d}"
        png_bytes, annotation = generate_sample(
            name,
            index,
            tuple(config.labels),
            dict(config.positive_ratios),
            width=width,
            height=height,
            base_seed=config.seed,
            variant=variant,
        )
        write_sample(output_dir, name, png_bytes, annotation)
        names.append(name)
    return names
