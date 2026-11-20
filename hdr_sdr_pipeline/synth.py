"""Deterministic synthetic HDR frame sequence generator and truth manifest.

Spec (frozen for reproducibility):
  * 64 frames of 960x540 linear-light RGB, authored in BT.709 primaries.
  * Content: HDR highlight window with a brighter specular sub-block,
    a horizontal colored linear-gradient banner, and a deep shadow band
    (including a small deterministic negative-value sensor block).
  * Randomness comes from a single ``numpy.random.default_rng(seed)``
    (PCG64); every draw order is fixed, so identical seeds rebuild
    byte-identical frames.
  * All arithmetic is performed in float32; the final conversion to
    16-bit half float uses IEEE 754 binary16 round-to-nearest-even
    (``numpy.ndarray.astype(np.float16)``) and values are stored as
    little-endian ``<f2`` raw bytes. The reverse conversion fp16 -> float32
    is exact (binary16 is a subset of binary32).
"""

import json
import os

import numpy as np

from .colorspace import luma
from .errors import FrameFormatError
from .hashing import hash_array, hash_json

GENERATOR_SEED = 20240517
FRAME_COUNT = 64
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Full-resolution (960x540) region bounds, inclusive start / exclusive end.
# Geometry is scaled proportionally for smaller test canvases; at full
# resolution the scaled values equal these constants exactly.
FULL_REGIONS = {
    "highlight": (90, 300, 120, 360),
    "specular": (90, 150, 120, 200),
    "banner": (360, 470, 100, 860),
    "shadow": (480, 540, 0, 960),
    "negative": (480, 520, 40, 100),
}
REGION_HIGHLIGHT = FULL_REGIONS["highlight"]
REGION_SPECULAR = FULL_REGIONS["specular"]
REGION_BANNER = FULL_REGIONS["banner"]
REGION_SHADOW = FULL_REGIONS["shadow"]
REGION_NEGATIVE = FULL_REGIONS["negative"]


def scaled_regions(height, width):
    """Scale full-res region bounds to a target canvas, keeping min size 1."""
    def sc(bounds):
        r0, r1, c0, c1 = bounds
        out = (
            min(height - 1, round(r0 * height / FRAME_HEIGHT)),
            min(height, round(r1 * height / FRAME_HEIGHT)),
            min(width - 1, round(c0 * width / FRAME_WIDTH)),
            min(width, round(c1 * width / FRAME_WIDTH)),
        )
        r0, r1, c0, c1 = out
        r1 = max(r1, min(height, r0 + 1))
        c1 = max(c1, min(width, c0 + 1))
        return (r0, r1, c0, c1)

    return {name: sc(bounds) for name, bounds in FULL_REGIONS.items()}

# Fixed truth sampling points: (row, col), spread across every region.
SAMPLE_POINTS = [
    (10, 10),
    (95, 130),      # highlight
    (120, 160),     # specular HDR core
    (290, 350),     # highlight edge
    (400, 120),     # banner left
    (420, 480),     # banner middle
    (450, 840),     # banner right
    (500, 30),      # shadow
    (505, 60),      # negative block
    (530, 900),     # shadow far right
    (330, 500),     # mid background
    (200, 700),     # background right
]

FRAME_NAME_TEMPLATE = "frame_{:04d}.fp16"


def _base_canvas(height, width, rng, regions):
    """Build the time-invariant float32 scene before per-frame modulation."""
    canvas = np.empty((height, width, 3), dtype=np.float32)

    # Deep, near-black shadow background with faint positive noise.
    noise = rng.random((height, width, 1), dtype=np.float32)
    canvas[:] = np.float32(0.02) + noise * np.float32(0.01)

    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]

    # HDR highlight window (~3.5 linear) with specular sub-block (~12).
    r0, r1, c0, c1 = regions["highlight"]
    canvas[r0:r1, c0:c1] = np.float32(3.5)
    s0, s1, sc0, sc1 = regions["specular"]
    canvas[s0:s1, sc0:sc1] = np.float32(12.0)

    # Colored linear gradient banner (all channels vary linearly with x).
    b0, b1, bc0, bc1 = regions["banner"]
    t = (cols[:, bc0:bc1] - np.float32(bc0)) / np.float32(bc1 - bc0)
    banner = np.empty((b1 - b0, bc1 - bc0, 3), dtype=np.float32)
    banner[..., 0] = np.float32(0.10) + t * np.float32(1.50)
    banner[..., 1] = np.float32(0.05) + t * np.float32(0.85)
    banner[..., 2] = np.float32(0.80) - t * np.float32(0.60)
    canvas[b0:b1, bc0:bc1] = banner

    # Deep shadow band with deterministic faint sinusoidal texture.
    sh0, sh1, _, _ = regions["shadow"]
    pattern = np.sin(rows[sh0:sh1, :] * np.float32(0.35)) * np.float32(0.004)
    canvas[sh0:sh1] = np.float32(0.012) + pattern[..., None]

    # Deterministic negative sensor block (exercises negative handling).
    n0, n1, nc0, nc1 = regions["negative"]
    canvas[n0:n1, nc0:nc1] = np.float32(-0.01)
    return canvas


def _modulate(base, frame_index, regions):
    """Apply the fixed per-frame exposure-like modulation in float32."""
    phase = np.float32(2.0 * np.pi * frame_index / 16.0)
    factor = np.float32(1.0) + np.float32(0.25) * np.sin(phase)
    frame = base * factor

    # Highlight window pulses on a slower phase.
    pulse = np.float32(1.0) + np.float32(0.20) * np.cos(
        np.float32(2.0 * np.pi * frame_index / 24.0)
    )
    r0, r1, c0, c1 = regions["highlight"]
    frame[r0:r1, c0:c1] = frame[r0:r1, c0:c1] * pulse
    return frame


def scaled_sample_points(height, width):
    pts = []
    for r, c in SAMPLE_POINTS:
        rr = min(height - 1, round(r * height / FRAME_HEIGHT))
        cc = min(width - 1, round(c * width / FRAME_WIDTH))
        pts.append((rr, cc))
    return pts


def _frame_stats(frame_fp32):
    """Luma stats with fixed row-major float64 accumulation."""
    y = luma(frame_fp32, "bt709")
    peak = float(np.max(y))
    mean = float(np.sum(y, dtype=np.float64) / float(y.size))
    return peak, mean


def generate_sequence(out_dir, *, frames=FRAME_COUNT, width=FRAME_WIDTH,
                      height=FRAME_HEIGHT, seed=GENERATOR_SEED):
    """Generate the frame sequence and truth manifest under ``out_dir``.

    Returns the manifest dictionary. Writes:
        out_dir/frames/frame_0000.fp16 ... raw little-endian binary16
        out_dir/truth.json              canonical truth manifest
    """
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    regions = scaled_regions(height, width)
    base = _base_canvas(height, width, rng, regions)

    manifest = {
        "generator": {
            "seed": seed,
            "rng": "numpy.random.default_rng / PCG64",
            "numpy": np.__version__,
            "frame_count": frames,
            "width": width,
            "height": height,
            "authoring_primaries": "BT.709 linear",
            "storage_dtype": "<f2 (IEEE 754 binary16, little-endian)",
            "arithmetic_dtype": "float32",
            "fp16_rounding": "round to nearest, ties to even (astype float16)",
            "fp16_to_float32": "exact widening, binary16 is a subset of binary32",
        },
        "regions": {name: list(bounds) for name, bounds in regions.items()},
        "sample_points": [list(p) for p in scaled_sample_points(height, width)],
        "frames": [],
    }

    for idx in range(frames):
        fp32 = _modulate(base, idx, regions)
        fp16_le = fp32.astype(np.float16).astype("<f2")
        frame_path = os.path.join(frames_dir, FRAME_NAME_TEMPLATE.format(idx))
        with open(frame_path, "wb") as fh:
            fh.write(np.ascontiguousarray(fp16_le).tobytes(order="C"))

        # Truth is derived from the stored representation (fp16 round trip),
        # guaranteeing manifest and on-disk frames describe the same data.
        stored = load_frame(frame_path, width=width, height=height)
        peak, mean = _frame_stats(stored)
        samples = [
            {
                "row": int(r),
                "col": int(c),
                "rgb": [float(v) for v in stored[r, c]],
            }
            for (r, c) in scaled_sample_points(height, width)
        ]
        manifest["frames"].append(
            {
                "frame": idx,
                "file": FRAME_NAME_TEMPLATE.format(idx),
                "peak_luma": peak,
                "mean_luma": mean,
                "frame_hash": hash_array(stored),
                "samples": samples,
            }
        )

    manifest["manifest_hash"] = hash_json(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True))
    return manifest


def load_frame(path, *, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Load a raw little-endian fp16 frame and validate its exact shape."""
    with open(path, "rb") as fh:
        raw = fh.read()
    expected = width * height * 3 * 2
    if len(raw) != expected:
        raise FrameFormatError(
            f"frame {path!r} has {len(raw)} bytes, expected {expected}",
            details={"path": path, "expected_bytes": expected,
                     "actual_bytes": len(raw)},
        )
    try:
        fp16 = np.frombuffer(raw, dtype="<f2").reshape(height, width, 3)
    except ValueError as exc:
        raise FrameFormatError(str(exc), details={"path": path}) from exc
    return fp16.astype(np.float32)
