from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorama_pipeline.codec import V1_HEADER, V1_MAGIC, V1_RECORD, V1_RUN, V1_VERSION
from panorama_pipeline.io_utils import atomic_write_json, sha256_file
from panorama_pipeline.png_codec import encode_panorama_png
from panorama_pipeline.rle import mask_to_runs


SEED = 20261120
WIDTH = 64
HEIGHT = 48


def rectangle(width: int, height: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def ellipse(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
        <= 1.0
    )


def shifted(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(mask, dy, axis=0), dx, axis=1)


def boundary_masks() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(WIDTH * HEIGHT, dtype=bool)
    right = np.zeros(WIDTH * HEIGHT, dtype=bool)
    left[:30] = True
    right[:15] = True
    right[45:60] = True
    return left.reshape(HEIGHT, WIDTH), right.reshape(HEIGHT, WIDTH)


def build_v1_masks() -> dict[int, tuple[int, np.ndarray]]:
    boundary_left, _ = boundary_masks()
    return {
        1: (1, rectangle(WIDTH, HEIGHT, 2, 2, 20, 18)),
        2: (2, boundary_left),
        3: (3, ellipse(WIDTH, HEIGHT, 18, 30, 12, 8)),
        4: (1, rectangle(WIDTH, HEIGHT, 45, 25, 60, 43)),
        5: (2, ellipse(WIDTH, HEIGHT, 50, 12, 7, 7)),
        6: (4, ellipse(WIDTH, HEIGHT, 35, 35, 8, 5)),
        7: (3, rectangle(WIDTH, HEIGHT, 0, 41, 64, 48)),
        8: (4, rectangle(WIDTH, HEIGHT, 58, 8, 62, 13)),
    }


def build_v2_masks(v1_masks: dict[int, tuple[int, np.ndarray]]) -> dict[int, tuple[int, np.ndarray]]:
    _, boundary_right = boundary_masks()
    return {
        1: (1, shifted(v1_masks[1][1], 1, 0)),
        2: (2, boundary_right),
        3: (3, shifted(v1_masks[3][1], 0, 1)),
        4: (1, rectangle(WIDTH, HEIGHT, 46, 26, 60, 42)),
        5: (2, shifted(v1_masks[5][1], 1, 1)),
        6: (4, v1_masks[6][1]),
        7: (3, shifted(v1_masks[7][1], 0, -1)),
        9: (5, rectangle(WIDTH, HEIGHT, 29, 2, 40, 10)),
    }


def frame_arrays(
    masks: dict[int, tuple[int, np.ndarray]],
    *,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    instance_ids = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    class_ids = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    rng = np.random.default_rng(SEED + seed_offset)
    for instance_id in sorted(masks):
        class_id, mask = masks[instance_id]
        dx = int(rng.integers(0, 2))
        dy = int(rng.integers(0, 2))
        placed = shifted(mask, dx, dy) if seed_offset else mask
        instance_ids[placed] = instance_id
        class_ids[placed] = class_id
    return instance_ids, class_ids


def write_v1(
    path: Path,
    masks: dict[int, tuple[int, np.ndarray]],
) -> None:
    pieces = [V1_HEADER.pack(V1_MAGIC, V1_VERSION, 0, WIDTH, HEIGHT, len(masks))]
    for instance_id in sorted(masks):
        class_id, mask = masks[instance_id]
        runs = mask_to_runs(mask)
        pieces.append(V1_RECORD.pack(instance_id, class_id, len(runs)))
        pieces.extend(V1_RUN.pack(start, length) for start, length in runs)
    path.write_bytes(b"".join(pieces))


def write_v2(
    path: Path,
    masks: dict[int, tuple[int, np.ndarray]],
    *,
    confidences: dict[int, float],
    crowds: dict[int, bool],
    palette: dict[int, list[int]],
) -> None:
    entries = []
    for instance_id in sorted(masks):
        class_id, mask = masks[instance_id]
        mask_bytes = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        entries.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "confidence": confidences[instance_id],
                "crowd": crowds.get(instance_id, False),
                "palette": palette[class_id],
                "mask_base64": base64.b64encode(mask_bytes).decode("ascii"),
            }
        )
    payload = {
        "format_version": "v2",
        "width": WIDTH,
        "height": HEIGHT,
        "palette": {str(key): value for key, value in palette.items()},
        "instances": entries,
    }
    atomic_write_json(path, payload)


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    v1_masks = build_v1_masks()
    v2_masks = build_v2_masks(v1_masks)
    palette = {
        1: [220, 60, 60, 255],
        2: [60, 180, 90, 255],
        3: [60, 100, 220, 255],
        4: [230, 190, 50, 255],
        5: [180, 70, 200, 255],
    }
    for frame_index in range(3):
        frame_masks = v1_masks if frame_index == 0 else v2_masks
        instance_ids, class_ids = frame_arrays(
            frame_masks,
            seed_offset=frame_index,
        )
        encode_panorama_png(
            instance_ids,
            class_ids,
            images_dir / f"panorama_{frame_index:03d}.png",
        )
    metadata = {
        "width": WIDTH,
        "height": HEIGHT,
        "instances": [
            {"instance_id": 1, "source_format": "v1", "class_id": 1, "confidence": 0.95, "crowd": False},
            {"instance_id": 2, "source_format": "v1", "class_id": 2, "confidence": 0.90, "crowd": False},
            {"instance_id": 3, "source_format": "v1", "class_id": 3, "confidence": 0.82, "crowd": False},
            {"instance_id": 4, "source_format": "v1", "class_id": 1, "confidence": 0.76, "crowd": False},
            {"instance_id": 5, "source_format": "v1", "class_id": 2, "confidence": 0.71, "crowd": False},
            {"instance_id": 6, "source_format": "v1", "class_id": 4, "confidence": 0.68, "crowd": False},
            {"instance_id": 7, "source_format": "v1", "class_id": 3, "crowd": True},
        ],
    }
    atomic_write_json(output / "instances.json", metadata)
    write_v1(output / "masks_v1.bin", v1_masks)
    v2_confidences = {
        1: 0.93,
        2: 0.88,
        3: 0.84,
        4: 0.74,
        5: 0.69,
        6: 0.72,
        7: 0.60,
        9: 0.80,
    }
    v2_crowds = {7: True, 9: False}
    write_v2(
        output / "masks_v2.json",
        v2_masks,
        confidences=v2_confidences,
        crowds=v2_crowds,
        palette=palette,
    )
    config = {
        "iou_match_threshold": 0.333333333,
        "pixel_retention": {
            "mode": "prefer_non_crowd",
            "min_coverage": 0.333333333,
        },
        "confidence_default": 0.5,
        "default_crowd": False,
        "palette": {str(key): value for key, value in palette.items()},
        "interaction_budget_ms": 15,
        "max_workers": 4,
        "snapshot_commit_interval": 2,
        "boundary_epsilon": 1e-6,
        "frame": "panorama_000",
    }
    atomic_write_json(output / "config.json", config)
    sample_files = [
        output / "config.json",
        output / "instances.json",
        output / "masks_v1.bin",
        output / "masks_v2.json",
        *sorted(images_dir.glob("*.png")),
    ]
    manifest = {
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "pixel_encoding": "packed=(instance_id<<16)|class_id stored in RGBA big-endian bytes",
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sample_files
        },
    }
    atomic_write_json(output / "sample_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    manifest = generate(Path(args.out))
    print(json.dumps({"generated": args.out, "seed": manifest["seed"], "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
