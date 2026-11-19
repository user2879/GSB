"""Deterministic synthetic PNG sequences, ground truth and config files."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .config import DEFAULT_CLASS_TABLE

WIDTH = 160
HEIGHT = 120
SEED_BASE = 20261120
CLASS_COLORS = {
    0: (220, 60, 60),     # circle
    1: (60, 200, 80),     # square
    2: (70, 120, 230),    # triangle
}
SEQUENCES = {"seqA": 8, "seqB": 6, "seqC": 5}


def _draw_shape(draw: ImageDraw.ImageDraw, kind: int, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    color = CLASS_COLORS[kind]
    if kind == 0:
        draw.ellipse(box, fill=color)
    elif kind == 1:
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(x1, y2), ((x1 + x2) // 2, y1), (x2, y2)], fill=color)


def _build_objects(seq_index: int) -> list[dict]:
    rng = random.Random(SEED_BASE + seq_index * 97)
    objects = []
    for obj_index in range(2):
        kind = (seq_index + obj_index) % 3
        size = int(rng.randint(14, 22))
        x = float(rng.randint(4, WIDTH - size - 4))
        y = float(rng.randint(4, HEIGHT - size - 4))
        vx = float(rng.choice([-2, -1, 1, 2]))
        vy = float(rng.choice([-1, 0, 1]))
        objects.append(
            {"id": obj_index, "class": kind, "size": size, "x": x, "y": y, "vx": vx, "vy": vy}
        )
    return objects


def _advance(obj: dict) -> tuple[int, int, int, int]:
    size = obj["size"]
    obj["x"] += obj["vx"]
    obj["y"] += obj["vy"]
    if obj["x"] <= 0 or obj["x"] + size >= WIDTH:
        obj["vx"] = -obj["vx"]
    if obj["y"] <= 0 or obj["y"] + size >= HEIGHT:
        obj["vy"] = -obj["vy"]
    obj["x"] = min(max(obj["x"], 0.0), float(WIDTH - size))
    obj["y"] = min(max(obj["y"], 0.0), float(HEIGHT - size))
    x1 = int(obj["x"])
    y1 = int(obj["y"])
    return x1, y1, x1 + size, y1 + size


def render_frame(seq_index: int, frame: int, objects: list[dict]) -> tuple[Image.Image, list[dict]]:
    rng = random.Random(SEED_BASE + seq_index * 1000 + frame)
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()
    for py in range(HEIGHT):
        for px in range(WIDTH):
            noise = int(rng.gauss(45, 12))
            value = max(0, min(255, noise))
            pixels[px, py] = (value, value, value)
    draw = ImageDraw.Draw(image)
    ground_truth = []
    for obj in objects:
        box = _advance(obj)
        _draw_shape(draw, obj["class"], box)
        ground_truth.append(
            {"object_id": obj["id"], "class_code": obj["class"], "box": list(box)}
        )
    return image, ground_truth


def generate_dataset(root: str | Path) -> Path:
    root = Path(root)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    all_gt: dict[str, list] = {}
    for seq_index, (sequence, frame_count) in enumerate(SEQUENCES.items()):
        objects = _build_objects(seq_index)
        seq_dir = image_root / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)
        all_gt[sequence] = []
        for frame in range(frame_count):
            image, gt = render_frame(seq_index, frame, objects)
            image.save(seq_dir / f"frame_{frame:04d}.png", format="PNG")
            all_gt[sequence].append({"frame": frame, "objects": gt})

    (root / "ground_truth.json").write_text(
        json.dumps(
            {
                "width": WIDTH,
                "height": HEIGHT,
                "class_table": {
                    "code_to_name": {str(k): v for k, v in DEFAULT_CLASS_TABLE.code_to_name.items()}
                },
                "sequences": all_gt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    detection_params = {
        "window_sizes": [12, 18, 24],
        "step": 3,
        "color_mean_threshold": 70.0,
        "edge_density_threshold": 0.05,
        "score_weights": {"color": 0.6, "edge": 0.4},
        "pool_capacity": 24,
    }
    (root / "detection_params.json").write_text(
        json.dumps(detection_params, indent=2), encoding="utf-8"
    )
    bad_weights = dict(detection_params)
    bad_weights["score_weights"] = {"color": float("inf"), "edge": 0.4}
    (root / "detection_params_nonfinite.json").write_text(
        json.dumps(bad_weights, indent=2), encoding="utf-8"
    )

    match_config = {
        "iou_threshold": 0.3,
        "max_gap_frames": 2,
        "eviction_rule": "score_desc_coord_lex",
    }
    (root / "match_config.json").write_text(
        json.dumps(match_config, indent=2), encoding="utf-8"
    )

    injections = {
        "injections": [
            {
                "injection_id": "INJ001",
                "sequence": "seqA",
                "frame": 3,
                "stage": "detect",
                "fault_type": "truncated_png",
            },
            {
                "injection_id": "INJ002",
                "sequence": "seqB",
                "frame": 2,
                "stage": "detect",
                "fault_type": "nonfinite_weight",
            },
            {
                "injection_id": "INJ003",
                "sequence": "seqA",
                "frame": 5,
                "stage": "match",
                "fault_type": "candidate_corrupt",
            },
            {
                "injection_id": "INJ004",
                "sequence": "seqC",
                "frame": 1,
                "stage": "serialize",
                "fault_type": "v1_missing_field",
            },
            {
                "injection_id": "INJ005",
                "sequence": "seqC",
                "frame": 3,
                "stage": "serialize",
                "fault_type": "coords_out_of_bounds",
            },
            {
                "injection_id": "INJ006",
                "sequence": "seqB",
                "frame": 4,
                "stage": "serialize",
                "fault_type": "class_unknown",
            },
        ]
    }
    (root / "injections.json").write_text(
        json.dumps(injections, indent=2), encoding="utf-8"
    )

    malformed = root / "malformed"
    malformed.mkdir(exist_ok=True)
    source_png = image_root / "seqA" / "frame_0000.png"
    raw = source_png.read_bytes()
    (malformed / "truncated.png").write_bytes(raw[: max(1, len(raw) // 3)])
    (malformed / "bad_v1_missing_field.txt").write_text(
        "seqA       00001001  010  020  030  04005000\n", encoding="utf-8"
    )
    (malformed / "bad_v1_out_of_bounds.txt").write_text(
        "seqA       00001001  0109999  020  030  04005000\n", encoding="utf-8"
    )
    unknown_class_line = (
        "seqA".ljust(10)
        + "1".rjust(5)
        + "9".rjust(3)
        + "10".rjust(5)
        + "20".rjust(5)
        + "30".rjust(5)
        + "40".rjust(5)
        + "500".rjust(7)
        + "0".rjust(1)
    )
    (malformed / "bad_v1_unknown_class.txt").write_text(
        unknown_class_line + "\n", encoding="utf-8"
    )
    bad_v2 = {
        "format": "v2",
        "sequences": [
            {
                "sequence": "seqA",
                "frames": [
                    {
                        "frame": 0,
                        "candidates": [
                            {
                                "box": [10, 10, 30, 30],
                                "score": 0.5,
                                "class_name": "hexagon",
                                "precision_label": "float",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (malformed / "bad_v2_unknown_class.json").write_text(
        json.dumps(bad_v2, indent=2), encoding="utf-8"
    )
    return root
