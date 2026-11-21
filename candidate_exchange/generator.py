from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .models import RawCandidate
from .util import atomic_write_json, atomic_write_text, quantize_int, quantize_milli, stable_hash
from .v1 import V1_HEADER, V1_ROW_WIDTH, format_v1_row
from .v2 import V2_FORMAT_VERSION


IMAGE_WIDTH = 192
IMAGE_HEIGHT = 128
SERIES_SPECS = (
    ("alpha_seq", 8, 0),
    ("beta_seq", 7, 1),
    ("gamma_seq", 6, 2),
)
CLASS_NAMES = {1: "pedestrian", 2: "vehicle", 3: "noise"}
CLASS_COLORS = {1: (224, 72, 72), 2: (64, 120, 224), 3: (72, 196, 112)}


def _base_canvas(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(8, 45, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    grid = np.indices((IMAGE_HEIGHT, IMAGE_WIDTH))
    gradient = ((grid[0] + grid[1]) % 24).astype(np.uint8)
    noise = np.clip(noise.astype(np.int16) + gradient[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(noise, mode="RGB")


def _draw_target(
    image: Image.Image,
    *,
    class_code: int,
    box: tuple[int, int, int, int],
    seed: int,
) -> None:
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    color = CLASS_COLORS[class_code]
    rng = np.random.default_rng(seed)
    if class_code == 1:
        draw.rectangle((x1, y1, x2, y2), fill=color, outline=(245, 220, 220), width=2)
        draw.ellipse((x1 + 5, y1 + 4, x1 + 13, y1 + 12), fill=(250, 230, 210))
    elif class_code == 2:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=6, fill=color, outline=(220, 235, 255), width=2)
        draw.rectangle((x1 + 8, y1 + 7, x2 - 8, y2 - 7), outline=(235, 245, 255), width=1)
    else:
        draw.polygon(
            [
                ((x1 + x2) // 2, y1),
                (x2, (y1 + y2) // 2),
                ((x1 + x2) // 2, y2),
                (x1, (y1 + y2) // 2),
            ],
            fill=color,
        )
    for _ in range(12):
        px = int(rng.integers(x1, max(x1 + 1, x2 + 1)))
        py = int(rng.integers(y1, max(y1 + 1, y2 + 1)))
        value = int(rng.integers(0, 80))
        draw.point((px, py), fill=(value, value, value))


def _truth_boxes(series_index: int, frame_index: int) -> list[dict[str, Any]]:
    phase = frame_index * 0.58 + series_index * 0.35
    pedestrian_x = 18 + series_index * 9 + frame_index * 13
    pedestrian_y = 24 + int(7 * math.sin(phase))
    vehicle_x = 35 + series_index * 7 + frame_index * 10
    vehicle_y = 72 + int(5 * math.cos(phase))
    return [
        {
            "class_code": 1,
            "class_name": CLASS_NAMES[1],
            "box": [pedestrian_x, pedestrian_y, pedestrian_x + 30, pedestrian_y + 42],
        },
        {
            "class_code": 2,
            "class_name": CLASS_NAMES[2],
            "box": [vehicle_x, vehicle_y, vehicle_x + 46, vehicle_y + 25],
        },
    ]


def _fractional_box(box: list[int], frame_index: int, object_index: int, series_index: int) -> list[float]:
    fractions = (0.0, 0.2, 0.4, 0.7, 0.8, 0.3)
    result: list[float] = []
    for axis, value in enumerate(box):
        fraction = fractions[(frame_index + object_index * 2 + axis + series_index) % len(fractions)]
        result.append(float(value) + fraction)
    return result


def _candidate_score(series_index: int, frame_index: int, object_index: int, seed: int) -> int:
    return 715 + ((seed + series_index * 97 + frame_index * 31 + object_index * 41) % 271)


def _candidate_set(series_index: int, frame_index: int, seed: int) -> list[RawCandidate]:
    result: list[RawCandidate] = []
    truth = _truth_boxes(series_index, frame_index)
    for object_index, target in enumerate(truth):
        x1, y1, x2, y2 = target["box"]
        jitter_x = ((seed + frame_index * 3 + object_index * 5 + series_index) % 5) - 2
        jitter_y = ((seed + frame_index * 7 + object_index * 2 + series_index) % 3) - 1
        integer_box = [x1 + jitter_x, y1 + jitter_y, x2 + jitter_x, y2 + jitter_y]
        v2_box = _fractional_box(integer_box, frame_index, object_index, series_index)
        score_pm = _candidate_score(series_index, frame_index, object_index, seed)
        score_sub_milli = 0.0001 + 0.0001 * (
            (seed + frame_index + object_index + series_index) % 3
        )
        result.append(
            RawCandidate(
                series_name=SERIES_SPECS[series_index][0],
                frame_index=frame_index,
                class_code=target["class_code"],
                class_name=target["class_name"],
                x1=v2_box[0],
                y1=v2_box[1],
                x2=v2_box[2],
                y2=v2_box[3],
                score=score_pm / 1000.0 + score_sub_milli,
                precision_label="f32",
                source_version="v2",
                source_ordinal=100 + object_index,
                extra={"generator": "seed-programmatic", "truth_index": object_index},
            )
        )
    # Low-score class-1 candidates create real candidate-pool eviction and re-entry.
    for noise_index in range(3):
        if noise_index == 2 and frame_index in {2, 3}:
            continue
        x1 = 12 + noise_index * 17 + (frame_index % 3) * 2
        y1 = 99 + noise_index * 7
        integer_box = [x1, y1, x1 + 11, y1 + 9]
        v2_box = _fractional_box(integer_box, frame_index, noise_index + 2, series_index)
        score_pm = 180 + ((seed + series_index * 13 + frame_index * 11 + noise_index * 29) % 170)
        if noise_index == 2:
            score_pm = 330  # tie-prone score for coordinate-lexicographic eviction checks
        result.append(
            RawCandidate(
                series_name=SERIES_SPECS[series_index][0],
                frame_index=frame_index,
                class_code=1,
                class_name=CLASS_NAMES[1],
                x1=v2_box[0],
                y1=v2_box[1],
                x2=v2_box[2],
                y2=v2_box[3],
                score=score_pm / 1000.0
                + 0.0001
                + 0.0001 * ((frame_index + noise_index + series_index) % 3),
                precision_label="f32",
                source_version="v2",
                source_ordinal=200 + noise_index,
                extra={"generator": "noise-track", "noise_index": noise_index},
            )
        )
    result.sort(
        key=lambda item: (
            -quantize_milli(item.score),
            quantize_int(item.x1),
            quantize_int(item.y1),
            quantize_int(item.x2),
            quantize_int(item.y2),
            item.class_code,
            item.source_ordinal,
        )
    )
    return result


def _v2_document(candidates: list[RawCandidate]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[RawCandidate]] = {}
    for item in candidates:
        groups.setdefault((item.series_name, item.frame_index), []).append(item)
    frames: list[dict[str, Any]] = []
    for key in sorted(groups):
        frames.append(
            {
                "series_name": key[0],
                "frame_index": key[1],
                "candidates": [
                    {
                        "series_name": item.series_name,
                        "frame_index": item.frame_index,
                        "class_code": item.class_code,
                        "class_name": item.class_name,
                        "x1": item.x1,
                        "y1": item.y1,
                        "x2": item.x2,
                        "y2": item.y2,
                        "score": item.score,
                        "precision_label": item.precision_label,
                        "extra": item.extra,
                    }
                    for item in groups[key]
                ],
            }
        )
    return {
        "format_version": V2_FORMAT_VERSION,
        "schema": "candidate-exchange-v2-json",
        "precision": "mixed-f32-and-subpixel",
        "frames": frames,
    }


def write_candidate_files(
    data_dir: str | Path, candidates: list[RawCandidate]
) -> None:
    root = Path(data_dir)
    atomic_write_json(root / "candidates_v2.json", _v2_document(candidates))
    lines = [V1_HEADER]
    for item in candidates:
        row = format_v1_row(
            item.series_name,
            item.frame_index,
            item.class_code,
            quantize_int(item.x1),
            quantize_int(item.y1),
            quantize_int(item.x2),
            quantize_int(item.y2),
            quantize_milli(item.score),
        )
        assert len(row) == V1_ROW_WIDTH
        lines.append(row)
    atomic_write_text(root / "candidates_v1.txt", "\n".join(lines) + "\n")


def generate_dataset(
    output_dir: str | Path,
    *,
    seed: int = 20261122,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"dataset directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    truth: dict[str, Any] = {"seed": seed, "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT], "series": {}}
    all_candidates: list[RawCandidate] = []
    for series_index, (series_name, frame_count, _) in enumerate(SERIES_SPECS):
        series_dir = images_root / series_name
        series_dir.mkdir(parents=True, exist_ok=True)
        series_truth: list[dict[str, Any]] = []
        for frame_index in range(frame_count):
            frame_seed = seed + series_index * 10_000 + frame_index * 97
            image = _base_canvas(frame_seed)
            frame_truth = _truth_boxes(series_index, frame_index)
            for object_index, target in enumerate(frame_truth):
                _draw_target(
                    image,
                    class_code=target["class_code"],
                    box=tuple(target["box"]),
                    seed=frame_seed + object_index * 17,
                )
            image_path = series_dir / f"frame_{frame_index:04d}.png"
            image.save(image_path, format="PNG", optimize=False)
            series_truth.append(
                {
                    "frame_index": frame_index,
                    "image": image_path.relative_to(root).as_posix(),
                    "objects": frame_truth,
                }
            )
            all_candidates.extend(_candidate_set(series_index, frame_index, seed))
        truth["series"][series_name] = series_truth
    all_candidates.sort(
        key=lambda item: (
            item.series_name,
            item.frame_index,
            -quantize_milli(item.score),
            quantize_int(item.x1),
            quantize_int(item.y1),
            quantize_int(item.x2),
            quantize_int(item.y2),
            item.class_code,
            item.source_ordinal,
        )
    )
    write_candidate_files(root, all_candidates)
    atomic_write_json(root / "ground_truth.json", truth)
    detection_params = {
        "schema_version": "1.0",
        "scales": [[28, 36], [32, 44], [44, 60], [48, 72]],
        "step": 8,
        "color_mean_threshold": 22.0,
        "edge_density_threshold": 0.055,
        "fusion_weights": {"color": 0.58, "edge": 0.42},
        "candidate_pool_capacity": 3,
        "class_map": {str(code): name for code, name in CLASS_NAMES.items()},
        "class_colors": {str(code): list(color) for code, color in CLASS_COLORS.items()},
        "frame_overrides": {},
    }
    match_config = {
        "iou_threshold": 0.35,
        "max_gap_frames": 2,
        "pool_eviction_rule": "score_desc_then_coord_lex",
        "fixed_decimal_places": 3,
        "concurrency": 4,
        "snapshot_commit_interval": 2,
    }
    atomic_write_json(root / "detection_params.json", detection_params)
    atomic_write_json(root / "match_config.json", match_config)
    atomic_write_json(root / "exchange_config.json", {"active_version": "v1"})
    sidecar_entries = []
    for series_name, frame_count, _ in SERIES_SPECS:
        for frame_index in range(frame_count):
            sidecar_entries.append(
                {
                    "series_name": series_name,
                    "frame_index": frame_index,
                    "precision_label": "f32",
                    "coordinate_residual": [0.0, 0.0, 0.0, 0.0],
                    "score_residual": 0.0,
                    "note": "initial exchange metadata",
                }
            )
    atomic_write_json(
        root / "sidecar_metadata.json",
        {
            "schema": "candidate-exchange-sidecar-v1",
            "sidecar_metadata_hash": stable_hash(sidecar_entries),
            "entries": sidecar_entries,
        },
    )
    injections = {
        "schema": "candidate-exchange-injections-v1",
        "injections": [
            {
                "id": "INJ-001",
                "type": "truncated_png",
                "series_name": "beta_seq",
                "frame_index": 2,
                "stage": "detection",
                "expected_code": "E_IMAGE_INVALID",
            },
            {
                "id": "INJ-002",
                "type": "v1_field_missing",
                "series_name": "alpha_seq",
                "frame_index": 1,
                "stage": "parsing_v1",
                "expected_code": "E_FIELD_TRUNCATED",
            },
            {
                "id": "INJ-003",
                "type": "coordinate_out_of_bounds",
                "series_name": "gamma_seq",
                "frame_index": 2,
                "stage": "parsing",
                "expected_code": "E_COORD_INVALID",
            },
            {
                "id": "INJ-004",
                "type": "non_finite_weight",
                "series_name": "beta_seq",
                "frame_index": 3,
                "stage": "detection",
                "expected_code": "E_NON_FINITE",
            },
            {
                "id": "INJ-005",
                "type": "class_code_not_in_map",
                "series_name": "alpha_seq",
                "frame_index": 3,
                "stage": "parsing",
                "expected_code": "E_CLASS_UNKNOWN",
            },
            {
                "id": "INJ-006",
                "type": "matching_forced_error",
                "series_name": "gamma_seq",
                "frame_index": 3,
                "stage": "matching",
                "expected_code": "E_MATCH_INVALID",
            },
            {
                "id": "INJ-007",
                "type": "serialization_forced_error",
                "series_name": "beta_seq",
                "frame_index": 4,
                "stage": "serialization",
                "expected_code": "E_SERIALIZE_INVALID",
            },
        ],
    }
    atomic_write_json(root / "injections.json", injections)
    edits = {
        "schema": "candidate-exchange-edit-sequence-v1",
        "operations": [
            {"id": "EDIT-001", "type": "switch_version", "active_version": "v2"},
            {
                "id": "EDIT-002",
                "type": "replace_frame_candidates",
                "series_name": "alpha_seq",
                "frame_index": 4,
            },
            {
                "id": "EDIT-003",
                "type": "replace_image",
                "series_name": "beta_seq",
                "frame_index": 3,
            },
            {"id": "EDIT-004", "type": "set_iou_threshold", "value": 0.42},
            {"id": "EDIT-005", "type": "set_pool_capacity", "value": 4},
            {
                "id": "EDIT-006",
                "type": "replace_frame_candidates",
                "series_name": "gamma_seq",
                "frame_index": 4,
            },
        ],
    }
    atomic_write_json(root / "edit_sequence.json", edits)
    return {
        "root": str(root),
        "seed": seed,
        "series": [name for name, _, _ in SERIES_SPECS],
        "frame_count": sum(count for _, count, _ in SERIES_SPECS),
        "candidate_count": len(all_candidates),
        "v1_path": str(root / "candidates_v1.txt"),
        "v2_path": str(root / "candidates_v2.json"),
    }
