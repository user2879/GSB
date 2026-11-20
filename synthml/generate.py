"""Deterministic synthetic sample, experiment and scale-plan generator."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

from . import SCHEMA_VERSION
from .config import feature_dimension
from .util import atomic_write_json


LABEL_SHAPES: dict[str, tuple[str, tuple[int, int, int]]] = {
    "bar": ("bar", (28, 156, 178)),
    "circle": ("circle", (49, 156, 80)),
    "cross": ("cross", (145, 70, 166)),
    "diamond": ("diamond", (225, 145, 34)),
    "ellipse": ("ellipse", (207, 72, 147)),
    "square": ("square", (42, 91, 202)),
    "star": ("star", (126, 94, 31)),
    "triangle": ("triangle", (207, 54, 58)),
}

LABEL_RATES: dict[str, float] = {
    "bar": 0.38,
    "circle": 0.48,
    "cross": 0.34,
    "diamond": 0.44,
    "ellipse": 0.40,
    "square": 0.50,
    "star": 0.36,
    "triangle": 0.46,
}


def _strategy(
    strategy_id: str,
    method: str,
    color_bins: int,
    hog_bins: int,
    classifier: str,
    *,
    knn_k: int = 3,
    delta: float = 0.0,
    base: float = 1.0,
    small: float = 0.0,
    medium: float = 0.0,
    large: float = 0.0,
    imbalance: float = 0.0,
    label_bonus: float = 0.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": strategy_id,
        "feature_method": method,
        "color_bins": color_bins,
        "hog_bins": hog_bins,
        "classifier": classifier,
        "threshold_delta": delta,
        "base_score": base,
        "scale_bonus": {"small": small, "medium": medium, "large": large},
        "imbalance_bonus": imbalance,
        "label_bonus": label_bonus,
    }
    if classifier == "knn":
        result["knn_k"] = knn_k
    return result


def default_config(
    labels: list[str],
    *,
    concurrency: int = 4,
    max_concurrency: int = 8,
    snapshot_interval: int = 8,
) -> dict[str, Any]:
    labels = sorted(labels)
    expected = []
    for method in ("color", "hog", "concat"):
        for color_bins, hog_bins in ((8, 6), (8, 9), (16, 6), (16, 9)):
            expected.append(
                {
                    "feature_method": method,
                    "color_bins": color_bins,
                    "hog_bins": hog_bins,
                    "dimension": feature_dimension(method, color_bins, hog_bins, (2, 2)),
                }
            )
    strategies = [
        _strategy(
            "small_color_centroid",
            "color",
            8,
            9,
            "nearest_centroid",
            base=1.0,
            small=0.82,
            medium=-0.08,
            large=-0.65,
            imbalance=0.28,
        ),
        _strategy(
            "small_color_knn3",
            "color",
            8,
            9,
            "knn",
            knn_k=3,
            delta=-0.01,
            base=0.96,
            small=0.48,
            medium=0.12,
            large=-0.30,
            imbalance=0.10,
        ),
        _strategy(
            "medium_color16_knn3",
            "color",
            16,
            9,
            "knn",
            knn_k=3,
            delta=-0.01,
            base=1.02,
            small=-0.12,
            medium=0.70,
            large=-0.05,
            imbalance=0.12,
        ),
        _strategy(
            "medium_hog9_knn5",
            "hog",
            8,
            9,
            "knn",
            knn_k=5,
            delta=0.0,
            base=1.0,
            small=-0.28,
            medium=0.52,
            large=0.12,
            imbalance=0.04,
            label_bonus=0.08,
        ),
        _strategy(
            "large_hog9_logreg",
            "hog",
            8,
            9,
            "logistic_regression",
            delta=0.01,
            base=0.98,
            small=-0.62,
            medium=0.14,
            large=0.72,
            imbalance=0.0,
            label_bonus=0.09,
        ),
        _strategy(
            "large_concat_logreg",
            "concat",
            8,
            9,
            "logistic_regression",
            delta=0.01,
            base=1.04,
            small=-0.82,
            medium=0.24,
            large=0.96,
            imbalance=0.0,
            label_bonus=0.14,
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "resolution": [96, 96],
        "labels": labels,
        "label_rates": {label: LABEL_RATES.get(label, 0.42) for label in labels},
        "features": {
            "supported": ["color", "hog", "concat"],
            "color_bins": 8,
            "color_bins_candidates": [8, 16],
            "hog_bins": 9,
            "hog_bins_candidates": [6, 9],
            "hog_cells": [2, 2],
            "expected_dimensions": expected,
        },
        "classifiers": {
            "supported": ["logistic_regression", "nearest_centroid", "knn"],
            "logistic_regression": {
                "learning_rate": 0.12,
                "epochs": 60,
                "l2": 0.001,
            },
            "nearest_centroid": {"metric": "euclidean"},
            "knn": {
                "k": 3,
                "k_candidates": [1, 3, 5],
                "tie_break": "distance_then_sample_name",
                "vote": "laplace_positive_fraction",
            },
        },
        "threshold": 0.5,
        "precision": "standard",
        "precision_levels": {
            "fast": {"quantization_levels": 1024, "rounding_decimals": 6},
            "standard": {"quantization_levels": 4096, "rounding_decimals": 8},
            "high": {"quantization_levels": 16384, "rounding_decimals": 10},
        },
        "concurrency": concurrency,
        "max_concurrency": max_concurrency,
        "snapshot_interval": snapshot_interval,
        "strategy_candidates": strategies,
        "evaluation": {
            "probability_decimals": 10,
            "metric_decimals": 8,
            "boundary_policy": "equality_positive_labels_lexicographic",
            "cost_weight": 0.035,
        },
    }


def default_plan(sample_count: int, label_count: int) -> dict[str, Any]:
    cuts = sorted(
        {
            max(1, min(sample_count, sample_count // 3)),
            max(1, min(sample_count, (2 * sample_count) // 3)),
            max(1, sample_count),
        }
    )
    label_cuts = sorted(
        {
            max(1, min(label_count, 3)),
            max(1, min(label_count, 5)),
            max(1, label_count),
        }
    )
    names = ["tier_s", "tier_m", "tier_l"]
    return {
        "schema_version": SCHEMA_VERSION,
        "tiers": [
            {
                "id": names[index],
                "max_samples": cuts[index],
                "max_labels": label_cuts[index],
                "description": f"deterministic scale tier {index + 1}",
            }
            for index in range(3)
        ],
    }


def default_injections(sample_names: list[str]) -> dict[str, Any]:
    ordered = sorted(sample_names)
    if len(ordered) >= 3:
        truncate_name, nonfinite_name, unknown_name = ordered[1], ordered[2], ordered[3] if len(ordered) > 3 else ordered[0]
    elif ordered:
        truncate_name = nonfinite_name = unknown_name = ordered[0]
    else:
        truncate_name = nonfinite_name = unknown_name = "sample_000000.png"
    return {
        "schema_version": SCHEMA_VERSION,
        "items": [
            {
                "id": "INJ-001",
                "kind": "truncate_png",
                "sample": truncate_name,
                "stage": "load",
                "enabled": True,
            },
            {
                "id": "INJ-002",
                "kind": "nonfinite_feature",
                "sample": nonfinite_name,
                "stage": "feature",
                "enabled": True,
            },
            {
                "id": "INJ-003",
                "kind": "unknown_label",
                "sample": unknown_name,
                "stage": "load",
                "enabled": True,
            },
            {
                "id": "INJ-THRESHOLD",
                "kind": "threshold_out_of_range",
                "stage": "config",
                "value": 1.7,
                "enabled": False,
            },
        ],
    }


def _positive_matrix(
    sample_count: int, labels: list[str], rates: dict[str, float], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = np.zeros((sample_count, len(labels)), dtype=np.uint8)
    order = rng.permutation(sample_count)
    for label_index, label in enumerate(labels):
        count = max(2 if sample_count >= 2 else 1, int(round(sample_count * rates.get(label, 0.42))))
        count = min(sample_count, count)
        offset = int(rng.integers(0, sample_count))
        chosen = np.array([(offset + label_index * 3 + step) % sample_count for step in range(count)])
        matrix[chosen, label_index] = 1
    for index in range(sample_count):
        if not matrix[index].any():
            matrix[index, int(order[index]) % len(labels)] = 1
    return matrix


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    width = max(2, (x2 - x1) // 7)
    if shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "ellipse":
        draw.ellipse(box, fill=color)
    elif shape == "triangle":
        draw.polygon([((x1 + x2) // 2, y1), (x1, y2), (x2, y2)], fill=color)
    elif shape == "diamond":
        draw.polygon([((x1 + x2) // 2, y1), (x2, (y1 + y2) // 2), ((x1 + x2) // 2, y2), (x1, (y1 + y2) // 2)], fill=color)
    elif shape == "cross":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.line((x1, y1, x2, y2), fill=color, width=width)
        draw.line((x1, y2, x2, y1), fill=color, width=width)
    elif shape == "bar":
        draw.rectangle((x1, y1 + (y2 - y1) // 3, x2, y2 - (y2 - y1) // 3), fill=color)
    elif shape == "star":
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx, ry = (x2 - x1) / 2.0, (y2 - y1) / 2.0
        points = []
        for point_index in range(10):
            angle = -math.pi / 2.0 + point_index * math.pi / 5.0
            radius = 1.0 if point_index % 2 == 0 else 0.44
            points.append((cx + math.cos(angle) * rx * radius, cy + math.sin(angle) * ry * radius))
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"unknown shape: {shape}")


def generate_dataset(
    out_dir: Path,
    config_path: Path,
    plan_path: Path,
    *,
    sample_count: int = 72,
    seed: int = 20260317,
    width: int = 96,
    height: int = 96,
    noise: float = 4.0,
    blur: float = 0.55,
    concurrency: int = 4,
    snapshot_interval: int = 8,
    injection_path: Path | None = None,
    serial_config_path: Path | None = None,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    labels = sorted(LABEL_SHAPES)
    rates = {label: LABEL_RATES[label] for label in labels}
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = _positive_matrix(sample_count, labels, rates, seed)
    sample_names: list[str] = []
    for sample_index in range(sample_count):
        rng = np.random.default_rng(seed * 1000003 + sample_index * 7919)
        base = np.full((height, width, 3), (239, 237, 229), dtype=np.uint8)
        background_noise = rng.normal(0.0, noise * 0.45, size=base.shape)
        image_array = np.clip(base.astype(np.float64) + background_noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode="RGB")
        draw = ImageDraw.Draw(image)
        boxes: dict[str, list[int]] = {}
        present = [labels[index] for index in np.flatnonzero(matrix[sample_index])]
        for label in present:
            shape, color = LABEL_SHAPES[label]
            shape_width = int(rng.integers(17, 35))
            shape_height = int(rng.integers(15, 34))
            x1 = int(rng.integers(3, max(4, width - shape_width - 3)))
            y1 = int(rng.integers(3, max(4, height - shape_height - 3)))
            x2 = min(width, x1 + shape_width)
            y2 = min(height, y1 + shape_height)
            clipped = (max(0, x1), max(0, y1), max(1, x2), max(1, y2))
            _draw_shape(draw, shape, clipped, color)
            boxes[label] = [int(clipped[0]), int(clipped[1]), int(clipped[2]), int(clipped[3])]
        if noise > 0:
            noised = np.asarray(image, dtype=np.float64) + rng.normal(0.0, noise, size=(height, width, 3))
            image = Image.fromarray(np.clip(noised, 0, 255).astype(np.uint8), mode="RGB")
        if blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(blur)))
        name = f"sample_{sample_index:06d}.png"
        sample_names.append(name)
        image.save(out_dir / name, format="PNG")
        annotation = {
            "image": name,
            "labels": sorted(boxes),
            "boxes": {label: boxes[label] for label in sorted(boxes)},
        }
        atomic_write_json(out_dir / f"{Path(name).stem}.json", annotation)

    config = default_config(
        labels,
        concurrency=concurrency,
        snapshot_interval=snapshot_interval,
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if serial_config_path is not None:
        serial_config = default_config(
            labels,
            concurrency=1,
            snapshot_interval=max(1, snapshot_interval // 2),
        )
        serial_config_path.parent.mkdir(parents=True, exist_ok=True)
        serial_config_path.write_text(
            yaml.safe_dump(serial_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    plan = default_plan(sample_count, len(labels))
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if injection_path is not None:
        injection_path.parent.mkdir(parents=True, exist_ok=True)
        injection_path.write_text(
            yaml.safe_dump(default_injections(sample_names), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return {
        "samples_dir": str(out_dir),
        "sample_count": sample_count,
        "labels": labels,
        "config": str(config_path),
        "plan": str(plan_path),
        "injections": str(injection_path) if injection_path else None,
        "serial_config": str(serial_config_path) if serial_config_path else None,
        "seed": seed,
        "resolution": [width, height],
        "noise": noise,
        "blur": blur,
    }


def replace_sample(samples_dir: Path, target_name: str, source_png: Path) -> dict[str, Any]:
    if not target_name.endswith(".png"):
        raise ValueError("target must be a .png filename")
    source_json = source_png.with_suffix(".json")
    if not source_png.exists() or not source_json.exists():
        raise FileNotFoundError("source PNG and matching JSON annotation are required")
    target_png = samples_dir / target_name
    target_json = samples_dir / f"{Path(target_name).stem}.json"
    target_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_png, target_png)
    annotation = json.loads(source_json.read_text(encoding="utf-8"))
    if not isinstance(annotation, dict):
        raise ValueError("source annotation must be an object")
    annotation["image"] = target_name
    atomic_write_json(target_json, annotation)
    return {"target": str(target_png), "source": str(source_png)}
