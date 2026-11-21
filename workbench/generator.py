"""Deterministic local sample and configuration generator."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 192
CHUNK_SIZE = 32

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (210, 58, 52),
    "green": (48, 157, 91),
    "blue": (54, 103, 201),
    "yellow": (225, 181, 47),
    "purple": (135, 73, 174),
}
SHAPES = ("circle", "rectangle", "triangle")
POSITIONS = ("left", "center", "right", "top", "middle", "bottom")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _position_names(rect: list[int]) -> tuple[str, str]:
    x, y, width, height = rect
    cx = x + width / 2.0
    cy = y + height / 2.0
    horizontal = "left" if cx < IMAGE_WIDTH / 3 else "right" if cx > 2 * IMAGE_WIDTH / 3 else "center"
    vertical = "top" if cy < IMAGE_HEIGHT / 3 else "bottom" if cy > 2 * IMAGE_HEIGHT / 3 else "middle"
    return horizontal, vertical


def _build_scene(sample_index: int, rng: random.Random) -> tuple[Image.Image, list[dict[str, Any]]]:
    base = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.int16)
    background = np.array([246, 244, 238], dtype=np.int16)
    base[:, :] = background
    image = Image.fromarray(base.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = list(COLORS)
    shapes = list(SHAPES)
    cells = [
        (22, 18, 94, 66),
        (94, 18, 166, 66),
        (166, 18, 234, 66),
        (22, 74, 94, 124),
        (94, 74, 166, 124),
        (166, 74, 234, 124),
        (22, 132, 94, 176),
        (94, 132, 166, 176),
        (166, 132, 234, 176),
    ]
    rng.shuffle(cells)
    regions: list[dict[str, Any]] = []
    shape_count = 3 + (sample_index % 2)
    for item_index in range(shape_count):
        shape = shapes[(sample_index + item_index * 2) % len(shapes)]
        color_name = colors[(sample_index * 2 + item_index) % len(colors)]
        color = COLORS[color_name]
        left, top, right, bottom = cells[item_index]
        margin = 4 + (item_index % 3)
        box = (left + margin, top + margin, right - margin, bottom - margin)
        if shape == "circle":
            draw.ellipse(box, fill=color, outline=(35, 35, 42), width=2)
        elif shape == "rectangle":
            draw.rectangle(box, fill=color, outline=(35, 35, 42), width=2)
        else:
            draw.polygon(
                [(box[0] + box[2]) // 2, box[1], box[2], box[3], box[0], box[3]],
                fill=color,
                outline=(35, 35, 42),
            )
        rect = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        horizontal, vertical = _position_names(rect)
        regions.append(
            {
                "label": shape,
                "rect": rect,
                "color": color_name,
                "entity_id": f"e{item_index + 1}",
                "position": f"{vertical}-{horizontal}",
            }
        )
    array = np.asarray(image, dtype=np.int16)
    noise_rng = np.random.default_rng(10_000 + sample_index)
    noise = noise_rng.integers(-5, 6, size=array.shape, dtype=np.int16)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB"), regions


def _build_text(sample_name: str, regions: list[dict[str, Any]]) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    assertions: list[dict[str, Any]] = []
    phrases: list[str] = []
    for index, region in enumerate(regions):
        entity_id = region["entity_id"]
        horizontal, vertical = _position_names(region["rect"])
        entity_attributes = {
            "shape": region["label"],
            "color": region["color"],
            "horizontal": horizontal,
            "vertical": vertical,
        }
        entities.append(
            {
                "id": entity_id,
                "type": region["label"],
                "attributes": entity_attributes,
            }
        )
        for attribute_name, value in (
            ("shape", region["label"]),
            ("color", region["color"]),
            ("position", f"{vertical}-{horizontal}"),
        ):
            attributes.append({"entity": entity_id, "name": attribute_name, "value": value})
            assertions.append(
                {
                    "id": f"a{index + 1}-{attribute_name}",
                    "entity": entity_id,
                    "attribute": attribute_name,
                    "value": value,
                }
            )
        phrases.append(f"a {region['color']} {region['label']} at {vertical}-{horizontal}")
    text = f"{sample_name}: " + "; ".join(phrases) + "."
    return {
        "schema_version": 1,
        "sample_name": sample_name,
        "text": text,
        "entities": entities,
        "attributes": attributes,
        "assertions": assertions,
    }


def default_alignment_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "chunk_size": CHUNK_SIZE,
        "quantization_level": 1,
        "quantization_levels": [0, 1, 2, 3],
        "visual_dim": 16,
        "text_dim": 16,
        "visual_weight": 0.55,
        "text_weight": 0.45,
        "similarity_threshold": 0.7451171875,
        "top_k": 4,
        "fixed_decimal_bits": 12,
        "thread_pool_size": 4,
        "feature_point_budget": 2_000_000,
        "snapshot_commit_interval": 2,
        "required_attributes": ["color", "shape", "position"],
        "edge_threshold": 0.08,
        "temperature": 0.25,
        "boundary_neighborhood": 0.035,
        "diagnostic_tolerance": 1e-6,
        "projection_salt": "local-cm-align-v1",
    }


def demo_injections() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "injections": [
            {
                "id": "INJ-PNG-TRUNCATED",
                "sample": "sample-03",
                "stage": "load_image",
                "type": "truncate_png",
                "bytes_removed": 32,
                "enabled": True,
            },
            {
                "id": "INJ-ASSERTION-FIELD",
                "sample": "sample-05",
                "stage": "load_text",
                "type": "missing_assertion_field",
                "field": "value",
                "assertion_index": 1,
                "enabled": True,
            },
            {
                "id": "INJ-FEATURE-NONFINITE",
                "sample": "sample-04",
                "stage": "visual_feature",
                "type": "nonfinite_feature",
                "value": "nan",
                "enabled": True,
            },
            {
                "id": "INJ-THRESHOLD-RANGE",
                "sample": "*",
                "stage": "retrieval",
                "type": "threshold_out_of_range",
                "value": 1.5,
                "enabled": True,
            },
            {
                "id": "INJ-CHUNK-INDIVISIBLE",
                "sample": "sample-06",
                "stage": "visual_feature",
                "type": "chunk_not_divisible",
                "chunk_size": 30,
                "enabled": True,
            },
        ],
    }


def generate_dataset(
    samples_dir: Path | str,
    config_dir: Path | str,
    *,
    seed: int = 20261122,
    sample_count: int = 8,
) -> dict[str, Any]:
    samples_dir = Path(samples_dir)
    config_dir = Path(config_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(seed)
    names: list[str] = []
    for index in range(sample_count):
        sample_name = f"sample-{index + 1:02d}"
        names.append(sample_name)
        sample_rng = random.Random(master_rng.getrandbits(64))
        directory = samples_dir / sample_name
        directory.mkdir(parents=True, exist_ok=True)
        image, regions = _build_scene(index, sample_rng)
        image.save(directory / "image.png", format="PNG", optimize=False)
        _write_json(
            directory / "regions.json",
            {
                "schema_version": 1,
                "sample_name": sample_name,
                "image_size": [image.width, image.height],
                "regions": regions,
            },
        )
        _write_json(directory / "text.json", _build_text(sample_name, regions))

    _write_json(config_dir / "alignment.json", default_alignment_config())
    _write_json(config_dir / "injections.json", {"schema_version": 1, "injections": []})
    _write_json(config_dir / "injections.demo.json", demo_injections())
    return {
        "seed": seed,
        "sample_count": sample_count,
        "samples": names,
        "samples_dir": str(samples_dir),
        "config_dir": str(config_dir),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "chunk_size": CHUNK_SIZE,
    }
