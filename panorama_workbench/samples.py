from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache import atomic_write_json, atomic_write_bytes
from .config import WorkbenchConfig, parse_config
from .inference import category_color


@dataclass(frozen=True)
class DemoPaths:
    commands: Path
    config: Path
    replacements: Path
    samples: Path


def generate_demo(root: str | Path, image_count: int = 4) -> DemoPaths:
    target = Path(root).resolve()
    samples_dir = target / "samples"
    replacements_dir = target / "replacements"
    samples_dir.mkdir(parents=True, exist_ok=True)
    replacements_dir.mkdir(parents=True, exist_ok=True)
    for old in list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.json")):
        old.unlink()
    for old in list(replacements_dir.glob("*")):
        if old.is_file():
            old.unlink()

    payload = _demo_config_payload()
    config = parse_config(payload)
    config_path = target / "config.yaml"
    atomic_write_bytes(config_path, render_config_yaml(payload).encode("utf-8"))

    for index in range(1, image_count + 1):
        name = f"scene_{index:02d}.png"
        rgb, truth = _render_scene(config, 7000 + index, 64, 64)
        Image.fromarray(rgb, mode="RGB").save(samples_dir / name)
        atomic_write_json(samples_dir / name.replace(".png", ".json"), truth)

    replacement_name = "scene_02.png" if image_count >= 2 else "scene_01.png"
    rgb, truth = _render_scene(config, 9902, 64, 64)
    atomic_write_bytes(replacements_dir / replacement_name, _png_bytes(rgb))
    atomic_write_json(
        replacements_dir / replacement_name.replace(".png", ".json"), truth
    )

    active_quant = config.quant_levels[-1]
    commands = {
        "commands": [
            {
                "image": replacement_name,
                "op": "replace_image",
                "source": f"replacements/{replacement_name}",
            },
            {
                "category": {"code": 4, "name": "accent", "weight": 0.001},
                "op": "add_category",
            },
            {"op": "set_merge_iou", "value": 0.12},
            {"op": "set_min_instance_area", "value": 28},
            {"op": "set_quant_level", "value": active_quant},
        ]
    }
    commands_path = target / "commands.json"
    atomic_write_json(commands_path, commands)
    return DemoPaths(
        commands=commands_path,
        config=config_path,
        replacements=replacements_dir,
        samples=samples_dir,
    )


def render_config_yaml(payload: dict[str, Any]) -> str:
    categories = payload["categories"]
    window = payload["window"]
    quantization = payload["quantization"]
    scheduling = payload["scheduling"]
    quality = payload["quality"]
    lines = ["categories:"]
    for category in categories:
        lines.extend(
            [
                f"  - code: {category['code']}",
                f"    name: {category['name']}",
                f"    weight: {category['weight']}",
            ]
        )
    lines.extend(
        [
            "window:",
            f"  size: {window['size']}",
            f"  stride: {window['stride']}",
            "merge:",
            f"  iou: {payload['merge_iou']}",
            f"  min_instance_area: {payload['min_instance_area']}",
            "quantization:",
            f"  fixed_bits: {quantization['fixed_bits']}",
            f"  levels: {list(quantization['levels'])}",
            f"  level: {quantization['level']}",
            "scheduling:",
            f"  max_workers: {scheduling['max_workers']}",
            f"  pixel_budget: {scheduling['pixel_budget']}",
            f"  max_instances_per_image: {scheduling['max_instances_per_image']}",
            f"  cache_budget: {scheduling['cache_budget']}",
            f"  snapshot_interval: {scheduling['snapshot_interval']}",
            "quality:",
            f"  min_semantic_accuracy: {quality['min_semantic_accuracy']}",
            f"  min_instance_quality: {quality['min_instance_quality']}",
            f"  instance_match_iou: {quality['instance_match_iou']}",
            "",
        ]
    )
    return "\n".join(lines)


def _demo_config_payload() -> dict[str, Any]:
    base_categories = [
        {"code": 0, "name": "background", "weight": 1.0},
        {"code": 1, "name": "shape_a", "weight": 1.0},
        {"code": 2, "name": "shape_b", "weight": 1.0},
        {"code": 3, "name": "shape_c", "weight": 1.0},
    ]
    return {
        "cache_budget": 1024,
        "categories": base_categories,
        "instance_match_iou": 0.35,
        "max_instances_per_image": 24,
        "merge_iou": 0.08,
        "min_instance_area": 20,
        "pixel_budget": 6000000,
        "quant_level": 4,
        "quant_levels": [1, 4, 16, 64],
        "quality": {
            "instance_match_iou": 0.35,
            "min_instance_quality": 0.12,
            "min_semantic_accuracy": 0.45,
        },
        "scheduling": {
            "cache_budget": 1024,
            "max_instances_per_image": 24,
            "max_workers": 3,
            "pixel_budget": 6000000,
            "snapshot_interval": 2,
        },
        "snapshot_interval": 2,
        "window": {"size": 16, "stride": 8},
        "fixed_bits": 10,
        "quantization": {"fixed_bits": 10, "level": 4, "levels": [1, 4, 16, 64]},
        "merge_iou": 0.08,
        "min_instance_area": 20,
    }


def _render_scene(
    config: WorkbenchConfig, seed: int, width: int, height: int
) -> tuple[np.ndarray, dict[str, Any]]:
    generator = random.Random(seed)
    background = np.asarray(category_color(config, config.category_by_code[0]), dtype=np.uint8)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = background
    owner = np.zeros((height, width), dtype=np.int32)
    instances_meta: dict[int, dict[str, Any]] = {}
    state_categories = [category for category in config.categories if category.code != 0]
    if not state_categories:
        raise ValueError("demo generator requires a non-background category")

    attempts = 0
    next_id = 1
    while len(instances_meta) < min(6, 3 + seed % 3) and attempts < 80:
        attempts += 1
        category = state_categories[(seed + attempts) % len(state_categories)]
        shape_width = generator.randint(12, 27)
        shape_height = generator.randint(12, 27)
        x = generator.randint(1, max(1, width - shape_width - 2))
        y = generator.randint(1, max(1, height - shape_height - 2))
        mask = np.zeros((height, width), dtype=bool)
        if (seed + attempts) % 3 == 0:
            yy, xx = np.ogrid[:height, :width]
            center_x = x + shape_width / 2.0
            center_y = y + shape_height / 2.0
            radius_x = shape_width / 2.0
            radius_y = shape_height / 2.0
            mask = (
                ((xx - center_x) / radius_x) ** 2
                + ((yy - center_y) / radius_y) ** 2
                <= 1.0
            )
        else:
            mask[y : y + shape_height, x : x + shape_width] = True
        if int(mask.sum()) < config.min_instance_area:
            continue
        visible_before = owner.copy()
        base = np.asarray(category_color(config, category), dtype=np.int32)
        jitter = np.asarray(
            [generator.randint(-3, 3) for _ in range(3)], dtype=np.int32
        )
        color = np.clip(base + jitter, 0, 255).astype(np.uint8)
        image[mask] = color
        owner[mask] = next_id
        instances_meta[next_id] = {
            "category": category.code,
            "original_mask": mask.copy(),
            "prior": visible_before,
        }
        next_id += 1

    noise = np.asarray(
        [generator.randint(-2, 2) for _ in range(height * width * 3)], dtype=np.int16
    ).reshape(height, width, 3)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    truth_instances: list[dict[str, Any]] = []
    for instance_id in sorted(instances_meta):
        meta = instances_meta[instance_id]
        visible = owner == instance_id
        area = int(visible.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(visible)
        pixels = [[int(x), int(y)] for y, x in zip(ys.tolist(), xs.tolist())]
        truth_instances.append(
            {
                "area": area,
                "bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ],
                "category": meta["category"],
                "id": instance_id,
                "mask_pixels": pixels,
            }
        )
    return image, {
        "height": height,
        "instances": truth_instances,
        "width": width,
    }


def _png_bytes(rgb: np.ndarray) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
