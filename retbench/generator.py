"""固定种子的样本/配置/命令流/注入清单生成器。

输出布局（均为本地文件、无需网络）：
    <out>/
      config.yaml
      commands.json
      injections.json
      samples/
        s000.png  s000.regions.json  s000.text.json ...
        <注入故障样本>（由 injections.json 声明，加载时进入失败清单）
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

from .miniyaml import parse_yaml  # 仅用于生成后自检
from .rng import DEFAULT_SEED, child_rng, root_rng

SHAPES = ("rect", "circle", "triangle")
PALETTE = (
    (210, 60, 60),
    (60, 150, 80),
    (60, 90, 200),
    (200, 150, 40),
    (140, 70, 180),
)
BG = (245, 245, 245)


def _dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    """与 miniyaml 子集对齐的最小 YAML 写出器。"""

    lines: list[str] = []
    pad = "  " * indent

    def emit(key: str, value: Any, level: int) -> None:
        prefix = "  " * level
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for sub_key, sub_value in value.items():
                emit(sub_key, sub_value, level + 1)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                lines.append(f"{prefix}{key}:")
                for row in value:
                    first = True
                    for sub_key, sub_value in row.items():
                        if first:
                            lines.append(f"{prefix}  - {sub_key}: {_scalar(sub_value)}")
                            first = False
                        else:
                            lines.append(f"{prefix}    {sub_key}: {_scalar(sub_value)}")
            else:
                lines.append(f"{prefix}{key}: [{', '.join(_scalar(v) for v in value)}]")
        else:
            lines.append(f"{prefix}{key}: {_scalar(value)}")

    for key, value in data.items():
        emit(key, value, indent)
    return "\n".join(lines) + "\n"


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str) and (value == "" or not value.replace("_", "").isalnum()):
        return json.dumps(value)
    return str(value)


def default_config() -> dict[str, Any]:
    return {
        "resolution": [64, 64],
        "vocabulary": {
            "entities": ["circle", "square", "triangle", "diamond"],
            "attributes": ["red", "green", "blue", "yellow", "purple", "large", "small"],
            "relations": ["left_of", "right_of", "above", "below", "same_color", "near"],
        },
        "image": {"tile_size": 16, "visual_dim": 24, "noise_level": 6},
        "text": {"text_dim": 24, "default_assertion_count": 4},
        "alignment": {"weights": {"color": 0.5, "gradient": 0.3, "edge": 0.2}},
        "recall": {
            "channels": {"lexical": True, "visual": True, "hybrid": True},
            "channel_weights": {"lexical": 0.4, "visual": 0.35, "hybrid": 0.25},
            "fusion_temperature": 1.0,
            "candidate_cap": 5,
        },
        "quantization": {"frac_bits": 12, "tier": "standard"},
        "scale": {"tiers": [4, 8, 16], "active_tier": 4},
        "interaction": {
            "latency_budget_ms": 200,
            "thread_pool_size": 2,
            "work_ms_per_task": 0,
        },
        "snapshot": {
            "commit_interval_commands": 2,
            "temp_dir": "_tmp",
            "snapshot_dir": "snapshots",
        },
        "strategy_table": [
            {
                "id": "ST_FULL",
                "capacity_samples": 6,
                "capacity_primitives": 24,
                "capacity_assertion_len": 6.0,
                "unit_cost_limit": 5_000,
                "feature_precision": "full",
                "tile_range": "full",
                "channels": ["lexical", "visual", "hybrid"],
                "cap_discount": 1.0,
            },
            {
                "id": "ST_REDUCED",
                "capacity_samples": 100000,
                "capacity_primitives": 100000,
                "capacity_assertion_len": 1000.0,
                "unit_cost_limit": 10_000_000,
                "feature_precision": "reduced",
                "tile_range": "center_half",
                "channels": ["lexical", "visual"],
                "cap_discount": 0.6,
            },
        ],
    }


def _draw_shape(canvas: np.ndarray, rng: np.random.Generator,
                width: int, height: int) -> tuple[dict[str, Any], tuple[int, int, int]]:
    shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
    color = PALETTE[int(rng.integers(0, len(PALETTE)))]
    size = int(rng.integers(8, max(9, min(width, height) // 3)))
    x0 = int(rng.integers(2, max(3, width - size - 2)))
    y0 = int(rng.integers(2, max(3, height - size - 2)))
    x1, y1 = min(width - 1, x0 + size), min(height - 1, y0 + size)

    if shape == "rect":
        canvas[y0:y1, x0:x1] = color
    elif shape == "circle":
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        radius = size // 2
        for yy in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for xx in range(max(0, cx - radius), min(width, cx + radius + 1)):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2:
                    canvas[yy, xx] = color
    else:  # triangle
        for yy in range(y0, y1):
            half = max(1, int(size * (yy - y0) / max(1, size)) // 2)
            mid = (x0 + x1) // 2
            canvas[yy, max(0, mid - half):min(width, mid + half + 1)] = color

    return {"shape": shape, "bbox": [x0, y0, x1 - x0, y1 - y0], "color": list(color)}, color


def _build_text(name: str, regions: list[dict[str, Any]], vocab: dict[str, list[str]],
                rng: np.random.Generator, assertion_count: int) -> dict[str, Any]:
    entity = vocab["entities"][int(rng.integers(0, len(vocab["entities"])))]
    attr_count = int(rng.integers(1, 4))
    attributes = list(
        vocab["attributes"][int(rng.integers(0, len(vocab["attributes"])))]
        for _ in range(attr_count)
    )
    attributes = sorted(set(attributes))
    assertions = []
    relations = vocab["relations"]
    objects = vocab["entities"]
    for _ in range(assertion_count):
        assertions.append({
            "predicate": relations[int(rng.integers(0, len(relations)))],
            "object": objects[int(rng.integers(0, len(objects)))],
        })
    return {
        "sample": name,
        "entity": entity,
        "attributes": attributes,
        "assertions": assertions,
    }


def generate_sample(name: str, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    width, height = cfg["resolution"]
    canvas = np.tile(np.array(BG, dtype=np.uint8), (height, width, 1))
    shape_count = int(rng.integers(2, 5))
    regions = []
    for _ in range(shape_count):
        region, _ = _draw_shape(canvas, rng, width, height)
        regions.append(region)

    noise_level = int(cfg["image"].get("noise_level", 0))
    if noise_level:
        noise = rng.integers(-noise_level, noise_level + 1, size=canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    text_rng = child_rng(rng, f"text:{name}")
    text = _build_text(
        name, regions, cfg["vocabulary"], text_rng,
        int(cfg["text"].get("default_assertion_count", 4)),
    )
    return {
        "image": canvas,
        "regions": {"sample": name, "regions": regions},
        "text": text,
    }


def generate_workspace(out_dir: str, sample_count: int = 12,
                       seed: int = DEFAULT_SEED,
                       cfg_overrides: dict[str, Any] | None = None,
                       inject_count: int = 2) -> dict[str, str]:
    rng = root_rng(seed)
    cfg = default_config()
    if cfg_overrides:
        _deep_update(cfg, cfg_overrides)

    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    paths: dict[str, str] = {}

    names = [f"s{index:03d}" for index in range(sample_count)]
    injections: list[dict[str, Any]] = []
    vocab = cfg["vocabulary"]

    for index, name in enumerate(names):
        sample_rng = child_rng(rng, f"sample:{name}")
        payload = generate_sample(name, cfg, sample_rng)
        image_path = os.path.join(sample_dir, f"{name}.png")
        Image.fromarray(payload["image"], mode="RGB").save(image_path, format="PNG")
        with open(os.path.join(sample_dir, f"{name}.regions.json"), "w", encoding="utf-8") as handle:
            json.dump(payload["regions"], handle, ensure_ascii=False, indent=2)
        with open(os.path.join(sample_dir, f"{name}.text.json"), "w", encoding="utf-8") as handle:
            json.dump(payload["text"], handle, ensure_ascii=False, indent=2)

    _inject_failures(sample_dir, names, injections, vocab, inject_count)

    config_path = os.path.join(out_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(_dump_yaml(cfg))
    with open(config_path, encoding="utf-8") as yaml_handle:
        parse_yaml(yaml_handle.read())  # 生成即自检

    commands = default_commands(cfg, names, injections)
    command_path = os.path.join(out_dir, "commands.json")
    with open(command_path, "w", encoding="utf-8") as handle:
        json.dump(commands, handle, ensure_ascii=False, indent=2)

    injection_path = os.path.join(out_dir, "injections.json")
    with open(injection_path, "w", encoding="utf-8") as handle:
        json.dump(injections, handle, ensure_ascii=False, indent=2)

    paths.update(config=config_path, commands=command_path,
                 injections=injection_path, samples=sample_dir)
    return paths


def _inject_failures(sample_dir: str, names: list[str],
                     injections: list[dict[str, Any]], vocab: dict[str, list[str]],
                     inject_count: int) -> None:
    if not names or inject_count <= 0:
        return
    bad_region = names[-1]
    region_path = os.path.join(sample_dir, f"{bad_region}.regions.json")
    with open(region_path, encoding="utf-8") as handle:
        region_data = json.load(handle)
    region_data["regions"][0]["bbox"] = [-5, 2, 10, 10]
    with open(region_path, "w", encoding="utf-8") as handle:
        json.dump(region_data, handle, ensure_ascii=False, indent=2)
    injections.append({"sample": bad_region, "kind": "region_out_of_bounds"})

    if inject_count >= 2 and len(names) >= 2:
        bad_text = names[-2]
        text_path = os.path.join(sample_dir, f"{bad_text}.text.json")
        with open(text_path, encoding="utf-8") as handle:
            text_data = json.load(handle)
        text_data["entity"] = "__unknown_entity__"
        with open(text_path, "w", encoding="utf-8") as handle:
            json.dump(text_data, handle, ensure_ascii=False, indent=2)
        injections.append({"sample": bad_text, "kind": "entity_not_in_vocabulary"})


def default_commands(cfg: dict[str, Any], names: list[str],
                     injections: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    injected = {row["sample"] for row in injections}
    valid = [name for name in names if name not in injected]
    target = valid[0] if valid else names[0]
    commands = [
        {"seq": 0, "op": "set_scale_tier", "value": cfg["scale"]["tiers"][-1]},
        {"seq": 1, "op": "set_weight", "channel": "color", "value": 0.7},
        {"seq": 2, "op": "replace_image", "sample": target},
        {"seq": 3, "op": "replace_text", "sample": target},
        {"seq": 4, "op": "disable_channel", "channel": "hybrid"},
        {"seq": 5, "op": "enable_channel", "channel": "hybrid"},
        {"seq": 6, "op": "set_quant", "frac_bits": 16, "tier": "fine"},
        {"seq": 7, "op": "set_scale_tier", "value": cfg["scale"]["tiers"][0]},
        {"seq": 8, "op": "set_quant", "frac_bits": 12, "tier": "standard"},
    ]
    return commands


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
