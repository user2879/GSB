"""固定种子的本地确定性样本生成器。

不访问网络与外部数据：所有 PNG 由 NumPy 随机数（固定种子）绘制几何图元
（圆/方/三角）与纹理块生成，文本断言由受控词表组合实体与属性列表，
区域真值与绘制坐标保持一致。另生成若干故意非法的样本用于失败清单验证。
"""
from __future__ import annotations

import json
import os

import numpy as np

from .pngio import write_png
from .validation import CONTROLLED_ENTITIES

WIDTH = 64
HEIGHT = 64
PALETTE = {
    "red": (200, 60, 60),
    "green": (60, 180, 80),
    "blue": (70, 90, 210),
    "yellow": (210, 200, 70),
    "purple": (160, 70, 190),
}
TEXTURES = ("flat", "dots", "stripes")
SIZES = ("small", "medium", "large")
ORIENTATIONS = ("north", "east", "south", "west")
POSITIONS = ("top", "center", "bottom")


def _draw_shape(canvas: np.ndarray, rng: np.random.RandomState, shape: dict) -> None:
    x, y, size = shape["x"], shape["y"], shape["px_size"]
    color = np.array(PALETTE[shape["color"]], dtype=np.uint8)
    kind = shape["type"]
    if kind == "square" or kind == "block":
        canvas[y : y + size, x : x + size] = color
    elif kind == "circle":
        yy, xx = np.ogrid[:HEIGHT, :WIDTH]
        mask = (xx - (x + size // 2)) ** 2 + (yy - (y + size // 2)) ** 2 <= (size // 2) ** 2
        canvas[mask] = color
    elif kind == "triangle":
        for row in range(size):
            half = max(1, int(size * (1 - row / max(1, size)) / 2))
            canvas[y + row, x + half : x + size - half] = color
    elif kind == "line":
        canvas[y : y + max(1, size // 4), x : x + size] = color
    # 纹理块叠加（确定性扰动）
    if shape["texture"] == "dots":
        for dy in range(2, size, 6):
            for dx in range(2, size, 6):
                if 0 <= y + dy < HEIGHT and 0 <= x + dx < WIDTH:
                    canvas[y + dy, x + dx] = (245, 245, 245)
    elif shape["texture"] == "stripes":
        for dy in range(0, size, 4):
            if 0 <= y + dy < HEIGHT:
                canvas[y + dy, x : x + size] = np.clip(
                    canvas[y + dy, x : x + size].astype(np.int16) + 40, 0, 255
                ).astype(np.uint8)


def _make_shape(rng: np.random.RandomState, index: int) -> dict:
    size = rng.choice([10, 14, 18, 22])
    x = int(rng.integers(0, WIDTH - size))
    y = int(rng.integers(0, HEIGHT - size))
    return {
        "type": CONTROLLED_ENTITIES[index % len(CROLLED_ENTITIES)],
        "color": list(PALETTE)[index % len(PALETTE)],
        "size": SIZES[index % len(SIZES)],
        "texture": TEXTURES[index % len(TEXTURES)],
        "orientation": ORIENTATIONS[index % len(ORIENTATIONS)],
        "position": POSITIONS[index % len(POSITIONS)],
        "x": x,
        "y": y,
        "px_size": int(size),
    }


CROLLED_ENTITIES = CONTROLLED_ENTITIES


def generate_sample(seed: int, sid: str) -> dict:
    rng = np.random.default_rng(seed)
    canvas = np.full((HEIGHT, WIDTH, 3), rng.integers(20, 60, size=3), dtype=np.uint8)
    shape_count = int(rng.integers(2, 5))
    shapes = [_make_shape(rng, idx) for idx in range(shape_count)]
    for shape in shapes:
        _draw_shape(canvas, rng, shape)
    entities = []
    regions = []
    for idx, shape in enumerate(shapes):
        entity_id = f"{sid}-e{idx}"
        entities.append(
            {
                "id": entity_id,
                "type": shape["type"],
                "attributes": {
                    "color": shape["color"],
                    "size": shape["size"],
                    "texture": shape["texture"],
                    "position": shape["position"],
                    "orientation": shape["orientation"],
                },
            }
        )
        regions.append(
            {
                "label": entity_id,
                "x": shape["x"],
                "y": shape["y"],
                "w": shape["px_size"],
                "h": shape["px_size"],
            }
        )
    assertion = {"sample_id": sid, "entities": entities}
    return {"pixels": canvas, "assertion": assertion, "regions": regions, "shapes": shapes}


def write_sample(out_dir: str, sid: str, sample: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    write_png(os.path.join(out_dir, f"{sid}.png"), sample["pixels"])
    with open(os.path.join(out_dir, f"{sid}.assertion.json"), "w", encoding="utf-8") as fh:
        json.dump(sample["assertion"], fh, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, f"{sid}.regions.json"), "w", encoding="utf-8") as fh:
        json.dump(sample["regions"], fh, ensure_ascii=False, indent=2, sort_keys=True)


def default_config() -> dict:
    return {
        "signature_bits": 64,
        "bucket_bits": 8,
        "similarity_permille": 850,
        "seed_version": "seed-v1",
        "bucket_capacity": 4,
        "frac_bits": 12,
        "tile_size": 8,
        "image_weight": 2,
        "text_weight": 3,
        "max_workers": 4,
        "global_budget": 5_000_000,
        "declared_seed_versions": ["seed-v1", "seed-v2", "seed-v3"],
    }


def generate_workspace(workspace: str, sample_count: int = 10, base_seed: int = 20261125) -> dict:
    """生成完整演示工作区：样本、近重复副本、非法样本、配置与命令流。"""
    sample_dir = os.path.join(workspace, "samples")
    invalid_dir = os.path.join(workspace, "invalid_samples")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    generated = {}
    for idx in range(sample_count):
        sid = f"s{idx:03d}"
        sample = generate_sample(base_seed + idx, sid)
        write_sample(sample_dir, sid, sample)
        generated[sid] = sample

    # 近重复样本：s000 的完全副本（不同 sid），保证至少产生一次真实合并
    dup = generate_sample(base_seed, "s100")
    write_sample(sample_dir, "s100", dup)
    # 近重复变体：复制 s001 但增加一个属性相同的实体（内容接近但不相同）
    variant = generate_sample(base_seed + 1, "s001")
    extra_entity = json.loads(json.dumps(variant["assertion"]["entities"][0]))
    extra_entity["id"] = "s101-eextra"
    variant["assertion"]["sample_id"] = "s101"
    variant["assertion"]["entities"].append(extra_entity)
    write_sample(sample_dir, "s101", variant)

    # 替换变体：replace s002 命令前，演示编排会把该目录提升到 samples/
    replacement = generate_sample(base_seed + 7777, "s002")
    write_sample(os.path.join(workspace, "replacements"), "s002", replacement)

    _generate_invalid_samples(invalid_dir, base_seed)

    with open(os.path.join(workspace, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(default_config(), fh, ensure_ascii=False, indent=2, sort_keys=True)
    commands = build_command_stream(sample_count)
    with open(os.path.join(workspace, "commands.json"), "w", encoding="utf-8") as fh:
        json.dump(commands, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return {"samples": generated, "commands": commands}


def _generate_invalid_samples(invalid_dir: str, base_seed: int) -> None:
    # 1) 区域落在图像外
    bad_region = generate_sample(base_seed + 900, "bad-region")
    bad_region["regions"][0]["x"] = WIDTH + 5
    write_sample(invalid_dir, "bad-region", bad_region)
    # 2) 断言实体类型不在受控词表
    bad_assert = generate_sample(base_seed + 901, "bad-assert")
    bad_assert["assertion"]["sample_id"] = "bad-assert"
    bad_assert["assertion"]["entities"][0]["type"] = "hexagon"
    write_sample(invalid_dir, "bad-assert", bad_assert)
    # 3) PNG 容器头损坏
    broken = generate_sample(base_seed + 902, "bad-png")
    write_sample(invalid_dir, "bad-png", broken)
    with open(os.path.join(invalid_dir, "bad-png.png"), "r+b") as fh:
        fh.seek(0)
        fh.write(b"\x00\x00\x00\x00")
    with open(os.path.join(invalid_dir, "bad-png.assertion.json"), "w", encoding="utf-8") as fh:
        json.dump(broken["assertion"], fh)
    with open(os.path.join(invalid_dir, "bad-png.regions.json"), "w", encoding="utf-8") as fh:
        json.dump(broken["regions"], fh)


def build_command_stream(sample_count: int) -> list[dict]:
    """覆盖新增/替换/删除/位宽调整/分桶位数变更/种子版本切换的命令流。"""
    ids = [f"s{idx:03d}" for idx in range(sample_count)]
    commands: list[dict] = []
    # 初始只安装一半样本，后续增量新增
    for sid in ids[: sample_count // 2]:
        commands.append({"cmd": "add", "sample_id": sid})
    commands.append({"cmd": "add", "sample_id": "s100"})
    for sid in ids[sample_count // 2 :]:
        commands.append({"cmd": "add", "sample_id": sid})
    commands.append({"cmd": "add", "sample_id": "s101"})
    # 单样本替换（生成器会重写 s002 的内容）
    commands.append({"cmd": "replace", "sample_id": "s002"})
    # 配置调整：签名位宽 64 -> 128
    commands.append({"cmd": "set_config", "config": {**default_config(), "signature_bits": 128}})
    # 分桶位数 8 -> 16（签名应被复用）
    commands.append({"cmd": "set_config", "config": {**default_config(), "signature_bits": 128, "bucket_bits": 16}})
    # 投影种子版本切换
    commands.append(
        {"cmd": "set_config", "config": {**default_config(), "signature_bits": 128, "bucket_bits": 16, "seed_version": "seed-v2"}}
    )
    # 回到原始配置后删除样本，验证簇标识重指定
    commands.append({"cmd": "set_config", "config": default_config()})
    commands.append({"cmd": "delete", "sample_id": "s000"})
    return commands

