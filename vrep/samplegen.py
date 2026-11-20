"""确定性合成样本生成器。

固定种子在浅色背景上绘制几何图元（矩形/椭圆/三角形）与纹理块，
每图输出 PNG 与同名区域真值 JSON；同时生成表征配置、编辑命令流、
注入配置。多次运行逐字节一致。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .hashing import deterministic_rng
from .jsonio import write_json

CATEGORIES = ("rectangle", "ellipse", "triangle", "texture_block")
BACKGROUND = 240


def _draw_rectangle(canvas, rng, size):
    x = int(rng.integers(4, size // 2))
    y = int(rng.integers(4, size // 2))
    w = int(rng.integers(size // 6, size // 3))
    h = int(rng.integers(size // 6, size // 3))
    w = min(w, size - x - 2)
    h = min(h, size - y - 2)
    color = tuple(int(value) for value in rng.integers(40, 200, size=3))
    canvas[y : y + h, x : x + w, :] = color
    return {"category": "rectangle", "box": [x, y, w, h]}


def _draw_ellipse(canvas, rng, size):
    cx = int(rng.integers(size // 4, 3 * size // 4))
    cy = int(rng.integers(size // 4, 3 * size // 4))
    rx = int(rng.integers(size // 8, size // 4))
    ry = int(rng.integers(size // 8, size // 4))
    color = tuple(int(value) for value in rng.integers(40, 200, size=3))
    yy, xx = np.ogrid[:size, :size]
    mask = ((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2 <= 1
    canvas[mask] = color
    x = max(0, cx - rx)
    y = max(0, cy - ry)
    w = min(size - x, 2 * rx + 1)
    h = min(size - y, 2 * ry + 1)
    return {"category": "ellipse", "box": [x, y, w, h]}


def _draw_triangle(canvas, rng, size):
    points = sorted(
        (int(rng.integers(4, size - 4)), int(rng.integers(4, size - 4)))
        for _ in range(3)
    )
    color = tuple(int(value) for value in rng.integers(40, 200, size=3))
    (x1, y1), (x2, y2), (x3, y3) = points
    yy, xx = np.mgrid[0:size, 0:size]
    d1 = (xx - x2) * (y1 - y2) - (x1 - x2) * (yy - y2)
    d2 = (xx - x3) * (y2 - y3) - (x2 - x3) * (yy - y3)
    d3 = (xx - x1) * (y3 - y1) - (x3 - x1) * (yy - y1)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    mask = ~(has_neg & has_pos)
    canvas[mask] = color
    xs = [x1, x2, x3]
    ys = [y1, y2, y3]
    x0, y0 = min(xs), min(ys)
    return {
        "category": "triangle",
        "box": [x0, y0, max(xs) - x0 + 1, max(ys) - y0 + 1],
    }


def _draw_texture(canvas, rng, size):
    x = int(rng.integers(4, size // 2))
    y = int(rng.integers(4, size // 2))
    w = int(rng.integers(size // 5, size // 3))
    h = int(rng.integers(size // 5, size // 3))
    w = min(w, size - x - 2)
    h = min(h, size - y - 2)
    base = int(rng.integers(60, 180))
    checker = (np.indices((h, w)).sum(axis=0) % 2) * 40
    patch = np.clip(base + checker, 0, 255).astype(np.uint8)
    canvas[y : y + h, x : x + w, 0] = patch
    canvas[y : y + h, x : x + w, 1] = np.roll(patch, 2, axis=1)
    canvas[y : y + h, x : x + w, 2] = np.roll(patch, 3, axis=0)
    return {"category": "texture_block", "box": [x, y, w, h]}


_DRAWERS = (_draw_rectangle, _draw_ellipse, _draw_triangle, _draw_texture)


def render_image(index: int, size: int, base_seed: int):
    rng = deterministic_rng(f"image|{base_seed}|{index:04d}")
    canvas = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    regions = []
    region_count = int(rng.integers(3, 6))
    for _ in range(region_count):
        drawer = _DRAWERS[int(rng.integers(0, len(_DRAWERS)))]
        regions.append(drawer(canvas, rng, size))
    regions.sort(key=lambda region: (region["box"][1], region["box"][0],
                                     region["category"]))
    return canvas, {"image": f"img_{index:04d}", "regions": regions}


def _base_config(image_size: int, tile_size: int) -> dict:
    return {
        "tile_size": tile_size,
        "color_dim": 6,
        "gradient_dim": 6,
        "embed_dim": 8,
        "quant_level": 16,
        "fixed_bits": 12,
        "similarity_threshold": 0.85,
        "max_tiles_per_image": (image_size // tile_size) ** 2,
        "worker_count": 4,
        "point_budget_per_batch": 24,
        "snapshot_commit_interval": 2,
    }


def _measure_boundary_threshold(output_dir: str, names: list[str],
                                base_config: dict) -> float:
    """生成后用真实管线计算相似度，选取中位值作为阈值，制造稳定边界样本。"""
    from .config import Config
    from .engine import Engine
    from .features import cosine_similarity
    from .injection import InjectionManager

    # 探针使用独立基础阈值，避免读取 output_dir 中可能存在的旧配置形成自引用。
    probe_config_dict = dict(base_config)
    probe_config_dict["similarity_threshold"] = 0.0
    config = Config.from_dict(probe_config_dict)
    import tempfile

    probe_root = tempfile.mkdtemp(prefix="vrep-probe-")
    engine = Engine(
        os.path.join(probe_root, "probe"), InjectionManager(None)
    )
    engine.seed_library(output_dir)
    batch_no = 1
    engine.run_batch(config, batch_no=batch_no)
    while engine.ledger["runtime"]["pending"]:
        before = len(engine.ledger["runtime"]["pending"])
        batch_no += 1
        engine.run_batch(config, batch_no=batch_no)
        if len(engine.ledger["runtime"]["pending"]) >= before:
            break
    vectors = {
        name: entry["embedding"]["vector"]
        for name, entry in engine.ledger["images"].items()
        if entry["embedding"] is not None
    }
    similarities = []
    ordered = sorted(vectors)
    for i, left_name in enumerate(ordered):
        for right_name in ordered[i + 1:]:
            similarities.append(
                cosine_similarity(vectors[left_name], vectors[right_name])
            )
    # 清理探针工作区，避免污染样本目录
    import shutil

    shutil.rmtree(os.path.join(output_dir, "_probe_workspace"), ignore_errors=True)
    if not similarities:
        return base_config["similarity_threshold"]
    similarities.sort()
    return float(similarities[len(similarities) // 2])


def generate_all(output_dir: str, image_size: int = 96, count: int = 12,
                 seed: int = 20261121) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    tile_size = 16 if image_size % 16 == 0 else image_size // 4
    names = []
    for index in range(count):
        name = f"img_{index:04d}"
        canvas, truth = render_image(index, image_size, seed)
        Image.fromarray(canvas, mode="RGB").save(
            os.path.join(output_dir, f"{name}.png"), format="PNG"
        )
        write_json(os.path.join(output_dir, f"{name}_truth.json"), truth)
        names.append(name)

    # 替换用变体（与原图同名内容不同的图像放在 pool 子目录，命令流引用其副本名）
    pool_dir = os.path.join(output_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    variant_names = []
    for index in (2, 7):
        variant = f"img_{index:04d}_v2"
        canvas, truth = render_image(10_000 + index, image_size, seed)
        Image.fromarray(canvas, mode="RGB").save(
            os.path.join(pool_dir, f"{variant}.png"), format="PNG"
        )
        truth["image"] = variant
        write_json(os.path.join(pool_dir, f"{variant}_truth.json"), truth)
        variant_names.append(variant)
    extra = "img_0099"
    canvas, truth = render_image(99, image_size, seed)
    Image.fromarray(canvas, mode="RGB").save(
        os.path.join(pool_dir, f"{extra}.png"), format="PNG"
    )
    truth["image"] = extra
    write_json(os.path.join(pool_dir, f"{extra}_truth.json"), truth)
    variant_names.append(extra)

    config = _base_config(image_size, tile_size)
    boundary_threshold = _measure_boundary_threshold(output_dir, names, config)
    config["similarity_threshold"] = boundary_threshold
    write_json(os.path.join(output_dir, "config.json"), config)

    # 多份配置文件用于配置编辑命令
    config_tile = dict(config)
    config_tile["tile_size"] = tile_size * 2
    config_tile["max_tiles_per_image"] = (image_size // config_tile["tile_size"]) ** 2
    write_json(os.path.join(output_dir, "config_tile.json"), config_tile)

    config_embed = dict(config)
    config_embed["embed_dim"] = 12
    write_json(os.path.join(output_dir, "config_embed.json"), config_embed)

    config_quant = dict(config)
    config_quant["quant_level"] = 8
    config_quant["fixed_bits"] = 8
    write_json(os.path.join(output_dir, "config_quant.json"), config_quant)

    config_threshold = dict(config)
    config_threshold["similarity_threshold"] = boundary_threshold
    write_json(os.path.join(output_dir, "config_threshold.json"), config_threshold)

    commands = [
        {"op": "replace", "image": "img_0002", "source": "pool/img_0002_v2.png"},
        {"op": "add", "image": "img_0099", "source": "pool/img_0099.png"},
        {"op": "config", "config_path": "config_tile.json"},
        {"op": "delete", "image": "img_0005"},
        {"op": "config", "config_path": "config_embed.json"},
        {"op": "replace", "image": "img_0007", "source": "pool/img_0007_v2.png"},
        {"op": "config", "config_path": "config_quant.json"},
        {"op": "add", "image": "img_0005", "source": "img_0005.png"},
        {"op": "config", "config_path": "config_threshold.json"},
        {"op": "config", "config_path": "config.json"},
    ]
    write_json(os.path.join(output_dir, "commands.json"),
               {"commands": commands})

    injections = {
        "injections": [
            {"id": "INJ-001", "type": "truncate_png", "image": "img_0003",
             "keep_bytes": 40},
            {"id": "INJ-002", "type": "truth_out_of_bounds", "image": "img_0006"},
            {"id": "INJ-003", "type": "nonfinite_feature", "image": "img_0001",
             "tile_seq": 0},
            {"id": "INJ-004", "type": "quant_level_out_of_range", "bad_level": 13},
        ]
    }
    write_json(os.path.join(output_dir, "injections.json"), injections)

    return {
        "samples_dir": output_dir,
        "image_size": image_size,
        "tile_size": tile_size,
        "image_count": count,
        "images": names,
        "pool": variant_names,
        "config_files": [
            "config.json",
            "config_tile.json",
            "config_embed.json",
            "config_quant.json",
            "config_threshold.json",
        ],
        "commands": len(commands),
        "similarity_threshold": boundary_threshold,
    }
