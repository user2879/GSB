"""固定种子样本生成：PNG 图像（几何图元 + 噪声）与文本描述 JSONL。

用法: python scripts/generate_samples.py --output data/samples [--count 32]
同一参数多次生成结果逐字节一致。
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

SEED = 20261121
SIZE = 64
COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (60, 90, 220),
    "yellow": (230, 210, 40),
    "magenta": (200, 60, 200),
    "cyan": (50, 200, 200),
}
SHAPES = ("rectangle", "ellipse", "triangle")
POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")


def _draw_shape(draw, shape, color, box):
    if shape == "rectangle":
        draw.rectangle(box, fill=color)
    elif shape == "ellipse":
        draw.ellipse(box, fill=color)
    else:
        x0, y0, x1, y1 = box
        draw.polygon([(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)], fill=color)


def gen_sample(rng, sample_id):
    """生成单个样本的图像数组与断言列表。"""
    noise = rng.integers(0, 32, size=(SIZE, SIZE, 3), dtype=np.uint8)
    image = noise.copy()
    assertions = []
    n_shapes = int(rng.integers(2, 5))
    for k in range(n_shapes):
        entity = f"shape_{k}"
        shape = SHAPES[rng.integers(0, len(SHAPES))]
        color_name = list(COLORS)[rng.integers(0, len(COLORS))]
        position = POSITIONS[rng.integers(0, len(POSITIONS))]
        w = int(rng.integers(12, 28))
        h = int(rng.integers(12, 28))
        x0 = int(rng.integers(0, SIZE - w))
        y0 = int(rng.integers(0, SIZE - h))
        img = Image.fromarray(image, "RGB")
        _draw_shape(ImageDraw.Draw(img), shape, COLORS[color_name],
                    (x0, y0, x0 + w, y0 + h))
        image = np.asarray(img, dtype=np.uint8).copy()
        assertions.append({"entity": entity, "attribute": "shape",
                           "value": shape})
        assertions.append({"entity": entity, "attribute": "color",
                           "value": color_name})
        assertions.append({"entity": entity, "attribute": "position",
                           "value": position})
    # 每第 5 个样本：注入一条冲突断言（同一实体同一属性不同取值）
    if int(sample_id.rsplit("_", 1)[1]) % 5 == 0 and assertions:
        other = [c for c in COLORS if c != assertions[1]["value"]][0]
        assertions.append({"entity": assertions[0]["entity"],
                           "attribute": "color", "value": other})
    return image, assertions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/samples")
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)
    desc_path = os.path.join(args.output, "descriptions.jsonl")
    lines = []
    for idx in range(args.count):
        sample_id = f"sample_{idx:03d}"
        rng = np.random.default_rng(SEED + idx)
        image, assertions = gen_sample(rng, sample_id)
        # 每第 7 个样本：断言为空，走默认向量补齐路径
        entry = {"sample_id": sample_id}
        entry["assertions"] = [] if idx % 7 == 0 else assertions
        Image.fromarray(image, "RGB").save(
            os.path.join(images_dir, f"{sample_id}.png"), format="PNG")
        lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=True))
    with open(desc_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"generated {args.count} samples -> {args.output}")


if __name__ == "__main__":
    main()
