"""固定种子确定性样本生成器。

生成的每个样本包含：
  image.png   固定分辨率 RGB 图（几何图元 + 噪声），纯本地 PNG 编码
  regions.json 图像区域真值（图元类型、颜色、包围盒）
  text.json   结构化文本描述（实体、属性、断言列表）

不依赖任何随机系统源，随机数全部来自 numpy.random.RandomState(固定种子)。
"""

import argparse
import json
import os

import numpy as np

from .png_codec import encode_png

PALETTE = {
    "red": (200, 40, 40),
    "green": (40, 170, 70),
    "blue": (50, 80, 210),
    "yellow": (220, 200, 60),
    "cyan": (60, 200, 210),
    "magenta": (200, 70, 190),
}
SHAPES = ("rectangle", "circle", "triangle")
BASE_SEED = 20261118


def _draw_rectangle(canvas, x0, y0, x1, y1, color):
    canvas[y0:y1, x0:x1] = color


def _draw_circle(canvas, cx, cy, radius, color):
    h, w = canvas.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[mask] = color


def _draw_triangle(canvas, points, color):
    (x0, y0), (x1, y1), (x2, y2) = points
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    d1 = (xx - x1) * (y0 - y1) - (x0 - x1) * (yy - y1)
    d2 = (xx - x2) * (y1 - y2) - (x1 - x2) * (yy - y2)
    d3 = (xx - x0) * (y2 - y0) - (x2 - x0) * (yy - y0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    canvas[~(has_neg & has_pos)] = color


def generate_sample(index, image_size):
    """生成单个样本的图像数组、区域真值与文本描述。"""
    width, height = image_size
    rng = np.random.RandomState(BASE_SEED + index * 97)
    base = np.array([24 + rng.randint(0, 20) for _ in range(3)], dtype=np.int16)
    noise = rng.randint(0, 18, size=(height, width, 3), dtype=np.int16)
    canvas = np.clip(base + noise, 0, 255).astype(np.uint8).copy()

    n_shapes = 2 + (index % 2)
    color_names = list(PALETTE.keys())
    rng.shuffle(color_names)
    regions = []
    entities = []
    for s in range(n_shapes):
        shape = SHAPES[(index + s) % len(SHAPES)]
        color_name = color_names[s]
        color = PALETTE[color_name]
        margin = 6
        x0 = int(rng.randint(margin, max(margin + 1, width // 2)))
        y0 = int(rng.randint(margin, max(margin + 1, height // 2)))
        span_x = int(rng.randint(width // 6, width // 3))
        span_y = int(rng.randint(height // 6, height // 3))
        x1 = min(width - 1, x0 + span_x)
        y1 = min(height - 1, y0 + span_y)
        name = "shape_%d" % (s + 1)
        if shape == "rectangle":
            _draw_rectangle(canvas, x0, y0, x1, y1, color)
        elif shape == "circle":
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            radius = max(3, min(x1 - x0, y1 - y0) // 2)
            _draw_circle(canvas, cx, cy, radius, color)
            x0, y0 = cx - radius, cy - radius
            x1, y1 = cx + radius, cy + radius
        else:
            pts = [(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)]
            _draw_triangle(canvas, pts, color)
        regions.append({
            "id": name,
            "shape": shape,
            "color": color_name,
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
        })
        entities.append({
            "name": name,
            "type": "shape",
            "attributes": [
                {"name": "shape", "value": shape},
                {"name": "color", "value": color_name},
                {"name": "area", "value": int((x1 - x0) * (y1 - y0))},
            ],
        })

    assertions = []
    for s in range(1, n_shapes):
        prev_box = regions[s - 1]["bbox"]
        cur_box = regions[s]["bbox"]
        relation = "left_of" if cur_box[0] >= prev_box[0] else "right_of"
        assertions.append({
            "subject": regions[s - 1]["id"],
            "predicate": relation,
            "object": regions[s]["id"],
        })
    assertions.append({
        "subject": "scene",
        "predicate": "contains",
        "object": regions[0]["id"],
    })
    text = {
        "scene_id": "scene_%03d" % index,
        "entities": entities + [{"name": "scene", "type": "scene",
                                 "attributes": [
                                     {"name": "shape_count",
                                      "value": n_shapes}]}],
        "assertions": assertions,
    }
    return canvas, regions, text


def write_sample(out_dir, index, image_size):
    sample_dir = os.path.join(out_dir, "sample_%03d" % index)
    os.makedirs(sample_dir, exist_ok=True)
    canvas, regions, text = generate_sample(index, image_size)
    with open(os.path.join(sample_dir, "image.png"), "wb") as fh:
        fh.write(encode_png(canvas))
    with open(os.path.join(sample_dir, "regions.json"), "w",
              encoding="utf-8") as fh:
        json.dump(regions, fh, sort_keys=True, ensure_ascii=False,
                  indent=2)
    with open(os.path.join(sample_dir, "text.json"), "w",
              encoding="utf-8") as fh:
        json.dump(text, fh, sort_keys=True, ensure_ascii=False, indent=2)
    return os.path.basename(sample_dir)


def generate_dataset(out_dir, count, image_size):
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for index in range(count):
        names.append(write_sample(out_dir, index, image_size))
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成确定性图文样本集")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args(argv)
    names = generate_dataset(args.out, args.count, (args.width, args.height))
    print(json.dumps({"generated": names}, ensure_ascii=False))


if __name__ == "__main__":
    main()
