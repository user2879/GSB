#!/usr/bin/env python3
"""Generate deterministic synthetic layout pages and corresponding truth JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


WIDTH = 720
HEIGHT = 960
BACKGROUND = 245
BORDER = 42
GLYPH = 68
IMAGE_INK = 104
PAGE_NAMES = [f"page_{index:03d}.png" for index in range(1, 7)]


def bbox(x0: int, y0: int, x1: int, y1: int) -> list[int]:
    if not (0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT):
        raise ValueError(f"invalid bbox {(x0, y0, x1, y1)}")
    return [x0, y0, x1, y1]


def block(
    role: str,
    rectangle: list[int],
    column_group: int,
    is_spanning: bool,
    row_group: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "bbox": rectangle,
        "column_group": column_group,
        "is_spanning": is_spanning,
        "row_group": row_group,
    }


def _seed_value(seed: int, page_name: str, block_index: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{page_name}:{block_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _draw_text_pattern(
    draw: ImageDraw.ImageDraw,
    rectangle: list[int],
    role: str,
    seed: int,
) -> None:
    x0, y0, x1, y1 = rectangle
    inner_x0 = x0 + 5
    inner_x1 = x1 - 5
    if inner_x1 <= inner_x0:
        return
    if role == "title":
        line_specs = [(y0 + 7, 7), (y0 + 20, 7)]
    elif role == "body":
        line_specs = [(y0 + 5, 6)]
    else:
        line_specs = [(y0 + 6, 5)]

    for line_index, (line_y, line_height) in enumerate(line_specs):
        if line_y + line_height > y1 - 3:
            continue
        cursor = inner_x0 + (seed + line_index * 13) % 8
        word_index = 0
        while cursor < inner_x1:
            word_length = 18 + ((seed >> (word_index % 17)) + word_index * 11) % 34
            end = min(inner_x1, cursor + word_length)
            if end - cursor >= 5:
                draw.rectangle(
                    [cursor, line_y, end - 1, line_y + line_height - 1],
                    fill=GLYPH,
                )
            gap = 4 + ((seed >> ((word_index + 3) % 19)) + word_index) % 6
            cursor = end + gap
            word_index += 1


def _draw_image_pattern(
    draw: ImageDraw.ImageDraw,
    rectangle: list[int],
    seed: int,
) -> None:
    x0, y0, x1, y1 = rectangle
    inner = [x0 + 3, y0 + 3, x1 - 4, y1 - 4]
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return
    draw.rectangle(inner, fill=IMAGE_INK)
    width = inner[2] - inner[0]
    height = inner[3] - inner[1]
    for index in range(4):
        spot_width = max(7, width // (7 + index))
        spot_height = max(5, height // (8 + index))
        spot_x = inner[0] + 4 + ((seed >> (index * 7 + 3)) % max(1, width - spot_width - 8))
        spot_y = inner[1] + 4 + ((seed >> (index * 9 + 5)) % max(1, height - spot_height - 8))
        draw.rectangle(
            [spot_x, spot_y, spot_x + spot_width, spot_y + spot_height],
            fill=BACKGROUND,
        )
    for stripe in range(3):
        stripe_y = inner[1] + (stripe + 1) * height // 4
        draw.line(
            [inner[0] + 3, stripe_y, inner[2] - 4, stripe_y],
            fill=180,
            width=1,
        )


def draw_page(page_name: str, specs: list[dict[str, Any]], seed: int) -> Image.Image:
    image = Image.new("L", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index, spec in enumerate(specs):
        x0, y0, x1, y1 = spec["bbox"]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BORDER, width=1)
        seed_value = _seed_value(seed, page_name, index)
        if spec["role"] == "image":
            _draw_image_pattern(draw, spec["bbox"], seed_value)
        else:
            _draw_text_pattern(draw, spec["bbox"], spec["role"], seed_value)
    return image


def _spanning_order(specs: list[dict[str, Any]], rule: str) -> list[dict[str, Any]]:
    spanning = sorted(
        [spec for spec in specs if spec["is_spanning"]],
        key=lambda spec: (spec["bbox"][1], spec["bbox"][3], spec["bbox"][0]),
    )
    remaining = [spec for spec in specs if not spec["is_spanning"]]

    if rule == "column_first":
        units = [[spec] for spec in sorted(
            remaining,
            key=lambda spec: (spec["column_group"], spec["bbox"][1], spec["bbox"][0]),
        )]
    else:
        units = []
        ordered = sorted(
            remaining, key=lambda spec: (spec["bbox"][1], spec["bbox"][0])
        )
        for spec in ordered:
            if (
                not units
                or spec["bbox"][1] - max(item["bbox"][3] for item in units[-1]) > 8
            ):
                units.append([spec])
            else:
                units[-1].append(spec)
        for unit in units:
            unit.sort(key=lambda spec: (spec["bbox"][0], spec["bbox"][1]))

    ordered_specs: list[dict[str, Any]] = []
    for span in spanning:
        above = [
            unit
            for unit in units
            if max((spec["bbox"][1] + spec["bbox"][3]) / 2 for spec in unit)
            < span["bbox"][1]
        ]
        if not above:
            above = [unit for unit in units if unit[0]["bbox"][1] < span["bbox"][1]]
        units = [unit for unit in units if unit not in above]
        for unit in above:
            ordered_specs.extend(unit)
        ordered_specs.append(span)
    for unit in units:
        ordered_specs.extend(unit)
    return ordered_specs


def _page_one() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        block("header_footer", bbox(40, 20, 680, 38), 0, False, "r_header"),
        block("title", bbox(160, 68, 560, 102), 0, False, "r_title"),
    ]
    for index in range(8):
        y = 126 + index * 30
        specs.append(block("body", bbox(80, y, 640, y + 16), 0, False, f"r_top_{index:02d}"))
    specs.append(block("image", bbox(120, 370, 600, 480), 0, False, "r_image"))
    for index in range(5):
        y = 510 + index * 30
        specs.append(block("body", bbox(80, y, 640, y + 16), 0, False, f"r_bottom_{index:02d}"))
    specs.append(block("header_footer", bbox(40, 922, 680, 940), 0, False, "r_footer"))
    return specs


def _page_two() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        block("header_footer", bbox(40, 20, 680, 38), -1, True, "r_header"),
        block("title", bbox(170, 64, 550, 98), -1, True, "r_title"),
    ]
    for index in range(9):
        y = 122 + index * 30
        specs.append(block("body", bbox(50, y, 335, y + 16), 0, False, f"r_top_{index:02d}"))
        specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_top_{index:02d}"))
    specs.append(block("image", bbox(140, 420, 580, 530), -1, True, "r_image"))
    for index in range(3):
        y = 560 + index * 30
        specs.append(block("body", bbox(50, y, 335, y + 16), 0, False, f"r_bottom_{index:02d}"))
        specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_bottom_{index:02d}"))
    specs.append(block("header_footer", bbox(40, 922, 680, 940), -1, True, "r_footer"))
    return specs


def _page_three() -> list[dict[str, Any]]:
    columns = [(40, 245), (270, 475), (500, 705)]
    specs: list[dict[str, Any]] = [
        block("header_footer", bbox(40, 20, 705, 38), -1, True, "r_header"),
        block("title", bbox(100, 58, 620, 92), -1, True, "r_title"),
    ]
    for index in range(9):
        y = 118 + index * 30
        for column_index, (x0, x1) in enumerate(columns):
            if column_index == 1 and index in {4, 5}:
                continue
            specs.append(
                block(
                    "body",
                    bbox(x0, y, x1, y + 16),
                    column_index,
                    False,
                    f"r_top_{index:02d}",
                )
            )
    specs.append(block("image", bbox(275, 238, 470, 290), 1, False, "r_image"))
    for index in range(4):
        y = 520 + index * 30
        for column_index, (x0, x1) in enumerate(columns):
            specs.append(
                block(
                    "body",
                    bbox(x0, y, x1, y + 16),
                    column_index,
                    False,
                    f"r_bottom_{index:02d}",
                )
            )
    specs.append(block("header_footer", bbox(40, 922, 705, 940), -1, True, "r_footer"))
    return specs


def _page_four() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        block("header_footer", bbox(40, 20, 680, 38), -1, True, "r_header"),
        block("title", bbox(150, 58, 570, 92), -1, True, "r_title"),
        block("image", bbox(120, 112, 600, 222), -1, True, "r_image_top"),
    ]
    for index in range(8):
        y = 252 + index * 30
        specs.append(block("body", bbox(50, y, 335, y + 16), 0, False, f"r_top_{index:02d}"))
        specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_top_{index:02d}"))
    specs.append(block("image", bbox(135, 500, 585, 605), -1, True, "r_image_mid"))
    for index in range(3):
        y = 632 + index * 30
        specs.append(block("body", bbox(50, y, 335, y + 16), 0, False, f"r_bottom_{index:02d}"))
        specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_bottom_{index:02d}"))
    specs.append(block("header_footer", bbox(40, 922, 680, 940), -1, True, "r_footer"))
    return specs


def _page_five() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        block("header_footer", bbox(40, 20, 680, 38), 0, False, "r_header"),
        block("title", bbox(110, 64, 610, 98), 0, False, "r_title"),
    ]
    for index in range(5):
        y = 124 + index * 30
        specs.append(block("body", bbox(80, y, 640, y + 16), 0, False, f"r_top_{index:02d}"))
    specs.append(block("image", bbox(190, 290, 530, 382), 0, False, "r_image_top"))
    for index in range(4):
        y = 412 + index * 30
        specs.append(block("body", bbox(80, y, 640, y + 16), 0, False, f"r_mid_{index:02d}"))
    specs.append(block("image", bbox(100, 552, 620, 662), 0, False, "r_image_mid"))
    for index in range(3):
        y = 692 + index * 30
        specs.append(block("body", bbox(80, y, 640, y + 16), 0, False, f"r_bottom_{index:02d}"))
    specs.append(block("header_footer", bbox(40, 922, 680, 940), 0, False, "r_footer"))
    return specs


def _page_six() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        block("header_footer", bbox(40, 20, 680, 38), -1, True, "r_header"),
        block("title", bbox(180, 62, 540, 96), -1, True, "r_title"),
    ]
    for index in range(10):
        y = 120 + index * 30
        specs.append(block("body", bbox(50, y, 335, y + 16), 0, False, f"r_top_{index:02d}"))
        if index < 5:
            specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_top_{index:02d}"))
    specs.append(block("image", bbox(390, 278, 665, 352), 1, False, "r_image_right"))
    for index in range(5):
        y = 382 + index * 30
        specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_right_{index:02d}"))
    for index in range(3):
        y = 580 + index * 30
        specs.append(block("body", bbox(50, y, 335, y + 16), 0, False, f"r_bottom_{index:02d}"))
        specs.append(block("body", bbox(385, y, 670, y + 16), 1, False, f"r_bottom_{index:02d}"))
    specs.append(block("header_footer", bbox(40, 922, 680, 940), -1, True, "r_footer"))
    return specs


LAYOUTS = {
    "page_001.png": _page_one,
    "page_002.png": _page_two,
    "page_003.png": _page_three,
    "page_004.png": _page_four,
    "page_005.png": _page_five,
    "page_006.png": _page_six,
}


def build_truth(seed: int) -> dict[str, Any]:
    pages: dict[str, Any] = {}
    for page_name in PAGE_NAMES:
        specs = LAYOUTS[page_name]()
        column_orders = _spanning_order(specs, "column_first")
        region_orders = _spanning_order(specs, "region_first")
        column_position = {id(spec): index for index, spec in enumerate(column_orders, 1)}
        region_position = {id(spec): index for index, spec in enumerate(region_orders, 1)}
        truth_blocks: list[dict[str, Any]] = []
        for index, spec in enumerate(specs, start=1):
            truth_blocks.append(
                {
                    "block_id": f"{page_name[:-4]}_truth_{index:03d}",
                    "role": spec["role"],
                    "bbox": list(spec["bbox"]),
                    "column_group": int(spec["column_group"]),
                    "is_spanning": bool(spec["is_spanning"]),
                    "row_group": spec["row_group"],
                    "reading_order": column_position[id(spec)],
                    "reading_order_region_first": region_position[id(spec)],
                }
            )
        pages[page_name] = {
            "width": WIDTH,
            "height": HEIGHT,
            "blocks": truth_blocks,
            "truth_block_count": len(truth_blocks),
        }
        # Attach stable ids to ordering arrays without changing truth block ids.
        order_ids = {
            id(spec): truth_blocks[index]["block_id"]
            for index, spec in enumerate(specs)
        }
        pages[page_name]["reading_order"] = [order_ids[id(spec)] for spec in column_orders]
        pages[page_name]["reading_order_region_first"] = [
            order_ids[id(spec)] for spec in region_orders
        ]
    return {
        "schema_version": 1,
        "generator": "scripts/generate_samples.py",
        "seed": seed,
        "width": WIDTH,
        "height": HEIGHT,
        "background": BACKGROUND,
        "binarization_note": "ink pixels are below 180; background is 245",
        "default_reading_order_rule": "column_first",
        "pages": pages,
    }


def generate(out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in out_dir.glob("*.png"):
        old_png.unlink()
    for page_name in PAGE_NAMES:
        specs = LAYOUTS[page_name]()
        image = draw_page(page_name, specs, seed)
        if page_name == "page_006.png":
            image = image.convert("RGB")
        image.save(out_dir / page_name, format="PNG", optimize=False, compress_level=9)
    truth = build_truth(seed)
    (out_dir / "layout_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="samples", help="sample output directory")
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    generate(Path(args.out), args.seed)
    print(f"generated {len(PAGE_NAMES)} PNG pages and truth JSON in {args.out}")


if __name__ == "__main__":
    main()
