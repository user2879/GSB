"""Deterministic local sample synthesis.

All samples are generated from fixed seeds / explicit plans; no network.
Also used by the edit runner to regenerate replacement images.
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

BACKGROUND = (245, 245, 240)


def draw_parts(image_size, parts, background=BACKGROUND):
    img = Image.new("RGB", tuple(image_size), background)
    draw = ImageDraw.Draw(img)
    for part in parts:
        x0, y0, x1, y1 = part["bbox"]
        rgb = tuple(part["rgb"])
        if part["category"] == "square":
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=rgb)
        elif part["category"] == "circle":
            draw.ellipse([x0, y0, x1 - 1, y1 - 1], fill=rgb)
        else:
            draw.polygon(
                [((x0 + x1) // 2, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)], fill=rgb
            )
    return img


def generate_image(seed, config):
    """Deterministic random image from a seed (used for replace_image edits)."""
    rng = random.Random(seed)
    width, height = config.image_size
    parts = []
    for _ in range(rng.randint(3, 5)):
        category = rng.choice(config.categories)
        color = rng.choice(list(config.colors))
        size = rng.randint(36, 64)
        x = rng.randint(4, width - size - 4)
        y = rng.randint(4, height - size - 4)
        parts.append(
            {
                "category": category,
                "color": color,
                "rgb": list(config.colors[color]),
                "bbox": [x, y, x + size, y + size],
            }
        )
    return draw_parts(config.image_size, parts), parts


def _part(category, color, bbox, config):
    return {
        "category": category,
        "color": color,
        "rgb": list(config.colors[color]),
        "bbox": list(bbox),
    }


def build_dataset(config, data_dir):
    """Create images/, descriptions/, labels_{a,b}.jsonl, edits.json,
    injections.json under data_dir. Fully deterministic."""
    data_dir = Path(data_dir)
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "descriptions").mkdir(parents=True, exist_ok=True)

    plans = {
        "s01": {
            "parts": [
                _part("square", "red", (20, 20, 70, 70), config),
                _part("square", "red", (150, 30, 200, 80), config),
                _part("circle", "blue", (40, 150, 100, 210), config),
            ],
            "assertions": [
                {"category": "square", "color": "red", "position": "top", "count": 2},
                {"category": "circle", "color": "blue", "position": "bottom-left", "count": 1},
            ],
            "label_b_extra": [],
            "label_a_extra": [],
        },
        "s02": {
            "parts": [
                _part("square", "red", (30, 30, 110, 110), config),
                _part("square", "blue", (60, 30, 80, 110), config),
            ],
            "assertions": [
                {"category": "square", "color": "red", "position": "left", "count": 1},
                {"category": "square", "color": "blue", "position": "center", "count": 1},
            ],
            "label_b_extra": [],
            "label_a_extra": [],
        },
        "s03": {
            "parts": [
                _part("triangle", "green", (30, 30, 90, 90), config),
                _part("triangle", "green", (150, 150, 210, 210), config),
            ],
            "assertions": [
                {"category": "triangle", "color": "green", "position": "diag", "count": 2},
            ],
            "label_b_extra": [
                {"category": "triangle", "color": "green", "bbox": [150, 30, 180, 60]}
            ],
            "label_a_extra": [],
        },
        "s04": {
            "parts": [
                _part("circle", "blue", (60, 60, 120, 120), config),
            ],
            "assertions": [
                {"category": "circle", "color": "blue", "position": "center", "count": 3},
            ],
            "label_b_extra": [
                {"category": "circle", "color": "blue", "bbox": [150, 150, 200, 200]}
            ],
            "label_a_extra": [
                {"category": "circle", "color": "blue", "bbox": [150, 150, 200, 200]}
            ],
        },
        "s05": {
            "parts": [
                _part("square", "red", (30, 30, 80, 80), config),
            ],
            "assertions": [
                {"category": "square", "color": "red", "position": "top-left", "count": 1},
                {"category": "circle", "color": "green", "position": "none", "count": 0},
            ],
            "label_b_extra": [],
            "label_a_extra": [],
        },
        "s06": {
            "parts": [
                _part("square", "red", (20, 20, 60, 60), config),
                _part("square", "red", (120, 120, 160, 160), config),
            ],
            "assertions": [
                {"category": "square", "color": "red", "position": "top-left", "count": 1},
                {"category": "square", "color": "red", "position": "center", "count": 1},
            ],
            "label_b_extra": [],
            "label_a_extra": [],
        },
        "s07": {
            "parts": [
                _part("circle", "green", (40, 40, 100, 100), config),
                _part("square", "blue", (150, 150, 200, 200), config),
            ],
            "assertions": [
                {"category": "circle", "color": "green", "position": "top-left", "count": 1},
                {"category": "square", "color": "blue", "position": "bottom-right", "count": 1},
            ],
            "label_b_extra": [],
            "label_a_extra": [],
        },
        "s08": {
            "parts": [
                _part("circle", "red", (30, 30, 90, 90), config),
                _part("circle", "red", (150, 150, 210, 210), config),
            ],
            "assertions": [
                {"category": "circle", "color": "red", "position": "diag", "count": 3},
            ],
            "label_b_extra": [],
            "label_a_extra": [],
        },
    }

    label_a_lines = []
    label_b_lines = []
    for sample_id in sorted(plans):
        plan = plans[sample_id]
        img = draw_parts(config.image_size, plan["parts"])
        img.save(data_dir / "images" / f"{sample_id}.png")
        desc = {"sample_id": sample_id, "assertions": plan["assertions"]}
        (data_dir / "descriptions" / f"{sample_id}.json").write_text(
            json.dumps(desc, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        for source, extra_key, sink in (
            ("label_a", "label_a_extra", label_a_lines),
            ("label_b", "label_b_extra", label_b_lines),
        ):
            order = 1
            for part in plan["parts"]:
                sink.append(
                    {
                        "sample_id": sample_id,
                        "category": part["category"],
                        "color": part["color"],
                        "bbox": list(part["bbox"]),
                        "order": order,
                    }
                )
                order += 1
            for extra in plan[extra_key]:
                sink.append(
                    {
                        "sample_id": sample_id,
                        "category": extra["category"],
                        "color": extra["color"],
                        "bbox": list(extra["bbox"]),
                        "order": order,
                    }
                )
                order += 1

    (data_dir / "labels_a.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in label_a_lines),
        encoding="utf-8",
    )
    (data_dir / "labels_b.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in label_b_lines),
        encoding="utf-8",
    )

    replaced_label_a = list(label_a_lines) + [
        {
            "sample_id": "s05",
            "category": "square",
            "color": "red",
            "bbox": [100, 100, 140, 140],
            "order": 2,
        }
    ]
    edits = [
        {"op": "desc_modify", "sample_id": "s01", "index": 0, "patch": {"count": 3}},
        {
            "op": "desc_add",
            "sample_id": "s05",
            "assertion": {"category": "circle", "color": "green", "position": "center", "count": 1},
        },
        {"op": "desc_remove", "sample_id": "s06", "index": 1},
        {
            "op": "label_modify",
            "source": "label_b",
            "sample_id": "s03",
            "order": 3,
            "patch": {"color": "blue"},
        },
        {"op": "replace_labels", "source": "label_a", "entries": replaced_label_a},
        {"op": "replace_image", "sample_id": "s07", "seed": 20241101},
        {"op": "set_tolerance", "value": 1},
        {
            "op": "set_priority",
            "value": ["label_majority", "detection", "description", "label_a"],
        },
    ]
    (data_dir / "edits.json").write_text(
        json.dumps(edits, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    injections = {
        "injections": [
            {
                "id": "INJ-001",
                "sample_id": "s07",
                "source": "image",
                "fault": "truncate_png",
                "params": {"keep_bytes": 100},
            },
            {
                "id": "INJ-002",
                "sample_id": "s07",
                "source": "description",
                "fault": "desc_category_enum",
                "params": {"index": 0, "value": "hexagon"},
            },
            {
                "id": "INJ-003",
                "sample_id": "s01",
                "source": "label_a",
                "fault": "label_order_gap",
                "params": {"entry_index": 1, "skip": 2},
            },
            {
                "id": "INJ-004",
                "sample_id": "s02",
                "source": "label_b",
                "fault": "box_out_of_range",
                "params": {"entry_index": 0, "dx": 500},
            },
        ]
    }
    (data_dir / "injections.json").write_text(
        json.dumps(injections, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return sorted(plans)
