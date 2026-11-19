from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .canonical import atomic_write_json
from .font import font_templates

IMAGE_WIDTH = 360
IMAGE_HEIGHT = 220
SCALE = 3
CHAR_GAP = 6


def _render_line(
    canvas: np.ndarray,
    text: str,
    x_start: int,
    y_start: int,
    ink_value: int = 24,
) -> dict[str, Any]:
    templates = font_templates()
    x = x_start
    chars: list[dict[str, Any]] = []
    for char in text:
        template = templates[char]
        rendered = np.repeat(np.repeat(template, SCALE, axis=0), SCALE, axis=1)
        h, w = rendered.shape
        region = canvas[y_start : y_start + h, x : x + w]
        if region.shape != rendered.shape:
            raise ValueError(f"line {text!r} does not fit image")
        region[rendered.astype(bool)] = ink_value
        ys, xs = np.nonzero(rendered)
        chars.append(
            {
                "bbox": [x + int(xs.min()), y_start + int(ys.min()), x + int(xs.max()) + 1, y_start + int(ys.max()) + 1],
                "text": char,
            }
        )
        x += w + CHAR_GAP
    if not chars:
        raise ValueError("a line must contain at least one character")
    return {
        "bbox": [
            min(c["bbox"][0] for c in chars),
            min(c["bbox"][1] for c in chars),
            max(c["bbox"][2] for c in chars),
            max(c["bbox"][3] for c in chars),
        ],
        "chars": chars,
    }


def _line_rect_contains(rect: list[int], x: int, y: int) -> bool:
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def make_layout_image(
    path: Path,
    truth_path: Path,
    lines: list[str],
    seed: int,
    *,
    noise_count: int = 36,
    bridge_mode: str = "none",
    challenge_noise: bool = False,
) -> None:
    canvas = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 242, dtype=np.uint8)
    truth_lines: list[dict[str, Any]] = []
    for line_index, text in enumerate(lines):
        y = 42 + line_index * 82
        truth_lines.append(_render_line(canvas, text, 35, y))

    rng = random.Random(seed)
    settled: list[list[int]] = []
    attempts = 0
    while len(settled) < noise_count and attempts < noise_count * 100:
        attempts += 1
        x = rng.randrange(5, IMAGE_WIDTH - 5)
        y = rng.randrange(5, IMAGE_HEIGHT - 5)
        rect = [x, y, x + 1, y + 1]
        if any(
            _line_rect_contains(
                [line["bbox"][0] - 5, line["bbox"][1] - 5, line["bbox"][2] + 5, line["bbox"][3] + 5],
                x,
                y,
            )
            for line in truth_lines
        ):
            continue
        value = rng.choice((78, 96, 132, 188, 205, 218))
        canvas[y, x] = value
        if rng.random() < 0.35:
            nx = min(IMAGE_WIDTH - 1, x + rng.choice((-1, 0, 1)))
            ny = min(IMAGE_HEIGHT - 1, y + rng.choice((0, 1)))
            canvas[ny, nx] = min(235, value + 18)
        settled.append(rect)

    if bridge_mode == "speck" and len(truth_lines) >= 1:
        first = truth_lines[0]
        if len(first["chars"]) >= 2:
            left = first["chars"][0]
            right = first["chars"][1]
            gap = right["bbox"][0] - left["bbox"][2]
            if gap >= 4:
                x = left["bbox"][2] + gap // 2
                y = (left["bbox"][1] + left["bbox"][3]) // 2
                canvas[y, x] = 70
    elif bridge_mode == "line_bridge" and len(truth_lines) >= 1:
        first = truth_lines[0]
        if len(first["chars"]) >= 2:
            left = first["chars"][0]
            right = first["chars"][1]
            y = (left["bbox"][1] + left["bbox"][3]) // 2
            for x in range(left["bbox"][2], right["bbox"][0]):
                canvas[y, x] = 70
    if challenge_noise and len(truth_lines) >= 1 and len(truth_lines[0]["chars"]) >= 2:
        target = truth_lines[0]["chars"][1]["bbox"]
        x = target[0] + min(9, max(0, target[2] - target[0] - 1))
        y = target[1] + min(3, max(0, target[3] - target[1] - 1))
        canvas[y, x] = 24

    image = Image.fromarray(canvas, mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    atomic_write_json(
        truth_path,
        {
            "schema": "layout-truth-v1",
            "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "lines": truth_lines,
        },
    )


def generate_samples(data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ("sample_a", ["HELLO", '"ABC"'], 1101, "none", True),
        ("sample_b", ["WORLD", "WQ12"], 2202, "line_bridge", False),
        ("sample_c", ["MINT", "X9"], 3303, "speck", False),
    ]
    written: list[Path] = []
    for name, lines, seed, bridge, challenge in samples:
        png = data_dir / f"{name}.png"
        truth = data_dir / f"{name}.truth.json"
        make_layout_image(
            png, truth, lines, seed, bridge_mode=bridge, challenge_noise=challenge
        )
        written.extend([png, truth])

    replacement_dir = data_dir / "replacements"
    replacement_dir.mkdir(parents=True, exist_ok=True)
    png = replacement_dir / "sample_b_v2.png"
    truth = replacement_dir / "sample_b_v2.truth.json"
    make_layout_image(png, truth, ["NEW1", "W8"], 9902)
    written.extend([png, truth])
    return written
