"""Deterministic local sample generation from fixed integer seeds.

Re-running the generator reproduces byte-identical prompts, PNGs and motion
tables. No network, GPU or external dataset is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .config import RenderConfig
from .rng import DeterministicRNG

SEED_VOCAB = 1101
SEED_PROMPT = 2202
SEED_IMAGE = 3303
SEED_MOTION = 4404

VOCABULARY = [
    "cube", "sphere", "cylinder", "cone", "torus", "prism",
    "red", "green", "blue", "amber", "violet", "teal",
]

# Geometry/texture primitives selected by integer seed.
SHAPES = ("square", "circle", "triangle")
TEXTURES = ("solid", "grid", "dots")


def generate_vocabulary() -> list[str]:
    rng = DeterministicRNG(SEED_VOCAB)
    ordered = list(VOCABULARY)
    # A fixed seeded shuffle keeps generation deterministic yet seed-derived.
    for index in range(len(ordered) - 1, 0, -1):
        target = rng.next_u64() % (index + 1)
        ordered[index], ordered[target] = ordered[target], ordered[index]
    return ordered


def generate_prompts(vocabulary: list[str], frame_count: int) -> list[list[str]]:
    rng = DeterministicRNG(SEED_PROMPT)
    prompts: list[list[str]] = []
    for _ in range(frame_count):
        length = 2 + int(rng.next_u64() % 3)
        terms = []
        for _ in range(length):
            terms.append(vocabulary[int(rng.next_u64() % len(vocabulary))])
        prompts.append(terms)
    return prompts


def _draw_png(seed: int, width: int, height: int) -> bytes:
    rng = DeterministicRNG(seed)
    pixels = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
    base = [int(40 + rng.next_u64() % 180) for _ in range(3)]
    shape = SHAPES[rng.next_u64() % len(SHAPES)]
    texture = TEXTURES[rng.next_u64() % len(TEXTURES)]
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 4
    for y in range(height):
        for x in range(width):
            inside = False
            if shape == "square":
                inside = abs(x - cx) <= radius and abs(y - cy) <= radius
            elif shape == "circle":
                inside = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            else:
                inside = (abs(x - cx) + abs(y - cy)) <= radius
            if texture == "grid":
                inside = inside or (x % 8 == 0) or (y % 8 == 0)
            elif texture == "dots" and (x % 6 == 0 and y % 6 == 0):
                inside = True
            if inside:
                jitter = int(rng.next_u64() % 24) - 12
                pixels[y][x] = [
                    max(0, min(255, base[channel] + jitter))
                    for channel in range(3)]
            else:
                pixels[y][x] = [channel // 2 for channel in base]
    image = Image.new("RGB", (width, height))
    flat = bytes(value for row in pixels for triple in row for value in triple)
    image.frombytes(flat)
    import io as _io
    buffer = _io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_images(out_dir: Path, frame_count: int, cfg: RenderConfig) -> None:
    for frame_id in range(frame_count):
        png = _draw_png(SEED_IMAGE + frame_id * 97,
                        cfg.image_width, cfg.image_height)
        (out_dir / f"cond_{frame_id:03d}.png").write_bytes(png)


def generate_motion_table(cfg: RenderConfig) -> dict:
    rng = DeterministicRNG(SEED_MOTION)
    rows = []
    for _ in range(cfg.tile_rows):
        row = []
        for _ in range(cfg.tile_cols):
            dx = int(rng.next_signed() * 8)
            dy = int(rng.next_signed() * 8)
            row.append({"dx": dx, "dy": dy})
        rows.append(row)
    return {"seed": SEED_MOTION, "rows": rows}


def generate_edit_stream(frame_count: int) -> dict:
    return {"commands": [
        {"op": "replace_prompt", "frame_id": 0,
         "terms": ["cube", "red"]},
        {"op": "set_weight", "condition": "image", "value": 0.45},
        {"op": "adjust_motion", "tile_row": 0, "tile_col": 0,
         "dx": 3, "dy": -2},
        {"op": "set_denoise_steps", "value": 8},
        {"op": "set_tile_rows", "value": 2},
        {"op": "set_quant", "width": 16, "tier": "fine"},
        {"op": "set_exchange_version", "value": 1},
        {"op": "set_exchange_version", "value": 2},
        {"op": "delete_image", "frame_id": frame_count - 1},
    ]}


def generate_sample_bundle(out_dir: str | Path,
                           cfg: RenderConfig | None = None,
                           frame_count: int = 6) -> Path:
    cfg = cfg or RenderConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocabulary = generate_vocabulary()
    (out / "vocabulary.txt").write_text(
        "\n".join(vocabulary) + "\n", encoding="utf-8")
    prompts = generate_prompts(vocabulary, frame_count)
    (out / "prompts.txt").write_text(
        "\n".join(" ".join(terms) for terms in prompts) + "\n",
        encoding="utf-8")
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    generate_images(img_dir, frame_count, cfg)
    (out / "motion.json").write_text(
        json.dumps(generate_motion_table(cfg), sort_keys=True, indent=1),
        encoding="utf-8")
    cfg.to_yaml(out / "render.yaml")
    (out / "edits.json").write_text(
        json.dumps(generate_edit_stream(frame_count), indent=1),
        encoding="utf-8")
    manifest = {
        "seeds": {"vocab": SEED_VOCAB, "prompt": SEED_PROMPT,
                  "image": SEED_IMAGE, "motion": SEED_MOTION},
        "frame_count": frame_count,
        "files": sorted(p.name for p in out.rglob("*") if p.is_file()),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8")
    return out
