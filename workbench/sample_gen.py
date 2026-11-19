"""Deterministic sample generation and per-sample validation.

All randomness derives from a numpy SeedSequence keyed by the configured
seed and the sample index, so regenerated PNG / JSON bytes are identical
bit-for-bit.  No network access or external assets are involved.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import Config, Tier
from .errors import E_REGION_OUT_OF_BOUND, SampleError
from .utils import digest_bytes, load_json, png_bytes

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REPAIR_CHOICES = ("auto", "neighbor_mean", "directional")
HINT_CHOICES = ("threshold", "edge")


def sample_rng(seed: int, sample_index: int) -> np.random.Generator:
    seq = np.random.SeedSequence([int(seed), int(sample_index)])
    return np.random.default_rng(seq)


def sample_name(index: int) -> str:
    return f"sample_{index:04d}"


def tier_dir(config: Config, tier: Tier | None = None) -> Path:
    tier = tier or config.tier()
    return Path(config.samples_dir) / tier.name


def sample_paths(config: Config, name: str, tier: Tier | None = None) -> dict:
    base = tier_dir(config, tier) / name
    return {"png": base.with_suffix(".png"), "json": base.with_suffix(".json")}


def _draw_primitive(draw: ImageDraw.ImageDraw, rng: np.random.Generator, width: int, height: int):
    kind = rng.integers(0, 3)
    color = tuple(int(v) for v in rng.integers(0, 256, size=3))
    x0 = int(rng.integers(0, max(1, width - 2)))
    y0 = int(rng.integers(0, max(1, height - 2)))
    x1 = min(width, x0 + int(rng.integers(3, max(4, width // 2 + 3))))
    y1 = min(height, y0 + int(rng.integers(3, max(4, height // 2 + 3))))
    if kind == 0:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)
    elif kind == 1:
        draw.ellipse([x0, y0, x1 - 1, y1 - 1], fill=color)
    else:
        draw.polygon(
            [(x0, y1 - 1), ((x0 + x1) // 2, y0), (x1 - 1, y1 - 1)], fill=color
        )


def _max_tier(config: Config) -> Tier:
    return max(config.tiers, key=lambda item: item.sample_count)


def render_image(config: Config, tier: Tier, index: int) -> np.ndarray:
    rng = sample_rng(config.seed, index)
    background = tuple(int(v) for v in rng.integers(20, 236, size=3))
    img = Image.new("RGB", config.resolution, background)
    draw = ImageDraw.Draw(img)
    # Primitive density is taken from the largest tier so that the same sample
    # renders identical pixels regardless of the active scale tier.
    for _ in range(_max_tier(config).primitives):
        _draw_primitive(draw, rng, config.width, config.height)
    arr = np.asarray(img, dtype=np.int16)
    noise_level = 6
    noise = rng.integers(-noise_level, noise_level + 1, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr


def build_instructions(config: Config, tier: Tier, index: int) -> dict:
    # Instruction files always carry the largest-tier region/op counts; the
    # active tier only decides how many samples participate in a run.  This
    # keeps content fingerprints stable when the scale tier narrows or widens.
    tier = _max_tier(config)
    rng = sample_rng(config.seed, 100_000 + index)
    regions = []
    min_side = 4
    for region_index in range(tier.regions):
        w = int(rng.integers(min_side, max(min_side + 1, config.width // 2 + 1)))
        h = int(rng.integers(min_side, max(min_side + 1, config.height // 2 + 1)))
        x = int(rng.integers(0, config.width - w + 1))
        y = int(rng.integers(0, config.height - h + 1))
        regions.append(
            {
                "region_id": f"region-{region_index:02d}",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )
    ops = []
    for op_index in range(tier.instruction_length):
        strength = float(np.round(rng.uniform(0.1, 1.0), 4))
        ops.append(
            {
                "op_id": f"op-{op_index:03d}",
                "region_index": int(op_index % tier.regions),
                "target_attribute": "hue",
                "attribute_value": int(rng.integers(0, 360)),
                "style_strength": strength,
                "repair": REPAIR_CHOICES[int(rng.integers(0, len(REPAIR_CHOICES)))],
                "mask_hint": HINT_CHOICES[int(rng.integers(0, len(HINT_CHOICES)))]
                if rng.random() < 0.5
                else None,
            }
        )
    return {
        "sample": sample_name(index),
        "tier": tier.name,
        "regions": regions,
        "ops": ops,
    }


def generate_tier(config: Config, tier: Tier | None = None, skip_existing: bool = True) -> list[str]:
    """Render every sample of a tier; existing files are preserved by default."""
    tier = tier or config.tier()
    out_dir = tier_dir(config, tier)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(tier.sample_count):
        name = sample_name(index)
        paths = sample_paths(config, name, tier)
        if not (skip_existing and paths["png"].exists() and paths["json"].exists()):
            image = render_image(config, tier, index)
            paths["png"].write_bytes(png_bytes(image))
            instruction = build_instructions(config, tier, index)
            from .utils import atomic_write_json

            atomic_write_json(paths["json"], instruction)
        names.append(name)
    index_path = out_dir / "index.json"
    from .utils import atomic_write_json

    atomic_write_json(
        index_path,
        {
            "tier": tier.name,
            "seed": config.seed,
            "sample_count": tier.sample_count,
            "names": names,
        },
    )
    return names


def validate_sample(config: Config, name: str, tier: Tier | None = None) -> dict:
    """Load and validate one sample; raise SampleError with a business code."""
    tier = tier or config.tier()
    paths = sample_paths(config, name, tier)
    if not paths["png"].exists() or not paths["json"].exists():
        raise SampleError("sample files missing", {"sample": name})

    png_data = paths["png"].read_bytes()
    if len(png_data) < 8 or png_data[:8] != PNG_SIGNATURE:
        raise SampleError("invalid PNG container header", {"sample": name})
    try:
        with Image.open(paths["png"]) as handle:
            handle.verify()
        with Image.open(paths["png"]) as handle:
            if handle.mode != "RGB":
                raise SampleError("image mode must be RGB", {"sample": name, "mode": handle.mode})
            if handle.size != config.resolution:
                raise SampleError(
                    "image resolution mismatch",
                    {"sample": name, "expected": list(config.resolution), "got": list(handle.size)},
                )
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except SampleError:
        raise
    except Exception as exc:  # corrupted image payload
        raise SampleError(f"unreadable PNG payload: {exc}", {"sample": name}) from exc

    try:
        instruction = load_json(paths["json"])
    except Exception as exc:
        raise SampleError(f"invalid instruction JSON: {exc}", {"sample": name}) from exc
    if not isinstance(instruction, dict):
        raise SampleError("instruction root must be an object", {"sample": name})

    regions = instruction.get("regions")
    ops = instruction.get("ops")
    if not isinstance(regions, list) or not regions:
        raise SampleError("'regions' must be a non-empty list", {"sample": name})
    if not isinstance(ops, list) or not ops:
        raise SampleError("'ops' must be a non-empty list", {"sample": name})
    max_tier = _max_tier(config)
    if len(regions) < tier.regions or len(ops) < tier.instruction_length:
        raise SampleError(
            "instruction length is shorter than the active tier profile",
            {
                "sample": name,
                "min_regions": tier.regions,
                "got_regions": len(regions),
                "min_ops": tier.instruction_length,
                "got_ops": len(ops),
            },
        )
    if len(regions) != max_tier.regions or len(ops) != max_tier.instruction_length:
        raise SampleError(
            "instruction length does not match the canonical max-tier profile",
            {
                "sample": name,
                "expected_regions": max_tier.regions,
                "got_regions": len(regions),
                "expected_ops": max_tier.instruction_length,
                "got_ops": len(ops),
            },
        )

    for idx, region in enumerate(regions):
        if not isinstance(region, dict) or any(
            not isinstance(region.get(key), int) or isinstance(region.get(key), bool)
            for key in ("x", "y", "w", "h")
        ):
            raise SampleError("malformed region rectangle", {"sample": name, "region_index": idx})
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            raise SampleError(
                "region rectangle must have positive size and non-negative origin",
                {"sample": name, "region_index": idx, "rect": [x, y, w, h]},
                code=E_REGION_OUT_OF_BOUND,
            )
        if x + w > config.width or y + h > config.height:
            raise SampleError(
                "region rectangle outside image bounds",
                {"sample": name, "region_index": idx, "rect": [x, y, w, h]},
                code=E_REGION_OUT_OF_BOUND,
            )

    for idx, op in enumerate(ops):
        if not isinstance(op, dict):
            raise SampleError("malformed edit op", {"sample": name, "op_index": idx})
        region_index = op.get("region_index")
        if not isinstance(region_index, int) or isinstance(region_index, bool) or not 0 <= region_index < len(regions):
            raise SampleError("op references unknown region", {"sample": name, "op_index": idx})
        if op.get("target_attribute") != "hue":
            raise SampleError("only 'hue' target attribute is supported", {"sample": name, "op_index": idx})
        value = op.get("attribute_value")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 360:
            raise SampleError("attribute_value must be an int in [0, 359]", {"sample": name, "op_index": idx})
        strength = op.get("style_strength")
        if not isinstance(strength, (int, float)) or isinstance(strength, bool) or not math.isfinite(strength):
            raise SampleError("style_strength must be a finite number", {"sample": name, "op_index": idx})
        if op.get("repair") not in REPAIR_CHOICES:
            raise SampleError("unknown repair method", {"sample": name, "op_index": idx, "repair": op.get("repair")})
        hint = op.get("mask_hint")
        if hint is not None and hint not in HINT_CHOICES:
            raise SampleError("unknown mask hint", {"sample": name, "op_index": idx})

    return {
        "name": name,
        "tier": tier.name,
        "image": image,
        "instruction": instruction,
        "image_hash": digest_bytes(png_data),
        "png_bytes": png_data,
    }
