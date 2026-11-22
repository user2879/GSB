from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .codec_v1 import encode_v1
from .codec_v2 import encode_v2
from .convert import downgrade_v2
from .models import RegionEdit, Rect


SAMPLE_SEED = 734921
IMAGE_SIZE = (256, 256)


def _base_image(width: int, height: int, seed: int, variant: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.int64)
    r = (40 + xx * (95 + variant * 7) // width + variant * 11) % 256
    g = (55 + yy * (80 + variant * 5) // height + variant * 17) % 256
    b = (75 + ((xx + yy) * (60 + variant * 3)) // (width + height) + variant * 23) % 256
    image = np.stack([r, g, b], axis=-1).astype(np.int64)
    noise = rng.integers(-14, 15, size=(height, width, 1), dtype=np.int64)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle((18 + variant * 5, 24, 95, 92), fill=(205, 78, 58))
    draw.rectangle((122, 30 + variant * 4, 224, 102), fill=(52, 111, 196))
    draw.ellipse((42, 118 + variant * 3, 124, 198), fill=(220, 184, 64))
    draw.polygon([(148, 118), (224, 190), (132, 214)], fill=(65, 172, 122))
    draw.line((8, 236, 247, 205 + variant * 6), fill=(245, 240, 225), width=3)
    draw.rectangle((188, 24, 232, 55), fill=(120, 61, 155))
    return np.asarray(pil, dtype=np.uint8).copy()


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, mode="RGB")
    image.save(path, format="PNG", compress_level=9, optimize=False)


def _sample_regions() -> list[RegionEdit]:
    records = [
        ("scene_alpha.png", 1, (18.6, 22.4, 68.0, 48.8), 0.4237, "warm", "mean", 6.5, 3.25),
        ("scene_alpha.png", 2, (112.0, 28.0, 78.0, 54.0), 0.2500, "cool", "edge_extend", 0.0, 0.0),
        ("scene_beta.png", 1, (30.5, 55.2, 62.0, 72.0), 0.8124, "vivid", "noise_guard", -12.5, 5.0),
        ("scene_beta.png", 2, (150.25, 142.75, 60.5, 50.25), 0.1270, "mono", "mean", 0.0, 1.5),
        ("scene_gamma.png", 1, (58.0, 64.0, 92.0, 84.0), 0.6666, "cool", "edge_extend", 21.75, 4.25),
        ("scene_gamma.png", 2, (176.0, 18.0, 52.0, 52.0), 0.5000, "warm", "mean", 0.0, 0.0),
    ]
    regions: list[RegionEdit] = []
    for image, region_index, rect, style, attribute, repair, rotation, feather in records:
        regions.append(
            RegionEdit(
                exchange_version=2,
                image_name=image,
                region_index=region_index,
                rect=Rect.from_values(*rect),
                style_strength=Decimal(str(style)),
                target_attribute=attribute,
                repair_method=repair,
                precision_label="v2:decimal-source",
                rotation_deg=Decimal(str(rotation)),
                feather_radius=Decimal(str(feather)),
                source_format="sample_v2",
            )
        )
    return regions


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "image_size": list(IMAGE_SIZE),
        "coordinate_tolerance": 0.25,
        "style_quantization_levels": 16,
        "fixed_fraction_bits": 12,
        "rounding": "half_up",
        "repair_neighborhood_radius": 2,
        "feather_radius_max": 8.0,
        "recompute_region_budget": 120000,
        "thread_pool_workers": 4,
        "snapshot_commit_interval_commands": 2,
        "pixel_difference_tolerance": 0,
    }


def _manifest_payload(regions: list[RegionEdit]) -> dict[str, Any]:
    return {
        "schema": "image-edit-manifest/v1",
        "edits": [
            {
                "image": region.image_name,
                "region_index": region.region_index,
                "rect": {key: float(Decimal(value)) for key, value in region.rect.to_dict().items()},
                "style_strength": float(region.style_strength),
                "target_attribute": region.target_attribute,
                "repair_method": region.repair_method,
                "rotation_deg": float(region.rotation_deg),
                "feather_radius": float(region.feather_radius),
            }
            for region in regions
        ],
    }


def _write_injections(root: Path, valid_v1: str, images: dict[str, Any], regions: list[RegionEdit]) -> None:
    injection_dir = root / "injections"
    injection_dir.mkdir(parents=True, exist_ok=True)

    truncated_line = valid_v1.splitlines()[0][:31]
    (injection_dir / "truncated_v1.txt").write_text(truncated_line + "\n", encoding="utf-8")
    (injection_dir / "mixed_v1.txt").write_text(valid_v1 + truncated_line + "\n", encoding="utf-8")

    v2_payload = json.loads(encode_v2(regions))
    invalid_structure = json.loads(json.dumps(v2_payload))
    del invalid_structure["regions"][0]["feather_radius"]
    (injection_dir / "invalid_structure_v2.json").write_text(
        json.dumps(invalid_structure, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    out_of_range = json.loads(json.dumps(v2_payload))
    out_of_range["regions"][0]["rect"]["x"] = 250.0
    out_of_range["regions"][0]["rect"]["w"] = 30.0
    (injection_dir / "out_of_range_v2.json").write_text(
        json.dumps(out_of_range, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    v1_lines = valid_v1.splitlines()
    invalid_v1 = list(v1_lines)
    first = list(invalid_v1[0])
    first[18:21] = "999"
    invalid_v1[0] = "".join(first)
    (injection_dir / "unknown_region_v1.txt").write_text("\n".join(invalid_v1) + "\n", encoding="utf-8")

    cases = {
        "schema": "image-edit-injection/v1",
        "cases": [
            {
                "id": "INJ-V1-TRUNCATED",
                "version": 1,
                "path": "injections/truncated_v1.txt",
                "expected_code": "E_FIELD_TRUNCATED",
            },
            {
                "id": "INJ-V2-STRUCTURE",
                "version": 2,
                "path": "injections/invalid_structure_v2.json",
                "expected_code": "E_STRUCTURE_INVALID",
            },
            {
                "id": "INJ-V2-OUT-OF-RANGE",
                "version": 2,
                "path": "injections/out_of_range_v2.json",
                "expected_code": "E_REGION_OUT_OF_RANGE",
            },
            {
                "id": "INJ-V1-UNKNOWN-REGION",
                "version": 1,
                "path": "injections/unknown_region_v1.txt",
                "expected_code": "E_REGION_OUT_OF_RANGE",
            },
        ],
    }
    (injection_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_scenarios(root: Path) -> None:
    scenarios = root / "scenarios"
    scenarios.mkdir(parents=True, exist_ok=True)
    incremental = [
        {"id": "CMD-001-V2", "op": "set_exchange_version", "version": 2},
        {"id": "CMD-002-QUANT", "op": "set_config", "key": "style_quantization_levels", "value": 8},
        {
            "id": "CMD-003-STYLE",
            "op": "update_region",
            "image": "scene_alpha.png",
            "region_index": 1,
            "changes": {"style_strength": "0.6125"},
        },
        {
            "id": "CMD-004-RECT",
            "op": "update_region",
            "image": "scene_beta.png",
            "region_index": 1,
            "changes": {"rect": {"x": 34.5, "y": 59.25, "w": 60.0, "h": 70.0}},
        },
        {
            "id": "CMD-005-REPAIR",
            "op": "update_region",
            "image": "scene_beta.png",
            "region_index": 1,
            "changes": {"repair_method": "edge_extend"},
        },
        {
            "id": "CMD-006-FEATHER",
            "op": "update_region",
            "image": "scene_gamma.png",
            "region_index": 1,
            "changes": {"feather_radius": "6.5"},
        },
        {
            "id": "CMD-007-ROTATE",
            "op": "update_region",
            "image": "scene_alpha.png",
            "region_index": 2,
            "changes": {"rotation_deg": "8.0", "feather_radius": "2.0"},
        },
        {"id": "CMD-008-BUDGET", "op": "set_config", "key": "recompute_region_budget", "value": 25000},
        {"id": "CMD-009-V1", "op": "set_exchange_version", "version": 1},
        {"id": "CMD-010-V2", "op": "set_exchange_version", "version": 2},
        {"id": "CMD-011-QUANT16", "op": "set_config", "key": "style_quantization_levels", "value": 16},
    ]
    (scenarios / "incremental_sequence.jsonl").write_text(
        "\n".join(json.dumps(command, ensure_ascii=False, sort_keys=True) for command in incremental) + "\n",
        encoding="utf-8",
    )

    budget = [
        {"id": "CMD-BUDGET-CONFIG", "op": "set_config", "key": "recompute_region_budget", "value": 25000},
        {"id": "CMD-BUDGET-RUN-1", "op": "run"},
        {"id": "CMD-BUDGET-RUN-2", "op": "run"},
        {"id": "CMD-BUDGET-DRAIN", "op": "drain"},
    ]
    (scenarios / "budget_sequence.jsonl").write_text(
        "\n".join(json.dumps(command, ensure_ascii=False, sort_keys=True) for command in budget) + "\n",
        encoding="utf-8",
    )

    abnormal = [
        {
            "id": "CMD-ABN-MIXED-V1",
            "op": "load_exchange_file",
            "version": 1,
            "path": "injections/mixed_v1.txt",
            "injection_id": "INJ-MIXED-V1",
            "expected_code": "E_FIELD_TRUNCATED",
        },
        {"id": "CMD-ABN-RESTORE-V2", "op": "set_exchange_version", "version": 2},
    ]
    (scenarios / "abnormal_sequence.jsonl").write_text(
        "\n".join(json.dumps(command, ensure_ascii=False, sort_keys=True) for command in abnormal) + "\n",
        encoding="utf-8",
    )


def generate_sample_workspace(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    sample_dir = root / "samples"
    image_dir = sample_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for variant, name in enumerate(("scene_alpha.png", "scene_beta.png", "scene_gamma.png")):
        rgb = _base_image(IMAGE_SIZE[0], IMAGE_SIZE[1], SAMPLE_SEED + variant * 101, variant)
        _write_png(image_dir / name, rgb)

    regions = _sample_regions()
    (sample_dir / "edits.json").write_text(
        json.dumps(_manifest_payload(regions), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    v2_text = encode_v2(regions)
    (sample_dir / "exchange_v2.json").write_text(v2_text, encoding="utf-8")

    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.json").write_text(
        json.dumps(_config(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (sample_dir / "config.json").write_text(
        json.dumps(_config(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    from .image_registry import load_images
    from .codec_v2 import parse_v2

    images = load_images(image_dir, expected_size=IMAGE_SIZE)
    v2_outcome = parse_v2(v2_text, images=images, manifest_keys={region.region_key for region in regions})
    if v2_outcome.failures:
        raise RuntimeError(f"generated v2 sample failed validation: {v2_outcome.failures}")
    # Emulate a real downgrade: parse the v2 exchange first so provenance and source
    # fingerprints in the sidecar match a direct v2 deserialization.
    downgraded = downgrade_v2(v2_outcome.regions)
    (sample_dir / "exchange_v1.txt").write_text(downgraded.text, encoding="utf-8")
    (sample_dir / "exchange_v1.sidecar.json").write_text(downgraded.sidecar_json or "", encoding="utf-8")
    _write_injections(root, downgraded.text, images, regions)
    _write_scenarios(root)
    return {
        "sample_dir": str(sample_dir),
        "image_count": len(images),
        "region_count": len(regions),
        "seed": SAMPLE_SEED,
        "size": list(IMAGE_SIZE),
    }
