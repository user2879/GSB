"""Deterministic vertical-page sample generator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from gsb.config import RunConfig
from gsb.fonts import BASE_GLYPHS, template_manifest_payload
from gsb.fixed_image import ink_bbox, regress_center
from gsb.models import BBox
from gsb.util import sha256_bytes, sha256_obj


@dataclass(frozen=True)
class PageSpec:
    sample_id: str
    seed: int
    columns: tuple[tuple[int, int], ...]
    characters: tuple[tuple[str, ...], ...]


PAGE_A = PageSpec(
    sample_id="page_a",
    seed=1701,
    columns=((260, 320), (208, 260), (156, 208), (104, 156), (52, 104)),
    characters=(
        ("古", "籍", "数", "字"),
        ("化", "首", "版", "竖"),
        ("排", "校", "己", "已"),
        ("未", "末", "日", "曰"),
        ("古", "字", "化", "首"),
    ),
)

PAGE_B = PageSpec(
    sample_id="page_b",
    seed=2903,
    columns=((260, 320), (208, 260), (156, 208), (104, 156)),
    characters=(
        ("版", "籍", "古", "数"),
        ("字", "化", "排", "校"),
        ("已", "己", "末", "未"),
        ("首", "曰", "日", "古"),
    ),
)

PAGE_C = PageSpec(
    sample_id="page_c",
    seed=3301,
    columns=((208, 320), (104, 208), (0, 104)),
    characters=(
        ("古", "籍", "数"),
        ("化", "首", "版"),
        ("己", "未", "日"),
    ),
)

FLIP_TEMPLATE_VERSION = "bitmap-flip-v1"
FLIP_PATTERN = (
    "............",
    "..########..",
    "..#......#..",
    ".........#..",
    "..#.##.###..",
    "....#..#.#..",
    "..#......#..",
    "..#......#.#",
    "..########..",
    "............",
    "............",
    "#........#..",
)

Y_CENTERS = (38, 88, 138, 188)
TILE_SIZE = 32
SCALE = 2


def _glyph_tile(char: str, rng: np.random.Generator) -> np.ndarray:
    rows = BASE_GLYPHS[char]
    mask = np.array([[1 if value == "#" else 0 for value in row] for row in rows], dtype=np.uint8)
    mask_image = Image.fromarray(mask * 255, mode="L").resize(
        (mask.shape[1] * SCALE, mask.shape[0] * SCALE),
        resample=Image.Resampling.NEAREST,
    )
    angle = float(rng.uniform(-2.2, 2.2))
    mask_image = mask_image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=0)
    shift_x = int(rng.integers(-1, 2))
    shift_y = int(rng.integers(-1, 2))
    mask_image = Image.fromarray(
        np.roll(np.asarray(mask_image), (shift_y, shift_x), axis=(0, 1)),
        mode="L",
    )
    alpha = np.asarray(mask_image, dtype=np.float64) / 255.0
    ink_value = float(rng.integers(18, 48))
    tile = np.full((TILE_SIZE, TILE_SIZE), 242.0, dtype=np.float64)
    glyph_width = mask_image.width
    glyph_height = mask_image.height
    left = (TILE_SIZE - glyph_width) // 2
    top = (TILE_SIZE - glyph_height) // 2
    local_alpha = alpha * (0.86 + 0.14 * float(rng.random()))
    tile[top : top + glyph_height, left : left + glyph_width] = (
        242.0 * (1.0 - local_alpha) + ink_value * local_alpha
    )
    noise_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    noise_positions = noise_rng.integers(0, TILE_SIZE, size=(5, 2))
    for y, x in noise_positions:
        tile[y, x] = min(tile[y, x], 205.0)
    return np.clip(np.rint(tile), 0, 255).astype(np.uint8)


def _render_page(
    spec: PageSpec,
    *,
    config: RunConfig,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(spec.seed)
    page = np.full((config.layout.height, config.layout.width), 242, dtype=np.uint8)
    keypoints: list[dict[str, Any]] = []
    reading_index = 0
    for column_index, (x_min, x_max) in enumerate(spec.columns):
        center_x = (x_min + x_max) // 2
        for row_index, char in enumerate(spec.characters[column_index]):
            center_y = Y_CENTERS[row_index]
            tile = _glyph_tile(char, rng)
            top = center_y - TILE_SIZE // 2
            left = center_x - TILE_SIZE // 2
            page[top : top + TILE_SIZE, left : left + TILE_SIZE] = tile
            search_bbox = BBox(left, top, left + TILE_SIZE, top + TILE_SIZE)
            bbox = ink_bbox(tile, config.thresholds.ink)
            assert bbox is not None
            global_bbox = BBox(
                bbox.left + left,
                bbox.top + top,
                bbox.right + left,
                bbox.bottom + top,
            )
            regression = regress_center(
                page,
                threshold=config.thresholds.ink,
                bits=config.fixed_bits,
                bbox=global_bbox,
            )
            assert regression is not None
            keypoint_id = f"{spec.sample_id}-c{column_index}-r{row_index}"
            keypoints.append(
                {
                    "id": keypoint_id,
                    "x": int(regression["x"]),
                    "y": int(regression["y"]),
                    "bbox": global_bbox.to_list(),
                    "column": column_index,
                    "reading_index": reading_index,
                    "text": char,
                    "template_id": f"U+{ord(char):04X}",
                }
            )
            reading_index += 1
    return page, keypoints, [
        {"index": index, "x_min": bounds[0], "x_max": bounds[1]}
        for index, bounds in enumerate(spec.columns)
    ]


def _annotation_payload(
    *,
    sample_id: str,
    keypoints: list[dict[str, Any]],
    version: str,
    base_version: str | None,
    changes: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    points = []
    for keypoint in keypoints:
        x = keypoint["x"]
        y = keypoint["y"]
        if changes and keypoint["id"] in changes:
            x, y = changes[keypoint["id"]]
        points.append({"id": keypoint["id"], "x": x, "y": y})
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "version": version,
        "keypoints": points,
    }
    if base_version is not None:
        payload["base_version"] = base_version
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_png(path: Path, array: np.ndarray) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array, mode="L")
    image.save(path, format="PNG", optimize=False, compress_level=9)
    payload = path.read_bytes()
    return sha256_bytes(payload), sha256_obj({"png_bytes_sha256": sha256_bytes(payload)})


def _generate_quantization_flip(
    *,
    root_path: Path,
    samples_dir: Path,
    annotations_dir: Path,
    configs_dir: Path,
    fonts_dir: Path,
    template_payload: dict[str, Any],
) -> dict[str, Any]:
    page = np.full((RunConfig().layout.height, RunConfig().layout.width), 242, dtype=np.uint8)
    mask = np.array(
        [[1 if character == "#" else 0 for character in row] for row in FLIP_PATTERN],
        dtype=np.uint8,
    )
    block_origin_x = 148
    block_origin_y = 116
    block = np.where(
        np.kron(mask, np.ones((2, 2), dtype=bool)),
        0,
        242,
    ).astype(np.uint8)
    page[
        block_origin_y : block_origin_y + mask.shape[0] * 2,
        block_origin_x : block_origin_x + mask.shape[1] * 2,
    ] = block

    image_path = samples_dir / "quantization_flip.png"
    image_sha, _ = _write_png(image_path, page)
    ys, xs = np.nonzero(mask)
    bbox = [
        block_origin_x + 2 * int(xs.min()),
        block_origin_y + 2 * int(ys.min()),
        block_origin_x + 2 * (int(xs.max()) + 1),
        block_origin_y + 2 * (int(ys.max()) + 1),
    ]
    keypoint_id = "quantization_flip-c0-r0"
    keypoint = {
        "id": keypoint_id,
        "x": 160,
        "y": 128,
        "bbox": bbox,
        "column": 0,
        "reading_index": 0,
        "text": "曰",
        "template_id": f"U+{ord('曰'):04X}",
    }
    manifest = {
        "schema_version": 1,
        "sample_id": "quantization_flip",
        "image": image_path.name,
        "image_sha256": image_sha,
        "width": RunConfig().layout.width,
        "height": RunConfig().layout.height,
        "pixel_mode": RunConfig().layout.pixel_mode,
        "template_version": FLIP_TEMPLATE_VERSION,
        "seed": 20261128,
        "columns": [{"index": 0, "x_min": 0, "x_max": RunConfig().layout.width}],
        "keypoints": [keypoint],
        "reading_order": [keypoint_id],
    }
    manifest_path = samples_dir / "quantization_flip.manifest.json"
    _write_json(manifest_path, manifest)
    annotation_path = annotations_dir / "quantization_flip.truth-v1.json"
    _write_json(
        annotation_path,
        {
            "schema_version": 1,
            "sample_id": "quantization_flip",
            "version": "quantization-flip-truth-v1",
            "keypoints": [{"id": keypoint_id, "x": 160, "y": 128}],
        },
    )
    dataset_path = samples_dir / "quantization_flip.dataset.yaml"
    _write_yaml(
        dataset_path,
        {
            "schema_version": 1,
            "samples": [
                {
                    "id": "quantization_flip",
                    "manifest": "quantization_flip.manifest.json",
                    "annotation": "annotations/quantization_flip.truth-v1.json",
                }
            ],
        },
    )

    flip_config = RunConfig(
        recognition=replace(
            RunConfig().recognition,
            template_version=FLIP_TEMPLATE_VERSION,
        )
    )
    config_path = configs_dir / "quantization-flip.yaml"
    _write_yaml(config_path, flip_config.to_dict())

    flip_templates = {
        key: value
        for key, value in template_payload.items()
        if key != "content_sha256"
    }
    flip_templates["template_version"] = FLIP_TEMPLATE_VERSION
    flip_templates["glyphs"] = [
        glyph
        for glyph in template_payload["glyphs"]
        if glyph["char"] in {"己", "曰"}
    ]
    flip_templates["content_sha256"] = sha256_obj(flip_templates)
    template_path = fonts_dir / "bitmap-flip-v1.json"
    _write_json(template_path, flip_templates)

    return {
        "sample_id": "quantization_flip",
        "image": str(image_path.relative_to(root_path)),
        "manifest": str(manifest_path.relative_to(root_path)),
        "annotation": str(annotation_path.relative_to(root_path)),
        "dataset": str(dataset_path.relative_to(root_path)),
        "config": str(config_path.relative_to(root_path)),
        "templates": str(template_path.relative_to(root_path)),
        "template_version": FLIP_TEMPLATE_VERSION,
        "config_hash": flip_config.config_hash,
        "image_sha256": image_sha,
        "keypoints": 1,
        "expected": {"Q4": "己", "Q6": "曰", "Q8": "曰", "Q10": "曰"},
    }


def generate_project(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    config = RunConfig()
    fonts_dir = root_path / "fonts"
    samples_dir = root_path / "samples"
    annotations_dir = samples_dir / "annotations"
    configs_dir = root_path / "configs"
    for directory in (fonts_dir, samples_dir, annotations_dir, configs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    template_payload = template_manifest_payload()
    _write_json(fonts_dir / "bitmap-v1.json", template_payload)
    _write_yaml(configs_dir / "default.yaml", config.to_dict())
    _write_yaml(
        configs_dir / "template_paths.yaml",
        {
            "bitmap-v1": "fonts/bitmap-v1.json",
            FLIP_TEMPLATE_VERSION: "fonts/bitmap-flip-v1.json",
        },
    )

    dataset_entries: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    for spec in (PAGE_A, PAGE_B):
        page, keypoints, columns = _render_page(spec, config=config)
        image_path = samples_dir / f"{spec.sample_id}.png"
        image_sha, _ = _write_png(image_path, page)
        manifest = {
            "schema_version": 1,
            "sample_id": spec.sample_id,
            "image": image_path.name,
            "image_sha256": image_sha,
            "width": config.layout.width,
            "height": config.layout.height,
            "pixel_mode": config.layout.pixel_mode,
            "template_version": config.recognition.template_version,
            "seed": spec.seed,
            "columns": columns,
            "keypoints": keypoints,
            "reading_order": [item["id"] for item in keypoints],
        }
        manifest_path = samples_dir / f"{spec.sample_id}.manifest.json"
        _write_json(manifest_path, manifest)

        v1_name = f"{spec.sample_id}.truth-v1.json"
        v1_as = _annotation_payload(
            sample_id=spec.sample_id,
            keypoints=keypoints,
            version=f"{spec.sample_id}-truth-v1",
            base_version=None,
        )
        v1_path = annotations_dir / v1_name
        _write_json(v1_path, v1_as)

        changes: dict[str, tuple[int, int]] = {}
        if keypoints:
            first = keypoints[0]
            changes[first["id"]] = (
                min(config.layout.width - 1, first["x"] + 3),
                min(config.layout.height - 1, first["y"] + 2),
            )
        if len(keypoints) > 2:
            second = keypoints[2]
            changes[second["id"]] = (
                max(0, second["x"] - 2),
                max(0, second["y"] + 1),
            )
        v2_name = f"{spec.sample_id}.ann-v2.json"
        v2_as = _annotation_payload(
            sample_id=spec.sample_id,
            keypoints=keypoints,
            version=f"{spec.sample_id}-ann-v2",
            base_version=f"{spec.sample_id}-truth-v1",
            changes=changes,
        )
        _write_json(annotations_dir / v2_name, v2_as)

        dataset_entries.append(
            {
                "id": spec.sample_id,
                "manifest": f"{spec.sample_id}.manifest.json",
                "annotation": f"annotations/{v1_name}",
            }
        )
        summaries.append(
            {
                "sample_id": spec.sample_id,
                "image": str(image_path.relative_to(root_path)),
                "image_sha256": image_sha,
                "keypoints": len(keypoints),
                "annotation_v1": str(v1_path.relative_to(root_path)),
                "annotation_v2": str((annotations_dir / v2_name).relative_to(root_path)),
            }
        )

    dataset_payload = {"schema_version": 1, "samples": dataset_entries}
    _write_yaml(samples_dir / "dataset.yaml", dataset_payload)

    extra_summaries: list[dict[str, Any]] = []
    extra_specs = (
        (replace(PAGE_A, seed=1702), "page_a.alt", "page_a"),
        (PAGE_C, "page_c", "page_c"),
    )
    for spec, stem, manifest_sample_id in extra_specs:
        page, keypoints, columns = _render_page(spec, config=config)
        image_path = samples_dir / f"{stem}.png"
        image_sha, _ = _write_png(image_path, page)
        manifest = {
            "schema_version": 1,
            "sample_id": manifest_sample_id,
            "image": image_path.name,
            "image_sha256": image_sha,
            "width": config.layout.width,
            "height": config.layout.height,
            "pixel_mode": config.layout.pixel_mode,
            "template_version": config.recognition.template_version,
            "seed": spec.seed,
            "columns": columns,
            "keypoints": keypoints,
            "reading_order": [item["id"] for item in keypoints],
        }
        manifest_path = samples_dir / f"{stem}.manifest.json"
        _write_json(manifest_path, manifest)
        annotation_name = f"{stem}.truth-v1.json"
        annotation_payload = _annotation_payload(
            sample_id=manifest_sample_id,
            keypoints=keypoints,
            version=f"{manifest_sample_id}-{stem}-truth-v1",
            base_version=None,
        )
        annotation_path = annotations_dir / annotation_name
        _write_json(annotation_path, annotation_payload)
        extra_summaries.append(
            {
                "sample_id": manifest_sample_id,
                "stem": stem,
                "image": str(image_path.relative_to(root_path)),
                "manifest": str(manifest_path.relative_to(root_path)),
                "annotation": str(annotation_path.relative_to(root_path)),
                "image_sha256": image_sha,
                "keypoints": len(keypoints),
            }
        )
    flip_summary = _generate_quantization_flip(
        root_path=root_path,
        samples_dir=samples_dir,
        annotations_dir=annotations_dir,
        configs_dir=configs_dir,
        fonts_dir=fonts_dir,
        template_payload=template_payload,
    )
    _write_json(
        samples_dir / "generation.json",
        {
            "schema_version": 1,
            "seed": 20261127,
            "config_hash": config.config_hash,
            "template_version": config.recognition.template_version,
            "samples": summaries,
            "extras": extra_summaries,
            "quantization_flip": flip_summary,
        },
    )
    return {
        "root": str(root_path),
        "config_hash": config.config_hash,
        "template_version": config.recognition.template_version,
        "samples": summaries,
        "extras": extra_summaries,
        "quantization_flip": flip_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic GSB samples")
    parser.add_argument("--root", default=".", help="project root")
    args = parser.parse_args(argv)
    summary = generate_project(args.root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
