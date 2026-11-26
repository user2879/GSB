from __future__ import annotations

import math
import random
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import WorkbenchConfig
from .fonts import builtin_glyph
from .hashing import fingerprint, sha256_bytes
from .pngcodec import encode_png

GLYPH_SEQUENCE = "永書學山水雲月竹風石田國天地玄黃宇宙洪荒日月盈昃辰宿列張"
BLOCK = 48
COLUMN_PITCH = 72
RIGHT_CENTER_X = 440
TOP_CENTER_Y = 82
ROW_PITCH = 66


@dataclass(frozen=True)
class SampleSpec:
    image_id: str
    seed: int
    column_counts: tuple[int, ...]
    glyph_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_specs(count: int = 6, seed: int = 20261127) -> list[SampleSpec]:
    rng = random.Random(seed)
    specs: list[SampleSpec] = []
    for index in range(count):
        columns = 3 + (index % 3)
        counts = tuple(5 + rng.randrange(5) for _ in range(columns))
        specs.append(
            SampleSpec(
                image_id=f"sample_{index + 1:03d}",
                seed=seed + index * 1009,
                column_counts=counts,
                glyph_offset=index * 7,
            )
        )
    return specs


def column_geometry(columns: int) -> list[dict[str, int]]:
    result = []
    for rank in range(columns):
        center_x = RIGHT_CENTER_X - rank * COLUMN_PITCH
        result.append(
            {
                "column_rank": rank,
                "center_x": center_x,
                "x_min": center_x - 28,
                "x_max": center_x + 29,
            }
        )
    return result


def _render_block(character: str, rng: random.Random) -> np.ndarray:
    glyph = builtin_glyph(character)
    raw = np.full((BLOCK, BLOCK), 255, dtype=np.uint8)
    scale = BLOCK // 16
    for gy, row in enumerate(glyph):
        for gx, on in enumerate(row):
            if not on:
                continue
            jx = rng.randint(-1, 1)
            jy = rng.randint(-1, 1)
            x0 = gx * scale + jx
            y0 = gy * scale + jy
            x1 = min(BLOCK, x0 + scale)
            y1 = min(BLOCK, y0 + scale)
            x0 = max(0, x0)
            y0 = max(0, y0)
            if x1 > x0 and y1 > y0:
                raw[y0:y1, x0:x1] = 0

    angle = math.radians(rng.randint(-6, 6))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    center = (BLOCK - 1) / 2.0
    rotated = np.full((BLOCK, BLOCK), 255, dtype=np.uint8)
    for y in range(BLOCK):
        dy = y - center
        for x in range(BLOCK):
            dx = x - center
            source_x = int(math.floor(center + cos_a * dx + sin_a * dy + 0.5))
            source_y = int(math.floor(center - sin_a * dx + cos_a * dy + 0.5))
            if 0 <= source_x < BLOCK and 0 <= source_y < BLOCK:
                rotated[y, x] = raw[source_y, source_x]
    return rotated


def build_sample(spec: SampleSpec, config: WorkbenchConfig) -> dict[str, Any]:
    width, height = config.width, config.height
    rng = random.Random(spec.seed)
    pixels = np.full((height, width), 255, dtype=np.uint8)
    noise = np.random.default_rng(spec.seed & 0xFFFF).integers(0, 4, size=(height, width))
    pixels = np.clip(pixels.astype(np.int16) - noise.astype(np.int16), 0, 255).astype(np.uint8)

    columns = column_geometry(len(spec.column_counts))
    keypoints: list[dict[str, Any]] = []
    reading_order = 0
    for column in columns:
        rank = column["column_rank"]
        for row_index in range(int(spec.column_counts[rank])):
            reading_order += 1
            center_x = column["center_x"] + rng.randint(-1, 1)
            center_y = TOP_CENTER_Y + row_index * ROW_PITCH + rng.randint(-2, 2)
            character = GLYPH_SEQUENCE[
                (spec.glyph_offset + reading_order - 1) % len(GLYPH_SEQUENCE)
            ]
            block = _render_block(character, rng)
            x0_place = center_x - BLOCK // 2
            y0_place = center_y - BLOCK // 2
            mask = block < 176
            region = pixels[y0_place : y0_place + BLOCK, x0_place : x0_place + BLOCK]
            region[mask] = block[mask]

            ys, xs = np.where(mask)
            if len(xs) == 0:
                raise RuntimeError("rendered empty glyph")
            bbox = [
                int(x0_place + xs.min()),
                int(y0_place + ys.min()),
                int(x0_place + xs.max() + 1),
                int(y0_place + ys.max() + 1),
            ]
            keypoints.append(
                {
                    "keypoint_id": f"{spec.image_id}-p{reading_order:03d}",
                    "x": int(center_x),
                    "y": int(center_y),
                    "column_rank": rank,
                    "reading_order": reading_order,
                    "bbox": bbox,
                    "character": character,
                }
            )

    png_bytes = encode_png(pixels)
    truth = {
        "schema": "vertical-truth/v1",
        "image_id": spec.image_id,
        "png_file": f"{spec.image_id}.png",
        "width": width,
        "height": height,
        "mode": "L",
        "seed": spec.seed,
        "columns": columns,
        "keypoints": keypoints,
    }
    truth["content_digest"] = fingerprint({k: v for k, v in truth.items() if k != "content_digest"})
    return {
        "spec": spec.to_dict(),
        "png_bytes": png_bytes,
        "image_fingerprint": sha256_bytes(pixels.tobytes()),
        "png_fingerprint": sha256_bytes(png_bytes),
        "truth": truth,
    }


def annotation_document_from_truth(
    truth: dict[str, Any], version_id: str, coordinate_jitter: int = 0
) -> dict[str, Any]:
    rng = random.Random(f"annotation:{version_id}:{truth['image_id']}")
    points = []
    for point in truth["keypoints"]:
        dx = rng.randint(-coordinate_jitter, coordinate_jitter) if coordinate_jitter else 0
        dy = rng.randint(-coordinate_jitter, coordinate_jitter) if coordinate_jitter else 0
        item = {key: point[key] for key in ("keypoint_id", "x", "y", "column_rank", "reading_order")}
        item["x"] = int(point["x"] + dx)
        item["y"] = int(point["y"] + dy)
        points.append(item)
    body = {"version_id": version_id, "keypoints": points}
    body["content_digest"] = fingerprint(points)
    return body


def build_annotation_bundle(
    truths: list[dict[str, Any]], version_id: str = "kv-0001", jitter: int = 0
) -> dict[str, Any]:
    documents = {}
    for index, truth in enumerate(truths, start=1):
        doc_version = f"{version_id}-doc{index:03d}"
        documents[truth["image_id"]] = annotation_document_from_truth(truth, doc_version, jitter)
    bundle = {
        "schema": "vertical-keypoint-versions/v1",
        "version_id": version_id,
        "documents": documents,
    }
    bundle["content_digest"] = fingerprint(
        {image_id: doc["content_digest"] for image_id, doc in sorted(documents.items())}
    )
    return bundle


def write_sample_workspace(
    root: str | Path,
    config: WorkbenchConfig,
    specs: list[SampleSpec] | None = None,
    annotation_version: str = "kv-0001",
) -> dict[str, Any]:
    root = Path(root)
    sample_dir = root / "samples"
    truth_dir = root / "truth"
    annotation_dir = root / "annotations"
    config_dir = root / "configs"
    for path in (sample_dir, truth_dir, annotation_dir, config_dir):
        path.mkdir(parents=True, exist_ok=True)

    specs = specs or default_specs(config.sample_count, config.seed)
    truths = []
    for spec in specs:
        bundle = build_sample(spec, config)
        (sample_dir / f"{spec.image_id}.png").write_bytes(bundle["png_bytes"])
        truth_path = truth_dir / f"{spec.image_id}.json"
        truth_path.write_text(
            __import__("json").dumps(bundle["truth"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        truths.append(bundle["truth"])

    (sample_dir / "specs.json").write_text(
        json.dumps([spec.to_dict() for spec in specs], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    annotations = build_annotation_bundle(truths, annotation_version)
    (annotation_dir / "keypoint_versions.json").write_text(
        __import__("json").dumps(annotations, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (config_dir / "workbench.json").write_text(
        __import__("json").dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"specs": [s.to_dict() for s in specs], "annotation": annotations}
