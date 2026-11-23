from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import Category, WorkbenchConfig
from .failures import Failure, sort_failures
from .fingerprint import PNG_SIGNATURE, hash_json, image_content_hash


@dataclass(frozen=True)
class ImageSpec:
    name: str
    image_path: Path
    truth_path: Path


@dataclass
class TruthInstance:
    instance_id: int
    category_code: int
    bbox: tuple[int, int, int, int]
    area: int
    mask_pixels: tuple[tuple[int, int], ...] | None = None

    def mask(self, width: int, height: int) -> np.ndarray:
        result = np.zeros((height, width), dtype=bool)
        if self.mask_pixels is not None:
            for x, y in self.mask_pixels:
                if 0 <= x < width and 0 <= y < height:
                    result[y, x] = True
            return result
        x, y, rect_width, rect_height = self.bbox
        remaining = self.area
        for row in range(y, y + rect_height):
            if remaining <= 0:
                break
            count = min(rect_width, remaining)
            result[row, x : x + count] = True
            remaining -= count
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "bbox": list(self.bbox),
            "category_code": self.category_code,
            "instance_id": self.instance_id,
        }


@dataclass
class Sample:
    name: str
    image_path: Path
    truth_path: Path
    width: int
    height: int
    rgb: np.ndarray
    content_hash: str
    truth_hash: str
    truth_instances: tuple[TruthInstance, ...]

    def truth_semantic_raster(self) -> np.ndarray:
        raster = np.zeros((self.height, self.width), dtype=np.int32)
        for instance in sorted(self.truth_instances, key=lambda item: (item.instance_id, item.category_code)):
            raster[instance.mask(self.width, self.height)] = instance.category_code
        return raster

    def truth_instance_raster(self) -> np.ndarray:
        raster = np.zeros((self.height, self.width), dtype=np.int32)
        for instance in sorted(self.truth_instances, key=lambda item: (item.instance_id, item.category_code)):
            mask = instance.mask(self.width, self.height)
            raster[mask] = instance.instance_id
        return raster


@dataclass
class ValidationResult:
    samples: list[Sample]
    failures: list[Failure]
    image_specs: list[ImageSpec]
    summary: dict[str, Any] = field(default_factory=dict)


def discover_image_specs(samples_dir: str | Path) -> list[ImageSpec]:
    root = Path(samples_dir).resolve()
    if not root.is_dir():
        return []
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    specs: list[ImageSpec] = []
    for image_path in sorted(images, key=lambda path: path.relative_to(root).as_posix()):
        relative = image_path.relative_to(root)
        candidates = [
            image_path.with_suffix(".json"),
            image_path.with_name(image_path.stem + ".truth.json"),
        ]
        truth_path = next((path for path in candidates if path.exists()), candidates[0])
        specs.append(
            ImageSpec(
                name=relative.as_posix(),
                image_path=image_path.resolve(),
                truth_path=truth_path.resolve(),
            )
        )
    return specs


def load_inputs(
    samples_dir: str | Path,
    config: WorkbenchConfig,
    specs: list[ImageSpec] | None = None,
) -> ValidationResult:
    return validate_samples(discover_image_specs(samples_dir), config, samples_dir)


def validate_samples(
    specs: list[ImageSpec],
    config: WorkbenchConfig,
    samples_dir: str | Path | None = None,
) -> ValidationResult:
    failures: list[Failure] = []
    samples: list[Sample] = []
    ordered_specs = sorted(specs, key=lambda spec: spec.name)
    category_by_name = {category.name: category for category in config.categories}
    category_by_code = config.category_by_code

    for spec in ordered_specs:
        image_failure = _load_image(spec, config)
        if isinstance(image_failure, Failure):
            failures.append(image_failure)
            continue
        width, height, rgb, content_hash = image_failure
        if width % config.window_size != 0 or height % config.window_size != 0:
            failures.append(
                Failure(
                    code="E_CONFIG_INVALID",
                    item=spec.name,
                    stage="load",
                    message="window size must divide both image dimensions",
                    details={
                        "height": height,
                        "width": width,
                        "window_size": config.window_size,
                    },
                )
            )
            continue
        truth_failures, truth_instances, truth_hash = _load_truth(
            spec,
            width,
            height,
            category_by_name,
            category_by_code,
        )
        if truth_failures:
            failures.extend(truth_failures)
            continue
        samples.append(
            Sample(
                name=spec.name,
                image_path=spec.image_path,
                truth_path=spec.truth_path,
                width=width,
                height=height,
                rgb=rgb,
                content_hash=content_hash,
                truth_hash=truth_hash,
                truth_instances=tuple(truth_instances),
            )
        )

    summary = {
        "failed_specs": len(ordered_specs) - len(samples),
        "failure_count": len(failures),
        "sample_count": len(samples),
        "sample_names": [sample.name for sample in samples],
    }
    if samples_dir is not None:
        summary["samples_dir"] = str(Path(samples_dir).resolve())
    return ValidationResult(
        samples=samples,
        failures=sort_failures(failures),
        image_specs=ordered_specs,
        summary=summary,
    )


def _load_image(spec: ImageSpec, config: WorkbenchConfig) -> tuple[int, int, np.ndarray, str] | Failure:
    path = spec.image_path
    if not path.exists():
        return Failure(
            code="E_IMAGE_FORMAT",
            item=spec.name,
            stage="load",
            message="image file does not exist",
            details={"path": str(path)},
        )
    try:
        header = path.read_bytes()[:33]
    except OSError as exc:
        return Failure(
            code="E_IMAGE_FORMAT",
            item=spec.name,
            stage="load",
            message="image file cannot be read",
            details={"error": str(exc)},
        )
    if len(header) < 33 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        return Failure(
            code="E_IMAGE_FORMAT",
            item=spec.name,
            stage="load",
            message="invalid PNG container header",
        )
    header_width, header_height = struct.unpack(">II", header[16:24])
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG":
                return Failure(
                    code="E_IMAGE_FORMAT",
                    item=spec.name,
                    stage="load",
                    message="image format is not PNG",
                    details={"format": image.format},
                )
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            if (width, height) != (header_width, header_height):
                return Failure(
                    code="E_IMAGE_FORMAT",
                    item=spec.name,
                    stage="load",
                    message="PNG IHDR dimensions do not match decoded image",
                    details={
                        "decoded_height": height,
                        "decoded_width": width,
                        "header_height": header_height,
                        "header_width": header_width,
                    },
                )
            rgb = np.asarray(rgb_image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return Failure(
            code="E_IMAGE_FORMAT",
            item=spec.name,
            stage="load",
            message="PNG decoding failed",
            details={"error": str(exc)},
        )
    if config.window_size <= 0:
        return Failure(
            code="E_CONFIG_INVALID",
            item=spec.name,
            stage="load",
            message="window size must be positive",
        )
    return width, height, rgb, image_content_hash(rgb)


def _load_truth(
    spec: ImageSpec,
    width: int,
    height: int,
    category_by_name: Mapping[str, Category],
    category_by_code: Mapping[int, Category],
) -> tuple[list[Failure], list[TruthInstance], str]:
    failures: list[Failure] = []
    path = spec.truth_path
    if not path.exists():
        return (
            [
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message="truth JSON is missing",
                    details={"path": str(path)},
                )
            ],
            [],
            "",
        )
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            [
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message="truth JSON cannot be parsed",
                    details={"error": str(exc)},
                )
            ],
            [],
            "",
        )
    if not isinstance(payload, dict):
        return (
            [
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message="truth JSON root must be a mapping",
                )
            ],
            [],
            "",
        )
    truth_width = payload.get("width")
    truth_height = payload.get("height")
    if truth_width is not None and truth_width != width:
        failures.append(
            Failure(
                code="E_TRUTH_INVALID",
                item=spec.name,
                stage="load",
                message="truth width does not match image width",
                details={"image_width": width, "truth_width": truth_width},
            )
        )
    if truth_height is not None and truth_height != height:
        failures.append(
            Failure(
                code="E_TRUTH_INVALID",
                item=spec.name,
                stage="load",
                message="truth height does not match image height",
                details={"image_height": height, "truth_height": truth_height},
            )
        )
    instances_raw = payload.get(
        "instances", payload.get("objects", payload.get("segments"))
    )
    if not isinstance(instances_raw, list):
        failures.append(
            Failure(
                code="E_TRUTH_INVALID",
                item=spec.name,
                stage="load",
                message="truth instances must be a list",
            )
        )
        return failures, [], hash_json(payload)

    instances: list[TruthInstance] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(instances_raw):
        prefix = f"instances[{index}]"
        if not isinstance(raw, dict):
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} must be a mapping",
                )
            )
            continue
        instance_id = raw.get("id", raw.get("instance_id", index + 1))
        if isinstance(instance_id, bool) or not isinstance(instance_id, int):
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} id must be an integer",
                )
            )
            continue
        if instance_id in seen_ids:
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} id is duplicated",
                    details={"instance_id": instance_id},
                )
            )
            continue
        seen_ids.add(instance_id)

        category_code = _resolve_category(raw, category_by_name, category_by_code)
        if category_code is None:
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} category is not in the configured category set",
                    details={
                        "category": raw.get("category", raw.get("category_code")),
                    },
                )
            )
            continue

        bbox_raw = raw.get(
            "bbox",
            raw.get(
                "rect",
                raw.get("mask_rect", raw.get("rectangle")),
            ),
        )
        if bbox_raw is None and all(
            key in raw for key in ("x", "y", "w", "h")
        ):
            bbox_raw = [raw["x"], raw["y"], raw["w"], raw["h"]]
        if (
            not isinstance(bbox_raw, (list, tuple))
            or len(bbox_raw) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in bbox_raw
            )
        ):
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} bbox must contain four integers",
                )
            )
            continue
        x, y, rect_width, rect_height = (int(value) for value in bbox_raw)
        if (
            x < 0
            or y < 0
            or rect_width <= 0
            or rect_height <= 0
            or x + rect_width > width
            or y + rect_height > height
        ):
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} bbox is outside the image bounds",
                    details={"bbox": [x, y, rect_width, rect_height], "height": height, "width": width},
                )
            )
            continue

        area = raw.get(
            "area", raw.get("mask_area", raw.get("pixel_area"))
        )
        if isinstance(area, bool) or not isinstance(area, int) or area <= 0:
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} mask area must be a positive integer",
                )
            )
            continue
        if area > rect_width * rect_height:
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} mask area exceeds bbox area",
                    details={"area": area, "bbox_area": rect_width * rect_height},
                )
            )
            continue

        mask_pixels = _parse_mask_pixels(raw)
        if isinstance(mask_pixels, Failure):
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} {mask_pixels.message}",
                    details=mask_pixels.details,
                )
            )
            continue
        if mask_pixels is not None and len(mask_pixels) != area:
            failures.append(
                Failure(
                    code="E_TRUTH_INVALID",
                    item=spec.name,
                    stage="load",
                    message=f"{prefix} mask pixel count does not equal area",
                    details={"area": area, "pixel_count": len(mask_pixels)},
                )
            )
            continue

        instances.append(
            TruthInstance(
                instance_id=instance_id,
                category_code=category_code,
                bbox=(x, y, rect_width, rect_height),
                area=area,
                mask_pixels=mask_pixels,
            )
        )
    return failures, instances, hash_json(payload)


def _resolve_category(
    raw: dict[str, Any],
    by_name: Mapping[str, Category],
    by_code: Mapping[int, Category],
) -> int | None:
    value = raw.get(
        "category_code",
        raw.get("category", raw.get("class_id", raw.get("class"))),
    )
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value in by_code else None
    if isinstance(value, str):
        if value in by_name:
            return by_name[value].code
        try:
            numeric = int(value)
        except ValueError:
            return None
        return numeric if numeric in by_code else None
    return None


def _parse_mask_pixels(raw: dict[str, Any]) -> tuple[tuple[int, int], ...] | None | Failure:
    value = raw.get("mask_pixels", raw.get("pixels"))
    if value is None:
        return None
    if not isinstance(value, list):
        return Failure(code="invalid", message="mask pixels must be a list", stage="load")
    result: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in item)
        ):
            return Failure(
                code="invalid",
                message="each mask pixel must be an integer pair",
                stage="load",
            )
        result.append((int(item[0]), int(item[1])))
    return tuple(result)
