"""Stage 1: sample directory and configuration validation."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .errors import (
    E_BBOX_OUT_OF_RANGE,
    E_LABEL_UNKNOWN,
    SampleError,
)
from .injection import (
    FAULT_BBOX_OUT_OF_RANGE,
    FAULT_DROP_LABEL,
    FAULT_TRUNCATE_PNG,
    FAULT_UNKNOWN_LABEL,
    VALIDATION_STAGE,
    first_active,
)
from .hashing import content_fingerprint, fingerprint_json

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class ValidSample:
    name: str
    image_path: Path
    annotation_path: Path
    png_bytes: bytes
    image_fingerprint: str
    width: int
    height: int
    labels: tuple[str, ...]
    boxes: dict[str, list[int]]
    annotation_fingerprint: str


@dataclass
class ValidationFailure:
    sample: str
    stage: str
    code: str
    message: str
    injection_id: str = ""

    def to_dict(self) -> dict:
        return {
            "sample": self.sample,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
        }


@dataclass
class ValidationResult:
    valid: list[ValidSample] = field(default_factory=list)
    failures: list[ValidationFailure] = field(default_factory=list)

    @property
    def valid_names(self) -> list[str]:
        return [sample.name for sample in self.valid]


def _load_annotation(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SampleError(f"annotation JSON unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise SampleError("annotation root must be an object")
    for required in ("image", "labels", "boxes"):
        if required not in data:
            raise SampleError(f"annotation missing field {required!r}")
    if not isinstance(data["image"], str) or not data["image"]:
        raise SampleError("annotation.image must be a non-empty string")
    if not isinstance(data["labels"], list) or any(
        not isinstance(label, str) for label in data["labels"]
    ):
        raise SampleError("annotation.labels must be a list of strings")
    if not isinstance(data["boxes"], dict):
        raise SampleError("annotation.boxes must be an object")
    return data


def _apply_validation_injection(sample_name: str, png_bytes: bytes, annotation: dict, injections):
    injection = first_active(injections, sample_name, VALIDATION_STAGE)
    if injection is None:
        return png_bytes, annotation, ""
    fault = injection.fault
    if fault == FAULT_TRUNCATE_PNG:
        # Keep the signature so we can prove the container body is corrupt.
        keep = max(len(PNG_SIGNATURE), min(len(png_bytes) - 1, len(png_bytes) // 2))
        return png_bytes[:keep], annotation, injection.injection_id
    if fault == FAULT_UNKNOWN_LABEL:
        patched = dict(annotation)
        patched["labels"] = list(annotation["labels"]) + ["__injected_unknown__"]
        boxes = dict(annotation["boxes"])
        boxes["__injected_unknown__"] = [0, 0, 4, 4]
        patched["boxes"] = boxes
        return png_bytes, patched, injection.injection_id
    if fault == FAULT_BBOX_OUT_OF_RANGE:
        patched = dict(annotation)
        if not patched["labels"]:
            patched["labels"] = ["__bbox_only__"]
        target = patched["labels"][0]
        boxes = dict(annotation["boxes"])
        width = int(annotation.get("width", 10**9))
        height = int(annotation.get("height", 10**9))
        boxes[target] = [0, 0, width + 25, height + 25]
        patched["boxes"] = boxes
        return png_bytes, patched, injection.injection_id
    if fault == FAULT_DROP_LABEL:
        # Drop a label entry while retaining its box, producing an orphan box
        # and therefore a structurally inconsistent annotation.
        patched = dict(annotation)
        if patched["labels"]:
            patched["labels"] = list(patched["labels"])[1:]
        else:
            boxes = dict(patched["boxes"])
            boxes["__orphan_box__"] = [0, 0, 2, 2]
            patched["boxes"] = boxes
        return png_bytes, patched, injection.injection_id
    raise SampleError(f"unsupported injection fault {fault!r}")


def _decode_png(png_bytes: bytes, expected_size: tuple[int, int]) -> tuple[int, int]:
    if not png_bytes.startswith(PNG_SIGNATURE):
        raise SampleError("missing or invalid PNG container header")
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
    except Exception as exc:  # Pillow raises a variety of decoder errors
        raise SampleError(f"PNG decoder failure: {exc}") from exc
    if image.mode != "RGB":
        raise SampleError(f"expected RGB image, got mode {image.mode!r}")
    width, height = image.size
    if (width, height) != tuple(expected_size):
        raise SampleError(
            f"image size {(width, height)} does not match config {tuple(expected_size)}"
        )
    return width, height


def validate_sample(
    sample_name: str,
    sample_dir: Path,
    config,
    injections,
) -> ValidSample:
    image_path = sample_dir / f"{sample_name}.png"
    annotation_path = sample_dir / f"{sample_name}.json"
    injection_id = ""
    try:
        if not image_path.is_file():
            raise SampleError(f"missing image file {image_path.name}")
        if not annotation_path.is_file():
            raise SampleError(f"missing annotation file {annotation_path.name}")
        png_bytes = image_path.read_bytes()
        annotation = _load_annotation(annotation_path)
        png_bytes, annotation, injection_id = _apply_validation_injection(
            sample_name, png_bytes, annotation, injections
        )
        width, height = _decode_png(png_bytes, config.image_size)

        labels = annotation["labels"]
        boxes = annotation["boxes"]
        # Structural completeness: every declared label needs a box and every
        # box needs a declared label. This also catches injected label drops.
        if set(labels) != set(boxes.keys()):
            missing_boxes = sorted(set(labels) - set(boxes.keys()))
            orphan_boxes = sorted(set(boxes.keys()) - set(labels))
            raise SampleError(
                "label/box pairing broken "
                f"(missing_boxes={missing_boxes}, orphan_boxes={orphan_boxes})",
                injection_id=injection_id,
            )
        label_set = set(config.labels)
        # Unknown labels are reported with a dedicated code and take precedence
        # over bbox issues so every failure maps to exactly one business code.
        for label in labels:
            if label not in label_set:
                raise SampleError(
                    f"annotation references unknown label {label!r}",
                    code=E_LABEL_UNKNOWN,
                    injection_id=injection_id,
                )
        for label in labels:
            if label not in boxes:
                raise SampleError(
                    f"label {label!r} has no bounding box",
                    injection_id=injection_id,
                )
        for label, box in boxes.items():
            if label not in labels:
                raise SampleError(
                    f"bounding box {label!r} has no matching label entry",
                    injection_id=injection_id,
                )
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
            ):
                raise SampleError(
                    f"bounding box for {label!r} must contain 4 integers",
                    injection_id=injection_id,
                )
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise SampleError(
                    f"bounding box for {label!r} {box} outside image {width}x{height}",
                    code=E_BBOX_OUT_OF_RANGE,
                    injection_id=injection_id,
                )
        if annotation.get("width") not in (None, width) or annotation.get("height") not in (
            None,
            height,
        ):
            raise SampleError(
                "annotation width/height inconsistent with decoded image",
                injection_id=injection_id,
            )
        return ValidSample(
            name=sample_name,
            image_path=image_path,
            annotation_path=annotation_path,
            png_bytes=png_bytes,
            image_fingerprint=content_fingerprint(png_bytes),
            width=width,
            height=height,
            labels=tuple(labels),
            boxes={label: list(boxes[label]) for label in labels},
            annotation_fingerprint=fingerprint_json(annotation),
        )
    except SampleError as exc:
        exc.sample = sample_name
        exc.stage = VALIDATION_STAGE
        if injection_id and not exc.injection_id:
            exc.injection_id = injection_id
        raise


def discover_sample_names(sample_dir: Path) -> list[str]:
    """Return the stable union of PNG/JSON stems, sorted lexicographically."""
    names = {path.stem for path in sample_dir.glob("*.png")}
    names |= {path.stem for path in sample_dir.glob("*.json")}
    return sorted(names)


def validate_directory(sample_dir: str | Path, config, injections=None) -> ValidationResult:
    sample_dir = Path(sample_dir)
    injections = injections or {}
    result = ValidationResult()
    for sample_name in discover_sample_names(sample_dir):
        try:
            valid = validate_sample(sample_name, sample_dir, config, injections)
        except SampleError as exc:
            result.failures.append(
                ValidationFailure(
                    sample=sample_name,
                    stage=VALIDATION_STAGE,
                    code=exc.code,
                    message=str(exc),
                    injection_id=exc.injection_id,
                )
            )
            continue
        result.valid.append(valid)
    result.valid.sort(key=lambda item: item.name)
    result.failures.sort(key=lambda item: (item.sample, item.code, item.injection_id))
    return result
