"""Sample discovery, structural validation and in-memory fault injection."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .errors import (
    E_DIMENSION_MISMATCH,
    E_IMAGE_MISSING,
    E_PNG_HEADER,
    E_PNG_TRUNCATED,
    E_REGION_INVALID,
    E_SAMPLE_INVALID,
    E_TEXT_MISSING,
    E_TEXT_SCHEMA,
    Failure,
)
from .hashing import canonical_json, hash_json, sha256_bytes
from .injections import InjectionSet


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Region:
    label: str
    rect: tuple[int, int, int, int]
    color: str | None = None
    entity_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rect": list(self.rect),
            "color": self.color,
            "entity_id": self.entity_id,
        }


@dataclass(frozen=True)
class ImageTruth:
    image_size: tuple[int, int]
    regions: tuple[Region, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "attributes": self.attributes}


@dataclass(frozen=True)
class Attribute:
    entity: str
    name: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"entity": self.entity, "name": self.name, "value": self.value}


@dataclass(frozen=True)
class Assertion:
    id: str
    entity: str
    attribute: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
        }


@dataclass(frozen=True)
class TextDescription:
    entities: tuple[Entity, ...]
    attributes: tuple[Attribute, ...]
    assertions: tuple[Assertion, ...]
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "attributes": [attribute.to_dict() for attribute in self.attributes],
            "assertions": [assertion.to_dict() for assertion in self.assertions],
            "text": self.text,
        }


@dataclass
class LoadedSample:
    name: str
    directory: Path
    image_path: Path
    regions_path: Path
    text_path: Path
    image_bytes: bytes
    image_size: tuple[int, int]
    image_truth: ImageTruth
    text: TextDescription
    image_fingerprint: str
    regions_fingerprint: str
    text_fingerprint: str
    image_injection_id: str | None = None
    text_injection_id: str | None = None

    def cache_reference(self) -> dict[str, Any]:
        return {
            "sample": self.name,
            "image_fingerprint": self.image_fingerprint,
            "regions_fingerprint": self.regions_fingerprint,
            "text_fingerprint": self.text_fingerprint,
        }


@dataclass
class LoadResult:
    samples: list[LoadedSample]
    failures: list[Failure]


def _find_one(directory: Path, explicit: str, suffixes: tuple[str, ...], predicate=None) -> Path | None:
    explicit_path = directory / explicit
    if explicit_path.is_file():
        return explicit_path
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.name.lower().endswith(suffixes)
        and (predicate is None or predicate(path))
    ]
    candidates.sort(key=lambda path: path.name)
    return candidates[0] if candidates else None


def _sample_directories(samples_dir: Path) -> list[Path]:
    if not samples_dir.exists() or not samples_dir.is_dir():
        return []
    return sorted(
        (path for path in samples_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _scalar_json(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _validate_truth(document: Any, image_size: tuple[int, int] | None = None) -> ImageTruth:
    if not isinstance(document, dict):
        raise ValueError("regions document must be an object")
    size = document.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or not all(_is_int(v) for v in size):
        raise ValueError("image_size must be [integer_width, integer_height]")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if image_size is not None and (width, height) != image_size:
        raise ValueError(
            f"image_size {(width, height)} does not match PNG dimensions {image_size}"
        )
    regions_raw = document.get("regions")
    if not isinstance(regions_raw, list):
        raise ValueError("regions must be a list")
    regions: list[Region] = []
    for index, item in enumerate(regions_raw):
        if not isinstance(item, dict):
            raise ValueError(f"regions[{index}] must be an object")
        label = item.get("label")
        rect = item.get("rect")
        if not isinstance(label, str) or not label:
            raise ValueError(f"regions[{index}].label must be a non-empty string")
        if not isinstance(rect, list) or len(rect) != 4 or not all(_is_int(v) for v in rect):
            raise ValueError(f"regions[{index}].rect must be four integers")
        x, y, region_width, region_height = rect
        if x < 0 or y < 0 or region_width <= 0 or region_height <= 0:
            raise ValueError(f"regions[{index}].rect must have non-negative origin and positive size")
        if x + region_width > width or y + region_height > height:
            raise ValueError(f"regions[{index}].rect lies outside the image")
        color = item.get("color")
        entity_id = item.get("entity_id")
        if color is not None and not isinstance(color, str):
            raise ValueError(f"regions[{index}].color must be a string")
        if entity_id is not None and not isinstance(entity_id, str):
            raise ValueError(f"regions[{index}].entity_id must be a string")
        regions.append(
            Region(
                label=label,
                rect=(x, y, region_width, region_height),
                color=color,
                entity_id=entity_id,
            )
        )
    return ImageTruth(image_size=(width, height), regions=tuple(regions))


def _validate_text(document: Any) -> TextDescription:
    if not isinstance(document, dict):
        raise ValueError("text document must be an object")
    for field_name in ("entities", "attributes", "assertions"):
        if field_name not in document or not isinstance(document[field_name], list):
            raise ValueError(f"text.{field_name} must be a list")
    text_value = document.get("text")
    if text_value is not None and not isinstance(text_value, str):
        raise ValueError("text.text must be a string when present")

    entities: list[Entity] = []
    entity_ids: set[str] = set()
    for index, item in enumerate(document["entities"]):
        if not isinstance(item, dict):
            raise ValueError(f"entities[{index}] must be an object")
        entity_id = item.get("id")
        entity_type = item.get("type")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"entities[{index}].id must be a non-empty string")
        if entity_id in entity_ids:
            raise ValueError(f"duplicate entity id: {entity_id}")
        if not isinstance(entity_type, str) or not entity_type:
            raise ValueError(f"entities[{index}].type must be a non-empty string")
        attrs = item.get("attributes", {})
        if attrs is None:
            attrs = {}
        if not isinstance(attrs, dict):
            raise ValueError(f"entities[{index}].attributes must be an object")
        entity_ids.add(entity_id)
        entities.append(Entity(id=entity_id, type=entity_type, attributes=attrs))

    attributes: list[Attribute] = []
    for index, item in enumerate(document["attributes"]):
        if not isinstance(item, dict):
            raise ValueError(f"attributes[{index}] must be an object")
        entity = item.get("entity")
        name = item.get("name")
        if entity not in entity_ids:
            raise ValueError(f"attributes[{index}].entity does not exist")
        if not isinstance(name, str) or not name:
            raise ValueError(f"attributes[{index}].name must be a non-empty string")
        if "value" not in item or not _scalar_json(item["value"]):
            raise ValueError(f"attributes[{index}].value must be a JSON scalar")
        attributes.append(Attribute(entity=entity, name=name, value=item["value"]))

    assertions: list[Assertion] = []
    assertion_ids: set[str] = set()
    for index, item in enumerate(document["assertions"]):
        if not isinstance(item, dict):
            raise ValueError(f"assertions[{index}] must be an object")
        missing = [name for name in ("id", "entity", "attribute", "value") if name not in item]
        if missing:
            raise ValueError(f"assertions[{index}] missing fields: {', '.join(missing)}")
        assertion_id = item["id"]
        entity = item["entity"]
        attribute = item["attribute"]
        if not isinstance(assertion_id, str) or not assertion_id:
            raise ValueError(f"assertions[{index}].id must be a non-empty string")
        if assertion_id in assertion_ids:
            raise ValueError(f"duplicate assertion id: {assertion_id}")
        if entity not in entity_ids:
            raise ValueError(f"assertions[{index}].entity does not exist")
        if not isinstance(attribute, str) or not attribute:
            raise ValueError(f"assertions[{index}].attribute must be a non-empty string")
        if not _scalar_json(item["value"]):
            raise ValueError(f"assertions[{index}].value must be a JSON scalar")
        assertion_ids.add(assertion_id)
        assertions.append(
            Assertion(
                id=assertion_id,
                entity=entity,
                attribute=attribute,
                value=item["value"],
            )
        )
    return TextDescription(
        entities=tuple(entities),
        attributes=tuple(attributes),
        assertions=tuple(assertions),
        text=text_value,
    )


def _validate_png(data: bytes) -> tuple[int, int]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("PNG signature is missing")
    if len(data) < 33 or data[12:16] != b"IHDR":
        raise ValueError("PNG IHDR chunk is missing")
    if b"IEND" not in data[-32:]:
        raise ValueError("PNG IEND chunk is missing or truncated")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if image.format != "PNG":
                raise ValueError("decoded image is not PNG")
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"PNG cannot be decoded: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    return width, height


def load_samples(samples_dir: Path | str, injections: InjectionSet | None = None) -> LoadResult:
    samples_dir = Path(samples_dir)
    injections = injections or InjectionSet.empty()
    samples: list[LoadedSample] = []
    failures: list[Failure] = []
    seen_names: set[str] = set()

    for directory in _sample_directories(samples_dir):
        name = directory.name
        if name in seen_names:
            failures.append(
                Failure(
                    sample=name,
                    code=E_SAMPLE_INVALID,
                    message="sample name is not unique",
                    stage="load",
                )
            )
            continue
        seen_names.add(name)
        image_path = _find_one(
            directory,
            "image.png",
            (".png",),
            predicate=lambda path: "regions" not in path.name.lower(),
        )
        regions_path = _find_one(
            directory,
            "regions.json",
            (".json",),
            predicate=lambda path: "region" in path.name.lower(),
        )
        text_path = _find_one(
            directory,
            "text.json",
            (".json",),
            predicate=lambda path: "text" in path.name.lower() and "region" not in path.name.lower(),
        )
        if image_path is None:
            failures.append(
                Failure(
                    name,
                    E_IMAGE_MISSING,
                    "image.png is missing",
                    "load_image",
                    parent_code=E_SAMPLE_INVALID,
                )
            )
            continue
        if regions_path is None:
            failures.append(
                Failure(
                    name,
                    E_IMAGE_MISSING,
                    "regions.json is missing",
                    "load_image",
                    parent_code=E_SAMPLE_INVALID,
                )
            )
            continue
        if text_path is None:
            failures.append(
                Failure(
                    name,
                    E_TEXT_MISSING,
                    "text.json is missing",
                    "load_text",
                    parent_code=E_SAMPLE_INVALID,
                )
            )
            continue

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            failures.append(
                Failure(
                    name,
                    E_IMAGE_MISSING,
                    str(exc),
                    "load_image",
                    parent_code=E_SAMPLE_INVALID,
                )
            )
            continue
        original_size = len(image_bytes)
        image_bytes, image_injection = injections.applied_image_bytes(name, image_bytes)
        if image_injection is not None:
            injections.mark_hit(
                image_injection,
                sample=name,
                stage="load_image",
                code=E_PNG_TRUNCATED,
                proof={
                    "original_bytes": original_size,
                    "injected_bytes": len(image_bytes),
                    "bytes_removed": original_size - len(image_bytes),
                },
            )
        try:
            png_size = _validate_png(image_bytes)
        except ValueError as exc:
            truncated_by_content = any(
                marker in str(exc).lower()
                for marker in ("iend", "truncat", "unexpected end")
            )
            code = (
                E_PNG_TRUNCATED
                if image_injection is not None or truncated_by_content
                else E_PNG_HEADER
            )
            failures.append(
                Failure(
                    sample=name,
                    code=code,
                    message=str(exc),
                    stage="load_image",
                    parent_code=E_SAMPLE_INVALID,
                    injection_id=image_injection.id if image_injection else None,
                )
            )
            continue

        try:
            regions_document = _load_json(regions_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                Failure(
                    name,
                    E_REGION_INVALID,
                    f"invalid regions JSON: {exc}",
                    "load_image",
                    parent_code=E_SAMPLE_INVALID,
                )
            )
            continue
        try:
            truth = _validate_truth(regions_document, image_size=png_size)
        except ValueError as exc:
            code = E_DIMENSION_MISMATCH if "dimensions" in str(exc) or "image_size" in str(exc) else E_REGION_INVALID
            failures.append(
                Failure(name, code, str(exc), "load_image", parent_code=E_SAMPLE_INVALID)
            )
            continue

        try:
            text_document = _load_json(text_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                Failure(
                    name,
                    E_TEXT_SCHEMA,
                    f"invalid text JSON: {exc}",
                    "load_text",
                    parent_code=E_SAMPLE_INVALID,
                )
            )
            continue
        text_document, text_injection = injections.applied_text_document(name, text_document)
        if text_injection is not None:
            if not text_injection.params.get("_not_applied"):
                injections.mark_hit(
                    text_injection,
                    sample=name,
                    stage="load_text",
                    code=E_TEXT_SCHEMA,
                    proof={
                        "field": text_injection.params.get("field", "value"),
                        "assertion_index": text_injection.params.get("assertion_index", 0),
                    },
                )
        try:
            if isinstance(text_document, dict):
                described_name = text_document.get("sample_name")
                if described_name is not None and described_name != name:
                    raise ValueError(
                        f"text.sample_name {described_name!r} does not match sample directory {name!r}"
                    )
            text = _validate_text(text_document)
        except ValueError as exc:
            failures.append(
                Failure(
                    sample=name,
                    code=E_TEXT_SCHEMA,
                    message=str(exc),
                    stage="load_text",
                    parent_code=E_SAMPLE_INVALID,
                    injection_id=(
                        text_injection.id
                        if text_injection is not None
                        and not text_injection.params.get("_not_applied")
                        else None
                    ),
                )
            )
            continue

        entity_ids = {entity.id for entity in text.entities}
        missing_entities = sorted(
            {
                region.entity_id
                for region in truth.regions
                if region.entity_id is not None and region.entity_id not in entity_ids
            }
        )
        if missing_entities:
            failures.append(
                Failure(
                    sample=name,
                    code=E_REGION_INVALID,
                    message="regions reference unknown text entities",
                    stage="index_validation",
                    parent_code=E_SAMPLE_INVALID,
                    details={"entity_ids": missing_entities},
                )
            )
            continue

        samples.append(
            LoadedSample(
                name=name,
                directory=directory,
                image_path=image_path,
                regions_path=regions_path,
                text_path=text_path,
                image_bytes=image_bytes,
                image_size=png_size,
                image_truth=truth,
                text=text,
                image_fingerprint=sha256_bytes(image_bytes),
                regions_fingerprint=hash_json(regions_document),
                text_fingerprint=hash_json(text.to_dict()),
                image_injection_id=image_injection.id if image_injection else None,
                text_injection_id=text_injection.id if text_injection else None,
            )
        )

    samples.sort(key=lambda sample: sample.name)
    failures.sort(key=lambda failure: (failure.sample, failure.code, canonical_json(failure.details)))
    return LoadResult(samples=samples, failures=failures)
