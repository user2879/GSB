from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from .errors import (
    BusinessFailure,
    E_REGION_OUT_OF_RANGE,
    E_STRUCTURE_INVALID,
    WorkbenchError,
)
from .models import (
    REPAIR_METHODS,
    TARGET_ATTRIBUTES,
    ImageInfo,
    ParseOutcome,
    Rect,
    RegionEdit,
)
from .util import dec, dec_str, fingerprint, sha256_bytes, sha256_json


V2_SCHEMA = "image-edit-exchange/v2"
REQUIRED_REGION_FIELDS = {
    "image",
    "region_index",
    "rect",
    "style_strength",
    "target_attribute",
    "repair_method",
    "precision_label",
    "rotation_deg",
    "feather_radius",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def encode_v2(regions: Iterable[RegionEdit]) -> str:
    ordered = sorted(regions, key=lambda region: (region.image_name, region.region_index))
    payload = {
        "schema": V2_SCHEMA,
        "version": 2,
        "precision": "decimal-source/float-number-serialization",
        "regions": [
            {
                "image": region.image_name,
                "region_index": region.region_index,
                "rect": {
                    "x": float(region.rect.x),
                    "y": float(region.rect.y),
                    "w": float(region.rect.w),
                    "h": float(region.rect.h),
                },
                "style_strength": float(region.style_strength),
                "target_attribute": region.target_attribute,
                "repair_method": region.repair_method,
                "precision_label": region.precision_label,
                "rotation_deg": float(region.rotation_deg),
                "feather_radius": float(region.feather_radius),
            }
            for region in ordered
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _region_from_v2_record(
    record: dict[str, Any],
    *,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str] | None,
    source: str,
) -> tuple[RegionEdit | None, BusinessFailure | None]:
    missing = sorted(REQUIRED_REGION_FIELDS - set(record))
    if missing:
        return None, BusinessFailure(
            E_STRUCTURE_INVALID,
            "parse_v2",
            source,
            f"region missing fields: {', '.join(missing)}",
            None,
        )

    image_name = record.get("image")
    if not isinstance(image_name, str) or not image_name:
        return None, BusinessFailure(E_STRUCTURE_INVALID, "parse_v2", source, "image must be a non-empty string")
    rect_value = record.get("rect")
    if not isinstance(rect_value, dict) or not {"x", "y", "w", "h"}.issubset(rect_value):
        return None, BusinessFailure(
            E_STRUCTURE_INVALID,
            "parse_v2",
            source,
            "rect must contain x, y, w and h",
            None,
        )
    try:
        region_index = int(record["region_index"])
        if isinstance(record["region_index"], bool):
            raise ValueError
        rect = Rect.from_values(rect_value["x"], rect_value["y"], rect_value["w"], rect_value["h"])
        style_strength = dec(record["style_strength"])
        rotation_deg = dec(record["rotation_deg"])
        feather_radius = dec(record["feather_radius"])
        precision_label = str(record["precision_label"])
    except (InvalidOperation, ValueError, TypeError) as exc:
        return None, BusinessFailure(
            E_STRUCTURE_INVALID,
            "parse_v2",
            source,
            f"invalid numeric/string field: {exc}",
        )

    region_key = f"{image_name}#{region_index:03d}"
    image = images.get(image_name)
    if image is None or (manifest_keys is not None and region_key not in manifest_keys):
        return None, BusinessFailure(
            E_REGION_OUT_OF_RANGE,
            "parse_v2",
            source,
            f"image/region not present in manifest: {region_key}",
            region_key,
        )
    bounds_error = rect.violates(image.width, image.height)
    if bounds_error:
        return None, BusinessFailure(
            E_REGION_OUT_OF_RANGE,
            "parse_v2",
            source,
            f"{region_key}: {bounds_error}",
            region_key,
        )

    target_attribute = record.get("target_attribute")
    repair_method = record.get("repair_method")
    if target_attribute not in TARGET_ATTRIBUTES:
        return None, BusinessFailure(
            E_STRUCTURE_INVALID,
            "parse_v2",
            source,
            f"unknown target_attribute: {target_attribute!r}",
            region_key,
        )
    if repair_method not in REPAIR_METHODS:
        return None, BusinessFailure(
            E_STRUCTURE_INVALID,
            "parse_v2",
            source,
            f"unknown repair_method: {repair_method!r}",
            region_key,
        )

    region = RegionEdit(
        exchange_version=2,
        image_name=image_name,
        region_index=region_index,
        rect=rect,
        style_strength=style_strength,
        target_attribute=target_attribute,
        repair_method=repair_method,
        precision_label=precision_label,
        rotation_deg=rotation_deg,
        feather_radius=feather_radius,
        sidecar_applied=False,
        source_format="v2_json",
        source_fingerprint=sha256_json(record),
    )
    try:
        region.validate_semantics()
    except WorkbenchError as exc:
        return None, BusinessFailure(exc.code, exc.stage, source, exc.message, region_key)
    return region, None


def parse_v2(
    text: str | bytes,
    *,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str] | None = None,
    source: str = "v2",
) -> ParseOutcome:
    raw_bytes = text.encode("utf-8") if isinstance(text, str) else text
    outcome = ParseOutcome(exchange_version=2)
    outcome.document_fingerprint = sha256_bytes(raw_bytes)
    try:
        decoded = raw_bytes.decode("utf-8")
        document = json.loads(decoded, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        outcome.failures.append(BusinessFailure(E_STRUCTURE_INVALID, "parse_v2", source, f"invalid JSON: {exc}"))
        return outcome

    if isinstance(document, list):
        version = 2
        records = document
    elif isinstance(document, dict):
        version = document.get("version")
        schema = document.get("schema")
        if version != 2 or (schema is not None and schema != V2_SCHEMA):
            outcome.failures.append(
                BusinessFailure(E_STRUCTURE_INVALID, "parse_v2", source, "expected version 2 and schema image-edit-exchange/v2")
            )
            return outcome
        if not isinstance(document.get("regions"), list):
            outcome.failures.append(
                BusinessFailure(E_STRUCTURE_INVALID, "parse_v2", source, "document.regions must be an array")
            )
            return outcome
        records = document["regions"]
    else:
        outcome.failures.append(
            BusinessFailure(E_STRUCTURE_INVALID, "parse_v2", source, "document root must be an object or array")
        )
        return outcome

    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            outcome.failures.append(
                BusinessFailure(E_STRUCTURE_INVALID, "parse_v2", source, f"region[{index}] must be an object")
            )
            continue
        region, failure = _region_from_v2_record(
            record,
            images=images,
            manifest_keys=manifest_keys,
            source=source,
        )
        if failure is not None:
            outcome.failures.append(failure)
            continue
        assert region is not None
        if region.region_key in seen:
            outcome.failures.append(
                BusinessFailure(
                    E_STRUCTURE_INVALID,
                    "parse_v2",
                    source,
                    f"duplicate region key {region.region_key}",
                    region.region_key,
                )
            )
            continue
        seen.add(region.region_key)
        outcome.regions.append(region)
    outcome.regions.sort(key=lambda region: (region.image_name, region.region_index))
    return outcome

