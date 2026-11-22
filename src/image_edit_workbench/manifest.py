from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .errors import BusinessFailure, E_REGION_OUT_OF_RANGE, E_STRUCTURE_INVALID, WorkbenchError
from .models import (
    REPAIR_METHODS,
    TARGET_ATTRIBUTES,
    ImageInfo,
    ParseOutcome,
    Rect,
    RegionEdit,
)
from .util import dec, read_json, sha256_json


def load_edit_manifest(
    path: Path,
    *,
    images: Mapping[str, ImageInfo],
) -> ParseOutcome:
    try:
        document = read_json(path)
    except (OSError, ValueError) as exc:
        outcome = ParseOutcome(exchange_version=2)
        outcome.failures.append(BusinessFailure(E_STRUCTURE_INVALID, "manifest", str(path), f"cannot read manifest: {exc}"))
        return outcome
    if not isinstance(document, dict) or not isinstance(document.get("edits"), list):
        outcome = ParseOutcome(exchange_version=2)
        outcome.failures.append(
            BusinessFailure(E_STRUCTURE_INVALID, "manifest", str(path), "manifest must contain an edits array")
        )
        return outcome

    outcome = ParseOutcome(exchange_version=2)
    outcome.document_fingerprint = sha256_json(document)
    seen: set[str] = set()
    for index, record in enumerate(document["edits"]):
        if not isinstance(record, dict):
            outcome.failures.append(
                BusinessFailure(E_STRUCTURE_INVALID, "manifest", str(path), f"edits[{index}] must be an object")
            )
            continue
        try:
            image_name = str(record["image"])
            region_index = int(record["region_index"])
            rect_value = record["rect"]
            rect = Rect.from_values(rect_value["x"], rect_value["y"], rect_value["w"], rect_value["h"])
            style = dec(record["style_strength"])
            attribute = str(record["target_attribute"])
            repair = str(record["repair_method"])
            rotation = dec(record.get("rotation_deg", 0))
            feather = dec(record.get("feather_radius", 0))
        except (KeyError, TypeError, ValueError) as exc:
            outcome.failures.append(
                BusinessFailure(E_STRUCTURE_INVALID, "manifest", str(path), f"invalid edit[{index}]: {exc}")
            )
            continue

        region_key = f"{image_name}#{region_index:03d}"
        if region_key in seen:
            outcome.failures.append(
                BusinessFailure(E_STRUCTURE_INVALID, "manifest", str(path), f"duplicate region {region_key}", region_key)
            )
            continue
        seen.add(region_key)
        image = images.get(image_name)
        if image is None:
            outcome.failures.append(
                BusinessFailure(
                    E_REGION_OUT_OF_RANGE,
                    "manifest",
                    str(path),
                    f"unknown image {image_name!r}",
                    region_key,
                )
            )
            continue
        bounds_error = rect.violates(image.width, image.height)
        if bounds_error:
            outcome.failures.append(
                BusinessFailure(
                    E_REGION_OUT_OF_RANGE,
                    "manifest",
                    str(path),
                    f"{region_key}: {bounds_error}",
                    region_key,
                )
            )
            continue
        if attribute not in TARGET_ATTRIBUTES or repair not in REPAIR_METHODS:
            outcome.failures.append(
                BusinessFailure(
                    E_STRUCTURE_INVALID,
                    "manifest",
                    str(path),
                    f"{region_key}: unknown attribute or repair method",
                    region_key,
                )
            )
            continue
        region = RegionEdit(
            exchange_version=2,
            image_name=image_name,
            region_index=region_index,
            rect=rect,
            style_strength=style,
            target_attribute=attribute,
            repair_method=repair,
            precision_label="manifest:logical",
            rotation_deg=rotation,
            feather_radius=feather,
            source_format="manifest",
        )
        try:
            region.validate_semantics()
        except WorkbenchError as exc:
            outcome.failures.append(BusinessFailure(exc.code, "manifest", str(path), exc.message, region_key))
            continue
        outcome.regions.append(region)
    outcome.regions.sort(key=lambda region: (region.image_name, region.region_index))
    return outcome

