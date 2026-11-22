from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .codec_v1 import V1_PRECISION_LABEL, encode_v1, parse_v1
from .codec_v2 import encode_v2, parse_v2
from .errors import E_STRUCTURE_INVALID, WorkbenchError
from .models import (
    ATTRIBUTE_CODE_TO_NAME,
    ATTRIBUTE_NAME_TO_CODE,
    REPAIR_CODE_TO_NAME,
    REPAIR_NAME_TO_CODE,
    ImageInfo,
    ParseOutcome,
    Rect,
    RegionEdit,
)
from .util import dec, dec_str, round_half_up, sha256_json


SIDECAR_SCHEMA = "image-edit-sidecar/v1"


@dataclass
class ConversionResult:
    text: str
    regions: list[RegionEdit]
    sidecar: dict[str, Any] | None = None

    @property
    def sidecar_json(self) -> str | None:
        if self.sidecar is None:
            return None
        return json.dumps(self.sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @property
    def sidecar_hash(self) -> str:
        return sha256_json(self.sidecar) if self.sidecar is not None else ""


def v1_to_v2_regions(regions: Iterable[RegionEdit]) -> list[RegionEdit]:
    result: list[RegionEdit] = []
    for region in regions:
        if region.exchange_version != 1:
            raise WorkbenchError(E_STRUCTURE_INVALID, "v1->v2 requires exchange_version=1", stage="convert")
        # v1 integers are exactly representable in v2's decimal model.
        result.append(
            RegionEdit(
                exchange_version=2,
                image_name=region.image_name,
                region_index=region.region_index,
                rect=Rect.from_values(region.rect.x, region.rect.y, region.rect.w, region.rect.h),
                style_strength=region.style_strength,
                target_attribute=region.target_attribute,
                repair_method=region.repair_method,
                precision_label=region.precision_label,
                rotation_deg=Decimal(0),
                feather_radius=Decimal(0),
                sidecar_applied=False,
                source_format="v1_upgraded",
                source_fingerprint=region.source_fingerprint,
            )
        )
    return sorted(result, key=lambda region: (region.image_name, region.region_index))


def upgrade_v1(text: str) -> ConversionResult:
    # This convenience entry point validates only the fixed-width structure. Callers that need
    # image/manifest validation should use parse_v1 followed by v1_to_v2_regions.
    regions: list[RegionEdit] = []
    for line in text.splitlines():
        if not line:
            continue
        region = _decode_v1_line_without_context(line)
        regions.append(region)
    upgraded = v1_to_v2_regions(regions)
    return ConversionResult(encode_v2(upgraded), upgraded)


def _decode_v1_line_without_context(line: str) -> RegionEdit:
    from .codec_v1 import FIELD_WIDTHS, LINE_WIDTH

    if len(line) != LINE_WIDTH:
        raise WorkbenchError(E_STRUCTURE_INVALID, "v1 line width mismatch", stage="convert")
    values: dict[str, str] = {}
    offset = 0
    for field in FIELD_WIDTHS:
        values[field.name] = line[offset : offset + field.width]
        offset += field.width
    repair_method = REPAIR_CODE_TO_NAME[int(values["repair_code"])]
    attribute = ATTRIBUTE_CODE_TO_NAME[int(values["attribute_code"])]
    region = RegionEdit(
        exchange_version=1,
        image_name=values["image_name"].strip(),
        region_index=int(values["region_index"]),
        rect=Rect.from_values(int(values["x"]), int(values["y"]), int(values["w"]), int(values["h"])),
        style_strength=Decimal(int(values["style_permille"])) / Decimal(1000),
        target_attribute=attribute,
        repair_method=repair_method,
        precision_label=V1_PRECISION_LABEL,
        source_format="v1_fixed",
    )
    region.validate_semantics()
    return region


def downgrade_v2(regions: Iterable[RegionEdit]) -> ConversionResult:
    ordered = sorted(regions, key=lambda region: (region.image_name, region.region_index))
    v1_regions: list[RegionEdit] = []
    records: list[dict[str, Any]] = []
    for region in ordered:
        if region.exchange_version != 2:
            raise WorkbenchError(E_STRUCTURE_INVALID, "v2->v1 requires exchange_version=2", stage="convert")
        ix, iy, iw, ih = region.rect.integer_half_up()
        style_permille = region.style_permille
        downgraded = RegionEdit(
            exchange_version=1,
            image_name=region.image_name,
            region_index=region.region_index,
            rect=Rect.from_values(ix, iy, iw, ih),
            style_strength=Decimal(style_permille) / Decimal(1000),
            target_attribute=region.target_attribute,
            repair_method=region.repair_method,
            precision_label=V1_PRECISION_LABEL,
            rotation_deg=Decimal(0),
            feather_radius=Decimal(0),
            sidecar_applied=False,
            source_format="v2_downgraded",
            source_fingerprint=region.source_fingerprint,
        )
        v1_regions.append(downgraded)
        records.append(
            {
                "image": region.image_name,
                "region_index": region.region_index,
                "source_rect": region.rect.to_dict(),
                "v1_rect": {"x": ix, "y": iy, "w": iw, "h": ih},
                "coordinate_residual": {
                    "x": dec_str(region.rect.x - Decimal(ix)),
                    "y": dec_str(region.rect.y - Decimal(iy)),
                    "w": dec_str(region.rect.w - Decimal(iw)),
                    "h": dec_str(region.rect.h - Decimal(ih)),
                },
                "source_style_strength": dec_str(region.style_strength),
                "v1_style_permille": style_permille,
                "strength_residual": dec_str(region.style_strength - Decimal(style_permille) / Decimal(1000)),
                "source_precision_label": region.precision_label,
                "rotation_deg": dec_str(region.rotation_deg),
                "feather_radius": dec_str(region.feather_radius),
                "target_attribute": region.target_attribute,
                "repair_method": region.repair_method,
                "source_format": region.source_format,
                "source_fingerprint": region.source_fingerprint,
            }
        )
    text = encode_v1(v1_regions)
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "version": 1,
        "source_exchange_version": 2,
        "v1_document_fingerprint": sha256_json([region.canonical_dict() for region in v1_regions]),
        "source_logic_fingerprint": sha256_json([region.canonical_dict() for region in ordered]),
        "records": records,
        "record_count": len(records),
    }
    return ConversionResult(text=text, regions=v1_regions, sidecar=sidecar)


def restore_v1_with_sidecar(
    v1_regions: Iterable[RegionEdit],
    sidecar: Mapping[str, Any] | str,
) -> ParseOutcome:
    if isinstance(sidecar, str):
        try:
            sidecar_obj = json.loads(sidecar)
        except json.JSONDecodeError as exc:
            raise WorkbenchError(E_STRUCTURE_INVALID, f"invalid sidecar JSON: {exc}", stage="sidecar") from exc
    else:
        sidecar_obj = dict(sidecar)
    if sidecar_obj.get("schema") != SIDECAR_SCHEMA or not isinstance(sidecar_obj.get("records"), list):
        raise WorkbenchError(E_STRUCTURE_INVALID, "invalid sidecar structure", stage="sidecar")

    record_map: dict[str, dict[str, Any]] = {}
    for record in sidecar_obj["records"]:
        if not isinstance(record, dict):
            continue
        try:
            key = f"{record['image']}#{int(record['region_index']):03d}"
        except (KeyError, TypeError, ValueError):
            continue
        record_map[key] = record

    outcome = ParseOutcome(exchange_version=2)
    for region in v1_regions:
        record = record_map.get(region.region_key)
        if record is None or not _sidecar_matches_v1(record, region):
            outcome.sidecar_misses += 1
            outcome.regions.append(v1_to_v2_regions([region])[0])
            continue
        try:
            source_rect = record["source_rect"]
            restored = RegionEdit(
                exchange_version=2,
                image_name=region.image_name,
                region_index=region.region_index,
                rect=Rect.from_values(source_rect["x"], source_rect["y"], source_rect["w"], source_rect["h"]),
                style_strength=dec(record["source_style_strength"]),
                target_attribute=region.target_attribute,
                repair_method=region.repair_method,
                precision_label=str(record["source_precision_label"]),
                rotation_deg=dec(record["rotation_deg"]),
                feather_radius=dec(record["feather_radius"]),
                # Sidecar restoration is reported through ParseOutcome.sidecar_hits;
                # provenance itself is restored so the logical record is field-identical
                # to the pre-downgrade v2 record.
                sidecar_applied=False,
                source_format=str(record.get("source_format", "v1_sidecar_restored")),
                source_fingerprint=str(record.get("source_fingerprint", region.source_fingerprint)),
            )
            restored.validate_semantics()
        except (KeyError, TypeError, ValueError) as exc:
            outcome.sidecar_misses += 1
            outcome.regions.append(v1_to_v2_regions([region])[0])
            continue
        outcome.sidecar_hits += 1
        outcome.regions.append(restored)
    outcome.regions.sort(key=lambda region: (region.image_name, region.region_index))
    return outcome


def _sidecar_matches_v1(record: Mapping[str, Any], region: RegionEdit) -> bool:
    v1_rect = record.get("v1_rect")
    if not isinstance(v1_rect, dict):
        return False
    ix, iy, iw, ih = region.rect.integer_half_up()
    expected_rect = {"x": ix, "y": iy, "w": iw, "h": ih}
    try:
        expected_style = int(record["v1_style_permille"])
    except (KeyError, TypeError, ValueError):
        return False
    return v1_rect == expected_rect and expected_style == region.style_permille


def upgrade_v1_with_sidecar(
    text: str,
    sidecar: Mapping[str, Any] | str,
    *,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str] | None = None,
) -> tuple[ParseOutcome, str]:
    parsed = parse_v1(text, images=images, manifest_keys=manifest_keys)
    restored = restore_v1_with_sidecar(parsed.regions, sidecar)
    restored.failures = list(parsed.failures) + list(restored.failures)
    restored.document_fingerprint = parsed.document_fingerprint
    restored.exchange_version = 2
    return restored, encode_v2(restored.regions)
