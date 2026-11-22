from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping

from .errors import (
    BusinessFailure,
    E_FIELD_TRUNCATED,
    E_REGION_OUT_OF_RANGE,
    E_STRUCTURE_INVALID,
    WorkbenchError,
)
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
from .util import fingerprint, sha256_bytes


@dataclass(frozen=True)
class FieldSpec:
    name: str
    width: int


FIELD_WIDTHS: tuple[FieldSpec, ...] = (
    FieldSpec("image_name", 18),
    FieldSpec("region_index", 3),
    FieldSpec("x", 5),
    FieldSpec("y", 5),
    FieldSpec("w", 5),
    FieldSpec("h", 5),
    FieldSpec("style_permille", 5),
    FieldSpec("repair_code", 2),
    FieldSpec("attribute_code", 2),
    FieldSpec("padding", 12),
)
LINE_WIDTH = sum(field.width for field in FIELD_WIDTHS)
V1_PRECISION_LABEL = "v1:integer-permille-exact"


def _format_fixed(value: int, width: int) -> str:
    text = str(value)
    if len(text) > width:
        raise WorkbenchError(
            E_STRUCTURE_INVALID,
            f"v1 numeric value {value!r} does not fit field width {width}",
            stage="v2_to_v1",
        )
    return text.rjust(width, "0")


def encode_region_line(region: RegionEdit) -> str:
    if region.exchange_version != 1:
        raise WorkbenchError(E_STRUCTURE_INVALID, "v1 encoder requires exchange_version=1", stage="v2_to_v1")
    x, y, w, h = region.rect.integer_half_up()
    image = region.image_name
    if len(image.encode("utf-8")) > 18:
        raise WorkbenchError(E_STRUCTURE_INVALID, f"image name too long for v1: {image}", stage="v2_to_v1")
    repair_code = REPAIR_NAME_TO_CODE.get(region.repair_method)
    attribute_code = ATTRIBUTE_NAME_TO_CODE.get(region.target_attribute)
    if repair_code is None or attribute_code is None:
        raise WorkbenchError(E_STRUCTURE_INVALID, "unknown named attribute or repair method", stage="v2_to_v1")
    pieces = [
        image.ljust(18),
        _format_fixed(region.region_index, 3),
        _format_fixed(x, 5),
        _format_fixed(y, 5),
        _format_fixed(w, 5),
        _format_fixed(h, 5),
        _format_fixed(region.style_permille, 5),
        _format_fixed(repair_code, 2),
        _format_fixed(attribute_code, 2),
        " " * 12,
    ]
    line = "".join(pieces)
    assert len(line) == LINE_WIDTH
    return line


def encode_v1(regions: Iterable[RegionEdit]) -> str:
    ordered = sorted(regions, key=lambda region: (region.image_name, region.region_index))
    return "\n".join(encode_region_line(region) for region in ordered) + ("\n" if ordered else "")


def _structure_failure(source: str, message: str, region_key: str | None = None) -> BusinessFailure:
    return BusinessFailure(E_STRUCTURE_INVALID, "parse_v1", source, message, region_key)


def parse_v1(
    text: str | bytes,
    *,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str] | None = None,
    source: str = "v1",
) -> ParseOutcome:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            outcome = ParseOutcome(exchange_version=1)
            outcome.failures.append(_structure_failure(source, f"invalid UTF-8: {exc}"))
            outcome.document_fingerprint = sha256_bytes(text)
            return outcome

    outcome = ParseOutcome(exchange_version=1)
    outcome.document_fingerprint = sha256_bytes(text.encode("utf-8"))
    seen: set[str] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if line == "":
            continue
        if len(line) < LINE_WIDTH:
            outcome.failures.append(
                BusinessFailure(
                    E_FIELD_TRUNCATED,
                    "parse_v1",
                    source,
                    f"line {line_number} has {len(line)} chars, expected {LINE_WIDTH}",
                    None,
                )
            )
            continue
        if len(line) > LINE_WIDTH:
            outcome.failures.append(
                _structure_failure(source, f"line {line_number} exceeds fixed width {LINE_WIDTH}")
            )
            continue

        values: dict[str, str] = {}
        offset = 0
        for field in FIELD_WIDTHS:
            values[field.name] = line[offset : offset + field.width]
            offset += field.width

        image_name = values["image_name"].strip()
        if not image_name:
            outcome.failures.append(_structure_failure(source, f"line {line_number}: empty image name"))
            continue
        if values["padding"] != " " * len(values["padding"]):
            outcome.failures.append(
                _structure_failure(source, f"line {line_number}: padding contains non-space bytes")
            )
            continue

        try:
            region_index = int(values["region_index"])
            ix = int(values["x"])
            iy = int(values["y"])
            iw = int(values["w"])
            ih = int(values["h"])
            style_permille = int(values["style_permille"])
            repair_code = int(values["repair_code"])
            attribute_code = int(values["attribute_code"])
        except ValueError as exc:
            outcome.failures.append(_structure_failure(source, f"line {line_number}: invalid integer field ({exc})"))
            continue

        if not 0 <= style_permille <= 1000:
            outcome.failures.append(
                _structure_failure(source, f"line {line_number}: style permille outside [0, 1000]")
            )
            continue
        repair_method = REPAIR_CODE_TO_NAME.get(repair_code)
        target_attribute = ATTRIBUTE_CODE_TO_NAME.get(attribute_code)
        if repair_method is None or target_attribute is None:
            outcome.failures.append(
                _structure_failure(source, f"line {line_number}: unknown repair/attribute code")
            )
            continue

        region_key = f"{image_name}#{region_index:03d}"
        if region_key in seen:
            outcome.failures.append(_structure_failure(source, f"duplicate region key {region_key}", region_key))
            continue
        seen.add(region_key)

        image = images.get(image_name)
        if image is None or (manifest_keys is not None and region_key not in manifest_keys):
            outcome.failures.append(
                BusinessFailure(
                    E_REGION_OUT_OF_RANGE,
                    "parse_v1",
                    source,
                    f"image/region not present in manifest: {region_key}",
                    region_key,
                )
            )
            continue

        rect = Rect.from_values(ix, iy, iw, ih)
        bounds_error = rect.violates(image.width, image.height)
        if bounds_error:
            outcome.failures.append(
                BusinessFailure(
                    E_REGION_OUT_OF_RANGE,
                    "parse_v1",
                    source,
                    f"{region_key}: {bounds_error}",
                    region_key,
                )
            )
            continue

        region = RegionEdit(
            exchange_version=1,
            image_name=image_name,
            region_index=region_index,
            rect=rect,
            style_strength=Decimal(style_permille) / Decimal(1000),
            target_attribute=target_attribute,
            repair_method=repair_method,
            precision_label=V1_PRECISION_LABEL,
            rotation_deg=Decimal(0),
            feather_radius=Decimal(0),
            sidecar_applied=False,
            source_format="v1_fixed",
            source_fingerprint=fingerprint("v1-line", line_number, line),
        )
        region.validate_semantics()
        outcome.regions.append(region)

    outcome.regions.sort(key=lambda region: (region.image_name, region.region_index))
    return outcome

