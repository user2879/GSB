"""v1/v2 calibration parsing, validation, conversion, and lossy sidecars."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

from .codes import (
    E_CALIB_PARSE,
    E_SIDECAR_INVALID,
    E_SIDECAR_MISSING,
)
from .io_utils import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from .models import CalibrationRecord, FailureRecord

V1_LAYER_WIDTH = 20
V1_SCALE_WIDTH = 8
V1_ZERO_WIDTH = 8
V1_BITS_WIDTH = 4
V1_LINE_WIDTH = V1_LAYER_WIDTH + V1_SCALE_WIDTH + V1_ZERO_WIDTH + V1_BITS_WIDTH
V1_HEADER = (
    f"{'layer':<{V1_LAYER_WIDTH}}{'scale_1e3':>{V1_SCALE_WIDTH}}"
    f"{'zero':>{V1_ZERO_WIDTH}}{'bits':>{V1_BITS_WIDTH}}"
)

SIDECAR_SCHEMA = "qcal-sidecar/1.0"


def signed_range(bit_width: int) -> tuple[int, int]:
    if bit_width <= 0:
        raise ValueError("bit width must be positive")
    return -(1 << (bit_width - 1)), (1 << (bit_width - 1)) - 1


@dataclass
class ParsedCalibration:
    version: str
    file_path: str
    file_sha256: str
    records: dict[str, CalibrationRecord]
    raw_by_layer: dict[str, dict[str, Any]]
    failures: list[FailureRecord]

    @property
    def content_fingerprint(self) -> str:
        return sha256_json(
            {
                "version": self.version,
                "file_sha256": self.file_sha256,
                "records": {key: value.to_dict() for key, value in sorted(self.records.items())},
            }
        )


def _injection_for(injections: Iterable[dict[str, Any]], injection_type: str, version: str, layer: str | None = None) -> dict[str, Any] | None:
    for item in injections:
        if item.get("type") != injection_type:
            continue
        if item.get("version") not in (None, version):
            continue
        if layer is not None and item.get("layer") not in (None, layer):
            continue
        return item
    return None


def _failure(layer: str, message: str, injection_id: str | None = None, detail: dict[str, Any] | None = None) -> FailureRecord:
    return FailureRecord(
        layer=layer or "<calibration>",
        stage="parse_validation",
        code=E_CALIB_PARSE,
        message=message,
        injection_id=injection_id,
        detail=detail or {},
    )


def parse_v1_text(text: str, injections: Iterable[dict[str, Any]] = ()) -> tuple[dict[str, dict[str, Any]], list[FailureRecord]]:
    """Parse fixed-width v1 rows into raw records.

    The layer field is left-padded/space-padded to 20 characters. The three
    numeric fields are right-aligned. Blank lines, comments, and the exact
    conventional header are ignored.
    """

    injections = list(injections)
    raw: dict[str, dict[str, Any]] = {}
    failures: list[FailureRecord] = []
    for line_number, original in enumerate(text.splitlines(), 1):
        line = original.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("layer") and "scale_1e3" in line:
            continue
        if len(line) < V1_LINE_WIDTH:
            layer_guess = line[:V1_LAYER_WIDTH].strip() or f"<v1-line-{line_number}>"
            failures.append(
                _failure(
                    layer_guess,
                    f"v1 line {line_number} is shorter than {V1_LINE_WIDTH} columns",
                    detail={"line": line_number, "length": len(line)},
                )
            )
            continue
        layer = line[0:V1_LAYER_WIDTH].strip()
        scale_token = line[V1_LAYER_WIDTH : V1_LAYER_WIDTH + V1_SCALE_WIDTH].strip()
        zero_token = line[
            V1_LAYER_WIDTH + V1_SCALE_WIDTH : V1_LAYER_WIDTH + V1_SCALE_WIDTH + V1_ZERO_WIDTH
        ].strip()
        bits_token = line[V1_LINE_WIDTH - V1_BITS_WIDTH : V1_LINE_WIDTH].strip()
        line_failure = None
        try:
            scale_thousandths = int(scale_token, 10)
            zero_point = int(zero_token, 10)
            bit_width = int(bits_token, 10)
        except ValueError:
            line_failure = _failure(
                layer or f"<v1-line-{line_number}>",
                f"v1 line {line_number} contains a non-integer fixed-width field",
                detail={
                    "line": line_number,
                    "scale_token": scale_token,
                    "zero_token": zero_token,
                    "bits_token": bits_token,
                },
            )
            failures.append(line_failure)
            continue

        zero_injection = _injection_for(injections, "zero_scale", "v1", layer)
        if zero_injection:
            scale_thousandths = 0
        zero_point_injection = _injection_for(injections, "zero_point_out_of_range", "v1", layer)
        if zero_point_injection:
            zero_point = int(zero_point_injection.get("value", 10**9))

        if layer in raw:
            failures.append(_failure(layer, f"duplicate v1 layer name at line {line_number}"))
            continue
        raw[layer] = {
            "layer": layer,
            "scale_thousandths": scale_thousandths,
            "scale": float(scale_thousandths) / 1000.0,
            "zero_point": zero_point,
            "bit_width": bit_width,
            "scheme": "affine",
            "rounding_mode": "nearest_even",
            "precision_label": "v1-thousandth-implicit",
            "per_channel": False,
            "source_version": "v1",
            "provenance": {
                "line": line_number,
                "fixed_widths": [V1_LAYER_WIDTH, V1_SCALE_WIDTH, V1_ZERO_WIDTH, V1_BITS_WIDTH],
                "scale_semantics": "integer_thousandths",
            },
        }
    return raw, failures


def parse_v2_json(data: Any, injections: Iterable[dict[str, Any]] = ()) -> tuple[dict[str, dict[str, Any]], list[FailureRecord]]:
    injections = list(injections)
    failures: list[FailureRecord] = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("layers", data.get("calibrations"))
        if rows is None:
            return {}, [_failure("<v2>", "v2 JSON must contain a layers or calibrations array")]
    else:
        return {}, [_failure("<v2>", "v2 JSON root must be an object or array")]
    if not isinstance(rows, list):
        return {}, [_failure("<v2>", "v2 layers value must be an array")]

    raw: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(_failure(f"<v2-row-{index}>", "v2 layer entry must be an object"))
            continue
        layer = str(row.get("layer", row.get("name", ""))).strip()
        if not layer:
            failures.append(_failure(f"<v2-row-{index}>", "v2 layer name is missing"))
            continue
        try:
            scale = float(row["scale"])
            zero_point = int(row["zero_point"])
            bit_width = int(row.get("bit_width", row.get("bits")))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(_failure(layer, f"v2 required numeric field is invalid: {exc}"))
            continue

        if _injection_for(injections, "zero_scale", "v2", layer):
            scale = 0.0
        zero_point_injection = _injection_for(injections, "zero_point_out_of_range", "v2", layer)
        if zero_point_injection:
            zero_point = int(zero_point_injection.get("value", 10**9))

        if layer in raw:
            failures.append(_failure(layer, "duplicate v2 layer name"))
            continue
        raw[layer] = {
            "layer": layer,
            "scale": scale,
            "zero_point": zero_point,
            "bit_width": bit_width,
            "scheme": str(row.get("scheme", row.get("quant_scheme", "affine"))),
            "rounding_mode": str(row.get("rounding_mode", row.get("rounding", "nearest_even"))),
            "precision_label": str(row.get("precision_label", row.get("precision", "unspecified"))),
            "per_channel": bool(row.get("per_channel", False)),
            "source_version": "v2",
            "provenance": {"row": index, "scale_semantics": "binary_float"},
        }
    return raw, failures


def _validate_raw(
    raw: dict[str, dict[str, Any]],
    version: str,
    weight_layers: set[str],
) -> tuple[dict[str, CalibrationRecord], list[FailureRecord]]:
    records: dict[str, CalibrationRecord] = {}
    failures: list[FailureRecord] = []
    for layer in sorted(raw):
        row = raw[layer]
        messages: list[str] = []
        try:
            bit_width = int(row["bit_width"])
        except (TypeError, ValueError):
            bit_width = 0
            messages.append("bit width is not an integer")
        if bit_width <= 0 or bit_width > 32:
            messages.append(f"bit width {bit_width} is outside [1, 32]")

        scale = row.get("scale")
        try:
            scale_float = float(scale)
        except (TypeError, ValueError):
            scale_float = float("nan")
        if not math.isfinite(scale_float) or scale_float <= 0.0:
            messages.append("scale must be a positive finite number")

        try:
            zero_point = int(row["zero_point"])
        except (TypeError, ValueError):
            zero_point = 0
            messages.append("zero point is not an integer")
        if bit_width > 0:
            qmin, qmax = signed_range(bit_width)
            if zero_point < qmin or zero_point > qmax:
                messages.append(f"zero point {zero_point} outside [{qmin}, {qmax}]")

        if layer not in weight_layers:
            messages.append("layer name is absent from the weight package")

        if messages:
            failures.append(_failure(layer, "; ".join(messages), detail={"version": version}))
            continue
        records[layer] = CalibrationRecord(
            layer=layer,
            scale=scale_float,
            zero_point=zero_point,
            bit_width=bit_width,
            scheme=str(row.get("scheme", "affine")),
            rounding_mode=str(row.get("rounding_mode", "nearest_even")),
            precision_label=str(row.get("precision_label", "unspecified")),
            per_channel=bool(row.get("per_channel", False)),
            source_version=version,
            provenance=dict(row.get("provenance", {})),
        )

    for missing in sorted(weight_layers - set(raw)):
        failures.append(
            _failure(missing, "weight package layer has no calibration record", detail={"version": version})
        )
    return records, failures


def parse_calibration_file(
    path: str | Path,
    version: str,
    weight_layers: set[str],
    injections: Iterable[dict[str, Any]] = (),
) -> ParsedCalibration:
    path = Path(path)
    file_hash = sha256_file(path) if path.exists() else "<missing>"
    failures: list[FailureRecord] = []
    raw: dict[str, dict[str, Any]] = {}
    if not path.exists():
        failures.append(_failure(f"<{version}>", f"calibration file does not exist: {path}"))
    elif version == "v1":
        raw, failures = parse_v1_text(path.read_text(encoding="utf-8"), injections)
    elif version == "v2":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            failures = [_failure("<v2>", f"invalid v2 JSON: {exc}")]
        else:
            declared_version = data.get("exchange_version") if isinstance(data, dict) else None
            if declared_version is not None and str(declared_version).lower() != "v2":
                version_failures = [
                    _failure(
                        "<v2>",
                        f"v2 JSON declares unsupported exchange_version {declared_version!r}",
                        detail={"declared_version": declared_version},
                    )
                ]
                raw, parser_failures = parse_v2_json([], injections)
                failures = version_failures + parser_failures
            else:
                raw, failures = parse_v2_json(data, injections)
    else:
        failures = [_failure(f"<{version}>", f"unsupported calibration version: {version}")]

    records, validation_failures = _validate_raw(raw, version, weight_layers)
    failures.extend(validation_failures)
    for failure in failures:
        if failure.injection_id:
            continue
        for injection in injections:
            if injection.get("type") not in ("zero_scale", "zero_point_out_of_range"):
                continue
            if injection.get("version") not in (None, version):
                continue
            if injection.get("layer") not in (None, failure.layer):
                continue
            failure.injection_id = str(injection.get("id", "INJ-UNKNOWN"))
            break
    failures.sort(key=failure_sort_key)
    return ParsedCalibration(
        version=version,
        file_path=str(path),
        file_sha256=file_hash,
        records=records,
        raw_by_layer=raw,
        failures=failures,
    )


def failure_sort_key(failure: FailureRecord) -> tuple[str, str, str, str]:
    return (failure.layer, failure.stage, failure.code, failure.message)


def render_v1_record(record: CalibrationRecord, scale_thousandths: int | None = None) -> str:
    if scale_thousandths is None:
        scale_thousandths = _half_up_thousandths(record.scale)
    layer = record.layer[:V1_LAYER_WIDTH]
    return (
        f"{layer:<{V1_LAYER_WIDTH}}"
        f"{scale_thousandths:>{V1_SCALE_WIDTH}d}"
        f"{record.zero_point:>{V1_ZERO_WIDTH}d}"
        f"{record.bit_width:>{V1_BITS_WIDTH}d}"
    )


def render_v1(records: dict[str, CalibrationRecord], scale_thousandths: dict[str, int] | None = None) -> str:
    scale_thousandths = scale_thousandths or {}
    lines = [V1_HEADER]
    for layer in sorted(records):
        lines.append(render_v1_record(records[layer], scale_thousandths.get(layer)))
    return "\n".join(lines) + "\n"


def upgrade_v1_to_v2(v1_records: dict[str, CalibrationRecord]) -> dict[str, CalibrationRecord]:
    result: dict[str, CalibrationRecord] = {}
    for layer, record in sorted(v1_records.items()):
        thousandths = record.provenance.get("scale_thousandths")
        if thousandths is None:
            # v1 records loaded from a renderer still have their provenance. This
            # branch provides a deterministic fallback for programmatic records.
            thousandths = _half_up_thousandths(record.scale)
        result[layer] = CalibrationRecord(
            layer=layer,
            scale=float(int(thousandths)) / 1000.0,
            zero_point=record.zero_point,
            bit_width=record.bit_width,
            scheme="affine",
            rounding_mode="nearest_even",
            precision_label="v1-thousandth-exact",
            per_channel=False,
            source_version="v2",
            provenance={
                "upgraded_from": "v1",
                "source_scale_thousandths": int(thousandths),
                "scale_conversion": "thousandths / 1000.0",
            },
        )
    return result


def _half_up_thousandths(scale: float) -> int:
    raw = (Decimal(str(float(scale))) * Decimal(1000)).to_integral_value(rounding=ROUND_HALF_UP)
    return int(raw)


def downgrade_v2_to_v1(
    v2_records: dict[str, CalibrationRecord],
) -> tuple[dict[str, CalibrationRecord], dict[str, Any], dict[str, int]]:
    v1_records: dict[str, CalibrationRecord] = {}
    sidecar_layers: dict[str, Any] = {}
    thousandths_by_layer: dict[str, int] = {}
    for layer, record in sorted(v2_records.items()):
        rounded_raw = _half_up_thousandths(record.scale)
        clamped = rounded_raw < 1
        thousandths = max(1, rounded_raw)
        restored = float(thousandths) / 1000.0
        residual = float(record.scale) - restored
        thousandths_by_layer[layer] = thousandths
        v1_records[layer] = CalibrationRecord(
            layer=layer,
            scale=restored,
            zero_point=record.zero_point,
            bit_width=record.bit_width,
            scheme="affine",
            rounding_mode="nearest_even",
            precision_label="v1-thousandth-rounded",
            per_channel=False,
            source_version="v1",
            provenance={
                "downgraded_from": "v2",
                "sidecar_required_for_lossless_reverse": True,
                "source_scale_thousandths": thousandths,
            },
        )
        sidecar_layers[layer] = {
            "layer": layer,
            "source_version": "v2",
            "source_scale": float(record.scale),
            "source_scale_hex": float(record.scale).hex(),
            "rounded_thousandths_raw": rounded_raw,
            "thousandths": thousandths,
            "restored_scale": restored,
            "truncation_residual": residual,
            "clamped_to_minimum": clamped,
            "precision_label": record.precision_label,
            "rounding_mode": record.rounding_mode,
            "per_channel": record.per_channel,
            "scheme": record.scheme,
            "zero_point": record.zero_point,
            "bit_width": record.bit_width,
            "source_provenance": dict(record.provenance),
            "recoverable_fields": [
                "scale",
                "precision_label",
                "rounding_mode",
                "per_channel",
                "scheme",
            ],
            "unrecoverable_fields": [],
            "recovery_rule": "restore source_scale_hex and preserved v2-only fields",
        }
    sidecar = {
        "schema_version": SIDECAR_SCHEMA,
        "source_version": "v2",
        "target_version": "v1",
        "scale_semantics": {
            "v1": "integer_thousandths/1000",
            "v2": "binary_float",
            "conversion": "Decimal ROUND_HALF_UP at 0.001",
        },
        "layers": sidecar_layers,
    }
    sidecar["sidecar_digest"] = sha256_json({k: v for k, v in sidecar.items() if k != "sidecar_digest"})
    return v1_records, sidecar, thousandths_by_layer


def write_downgrade_artifacts(
    directory: str | Path,
    v1_records: dict[str, CalibrationRecord],
    sidecar: dict[str, Any],
    thousandths_by_layer: dict[str, int],
) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / "calibration_v2_to_v1.txt"
    sidecar_path = directory / "calibration_v2_to_v1.sidecar.json"
    atomic_write_text(text_path, render_v1(v1_records, thousandths_by_layer))
    atomic_write_json(sidecar_path, sidecar)
    return text_path, sidecar_path


def reverse_downgrade(
    v1_records: dict[str, CalibrationRecord],
    sidecar_path: str | Path,
) -> tuple[dict[str, CalibrationRecord], list[FailureRecord], str]:
    path = Path(sidecar_path)
    if not path.exists():
        failures = [
            FailureRecord(
                layer=layer,
                stage="conversion",
                code=E_SIDECAR_MISSING,
                message="downgrade sidecar is missing; v2-only information cannot be restored",
            )
            for layer in sorted(v1_records)
        ]
        return {}, failures, "sidecar_missing"
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        sidecar_layers = sidecar["layers"]
        if not isinstance(sidecar_layers, dict):
            raise ValueError("layers is not an object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        failures = [
            FailureRecord(
                layer=layer,
                stage="conversion",
                code=E_SIDECAR_INVALID,
                message=f"downgrade sidecar is invalid: {exc}",
            )
            for layer in sorted(v1_records)
        ]
        return {}, failures, "sidecar_invalid"

    restored: dict[str, CalibrationRecord] = {}
    failures: list[FailureRecord] = []
    for layer, v1_record in sorted(v1_records.items()):
        item = sidecar_layers.get(layer)
        if not isinstance(item, dict):
            failures.append(
                FailureRecord(
                    layer=layer,
                    stage="conversion",
                    code=E_SIDECAR_MISSING,
                    message="sidecar has no entry for this layer",
                )
            )
            continue
        unrecoverable = list(item.get("unrecoverable_fields", []))
        required = {"source_scale_hex", "precision_label", "rounding_mode", "per_channel", "scheme"}
        missing_fields = sorted(field for field in required if field not in item or field in unrecoverable)
        if "source_scale_hex" in item and "source_scale_hex" not in missing_fields:
            scale = float.fromhex(str(item["source_scale_hex"]))
        else:
            scale = v1_record.scale
        source_provenance = item.get("source_provenance")
        if isinstance(source_provenance, dict):
            provenance = dict(source_provenance)
        else:
            provenance = {
                "restored_from_sidecar": True,
                "downgrade_residual": item.get("truncation_residual"),
                "exact": not missing_fields,
                "unrecoverable_fields": missing_fields,
            }
        restored[layer] = CalibrationRecord(
            layer=layer,
            scale=scale,
            zero_point=int(item.get("zero_point", v1_record.zero_point)),
            bit_width=int(item.get("bit_width", v1_record.bit_width)),
            scheme=str(item.get("scheme", "affine")),
            rounding_mode=str(item.get("rounding_mode", "nearest_even")),
            precision_label=str(item.get("precision_label", "unspecified")),
            per_channel=bool(item.get("per_channel", False)),
            source_version="v2",
            provenance=provenance,
        )
    status = "restored" if restored and not failures else "partially_restored" if restored else "failed"
    return restored, failures, status


def compare_v2_records(
    left: dict[str, CalibrationRecord],
    right: dict[str, CalibrationRecord],
) -> dict[str, Any]:
    layers = sorted(set(left) | set(right))
    exact_layers: list[str] = []
    mismatch_layers: list[str] = []
    residuals: dict[str, float] = {}
    for layer in layers:
        a = left.get(layer)
        b = right.get(layer)
        equal = a is not None and b is not None and a.to_dict() == b.to_dict()
        if equal:
            exact_layers.append(layer)
        else:
            mismatch_layers.append(layer)
            if a is not None and b is not None:
                residuals[layer] = float(b.scale) - float(a.scale)
    return {
        "exact": not mismatch_layers,
        "exact_layers": exact_layers,
        "mismatch_layers": mismatch_layers,
        "residuals": residuals,
    }
