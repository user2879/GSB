"""Bidirectional v1/v2 weight conversion with loss sidecars."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .config import LabConfig
from .errors import E_FIELD_TRUNCATED, E_PRECISION_UNKNOWN, E_WIDTH_OUT_OF_RANGE, LabError
from .loaders import make_v1_line
from .models import WeightRow
from .util import round_half_away, sha256_obj


CONVERSION_IMPLEMENTATION = "opexlab-conversion-v1"


def _sidecar_hash(sidecar: Dict[str, Any]) -> str:
    return sha256_obj(sidecar)


def v1_to_v2(config: LabConfig, row: WeightRow) -> WeightRow:
    scale_permille = int(row.raw["scale_permille"])
    bias_code = int(row.raw["bias_code"])
    quant_bits = int(row.raw["quant_bits"])
    precision = config.bits_to_precision.get(quant_bits)
    if precision is None:
        raise LabError(
            E_WIDTH_OUT_OF_RANGE,
            "v1 quantization width has no named v2 precision",
            stage="version_conversion",
            layer=row.layer,
            row=row.row,
            details={"quant_bits": quant_bits},
        )
    sidecar = {
        "implementation": CONVERSION_IMPLEMENTATION,
        "direction": "v1_to_v2",
        "source_fields": {
            "scale_permille": scale_permille,
            "bias_code": bias_code,
            "quant_bits": quant_bits,
        },
        "precision_tag": "permille_exact_to_float",
        "exact_fields": ["scale", "bias", "quant_bits"],
        "dropped_fields": [],
    }
    sidecar["sidecar_hash"] = _sidecar_hash(sidecar)
    return WeightRow(
        layer=row.layer,
        row=row.row,
        scale=scale_permille / 1000.0,
        bias=bias_code / 1000.0,
        scheme=config.v1_upgrade_scheme,
        precision=precision,
        quant_bits=quant_bits,
        source_version="v2",
        raw={
            "precision_tag": sidecar["precision_tag"],
            "named_width": precision,
        },
        sidecar=sidecar,
        source_line=row.source_line,
    )


def v2_to_v1(config: LabConfig, row: WeightRow) -> WeightRow:
    if row.layer and len(row.layer) > config.field_widths["layer"]:
        raise LabError(
            E_FIELD_TRUNCATED,
            "layer name cannot fit the v1 fixed layer field",
            stage="version_conversion",
            layer=row.layer,
            row=row.row,
            details={"width": config.field_widths["layer"], "length": len(row.layer)},
        )
    quant_bits = row.quant_bits
    if quant_bits not in config.bits_to_precision:
        raise LabError(
            E_WIDTH_OUT_OF_RANGE,
            "v2 named precision cannot be mapped to a v1 quantization width",
            stage="version_conversion",
            layer=row.layer,
            row=row.row,
            details={"precision": row.precision, "quant_bits": quant_bits},
        )
    scale_permille = round_half_away(row.scale * 1000.0)
    bias_code = round_half_away(row.bias * 1000.0)
    restored_scale = scale_permille / 1000.0
    restored_bias = bias_code / 1000.0
    sidecar = {
        "implementation": CONVERSION_IMPLEMENTATION,
        "direction": "v2_to_v1",
        "precision_label": row.precision,
        "precision_tag": row.raw.get("precision_tag"),
        "dropped_scheme": row.scheme,
        "scale_residual": row.scale - restored_scale,
        "bias_residual": row.bias - restored_bias,
        "scale_permille": scale_permille,
        "bias_code": bias_code,
        "quant_bits": quant_bits,
        "rounding": "half_away_from_zero",
        "exact_fields": ["quant_bits"],
        "lossy_fields": ["scale", "bias", "scheme", "precision_label"],
    }
    sidecar["sidecar_hash"] = _sidecar_hash(sidecar)
    try:
        line = make_v1_line(
            layer=row.layer,
            row=row.row,
            scale_permille=scale_permille,
            bias_code=bias_code,
            quant_bits=quant_bits,
            widths=config.field_widths,
        )
    except ValueError as exc:
        raise LabError(
            E_FIELD_TRUNCATED,
            "v1 fixed-width encoding failed",
            stage="version_conversion",
            layer=row.layer,
            row=row.row,
            details={"reason": str(exc)},
        )
    return WeightRow(
        layer=row.layer,
        row=row.row,
        scale=restored_scale,
        bias=restored_bias,
        scheme=config.v1_upgrade_scheme,
        precision=row.precision,
        quant_bits=quant_bits,
        source_version="v1",
        raw={
            "scale_permille": scale_permille,
            "bias_code": bias_code,
            "quant_bits": quant_bits,
            "line_width": sum(config.field_widths.values()),
            "encoded_line": line,
        },
        sidecar=sidecar,
        source_line=row.source_line,
    )


def convert_row(config: LabConfig, row: WeightRow, target_version: str) -> WeightRow:
    if target_version == row.source_version:
        return row.clone()
    if row.source_version == "v1" and target_version == "v2":
        return v1_to_v2(config, row)
    if row.source_version == "v2" and target_version == "v1":
        return v2_to_v1(config, row)
    raise LabError(
        E_PRECISION_UNKNOWN,
        "unsupported conversion direction",
        stage="version_conversion",
        layer=row.layer,
        row=row.row,
        details={"source": row.source_version, "target": target_version},
    )


def conversion_artifact(
    config: LabConfig,
    row: WeightRow,
    target_version: str,
) -> Dict[str, Any]:
    converted = convert_row(config, row, target_version)
    payload = {
        "source": row.source_payload(),
        "target": converted.source_payload(),
        "sidecar": converted.sidecar,
        "sidecar_hash": converted.sidecar.get("sidecar_hash"),
        "implementation": CONVERSION_IMPLEMENTATION,
    }
    return {
        "layer": row.layer,
        "row": row.row,
        "direction": f"{row.source_version}_to_{target_version}",
        "source_version": row.source_version,
        "target_version": target_version,
        "source": row.source_payload(),
        "target": converted.source_payload(),
        "sidecar": copy.deepcopy(converted.sidecar),
        "sidecar_hash": converted.sidecar.get("sidecar_hash"),
        "converted_row": converted,
        "artifact_fingerprint": sha256_obj(payload),
    }


def conversion_artifact_key(
    config: LabConfig,
    row: WeightRow,
    target_version: str,
    *,
    exchange_version: str,
    fractional_bits: int,
    quant_level: int,
) -> str:
    return sha256_obj(
        {
            "stage": "version_conversion",
            "implementation": CONVERSION_IMPLEMENTATION,
            "exchange_version": exchange_version,
            "target_version": target_version,
            "fractional_bits": fractional_bits,
            "quant_level": quant_level,
            "source": row.source_payload(),
        }
    )

