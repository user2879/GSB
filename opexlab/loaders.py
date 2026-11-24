"""Line-oriented activation, v1 fixed-width and v2 JSON loaders."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .config import LabConfig
from .errors import (
    E_DUPLICATE_LAYER,
    E_DUPLICATE_WEIGHT_ROW,
    E_FIELD_TRUNCATED,
    E_NONFINITE,
    E_PARSE,
    E_PRECISION_UNKNOWN,
    E_ROW_RANGE,
    E_SCHEME_UNKNOWN,
    E_SHAPE,
    E_WIDTH_OUT_OF_RANGE,
    make_failure,
)
from .models import LayerTensor, LoadedSources, SourceBundle, WeightRow
from .util import is_finite_number, is_nonnegative_int, is_positive_int, sha256_obj


def read_source_bundle(config: LabConfig) -> SourceBundle:
    activation_lines = config.paths["activations"].read_text(encoding="utf-8").splitlines()
    v1_lines = config.paths["weights_v1"].read_text(encoding="utf-8").splitlines()
    v2_text = config.paths["weights_v2"].read_text(encoding="utf-8")
    stripped = v2_text.lstrip()
    if stripped.startswith("["):
        parsed = json.loads(v2_text)
        if not isinstance(parsed, list):
            raise ValueError("v2 JSON root array expected")
        v2_values: list[Any] = [dict(item) for item in parsed]
    else:
        v2_values = [json.loads(line) for line in v2_text.splitlines() if line.strip()]
    return SourceBundle(
        activation_lines=activation_lines,
        v1_lines=v1_lines,
        v2_values=v2_values,
    )


def apply_activation_overrides(
    lines: list[str], overrides: Dict[str, Any]
) -> tuple[list[Any], list[str]]:
    """Return parsed mutable activation objects plus JSON parse failures.

    The function first parses JSON lines, applies ``layer -> {row: values}``
    overrides, and then serializes back to canonical JSON.  This mirrors an
    editor replacing one activation row in the tensor package.
    """
    objects: list[Any] = []
    failures: list[str] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            objects.append(json.loads(line))
        except Exception as exc:
            failures.append(f"line {line_no}: {exc}")
    for layer_name, row_map in overrides.items():
        if not isinstance(row_map, dict):
            continue
        for obj in objects:
            if not isinstance(obj, dict) or obj.get("layer") != layer_name:
                continue
            activation = obj.get("activation")
            if not isinstance(activation, list):
                continue
            for row_key, values in row_map.items():
                try:
                    row_index = int(row_key)
                except Exception:
                    continue
                if 0 <= row_index < len(activation):
                    activation[row_index] = list(values)
    return objects, failures


def load_activation_objects(
    objects: Iterable[Any], config: LabConfig
) -> tuple[Dict[str, LayerTensor], list[Dict[str, Any]]]:
    layers: Dict[str, LayerTensor] = {}
    failures: list[Dict[str, Any]] = []
    for obj_index, obj in enumerate(objects, start=1):
        if not isinstance(obj, dict):
            failures.append(
                make_failure(
                    E_PARSE,
                    "load_validation",
                    "activation row package must be a JSON object",
                    details={"object_index": obj_index},
                )
            )
            continue
        layer_name = obj.get("layer")
        if not isinstance(layer_name, str) or not layer_name.strip():
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    "layer name must be a non-empty string",
                    details={"object_index": obj_index},
                )
            )
            continue
        layer_name = layer_name.strip()
        if layer_name in layers:
            failures.append(
                make_failure(
                    E_DUPLICATE_LAYER,
                    "load_validation",
                    "layer name must be unique",
                    layer=layer_name,
                    details={"object_index": obj_index},
                )
            )
            continue
        rows_declared = obj.get("rows")
        dim = obj.get("dim")
        if not is_positive_int(rows_declared) or not is_positive_int(dim):
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    "rows and dim must be positive integers",
                    layer=layer_name,
                    details={"rows": rows_declared, "dim": dim},
                )
            )
            continue
        activation = obj.get("activation")
        if not isinstance(activation, list) or len(activation) != rows_declared:
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    "activation row count does not match rows",
                    layer=layer_name,
                    details={"declared": rows_declared, "actual": len(activation) if isinstance(activation, list) else None},
                )
            )
            continue
        accepted_rows: Dict[int, Tuple[float, ...]] = {}
        for row_index, values in enumerate(activation):
            if not isinstance(values, list) or len(values) != dim:
                failures.append(
                    make_failure(
                        E_SHAPE,
                        "load_validation",
                        "activation row dimension does not match dim",
                        layer=layer_name,
                        row=row_index,
                        details={"expected": dim, "actual": len(values) if isinstance(values, list) else None},
                    )
                )
                continue
            if not all(is_finite_number(value) for value in values):
                failures.append(
                    make_failure(
                        E_NONFINITE,
                        "load_validation",
                        "activation contains a non-finite element",
                        layer=layer_name,
                        row=row_index,
                    )
                )
                continue
            accepted_rows[row_index] = tuple(float(value) for value in values)
        layernorm = obj.get("layernorm", {})
        scale_values = layernorm.get("scale") if isinstance(layernorm, dict) else None
        bias_values = layernorm.get("bias") if isinstance(layernorm, dict) else None
        reference_scale, scale_ok = _coerce_parameter_package(
            scale_values, rows_declared, dim, layer_name, "scale", failures
        )
        reference_bias, bias_ok = _coerce_parameter_package(
            bias_values, rows_declared, dim, layer_name, "bias", failures
        )
        if not scale_ok or not bias_ok:
            continue
        seed = obj.get("seed")
        if seed is not None and not isinstance(seed, int):
            seed = None
        layers[layer_name] = LayerTensor(
            name=layer_name,
            seed=seed,
            dim=dim,
            rows=accepted_rows,
            reference_scale=reference_scale,
            reference_bias=reference_bias,
            source_line=obj_index,
        )
    return layers, failures


def _coerce_parameter_package(
    value: Any,
    rows: int,
    dim: int,
    layer: str,
    field: str,
    failures: list[Dict[str, Any]],
) -> tuple[Tuple[float, ...], bool]:
    if is_finite_number(value):
        return (float(value),) * dim, True
    if not isinstance(value, list) or not value:
        failures.append(
            make_failure(
                E_SHAPE,
                "load_validation",
                f"layernorm.{field} must be a finite scalar or non-empty vector",
                layer=layer,
            )
        )
        return (), False
    if all(is_finite_number(item) for item in value):
        if len(value) != dim:
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    f"layernorm.{field} length must equal dim",
                    layer=layer,
                    details={"expected": dim, "actual": len(value)},
                )
            )
            return (), False
        return tuple(float(item) for item in value), True
    if all(isinstance(row, list) for row in value):
        if len(value) != rows:
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    f"layernorm.{field} row count must equal rows",
                    layer=layer,
                )
            )
            return (), False
        flattened: list[float] = []
        for row_index, row_values in enumerate(value):
            if len(row_values) != dim or not all(is_finite_number(item) for item in row_values):
                failures.append(
                    make_failure(
                        E_NONFINITE if any(not is_finite_number(item) for item in row_values) else E_SHAPE,
                        "load_validation",
                        f"layernorm.{field} row must contain finite dim elements",
                        layer=layer,
                        row=row_index,
                    )
                )
                return (), False
            flattened.extend(float(item) for item in row_values)
        if any(item is None for item in flattened):
            return (), False
        # Per-row vectors are validated, but the reference package is retained
        # as a flattened dimension vector for deterministic hashing.
        return tuple(flattened), True
    failures.append(
        make_failure(
            E_NONFINITE,
            "load_validation",
            f"layernorm.{field} contains a non-finite value",
            layer=layer,
        )
    )
    return (), False


def make_v1_line(
    *,
    layer: str,
    row: int,
    scale_permille: int,
    bias_code: int,
    quant_bits: int,
    widths: Dict[str, int],
) -> str:
    if len(layer) > widths["layer"]:
        raise ValueError(E_FIELD_TRUNCATED)
    fields = [
        f"{layer:<{widths['layer']}}",
        f"{row:>{widths['row']}d}",
        f"{scale_permille:>{widths['scale_permille']}d}",
        f"{bias_code:>{widths['bias_code']}d}",
        f"{quant_bits:>{widths['quant_bits']}d}",
    ]
    for field_name, rendered in zip(
        ("layer", "row", "scale_permille", "bias_code", "quant_bits"), fields
    ):
        if len(rendered) > widths[field_name]:
            raise ValueError(E_FIELD_TRUNCATED)
    return "".join(fields)


def _parse_int_field(text: str) -> int:
    stripped = text.strip()
    if not stripped or stripped in ("+", "-") or not stripped.lstrip("+-").isdigit():
        raise ValueError(f"not an integer field: {text!r}")
    return int(stripped)


def load_weights_v1(config: LabConfig, lines: Iterable[str]) -> tuple[Dict[Tuple[str, int], WeightRow], list[Dict[str, Any]]]:
    weights: Dict[Tuple[str, int], WeightRow] = {}
    failures: list[Dict[str, Any]] = []
    widths = config.field_widths
    total_width = sum(widths.values())
    bounds = []
    cursor = 0
    for name in ("layer", "row", "scale_permille", "bias_code", "quant_bits"):
        bounds.append((name, cursor, cursor + widths[name]))
        cursor += widths[name]
    for line_no, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        line = raw_line.rstrip("\r\n")
        fields = {name: line[start:end] for name, start, end in bounds}
        layer_value = fields["layer"].strip() if len(line) >= widths["layer"] else ""
        row_value: Optional[int] = None
        try:
            if len(line) >= widths["layer"] + widths["row"]:
                row_value = _parse_int_field(fields["row"])
        except Exception:
            row_value = None
        if len(line) < total_width:
            failures.append(
                make_failure(
                    E_FIELD_TRUNCATED,
                    "load_validation",
                    "v1 fixed-width line is shorter than the configured field widths",
                    layer=layer_value or None,
                    row=row_value,
                    details={"line": line_no, "expected_width": total_width, "actual_width": len(line)},
                )
            )
            continue
        if len(line) > total_width:
            failures.append(
                make_failure(
                    E_FIELD_TRUNCATED,
                    "load_validation",
                    "v1 fixed-width line does not match the configured field widths",
                    layer=layer_value or None,
                    row=row_value,
                    details={"line": line_no, "expected_width": total_width, "actual_width": len(line)},
                )
            )
            continue
        try:
            layer_name = fields["layer"].strip()
            row = _parse_int_field(fields["row"])
            scale_permille = _parse_int_field(fields["scale_permille"])
            bias_code = _parse_int_field(fields["bias_code"])
            quant_bits = _parse_int_field(fields["quant_bits"])
        except Exception as exc:
            failures.append(
                make_failure(
                    E_PARSE,
                    "load_validation",
                    f"cannot parse fixed-width fields: {exc}",
                    layer=layer_value or None,
                    row=row_value,
                    details={"line": line_no},
                )
            )
            continue
        if not layer_name:
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    "v1 layer name is empty",
                    row=row,
                    details={"line": line_no},
                )
            )
            continue
        if row < 0:
            failures.append(
                make_failure(
                    E_ROW_RANGE,
                    "load_validation",
                    "v1 row index must be non-negative",
                    layer=layer_name,
                    row=row,
                    details={"line": line_no},
                )
            )
            continue
        if quant_bits not in config.bits_to_precision:
            failures.append(
                make_failure(
                    E_WIDTH_OUT_OF_RANGE,
                    "load_validation",
                    "quantization width is not in the configured code table",
                    layer=layer_name,
                    row=row,
                    details={"quant_bits": quant_bits, "allowed_bits": sorted(config.bits_to_precision)},
                )
            )
            continue
        key = (layer_name, row)
        if key in weights:
            failures.append(
                make_failure(
                    E_DUPLICATE_WEIGHT_ROW,
                    "load_validation",
                    "duplicate v1 layer/row weight",
                    layer=layer_name,
                    row=row,
                    details={"line": line_no},
                )
            )
            continue
        weights[key] = WeightRow(
            layer=layer_name,
            row=row,
            scale=scale_permille / 1000.0,
            bias=bias_code / 1000.0,
            scheme=config.v1_upgrade_scheme,
            precision=config.bits_to_precision[quant_bits],
            quant_bits=quant_bits,
            source_version="v1",
            raw={
                "scale_permille": scale_permille,
                "bias_code": bias_code,
                "quant_bits": quant_bits,
                "line_width": total_width,
            },
            source_line=line_no,
        )
    return weights, failures


def load_weights_v2(
    config: LabConfig, values: Iterable[Any]
) -> tuple[Dict[Tuple[str, int], WeightRow], list[Dict[str, Any]]]:
    weights: Dict[Tuple[str, int], WeightRow] = {}
    failures: list[Dict[str, Any]] = []
    for obj_index, obj in enumerate(values, start=1):
        if not isinstance(obj, dict):
            failures.append(
                make_failure(
                    E_PARSE,
                    "load_validation",
                    "v2 weight row must be a JSON object",
                    details={"object_index": obj_index},
                )
            )
            continue
        layer_name = obj.get("layer")
        row = obj.get("row")
        failure_layer = layer_name.strip() if isinstance(layer_name, str) else None
        failure_row = row if isinstance(row, int) and not isinstance(row, bool) else None
        if not isinstance(layer_name, str) or not layer_name.strip():
            failures.append(
                make_failure(
                    E_SHAPE,
                    "load_validation",
                    "v2 layer name must be a non-empty string",
                    row=failure_row,
                    details={"object_index": obj_index},
                )
            )
            continue
        layer_name = layer_name.strip()
        if not is_nonnegative_int(row):
            failures.append(
                make_failure(
                    E_ROW_RANGE,
                    "load_validation",
                    "v2 row index must be a non-negative integer",
                    layer=layer_name,
                    details={"object_index": obj_index, "row": row},
                )
            )
            continue
        scale = obj.get("scale")
        bias = obj.get("bias", 0.0)
        scheme = obj.get("scheme")
        precision = obj.get("precision")
        if not is_finite_number(scale) or not is_finite_number(bias):
            failures.append(
                make_failure(
                    E_NONFINITE,
                    "load_validation",
                    "v2 scale and bias must be finite",
                    layer=layer_name,
                    row=row,
                    details={"scale": scale, "bias": bias},
                )
            )
            continue
        if scheme not in config.allowed_schemes:
            failures.append(
                make_failure(
                    E_SCHEME_UNKNOWN,
                    "load_validation",
                    "v2 normalization scheme is not in the allowed enum",
                    layer=layer_name,
                    row=row,
                    details={"scheme": scheme, "allowed": list(config.allowed_schemes)},
                )
            )
            continue
        if precision not in config.quant_codebook:
            failures.append(
                make_failure(
                    E_PRECISION_UNKNOWN,
                    "load_validation",
                    "v2 precision label is not in the quantization code table",
                    layer=layer_name,
                    row=row,
                    details={"precision": precision, "allowed": sorted(config.quant_codebook)},
                )
            )
            continue
        key = (layer_name, row)
        if key in weights:
            failures.append(
                make_failure(
                    E_DUPLICATE_WEIGHT_ROW,
                    "load_validation",
                    "duplicate v2 layer/row weight",
                    layer=layer_name,
                    row=row,
                    details={"object_index": obj_index},
                )
            )
            continue
        weights[key] = WeightRow(
            layer=layer_name,
            row=row,
            scale=float(scale),
            bias=float(bias),
            scheme=str(scheme),
            precision=str(precision),
            quant_bits=config.quant_codebook[str(precision)],
            source_version="v2",
            raw={
                "precision_tag": obj.get("precision_tag"),
                "named_width": obj.get("named_width"),
            },
            sidecar=copy.deepcopy(obj.get("sidecar", {})) if isinstance(obj.get("sidecar", {}), dict) else {},
            source_line=obj_index,
        )
    return weights, failures


def load_all_sources(
    config: LabConfig,
    bundle: SourceBundle,
    *,
    activation_overrides: Optional[Dict[str, Any]] = None,
) -> LoadedSources:
    objects, parse_failures = apply_activation_overrides(
        bundle.activation_lines, activation_overrides or {}
    )
    layers, activation_failures = load_activation_objects(objects, config)
    v1_weights, v1_failures = load_weights_v1(config, bundle.v1_lines)
    v2_weights, v2_failures = load_weights_v2(config, bundle.v2_values)
    failures = list(activation_failures) + list(v1_failures) + list(v2_failures)
    for message in parse_failures:
        failures.append(make_failure(E_PARSE, "load_validation", message))
    fingerprint_parts = {
        "activation_objects": objects,
        "v1_lines": bundle.v1_lines,
        "v2_values": bundle.v2_values,
        "activation_overrides": activation_overrides or {},
    }
    return LoadedSources(
        layers=layers,
        weights_v1=v1_weights,
        weights_v2=v2_weights,
        failures=failures,
        load_fingerprint_parts=fingerprint_parts,
    )


def source_fingerprint(loaded: LoadedSources) -> str:
    return sha256_obj(loaded.load_fingerprint_parts)
