"""JSON v2 exchange codec and v1/v2 conversion with explicit loss metadata."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterable

from .errors import QuantCalError
from .format_v1 import V1Widths, render_v1
from .models import (
    SUPPORTED_ROUNDING,
    SUPPORTED_SCHEMES,
    ExchangeSet,
    TensorParam,
    quant_bounds,
    scheme_saturation_max,
    symmetric_saturation_max,
)
from .util import clean_float, half_away_round, is_int


def _parse_bits(value: Any, tensor: str | None = None) -> int:
    if not is_int(value) or not (2 <= value <= 32):
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"quantization_bits must be an integer in [2, 32], got {value!r}",
            tensor=tensor,
            stage="parse",
        )
    return int(value)


def _parse_sat(value: Any, tensor: str) -> int:
    if not is_int(value) or value <= 0:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            "saturation max must be a positive integer",
            tensor=tensor,
            stage="parse",
        )
    return int(value)


def _parse_scale(value: Any, tensor: str) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise QuantCalError(
            "E_FORMAT_INVALID", "scale must be numeric", tensor=tensor, stage="parse"
        ) from exc
    if not math.isfinite(scale):
        raise QuantCalError(
            "E_FORMAT_INVALID", "scale must be finite", tensor=tensor, stage="parse"
        )
    if scale <= 0.0:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            "scale must be positive",
            tensor=tensor,
            stage="parse",
        )
    return scale


def _parse_range(value: Any, tensor: str) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        lower = value.get("lower")
        upper = value.get("upper")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        raise QuantCalError(
            "E_FORMAT_INVALID",
            "quantile_range must be [lower, upper] or an object with lower/upper",
            tensor=tensor,
            stage="parse",
        )
    if lower is None and upper is None:
        return None, None
    try:
        lower_float = float(lower)
        upper_float = float(upper)
    except (TypeError, ValueError) as exc:
        raise QuantCalError(
            "E_FORMAT_INVALID",
            "quantile_range bounds must be numeric",
            tensor=tensor,
            stage="parse",
        ) from exc
    if not (math.isfinite(lower_float) and math.isfinite(upper_float)):
        raise QuantCalError(
            "E_FORMAT_INVALID",
            "quantile_range bounds must be finite",
            tensor=tensor,
            stage="parse",
        )
    if not lower_float < upper_float:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"quantile lower {lower_float} must be less than upper {upper_float}",
            tensor=tensor,
            stage="parse",
        )
    return lower_float, upper_float


def parse_v2(
    document: Any,
    expected_tensor_names: Iterable[str] | None = None,
    require_complete: bool = True,
) -> ExchangeSet:
    if not isinstance(document, dict):
        raise QuantCalError("E_FORMAT_INVALID", "v2 exchange must be a JSON object", stage="parse")
    if document.get("version") != "v2":
        raise QuantCalError("E_FORMAT_INVALID", "v2 exchange requires version='v2'", stage="parse")
    default_bits = _parse_bits(document.get("quantization_bits", 8))
    rows = document.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise QuantCalError(
            "E_FORMAT_INVALID", "v2 tensors must be a non-empty array", stage="parse"
        )
    tensors: list[TensorParam] = []
    seen: set[str] = set()
    expected = set(expected_tensor_names) if expected_tensor_names is not None else None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise QuantCalError(
                "E_FORMAT_INVALID", f"v2 tensor row {index} must be an object", stage="parse"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise QuantCalError(
                "E_FORMAT_INVALID", f"v2 tensor row {index} has invalid name", stage="parse"
            )
        if name in seen:
            raise QuantCalError(
                "E_FORMAT_INVALID", "duplicate tensor name in v2 package", tensor=name, stage="parse"
            )
        seen.add(name)
        if expected is not None and name not in expected:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                "v2 tensor name is absent from tensor manifest",
                tensor=name,
                stage="parse",
            )
        bits = _parse_bits(row.get("quantization_bits", default_bits), tensor=name)
        scheme = row.get("scheme")
        if scheme not in SUPPORTED_SCHEMES:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"unsupported scheme {scheme!r}",
                tensor=name,
                stage="parse",
            )
        rounding = row.get("rounding")
        if rounding not in SUPPORTED_ROUNDING:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"unsupported rounding mode {rounding!r}",
                tensor=name,
                stage="parse",
            )
        scale = _parse_scale(row.get("scale"), name)
        zero_point = row.get("zero_point")
        if not is_int(zero_point):
            raise QuantCalError(
                "E_FORMAT_INVALID", "zero_point must be an integer", tensor=name, stage="parse"
            )
        qmin, qmax = quant_bounds(scheme, bits)
        if not (qmin <= zero_point <= qmax):
            raise QuantCalError(
                "E_ZERO_POINT_INVALID",
                f"zero point {zero_point} outside [{qmin}, {qmax}]",
                tensor=name,
                stage="parse",
            )
        saturation_max = _parse_sat(
            row.get("saturation_max", scheme_saturation_max(scheme, bits)), name
        )
        if saturation_max > max(abs(qmin), qmax):
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                f"saturation max {saturation_max} exceeds {bits}-bit range",
                tensor=name,
                stage="parse",
            )
        lower, upper = _parse_range(row.get("quantile_range"), name)
        precision_tag = row.get("precision_tag", "unspecified")
        if not isinstance(precision_tag, str) or not precision_tag:
            raise QuantCalError(
                "E_FORMAT_INVALID", "precision_tag must be a non-empty string", tensor=name, stage="parse"
            )
        source = row.get("source", {})
        if source is None:
            source = {}
        if not isinstance(source, dict):
            raise QuantCalError(
                "E_FORMAT_INVALID", "source must be an object", tensor=name, stage="parse"
            )
        tensors.append(
            TensorParam(
                name=name,
                scale=scale,
                zero_point=int(zero_point),
                scheme=scheme,
                rounding=rounding,
                precision_tag=precision_tag,
                quantization_bits=bits,
                saturation_max=saturation_max,
                quantile_lower=lower,
                quantile_upper=upper,
                source=deepcopy(source),
            )
        )
    if require_complete and expected is not None:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"v2 tensor set mismatch; missing={missing}, extra={extra}",
                stage="parse",
            )
    bits_values = {tensor.quantization_bits for tensor in tensors}
    bundle_bits = next(iter(bits_values)) if len(bits_values) == 1 else default_bits
    return ExchangeSet(
        version="v2",
        quantization_bits=bundle_bits,
        tensors=tensors,
        metadata={
            "format": "quantcal.exchange",
            "quantization_bits": default_bits,
            "mixed_quantization_bits": len(bits_values) > 1,
            "metadata": deepcopy(document.get("metadata", {})),
        },
    )


def to_v2_document(exchange: ExchangeSet) -> dict[str, Any]:
    bits_values = {tensor.quantization_bits for tensor in exchange.tensors}
    rows: list[dict[str, Any]] = []
    for tensor in exchange.tensors:
        if tensor.quantile_lower is None or tensor.quantile_upper is None:
            quantile_range: Any = None
        else:
            quantile_range = [tensor.quantile_lower, tensor.quantile_upper]
        rows.append(
            {
                "name": tensor.name,
                "scale": tensor.scale,
                "zero_point": tensor.zero_point,
                "scheme": tensor.scheme,
                "rounding": tensor.rounding,
                "precision_tag": tensor.precision_tag,
                "quantization_bits": tensor.quantization_bits,
                "saturation_max": tensor.saturation_max,
                "quantile_range": quantile_range,
                "source": deepcopy(tensor.source),
            }
        )
    return {
        "format": "quantcal.exchange",
        "version": "v2",
        "quantization_bits": (
            next(iter(bits_values))
            if len(bits_values) == 1
            else exchange.quantization_bits
        ),
        "tensors": rows,
        "metadata": deepcopy(exchange.metadata),
    }


def upgrade_v1_to_v2(exchange: ExchangeSet) -> tuple[ExchangeSet, dict[str, Any]]:
    """Lift v1 while making every information loss explicit."""

    if exchange.version != "v1":
        raise QuantCalError("E_FORMAT_INVALID", "upgrade_v1_to_v2 requires a v1 exchange")
    tensors: list[TensorParam] = []
    loss_by_tensor: list[dict[str, Any]] = []
    for tensor in exchange.tensors:
        milli = int(tensor.source.get("scale_milli", half_away_round(tensor.scale * 1000.0)))
        restored_scale = milli / 1000.0
        source = deepcopy(tensor.source)
        source.update(
            {
                "upgraded_from": "v1",
                "scale_source": "milli_integer",
                "scale_milli": milli,
                "approximation": {
                    "asymmetric_scheme": "not_representable_in_v1;normalized_to_symmetric",
                    "named_rounding": "not_representable_in_v1;nearest_even_assumed",
                    "quantile_range": "not_representable_in_v1;unavailable",
                },
            }
        )
        tensors.append(
            TensorParam(
                name=tensor.name,
                scale=restored_scale,
                zero_point=tensor.zero_point,
                scheme="symmetric",
                rounding="nearest_even",
                precision_tag="v1_milli_exact_1e-3",
                quantization_bits=tensor.quantization_bits,
                saturation_max=tensor.saturation_max,
                quantile_lower=None,
                quantile_upper=None,
                source=source,
            )
        )
        loss_by_tensor.append(
            {
                "name": tensor.name,
                "approximated_as": "symmetric",
                "source": "v1_cannot_encode_asymmetric_scheme",
                "named_rounding_lost": True,
                "quantile_range_lost": True,
                "scale_exact": True,
                "scale_milli": milli,
            }
        )
    upgraded = ExchangeSet(
        version="v2",
        quantization_bits=exchange.quantization_bits,
        tensors=tensors,
        metadata={
            "upgraded_from": "v1",
            "approximations": "asymmetric/rounding/quantile metadata normalized",
            "source_metadata": deepcopy(exchange.metadata),
        },
    )
    loss_by_tensor.sort(key=lambda row: row["name"])
    losses = {
        "format": "quantcal.upgrade-losses.v1",
        "source_version": "v1",
        "target_version": "v2",
        "losses": loss_by_tensor,
    }
    return upgraded, losses


def downgrade_v2_to_v1(
    exchange: ExchangeSet, widths: V1Widths | None = None
) -> tuple[ExchangeSet, str, dict[str, Any]]:
    """Lower v2 to fixed-width v1 and return text plus stable loss sidecar."""

    if exchange.version != "v2":
        raise QuantCalError("E_FORMAT_INVALID", "downgrade_v2_to_v1 requires a v2 exchange")
    bits_values = {tensor.quantization_bits for tensor in exchange.tensors}
    if len(bits_values) != 1:
        if bits_values:
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                f"v1 cannot represent mixed bit widths {sorted(bits_values)}",
            )
    bits = next(iter(bits_values)) if bits_values else exchange.quantization_bits
    v1_tensors: list[TensorParam] = []
    loss_rows: list[dict[str, Any]] = []
    for tensor in exchange.tensors:
        exact_milli = tensor.scale * 1000.0
        rounded_milli = half_away_round(exact_milli)
        clamped = False
        if rounded_milli <= 0:
            rounded_milli = 1
            clamped = True
        v1_scale = rounded_milli / 1000.0
        v1_zero = tensor.zero_point if tensor.scheme == "symmetric" else 0
        symmetric_max = symmetric_saturation_max(bits)
        v1_saturation = (
            tensor.saturation_max
            if tensor.scheme == "symmetric" and 0 < tensor.saturation_max <= symmetric_max
            else symmetric_max
        )
        rounding_residual = clean_float(tensor.scale - v1_scale)
        milli_residual = clean_float(exact_milli - rounded_milli)
        loss_codes: list[str] = []
        if milli_residual != 0.0 or clamped:
            loss_codes.append("scale_quantized_to_milli")
        loss_codes.append("named_rounding_mode_discarded")
        if tensor.scheme == "asymmetric":
            loss_codes.append("asymmetric_scheme_approximated_as_symmetric")
            if tensor.zero_point != 0:
                loss_codes.append("asymmetric_zero_point_offset")
        if tensor.quantile_lower is not None or tensor.quantile_upper is not None:
            loss_codes.append("quantile_range_discarded")
        if v1_saturation != tensor.saturation_max:
            loss_codes.append("saturation_cap_changed")
        loss_codes.append("precision_tag_discarded")
        v1_tensors.append(
            TensorParam(
                name=tensor.name,
                scale=v1_scale,
                zero_point=v1_zero,
                scheme="symmetric",
                rounding="nearest_even",
                precision_tag="v1_milli_exact_1e-3",
                quantization_bits=bits,
                saturation_max=v1_saturation,
                quantile_lower=None,
                quantile_upper=None,
                source={
                    "downgraded_from": "v2",
                    "scale_milli": rounded_milli,
                    "original_scheme": tensor.scheme,
                    "original_zero_point": tensor.zero_point,
                    "original_precision_tag": tensor.precision_tag,
                },
            )
        )
        loss_rows.append(
            {
                "name": tensor.name,
                "loss_codes": sorted(loss_codes),
                "scale_original": tensor.scale,
                "scale_milli_exact": exact_milli,
                "scale_milli_rounded": rounded_milli,
                "scale_after": v1_scale,
                "scale_truncation_residual_milli": milli_residual,
                "scale_rounding_residual": rounding_residual,
                "scale_clamped_to_min_milli": clamped,
                "named_rounding_mode": tensor.rounding,
                "asymmetric_zero_point_offset": (
                    tensor.zero_point if tensor.scheme == "asymmetric" else 0
                ),
                "original_scheme": tensor.scheme,
                "discarded_quantile_range": (
                    [tensor.quantile_lower, tensor.quantile_upper]
                    if tensor.quantile_lower is not None
                    else None
                ),
                "precision_tag_original": tensor.precision_tag,
                "saturation_max_original": tensor.saturation_max,
                "saturation_max_v1": v1_saturation,
            }
        )
    v1_exchange = ExchangeSet(
        version="v1",
        quantization_bits=bits,
        tensors=v1_tensors,
        metadata={
            "downgraded_from": "v2",
            "source_metadata": deepcopy(exchange.metadata),
        },
    )
    text = render_v1(v1_exchange, widths)
    loss_rows.sort(key=lambda row: row["name"])
    sidecar = {
        "format": "quantcal.downgrade-sidecar.v1",
        "source_version": "v2",
        "target_version": "v1",
        "quantization_bits": bits,
        "tensors": loss_rows,
        "losses": sorted({code for row in loss_rows for code in row["loss_codes"]}),
    }
    return v1_exchange, text, sidecar


v1_to_v2 = upgrade_v1_to_v2
v2_to_v1 = downgrade_v2_to_v1
