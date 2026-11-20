"""Fixed-width v1 text exchange codec.

The header declares the exact field widths. Each data row contains four fixed
width fields separated by ``|``:

    name | scale_milli | zero_point | saturation_max

The header also carries the shared quantisation bit width. Keeping bit width in
the header lets a v1 -> v2 -> v1 round trip preserve tensor order and bit width
without adding a fifth data field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import QuantCalError
from .models import (
    ExchangeSet,
    TensorParam,
    quant_bounds,
    symmetric_saturation_max,
)
from .util import half_away_round, is_int


V1_HEADER_PREFIX = "# QSCALE-V1"
DEFAULT_V1_WIDTHS = {
    "name_width": 24,
    "scale_milli_width": 12,
    "zero_point_width": 8,
    "saturation_max_width": 8,
}


@dataclass(frozen=True)
class V1Widths:
    name_width: int = 24
    scale_milli_width: int = 12
    zero_point_width: int = 8
    saturation_max_width: int = 8

    @property
    def row_width(self) -> int:
        return (
            self.name_width
            + self.scale_milli_width
            + self.zero_point_width
            + self.saturation_max_width
            + 3
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "name_width": self.name_width,
            "scale_milli_width": self.scale_milli_width,
            "zero_point_width": self.zero_point_width,
            "saturation_max_width": self.saturation_max_width,
        }


def _parse_header(line: str) -> tuple[V1Widths, int | None]:
    parts = line[len(V1_HEADER_PREFIX) :].strip().lstrip(";").split(";")
    values: dict[str, str] = {}
    for part in parts:
        if not part.strip():
            continue
        if "=" not in part:
            raise QuantCalError("E_FORMAT_INVALID", f"invalid v1 header token: {part}")
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    try:
        widths = V1Widths(
            **{
                key: int(values.get(key, default))
                for key, default in DEFAULT_V1_WIDTHS.items()
            }
        )
    except ValueError as exc:
        raise QuantCalError("E_FORMAT_INVALID", f"invalid v1 width header: {exc}") from exc
    for key, width in widths.as_dict().items():
        if width < 1 or width > 256:
            raise QuantCalError(
                "E_FORMAT_INVALID", f"invalid v1 field width {key}={width}"
            )
    bits: int | None = None
    if "bits" in values:
        try:
            bits = int(values["bits"])
        except ValueError as exc:
            raise QuantCalError("E_FORMAT_INVALID", "v1 bits must be an integer") from exc
        if not (2 <= bits <= 32):
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID", f"v1 bits must be in [2, 32], got {bits}"
            )
    return widths, bits


def _infer_bits_from_saturation(saturation_max: int) -> int | None:
    for bits in range(2, 33):
        if symmetric_saturation_max(bits) == saturation_max:
            return bits
    return None


def _validate_tensor(param: TensorParam, expected_names: Iterable[str] | None) -> None:
    qmin, qmax = quant_bounds("symmetric", param.quantization_bits)
    if not (qmin <= param.zero_point <= qmax):
        raise QuantCalError(
            "E_ZERO_POINT_INVALID",
            f"zero point {param.zero_point} outside [{qmin}, {qmax}]",
            tensor=param.name,
            stage="parse",
        )
    if not (param.scale > 0.0):
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            "v1 milli scale must be positive",
            tensor=param.name,
            stage="parse",
        )
    if param.saturation_max <= 0:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            "saturation max must be positive",
            tensor=param.name,
            stage="parse",
        )
    expected_qmax = symmetric_saturation_max(param.quantization_bits)
    if param.saturation_max > expected_qmax:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"saturation max {param.saturation_max} exceeds {param.quantization_bits}-bit symmetric maximum {expected_qmax}",
            tensor=param.name,
            stage="parse",
        )
    if expected_names is not None and param.name not in set(expected_names):
        raise QuantCalError(
            "E_FORMAT_INVALID",
            "v1 tensor name is absent from tensor manifest",
            tensor=param.name,
            stage="parse",
        )


def parse_v1(
    text: str,
    expected_tensor_names: Iterable[str] | None = None,
    require_complete: bool = True,
) -> ExchangeSet:
    """Parse a v1 package, preserving row order as tensor order."""

    lines = text.splitlines()
    header_line = next((line for line in lines if line.startswith(V1_HEADER_PREFIX)), None)
    widths = V1Widths()
    header_bits: int | None = None
    if header_line is not None:
        widths, header_bits = _parse_header(header_line)

    tensors: list[TensorParam] = []
    seen: set[str] = set()
    inferred_bits: int | None = header_bits
    for line_number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        if len(line) != widths.row_width:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"v1 row {line_number} has width {len(line)}, expected {widths.row_width}",
                stage="parse",
            )
        boundaries = (
            widths.name_width,
            widths.name_width + 1,
            widths.name_width + widths.scale_milli_width + 1,
            widths.name_width + widths.scale_milli_width + widths.zero_point_width + 2,
        )
        if any(line[index] != "|" for index in (boundaries[0], boundaries[2], boundaries[3])):
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"v1 row {line_number} has misplaced pipe separators",
                stage="parse",
            )
        name = line[: widths.name_width].strip()
        scale_field = line[
            widths.name_width + 1 : widths.name_width + 1 + widths.scale_milli_width
        ]
        zero_field = line[
            widths.name_width + widths.scale_milli_width + 2 :
            widths.name_width + widths.scale_milli_width + 2 + widths.zero_point_width
        ]
        saturation_field = line[
            widths.name_width + widths.scale_milli_width + widths.zero_point_width + 3 :
        ]
        if not name or "|" in name:
            raise QuantCalError(
                "E_FORMAT_INVALID", f"v1 row {line_number} has invalid tensor name", stage="parse"
            )
        if name in seen:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                "duplicate tensor name in v1 package",
                tensor=name,
                stage="parse",
            )
        seen.add(name)
        try:
            scale_milli = int(scale_field.strip())
            zero_point = int(zero_field.strip())
            saturation_max = int(saturation_field.strip())
        except ValueError as exc:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"v1 row {line_number} contains a non-integer fixed-width field",
                stage="parse",
            ) from exc
        if saturation_max <= 0:
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                "saturation max must be positive",
                tensor=name,
                stage="parse",
            )
        if scale_milli <= 0:
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                "v1 milli scale must be positive",
                tensor=name,
                stage="parse",
            )
        row_bits = header_bits or _infer_bits_from_saturation(saturation_max)
        if row_bits is None:
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                "cannot infer symmetric quantisation width from saturation max",
                tensor=name,
                stage="parse",
            )
        if saturation_max > symmetric_saturation_max(row_bits):
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                f"saturation max {saturation_max} exceeds {row_bits}-bit symmetric maximum",
                tensor=name,
                stage="parse",
            )
        if inferred_bits is None:
            inferred_bits = row_bits
        elif inferred_bits != row_bits:
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                f"mixed quantisation widths {inferred_bits} and {row_bits} are not representable in v1",
                tensor=name,
                stage="parse",
            )
        param = TensorParam(
            name=name,
            scale=scale_milli / 1000.0,
            zero_point=zero_point,
            scheme="symmetric",
            rounding="nearest_even",
            precision_tag="v1_milli_exact_1e-3",
            quantization_bits=row_bits,
            saturation_max=saturation_max,
            source={
                "format": "v1",
                "scale_milli": scale_milli,
                "field_widths": widths.as_dict(),
            },
        )
        _validate_tensor(param, expected_tensor_names)
        tensors.append(param)

    if not tensors:
        raise QuantCalError("E_FORMAT_INVALID", "v1 package contains no tensor rows", stage="parse")
    if inferred_bits is None:
        raise QuantCalError("E_QUANT_RANGE_INVALID", "cannot determine v1 bit width", stage="parse")
    if require_complete and expected_tensor_names is not None:
        expected = set(expected_tensor_names)
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"v1 tensor set mismatch; missing={missing}, extra={extra}",
                stage="parse",
            )
    return ExchangeSet(
        version="v1",
        quantization_bits=inferred_bits,
        tensors=tensors,
        metadata={
            "format": "QSCALE-V1",
            "field_widths": widths.as_dict(),
            "row_width": widths.row_width,
            "quantization_bits": inferred_bits,
        },
    )


def render_v1(exchange: ExchangeSet, widths: V1Widths | None = None) -> str:
    widths = widths or V1Widths()
    if exchange.version != "v1":
        raise QuantCalError("E_FORMAT_INVALID", "render_v1 expects a v1 exchange set")
    bits_set = {tensor.quantization_bits for tensor in exchange.tensors}
    if exchange.tensors and len(bits_set) != 1:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"v1 requires one shared bit width, got {sorted(bits_set)}",
        )
    output_bits = next(iter(bits_set)) if bits_set else exchange.quantization_bits
    header = (
        f"{V1_HEADER_PREFIX};"
        f"name_width={widths.name_width};"
        f"scale_milli_width={widths.scale_milli_width};"
        f"zero_point_width={widths.zero_point_width};"
        f"saturation_max_width={widths.saturation_max_width};"
        f"bits={output_bits}"
    )
    lines = [header]
    for param in exchange.tensors:
        if param.scheme != "symmetric":
            raise QuantCalError(
                "E_FORMAT_INVALID",
                "v1 cannot represent an asymmetric scheme without conversion",
                tensor=param.name,
            )
        source_milli = param.source.get("scale_milli")
        if is_int(source_milli) and abs(source_milli / 1000.0 - param.scale) < 1e-15:
            scale_milli = int(source_milli)
        else:
            scale_milli = half_away_round(param.scale * 1000.0)
        if scale_milli <= 0:
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                "v1 milli scale must be positive",
                tensor=param.name,
            )
        if len(param.name.encode("utf-8")) > widths.name_width:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"tensor name exceeds v1 field width {widths.name_width}",
                tensor=param.name,
            )
        fields = (
            param.name.ljust(widths.name_width),
            str(scale_milli).rjust(widths.scale_milli_width),
            str(param.zero_point).rjust(widths.zero_point_width),
            str(param.saturation_max).rjust(widths.saturation_max_width),
        )
        row = "|".join(fields)
        if len(row) != widths.row_width:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                "fixed-width v1 field overflow",
                tensor=param.name,
            )
        lines.append(row)
    return "\n".join(lines) + "\n"
