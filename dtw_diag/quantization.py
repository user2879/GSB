"""Deterministic fixed-point quantization."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .config import Config
from .fingerprints import canonical_json, fingerprint


@dataclass(frozen=True)
class QuantizedPoint:
    index: int
    value: Any
    missing: bool
    q_level: int | None
    fixed_units: int | None
    residual: str | None
    point_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "missing": self.missing,
            "value": self.value,
            "q_level": self.q_level,
            "fixed_units": self.fixed_units,
            "residual": self.residual,
            "point_key": self.point_key,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuantizedPoint":
        return cls(
            index=raw["index"],
            value=raw.get("value"),
            missing=raw["missing"],
            q_level=raw.get("q_level"),
            fixed_units=raw.get("fixed_units"),
            residual=raw.get("residual"),
            point_key=raw["point_key"],
        )


@dataclass(frozen=True)
class QuantizedSequence:
    sequence_id: str
    points: tuple[QuantizedPoint, ...]
    segment_ids: tuple[Any, ...]
    timestamp_interval: Any
    point_content_key: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "points": [point.to_dict() for point in self.points],
            "segment_ids": list(self.segment_ids),
            "timestamp_interval": self.timestamp_interval,
            "point_content_key": self.point_content_key,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuantizedSequence":
        return cls(
            sequence_id=raw["sequence_id"],
            points=tuple(QuantizedPoint.from_dict(item) for item in raw["points"]),
            segment_ids=tuple(raw.get("segment_ids", [])),
            timestamp_interval=raw.get("timestamp_interval"),
            point_content_key=raw["point_content_key"],
            fingerprint=raw["fingerprint"],
        )


def _decimal_text(value: Any) -> str:
    return format(Decimal(str(value)), "f")


def _is_missing(value: Any, marker: Any) -> bool:
    if value is None:
        return True
    if marker is None:
        return False
    try:
        return value == marker
    except Exception:  # pragma: no cover - defensive for exotic scalar types
        return False


def quantize_sequence(record: Any, config: Config) -> QuantizedSequence:
    """Quantize each observation under one immutable configuration.

    ``fixed_units = round_half_up(value * 2**fraction_bits / step) * step``.
    The residual is emitted as an exact decimal string.
    """

    scale = 1 << config.fraction_bits
    step = config.quantization_step
    points: list[QuantizedPoint] = []
    for index, value in enumerate(record.observations):
        missing = _is_missing(value, config.missing_marker)
        if missing:
            point_key = fingerprint(
                {
                    "quant": {
                        "fraction_bits": config.fraction_bits,
                        "quantization_step": step,
                    },
                    "index": index,
                    "missing": True,
                    "value": None if value is None else value,
                }
            )
            points.append(
                QuantizedPoint(
                    index=index,
                    value=value,
                    missing=True,
                    q_level=None,
                    fixed_units=None,
                    residual=None,
                    point_key=point_key,
                )
            )
            continue

        decimal_value = Decimal(str(value))
        raw_units = decimal_value * Decimal(scale) / Decimal(step)
        q_level = int(raw_units.to_integral_value(rounding=ROUND_HALF_UP))
        fixed_units = q_level * step
        quantized_value = Decimal(fixed_units) / Decimal(scale)
        residual = decimal_value - quantized_value
        residual_text = format(residual.normalize(), "f")
        if residual_text == "-0":
            residual_text = "0"
        value_text = _decimal_text(value)
        point_key = fingerprint(
            {
                "quant": {
                    "fraction_bits": config.fraction_bits,
                    "quantization_step": step,
                },
                "index": index,
                "missing": False,
                "value": value_text,
                "q_level": q_level,
                "fixed_units": fixed_units,
                "residual": residual_text,
            }
        )
        points.append(
            QuantizedPoint(
                index=index,
                value=value,
                missing=False,
                q_level=q_level,
                fixed_units=fixed_units,
                residual=residual_text,
                point_key=point_key,
            )
        )

    point_content_key = fingerprint(
        {
            "sequence_id": record.seq_id,
            "fraction_bits": config.fraction_bits,
            "quantization_step": step,
            "points": [point.point_key for point in points],
        }
    )
    metadata_key = fingerprint(
        {
            "segment_ids": list(record.segment_ids),
            "timestamp_interval": canonical_json(record.timestamp_interval),
        }
    )
    seq_fingerprint = fingerprint(
        {"point_content_key": point_content_key, "metadata_key": metadata_key}
    )
    return QuantizedSequence(
        sequence_id=record.seq_id,
        points=tuple(points),
        segment_ids=tuple(record.segment_ids),
        timestamp_interval=record.timestamp_interval,
        point_content_key=point_content_key,
        fingerprint=seq_fingerprint,
    )
