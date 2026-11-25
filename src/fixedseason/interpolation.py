"""Deterministic window-only missing-value imputation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any

from .fingerprints import referenced_observations
from .jsonutil import fingerprint
from .model import INTERPOLATION_DECIMALS, SeriesRecord

_INTERPOLATION_QUANTUM = Decimal(1).scaleb(-INTERPOLATION_DECIMALS)


def decimal_from_float(value: float) -> Decimal:
    return Decimal(str(value))


def round_decimal_half_away(value: Decimal, places: int | None = None) -> Decimal:
    with localcontext() as context:
        context.prec = 60
        quantum = Decimal(1).scaleb(-places) if places is not None else Decimal(1)
        try:
            return value.quantize(quantum, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return value


def decimal_text(value: Decimal) -> str:
    if value == value.to_integral_value():
        return format(value, "f").split(".")[0]
    return format(value, "f").rstrip("0").rstrip(".")


@dataclass(frozen=True)
class InterpolationResult:
    values: tuple[float, ...]
    metadata: dict[str, Any]
    artifact_fingerprint: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "metadata": self.metadata,
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InterpolationResult":
        return cls(
            values=tuple(float(value) for value in payload["values"]),
            metadata=dict(payload["metadata"]),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
        )


def interpolate_series(series: SeriesRecord) -> InterpolationResult:
    """Interpolate missing values using only observations in this series.

    Interior gaps use linear interpolation between the immediately adjacent
    observed indices.  Leading/trailing gaps copy the nearest edge observation.
    Distance ties (only possible in nearest-edge mode) select the lower index.
    All reported imputed values are rounded to 12 decimal places, ties away
    from zero.  Quantization to the fixed-point lattice repeats the same tie
    rule.
    """

    observed = series.observed_indices
    if not observed:
        raise ValueError("series has no observed values")
    values = list(series.values)
    imputed: list[dict[str, Any]] = []
    with localcontext() as context:
        context.prec = 60
        for index, value in enumerate(series.values):
            if value is not None:
                continue
            left_candidates = [item for item in observed if item < index]
            right_candidates = [item for item in observed if item > index]
            left = left_candidates[-1] if left_candidates else None
            right = right_candidates[0] if right_candidates else None
            if left is not None and right is not None:
                left_value = decimal_from_float(series.values[left])  # type: ignore[arg-type]
                right_value = decimal_from_float(series.values[right])  # type: ignore[arg-type]
                ratio = Decimal(index - left) / Decimal(right - left)
                raw = left_value + ratio * (right_value - left_value)
                rounded = round_decimal_half_away(raw, INTERPOLATION_DECIMALS)
                values[index] = float(rounded)
                imputed.append(
                    {
                        "index": index,
                        "rule": "inside_linear",
                        "left_index": left,
                        "right_index": right,
                        "fraction_numerator": index - left,
                        "fraction_denominator": right - left,
                        "raw_value": decimal_text(raw),
                        "value": decimal_text(rounded),
                    }
                )
            else:
                edge = left if left is not None else right
                if edge is None:
                    raise ValueError(f"cannot interpolate index {index}")
                edge_value = decimal_from_float(series.values[edge])  # type: ignore[arg-type]
                rounded = round_decimal_half_away(edge_value, INTERPOLATION_DECIMALS)
                values[index] = float(rounded)
                imputed.append(
                    {
                        "index": index,
                        "rule": "edge_nearest",
                        "source_index": edge,
                        "source_side": "left" if edge < index else "right",
                        "distance": abs(index - edge),
                        "value": decimal_text(rounded),
                    }
                )

    metadata = {
        "method": "inside_linear_nearest_edge",
        "window": "entire_series_only",
        "uses_only_observed_in_window": True,
        "observed_indices": list(observed),
        "missing_indices": list(series.missing_indices),
        "imputed": imputed,
        "rounding": "nearest_half_away_from_zero",
        "rounding_decimals": INTERPOLATION_DECIMALS,
        "tie_breaks": {
            "interior": "linear_between_bracketing_observed",
            "nearest_distance_equal": "lower_index_left",
            "rounding_midpoint": "away_from_zero",
        },
        "referenced_observations": referenced_observations(series),
    }
    artifact_fp = fingerprint(
        {
            "values": list(values),
            "metadata": metadata,
        }
    )
    return InterpolationResult(
        values=tuple(values),
        metadata=metadata,
        artifact_fingerprint=artifact_fp,
    )
