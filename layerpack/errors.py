"""Stable business error codes and failure records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_WEIGHT_SHAPE_MISMATCH = "E_WEIGHT_SHAPE_MISMATCH"
E_NONFINITE_WEIGHT = "E_NONFINITE_WEIGHT"
E_CALIB_MISSING = "E_CALIB_MISSING"
E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"

KNOWN_CODES = frozenset(
    {
        E_CONFIG_INVALID,
        E_WEIGHT_SHAPE_MISMATCH,
        E_NONFINITE_WEIGHT,
        E_CALIB_MISSING,
        E_INTERACT_BUDGET_PENDING,
        E_POINT_BUDGET_PENDING,
    }
)

STAGE_ORDER = {
    "loading": 0,
    "quantization": 1,
    "calibration": 2,
    "deployment": 3,
    "interaction": 4,
}


@dataclass(frozen=True)
class Failure:
    """One deterministic entry in the failure list."""

    code: str
    stage: str
    layer: str | None
    message: str
    detail: dict[str, Any] | None = None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            STAGE_ORDER.get(self.stage, 99),
            self.layer or "",
            self.code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if value["detail"] is None:
            value.pop("detail")
        return value


class LayerPackError(ValueError):
    """An input error carrying a stable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "loading",
        layer: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code not in KNOWN_CODES:
            raise ValueError(f"unknown business code: {code}")
        self.failure = Failure(code, stage, layer, message, detail)

    @property
    def code(self) -> str:
        return self.failure.code


def failure(
    code: str,
    message: str,
    *,
    stage: str = "loading",
    layer: str | None = None,
    detail: dict[str, Any] | None = None,
) -> Failure:
    return Failure(code, stage, layer, message, detail)


def sort_failures(failures: Iterable[Failure]) -> list[Failure]:
    """Return failures in the public, field-stable order."""

    # A duplicate issue can be produced by both the loader and config validator.
    # Keeping the first occurrence avoids noisy diagnostics while retaining order.
    seen: set[tuple[Any, ...]] = set()
    ordered: list[Failure] = []
    for item in sorted(failures, key=Failure.sort_key):
        key = (item.code, item.stage, item.layer, item.message)
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered
