"""Shared constants, business codes, fixed-point arithmetic and deterministic IO.

Everything here is pure-Python and platform independent so that two runs over
the same inputs always produce byte-identical outputs.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

# --- Business codes (also mapped through config.business_codes at output) ---
CODE_EVENT_INVALID = "E_EVENT_INVALID"
CODE_DEF_CYCLE = "E_DEF_CYCLE"
CODE_DEF_PARAM = "E_DEF_PARAM"
CODE_BACKFILL_ORDER = "E_BACKFILL_ORDER"
CODE_SAMPLE_PENDING = "E_SAMPLE_PENDING"
CODE_FEAT_BUDGET_EXHAUSTED = "E_FEAT_BUDGET_EXHAUSTED"
CODE_GLOBAL_BUDGET_EXHAUSTED = "E_GLOBAL_BUDGET_EXHAUSTED"
CODE_INCREMENT_MISMATCH = "E_INCREMENT_MISMATCH"
CODE_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
CODE_INJECTION_HIT = "E_INJECTION_HIT"

ALL_CODES = [
    CODE_EVENT_INVALID,
    CODE_DEF_CYCLE,
    CODE_DEF_PARAM,
    CODE_BACKFILL_ORDER,
    CODE_SAMPLE_PENDING,
    CODE_FEAT_BUDGET_EXHAUSTED,
    CODE_GLOBAL_BUDGET_EXHAUSTED,
    CODE_INCREMENT_MISMATCH,
    CODE_SNAPSHOT_STALE,
    CODE_INJECTION_HIT,
]

AGG_COUNT = "count"
AGG_SUM = "sum"
AGG_MEAN = "mean"
AGG_LAST = "last"
AGG_FUNCTIONS = (AGG_COUNT, AGG_SUM, AGG_MEAN, AGG_LAST)

SIDE_USER = "user"
SIDE_ITEM = "item"
SIDES = (SIDE_USER, SIDE_ITEM)

STAGE_LOAD = "load"
STAGE_FEATURE = "feature"
STAGE_SAMPLE = "sample"
STAGES = (STAGE_LOAD, STAGE_FEATURE, STAGE_SAMPLE)


@dataclass(frozen=True)
class Failure:
    """One failure-list entry. Stable ordering is by (stage, target, code, detail)."""

    stage: str
    target: str          # event id / feature name / sample id / injection id
    code: str
    detail: str = ""
    injection_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_dumps(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, fixed separators, no ascii escaping."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical_dumps(obj))
        fh.write("\n")
    os.replace(tmp, path)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(canonical_dumps(row))
            fh.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fixed-point quantization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedPoint:
    """Quantization rule: quantized = round(raw / step) then scaled by 10**scale."""

    scale: int          # number of fractional digits kept
    step: Decimal       # quantization step (raw units), strictly positive

    @classmethod
    def build(cls, scale: int, quant_step: str | float | int) -> "FixedPoint":
        if not isinstance(scale, int) or scale < 0 or scale > 18:
            raise ValueError("fixed_point_scale must be an integer in [0, 18]")
        step = Decimal(str(quant_step))
        if not step.is_finite() or step <= 0:
            raise ValueError("quant_step must be a positive finite number")
        return cls(scale=scale, step=step)

    def quantize(self, raw: Decimal | int) -> int:
        """Return a fixed-point integer: scaled value of the quantized raw number."""
        raw = Decimal(raw) if not isinstance(raw, Decimal) else raw
        # Half-up rounding on the integer number of quantization steps.
        steps = (raw / self.step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantized = steps * self.step
        scaled = (quantized * (Decimal(10) ** self.scale)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        return int(scaled)

    def empty_value(self, configured: Any) -> int:
        if configured is None:
            return 0
        return self.quantize(Decimal(str(configured)))


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)
