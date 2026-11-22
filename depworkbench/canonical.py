from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def quantize_decimal(value: float | int, fixed_point_decimals: int, quantization_level: int) -> int:
    """Quantize a finite real into an integer score domain.

    ``10**fixed_point_decimals / quantization_level`` is the effective scale.
    Ties are rounded away from zero, which keeps negative and positive scores
    symmetric and independent of the platform's float formatting.
    """

    decimal_value = Decimal(str(value))
    scale = (Decimal(10) ** fixed_point_decimals) / Decimal(quantization_level)
    return int((decimal_value * scale).quantize(Decimal("1")))


def quantization_fingerprint(fixed_point_decimals: int, quantization_level: int) -> str:
    return sha256_json(
        {
            "fixed_point_decimals": fixed_point_decimals,
            "quantization_level": quantization_level,
        }
    )


def stable_task_key(sentence_id: str, step_index: int, stage: str) -> str:
    return f"{sentence_id}\x1f{step_index:08d}\x1f{stage}"

