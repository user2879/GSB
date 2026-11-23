"""Deterministic block-level fixed-point scoring.

Accumulation order is fixed: query row ascending, then key column ascending,
then embedding dimension ascending. Saturation and rounding are implemented in
pure Python (with Decimal for the final rounding) so results are bit-identical
across runs, platforms and thread completion order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_DOWN, Decimal, getcontext
from typing import Any

import numpy as np

from .util import short_hash

getcontext().prec = 50


@dataclass
class OverflowRecord:
    qb: int
    kb: int
    row: int
    col: int
    dim: int
    raw: int
    clamped: int


@dataclass
class BlockPairScore:
    qb: int
    kb: int
    accumulators: list[list[int]]
    scaled_rows: list[float]
    block_score: float
    scale_factor: float
    temperature: float
    rounding: str
    saturation: str
    frac_bits: int
    acc_bits: int
    overflows: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qb": self.qb,
            "kb": self.kb,
            "accumulators": self.accumulators,
            "scaled_rows": self.scaled_rows,
            "block_score": self.block_score,
            "scale_factor": self.scale_factor,
            "temperature": self.temperature,
            "rounding": self.rounding,
            "saturation": self.saturation,
            "frac_bits": self.frac_bits,
            "acc_bits": self.acc_bits,
            "overflows": self.overflows,
            "fingerprint": self.fingerprint,
        }


def signed_range(bits: int) -> tuple[int, int]:
    limit = 1 << (bits - 1)
    return -limit, limit - 1


def quantize(values: np.ndarray, frac_bits: int, rounding: str) -> np.ndarray:
    scale = float(1 << frac_bits)
    out = np.empty(values.shape, dtype=np.int64)
    flat_in = values.reshape(-1)
    flat_out = out.reshape(-1)
    for idx, value in enumerate(flat_in):
        flat_out[idx] = _round_decimal(Decimal(repr(float(value))) * Decimal(scale), rounding)
    q_lo, q_hi = signed_range(frac_bits)
    np.clip(out, q_lo, q_hi, out=out)
    return out


def _round_decimal(value: Decimal, rounding: str) -> int:
    if rounding == "truncate":
        return int(value.to_integral_value(rounding=ROUND_DOWN))
    if rounding == "nearest_even":
        return int(value.to_integral_value(rounding=ROUND_HALF_EVEN))
    if rounding == "nearest_up":
        return int(value.to_integral_value(rounding=ROUND_HALF_UP))
    if rounding == "floor":
        return int(value.to_integral_value(rounding=ROUND_FLOOR))
    raise ValueError(f"unknown rounding {rounding!r}")


def _sat_add(current: int, product: int, bits: int, saturation: str) -> tuple[int, bool, int]:
    raw = current + product
    lo, hi = signed_range(bits)
    if lo <= raw <= hi:
        return raw, False, raw
    if saturation == "wrap":
        span = 1 << bits
        wrapped = ((raw - lo) % span) + lo
        return wrapped, True, raw
    clamped = max(lo, min(hi, raw))
    return clamped, True, raw


def deterministic_round_float(value: float, rounding: str) -> float:
    return float(_round_decimal(Decimal(repr(value)), rounding))


def score_block_pair(
    qb: int,
    kb: int,
    query_block: np.ndarray,
    key_block: np.ndarray,
    frac_bits: int,
    acc_bits: int,
    saturation: str,
    rounding: str,
    temperature: float,
    quant_q: np.ndarray | None = None,
    quant_k: np.ndarray | None = None,
    injected_scale: float | None = None,
    force_overflow: bool = False,
) -> BlockPairScore:
    rows, dim = query_block.shape
    cols = key_block.shape[0]
    if quant_q is None:
        quant_q = quantize(query_block, frac_bits, rounding)
    if quant_k is None:
        quant_k = quantize(key_block, frac_bits, rounding)

    acc: list[list[int]] = [[0] * cols for _ in range(rows)]
    overflows: list[OverflowRecord] = []
    for i in range(rows):
        for j in range(cols):
            total = 0
            for d in range(dim):
                product = int(quant_q[i, d]) * int(quant_k[j, d])
                if force_overflow and i == 0 and j == 0 and d == 0:
                    lo, hi = signed_range(acc_bits)
                    raw = total + product
                    total = hi if product >= 0 else lo
                    overflows.append(OverflowRecord(qb, kb, i, j, d, raw, total))
                    continue
                total, did_overflow, raw = _sat_add(total, product, acc_bits, saturation)
                if did_overflow:
                    overflows.append(OverflowRecord(qb, kb, i, j, d, raw, total))
            acc[i][j] = total

    base_scale = 1.0 / float(1 << (2 * frac_bits))
    scale_factor = base_scale if injected_scale is not None else base_scale / temperature
    if injected_scale is not None:
        scale_factor = float(injected_scale)
        if not math.isfinite(scale_factor):
            raise FloatingPointError(f"non-finite injected scale for qb={qb} kb={kb}")

    scaled_rows: list[float] = []
    for i in range(rows):
        row_sum = 0.0
        for j in range(cols):
            raw_float = float(acc[i][j]) * scale_factor
            row_sum += deterministic_round_float(raw_float, rounding)
        scaled_rows.append(row_sum)
    block_score = math.fsum(scaled_rows)
    fingerprint = _fingerprint(qb, kb, acc, scale_factor, rounding)
    return BlockPairScore(
        qb=qb, kb=kb, accumulators=acc, scaled_rows=scaled_rows,
        block_score=block_score, scale_factor=scale_factor,
        temperature=temperature, rounding=rounding, saturation=saturation,
        frac_bits=frac_bits, acc_bits=acc_bits,
        overflows=[vars(item) for item in overflows], fingerprint=fingerprint,
    )


def _fingerprint(qb: int, kb: int, acc: list[list[int]], scale_factor: float,
                 rounding: str) -> str:
    payload = repr((qb, kb, acc, repr(scale_factor), rounding)).encode("utf-8")
    return short_hash(payload)
