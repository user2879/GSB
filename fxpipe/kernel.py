"""Deterministic fixed-point integer kernels.

All arithmetic uses Python integers (arbitrary precision) with explicit q-format
scaling, so results are identical regardless of host word size or thread count.
The fractional bit width ``f`` defines the unit 2**f; every multiplication is
followed by an arithmetic right shift of ``f`` bits and saturation to the
configured integer container width.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .config import GEAR_SHIFTS, PipelineConfig


def isqrt(value: int) -> int:
    """Floor integer square root for non-negative integers."""
    if value < 0:
        raise ValueError("isqrt requires a non-negative value")
    if value == 0:
        return 0
    x = value
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + value // x) // 2
    return x


@dataclass(frozen=True)
class QuantContext:
    frac_bits: int
    gear: str
    width: int
    scale_permille: int

    @property
    def unit(self) -> int:
        return 1 << self.frac_bits

    @property
    def limit(self) -> int:
        return 1 << (self.width - 1)

    def saturate(self, value: int) -> int:
        bound = self.limit - 1
        if value > bound:
            return bound
        if value < -bound - 1:
            return -bound - 1
        return value

    def qmul(self, lhs: int, rhs: int) -> int:
        # Rounded fixed-point multiply; the rounding term depends on the
        # fractional bit width so different frac_bits genuinely change values.
        numerator = lhs * rhs
        if numerator >= 0:
            raw = (numerator + (1 << (self.frac_bits - 1))) >> self.frac_bits
        else:
            raw = -(((-numerator) + (1 << (self.frac_bits - 1))) >> self.frac_bits)
        return self.saturate(raw)

    def gear_scale(self, value: int) -> int:
        numerator, denominator = GEAR_SHIFTS[self.gear]
        if numerator == denominator:
            return self.saturate(value)
        scaled = (value * numerator) // denominator
        return self.saturate(scaled)


def make_context(config: PipelineConfig, width: int, permille: int) -> QuantContext:
    return QuantContext(
        frac_bits=config.frac_bits,
        gear=config.gear,
        width=width,
        scale_permille=permille,
    )


# --------------------------------------------------------------------------- #
# Deterministic synthetic helpers
# --------------------------------------------------------------------------- #
def deterministic_int(seed: str, span: int) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % span


def attention_weights(node_id: str, head: int, token_index: int, length: int) -> list[int]:
    """Integer per-mille weights for one (head, token) query, summing to 1000."""
    raw = []
    for key_index in range(length):
        seed = f"{node_id}|h{head}|q{token_index}|k{key_index}"
        raw.append(deterministic_int(seed, 100_000) + 1)
    total = sum(raw)
    weights = [value * 1000 // total for value in raw]
    remainder = 1000 - sum(weights)
    order = sorted(range(length), key=lambda idx: ((raw[idx] * 1000) % total, -idx), reverse=True)
    cursor = 0
    while remainder > 0:
        weights[order[cursor % length]] += 1
        remainder -= 1
        cursor += 1
    return weights


# --------------------------------------------------------------------------- #
# Stage kernels. Each accepts ascending-order inputs and keeps fixed order.
# --------------------------------------------------------------------------- #
def embed_lookup(token: int, table_row: list[int], ctx: QuantContext) -> list[int]:
    """Element-wise, token index ascending; gear + frac_bits affect values."""
    scaled = [ctx.gear_scale(value) for value in table_row]
    # The fractional LSB (unit//16) folds the configured frac_bits into the
    # integer embedding, guaranteeing distinct intermediates per frac width.
    frac_lsb = max(1, ctx.unit // 16)
    factor = ctx.unit * ctx.scale_permille // 1000
    out = []
    for value in scaled:
        adjusted = ctx.saturate(value + (frac_lsb if value >= 0 else -frac_lsb))
        out.append(ctx.qmul(adjusted, factor))
    return out


def segment_pool(rows: list[list[int]], ctx: QuantContext) -> dict:
    """Sum pooled rows in token ascending order, record valid token count."""
    dim = len(rows[0])
    accumulator = [0] * dim
    for row in rows:  # rows are already ordered by ascending token index
        for element_index in range(dim):
            accumulator[element_index] = ctx.saturate(accumulator[element_index] + row[element_index])
    return {"sum": accumulator, "count": len(rows)}


def multihead_attention(
    node_id: str,
    rows: list[list[int]],
    heads: int,
    ctx: QuantContext,
) -> dict:
    """Heads ascending, dim ascending; weighted value accumulation in q-format."""
    length = len(rows)
    dim = len(rows[0])
    head_dim = dim // heads
    per_head: list[list[list[int]]] = []
    vectors: list[list[int]] = [[0] * dim for _ in range(length)]
    for head in range(heads):  # head index ascending
        head_rows: list[list[int]] = []
        for query_index in range(length):
            weights = attention_weights(node_id, head, query_index, length)
            out = [0] * head_dim
            for dim_index in range(head_dim):  # dimension index ascending
                acc = 0
                for key_index in range(length):
                    value = rows[key_index][head * head_dim + dim_index]
                    product = ctx.qmul(weights[key_index], value)
                    acc = ctx.saturate(acc + product)
                out[dim_index] = ctx.saturate(acc)
            head_rows.append(out)
            for dim_index, value in enumerate(out):
                vectors[query_index][head * head_dim + dim_index] = value
        per_head.append(head_rows)
    return {"heads": per_head, "vectors": vectors}


def layer_norm(rows: list[list[int]], ctx: QuantContext, epsilon: int | None = None) -> dict:
    """Row-wise fixed accumulation order for mean, variance and normalization."""
    epsilon = epsilon if epsilon is not None else max(1, ctx.unit // 16)
    normalized: list[list[int]] = []
    means: list[int] = []
    variances: list[int] = []
    for row in rows:
        total = 0
        for value in row:  # fixed ascending accumulation order
            total += value
        mean = total // len(row)
        variance_acc = 0
        for value in row:
            delta = value - mean
            variance_acc += (delta * delta) >> ctx.frac_bits
        variance = variance_acc // len(row)
        std = isqrt(variance + epsilon)
        out = []
        for value in row:
            centered = value - mean
            scaled = (centered * ctx.unit) // (std if std > 0 else 1)
            out.append(ctx.saturate(scaled))
        normalized.append(out)
        means.append(mean)
        variances.append(variance)
    return {"rows": normalized, "means": means, "variances": variances}
