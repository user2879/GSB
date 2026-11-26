"""Temperature scaling, stable softmax and deterministic fixed-point kernel.

Pipeline (all in the integer domain once quantized):

    logits            finite, non-positive log probabilities
    scaled = logits / temperature
    stable softmax    subtract max, exponentiate, normalize
    quantize          map probabilities in [0,1] to integers summing to scale

The kernel records, for every token, the quantization step, raw integer,
final integer and the integer residual applied so the ledger is fully
reproducible.  A one-digit-wider unquantized reference distribution is
produced alongside for error attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import DecodeConfig


@dataclass
class TokenQuant:
    token: int
    scaled_logit: float
    probability: float
    raw_int: int
    final_int: int
    residual: int          # final_int - raw_int (integer units)


@dataclass
class KernelResult:
    width: int
    tier: str
    scale: int
    tokens: list[int]
    prob: dict[int, float]
    quant: dict[int, int]                 # final fixed-point integers
    records: list[TokenQuant]
    ref_width: int
    ref_scale: int
    ref_prob: dict[int, float]
    ref_quant: dict[int, int]             # one-digit-wider reference integers
    total_int: int

    def record_for(self, token: int) -> TokenQuant:
        for rec in self.records:
            if rec.token == token:
                return rec
        raise KeyError(token)


def stable_softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    shifted = arr - np.max(arr)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _quantize(probs: np.ndarray, scale: int, tier: str) -> tuple[list[int], list[int]]:
    """Return (raw integers, deterministic residual corrections).

    Final integers are raw integers plus residuals and sum exactly to scale.
    """
    exact = probs * scale
    if tier == "FLOOR":
        raw = [int(math.floor(x)) for x in exact]
    else:  # ROUND_HALF_UP (default) and REF use half-up
        raw = [int(math.floor(x + 0.5)) for x in exact]

    deficit = scale - sum(raw)
    residual = [0] * len(raw)
    if deficit > 0:
        # Distribute the missing units to the largest fractional remainders.
        if tier == "FLOOR":
            remainders = [exact[i] - raw[i] for i in range(len(raw))]
        else:
            remainders = [exact[i] - (raw[i] - 0.5) for i in range(len(raw))]
        # Deterministic ordering: biggest remainder, then lowest array index.
        order = sorted(range(len(raw)),
                       key=lambda i: (-remainders[i], i))
        for j in range(deficit):
            residual[order[j % len(order)]] += 1
    elif deficit < 0:
        # Remove surplus units from the smallest fractional remainders.
        if tier == "FLOOR":
            remainders = [exact[i] - raw[i] for i in range(len(raw))]
        else:
            remainders = [exact[i] - (raw[i] - 0.5) for i in range(len(raw))]
        order = sorted(range(len(raw)),
                       key=lambda i: (remainders[i], i))
        for j in range(-deficit):
            residual[order[j % len(order)]] -= 1
    return raw, residual


def run_kernel(tokens: Sequence[int], logprobs: Sequence[float],
               temperature: float, config: DecodeConfig) -> KernelResult:
    if len(tokens) != len(logprobs):
        raise ValueError("tokens and logprobs length mismatch")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")

    token_list = [int(t) for t in tokens]
    lp = np.asarray(logprobs, dtype=np.float64)
    if not np.all(np.isfinite(lp)):
        raise ValueError("all logprobs must be finite")

    scaled = lp / float(temperature)
    probs = stable_softmax(scaled)

    scale = config.scale
    raw, residual = _quantize(probs, scale, config.tier)
    final = [raw[i] + residual[i] for i in range(len(token_list))]

    records = [
        TokenQuant(
            token=token_list[i],
            scaled_logit=float(scaled[i]),
            probability=float(probs[i]),
            raw_int=raw[i],
            final_int=final[i],
            residual=residual[i],
        )
        for i in range(len(token_list))
    ]

    # One-digit-wider reference path (always half-up, no tier-specific rounding).
    ref_width = config.ref_width
    ref_scale = 10 ** ref_width
    ref_raw, ref_res = _quantize(probs, ref_scale, "ROUND_HALF_UP")
    ref_final = [ref_raw[i] + ref_res[i] for i in range(len(token_list))]

    return KernelResult(
        width=config.width,
        tier=config.tier,
        scale=scale,
        tokens=token_list,
        prob={token_list[i]: float(probs[i]) for i in range(len(token_list))},
        quant={token_list[i]: final[i] for i in range(len(token_list))},
        records=records,
        ref_width=ref_width,
        ref_scale=ref_scale,
        ref_prob={token_list[i]: float(probs[i]) for i in range(len(token_list))},
        ref_quant={token_list[i]: ref_final[i]
                   for i in range(len(token_list))},
        total_int=sum(final),
    )
