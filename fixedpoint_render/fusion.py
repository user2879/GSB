"""Condition weight normalization and fixed point fusion (business rule 2)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CONDITION_TYPES, RenderConfig
from .fixedpoint import (
    QuantTrace, q_div_round, q_mul, q_scale,
)
from .encoding import EncodedCondition
from .fingerprints import fingerprint


@dataclass
class FusionResult:
    frame: int
    weights_raw: dict[str, int]          # configured weights in Q
    weights_norm: dict[str, int]         # normalized weights in Q, sum == scale
    vector: list[int]                    # fused condition vector in Q
    traces: list[QuantTrace] = field(default_factory=list)

    @property
    def content_id(self) -> str:
        # The fused vector already fully encodes the normalized weights and
        # typed inputs; keying on it plus the frame keeps the id stable across
        # serialize/rehydrate cycles.
        return fingerprint({"frame": self.frame, "vector": self.vector},
                           f"fusion-id-f{self.frame}")

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "weights_raw": self.weights_raw,
            "weights_norm": self.weights_norm,
            "vector": self.vector,
            "content_id": self.content_id,
            "traces": [t.to_dict() for t in self.traces],
        }


def normalize_weights(config: RenderConfig) -> tuple[dict[str, int], list[QuantTrace]]:
    """Normalize the three configured weights so they sum to 1.0 (== scale)."""
    scale = q_scale(config.frac_bits)
    raw = {name: config.weight_q(name) for name in CONDITION_TYPES}
    total = sum(raw.values())
    norm: dict[str, int] = {}
    traces: list[QuantTrace] = []
    if total == 0:
        # Degenerate config: uniform fallback, still fully deterministic.
        share = q_div_round(scale, len(CONDITION_TYPES), config.qmode)
        for name in CONDITION_TYPES:
            norm[name] = share
        # Correct the last component so the sum is exact.
        norm[CONDITION_TYPES[-1]] += scale - sum(norm.values())
        return norm, traces
    assigned = 0
    ordered = sorted(CONDITION_TYPES)  # condition type name ascending
    for order, name in enumerate(ordered):
        if order < len(ordered) - 1:
            value = q_div_round(raw[name] * scale, total, config.qmode)
            residual = raw[name] * scale - value * total
            traces.append(QuantTrace(f"weight_norm[{name}]",
                                     config.frac_bits, config.frac_bits,
                                     config.qmode, total, raw[name] * scale,
                                     value, residual))
            norm[name] = value
            assigned += value
        else:
            # Last component absorbs rounding so the normalized sum is exact.
            value = scale - assigned
            residual = raw[name] * scale - value * total
            traces.append(QuantTrace(f"weight_norm[{name}]",
                                     config.frac_bits, config.frac_bits,
                                     config.qmode, total, raw[name] * scale,
                                     value, residual))
            norm[name] = value
    return norm, traces


def fuse_conditions(frame: int, encoded: dict[str, EncodedCondition],
                    config: RenderConfig) -> FusionResult:
    """Weighted accumulation.

    Iteration order is fixed: condition type name ascending then dimension
    index ascending.  Every multiply is a real fixed point operation whose
    integer result and residual are recorded, so changing a weight, frac width
    or qmode changes the actual intermediate integers, not just their display.
    """
    weights_norm, norm_traces = normalize_weights(config)
    scale = q_scale(config.frac_bits)
    dim = config.condition_dim
    accum = [0] * dim
    traces: list[QuantTrace] = list(norm_traces)
    for name in sorted(CONDITION_TYPES):
        cond = encoded[name]
        weight = weights_norm[name]
        for d in range(dim):  # dimension index ascending
            product, trace = q_mul(weight, cond.vector[d],
                                   config.frac_bits, config.qmode,
                                   op=f"fuse:{name}[{d}]")
            traces.append(trace)
            accum[d] += product
            traces.append(QuantTrace(f"accum:{name}[{d}]",
                                     config.frac_bits, config.frac_bits,
                                     config.qmode, 1, accum[d] - product,
                                     accum[d], 0))
    return FusionResult(frame=frame, weights_raw={n: config.weight_q(n)
                                                  for n in CONDITION_TYPES},
                        weights_norm=weights_norm, vector=accum, traces=traces)
