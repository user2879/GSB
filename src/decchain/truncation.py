"""Deterministic TopK + nucleus truncation and boundary diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DecodeConfig
from .kernel import KernelResult


@dataclass
class BoundaryToken:
    token: int
    position: int                       # 1-based position in the sorted prefix
    working_int: int
    cumulative_before: int
    cumulative_after: int
    target_int: int
    decision: str                       # EXACT / HALF_STEP_INCLUDE / OVERSHOOT / AT_TARGET_EXCLUDE
    ref_int_scaled: int                 # higher-width reference rescaled to W
    rounding_diff: int                  # reference-vs-working at same token
    accum_diff: int                     # error attributable to accumulation order
    attribution: str                    # TEMP_ROUNDING / ACCUM_ORDER / BOTH / NONE


@dataclass
class TruncationResult:
    kept_tokens: list[int]
    sorted_tokens: list[int]            # full deterministic ordering
    topk_tokens: list[int]              # after TopK only
    target_int: int
    half_step: int
    boundaries: list[BoundaryToken] = field(default_factory=list)
    kept_quant: dict[int, int] = field(default_factory=dict)
    dropped_quant: dict[int, int] = field(default_factory=dict)

    @property
    def kept_total(self) -> int:
        return sum(self.kept_quant.values())


def _order(kr: KernelResult) -> list[int]:
    # Fixed-point descending, ties broken by ascending token id.
    return sorted(kr.tokens, key=lambda t: (-kr.quant[t], t))


def truncate(kr: KernelResult, config: DecodeConfig) -> TruncationResult:
    ordered = _order(kr)
    topk = ordered[: min(config.top_k, len(ordered))]

    target = round(config.top_p_mille * kr.scale / 1000)
    half = config.half_step

    cumulative = 0
    kept: list[int] = []
    boundary: BoundaryToken | None = None

    for pos, token in enumerate(topk, start=1):
        value = kr.quant[token]
        before = cumulative
        after = cumulative + value

        if pos == 1:
            # Always keep at least one token; it is only a boundary if it
            # already reaches the threshold by itself.
            kept.append(token)
            cumulative = after
            if after >= target:
                decision = ("EXACT" if after == target
                            else ("HALF_STEP_INCLUDE"
                                  if after - target <= half else "OVERSHOOT"))
                boundary = _build_boundary(
                    kr, token, pos, value, before, after, target, decision)
                break
            continue

        if cumulative == target:
            boundary = _build_boundary(
                kr, token, pos, value, before, after, target,
                "AT_TARGET_EXCLUDE")
            break

        if after <= target:
            kept.append(token)
            cumulative = after
            if after == target:
                boundary = _build_boundary(
                    kr, token, pos, value, before, after, target, "EXACT")
                break
            continue

        # Crossing the threshold: the half-step rule decides inclusion.
        overshoot = after - target
        decision = ("HALF_STEP_INCLUDE" if overshoot <= half
                    else "OVERSHOOT")
        kept.append(token)
        cumulative = after
        boundary = _build_boundary(
            kr, token, pos, value, before, after, target, decision)
        break

    kept_set = set(kept)
    kept_quant = {t: kr.quant[t] for t in kept}
    dropped_quant = {t: kr.quant[t] for t in topk if t not in kept_set}
    boundaries = [boundary] if boundary is not None else []
    return TruncationResult(
        kept_tokens=kept,
        sorted_tokens=ordered,
        topk_tokens=topk,
        target_int=target,
        half_step=half,
        boundaries=boundaries,
        kept_quant=kept_quant,
        dropped_quant=dropped_quant,
    )


def _build_boundary(kr: KernelResult, token: int, position: int,
                    value: int, before: int, after: int,
                    target: int, decision: str) -> BoundaryToken:
    # Rescale the one-digit-wider reference integer down to the working width
    # (half-up), so both quantities live on the same scale.
    ref_raw = kr.ref_quant[token]
    ref_scaled = (ref_raw + 5) // 10
    rounding_diff = ref_scaled - value

    # Cumulative-order component: signed distance of the prefix mass from the
    # threshold at the boundary.
    accum_diff = after - target

    if rounding_diff and accum_diff:
        attribution = "BOTH"
    elif rounding_diff:
        attribution = "TEMP_ROUNDING"
    elif accum_diff:
        attribution = "ACCUM_ORDER"
    else:
        attribution = "NONE"

    return BoundaryToken(
        token=token,
        position=position,
        working_int=value,
        cumulative_before=before,
        cumulative_after=after,
        target_int=target,
        decision=decision,
        ref_int_scaled=ref_scaled,
        rounding_diff=rounding_diff,
        accum_diff=accum_diff,
        attribution=attribution,
    )
