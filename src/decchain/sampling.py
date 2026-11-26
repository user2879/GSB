"""Deterministic integer-domain sampling over a truncated distribution."""

from __future__ import annotations

from dataclasses import dataclass, field

from .kernel import KernelResult
from .rng import DeterministicRng, derive_seed
from .truncation import TruncationResult


@dataclass
class SamplingConclusion:
    request_id: str
    seq_id: int
    step_index: int
    selected_token: int
    rng_unit: float
    rng_target: int                     # integer target within kept mass
    kept_total: int
    cumulative_intervals: dict[int, tuple[int, int]]  # token -> [lo, hi)
    tie_tokens: list[int]
    basis: str
    seed: str
    content_fingerprint: str
    config_fingerprint: str
    kept_tokens: list[int]
    kept_quant: dict[int, int]
    residuals: dict[int, int]
    ref_quant: dict[int, int]
    boundary_tokens: list[int]

    def cache_key(self) -> str:
        # Stable key covering every input that can change the conclusion.
        mat = "|".join([
            self.content_fingerprint,
            self.config_fingerprint,
            str(self.seq_id),
            str(self.step_index),
        ])
        import hashlib
        return hashlib.sha256(mat.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "seq_id": self.seq_id,
            "step_index": self.step_index,
            "selected_token": self.selected_token,
            "rng_unit": self.rng_unit,
            "rng_target": self.rng_target,
            "kept_total": self.kept_total,
            "cumulative_intervals": {
                str(tok): list(iv)
                for tok, iv in self.cumulative_intervals.items()
            },
            "tie_tokens": self.tie_tokens,
            "basis": self.basis,
            "seed": self.seed,
            "content_fingerprint": self.content_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "kept_tokens": list(self.kept_tokens),
            "kept_quant": {str(k): v for k, v in self.kept_quant.items()},
            "residuals": {str(k): v for k, v in self.residuals.items()},
            "ref_quant": {str(k): v for k, v in self.ref_quant.items()},
            "boundary_tokens": list(self.boundary_tokens),
            "cache_key": self.cache_key(),
        }


def sample(request_id: str, seq_id: int, step_index: int,
           content_fp: str, config_fp: str,
           kr: KernelResult, tr: TruncationResult) -> SamplingConclusion:
    kept = list(tr.kept_tokens)  # already deterministic (quant desc, id asc)
    kept_quant = {t: kr.quant[t] for t in kept}
    residuals = {t: kr.record_for(t).residual for t in kept}
    ref_quant = {t: kr.ref_quant[t] for t in kept}
    total = sum(kept_quant.values())

    seed = derive_seed(content_fp, seq_id, step_index, config_fp)
    rng = DeterministicRng(seed)
    unit = rng.next_unit()
    target = int(unit * total)          # integer draw in [0, total)
    if target >= total:                 # guard for the float boundary
        target = total - 1

    intervals: dict[int, tuple[int, int]] = {}
    cumulative = 0
    selected = kept[-1]
    for token in kept:
        lo = cumulative
        hi = cumulative + kept_quant[token]
        intervals[token] = (lo, hi)
        if lo <= target < hi:
            selected = token
        cumulative = hi

    # Tie detection: tokens sharing the selected token's integer mass are
    # resolved by the deterministic ordering (ascending id among equals),
    # which is already the order inside ``kept``.
    tie_tokens = sorted(t for t in kept
                        if kept_quant[t] == kept_quant[selected])

    basis = (f"target={target} of kept_total={total}; token {selected} "
             f"interval {intervals[selected]}; ties resolved by ascending id")

    return SamplingConclusion(
        request_id=request_id,
        seq_id=seq_id,
        step_index=step_index,
        selected_token=selected,
        rng_unit=unit,
        rng_target=target,
        kept_total=total,
        cumulative_intervals=intervals,
        tie_tokens=tie_tokens,
        basis=basis,
        seed=seed,
        content_fingerprint=content_fp,
        config_fingerprint=config_fp,
        kept_tokens=kept,
        kept_quant=kept_quant,
        residuals=residuals,
        ref_quant=ref_quant,
        boundary_tokens=[b.token for b in tr.boundaries],
    )
