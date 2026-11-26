"""Stage-wise error attribution against a wider unquantized reference."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .config import DecodeConfig
from .kernel import run_kernel
from .truncation import truncate
from .rng import DeterministicRng, derive_seed
from .models import Request, CandidateSequence, Step


@dataclass
class StageDiff:
    stage: str
    max_abs_diff: float
    first_offending_token: int | None
    within_tolerance: bool


@dataclass
class AttributionRow:
    request_id: str
    seq_id: int
    step_index: int
    stages: list[StageDiff]
    first_breach_stage: str | None
    tolerance_units: int

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "seq_id": self.seq_id,
            "step_index": self.step_index,
            "tolerance_units": self.tolerance_units,
            "first_breach_stage": self.first_breach_stage,
            "stages": [asdict(s) for s in self.stages],
        }


def _diff_ints(working: dict[int, int], reference_scaled: dict[int, int],
               tolerance: int) -> StageDiff:
    tokens = sorted(working)
    worst = 0
    first_token = None
    for token in tokens:
        delta = abs(working[token] - reference_scaled[token])
        if delta > worst:
            worst = delta
            if delta > tolerance and first_token is None:
                first_token = token
    return worst, first_token


def attribute_step(request: Request, seq: CandidateSequence, step: Step,
                   config: DecodeConfig) -> AttributionRow:
    tokens = step.sorted_tokens()
    logprobs = [step.logprobs[t] for t in tokens]
    kr = run_kernel(tokens, logprobs, seq.temperature, config)

    # Reference path: one digit wider, treated as the unquantized benchmark.
    ref_config = config.with_replacements(width=config.ref_width, tier="REF")
    kr_ref = run_kernel(tokens, logprobs, seq.temperature, ref_config)
    ref_scaled = {t: (kr_ref.quant[t] + 5) // 10 for t in tokens}

    tolerance = config.tolerance_units
    stages: list[StageDiff] = []

    # Stage 1: temperature scaling + quantization.
    worst, first = _diff_ints(kr.quant, ref_scaled, tolerance)
    stages.append(StageDiff("temperature_scaling", worst, first,
                            worst <= tolerance))

    # Stage 2: truncation accumulation (prefix mass working vs reference).
    tr = truncate(kr, config)
    tr_ref = truncate(kr_ref, ref_config)
    work_prefix = _prefix_mass(tr, kr)
    ref_prefix = {t: (v + 5) // 10 for t, v in
                  _prefix_mass(tr_ref, kr_ref).items()}
    worst2, first2 = _diff_ints(work_prefix, ref_prefix, tolerance)
    stages.append(StageDiff("truncation_accumulation", worst2, first2,
                            worst2 <= tolerance))

    # Stage 3: sampling decision. Compare the selected token using working vs
    # reference RNG streams (same seed material, different distribution scale).
    content_fp = request.content_fingerprint()
    seed = derive_seed(content_fp, seq.seq_id, step.index,
                       config.fingerprint())
    work_choice = _choose(kr, tr, seed)
    ref_choice = _choose(kr_ref, tr_ref, seed)
    choice_diff = 0 if work_choice == ref_choice else 1
    stages.append(StageDiff("sampling_decision", float(choice_diff),
                            None if work_choice == ref_choice else work_choice,
                            work_choice == ref_choice))

    # Stage 4: version conversion. Per-mille rounding distance from working ints.
    conversion_diff = 0
    conv_token = None
    for token in tr.kept_tokens:
        working = kr.quant[token]
        mille = round(working * 1000 / config.scale)
        back = round(mille * config.scale / 1000)
        delta = abs(working - back)
        if delta > conversion_diff:
            conversion_diff = delta
            if delta > tolerance and conv_token is None:
                conv_token = token
    stages.append(StageDiff("version_conversion", float(conversion_diff),
                            conv_token, conversion_diff <= tolerance))

    first_breach = next((s.stage for s in stages
                         if not s.within_tolerance), None)
    return AttributionRow(
        request_id=request.request_id, seq_id=seq.seq_id,
        step_index=step.index, stages=stages,
        first_breach_stage=first_breach,
        tolerance_units=tolerance)


def _prefix_mass(tr, kr) -> dict[int, int]:
    out = {}
    cumulative = 0
    for token in tr.kept_tokens:
        cumulative += kr.quant[token]
        out[token] = cumulative
    return out


def _choose(kr, tr, seed: str) -> int:
    kept = list(tr.kept_tokens)
    total = sum(kr.quant[t] for t in kept)
    rng = DeterministicRng(seed)
    target = int(rng.next_unit() * total)
    if target >= total:
        target = total - 1
    cumulative = 0
    for token in kept:
        hi = cumulative + kr.quant[token]
        if cumulative <= target < hi:
            return token
        cumulative = hi
    return kept[-1]
