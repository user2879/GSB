"""Fixed-point window posterior synthesis and frame-level endpoint restore.

A fixed, named projection maps each window's integer feature vector to a speech
posterior.  All arithmetic stays in the integer (Qn) domain:

1. saturating weighted integer accumulation (frame/dim order is fixed upstream),
2. deterministic piecewise-linear squashing to a Qn posterior in ``[0, 1]``,
3. threshold decision with an explicit, documented half-step tie rule.

Frame-level conclusions are then restored by taking the *maximum* posterior
over the windows that cover each frame; ownership ties are broken by the lowest
window index and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .config import Config
from .fixedpoint import (fp_add, fp_mul, half_step, saturate, to_fixed,
                         to_float)
from .windows import Window, WindowFeature

# Fixed, named projection weights in the same dimension order as the feature
# vector (6 Mel bins + ZCR + centroid).  These constants are part of the
# component's deterministic contract and therefore fingerprinted.
PROJECTION_NAME = "mel6_zcr_centroid_v1"
PROJECTION_WEIGHTS_REAL = (
    -0.20, -0.20, 0.05, 0.15, 0.35, 0.45,  # Mel: reward high-freq tilt
    -1.60,                                  # zero-crossing rate (noise high)
    -0.90,                                  # spectral centroid (noise high)
)
PROJECTION_BIAS_REAL = 0.55

# Piecewise-linear squash breakpoints on the (real) linear score.
_SQUASH_POINTS = (
    (-3.0, 0.02),
    (-1.5, 0.15),
    (-0.5, 0.35),
    (0.0, 0.50),
    (0.5, 0.65),
    (1.5, 0.85),
    (3.0, 0.98),
)


@dataclass(frozen=True)
class WindowPosterior:
    window_id: str
    linear_q: int
    posterior_q: int
    posterior: float
    is_speech: bool
    boundary_window: bool
    tie_kind: str                 # "exact_threshold" | "within_half_step" | ""
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "linear_q": self.linear_q,
            "posterior_q": self.posterior_q,
            "posterior": self.posterior,
            "is_speech": self.is_speech,
            "boundary_window": self.boundary_window,
            "tie_kind": self.tie_kind,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class FrameDecision:
    frame_index: int
    covering_windows: Tuple[str, ...]
    owner_window: str
    max_posterior: float
    max_posterior_q: int
    is_speech: bool
    boundary_frame: bool
    tie_kind: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "covering_windows": list(self.covering_windows),
            "owner_window": self.owner_window,
            "max_posterior": self.max_posterior,
            "max_posterior_q": self.max_posterior_q,
            "is_speech": self.is_speech,
            "boundary_frame": self.boundary_frame,
            "tie_kind": self.tie_kind,
        }


@dataclass(frozen=True)
class EndpointInterval:
    endpoint_id: str
    frame_start: int
    frame_end: int
    sample_start: int
    sample_end: int
    frame_indices: Tuple[int, ...]
    window_ids: Tuple[str, ...]
    boundary_frames: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "frame_indices": list(self.frame_indices),
            "window_ids": sorted(self.window_ids),
            "boundary_frames": list(self.boundary_frames),
        }


def projection_ints(frac_bits: int, quant_mode: str) -> Tuple[List[int], int]:
    weights = [to_fixed(w, frac_bits, quant_mode)
               for w in PROJECTION_WEIGHTS_REAL]
    bias = to_fixed(PROJECTION_BIAS_REAL, frac_bits, quant_mode)
    return weights, bias


def _piecewise_squash(score_real: float) -> float:
    points = _SQUASH_POINTS
    if score_real <= points[0][0]:
        return points[0][1]
    if score_real >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= score_real <= x1:
            ratio = (score_real - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return 0.5


def synthesize_posterior(feature: WindowFeature, config: Config,
                         cache_key: str = "") -> WindowPosterior:
    weights_q, bias_q = projection_ints(config.frac_bits, config.quant_mode)
    if len(feature.int_features) != len(weights_q):
        raise ValueError(
            f"feature dim {len(feature.int_features)} != projection dim "
            f"{len(weights_q)} for {feature.window_id}")

    # Weighted integer accumulation in dimension-ascending order, saturated.
    acc = 0
    for dim in range(len(weights_q)):
        product = fp_mul(feature.int_features[dim], weights_q[dim],
                         config.frac_bits)
        acc = fp_add(acc, product)
    linear_q = fp_add(acc, bias_q)
    linear_real = to_float(linear_q, config.frac_bits)

    posterior_real = _piecewise_squash(linear_real)
    posterior_q = saturate(to_fixed(posterior_real, config.frac_bits,
                                    config.quant_mode))
    posterior = to_float(posterior_q, config.frac_bits)

    threshold_q = to_fixed(config.speech_threshold, config.frac_bits,
                           config.quant_mode)
    half = half_step(config.frac_bits)
    delta = abs(posterior - config.speech_threshold)

    tie_kind = ""
    boundary = False
    if posterior_q == threshold_q:
        # Exact-threshold tie: fixed rule -> speech (upper side wins), marked.
        tie_kind = "exact_threshold"
        boundary = True
        is_speech = True
    elif delta <= half:
        # Within one quantization half step: fixed rule -> speech, marked.
        tie_kind = "within_half_step"
        boundary = True
        is_speech = True
    else:
        is_speech = posterior_q > threshold_q
        boundary = False

    return WindowPosterior(
        window_id=feature.window_id,
        linear_q=linear_q,
        posterior_q=posterior_q,
        posterior=posterior,
        is_speech=is_speech,
        boundary_window=boundary,
        tie_kind=tie_kind,
        cache_key=cache_key)


def restore_frame_decisions(windows: Sequence[Window],
                            posteriors: Dict[str, WindowPosterior],
                            frame_intervals: Sequence[Tuple[int, int]],
                            config: Config) -> List[FrameDecision]:
    """Reduce overlapping window posteriors to per-frame endpoint decisions."""
    # Map each frame to the windows that fully cover it (window index order).
    covering: Dict[int, List[Window]] = {i: [] for i in
                                         range(len(frame_intervals))}
    for window in sorted(windows, key=lambda w: w.index):
        # A window whose feature stage failed (injection / over-budget) has no
        # posterior and must not participate in frame restoration.
        if window.window_id not in posteriors:
            continue
        for fi in window.frame_indices:
            covering[fi].append(window)

    decisions: List[FrameDecision] = []
    for fi in range(len(frame_intervals)):
        owners = covering[fi]
        if not owners:
            # No window covers this frame: deterministic non-speech default.
            decisions.append(FrameDecision(
                frame_index=fi, covering_windows=tuple(), owner_window="",
                max_posterior=0.0, max_posterior_q=0, is_speech=False,
                boundary_frame=False, tie_kind="uncovered"))
            continue
        # Maximum posterior; ties broken by lowest window index (deterministic).
        best = owners[0]
        best_post = posteriors[best.window_id]
        tied = False
        for cand in owners[1:]:
            cand_post = posteriors[cand.window_id]
            if cand_post.posterior_q > best_post.posterior_q:
                best = cand
                best_post = cand_post
                tied = False
            elif cand_post.posterior_q == best_post.posterior_q:
                tied = True
        owner_decision = best_post
        decisions.append(FrameDecision(
            frame_index=fi,
            covering_windows=tuple(w.window_id for w in owners),
            owner_window=best.window_id,
            max_posterior=owner_decision.posterior,
            max_posterior_q=owner_decision.posterior_q,
            is_speech=owner_decision.is_speech,
            boundary_frame=owner_decision.boundary_window,
            tie_kind=("max_posterior_tie" if tied
                      else owner_decision.tie_kind)))
    return decisions


def build_endpoint_intervals(decisions: Sequence[FrameDecision],
                             frame_intervals: Sequence[Tuple[int, int]],
                             windows: Sequence[Window],
                             posteriors: Dict[str, WindowPosterior]
                             ) -> List[EndpointInterval]:
    """Group consecutive speech frames into endpoint intervals with provenance.
    """
    window_by_id = {w.window_id: w for w in windows}
    intervals: List[EndpointInterval] = []
    ordinal = 0
    i = 0
    n = len(decisions)
    while i < n:
        if not decisions[i].is_speech:
            i += 1
            continue
        start = i
        while i < n and decisions[i].is_speech:
            i += 1
        end = i  # exclusive
        frame_indices = tuple(range(start, end))
        window_ids = sorted({
            wid
            for fi in frame_indices
            for wid in decisions[fi].covering_windows
        })
        boundary_frames = tuple(
            fi for fi in frame_indices if decisions[fi].boundary_frame)
        sample_start = frame_intervals[start][0]
        sample_end = frame_intervals[end - 1][1]
        intervals.append(EndpointInterval(
            endpoint_id=f"E{ordinal:04d}",
            frame_start=start, frame_end=end,
            sample_start=sample_start, sample_end=sample_end,
            frame_indices=frame_indices,
            window_ids=tuple(window_ids),
            boundary_frames=boundary_frames))
        ordinal += 1
    return intervals
