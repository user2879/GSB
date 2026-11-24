"""Integer fixed-point window statistics, drift and the stable verdict.

All aggregates are order independent:
* mean  = round_half_away(sum(values), count)  - integer sum is associative;
* decile bucket counts are pure tallies;
* missing rate is a basis-point ratio of two counters.

Drift is reported in basis points (integer) for three components, plus the
composite max. Each component carries its own propagated error bound so the
verdict can distinguish "certainly over", "certainly under" and
"indistinguishable" without any floating point.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import fixedpoint as fp

DECILE_EDGES_REL = tuple(range(1000, 10000, 1000))  # 10%..90% in bps
VERDICT_OVER = "drift_confirmed"
VERDICT_UNDER = "stable"
VERDICT_INDISTINGUISHABLE = "indistinguishable"


@dataclass
class WindowStats:
    count: int
    nulls: int
    mean_q: int                 # mean at session scale Q
    sum_q: int
    missing_bps: int
    decile_hist: list[int]      # 10 tallies
    scale_exp: int

    def to_dict(self) -> dict:
        return {"count": self.count, "nulls": self.nulls,
                "mean_q": self.mean_q, "sum_q": self.sum_q,
                "missing_bps": self.missing_bps,
                "decile_hist": list(self.decile_hist),
                "scale_exp": self.scale_exp}


def reference_decile_edges(values: list[int | None], in_exp: int,
                            session_exp: int) -> list[int]:
    """Fixed decile edges derived once from the reference window.

    Applying the *same* integer edges to both windows makes the histogram a
    genuine shape comparison (quantile-relative buckets would always be 10%).
    """
    rescaled = sorted(fp.rescale_int(v, in_exp, session_exp)
                      for v in values if v is not None)
    n = len(rescaled)
    edges = []
    for p in DECILE_EDGES_REL:
        num = p * n
        idx = (num + 9999) // 10000  # integer ceiling of num / 10000
        edges.append(rescaled[min(n - 1, max(0, idx - 1))])
    return edges


def _histogram_with_edges(ordered: list[int], edges: list[int]) -> list[int]:
    hist = [0] * 10
    for v in ordered:
        bid = 0
        for e in edges:
            if v <= e:
                break
            bid += 1
        hist[min(9, bid)] += 1
    return hist


def compute_window_stats(values: list[int | None], in_exp: int,
                         session_exp: int,
                         decile_edges: list[int] | None = None) -> WindowStats:
    rescaled = [fp.rescale_int(v, in_exp, session_exp) for v in values if v is not None]
    count = len(rescaled)
    nulls = len(values) - count
    total = sum(rescaled)
    mean_q = fp.div_half_away(total, count) if count else 0
    ordered = sorted(rescaled)  # stable, deterministic ordering
    if decile_edges is None:
        decile_edges = reference_decile_edges(values, in_exp, session_exp)
    hist = _histogram_with_edges(ordered, decile_edges)
    return WindowStats(count, nulls, mean_q, total,
                       fp.bps_ratio(nulls, len(values)), hist, session_exp)


def mean_drift_bps(ref: WindowStats, obs: WindowStats):
    """|obs.mean - ref.mean| / |ref.mean| in bps; zero denom -> absolute bps."""
    diff = abs(obs.mean_q - ref.mean_q)
    if ref.mean_q == 0:
        return diff, "abs_bps_zero_ref"
    return fp.bps_ratio(diff, abs(ref.mean_q)), "relative_to_ref_mean"


def tv_drift_bps(ref: WindowStats, obs: WindowStats) -> int:
    """Total-variation distance between decile histograms, in bps."""
    n_ref = max(1, sum(ref.decile_hist))
    n_obs = max(1, sum(obs.decile_hist))
    acc = 0
    for a, b in zip(ref.decile_hist, obs.decile_hist):
        acc += abs(fp.bps_ratio(a, n_ref) - fp.bps_ratio(b, n_obs))
    return fp.div_half_away(acc, 2)


def drift_error_bounds(ref: WindowStats, obs: WindowStats,
                       feature_bound_q: int, feature_uncertain_rows: int):
    # Mean of n values each bounded by b ulps has mean error <= b (the n
    # cancels in sum/n). The relative bps bound is b / |ref.mean| * 1e4.
    mean_bound = fp.bps_ratio(feature_bound_q, max(1, abs(ref.mean_q)))
    # Total-variation moves only when boundary-uncertain rows change bucket.
    tv_bound = min(10000, fp.bps_ratio(feature_uncertain_rows,
                                       min(ref.count, obs.count) or 1))
    missing_bound = 0  # missing counters are exact; no numeric error
    return {"mean_bps": mean_bound, "tv_bps": tv_bound,
            "missing_bps": missing_bound}


def decide(drift_bps: int, error_bps: int, threshold_bps: int) -> dict:
    """Stable three-way decision with an explicit tie-break rule.

    over  iff drift - error > threshold;
    under iff drift + error < threshold;
    otherwise indistinguishable. At exact equality (drift +/- error ==
    threshold) the conservative stable rule selects "indistinguishable",
    never "over", and the result is flagged error-affected.
    """
    lower = max(0, drift_bps - error_bps)
    upper = drift_bps + error_bps
    if lower > threshold_bps:
        verdict = VERDICT_OVER
    elif upper < threshold_bps:
        verdict = VERDICT_UNDER
    else:
        verdict = VERDICT_INDISTINGUISHABLE
    return {"verdict": verdict, "threshold_bps": threshold_bps,
            "drift_bps": drift_bps, "error_bps": error_bps,
            "interval_lower_bps": lower, "interval_upper_bps": upper,
            "error_affected": verdict == VERDICT_INDISTINGUISHABLE,
            "tie_break": "indistinguishable_when_interval_meets_threshold"}


def evaluate_feature(stage_id: str, ref: WindowStats, obs: WindowStats,
                     threshold_bps: int, feature_bound_q: int,
                     uncertain_rows: int, scale_exp: int) -> dict:
    mean_bps, mean_basis = mean_drift_bps(ref, obs)
    tv_bps = tv_drift_bps(ref, obs)
    missing_bps = abs(obs.missing_bps - ref.missing_bps)
    components = {"mean_bps": mean_bps, "tv_bps": tv_bps,
                  "missing_bps": missing_bps}
    drift_bps = max(mean_bps, tv_bps, missing_bps)
    dominant_component = max(components, key=lambda k: components[k])
    err = drift_error_bounds(ref, obs, feature_bound_q, uncertain_rows)
    composite_error = max(err["mean_bps"], err["tv_bps"], err["missing_bps"])
    verdict = decide(drift_bps, composite_error, threshold_bps)
    return {"stage_id": stage_id, "scale_exp": scale_exp,
            "aggregation": "sum_then_divide_half_away;histogram_tally;bps_counts",
            "reference": ref.to_dict(), "observation": obs.to_dict(),
            "components": components, "dominant_component": dominant_component,
            "drift_bps": drift_bps,
            "component_error_bps": err, "error_bps": composite_error,
            "verdict": verdict}
