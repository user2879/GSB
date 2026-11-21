"""Base-frequency / scale-factor boundary sensitivity diagnosis.

For every scheme we scan the configured neighborhood grids of the base
frequency and the scale factor.  Each valid grid point recomputes float64
extrapolation scores and reports their mean deviation from the baseline
(configured parameter value).  Invalid grid points (zero base, non-positive
scale) and dimension-level truncation / scaling-overflow events are captured as
boundary conditions instead of crashing.  Rankings are stable: descending
deviation, then ascending parameter value on ties.
"""

from __future__ import annotations

import numpy as np

from .errors import DiagError
from .scoring import score_sample

# parameter names
P_BASE = "base"
P_SCALE = "scale"


def _extrap_deviation(sample, config, decision, base_override, scale_override,
                      baseline) -> tuple[float | None, str | None]:
    try:
        res = score_sample(
            sample, config, decision, "float64",
            base_override=base_override, scale_override=scale_override)
    except DiagError as exc:
        return None, exc.code
    extrap_idx = [i for i, region in enumerate(res.regions)
                  if region == "extrap"]
    if not extrap_idx:
        return 0.0, None
    diffs = [abs(float(res.scaled_scores[i]) - float(baseline[i]))
             for i in extrap_idx]
    return float(np.mean(diffs)), None


def scan_parameter(samples, config, decision, scheme: str, param: str,
                   grid: tuple[float, ...], baselines: dict) -> list[dict]:
    decision = dict(decision)
    decision["scheme"] = scheme
    rows = []
    for value in grid:
        base_override = value if param == P_BASE else None
        scale_override = value if param == P_SCALE else None
        deviations: list[float] = []
        boundary: list[dict] = []
        error_code: str | None = None
        for sample in samples:
            baseline = baselines[sample.sample_id]
            dev, code = _extrap_deviation(
                sample, config, decision, base_override, scale_override,
                baseline)
            if code is not None:
                error_code = code
                boundary.append({
                    "sample_id": sample.sample_id,
                    "trigger": "invalid_parameter",
                    "code": code,
                    "value": float(value),
                })
                continue
            deviations.append(dev)
        mean_dev = float(np.mean(deviations)) if deviations else 0.0
        rows.append({
            "scheme": scheme,
            "parameter": param,
            "value": float(value),
            "mean_extrap_score_deviation": mean_dev,
            "is_baseline": bool(
                (param == P_BASE and np.isclose(value, config.base)) or
                (param == P_SCALE and np.isclose(value, config.scale))),
            "boundary_events": boundary,
            "error_code": error_code,
        })
    # deterministic ordering: deviation desc, then value asc
    rows.sort(key=lambda row: (-row["mean_extrap_score_deviation"],
                               row["value"], row["scheme"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def run_sensitivity(samples, config, decision, schemes,
                    sample_limit: int | None = None) -> dict:
    # deterministic representative subset (fixed ordering, no RNG)
    ordered = sorted(samples, key=lambda smp: smp.sample_id)
    if sample_limit is not None and len(ordered) > sample_limit:
        ordered = ordered[:sample_limit]
    samples = ordered
    # baseline float64 extrap score vectors per sample/scheme
    baselines_by_scheme: dict[str, dict] = {}
    base_rows: list[dict] = []
    scale_rows: list[dict] = []
    for scheme in schemes:
        scheme_decision = dict(decision)
        scheme_decision["scheme"] = scheme
        baselines = {}
        for sample in samples:
            res = score_sample(sample, config, scheme_decision, "float64")
            baselines[sample.sample_id] = np.asarray(res.scaled_scores)
        baselines_by_scheme[scheme] = baselines
        base_rows.extend(scan_parameter(
            samples, config, scheme_decision, scheme, P_BASE,
            config.sensitivity_base_grid, baselines))
        scale_rows.extend(scan_parameter(
            samples, config, scheme_decision, scheme, P_SCALE,
            config.sensitivity_scale_grid, baselines))
    return {
        "tier_id": config.tier_id,
        "base_ranking": base_rows,
        "scale_ranking": scale_rows,
    }
