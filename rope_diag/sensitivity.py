"""Base / scale boundary sensitivity diagnosis for every scheme."""
from __future__ import annotations

import numpy as np

from .error_trace import trace_sample
from .schemes import SCHEMES, SchemeFault, SchemeParams, apply_scheme, boundary_scan
from .scoring import compute_scores


def _extrapolation_score_deviation(
    params: SchemeParams, samples: list[dict], dim_range, score_chunk, tolerance
) -> float:
    """Mean score deviation over extrapolation keys, or inf on scheme faults."""
    deviations: list[float] = []
    for sample in samples:
        try:
            out = apply_scheme(
                params,
                np.asarray(sample["query"], dtype=np.float64),
                np.asarray(sample["keys"], dtype=np.float64),
                sample["query_pos"],
                np.asarray(sample["key_positions"], dtype=np.float64),
            )
        except SchemeFault:
            return float("inf")
        scores = compute_scores(out.rotated_query, out.rotated_keys, dim_range, score_chunk)
        trace = trace_sample(
            params,
            sample["query"],
            sample["keys"],
            sample["query_pos"],
            sample["key_positions"],
            out.rotated_query,
            out.rotated_keys,
            scores,
            tolerance,
            dim_range,
        )
        for rec in trace["records"]:
            deviations.append(rec["deviations"]["score"])
    if not deviations:
        return 0.0
    return float(np.mean(np.asarray(deviations, dtype=np.float64)))


def diagnose(config: dict, samples: list[dict], dim_range, score_chunk: int) -> dict:
    """Scan base/scale neighbourhoods per scheme and rank sensitivity."""
    tolerance = float(config["error_tolerance"])
    base_factors = list(config.get("base_scan_factors", [0.5, 1.0, 2.0]))
    scale_factors = list(config.get("scale_scan_factors", [0.5, 1.0, 2.0]))
    report: dict[str, dict] = {}
    for scheme in SCHEMES:
        base_params = SchemeParams(
            scheme=scheme,
            base=float(config["base"]),
            scale=float(config["scale"]),
            dim=int(config["dim"]),
            train_len=int(config["train_len"]),
            target_len=int(config["target_len"]),
        )
        baseline_dev = _extrapolation_score_deviation(
            base_params, samples, dim_range, score_chunk, tolerance
        )
        entries = []
        for axis, factors in (("base", base_factors), ("scale", scale_factors)):
            for factor in factors:
                candidate = SchemeParams(
                    scheme=scheme,
                    base=base_params.base * factor if axis == "base" else base_params.base,
                    scale=base_params.scale * factor if axis == "scale" else base_params.scale,
                    dim=base_params.dim,
                    train_len=base_params.train_len,
                    target_len=base_params.target_len,
                )
                flags = boundary_scan(candidate)
                value = candidate.base if axis == "base" else candidate.scale
                entry = {
                    "axis": axis,
                    "factor": float(factor),
                    "value": float(value),
                    "unusable": bool(flags["unusable"]),
                    "reasons": list(flags["reasons"]),
                    "truncated_dims": list(flags["truncated_dims"]),
                    "boundary_samples": [],
                    "deviation": None,
                    "sensitivity": None,
                }
                if flags["unusable"]:
                    entry["boundary_samples"] = [
                        {"sample_id": s["sample_id"], "trigger": ",".join(flags["reasons"])}
                        for s in samples
                    ]
                else:
                    dev = _extrapolation_score_deviation(
                        candidate, samples, dim_range, score_chunk, tolerance
                    )
                    entry["deviation"] = dev
                    if factor != 1.0 and np.isfinite(dev) and np.isfinite(baseline_dev):
                        delta = abs(dev - baseline_dev)
                        entry["sensitivity"] = float(delta / abs(np.log(factor)))
                    elif factor == 1.0:
                        entry["sensitivity"] = 0.0
                entries.append(entry)
        ranked = sorted(
            (e for e in entries if e["sensitivity"] is not None),
            key=lambda e: (-e["sensitivity"], e["axis"], e["factor"]),
        )
        report[scheme] = {
            "baseline_deviation": baseline_dev,
            "entries": entries,
            "sensitivity_ranking": [
                {"axis": e["axis"], "factor": e["factor"], "sensitivity": e["sensitivity"]}
                for e in ranked
            ],
            "unusable_tiers": [
                {"axis": e["axis"], "factor": e["factor"], "reasons": e["reasons"]}
                for e in entries
                if e["unusable"]
            ],
        }
    return report
