"""Reconstruction error diagnostics with independent recomputation support.

For every query the chosen strategy is compared against the full-SH
interpolation reference at the same order.  The report embeds everything a
third party needs to recompute identical numbers from the probe file alone:
probe file hash, sample-direction count, SH order, strategy, query-set hash
and the declared model constants.
"""
import numpy as np

from . import strategies as st
from .errors import StrategyDivergeError
from .hashing import floats_hex, sha256_json
from .queries import query_set_hash, solve_batch
from .strategies import ProbeField

EPSILON = 1e-9
# Declared slack between estimated-bound ordering and measured ordering.
ORDERING_SLACK = 0.05


def diagnose(field: ProbeField, strategy: str, queries: list,
             n_directions: int, probe_file_hash: str,
             extra_strategies=st.STRATEGIES) -> dict:
    reference = solve_batch(field, "full", queries)
    chosen = solve_batch(field, strategy, queries)
    per_query = []
    for q in sorted(queries, key=lambda x: x["key"]):
        key = q["key"]
        dev = chosen[key] - reference[key]
        abs_dev = float(np.linalg.norm(dev))
        rel = float(np.linalg.norm(dev) /
                    (np.linalg.norm(reference[key]) + EPSILON))
        per_query.append({
            "key": key,
            "abs_deviation": abs_dev,
            "relative_error": rel,
            "error_interval": [0.0, st.est_error_bound(
                strategy, field.grid_n, field.order, field.field_scale)],
            "irradiance": floats_hex(chosen[key]),
            "reference": floats_hex(reference[key]),
        })
    measured = {strategy: float(np.mean([e["relative_error"]
                                         for e in per_query]))}
    for other in extra_strategies:
        if other == strategy:
            continue
        vals = solve_batch(field, other, queries)
        errs = [float(np.linalg.norm(vals[q["key"]] - reference[q["key"]]) /
                      (np.linalg.norm(reference[q["key"]]) + EPSILON))
                for q in queries]
        measured[other] = float(np.mean(errs))
    report = {
        "grid_n": field.grid_n,
        "order": field.order,
        "n_directions": n_directions,
        "strategy": strategy,
        "probe_file_sha256": probe_file_hash,
        "coeff_sha256": _coeff_hash(field),
        "query_set_sha256": query_set_hash(queries),
        "model_constants": {
            "C_TRILINEAR": st.C_TRILINEAR,
            "C_NN_GRADIENT": st.C_NN_GRADIENT,
            "CLUSTER_BLOCK": st.CLUSTER_BLOCK,
            "ORDER_ERROR_FACTOR": st.ORDER_ERROR_FACTOR,
        },
        "per_query": per_query,
        "aggregate": {
            "mean_relative_error": measured[strategy],
            "max_relative_error": float(max(e["relative_error"]
                                            for e in per_query)),
            "mean_abs_deviation": float(np.mean([e["abs_deviation"]
                                                 for e in per_query])),
        },
        "measured_by_strategy": measured,
        "est_bounds_by_strategy": {
            s: st.est_error_bound(s, field.grid_n, field.order,
                                  field.field_scale)
            for s in st.STRATEGIES},
    }
    report["sha256"] = sha256_json(report)
    return report


def _coeff_hash(field: ProbeField) -> str:
    from .hashing import array_hash
    return array_hash(field.coeffs.astype("<f8"))


def check_ordering_consistency(report: dict) -> dict:
    """The estimated error-bound ordering must not contradict measured
    errors: if est(A) < est(B) then measured(A) <= measured(B) * (1+slack)."""
    est = report["est_bounds_by_strategy"]
    meas = report["measured_by_strategy"]
    violations = []
    names = sorted(est)
    for a in names:
        for b in names:
            if est[a] < est[b] and meas[a] > meas[b] * (1.0 + ORDERING_SLACK):
                violations.append({"lower_est": a, "higher_est": b,
                                   "measured_a": meas[a], "measured_b": meas[b]})
    result = {"consistent": not violations, "violations": violations,
              "slack": ORDERING_SLACK}
    if violations:
        raise StrategyDivergeError(
            "estimated error ordering contradicts measured errors",
            **result)
    return result


def recompute_report(probe_data: dict, order: int, strategy: str,
                     queries: list, report: dict) -> dict:
    """Independent recomputation from the probe file + report parameters."""
    from .sh import project
    coeffs = project(probe_data["radiance"], probe_data["directions"], order)
    field = ProbeField(probe_data["grid_n"], probe_data["bbox"], coeffs)
    return diagnose(field, strategy, queries,
                    probe_data["n_directions"],
                    report["probe_file_sha256"])


def verify_report(probe_data: dict, queries: list, report: dict) -> dict:
    recomputed = recompute_report(probe_data, report["order"],
                                  report["strategy"], queries, report)
    keys = ("per_query", "aggregate", "measured_by_strategy")
    match = all(recomputed[k] == report[k] for k in keys)
    return {"verified": bool(match),
            "report_sha256": report["sha256"],
            "recomputed_sha256": recomputed["sha256"]}
