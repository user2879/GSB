"""误差传播账本与边界报告。"""

from __future__ import annotations

from .util import content_hash


def build_ledger(spec, continuous: dict, mapping: dict, grid: dict,
                 entries: list[dict]) -> dict:
    uncertain = [e["constraint_id"] for e in entries
                 if e["status"] == "UNCERTAIN"]
    boundary = [e["constraint_id"] for e in entries
                if e.get("on_exact_boundary")]
    ledger = {
        "spec_hash": spec.spec_hash,
        "continuous_hash": continuous["continuous_hash"],
        "mapping_hash": mapping["mapping_hash"],
        "grid_hash": grid["grid_hash"],
        "seed_tree": continuous["seed_tree"],
        "quantization_rules": mapping["rules"],
        "observed_max_quantization_error": mapping["observed_max_error"],
        "declared_quantization_error_bounds": {
            "elevation": mapping["rules"]["elevation_error_bound"],
            "density": mapping["rules"]["density_error_bound"]},
        "weight_normalization": continuous["weight_normalization"],
        "entries": entries,
        "summary": {
            "total": len(entries),
            "pass": sum(1 for e in entries if e["status"] == "PASS"),
            "fail": sum(1 for e in entries if e["status"] == "FAIL"),
            "uncertain": len(uncertain),
            "uncertain_ids": uncertain,
            "on_exact_boundary_ids": boundary,
        },
    }
    ledger["ledger_hash"] = content_hash(
        {"spec_hash": ledger["spec_hash"],
         "continuous_hash": ledger["continuous_hash"],
         "mapping_hash": ledger["mapping_hash"],
         "grid_hash": ledger["grid_hash"],
         "entries": entries})
    return ledger


def boundary_report(ledger: dict) -> dict:
    """可复算边界报告：所有恰好落在边界或误差区间内的判定。"""
    reports = []
    for e in ledger["entries"]:
        if e["status"] == "UNCERTAIN" or e.get("on_exact_boundary"):
            reports.append({
                "constraint_id": e["constraint_id"],
                "kind": e["kind"],
                "observed": e["observed"],
                "threshold": e["threshold"],
                "direction": e["direction"],
                "total_band": e["total_band"],
                "plausible_interval": [e["plausible_low"], e["plausible_high"]],
                "deterministic_verdict": e["status"],
                "note": ("严格相等边界不静默四舍五入；落在总比较带内判 UNCERTAIN"
                         if e["status"] == "UNCERTAIN"
                         else "观测值与阈值严格相等，按声明方向与容差判定"),
            })
    return {"spec_hash": ledger["spec_hash"], "boundary_items": reports}
