"""Auditable diagnostic report assembly (business rule 8)."""
from __future__ import annotations

import json
import os

from .fingerprints import fingerprint


def build_report(workdir: str, state, plan, latents, ledger,
                 version: int, recovered: bool,
                 reused_by_stage: dict, recomputed_by_stage: dict,
                 error_counts: dict, timing: dict | None = None) -> dict:
    degraded = [e.frame for e in plan.entries if e.degraded]
    budget_basis = [
        {
            "frame": e.frame,
            "demand": e.demand,
            "score_q": e.score_q,
            "raw_share_q": e.raw_share_q,
            "allocated": e.allocated,
            "truncated": e.truncated,
            "redistributed": e.redistributed,
            "degraded": e.degraded,
            "rounding_basis": e.rounding_basis,
        }
        for e in plan.entries
    ]
    fill_basis = {frame: latent.fill_basis
                  for frame, latent in latents.items() if latent.fill_basis}

    # Per-frame error attribution: quantization residuals summed per stage act
    # as the auditable rounding/error attribution per frame.
    attribution = {}
    for frame, latent in latents.items():
        residual_sum = 0
        residual_count = 0
        for step in latent.steps:
            for trace in step.trace:
                residual_sum += abs(trace.residual)
                residual_count += 1
        attribution[frame] = {
            "denoise_residual_abs_sum": residual_sum,
            "denoise_residual_events": residual_count,
            "final_latent_fp": fingerprint(latent.final,
                                           f"final-latent-f{frame}"),
            "degraded": frame in degraded,
        }

    return {
        "version": version,
        "recovered": recovered,
        "config_hash": fingerprint(state.config.canonical_json(),
                                   "config-hash"),
        "frame_count": state.frame_count,
        "quantization": {"qmode": state.config.qmode,
                         "frac_bits": state.config.frac_bits},
        "budget": {
            "global_budget": plan.global_budget,
            "total_demand": plan.total_demand,
            "total_allocated": plan.total_allocated,
            "truncated_total": plan.truncated_total,
            "per_frame_basis": budget_basis,
        },
        "degraded_frames": degraded,
        "fill_basis": fill_basis,
        "reuse": {
            "reused_total": ledger.reused_count,
            "recomputed_total": ledger.recomputed_count,
            "reused_by_stage": reused_by_stage,
            "recomputed_by_stage": recomputed_by_stage,
        },
        "error_counts": error_counts,
        "per_frame_error_attribution": attribution,
        "timing": timing or {},
    }


def write_report(workdir: str, report: dict) -> str:
    path = os.path.join(workdir, "diagnostic_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=1, ensure_ascii=False)
    return path
