"""Diagnostic report serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .pipeline import RunReport


def build_report(run: RunReport) -> Dict[str, Any]:
    gr = run.global_raster
    return {
        "global_version": run.global_version,
        "config_hash": run.config_hash,
        "stages_committed": run.stages_committed,
        "budget": run.budget,
        "invalidation": {
            "recomputed_tiles": run.recomputed_tiles,
            "cached_tiles": run.cached_tiles,
            "invalidated_tiles": run.invalidated_tiles,
            "recomputed_fusions": run.recomputed_fusions,
            "cached_fusions": run.cached_fusions,
            "pending_fusions": run.pending_fusions,
        },
        "tile_score_summary": {
            tid: {
                "prob_summary": r.prob_summary,
                "score_summary": r.score_summary,
                "cache_key": r.cache_key,
            }
            for tid, r in sorted(run.tile_results.items())
        },
        "fusion_vote_detail": [
            {
                "pair_key": f.pair_key,
                "direction": f.direction,
                "fingerprint": f.fingerprint,
                "votes": [v.__dict__ for v in f.votes],
            }
            for f in sorted(run.fusions, key=lambda f: f.pair_key)
        ],
        "boundary_diagnostics": run.diagnostic_dicts(),
        "removed_components": [c.__dict__ for c in gr.removed] if gr else [],
        "global_raster": gr.raster.tolist() if gr else None,
        "failures": run.failures.to_dicts(),
    }


def write_report(run: RunReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_report(run), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
