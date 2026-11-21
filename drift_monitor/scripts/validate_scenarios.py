"""End-to-end acceptance scenarios.

Runs an interactive edit sequence incrementally, then rebuilds the same final
configuration from a clean workspace and compares every business field. Also
verifies concurrency determinism, snapshot resume/stale handling and the
quantization flip diagnostics.  Emits a machine-readable validation report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_monitor.config import load_config
from drift_monitor.interactive import Session
from drift_monitor.runner import run_pipeline

CONFIG = ROOT / "configs" / "default_monitor.yaml"
OUT = ROOT / "output" / "validation"

EDIT_SEQUENCE = [
    "run",
    "set-windows 6",
    "run",
    "set-threshold 0.5",
    "run",
    "set-tier coarse",
    "run",
    "set-tier fine",
    "set-metric age mean_shift",
    "run",
    "set-size-tier medium",
    "run",
    "set-concurrency 8",
    "run",
]

COMPARE_FIELDS = (
    "scores_and_judgements",
    "flip_diagnostics",
    "strategy_decision",
    "failure_list",
    "windows",
    "data_fingerprint",
    "budget_deduction_sequence",
)


def comparable(report: dict) -> dict:
    return {field: report[field] for field in COMPARE_FIELDS}


def main() -> int:
    inc_dir = OUT / "incremental"
    full_dir = OUT / "full_rebuild"
    # Clean validation workspace so the full rebuild truly starts empty.
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    session = Session(load_config(CONFIG), "small", inc_dir)
    transcript = []
    for command in EDIT_SEQUENCE:
        transcript.append({"command": command, "result": session.apply(command)})

    incremental = run_pipeline(session.config, session.tier_name,
                               inc_dir, resume=False)
    full = run_pipeline(session.config, session.tier_name,
                        full_dir, resume=False)
    consistency = {
        "business_fields_equal": comparable(incremental) == comparable(full),
        "required_keys_equal":
            incremental["incremental"]["required_keys"]
            == full["incremental"]["required_keys"],
        "incremental_reused": incremental["incremental"]["reused_count"],
        "incremental_recomputed": incremental["incremental"]["recomputed_count"],
        "full_rebuild_recomputed": full["incremental"]["recomputed_count"],
    }

    concurrency = {}
    for workers in (1, 4, 8):
        cfg = replace(load_config(CONFIG), concurrency=workers)
        report = run_pipeline(cfg, "small", OUT / f"concurrency_{workers}",
                              resume=False)
        concurrency[workers] = comparable(report)
    concurrency_equal = concurrency[1] == concurrency[4] == concurrency[8]

    flip_summary = {
        "total_pairs": len(incremental["flip_diagnostics"]),
        "flipped_judgements": sum(
            1 for f in incremental["flip_diagnostics"] if f["flipped"]),
        "breach_stages": sorted({
            f["first_breach_stage"] for f in incremental["flip_diagnostics"]
            if f["first_breach_stage"]
        }),
    }

    resume = run_pipeline(session.config, session.tier_name, inc_dir, resume=True)
    session.apply("set-threshold 0.33")
    stale = run_pipeline(session.config, session.tier_name, inc_dir, resume=True)

    report = {
        "edit_sequence": transcript,
        "incremental_vs_full_rebuild": consistency,
        "concurrency_deterministic": concurrency_equal,
        "flip_summary": flip_summary,
        "snapshot": {
            "healthy_resume": resume["snapshot"]["resume_status"],
            "stale_status": stale["snapshot"]["resume_status"],
            "global_version": stale["snapshot"]["global_version"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    ok = (consistency["business_fields_equal"]
          and consistency["required_keys_equal"]
          and concurrency_equal)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
