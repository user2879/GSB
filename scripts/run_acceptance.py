from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from posdiag.models import (  # noqa: E402
    canonical_json,
    load_config,
    load_edit_plan,
    load_samples,
    sha256_value,
)
from posdiag.pipeline import deterministic_view, run_pipeline  # noqa: E402
from posdiag.samples import generate_sample_directory  # noqa: E402


ARTIFACTS = ROOT / "artifacts"
SAMPLES = ROOT / "samples"


def _report_hash(report: dict[str, Any]) -> str:
    return sha256_value(deterministic_view(report))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_tests() -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    return completed.returncode == 0, completed.stdout


def _write_budget_inputs(root: Path, samples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    samples_path = root / "samples.jsonl"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    samples_path.write_text(
        "".join(canonical_json(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    return config_path, samples_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end local acceptance and write evidence files.")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    generated = generate_sample_directory(SAMPLES, seed=20261124)
    config = load_config(generated["config"])
    loaded = load_samples(generated["samples"], dimension=config.dimension)
    checks: dict[str, Any] = {}

    # 1. Fresh three-scheme run with a persisted, inspectable stage snapshot.
    snapshot_all = ARTIFACTS / "snapshot_all_schemes"
    if snapshot_all.exists():
        shutil.rmtree(snapshot_all)
    all_schemes = run_pipeline(
        config,
        loaded.samples,
        snapshot_all,
        sample_issues=loaded.issues,
        report_path=ARTIFACTS / "all_schemes_report.json",
        schemes_override=("rope_standard", "rope_linear", "rope_ntk"),
    )
    sample_for_scheme_check = next(item for item in all_schemes.report["samples"] if item["sample_id"] == "s0007")
    scheme_details = {}
    for item in sample_for_scheme_check["schemes"]:
        scheme_details[item["scheme"]] = {
            "position_digest": item["position_summary"]["digest"],
            "scores": item["attention"]["scores"],
            "first_exceeded": item["error_trace"]["first_exceeded"],
        }
    checks["three_schemes"] = {
        "status": all_schemes.report["status"],
        "successful_task_count": all_schemes.report["summary"]["successful_task_count"],
        "distinct_position_digests": len({value["position_digest"] for value in scheme_details.values()}),
        "distinct_score_vectors": len({tuple(value["scores"]) for value in scheme_details.values()}),
        "details": scheme_details,
        "observed_max_concurrency": all_schemes.report["run_metadata"]["execution"]["observed_max_concurrency"],
    }

    with tempfile.TemporaryDirectory(prefix="posdiag-acceptance-") as temporary:
        scratch = Path(temporary)

        # 2. Incremental edit sequence vs. blank rebuild.
        incremental_workspace = scratch / "incremental"
        baseline = run_pipeline(config, loaded.samples, incremental_workspace)
        sample_edit = load_edit_plan(config, loaded.samples, generated["edit_plan_sample_only.json"])
        sample_incremental = run_pipeline(
            sample_edit.config,
            sample_edit.samples,
            incremental_workspace,
            sample_issues=loaded.issues,
            edit_issues=sample_edit.issues,
            edit_plan_fingerprint=sample_edit.plan_fingerprint,
            applied_edit_operations=sample_edit.applied_operations,
            report_path=ARTIFACTS / "incremental_sample_only_report.json",
        )
        sample_full = run_pipeline(
            sample_edit.config,
            sample_edit.samples,
            scratch / "full-sample-only",
            sample_issues=loaded.issues,
            edit_issues=sample_edit.issues,
        )
        checks["sample_only_incremental"] = {
            "baseline_status": baseline.report["status"],
            "deterministic_equal_to_blank_full": report_without_metadata(sample_incremental.report)
            == report_without_metadata(sample_full.report),
            "incremental_result_hash": _report_hash(sample_incremental.report),
            "full_result_hash": _report_hash(sample_full.report),
            "cache": sample_incremental.report["run_metadata"]["cache"],
            "stale": sample_incremental.report["run_metadata"]["stale"],
            "resumed": sample_incremental.report["run_metadata"]["resumed"],
        }

        combined_edit = load_edit_plan(config, loaded.samples, generated["edit_plan.json"])
        combined_incremental = run_pipeline(
            combined_edit.config,
            combined_edit.samples,
            incremental_workspace,
            sample_issues=loaded.issues,
            edit_issues=combined_edit.issues,
            edit_plan_fingerprint=combined_edit.plan_fingerprint,
            applied_edit_operations=combined_edit.applied_operations,
            report_path=ARTIFACTS / "incremental_combined_report.json",
        )
        combined_full = run_pipeline(
            combined_edit.config,
            combined_edit.samples,
            scratch / "full-combined",
            sample_issues=loaded.issues,
            edit_issues=combined_edit.issues,
            report_path=ARTIFACTS / "full_combined_report.json",
        )
        checks["combined_edit_sequence"] = {
            "deterministic_equal_to_blank_full": report_without_metadata(combined_incremental.report)
            == report_without_metadata(combined_full.report),
            "incremental_result_hash": _report_hash(combined_incremental.report),
            "full_result_hash": _report_hash(combined_full.report),
            "stale": combined_incremental.report["run_metadata"]["stale"],
            "diagnostic_codes": combined_incremental.report["run_metadata"]["diagnostic_codes"],
            "cache": combined_incremental.report["run_metadata"]["cache"],
        }

        # 3. Graceful stage interruption and resume.
        resume_workspace = scratch / "resume"
        stop_event = threading.Event()

        def stop_after_encoding(stage: str) -> None:
            if stage == "encoding":
                stop_event.set()

        interrupted = run_pipeline(
            config,
            loaded.samples,
            resume_workspace,
            stop_event=stop_event,
            stage_complete_callback=stop_after_encoding,
            report_path=ARTIFACTS / "interrupted_after_encoding.json",
        )
        resumed = run_pipeline(
            config,
            loaded.samples,
            resume_workspace,
            report_path=ARTIFACTS / "resumed_after_interrupt.json",
        )
        clean = run_pipeline(config, loaded.samples, scratch / "clean")
        manifest = json.loads((resume_workspace / "manifest.json").read_text(encoding="utf-8"))
        checks["interrupt_and_resume"] = {
            "interrupted_status": interrupted.report["status"],
            "interrupted_attention_computed": interrupted.report["run_metadata"]["cache"]["attention"]["computed"],
            "resumed_status": resumed.report["status"],
            "resumed_encoding_reused": resumed.report["run_metadata"]["cache"]["encoding"]["reused"],
            "deterministic_equal_to_clean": report_without_metadata(resumed.report)
            == report_without_metadata(clean.report),
            "manifest_status": manifest["status"],
            "manifest_global_version": manifest["global_version"],
            "manifest_stage_versions": {name: value["version"] for name, value in manifest["stages"].items()},
            "manifest_consumed_budget": manifest["consumed_budget"],
        }

        # 4. Budget edge cases.
        budget_samples = [
            {
                "sample_id": "s0001",
                "query_position": 2,
                "key_positions": [],
                "query_vector": [0.1, 0.2, 0.3, 0.4],
                "key_vectors": [],
            },
            {
                "sample_id": "s0002",
                "query_position": 2,
                "key_positions": [1],
                "query_vector": [0.1, 0.2, 0.3, 0.4],
                "key_vectors": [[0.4, 0.3, 0.2, 0.1]],
            },
        ]
        base_budget_config = {
            "scheme": "rope_standard",
            "base_frequency": 100.0,
            "training_context_length": 8,
            "extrapolation_target_length": 16,
            "dimension": 4,
            "scaling_factor": 2.0,
            "precision": "strict",
            "batch_point_budget": 3,
            "global_mac_budget": 4,
            "thread_pool_concurrency": 1,
            "snapshot_commit_interval": 1,
            "tolerance": 0.35,
            "global_version": 1,
        }
        pending_config_path, pending_samples_path = _write_budget_inputs(
            scratch / "budget-pending", budget_samples, base_budget_config
        )
        pending_config = load_config(pending_config_path)
        pending_loaded = load_samples(pending_samples_path, dimension=pending_config.dimension)
        pending = run_pipeline(
            pending_config,
            pending_loaded.samples,
            scratch / "budget-pending-state",
            sample_issues=pending_loaded.issues,
            report_path=ARTIFACTS / "budget_pending_report.json",
        )

        exhausted_config = dict(base_budget_config, batch_point_budget=2, global_mac_budget=1000)
        exhausted_config_path, exhausted_samples_path = _write_budget_inputs(
            scratch / "budget-exhausted", budget_samples[1:], exhausted_config
        )
        exhausted_cfg = load_config(exhausted_config_path)
        exhausted_loaded = load_samples(exhausted_samples_path, dimension=exhausted_cfg.dimension)
        exhausted = run_pipeline(
            exhausted_cfg,
            exhausted_loaded.samples,
            scratch / "budget-exhausted-state",
            sample_issues=exhausted_loaded.issues,
            report_path=ARTIFACTS / "budget_exhausted_report.json",
        )
        checks["budget_edges"] = {
            "pending_count": pending.report["summary"]["pending_task_count"],
            "pending_queue": pending.report["budget_usage"]["pending_queue"],
            "exhausted_count": exhausted.report["summary"]["budget_exhausted_task_count"],
            "exhausted_samples": exhausted.report["budget_usage"]["budget_exhausted_samples"],
            "deduction_sequence_stable": pending.report["budget_usage"]["deduction_sequence"]
            == pending.report["budget_usage"]["deduction_sequence"],
        }

    test_ok = True
    if not args.skip_tests:
        test_ok, test_output = _run_tests()
        (ARTIFACTS / "test_results.txt").write_text(test_output, encoding="utf-8")

    required_checks = [
        checks["three_schemes"]["status"] == "complete",
        checks["three_schemes"]["successful_task_count"] == 42,
        checks["three_schemes"]["distinct_position_digests"] == 3,
        checks["three_schemes"]["distinct_score_vectors"] == 3,
        checks["sample_only_incremental"]["deterministic_equal_to_blank_full"],
        all(stage["computed"] == 1 and stage["reused"] == 13 for stage in checks["sample_only_incremental"]["cache"].values()),
        checks["combined_edit_sequence"]["deterministic_equal_to_blank_full"],
        checks["combined_edit_sequence"]["stale"],
        "E_SNAPSHOT_STALE" in checks["combined_edit_sequence"]["diagnostic_codes"],
        checks["interrupt_and_resume"]["interrupted_status"] == "interrupted",
        checks["interrupt_and_resume"]["resumed_encoding_reused"] == 14,
        checks["interrupt_and_resume"]["deterministic_equal_to_clean"],
        checks["interrupt_and_resume"]["manifest_status"] == "complete",
        checks["budget_edges"]["pending_count"] == 1,
        checks["budget_edges"]["exhausted_count"] == 1,
        test_ok,
    ]
    summary = {
        "acceptance_passed": all(required_checks),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "task_concurrency": config.thread_pool_concurrency,
        },
        "checks": checks,
        "required_check_count": len(required_checks),
        "required_checks_passed": sum(1 for value in required_checks if value),
        "artifacts": {
            "all_schemes_report": "artifacts/all_schemes_report.json",
            "incremental_sample_only_report": "artifacts/incremental_sample_only_report.json",
            "incremental_combined_report": "artifacts/incremental_combined_report.json",
            "full_combined_report": "artifacts/full_combined_report.json",
            "interrupted_report": "artifacts/interrupted_after_encoding.json",
            "resumed_report": "artifacts/resumed_after_interrupt.json",
            "budget_pending_report": "artifacts/budget_pending_report.json",
            "budget_exhausted_report": "artifacts/budget_exhausted_report.json",
            "snapshot": "artifacts/snapshot_all_schemes",
            "test_log": "artifacts/test_results.txt" if not args.skip_tests else None,
        },
    }
    _write_json(ARTIFACTS / "acceptance_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if summary["acceptance_passed"] else 1


def report_without_metadata(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "run_metadata"}


if __name__ == "__main__":
    raise SystemExit(main())
