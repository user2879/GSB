from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import repair_studio  # noqa: F401,E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from repair_studio.config import apply_command, command_affected_stages  # noqa: E402
from repair_studio.engine import RunController, recover_output, render_full_reference  # noqa: E402
from repair_studio.imaging import load_sample  # noqa: E402
from repair_studio.rendering import Renderer  # noqa: E402


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB")).copy()


def _final_params(sample: Any) -> Any:
    params = sample.config.params
    for command in sample.commands:
        operation = command.get("op", command.get("type"))
        if operation in {"exit", "request_exit", "shutdown", "snapshot"}:
            continue
        params = apply_command(params, sample.config.width, sample.config.height, command)
    return params


def verify(quick: bool) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_root = PROJECT_ROOT / "output" / f"acceptance-{stamp}"
    sync_output = run_root / "incremental_sync"
    started = time.perf_counter()
    sample_root = PROJECT_ROOT / "samples" / "procedural_a" if quick else PROJECT_ROOT / "samples"
    sync_report = RunController(sample_root, sync_output, mode="sync").run()
    sample_directories = [PROJECT_ROOT / "samples" / "procedural_a"]
    if not quick:
        sample_directories.append(PROJECT_ROOT / "samples" / "procedural_b")
    samples = {sample.sample_id: sample for sample in (load_sample(path) for path in sample_directories)}
    checks: dict[str, Any] = {}
    sample_evidence: dict[str, Any] = {}
    final_reference = None
    deep_sample = None

    for sample_report in sync_report["samples"]:
        sample_id = sample_report["sample_id"]
        sample = samples[sample_id]
        params = _final_params(sample)
        reference = render_full_reference(
            sample,
            run_root / "references" / sample_id / "final_cache",
            params,
        )
        current_png = _read_png(sync_output / "current" / sample_id / "final.png")
        summary_match = sample_report["final"]["stage_summaries"] == reference.stage_summaries
        artifact_match = sample_report["final"]["artifact_versions"] == reference.artifact_versions()
        pixel_match = bool(np.array_equal(current_png, reference.final_pixels))
        sample_evidence[sample_id] = {
            "final_params_fingerprint": params.fingerprint(),
            "incremental_png_sha256": reference_png_hash(current_png),
            "reference_png_sha256": reference.final_png_sha256,
            "pixel_match": pixel_match,
            "stage_summary_match": summary_match,
            "artifact_version_match": artifact_match,
            "task_keys_by_stage": {
                stage: sample_report["final"]["stage_summaries"][stage]["task_keys"]
                for stage in ("fill", "style", "composite")
            },
        }
        checks[f"{sample_id}_final_pixel_match"] = pixel_match
        checks[f"{sample_id}_final_stage_summary_match"] = summary_match
        checks[f"{sample_id}_final_artifact_version_match"] = artifact_match
        if deep_sample is None:
            deep_sample = sample
            final_reference = reference

    assert deep_sample is not None and final_reference is not None
    deep_report = next(item for item in sync_report["samples"] if item["sample_id"] == deep_sample.sample_id)
    params = deep_sample.config.params
    command_checks: list[dict[str, Any]] = []
    for command, response in zip(deep_sample.commands, deep_report["commands"]):
        operation = str(command.get("op", command.get("type")))
        if operation in {"exit", "request_exit", "shutdown", "snapshot"}:
            continue
        if response.get("business_code") != "OK":
            command_checks.append(
                {
                    "command_index": response["command_index"],
                    "op": operation,
                    "business_code": response["business_code"],
                    "skipped_full_compare": True,
                }
            )
            continue
        params = apply_command(params, deep_sample.config.width, deep_sample.config.height, command)
        reference = render_full_reference(
            deep_sample,
            run_root / "references" / deep_sample.sample_id / f"command_{response['command_index']}",
            params,
        )
        actual = response["stage_summaries"]
        invalidated = response["invalidated_stages"]
        expected = command_affected_stages(operation)
        command_checks.append(
            {
                "command_index": response["command_index"],
                "op": operation,
                "pixel_match": response["final_png_sha256"] == reference.final_png_sha256,
                "stage_summary_match": actual == reference.stage_summaries,
                "artifact_version_match": response["latest_completed"]["artifact_versions"]
                == reference.artifact_versions(),
                "expected_invalidated_stages": expected,
                "actual_invalidated_stages": invalidated,
                "reused_stages": response["reused_stages"],
                "invalidation_matches_downstream_rule": invalidated == expected,
            }
        )
    checks["deep_all_commands_pixel_match"] = all(
        item.get("pixel_match", False)
        for item in command_checks
        if not item.get("skipped_full_compare")
    )
    checks["deep_all_commands_stage_match"] = all(
        item.get("stage_summary_match", False)
        for item in command_checks
        if not item.get("skipped_full_compare")
    )
    checks["deep_invalidation_rule_match"] = all(
        item.get("invalidation_matches_downstream_rule", False)
        for item in command_checks
        if not item.get("skipped_full_compare")
    )

    final_params = _final_params(deep_sample)
    single_sample = replace(
        deep_sample,
        config=replace(deep_sample.config, thread_workers=1),
    )
    multi_sample = replace(
        deep_sample,
        config=replace(deep_sample.config, thread_workers=4),
    )
    single_renderer = Renderer(run_root / "concurrency" / "single_cache")
    multi_renderer = Renderer(run_root / "concurrency" / "multi_cache")
    try:
        single = single_renderer.render(single_sample, final_params)
        multi = multi_renderer.render(multi_sample, final_params)
    finally:
        single_renderer.close()
        multi_renderer.close()
    concurrency_evidence = {
        "pixel_match": bool(np.array_equal(single.final_pixels, multi.final_pixels)),
        "stage_summary_match": single.stage_summaries == multi.stage_summaries,
        "max_workers_configured": multi.max_workers_configured,
        "max_workers_observed": multi.max_workers_observed,
        "task_keys_ordered": all(
            multi.stage_summaries[stage]["task_keys"] == sorted(multi.stage_summaries[stage]["task_keys"])
            for stage in ("fill", "style", "composite")
        ),
        "failure_lists_equal": single.stage_summaries["fill"]["failures"]
        == multi.stage_summaries["fill"]["failures"],
    }
    checks["concurrency_pixel_match"] = concurrency_evidence["pixel_match"]
    checks["concurrency_stage_summary_match"] = concurrency_evidence["stage_summary_match"]
    checks["concurrency_worker_limit_respected"] = multi.max_workers_observed <= 4
    checks["concurrency_task_order_stable"] = concurrency_evidence["task_keys_ordered"]

    interactive_output = run_root / "interactive_budget"
    interactive_report = RunController(deep_sample.directory, interactive_output, mode="interactive").run()
    interactive_sample = interactive_report["samples"][0]
    budget_ms = interactive_sample["commands"][0]["budget_ms"]
    elapsed_values = [record["elapsed_ms"] for record in interactive_sample["commands"]]
    pending_records = [record for record in interactive_sample["commands"] if record["business_code"] == "E_RENDER_BUDGET_PENDING"]
    pending_honesty = all(
        record["result_is_latest"] is False
        and record["latest_completed"] is not None
        and record["progress"] is not None
        for record in pending_records
    )
    interactive_pixels = _read_png(interactive_output / "current" / deep_sample.sample_id / "final.png")
    budget_evidence = {
        "budget_ms": budget_ms,
        "max_elapsed_ms": max(elapsed_values),
        "all_responses_within_budget": max(elapsed_values) <= budget_ms,
        "pending_count": len(pending_records),
        "pending_has_real_preview_and_no_stale_masquerade": pending_honesty,
        "final_cursor": interactive_sample["completed_command_cursor"],
        "final_pixel_match": bool(np.array_equal(interactive_pixels, final_reference.final_pixels)),
    }
    checks["budget_all_responses_within_limit"] = budget_evidence["all_responses_within_budget"]
    checks["budget_pending_contract"] = pending_honesty
    checks["budget_final_pixel_match"] = budget_evidence["final_pixel_match"]

    recovery = recover_output(sync_output)
    recovery_evidence = {
        "status": recovery["status"],
        "recovered_versions": recovery["recovered_versions"],
        "exit_report_written": recovery["exit_report_written"],
    }
    checks["complete_snapshot_recovered"] = recovery["status"] == "recovered"
    checks["forced_recovery_does_not_claim_exit_report"] = recovery["exit_report_written"] is False

    diagnostics_evidence = {
        "sync_report_path": sync_report["report_path"],
        "interactive_report_path": interactive_report["report_path"],
        "per_command_fields": sorted(
            key
            for key in interactive_sample["commands"][0]
            if key
            in {
                "elapsed_ms",
                "business_code",
                "reused_count",
                "invalidated_count",
                "reused_stages",
                "invalidated_stages",
                "progress",
            }
        ),
        "sync_business_codes": sync_report["summary"]["business_codes"],
        "interactive_business_codes": interactive_report["summary"]["business_codes"],
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "run_root": str(run_root),
        "checks": checks,
        "samples": sample_evidence,
        "deep_command_checks": command_checks,
        "concurrency": concurrency_evidence,
        "budget": budget_evidence,
        "recovery": recovery_evidence,
        "diagnostics": diagnostics_evidence,
        "audit_note": (
            "Runtime reuse/invalidation histories depend on command order; canonical stage cache keys, "
            "content hashes and artifact versions are compared field-for-field against a blank-cache "
            "full render. Per-command runtime audits are emitted separately and satisfy the downstream "
            "invalidation rule."
        ),
    }


def reference_png_hash(pixels: np.ndarray) -> str:
    from repair_studio.imaging import array_sha256

    return array_sha256(pixels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local acceptance suite")
    parser.add_argument("--quick", action="store_true", help="verify only procedural_a")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "verification.json",
    )
    arguments = parser.parse_args()
    result = verify(arguments.quick)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
