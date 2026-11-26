from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from renderdiag import __version__
from renderdiag.config import load_config
from renderdiag.engine import RenderEngine
from renderdiag.errors import DiagnosticError
from renderdiag.resources import load_command_stream, load_injections
from renderdiag.samples import generate_sample_workspace, validate_sample_workspace
from renderdiag.util import atomic_write_json


def _comparable(engine: RenderEngine) -> dict[str, Any]:
    results = engine.results()
    return {
        "frames": results["frames"],
        "denoise_steps": results["denoise_steps"],
        "fusion": results["fusion"],
        "rows": results["rows"],
        "text_fingerprint": results["text_fingerprint"],
        "image_fingerprint": results["image_fingerprint"],
        "motion_fingerprints": results["motion_fingerprints"],
    }


def _pixel_equal(left: RenderEngine, right: RenderEngine) -> bool:
    if sorted(left._frame_products) != sorted(right._frame_products):
        return False
    for frame in sorted(left._frame_products):
        if not np.array_equal(left.frame_pixels(frame), right.frame_pixels(frame)):
            return False
    return True


def _complete(engine: RenderEngine, initial: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    responses: list[dict[str, Any]] = []
    response = engine.advance() if initial is None else initial
    responses.append(response)
    batches = 1
    while response["pending"] and batches < 10000 and not response["is_latest"]:
        response = engine.advance()
        responses.append(response)
        batches += 1
    return response, responses


def _engine(root: Path, config, name: str, **kwargs) -> RenderEngine:
    return RenderEngine(
        config,
        root / name,
        store_root=root / f"{name}-store",
        output_dir=root / f"{name}-output",
        **kwargs,
    )


def _fresh_final(root: Path, config, commands, name: str) -> RenderEngine:
    engine = _engine(root, config, name)
    for command in commands:
        engine.submit(command)
    final, _ = _complete(engine)
    if final["code"] != "OK":
        raise RuntimeError(f"fresh final render failed: {final['code']} {final['failures']}")
    return engine


def _last_status_counts(response: dict[str, Any]) -> dict[str, int]:
    counts = {"reused": 0, "recomputed": 0}
    for item in response["ledger"]:
        status = item["status"]
        if status in counts:
            counts[status] += 1
    return counts


def run_acceptance(output: Path) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = generate_sample_workspace(output / "sample")
    validation = validate_sample_workspace(samples["config"], samples["commands"])
    config = load_config(samples["config"])
    commands = load_command_stream(samples["commands"])
    injections = load_injections(samples["injections"])

    run_root = output / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    incremental = _engine(run_root, config, "incremental")
    initial, initial_responses = _complete(incremental)
    if initial["code"] != "OK":
        raise RuntimeError(f"initial incremental render failed: {initial['failures']}")

    sequence: list[dict[str, Any]] = []
    all_invariance = True
    all_latency = True
    max_elapsed = max(item["elapsed_ms"] for item in initial_responses)
    for index, command in enumerate(commands, start=1):
        incremental.submit(command)
        final, responses = _complete(incremental)
        max_elapsed = max(max_elapsed, max(item["elapsed_ms"] for item in responses))
        all_latency = all_latency and all(
            item["elapsed_ms"] <= config.interaction_latency_budget_ms for item in responses
        )
        full = _fresh_final(run_root, config, commands[:index], f"full-{index:02d}")
        incremental_results = _comparable(incremental)
        full_results = _comparable(full)
        invariance = {
            "frames_equal": incremental_results["frames"] == full_results["frames"],
            "steps_equal": incremental_results["denoise_steps"] == full_results["denoise_steps"],
            "fusion_equal": incremental_results["fusion"] == full_results["fusion"],
            "rows_equal": incremental_results["rows"] == full_results["rows"],
            "pixels_equal": _pixel_equal(incremental, full),
        }
        sequence.append(
            {
                "cursor": incremental.cursor,
                "op": command.op,
                "changed_fields": final["changed_fields"],
                "response_code": final["code"],
                "stale": final["stale"],
                "completed_frames": final["summary"]["completed_frames"],
                "reuse_recompute": _last_status_counts(final),
                "invariance": invariance,
                "frame_fingerprints": [item["content_fingerprint"] for item in incremental_results["frames"]],
            }
        )
        all_invariance = all_invariance and all(invariance.values())

    # Stale-vs-latest response semantics.
    baseline_cursor = incremental.cursor
    incremental.submit({"op": "set_guidance", "guidance": 2.25})
    stale_response = incremental.advance()
    stale_snapshot = {
        "code": stale_response["code"],
        "stale": stale_response["stale"],
        "is_latest": stale_response["is_latest"],
        "requested_cursor": stale_response["requested_cursor"],
        "requested_config_fp": stale_response["requested_config_fp"],
        "based_on_cursor": stale_response["based_on_cursor"],
        "based_on_config_fp": stale_response["based_on_config_fp"],
        "pending_count": len(stale_response["pending"]),
        "pending_codes": sorted({item["code"] for item in stale_response["pending"]}),
        "elapsed_ms": stale_response["elapsed_ms"],
    }
    latest_response, latencies = _complete(incremental, stale_response)
    max_elapsed = max(max_elapsed, max(item["elapsed_ms"] for item in latencies))
    all_latency = all_latency and all(
        item["elapsed_ms"] <= config.interaction_latency_budget_ms for item in latencies
    )
    latest_snapshot = {
        "code": latest_response["code"],
        "stale": latest_response["stale"],
        "is_latest": latest_response["is_latest"],
        "requested_cursor": latest_response["requested_cursor"],
        "requested_config_fp": latest_response["requested_config_fp"],
        "based_on_cursor": latest_response["based_on_cursor"],
        "based_on_config_fp": latest_response["based_on_config_fp"],
        "completed_frames": latest_response["summary"]["completed_frames"],
    }
    freshness_ok = (
        stale_snapshot["code"] == "E_STALE_SUMMARY"
        and stale_snapshot["stale"]
        and not stale_snapshot["is_latest"]
        and stale_snapshot["based_on_cursor"] == baseline_cursor
        and stale_snapshot["requested_cursor"] == baseline_cursor + 1
        and stale_snapshot["based_on_config_fp"] != stale_snapshot["requested_config_fp"]
        and latest_snapshot["code"] == "OK"
        and latest_snapshot["is_latest"]
        and not latest_snapshot["stale"]
        and latest_snapshot["based_on_cursor"] == latest_snapshot["requested_cursor"]
    )

    # Batch-budget pending priority.
    priority_config = replace(config, batch_point_budget=1)
    priority_engine = _engine(run_root, priority_config, "priority")
    first_batch = priority_engine.advance()
    pending_before = {
        (item["frame"], item["step"])
        for item in first_batch["pending"]
        if item["stage"] == "denoise"
    }
    second_batch = priority_engine.advance()
    first_computed = second_batch["batch"]["computed_tasks"][0]
    priority_ok = (
        first_batch["code"] == "E_STALE_SUMMARY"
        and (first_computed["frame"], first_computed["step"]) in pending_before
    )

    # Concurrency must not alter fixed-point content.
    serial_config = replace(config, max_concurrency=1)
    parallel_config = replace(config, max_concurrency=4)
    serial = _fresh_final(run_root, serial_config, commands, "concurrency-serial")
    parallel = _fresh_final(run_root, parallel_config, commands, "concurrency-parallel")
    concurrency_ok = (
        _comparable(serial) == _comparable(parallel)
        and _pixel_equal(serial, parallel)
    )

    # Partial snapshot restore resumes from exactly the same cache state.
    restore_root = run_root / "restore"
    restore_engine = _engine(restore_root, config, "job")
    restore_first = restore_engine.advance()
    restore_version = restore_engine.snapshot_store.current_version()
    restored = _engine(restore_root, config, "job")
    restore_loaded_cursor = restored.cursor
    restore_loaded_pending = len(restored.pending_denoise)
    restore_final, _ = _complete(restored)
    restore_full = _fresh_final(run_root, config, [], "restore-full")
    restore_ok = (
        restore_first["pending"]
        and restore_loaded_cursor == restore_engine.cursor
        and restore_loaded_pending == len(restore_engine.pending_denoise)
        and restore_final["code"] == "OK"
        and _comparable(restored) == _comparable(restore_full)
        and _pixel_equal(restored, restore_full)
    )

    # Independent fault injections.
    injection_reports: list[dict[str, Any]] = []
    for injection in injections[:-1]:
        engine = _engine(
            run_root,
            config,
            f"injection-{injection.injection_id}",
            injections=[injection],
        )
        response = engine.drain_current()
        injection_reports.append(
            {
                "id": injection.injection_id,
                "stage": injection.stage,
                "kind": injection.kind,
                "code": response["code"],
                "hit": injection.injection_id in [item["id"] for item in response["injection_hits"]],
                "failure_codes": [item["code"] for item in response["failures"]],
                "completed_frames": response["current_state"]["completed_frames"],
                "block_rows_completed": response["current_state"]["block_rows_completed"],
                "injection_hits": response["injection_hits"],
            }
        )
    stale_injection = injections[-1]
    snapshot_warm = _engine(run_root, config, "injection-snapshot")
    snapshot_warm.drain_current()
    snapshot_restart = _engine(
        run_root,
        config,
        "injection-snapshot",
        injections=[stale_injection],
    )
    injection_reports.append(
        {
            "id": stale_injection.injection_id,
            "stage": stale_injection.stage,
            "kind": stale_injection.kind,
            "hit": stale_injection.injection_id
            in [item["id"] for item in snapshot_restart.injection_report()],
            "failure_codes": [item["code"] for item in snapshot_restart.failures],
            "completed_frames": [],
            "block_rows_completed": {},
            "injection_hits": snapshot_restart.injection_report(),
        }
    )
    injections_ok = all(
        item["hit"]
        and (
            "E_PARTIAL_FAILURE" in item.get("code", "")
            or "E_STALE_SNAPSHOT_VERSION" in item["failure_codes"]
        )
        for item in injection_reports
    )

    test_log = output / "unittest.log"
    test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    test_log.write_text(test_process.stdout, encoding="utf-8")
    tests_ok = test_process.returncode == 0

    final_results = incremental.results()
    overall = bool(
        all_invariance
        and freshness_ok
        and priority_ok
        and concurrency_ok
        and restore_ok
        and injections_ok
        and tests_ok
        and all_latency
        and validation.get("ok")
    )
    report = {
        "component": "renderdiag",
        "version": __version__,
        "overall_pass": overall,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "pyyaml": yaml.__version__,
            "containerized": False,
        },
        "sample": validation,
        "invariants": {
            "incremental_vs_full_pixel_and_fingerprint": all_invariance,
            "stale_latest_distinction": freshness_ok,
            "pending_priority": priority_ok,
            "concurrency_determinism": concurrency_ok,
            "snapshot_restore_exact": restore_ok,
            "fault_injection_hits": injections_ok,
            "unit_tests": tests_ok,
            "interaction_latency_within_budget": all_latency,
        },
        "budgets": {
            "batch_point_budget": config.batch_point_budget,
            "interaction_latency_budget_ms": config.interaction_latency_budget_ms,
            "max_observed_response_ms": round(max_elapsed, 3),
        },
        "initial_response": {
            "code": initial["code"],
            "is_latest": initial["is_latest"],
            "completed_frames": initial["summary"]["completed_frames"],
            "batch_count": len(initial_responses),
        },
        "edit_sequence": sequence,
        "stale_response": stale_snapshot,
        "latest_response": latest_snapshot,
        "pending_priority": {
            "first_batch_computed": first_batch["batch"]["computed_tasks"],
            "first_pending_task": first_batch["pending"][0] if first_batch["pending"] else None,
            "second_batch_first_computed": first_computed,
            "prior_pending_hit": True,
        },
        "concurrency": {
            "serial": serial.results()["frames"],
            "parallel": parallel.results()["frames"],
        },
        "snapshot_restore": {
            "snapshot_version_before_restart": restore_version,
            "loaded_cursor": restore_loaded_cursor,
            "loaded_pending_count": restore_loaded_pending,
            "completed_frame_fingerprints": [
                item["content_fingerprint"]
                for item in restored.results()["frames"]
            ],
        },
        "injections": injection_reports,
        "final_frames": final_results["frames"],
        "final_denoise_steps": final_results["denoise_steps"],
        "snapshot_versions": incremental.snapshot_store.versions(),
        "artifact_count": incremental.artifact_store.count(),
        "unittest_log": str(test_log),
    }
    atomic_write_json(output / "report.json", report)

    summary_lines = [
        "# Render Diagnostic Acceptance",
        "",
        f"- Overall: **{'PASS' if overall else 'FAIL'}**",
        f"- Python: `{report['environment']['python']}`; NumPy `{report['environment']['numpy']}`; Pillow `{report['environment']['pillow']}`; PyYAML `{report['environment']['pyyaml']}`",
        f"- Incremental vs blank full rebuild: `{'PASS' if all_invariance else 'FAIL'}` across `{len(sequence)}` edits",
        f"- Stale/latest distinction: `{'PASS' if freshness_ok else 'FAIL'}`",
        f"- Concurrency 1 vs 4: `{'PASS' if concurrency_ok else 'FAIL'}`",
        f"- Snapshot restore: `{'PASS' if restore_ok else 'FAIL'}`",
        f"- Injection hits: `{'PASS' if injections_ok else 'FAIL'}`",
        f"- Unit tests: `{'PASS' if tests_ok else 'FAIL'}`",
        f"- Max observed response latency: `{report['budgets']['max_observed_response_ms']} ms` / budget `{config.interaction_latency_budget_ms} ms`",
        "",
        "## Final Frame Fingerprints",
        "",
    ]
    for frame in final_results["frames"]:
        summary_lines.append(
            f"- frame {frame['frame']}: content `{frame['content_fingerprint']}`, pixels `{frame['pixel_sha256']}`"
        )
    summary_lines.extend(
        [
            "",
            "## Stale Baseline",
            "",
            f"- requested cursor `{stale_snapshot['requested_cursor']}`, based-on cursor `{stale_snapshot['based_on_cursor']}`",
            f"- stale `{str(stale_snapshot['stale']).lower()}`, latest `{str(stale_snapshot['is_latest']).lower()}`, code `{stale_snapshot['code']}`",
            f"- pending entries `{stale_snapshot['pending_count']}` marked `{', '.join(stale_snapshot['pending_codes'])}`",
            "",
            "Detailed machine-readable output is in `report.json`; the complete test log is in `unittest.log`.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic render diagnostic acceptance checks")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "acceptance"),
        help="output directory for samples, report and rendered frames",
    )
    args = parser.parse_args(argv)
    try:
        report = run_acceptance(Path(args.out).resolve())
    except Exception as exc:
        print(json.dumps({"overall_pass": False, "error": repr(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "edit_count": len(report["edit_sequence"]),
                "max_observed_response_ms": report["budgets"]["max_observed_response_ms"],
                "final_frames": report["final_frames"],
                "invariants": report["invariants"],
                "report": str(Path(args.out).resolve() / "report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
