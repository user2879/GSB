"""End-to-end local acceptance checks for deterministic incremental reconstruction."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .commands import apply_commands
from .fixed import stable_hash
from .models import ProjectState
from .pipeline import ReconstructionRunner, RunResult
from .synth import generate_demo, load_project


@dataclass
class Check:
    name: str
    passed: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "details": self.details}


def _ledger_stage(result: RunResult, stage: str) -> dict[str, int]:
    return dict(result.report["ledger"]["summary"]["by_stage"].get(stage, {}))


def _set_recon(state: ProjectState, key: str, value: Any) -> ProjectState:
    clone = state.clone()
    clone.data.setdefault("reconstruction", {})[key] = value
    return clone


def _set_quantization(state: ProjectState, bits: int, quant_level: int) -> ProjectState:
    clone = state.clone()
    clone.data.setdefault("fixed", {})["bits"] = bits
    clone.data.setdefault("fixed", {})["quant_level"] = quant_level
    return clone


def _fresh_workspace(root: Path, name: str) -> Path:
    workspace = root / name
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _run_fresh(root: Path, name: str, state: ProjectState, workers: int) -> RunResult:
    workspace = _fresh_workspace(root, name)
    return ReconstructionRunner(workspace, acquire_lock=False).run(
        state,
        workers_override=workers,
    )


def _compare_quantization(
    root: Path,
    final_state: ProjectState,
    final_full: RunResult,
) -> dict[str, Any]:
    quant2 = _run_fresh(
        root,
        "quant2",
        _set_quantization(final_state, final_state.config().bits, 2),
        workers=4,
    )
    bits12 = _run_fresh(
        root,
        "bits12",
        _set_quantization(final_state, 12, final_state.config().quant_level),
        workers=4,
    )
    common_quant = sorted(
        set(final_full.stages["triangulation"]["points"])
        & set(quant2.stages["triangulation"]["points"])
    )
    common_bits = sorted(
        set(final_full.stages["triangulation"]["points"])
        & set(bits12.stages["triangulation"]["points"])
    )
    triangle_quant = common_quant[0] if common_quant else ""
    triangle_bits = common_bits[0] if common_bits else ""
    quant_xyz_changed = bool(triangle_quant) and (
        final_full.stages["triangulation"]["points"][triangle_quant]["xyz_fixed"]
        != quant2.stages["triangulation"]["points"][triangle_quant]["xyz_fixed"]
    )
    quant_residual_changed = bool(triangle_quant) and (
        final_full.stages["triangulation"]["points"][triangle_quant]["residual_fixed"]
        != quant2.stages["triangulation"]["points"][triangle_quant]["residual_fixed"]
    )
    depth_quant_values_changed = False
    for view_id in final_full.stages["depth"]["order"]:
        left = final_full.stages["depth"]["views"][view_id]["pixels"]
        right = quant2.stages["depth"]["views"][view_id]["pixels"]
        if any(a["depth_q"] != b["depth_q"] for a, b in zip(left, right)):
            depth_quant_values_changed = True
            break
    bits_xyz_changed = bool(triangle_bits) and (
        final_full.stages["triangulation"]["points"][triangle_bits]["xyz_fixed"]
        != bits12.stages["triangulation"]["points"][triangle_bits]["xyz_fixed"]
    )
    bits_depth_changed = False
    for view_id in final_full.stages["depth"]["order"]:
        left = final_full.stages["depth"]["views"][view_id]["pixels"]
        right = bits12.stages["depth"]["views"][view_id]["pixels"]
        if any(a["depth_q"] != b["depth_q"] for a, b in zip(left, right)):
            bits_depth_changed = True
            break
    return {
        "quant_level_changes_xyz": quant_xyz_changed,
        "quant_level_changes_residual": quant_residual_changed,
        "quant_level_changes_depth": depth_quant_values_changed,
        "bits_change_xyz": bits_xyz_changed,
        "bits_change_depth": bits_depth_changed,
        "quant_digest": quant2.canonical_digest,
        "bits_digest": bits12.canonical_digest,
    }


def _run_crash_check(
    root: Path,
    repo_root: Path,
    project_path: Path,
    commands_path: Path | None,
    expected_digest: str,
    full_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    workspace = _fresh_workspace(root, "crash_workspace")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    base = [
        sys.executable,
        "-m",
        "mv3d",
        "run",
        "--project",
        str(project_path),
        "--workspace-root",
        str(workspace),
    ]
    if commands_path is not None:
        base.extend(["--commands", str(commands_path)])
    killed = subprocess.run(
        base + ["--sigkill-after-stage", "matches"],
        cwd=str(repo_root),
        env=environment,
        capture_output=True,
        text=True,
    )
    kill_ok = killed.returncode < 0 or killed.returncode == 137
    report_path = workspace / "resume-report.json"
    resumed = subprocess.run(
        base + ["--report", str(report_path)],
        cwd=str(repo_root),
        env=environment,
        capture_output=True,
        text=True,
    )
    resume_report: dict[str, Any] = {}
    if report_path.is_file():
        import json

        resume_report = json.loads(report_path.read_text(encoding="utf-8"))
    resume_used = bool(resume_report.get("snapshot", {}).get("resume_used"))
    resume_stage = resume_report.get("snapshot", {}).get("resume_from_stage")
    resume_digest = str(resume_report.get("canonical_digest", ""))
    failures_equal = resume_report.get("failures", []) == full_failures
    return {
        "kill_returncode": killed.returncode,
        "kill_ok": kill_ok,
        "restart_returncode": resumed.returncode,
        "resume_used": resume_used,
        "resume_from_stage": resume_stage,
        "resume_digest": resume_digest,
        "expected_digest": expected_digest,
        "digest_equal": resume_digest == expected_digest,
        "failures_equal": failures_equal,
        "stdout_tail": resumed.stdout[-500:],
        "stderr_tail": resumed.stderr[-500:],
    }


def run_acceptance(
    *,
    project: ProjectState,
    commands: list[dict[str, Any]],
    seed: int,
    workers: tuple[int, ...],
    run_crash_check: bool,
    repo_root: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mv3d-acceptance-") as temporary:
        root = Path(temporary)
        demo_dir = generate_demo(root / "demo", seed=seed, force=True)
        project_path = demo_dir / "project.json"
        commands_path = demo_dir / "commands.jsonl"
        initial = load_project(project_path)
        incremental_workspace = _fresh_workspace(root, "incremental")
        runner = ReconstructionRunner(incremental_workspace, acquire_lock=False)
        initial_result = runner.run(initial)
        state = initial
        edit_results: list[RunResult] = []
        for command in commands:
            state, command_failures = apply_commands(state, [command])
            result = runner.run(state, external_failures=command_failures)
            edit_results.append(result)
        final_incremental = edit_results[-1] if edit_results else initial_result
        full_result = _run_fresh(root, "full", state, workers=max(workers))
        concurrency_results = [
            _run_fresh(root, f"workers-{count}", state, workers=count)
            for count in workers
        ]
        budget_state = _set_recon(state, "global_point_budget", 1)
        budget_result = _run_fresh(root, "budget", budget_state, workers=4)
        quota_state = _set_recon(state, "global_point_budget", 1_000_000)
        quota_state.data["reconstruction"]["view_point_quota"] = 1
        quota_result = _run_fresh(root, "quota", quota_state, workers=4)
        quantization = _compare_quantization(root, state, full_result)

        checks: list[Check] = []
        checks.append(
            Check(
                "incremental_equals_full",
                final_incremental.canonical_digest == full_result.canonical_digest,
                {
                    "incremental": final_incremental.canonical_digest,
                    "full": full_result.canonical_digest,
                    "stage_equal": final_incremental.stages == full_result.stages,
                    "failures_equal": [f.as_dict() for f in final_incremental.failures]
                    == [f.as_dict() for f in full_result.failures],
                },
            )
        )
        concurrent_digests = [result.canonical_digest for result in concurrency_results]
        checks.append(
            Check(
                "concurrency_stable",
                all(digest == final_incremental.canonical_digest for digest in concurrent_digests),
                {
                    "workers": list(workers),
                    "digests": concurrent_digests,
                },
            )
        )
        checks.append(
            Check(
                "budget_pending",
                budget_result.exit_code == 3
                and any(f.code == "E_POINT_BUDGET_PENDING" for f in budget_result.failures)
                and bool(budget_result.pending_queue),
                {
                    "exit_code": budget_result.exit_code,
                    "pending_queue": budget_result.pending_queue,
                    "failure_counts": budget_result.report["failure_counts"],
                },
            )
        )
        checks.append(
            Check(
                "view_quota",
                any(f.code == "E_VIEW_QUOTA_EXCEEDED" for f in quota_result.failures),
                {
                    "failure_counts": quota_result.report["failure_counts"],
                    "pending_queue": quota_result.pending_queue,
                },
            )
        )
        if edit_results:
            first_command = commands[0] if commands else {}
            checks.append(
                Check(
                    "replace_view_partial_invalidation",
                    first_command.get("op") == "replace_view"
                    and _ledger_stage(edit_results[0], "features").get("reused", 0)
                    == len(initial_result.stages.get("features", {}).get("views", {})) - 1
                    and _ledger_stage(edit_results[0], "features").get("invalidated", 0) == 1,
                    {
                        "features": _ledger_stage(edit_results[0], "features"),
                        "depth": _ledger_stage(edit_results[0], "depth"),
                        "voxels": _ledger_stage(edit_results[0], "voxels"),
                    },
                )
            )
        checks.append(
            Check(
                "quantization_effects",
                quantization["quant_level_changes_xyz"]
                and quantization["quant_level_changes_residual"]
                and quantization["quant_level_changes_depth"]
                and quantization["bits_change_xyz"]
                and quantization["bits_change_depth"],
                quantization,
            )
        )
        if run_crash_check:
            crash = _run_crash_check(
                root,
                repo_root,
                project_path,
                commands_path if commands else None,
                full_result.canonical_digest,
                [failure.as_dict() for failure in full_result.failures],
            )
            checks.append(
                Check(
                    "sigkill_snapshot_recovery",
                    crash["kill_ok"]
                    and crash["restart_returncode"] == 0
                    and crash["resume_used"]
                    and crash["digest_equal"],
                    crash,
                )
            )
        report = {
            "passed": all(check.passed for check in checks),
            "seed": seed,
            "checks": [check.as_dict() for check in checks],
            "digests": {
                "initial": initial_result.canonical_digest,
                "final_incremental": final_incremental.canonical_digest,
                "full": full_result.canonical_digest,
            },
            "edit_ledgers": [
                {
                    "command": commands[index] if index < len(commands) else {},
                    "ledger_summary": result.report["ledger"]["summary"],
                    "digest": result.canonical_digest,
                }
                for index, result in enumerate(edit_results)
            ],
            "workspace_fingerprint": stable_hash(
                {
                    "final_digest": final_incremental.canonical_digest,
                    "initial_digest": initial_result.canonical_digest,
                }
            ),
        }
        return report

