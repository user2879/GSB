from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import digest
from .engine import BuildEngine, BuildResult, build_workspace
from .snapshot import SnapshotStore
from .workspace import WorkspaceEditor, read_commands


@dataclass
class VerificationResult:
    passed: bool
    report: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.report


def _copy_workspace(source: Path, target: Path) -> None:
    for name in ("data", "config", "commands", "injections"):
        source_path = source / name
        if not source_path.exists():
            continue
        if source_path.is_dir():
            shutil.copytree(source_path, target / name, dirs_exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target / name)


def _cache_counts(result: BuildResult) -> dict[str, Any]:
    ledger = result.result.get("execution", {}).get("cache_ledger", [])
    counts: dict[str, dict[str, int]] = {}
    for item in ledger:
        stage = str(item.get("stage", "unknown"))
        action = str(item.get("action", "recompute"))
        counts.setdefault(stage, {"reuse": 0, "recompute": 0})
        counts[stage][action] = counts[stage].get(action, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _canonical_projection_equal(left: BuildResult, right: BuildResult) -> bool:
    return left.canonical == right.canonical


def run_verification(source_root: Path) -> VerificationResult:
    source_root = Path(source_root).resolve()
    command_path = source_root / "commands" / "edit-sequence.yaml"
    commands = read_commands(command_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "checks": [],
        "edit_steps": [],
        "passed": False,
    }
    checks: list[dict[str, Any]] = report["checks"]
    with tempfile.TemporaryDirectory(prefix="edge-ledger-verify-") as temporary:
        work_root = Path(temporary) / "workspace"
        _copy_workspace(source_root, work_root)
        initial = build_workspace(work_root)
        initial_full = build_workspace(
            work_root,
            state_dir=Path(temporary) / "full-initial-state",
            full=True,
        )
        checks.append(
            {
                "name": "initial_incremental_vs_full",
                "passed": _canonical_projection_equal(initial, initial_full),
                "incremental_digest": initial.canonical_digest,
                "full_digest": initial_full.canonical_digest,
            }
        )
        if initial.result["injection_hits"]:
            checks.append(
                {
                    "name": "injection_isolation",
                    "passed": all(
                        item["hit"] for item in initial.result["injection_hits"]
                    ),
                    "hits": [
                        {
                            "id": item["injection_id"],
                            "hit": item["hit"],
                            "codes": item["observed_codes"],
                        }
                        for item in initial.result["injection_hits"]
                    ],
                }
            )

        current_result = initial
        for index, command in enumerate(commands):
            command_id = str(command.get("id", f"command-{index + 1:03d}"))
            editor = WorkspaceEditor.from_root(work_root)
            editor.apply_command(command, command_id=command_id)
            editor.save()
            incremental = build_workspace(work_root)
            full = build_workspace(
                work_root,
                state_dir=Path(temporary) / f"full-state-{index:03d}",
                full=True,
            )
            equal = _canonical_projection_equal(incremental, full)
            previous_render = {
                item["entity_id"]: item["action"]
                for item in current_result.result["execution"]["cache_ledger"]
                if item["stage"] == "render"
            }
            render_actions = {
                item["entity_id"]: item["action"]
                for item in incremental.result["execution"]["cache_ledger"]
                if item["stage"] == "render"
            }
            affected = sorted(
                sample_id
                for sample_id, action in render_actions.items()
                if action == "recompute"
            )
            step = {
                "index": index + 1,
                "command_id": command_id,
                "op": command.get("op"),
                "incremental_vs_full": equal,
                "incremental_digest": incremental.canonical_digest,
                "full_digest": full.canonical_digest,
                "render_recomputed": affected,
                "cache_counts": _cache_counts(incremental),
                "render_reused_count": sum(
                    1 for action in render_actions.values() if action == "reuse"
                ),
                "score_reused_count": sum(
                    1
                    for item in incremental.result["execution"]["cache_ledger"]
                    if item["stage"] == "score" and item["action"] == "reuse"
                ),
            }
            report["edit_steps"].append(step)
            checks.append(
                {
                    "name": f"edit_{index + 1:03d}_{command_id}_incremental_vs_full",
                    "passed": equal,
                    "digest": incremental.canonical_digest,
                }
            )
            current_result = incremental

        threshold_checks: list[dict[str, Any]] = []
        for step in report["edit_steps"]:
            if step["op"] == "set_dedupe_threshold":
                threshold_checks.append(
                    {
                        "name": "dedupe_threshold_preserves_render_and_score",
                        "passed": not step["render_recomputed"]
                        and step["render_reused_count"] > 0,
                        "render_recomputed": step["render_recomputed"],
                        "render_reused_count": step["render_reused_count"],
                        "score_reused_count": step["score_reused_count"],
                    }
                )
        checks.extend(threshold_checks)

        final_canonical = current_result.canonical
        final_digest = current_result.canonical_digest
        recovery_root = Path(temporary) / "recovery-workspace"
        shutil.copytree(work_root, recovery_root, dirs_exist_ok=True, symlinks=True)
        recovery_store = SnapshotStore(recovery_root / "state")
        before = recovery_store.load_current()
        before_generation = before.snapshot.generation if before.snapshot else None
        crash = subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_ledger.cli",
                "build",
                "--root",
                str(recovery_root),
                "--crash-after-stage",
                "render",
                "--crash-mode",
                "exit",
                "--no-report",
            ],
            cwd=str(source_root),
            text=True,
            capture_output=True,
        )
        after = recovery_store.load_current()
        after_generation = after.snapshot.generation if after.snapshot else None
        recovered = build_workspace(recovery_root)
        recovery_equal = recovered.canonical == final_canonical
        checks.append(
            {
                "name": "forced_process_termination_keeps_committed_snapshot",
                "passed": crash.returncode == 91
                and before_generation == after_generation,
                "exit_code": crash.returncode,
                "before_generation": before_generation,
                "after_generation": after_generation,
            }
        )
        checks.append(
            {
                "name": "post_crash_rebuild_matches_final_incremental",
                "passed": recovery_equal,
                "expected_digest": final_digest,
                "recovered_digest": recovered.canonical_digest,
            }
        )

        unchanged = build_workspace(recovery_root)
        unchanged_actions = {
            item["stage"] + ":" + item["action"]
            for item in unchanged.result["execution"]["cache_ledger"]
        }
        unexpected_recomputes = [
            item
            for item in unchanged.result["execution"]["cache_ledger"]
            if item["action"] != "reuse"
            and not (
                item["stage"] == "score"
                and item["entity_id"] == "S-FAIL-NONFINITE"
            )
        ]
        checks.append(
            {
                "name": "unchanged_resume_reuses_persisted_products",
                "passed": unchanged.canonical == final_canonical
                and not unexpected_recomputes,
                "digest": unchanged.canonical_digest,
                "actions": sorted(unchanged_actions),
                "unexpected_recomputes": unexpected_recomputes,
            }
        )
        report["final_digest"] = final_digest

    # A result may legitimately differ in execution provenance; only the canonical
    # projection is compared field by field.
    report["canonical_projection_digest"] = digest(report["edit_steps"])
    report["passed"] = all(check.get("passed", False) for check in checks)
    return VerificationResult(passed=report["passed"], report=report)


def write_verification(result: VerificationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
