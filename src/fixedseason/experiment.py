"""Reproducible incremental experiment harness."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .commands import CommandProcessor
from .engine import FitEngine
from .errors import E_COMMAND_INVALID, Issue
from .jsonutil import loads_strict, write_json_atomic
from .model import FitConfig, SeriesRecord


GENERATED_DIRS = ("cache", "snapshot", "reports", "state")
GENERATED_FILES = (
    "final_report.json",
    "equivalence-report.json",
    "incremental-trace.json",
    "width-sign-report.json",
    "width-sign-flip-report.json",
)


def clear_generated_artifacts(workspace: str | Path) -> None:
    root = Path(workspace)
    for name in GENERATED_DIRS:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
    for name in GENERATED_FILES:
        path = root / name
        if path.exists():
            path.unlink()


def workspace_generated_entries(workspace: str | Path) -> list[str]:
    root = Path(workspace)
    return [
        name
        for name in GENERATED_DIRS + GENERATED_FILES
        if (root / name).exists()
    ]


@dataclass
class ExperimentOutcome:
    reports: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    final_report: dict[str, Any]
    final_series: tuple[SeriesRecord, ...]
    final_config: FitConfig
    issues: tuple[Issue, ...]


def run_incremental_experiment(
    workspace: str | Path,
    series: Iterable[SeriesRecord],
    config: FitConfig,
    *,
    initial_issues: Iterable[Issue] = (),
    commands_path: str | Path | None = None,
    mode: str = "concurrent",
    fresh: bool = False,
    save_reports: bool = True,
) -> ExperimentOutcome:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if fresh:
        clear_generated_artifacts(workspace)
    reports_dir = workspace / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processor = CommandProcessor(
        series,
        config,
        source=str(commands_path) if commands_path is not None else "<commands>",
    )
    persistent_issues: list[Issue] = list(initial_issues)
    reports: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    def execute(step: int, line: int | None, op: str, applied: bool) -> None:
        state = processor.state
        all_issues = persistent_issues + processor.issues
        engine = FitEngine(
            workspace,
            state.config,
            state.series,
            validation_issues=all_issues,
            mode=mode,
        )
        report = engine.run(version=state.config.version, force_rerun=True)
        reports.append(report)
        entry = {
            "step": step,
            "line": line,
            "op": op,
            "applied": applied,
            "version": state.config.version,
            "config_hash": report["run"]["config_hash"],
            "report_file": f"reports/step-{step:03d}-v{state.config.version}.json",
        }
        trace.append(entry)
        if save_reports:
            write_json_atomic(workspace / entry["report_file"], report)

    execute(0, None, "<baseline>", True)
    if commands_path is not None and Path(commands_path).exists():
        text = Path(commands_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                command = loads_strict(line)
            except (ValueError, TypeError) as exc:
                processor.issues.append(
                    Issue(
                        code=E_COMMAND_INVALID,
                        message=f"invalid JSON: {exc}",
                        source=str(commands_path),
                        line=line_number,
                        stage="command",
                    )
                )
                execute(len(trace), line_number, "<invalid-json>", False)
                continue
            if not isinstance(command, dict):
                processor.issues.append(
                    Issue(
                        code=E_COMMAND_INVALID,
                        message="command line must be a JSON object",
                        source=str(commands_path),
                        line=line_number,
                        stage="command",
                    )
                )
                execute(len(trace), line_number, "<invalid-shape>", False)
                continue
            applied = processor.apply(command, line=line_number)
            execute(len(trace), line_number, str(command.get("op")), applied)
    elif commands_path is not None:
        processor.issues.append(
            Issue(
                code=E_COMMAND_INVALID,
                message="command file does not exist",
                source=str(commands_path),
                stage="command",
            )
        )
        execute(len(trace), None, "<missing-command-file>", False)

    final_state = processor.state
    return ExperimentOutcome(
        reports=reports,
        trace=trace,
        final_report=reports[-1],
        final_series=final_state.series,
        final_config=final_state.config,
        issues=tuple(persistent_issues + processor.issues),
    )


def canonical_equivalence(report: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match between incremental final state and a fresh build.

    Cache hit/miss actions and consumed-point ledgers legitimately depend on
    prior history, but stage identities, reused artifacts, coefficients,
    attribution, pending and failure order must not.
    """

    return {
        "schema_version": report["schema_version"],
        "run_identity": {
            "global_version": report["run"]["global_version"],
            "config_hash": report["run"]["config_hash"],
            "compute_config_hash": report["run"]["compute_config_hash"],
            "series_library_fingerprint": report["run"]["series_library_fingerprint"],
            "bit_width_tier": report["run"]["bit_width_tier"],
        },
        "config": report["config"],
        "validation_failures": report["validation_failures"],
        "fit_failures": report["fit_failures"],
        "failures": report["failures"],
        "pending": report["pending"],
        "unfinished": report["unfinished"],
        "series": report["series"],
        "reuse_ledger": report["reuse_ledger"],
        "budget_settings": report["budget"]["settings"],
        "pending_queue": report["budget"]["pending_queue"],
    }


def compare_reports(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    left_canon = canonical_equivalence(left)
    right_canon = canonical_equivalence(right)
    differences: list[dict[str, Any]] = []

    def walk(prefix: str, a: Any, b: Any) -> None:
        if type(a) is not type(b):
            differences.append({"path": prefix, "left": a, "right": b})
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(
                        {
                            "path": f"{prefix}.{key}" if prefix else key,
                            "left": a.get(key, "<missing>"),
                            "right": b.get(key, "<missing>"),
                        }
                    )
                else:
                    walk(f"{prefix}.{key}" if prefix else key, a[key], b[key])
            return
        if isinstance(a, list):
            if len(a) != len(b):
                differences.append(
                    {"path": f"{prefix}.length", "left": len(a), "right": len(b)}
                )
            for index in range(min(len(a), len(b))):
                walk(f"{prefix}[{index}]", a[index], b[index])
            return
        if a != b:
            differences.append({"path": prefix, "left": a, "right": b})

    walk("", left_canon, right_canon)
    return not differences, differences


def run_fresh_build(
    workspace: str | Path,
    series: Iterable[SeriesRecord],
    config: FitConfig,
    *,
    validation_issues: Iterable[Issue] = (),
    mode: str = "concurrent",
) -> dict[str, Any]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    clear_generated_artifacts(workspace)
    engine = FitEngine(
        workspace,
        config,
        series,
        validation_issues=validation_issues,
        mode=mode,
    )
    return engine.run(version=config.version, force_rerun=True)
