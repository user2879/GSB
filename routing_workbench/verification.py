"""End-to-end acceptance ladder for the local deterministic sample."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .commands import apply_command, ensure_config_shape
from .config import parse_config
from .engine import Workbench
from .errors import SimulatedCrash
from .injection import run_injections
from .io_layer import load_commands, load_utterances, load_yaml
from .util import atomic_write_json, atomic_write_text, fingerprint


COMPARE_FIELDS = ("business_digest", "config_hash", "results", "failures", "boundary_samples", "pending_queue")


def compare_reports(left: dict[str, Any], right: dict[str, Any], fields: tuple[str, ...] = COMPARE_FIELDS) -> dict[str, Any]:
    differences = [field for field in fields if left.get(field) != right.get(field)]
    return {
        "equal": not differences,
        "differences": differences,
        "left_digest": left.get("business_digest"),
        "right_digest": right.get("business_digest"),
    }


def materialize_final_inputs(
    utterances_path: str | Path,
    config_path: str | Path,
    commands_path: str | Path,
    target: str | Path,
) -> dict[str, Path]:
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    config = ensure_config_shape(load_yaml(config_path))
    parsed_utterances, _ = load_utterances(utterances_path)
    raw_map = {utterance.utterance_id: copy.deepcopy(utterance.raw) for utterance in parsed_utterances}
    commands, _ = load_commands(commands_path)
    for index, command in enumerate(commands):
        command = dict(command)
        command["command_index"] = index
        config_before = copy.deepcopy(config)
        raw_before = copy.deepcopy(raw_map)
        result = apply_command(command, config, raw_map, set(raw_map))
        if result.status == "failed":
            config = config_before
            raw_map = raw_before
            continue
        try:
            parse_config(config)
        except Exception:
            config = config_before
            raw_map = raw_before
    config_path_out = root / "routing.yaml"
    utterances_path_out = root / "utterances.jsonl"
    atomic_write_text(config_path_out, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    atomic_write_text(
        utterances_path_out,
        "".join(json.dumps(raw_map[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(raw_map)),
    )
    atomic_write_json(root / "commands.json", {"commands": []})
    return {"config": config_path_out, "utterances": utterances_path_out, "commands": root / "commands.json"}


def run_verification(
    *,
    samples_dir: str | Path,
    workdir: str | Path,
    report_path: str | Path | None = None,
    run_pytest: bool = True,
) -> dict[str, Any]:
    samples = Path(samples_dir)
    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    utterance_path = samples / "utterances.jsonl"
    config_path = samples / "routing.yaml"
    command_path = samples / "commands.json"
    vocabulary_path = samples / "vocabulary.txt"
    injection_path = samples / "injections.yaml"
    inputs = materialize_final_inputs(utterance_path, config_path, command_path, work / "final-inputs")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: dict[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    incremental = Workbench(
        samples_dir=samples,
        state_dir=work / "incremental-state",
        full=False,
        batch_budget=10**9,
    ).run()
    check(
        "sample_route",
        incremental["summary"]["failure_count"] == 0
        and incremental["summary"]["result_count"] == incremental["summary"]["valid_count"],
        {
            "summary": incremental["summary"],
            "first_results": incremental["results"][:6],
        },
    )

    full = Workbench(
        utterances_path=inputs["utterances"],
        config_path=inputs["config"],
        commands_path=inputs["commands"],
        vocabulary_path=vocabulary_path,
        state_dir=work / "full-state",
        full=True,
        batch_budget=10**9,
    ).run()
    comparison = compare_reports(incremental, full)
    check(
        "incremental_equals_full_rebuild",
        comparison["equal"],
        {
            **comparison,
            "compared_fields": list(COMPARE_FIELDS),
            "full_digest": full["business_digest"],
            "incremental_digest": incremental["business_digest"],
        },
    )

    noop = Workbench(
        samples_dir=samples,
        state_dir=work / "incremental-state",
        full=False,
        batch_budget=10**9,
    ).run()
    cache_stages = noop["cache"]["stages"]
    noop_all_reused = all(cache_stages[stage]["recomputed"] == 0 for stage in cache_stages)
    check(
        "noop_incremental_reuses_all",
        noop_all_reused and noop["business_digest"] == incremental["business_digest"],
        {"cache_stages": cache_stages, "digest": noop["business_digest"]},
    )

    serial = Workbench(
        utterances_path=inputs["utterances"],
        config_path=inputs["config"],
        commands_path=inputs["commands"],
        vocabulary_path=vocabulary_path,
        state_dir=work / "serial-state",
        full=True,
        workers=1,
        batch_budget=10**9,
    ).run()
    parallel = Workbench(
        utterances_path=inputs["utterances"],
        config_path=inputs["config"],
        commands_path=inputs["commands"],
        vocabulary_path=vocabulary_path,
        state_dir=work / "parallel-state",
        full=True,
        workers=4,
        batch_budget=10**9,
    ).run()
    serial_parallel = compare_reports(serial, parallel)
    serial_parallel["equal"] = (
        serial_parallel["equal"]
        and serial["cache"]["reuse_recompute"] == parallel["cache"]["reuse_recompute"]
        and serial["budget"] == parallel["budget"]
    )
    check(
        "serial_equals_parallel",
        serial_parallel["equal"],
        {
            "fields": ["summary", "results", "failures", "boundary_samples", "cache", "budget"],
            "workers": [serial["cache"]["workers"], parallel["cache"]["workers"]],
        },
    )

    resume_state = work / "resume-state"
    crashed = False
    try:
        Workbench(
            samples_dir=samples,
            state_dir=resume_state,
            full=False,
            crash_after_commands=4,
            batch_budget=10**9,
        ).run()
    except SimulatedCrash:
        crashed = True
    resumed = Workbench(
        samples_dir=samples,
        state_dir=resume_state,
        full=False,
        batch_budget=10**9,
    ).run()
    resume_comparison = compare_reports(resumed, full)
    check(
        "snapshot_interrupt_resume",
        crashed and resumed["snapshot"]["resumed"] and resume_comparison["equal"],
        {
            "interrupted_cursor": 4,
            "resumed": resumed["snapshot"]["resumed"],
            "manifest_stage_versions": resumed["snapshot"]["stage_versions"],
            "comparison": resume_comparison,
        },
    )

    stale_state = work / "stale-state"
    shutil.copytree(resume_state, stale_state)
    stale = Workbench(
        samples_dir=samples,
        state_dir=stale_state,
        full=False,
        expected_global_version=9999,
        batch_budget=10**9,
    ).run()
    stale_codes = [diagnostic.get("business_code") for diagnostic in stale["diagnostics"]]
    check(
        "snapshot_stale_full_rebuild",
        "E_SNAPSHOT_STALE" in stale_codes
        and stale["snapshot"]["resumed"] is False
        and stale["business_digest"] == full["business_digest"],
        {
            "diagnostics": stale["diagnostics"],
            "replacement": "whole-directory",
            "digest": stale["business_digest"],
        },
    )

    budget_state = work / "budget-state"
    constrained = Workbench(
        samples_dir=samples,
        state_dir=budget_state,
        full=False,
        batch_budget=24,
    ).run()
    constrained_pending = [item["utterance_id"] for item in constrained["pending_queue"]]
    drained = Workbench(
        samples_dir=samples,
        state_dir=budget_state,
        full=False,
        batch_budget=10**9,
    ).run()
    budget_comparison = compare_reports(drained, full)
    check(
        "budget_pending_and_priority",
        bool(constrained_pending)
        and not drained["pending_queue"]
        and budget_comparison["equal"]
        and any(failure["business_code"] == "E_ROUTE_BUDGET_PENDING" for failure in constrained["failures"]),
        {
            "pending_before": len(constrained_pending),
            "pending_prefix": constrained_pending[:6],
            "ledger_entries": len(drained["budget"]["sequence"]),
            "total_consumed": drained["budget"]["consumed_in_snapshot"],
            "comparison": budget_comparison,
        },
    )

    injections = run_injections(
        utterances_path=utterance_path,
        config_path=config_path,
        injections_path=injection_path,
        vocabulary_path=vocabulary_path,
    )
    check(
        "fault_injection_and_isolation",
        injections["all_hit"],
        {
            "injections": [
                {
                    "id": item["id"],
                    "fault": item["fault"],
                    "hit": item["hit"],
                    "failure_codes": item["failure_codes"],
                    "valid_count": item["valid_count"],
                    "result_count": item["result_count"],
                }
                for item in injections["injections"]
            ],
            "all_hit": injections["all_hit"],
        },
    )

    pytest_check: dict[str, Any] | None = None
    if run_pytest:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        pytest_check = {
            "returncode": completed.returncode,
            "output": [line for line in completed.stdout.splitlines() if line.strip()],
            "stderr": completed.stderr[-2000:],
            "command": f"{sys.executable} -m pytest -q",
        }
        check("pytest_suite", completed.returncode == 0, pytest_check)

    report = {
        "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "sample": {
            "utterances": str(utterance_path),
            "config": str(config_path),
            "commands": str(command_path),
            "injections": str(injection_path),
            "business_digest": full["business_digest"],
        },
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": _numpy_version(),
        },
        "schema_version": 1,
    }
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
    return report


def _numpy_version() -> str:
    try:
        import numpy

        return numpy.__version__
    except Exception:
        return "unavailable"

