from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnostics import build_diagnostic_report
from .engine import (
    DiagnosticEngine,
    config_fingerprint,
    evidence_library_fingerprint,
    run_sequence,
)
from .models import AppConfig, Session
from .sample import DEFAULT_SEED, generate_sample
from .snapshot import SnapshotStore
from .stages import compute_boundary
from .util import load_json, load_jsonl, sha256_json, write_json
from .validation import ERROR_CODE, validate_commands, validate_config, validate_session


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / "config" / "demo.json",
        "session": root / "sample" / "session.json",
        "commands": root / "sample" / "commands.jsonl",
        "snapshots": root / "snapshots",
        "reports": root / "reports",
    }


def _input_failure(errors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "code": ERROR_CODE,
        "isolated_failures": [dict(error) for error in errors],
    }


def _persist_input_failure(
    root: Path, errors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    failure = _input_failure(errors)
    reports = _paths(root)["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "input_failures.json", failure)
    return failure


def load_project(root: Path) -> tuple[AppConfig, Session, list[dict[str, Any]], list[dict[str, str]]]:
    paths = _paths(root)
    issues: list[dict[str, str]] = []
    for key in ("config", "session", "commands"):
        if not paths[key].exists():
            issues.append(
                {
                    "code": ERROR_CODE,
                    "path": str(paths[key]),
                    "message": "file does not exist; run generate first",
                }
            )
    if issues:
        return AppConfig.from_dict({"channel_weights": {}}), Session.from_dict(
            {"session_id": "", "turns": [], "evidence": []}
        ), [], issues
    try:
        config = AppConfig.from_dict(load_json(paths["config"]))
        session = Session.from_dict(load_json(paths["session"]))
        commands = load_jsonl(paths["commands"])
    except Exception as exc:  # parse/type errors are input failures
        return AppConfig.from_dict({"channel_weights": {}}), Session.from_dict(
            {"session_id": "", "turns": [], "evidence": []}
        ), [], [
            {
                "code": ERROR_CODE,
                "path": "input",
                "message": f"cannot parse input: {exc}",
            }
        ]
    issues.extend(validate_config(config))
    issues.extend(validate_session(config, session))
    issues.extend(validate_commands(commands))
    return config, session, commands, issues


def _prefix_hash(commands: Sequence[Mapping[str, Any]], cursor: int) -> str:
    return sha256_json(list(commands[:cursor]))


def _canonical_trace_hash(result: Mapping[str, Any]) -> str:
    return sha256_json(result)


def _deep_diff(
    left: Any, right: Any, path: str = "$", diffs: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if diffs is None:
        diffs = []
    if len(diffs) >= 50:
        return diffs
    if type(left) is not type(right):
        diffs.append(
            {
                "path": path,
                "left_type": type(left).__name__,
                "right_type": type(right).__name__,
            }
        )
        return diffs
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            next_path = f"{path}.{key}"
            if key not in left or key not in right:
                diffs.append(
                    {
                        "path": next_path,
                        "missing": "left" if key not in left else "right",
                    }
                )
            else:
                _deep_diff(left[key], right[key], next_path, diffs)
    elif isinstance(left, list):
        if len(left) != len(right):
            diffs.append(
                {
                    "path": path,
                    "left_length": len(left),
                    "right_length": len(right),
                }
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _deep_diff(left_item, right_item, f"{path}[{index}]", diffs)
    elif left != right:
        diffs.append({"path": path, "left": left, "right": right})
    return diffs


def _find_turn(result: Mapping[str, Any], turn_id: int) -> Mapping[str, Any]:
    for turn in result["turns"]:
        if turn["turn_id"] == turn_id:
            return turn
    raise KeyError(turn_id)


def _build_equivalence_report(
    root: Path,
    config: AppConfig,
    session: Session,
    commands: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    incremental = run_sequence(config, session, commands, mode="incremental")
    full = run_sequence(config, session, commands, mode="full")
    trace_checks: list[dict[str, Any]] = []
    for increment_step, full_step in zip(
        incremental["trace"], full["trace"], strict=True
    ):
        diffs = _deep_diff(increment_step["result"], full_step["result"])
        trace_checks.append(
            {
                "cursor": increment_step["cursor"],
                "equal": not diffs,
                "canonical_hash_incremental": _canonical_trace_hash(
                    increment_step["result"]
                ),
                "canonical_hash_full": _canonical_trace_hash(
                    full_step["result"]
                ),
                "diffs": diffs,
            }
        )
    final = incremental["final_result"]
    pending_steps = [
        {
            "cursor": step["cursor"],
            "todo": step["result"]["todo"],
        }
        for step in incremental["trace"]
        if step["result"].get("todo")
    ]
    failure_steps: list[dict[str, Any]] = []
    for step in incremental["trace"]:
        cursor = step["cursor"]
        current_failures = [
            failure
            for failure in step["result"].get("failures", [])
            if failure.get("seq") == cursor
        ]
        if current_failures:
            failure_steps.append(
                {"cursor": cursor, "failures": current_failures}
            )
    report = {
        "schema_version": 1,
        "mode_pair": ["incremental", "full"],
        "all_cursors_equal": all(item["equal"] for item in trace_checks),
        "cursor_count": len(trace_checks),
        "trace_checks": trace_checks,
        "final_equal": incremental["final_result"] == full["final_result"],
        "pending_cursors": pending_steps,
        "failure_cursors": failure_steps,
        "final_todo_empty": not final["todo"],
        "final_failure_codes": [
            item["code"] for item in final.get("failures", [])
        ],
        "command_count": len(commands),
        "config_fingerprint": config_fingerprint(config),
        "initial_evidence_fingerprint": evidence_library_fingerprint(session),
    }
    return report, incremental, full


def _build_precision_report(
    config: AppConfig,
    session: Session,
    incremental: Mapping[str, Any],
) -> dict[str, Any]:
    trace = incremental["trace"]
    fixed_switch_checks: list[dict[str, Any]] = []
    if len(trace) > 9:
        before = trace[8]["result"]
        after = trace[9]["result"]
        for turn_id in sorted(
            {turn["turn_id"] for turn in before["turns"]}
            & {turn["turn_id"] for turn in after["turns"]}
        ):
            old_turn = _find_turn(before, turn_id)
            new_turn = _find_turn(after, turn_id)
            old_conf = old_turn["confidence"].get("confidence")
            new_conf = new_turn["confidence"].get("confidence")
            old_units = old_turn["recall"]["evidence"][0]["authority_score_units"]
            new_units = new_turn["recall"]["evidence"][0]["authority_score_units"]
            fixed_switch_checks.append(
                {
                    "turn_id": turn_id,
                    "old_domain": before["fixed_point"],
                    "new_domain": after["fixed_point"],
                    "old_recall_units": old_units,
                    "new_recall_units": new_units,
                    "recall_integer_changed": old_units != new_units,
                    "old_confidence": old_conf,
                    "new_confidence": new_conf,
                    "confidence_decimal_changed": (
                        old_conf is not None
                        and new_conf is not None
                        and old_conf != new_conf
                    ),
                    "new_status": new_turn["status"],
                }
            )
    changed_confidence = [
        item for item in fixed_switch_checks if item["confidence_decimal_changed"]
    ]
    changed_units = [
        item for item in fixed_switch_checks if item["recall_integer_changed"]
    ]
    final = incremental["final_result"]
    final_engine = DiagnosticEngine.from_state(incremental["state"])
    exact_rule = None
    if len(final["turns"]) >= 2:
        boundary = final["turns"][1]["boundary_before"]
        equal_threshold = boundary["combined_score_bp"]
        from dataclasses import replace

        equal_config = replace(
            final_engine.config,
            topic=replace(
                final_engine.config.topic,
                switch_threshold_bp=equal_threshold,
            ),
        )
        turns_by_id = {
            turn.turn_id: turn for turn in final_engine.session.turns
        }
        dedupe_by_id = {
            turn["turn_id"]: turn["dedupe"] for turn in final["turns"]
        }
        left_id = int(boundary["left_turn_id"])
        right_id = int(boundary["right_turn_id"])
        equal_boundary = compute_boundary(
            turns_by_id[left_id],
            turns_by_id[right_id],
            dedupe_by_id[left_id],
            dedupe_by_id[right_id],
            equal_config,
        )
        exact_rule = {
            "left_turn_id": left_id,
            "right_turn_id": right_id,
            "query_similarity_bp": equal_boundary["query_similarity_bp"],
            "evidence_similarity_bp": equal_boundary[
                "evidence_similarity_bp"
            ],
            "combined_score_bp": equal_boundary["combined_score_bp"],
            "threshold_bp": equal_threshold,
            "exact_equal": equal_boundary["exact_equal"],
            "split": equal_boundary["split"],
            "decision": equal_boundary["decision"],
            "equal_rule": equal_boundary["equal_rule"],
        }
    precision = {
        "schema_version": 1,
        "fixed_point_switch": {
            "checked_turns": len(fixed_switch_checks),
            "recall_integer_changed_turns": [
                item["turn_id"] for item in changed_units
            ],
            "confidence_decimal_changed_turns": [
                item["turn_id"] for item in changed_confidence
            ],
            "real_change_verified": bool(changed_confidence and changed_units),
            "checks": fixed_switch_checks,
        },
        "exact_equal_topic_rule": exact_rule,
        "invalidation_precision": {
            "query_replace_cursor": 2,
            "query_replace_invalidated_turns": trace[2]["result"][
                "invalidation"
            ]["invalidated_turns"],
            "query_replace_reused_turns": trace[2]["result"]["invalidation"][
                "reused_turns"
            ],
        },
        "budget": {
            "pending_cursor": 9,
            "pending_todo": trace[9]["result"]["todo"] if len(trace) > 9 else [],
            "final_todo": final["todo"],
        },
    }
    return precision


def _snapshot_run(
    root: Path,
    *,
    mode: str,
    reset: bool,
    resume: bool,
    crash_at: int | None,
) -> dict[str, Any]:
    config, session, commands, issues = load_project(root)
    if issues:
        raise ValueError(
            json.dumps(_persist_input_failure(root, issues), ensure_ascii=False)
        )
    paths = _paths(root)
    paths["snapshots"].mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(paths["snapshots"])
    if reset and resume:
        raise ValueError("--reset and --resume are mutually exclusive")
    if crash_at is not None:
        os.environ["EVIDENCE_DIAG_CRASH_AT"] = (
            f"after-stage-write:{crash_at}"
        )
    base_config_hash = config_fingerprint(config)
    base_evidence_fp = evidence_library_fingerprint(session)
    parent_version = store.current_version()
    engine: DiagnosticEngine
    start_cursor = 0
    recovery: dict[str, Any]
    if resume:
        bundle = store.load_latest_coherent()
        if bundle is None:
            recovery = {
                "decision": "full_recompute",
                "reason": "no_coherent_snapshot",
            }
            store.reset()
            parent_version = 0
            engine = DiagnosticEngine(config, session)
        else:
            cursor = int(bundle.manifest["command_cursor"])
            decision = store.recover(
                base_config_hash=base_config_hash,
                base_evidence_fingerprint=base_evidence_fp,
                command_prefix_hash=_prefix_hash(commands, cursor),
            )
            recovery = {
                "decision": decision["decision"],
                "reason": decision["reason"],
                "snapshot_global_version": bundle.manifest["global_version"],
                "snapshot_cursor": cursor,
            }
            if decision["decision"] == "resume":
                engine = bundle.engine()
                start_cursor = engine.command_cursor
                parent_version = int(bundle.manifest["global_version"])
            else:
                store.reset()
                parent_version = 0
                engine = DiagnosticEngine(config, session)
    else:
        if reset:
            store.reset()
        elif store.current_version() != 0:
            raise ValueError(
                "snapshots exist; pass --resume or --reset"
            )
        parent_version = 0
        engine = DiagnosticEngine(config, session)
        recovery = {"decision": "fresh", "reason": "new_snapshot_sequence"}

    commits: list[dict[str, Any]] = []
    if start_cursor == 0:
        computed = engine.compute(force_recompute=(mode == "full"))
        manifest = store.commit(
            state=engine.export_state(),
            result=computed.result,
            execution=computed.execution,
            expected_parent_version=parent_version,
            base_config_hash=base_config_hash,
            base_evidence_fingerprint=base_evidence_fp,
            command_prefix_hash=_prefix_hash(commands, 0),
        )
        parent_version = int(manifest["global_version"])
        commits.append(
            {
                "cursor": 0,
                "global_version": parent_version,
                "manifest_hash": manifest["manifest_hash"],
            }
        )

    for command in commands[start_cursor:]:
        outcome = engine.apply_command(command)
        if outcome["valid"]:
            computed = engine.compute(force_recompute=(mode == "full"))
            result = computed.result
            execution = computed.execution
        else:
            result = engine.invalid_result_after_failure()
            execution = {
                "mode": mode,
                "stages": [],
                "command_rejected": True,
            }
        cursor = int(engine.command_cursor)
        manifest = store.commit(
            state=engine.export_state(),
            result=result,
            execution=execution,
            expected_parent_version=parent_version,
            base_config_hash=base_config_hash,
            base_evidence_fingerprint=base_evidence_fp,
            command_prefix_hash=_prefix_hash(commands, cursor),
        )
        parent_version = int(manifest["global_version"])
        commits.append(
            {
                "cursor": cursor,
                "global_version": parent_version,
                "manifest_hash": manifest["manifest_hash"],
            }
        )
    final = engine.last_result
    report = {
        "schema_version": 1,
        "mode": mode,
        "recovery": recovery,
        "commits": commits,
        "final_cursor": engine.command_cursor,
        "final_global_version": parent_version,
        "final_result_hash": sha256_json(final),
        "final_todo": final["todo"] if final else [],
        "failures": final["failures"] if final else [],
    }
    write_json(paths["reports"] / "last_snapshot_run.json", report)
    if final is not None:
        write_json(paths["reports"] / "last_snapshot_result.json", final)
    return report


def _run_subprocess(
    root: Path, args: Sequence[str], *, env_extra: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (package_root, str(root), existing_pythonpath)
        if value
    )
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "evidence_diag", "--root", str(root), *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_demo(root: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    generation = generate_sample(root, seed)
    config, session, commands, issues = load_project(root)
    if issues:
        failure = _persist_input_failure(root, issues)
        raise ValueError(json.dumps(failure, ensure_ascii=False))
    equivalence, incremental, full = _build_equivalence_report(
        root, config, session, commands
    )
    precision = _build_precision_report(config, session, incremental)
    final_engine = DiagnosticEngine.from_state(incremental["state"])
    diagnostic = build_diagnostic_report(
        final_engine.config, final_engine.session
    )
    reports = _paths(root)["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "equivalence.json", equivalence)
    write_json(reports / "precision.json", precision)
    write_json(reports / "diagnostic.json", diagnostic)
    write_json(reports / "final_incremental.json", incremental["final_result"])
    write_json(reports / "final_full.json", full["final_result"])

    store = SnapshotStore(_paths(root)["snapshots"])
    store.reset()
    crash_cursor = min(10, len(commands) + 1)
    crash_env = {
        "EVIDENCE_DIAG_CRASH_AT": f"after-stage-write:{crash_cursor}"
    }
    crashed = _run_subprocess(
        root,
        ["snapshot-run", "--mode", "incremental", "--reset"],
        env_extra=crash_env,
    )
    if crashed.returncode == 0:
        raise RuntimeError("crash simulation did not terminate the process")
    before_resume = store.load_latest_coherent()
    if before_resume is None:
        raise RuntimeError("no coherent snapshot after forced termination")
    orphan_temp_before_resume = len(
        list(_paths(root)["snapshots"].glob(".tmp-*"))
    )
    store.cleanup_temp()
    resumed = _run_subprocess(
        root, ["snapshot-run", "--mode", "incremental", "--resume"]
    )
    if resumed.returncode != 0:
        raise RuntimeError(
            f"resume failed ({resumed.returncode}): {resumed.stderr}"
        )
    resume_report = json.loads(resumed.stdout)
    recovered_bundle = store.load_latest_coherent()
    if recovered_bundle is None:
        raise RuntimeError("recovered snapshot missing")
    recovered_result = recovered_bundle.result
    expected_config_hash = config_fingerprint(config)
    expected_evidence_fp = evidence_library_fingerprint(session)
    mismatch = store.recover(
        base_config_hash="0" * 64,
        base_evidence_fingerprint=expected_evidence_fp,
        command_prefix_hash=_prefix_hash(
            commands, int(recovered_bundle.manifest["command_cursor"])
        ),
    )
    recovery_report = {
        "schema_version": 1,
        "crash_exit_code": crashed.returncode,
        "crash_at_cursor": crash_cursor,
        "coherent_version_before_resume": before_resume.manifest[
            "global_version"
        ],
        "cursor_before_resume": before_resume.manifest["command_cursor"],
        "orphan_temp_directories_before_resume": orphan_temp_before_resume,
        "orphan_temp_directories_after_resume": len(
            list(_paths(root)["snapshots"].glob(".tmp-*"))
        ),
        "resume_mode": resume_report["recovery"],
        "resumed_cursor": resume_report["final_cursor"],
        "resumed_global_version": resume_report["final_global_version"],
        "result_equal_to_incremental": recovered_result
        == incremental["final_result"],
        "result_hash_recovered": sha256_json(recovered_result),
        "result_hash_expected": sha256_json(incremental["final_result"]),
        "config_mismatch_decision": {
            "decision": mismatch["decision"],
            "reason": mismatch["reason"],
        },
        "snapshot_manifest_verified": True,
        "expected_config_hash": expected_config_hash,
    }
    if not recovery_report["result_equal_to_incremental"]:
        raise RuntimeError("recovered snapshot result differs from full replay")
    write_json(reports / "recovery.json", recovery_report)
    summary = {
        "schema_version": 1,
        "generation": generation,
        "checks": {
            "all_cursor_results_equal": equivalence["all_cursors_equal"],
            "cursor_count": equivalence["cursor_count"],
            "budget_pending_cursor": 9,
            "budget_pending_count": len(
                precision["budget"]["pending_todo"]
            ),
            "final_todo_count": len(incremental["final_result"]["todo"]),
            "failure_codes": equivalence["final_failure_codes"],
            "fixed_point_change_verified": precision["fixed_point_switch"][
                "real_change_verified"
            ],
            "diagnostic_first_exceed_count": len(
                diagnostic["first_exceed_by_turn"]
            ),
            "diagnostic_flip_count": diagnostic["flip_count"],
            "diagnostic_primary_causes": sorted(
                {item["primary_cause"] for item in diagnostic["flips"]}
            ),
            "snapshot_crash_recovered": recovery_report[
                "result_equal_to_incremental"
            ],
            "config_mismatch_full_recompute": recovery_report[
                "config_mismatch_decision"
            ]["decision"]
            == "full_recompute",
        },
        "reports": {
            "equivalence": "reports/equivalence.json",
            "precision": "reports/precision.json",
            "diagnostic": "reports/diagnostic.json",
            "recovery": "reports/recovery.json",
            "final_result": "reports/final_incremental.json",
        },
    }
    write_json(reports / "demo_summary.json", summary)
    return summary


def _diagnose(root: Path, at_cursor: int | None) -> dict[str, Any]:
    config, session, commands, issues = load_project(root)
    if issues:
        failure = _persist_input_failure(root, issues)
        return failure
    sequence = run_sequence(config, session, commands, mode="incremental")
    engine = DiagnosticEngine.from_state(sequence["state"])
    if at_cursor is not None:
        engine = DiagnosticEngine(config, session)
        for command in commands:
            if command["seq"] > at_cursor:
                break
            result = engine.apply_command(command)
            if result["valid"]:
                engine.compute()
            else:
                engine.invalid_result_after_failure()
    report = build_diagnostic_report(engine.config, engine.session)
    write_json(_paths(root)["reports"] / "diagnostic.json", report)
    return report


def _check_environment(root: Path) -> dict[str, Any]:
    import platform

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "root": str(root),
    }
    try:
        import numpy

        result["numpy"] = numpy.__version__
    except Exception as exc:
        result["numpy_error"] = str(exc)
    try:
        from .gui import _bootstrap_local_tk

        result["tk_bootstrap"] = _bootstrap_local_tk(root)
        import tkinter

        result["tkinter"] = tkinter.TkVersion
        result["tk_patchlevel"] = tkinter.Tcl().eval("info patchlevel")
    except Exception as exc:
        result["tkinter_error"] = str(exc)
        result["tkinter_hint"] = (
            "run scripts/install_local_tk.sh or sudo apt install python3-tk"
        )
    for name, path in _paths(root).items():
        if name in ("config", "session", "commands"):
            result[f"{name}_exists"] = path.exists()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evidence-diag",
        description="Deterministic conversational evidence-chain diagnostics",
    )
    parser.add_argument("--root", default=str(Path.cwd()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic sample")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run = subparsers.add_parser("run", help="run command stream in memory")
    run.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    run.add_argument(
        "--trace",
        action="store_true",
        help="also write every cursor result to reports/run_<mode>_trace.json",
    )

    snapshot_run = subparsers.add_parser(
        "snapshot-run", help="run command stream with atomic snapshots"
    )
    snapshot_run.add_argument(
        "--mode", choices=("incremental", "full"), default="incremental"
    )
    snapshot_run.add_argument("--reset", action="store_true")
    snapshot_run.add_argument("--resume", action="store_true")
    snapshot_run.add_argument("--crash-at", type=int)

    diagnose = subparsers.add_parser("diagnose", help="write fixed-point diagnostic")
    diagnose.add_argument("--at-cursor", type=int)

    subparsers.add_parser("check-env", help="report environment compatibility")
    subparsers.add_parser("demo", help="run complete reproducible demo")

    gui = subparsers.add_parser("gui", help="open optional Tk dashboard")
    gui.add_argument("--demo", action="store_true", help="run demo before opening")
    gui.add_argument(
        "--smoke-test",
        action="store_true",
        help="create and destroy the Tk window without entering mainloop",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "generate":
            print(
                json.dumps(
                    generate_sample(root, args.seed),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "check-env":
            print(
                json.dumps(
                    _check_environment(root),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run":
            config, session, commands, issues = load_project(root)
            if issues:
                print(
                    json.dumps(
                        _persist_input_failure(root, issues),
                        ensure_ascii=False,
                    )
                )
                return 2
            sequence = run_sequence(config, session, commands, mode=args.mode)
            paths = _paths(root)
            write_json(paths["reports"] / f"run_{args.mode}_final.json", sequence["final_result"])
            if args.trace:
                write_json(
                    paths["reports"] / f"run_{args.mode}_trace.json",
                    sequence["trace"],
                )
            summary = {
                "mode": args.mode,
                "cursor": sequence["final_result"]["command_cursor"],
                "turns": len(sequence["final_result"]["turns"]),
                "topics": len(sequence["final_result"]["topics"]),
                "todo": sequence["final_result"]["todo"],
                "failure_codes": [
                    item["code"] for item in sequence["final_result"]["failures"]
                ],
                "output": f"reports/run_{args.mode}_final.json",
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "snapshot-run":
            report = _snapshot_run(
                root,
                mode=args.mode,
                reset=args.reset,
                resume=args.resume,
                crash_at=args.crash_at,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "diagnose":
            report = _diagnose(root, args.at_cursor)
            print(
                json.dumps(
                    {
                        "report": "reports/diagnostic.json",
                        "actual_domain": report.get("actual_domain"),
                        "reference_domain": report.get("reference_domain"),
                        "first_exceed_count": len(
                            report.get("first_exceed_by_turn", {})
                        ),
                        "flip_count": report.get("flip_count"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "demo":
            summary = run_demo(root)
            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "gui":
            if args.demo:
                run_demo(root)
            from .gui import run_gui

            return run_gui(root, smoke_test=args.smoke_test)
        parser.error("unknown command")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.SubprocessError as exc:
        print(f"subprocess failure: {exc}", file=sys.stderr)
        return 3
    return 0
