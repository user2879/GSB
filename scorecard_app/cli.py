from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import VALID_CONFLICT_RULES, load_yaml, normalize_direction, validate_inputs
from .engine import ALGORITHM_VERSION, REPORT_SCHEMA_VERSION, ScorecardEngine
from .errors import E_CONFIG_INVALID
from .snapshot import atomic_write_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scorecard-lab",
        description="Deterministic local scorecard binning, WOE and monotonic solver",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {ALGORITHM_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute one full or resumed run")
    _add_common_arguments(run)

    shell = subparsers.add_parser("shell", help="interactive edit/run loop")
    _add_common_arguments(shell)
    shell.add_argument("--commands", help="read shell commands from a UTF-8 text file")

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/experiment.yaml", help="experiment YAML path")
    parser.add_argument("--data", default="configs/synthetic.yaml", help="synthetic data description YAML path")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="snapshot/cache directory (default: .scorecard-state)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: <state-dir>/last_report.json)",
    )
    parser.add_argument(
        "--revision",
        action="store_true",
        help="treat changed config/data as an intentional edit and reuse content-addressed caches",
    )


def _load_inputs(config_path: str, data_path: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    try:
        experiment = load_yaml(config_path)
    except Exception as exc:
        return None, None, f"cannot load experiment YAML {config_path}: {exc}"
    try:
        schema = load_yaml(data_path)
    except Exception as exc:
        return None, None, f"cannot load synthetic data YAML {data_path}: {exc}"
    return experiment, schema, None


def _input_error_report(message: str, config_path: str, data_path: str) -> dict[str, Any]:
    def digest(path: str) -> str:
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return hashlib.sha256(path.encode("utf-8")).hexdigest()

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "error",
        "error": "config_invalid",
        "business_codes": [E_CONFIG_INVALID],
        "snapshot_version": 0,
        "recovery_point_version": 0,
        "input": {
            "config_fingerprint": digest(config_path),
            "data_fingerprint": digest(data_path),
        },
        "snapshot": {"version": 0, "recovery_point_version": 0, "last_completed_stage": None},
        "recovery": {
            "recovered": False,
            "refused": True,
            "resume_version": 0,
            "resumed_from_stage": "none",
        },
        "columns": [],
        "evaluation": None,
        "cache": {
            "hits": 0,
            "misses": 0,
            "invalidated_artifacts": 0,
            "invalidated_columns": [],
            "by_column": {},
        },
        "failures": [
            {
                "column": "",
                "stage": "config",
                "code": E_CONFIG_INVALID,
                "exception_type": None,
                "message": message,
            }
        ],
    }


def _state_dir(args: argparse.Namespace, experiment: dict[str, Any] | None) -> Path:
    if args.state_dir:
        return Path(args.state_dir)
    configured = ((experiment or {}).get("runtime") or {}).get("state_dir")
    return Path(configured or ".scorecard-state")


def _report_path(args: argparse.Namespace, state_dir: Path) -> Path:
    return Path(args.report) if args.report else state_dir / "last_report.json"


def _install_signal_handlers(engine_holder: dict[str, Any]) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        engine_holder["stop_requested"] = True
        engine = engine_holder.get("engine")
        if engine is not None:
            engine.request_stop()

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signum = getattr(signal, name)
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _print_summary(report: dict[str, Any], report_path: Path) -> None:
    failures = len(report.get("failures", []))
    cache = report.get("cache", {})
    evaluation = report.get("evaluation") or {}
    overall = evaluation.get("overall") or {}
    auc = overall.get("auc")
    snapshot = report.get("snapshot", {})
    print(
        f"[{report.get('status')}] snapshot={snapshot.get('version', 0)} "
        f"recovery_point={snapshot.get('recovery_point_version', 0)} "
        f"columns={len(report.get('columns', []))} failures={failures} "
        f"auc={auc} cache={cache.get('hits', 0)}/{cache.get('misses', 0)} "
        f"report={report_path}"
    )
    for failure in report.get("failures", []):
        print(
            f"  {failure.get('code')} column={failure.get('column') or '-'} "
            f"stage={failure.get('stage')} type={failure.get('exception_type') or '-'} "
            f"{failure.get('message')}"
        )


def _exit_code(report: dict[str, Any]) -> int:
    if report.get("status") in {"ok", "interrupted"}:
        return 0
    return 1


def _run_once(args: argparse.Namespace, engine_holder: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    experiment, schema, error = _load_inputs(args.config, args.data)
    state_dir = _state_dir(args, experiment)
    report_path = _report_path(args, state_dir)
    if error is not None or experiment is None or schema is None:
        report = _input_error_report(error or "input loading failed", args.config, args.data)
        atomic_write_json(report_path, report, indent=2)
        return report, report_path
    engine = ScorecardEngine(experiment, schema, state_dir, allow_revision=args.revision)
    if engine_holder.get("stop_requested"):
        engine.request_stop()
    engine_holder["engine"] = engine
    report = engine.run(report_path=report_path, allow_revision=args.revision)
    engine_holder["engine"] = None
    return report, report_path


def _shell_help() -> None:
    print(
        "\n".join(
            [
                "Commands:",
                "  run                              rerun current state",
                "  max-bins <n> [column]            set global or per-column bin cap",
                "  min-ratio <r> [column]           set global or per-column minimum bin share",
                "  direction <column> <ascending|descending|none>",
                "  priority <rule[,rule...]>         rules: " + ", ".join(sorted(VALID_CONFLICT_RULES)),
                "  max-rounds <n>                   set monotonic merge round ceiling",
                "  replace-column <column> <offset> regenerate one column with a new seed offset",
                "  save <experiment.yaml> <data.yaml>",
                "  show [column]                    show current report summary or one column",
                "  report [path]                    write current report to a path",
                "  help | quit",
            ]
        )
    )


def _mutate_and_run(
    experiment: dict[str, Any],
    schema: dict[str, Any],
    state_dir: Path,
    report_path: Path,
    engine_holder: dict[str, Any],
) -> dict[str, Any] | None:
    _, failures = validate_inputs(experiment, schema)
    if failures:
        for failure in failures:
            print(f"rejected: {failure.code} column={failure.column or '-'} {failure.message}")
        return None
    engine = ScorecardEngine(experiment, schema, state_dir, allow_revision=True)
    if engine_holder.get("stop_requested"):
        engine.request_stop()
    engine_holder["engine"] = engine
    report = engine.run(report_path=report_path, allow_revision=True)
    engine_holder["last_report"] = report
    engine_holder["engine"] = None
    return report


def _handle_shell_command(
    line: str,
    experiment: dict[str, Any],
    schema: dict[str, Any],
    state_dir: Path,
    report_path: Path,
    engine_holder: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    parts = shlex.split(line)
    if not parts:
        return True, None
    command = parts[0].lower()
    if command in {"quit", "exit"}:
        return False, None
    if command in {"help", "?"}:
        _shell_help()
        return True, None
    if command == "show":
        report = engine_holder.get("last_report") or {}
        if len(parts) == 1:
            for column in report.get("columns", []):
                print(
                    f"{column['name']}: type={column['type']} bins={column['bin_count']} "
                    f"iv={column['iv']} direction={column['direction']} "
                    f"corrections={column['monotonic_correction_count']} status={column['status']}"
                )
        else:
            target = parts[1]
            matches = [column for column in report.get("columns", []) if column["name"] == target]
            print(json.dumps(matches[0], ensure_ascii=False, indent=2) if matches else f"column not found: {target}")
        return True, None
    if command == "report":
        if len(parts) > 2:
            print("usage: report [path]")
            return True, None
        target = Path(parts[1]) if len(parts) == 2 else report_path
        report = engine_holder.get("last_report")
        if report is None:
            print("no report is available yet")
        else:
            atomic_write_json(target, report, indent=2)
            print(f"report written: {target}")
        return True, None
    if command == "save":
        if len(parts) != 3:
            print("usage: save <experiment.yaml> <data.yaml>")
            return True, None
        _write_yaml_atomic(Path(parts[1]), experiment)
        _write_yaml_atomic(Path(parts[2]), schema)
        print(f"saved effective config to {parts[1]} and {parts[2]}")
        return True, None
    if command == "run":
        report = _mutate_and_run(experiment, schema, state_dir, report_path, engine_holder)
        if report is None:
            return True, None
        _print_summary(report, report_path)
        return report.get("status") != "interrupted", report
    if command in {"max-bins", "min-ratio", "max-rounds", "direction", "priority", "replace-column"}:
        original_experiment = copy.deepcopy(experiment)
        original_schema = copy.deepcopy(schema)
        try:
            if command == "max-bins":
                if len(parts) not in {2, 3}:
                    raise ValueError("usage: max-bins <n> [column]")
                value = int(parts[1])
                if value <= 0:
                    raise ValueError("max-bins must be positive")
                if len(parts) == 2:
                    experiment.setdefault("binning", {})["max_bins"] = value
                else:
                    setting = experiment.setdefault("columns", {}).setdefault(parts[2], {})
                    setting["max_bins"] = value
            elif command == "min-ratio":
                if len(parts) not in {2, 3}:
                    raise ValueError("usage: min-ratio <r> [column]")
                value = float(parts[1])
                if not (0.0 <= value <= 1.0):
                    raise ValueError("min-ratio must be in [0, 1]")
                if len(parts) == 2:
                    experiment.setdefault("binning", {})["min_bin_ratio"] = value
                else:
                    setting = experiment.setdefault("columns", {}).setdefault(parts[2], {})
                    setting["min_bin_ratio"] = value
            elif command == "max-rounds":
                if len(parts) != 2:
                    raise ValueError("usage: max-rounds <n>")
                value = int(parts[1])
                if value < 0:
                    raise ValueError("max-rounds must be non-negative")
                experiment.setdefault("monotonic", {})["max_rounds"] = value
            elif command == "direction":
                if len(parts) != 3:
                    raise ValueError("usage: direction <column> <ascending|descending|none>")
                direction = normalize_direction(parts[2])
                if direction is None:
                    raise ValueError("invalid direction")
                column = parts[1]
                if column not in experiment.get("features", []):
                    raise ValueError(f"unknown feature: {column}")
                experiment.setdefault("columns", {}).setdefault(column, {})["monotonic_direction"] = direction
            elif command == "priority":
                if len(parts) != 2:
                    raise ValueError("usage: priority <rule[,rule...]>")
                rules = [item.strip() for item in parts[1].split(",") if item.strip()]
                if not rules or any(item not in VALID_CONFLICT_RULES for item in rules):
                    raise ValueError("invalid conflict priority list")
                experiment.setdefault("monotonic", {})["conflict_priority"] = rules
            elif command == "replace-column":
                if len(parts) != 3:
                    raise ValueError("usage: replace-column <column> <seed_offset>")
                column = parts[1]
                offset = int(parts[2])
                matches = [item for item in schema.get("columns", []) if item.get("name") == column]
                if not matches:
                    raise ValueError(f"unknown column: {column}")
                matches[0]["seed_offset"] = offset
        except (TypeError, ValueError) as exc:
            experiment.clear()
            experiment.update(original_experiment)
            schema.clear()
            schema.update(original_schema)
            print(f"rejected: {exc}")
            return True, None
        report = _mutate_and_run(experiment, schema, state_dir, report_path, engine_holder)
        if report is None:
            experiment.clear()
            experiment.update(original_experiment)
            schema.clear()
            schema.update(original_schema)
            return True, None
        _print_summary(report, report_path)
        return report.get("status") != "interrupted", report
    print(f"unknown command: {command}. Type 'help' for commands.")
    return True, None


def _write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_shell(args: argparse.Namespace, engine_holder: dict[str, Any]) -> int:
    experiment, schema, error = _load_inputs(args.config, args.data)
    state_dir = _state_dir(args, experiment)
    report_path = _report_path(args, state_dir)
    if error is not None or experiment is None or schema is None:
        report = _input_error_report(error or "input loading failed", args.config, args.data)
        atomic_write_json(report_path, report, indent=2)
        _print_summary(report, report_path)
        return 1

    report = _mutate_and_run(experiment, schema, state_dir, report_path, engine_holder) if args.revision else None
    if report is None:
        engine = ScorecardEngine(experiment, schema, state_dir, allow_revision=False)
        if engine_holder.get("stop_requested"):
            engine.request_stop()
        engine_holder["engine"] = engine
        report = engine.run(report_path=report_path, allow_revision=False)
        engine_holder["last_report"] = report
        engine_holder["engine"] = None
    _print_summary(report, report_path)

    if args.commands:
        with Path(args.commands).open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        interactive = False
    else:
        lines = []
        interactive = sys.stdin.isatty()

    if interactive:
        _shell_help()
        while True:
            try:
                line = input("scorecard> ")
            except EOFError:
                break
            keep_going, _ = _handle_shell_command(
                line, experiment, schema, state_dir, report_path, engine_holder
            )
            if not keep_going:
                break
    else:
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            keep_going, _ = _handle_shell_command(
                stripped, experiment, schema, state_dir, report_path, engine_holder
            )
            if not keep_going:
                break
    final_report = engine_holder.get("last_report") or {}
    return 0 if final_report.get("status") in {None, "ok", "interrupted"} else 1


def main(argv: list[str] | None = None) -> int:
    engine_holder: dict[str, Any] = {
        "engine": None,
        "last_report": None,
        "stop_requested": False,
    }
    previous_handlers = _install_signal_handlers(engine_holder)
    try:
        parser = _build_parser()
        args = parser.parse_args(argv)
        if args.command == "run":
            report, report_path = _run_once(args, engine_holder)
            _print_summary(report, report_path)
            return _exit_code(report)
        if args.command == "shell":
            return _run_shell(args, engine_holder)
        parser.error(f"unknown command: {args.command}")
        return 2
    finally:
        _restore_signal_handlers(previous_handlers)
