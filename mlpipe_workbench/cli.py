"""Command-line interface for the workbench and its repeatable demo."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from .demo_data import (
    amplify_outliers,
    corrupt_sample,
    materialise_demo_files,
    out_of_bounds_pipeline,
    write_json,
)
from .engine import Engine
from .errors import (
    E_DATA,
    E_ENV,
    E_OUTLIER,
    E_PARAM,
    E_REPLAY,
    E_SPEC,
    E_STATE,
    WorkbenchError,
)
from .ledger import Ledger
from .loader import load_workspace
from .logs import JsonlLogger
from .recovery import RecoveryManager


def _exit_for_code(code: str) -> int:
    return {
        E_SPEC: 2,
        E_DATA: 2,
        E_PARAM: 2,
        E_OUTLIER: 3,
        E_REPLAY: 4,
        E_STATE: 5,
        E_ENV: 6,
    }.get(code, 1)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record_standalone_error(
    db_path: str | Path,
    log_path: str | Path,
    error: WorkbenchError,
    run_id: str | None = None,
) -> None:
    logger = JsonlLogger(log_path)
    try:
        with Ledger(db_path, logger) as ledger:
            ledger.record_error(run_id, error)
    except Exception:
        logger.write(error.code, {"run_id": run_id, **error.to_dict(), "ledger_write": "failed"})


def _apply_injections(
    args: argparse.Namespace,
    data_path: Path,
    pipeline_path: Path,
) -> tuple[Path, Path, str | None]:
    modes = [
        bool(args.inject_data_corrupt),
        bool(args.inject_param_bounds),
        args.inject_outlier_ratio is not None,
    ]
    if sum(1 for mode in modes if mode) > 1:
        raise WorkbenchError(
            E_PARAM,
            "only one anomaly injection may be active per invocation",
            param="injection",
            location="cli.injection",
        )
    if not any(modes):
        return data_path, pipeline_path, None
    injection_dir = Path(args.injection_dir)
    injection_dir.mkdir(parents=True, exist_ok=True)
    if args.inject_data_corrupt:
        dataset = corrupt_sample(_read_json(data_path))
        target = injection_dir / "sample.injected-corrupt.json"
        write_json(target, dataset)
        return target, pipeline_path, "data_corruption"
    if args.inject_param_bounds:
        pipeline = out_of_bounds_pipeline(_read_json(pipeline_path))
        target = injection_dir / "pipeline.injected-param.json"
        write_json(target, pipeline)
        return data_path, target, "parameter_out_of_bounds"
    if args.inject_outlier_ratio is not None:
        if not 1 <= args.inject_outlier_ratio <= 1000:
            raise WorkbenchError(
                E_PARAM,
                "inject-outlier-ratio must be in [1, 1000] permille",
                param="inject_outlier_ratio",
                param_value=args.inject_outlier_ratio,
                location="cli.inject_outlier_ratio",
            )
        dataset = _read_json(data_path)
        pipeline = _read_json(pipeline_path)
        clip_stage = next(
            (stage for stage in pipeline.get("stages", []) if stage.get("type") == "clip_outliers"),
            None,
        )
        if clip_stage is None:
            raise WorkbenchError(
                E_PARAM,
                "outlier injection requires a clip_outliers stage",
                param="inject_outlier_ratio",
                location="cli.inject_outlier_ratio",
            )
        column_index = int(clip_stage.get("params", {}).get("column", 0))
        target = injection_dir / "sample.injected-outliers.json"
        write_json(target, amplify_outliers(dataset, column_index, args.inject_outlier_ratio))
        return target, pipeline_path, "outlier_amplification"
    return data_path, pipeline_path, None


def _apply_error_context(error: WorkbenchError, injection: str | None) -> WorkbenchError:
    if injection == "data_corruption" and error.code == E_SPEC:
        error.code = E_DATA
        error.stage_id = error.stage_id or "<dataset>"
        error.param = error.param or "nulls"
        error.details = {**error.details, "injection": injection}
    elif injection == "parameter_out_of_bounds" and error.code == E_SPEC:
        error.code = E_PARAM
        error.stage_id = error.stage_id or "<pipeline>"
        error.param = error.param or "params"
        error.details = {**error.details, "injection": injection}
    elif injection is not None:
        error.details = {**error.details, "injection": injection}
    return error


def _run_result_payload(result: Any) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status,
        "spec_version": result.spec_version,
        "fingerprint": result.fingerprint,
        "dataset_digest": result.dataset_digest,
        "final_data_digest": result.final_data_digest,
        "stage_statuses": result.stage_statuses,
        "shards": result.shards,
        "shard_count": len(result.shards),
        "evolution": result.evolution,
        "elapsed_ms": result.elapsed_ms,
    }


def cmd_init_demo(args: argparse.Namespace) -> int:
    paths = materialise_demo_files(args.out_dir, force=args.force)
    _print_json(
        {
            "status": "OK",
            "files": paths,
            "row_count": 137,
            "sources": ["events", "profiles"],
        }
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    prepared = load_workspace(args.data, args.pipeline)
    _print_json(
        {
            "status": "OK",
            "row_count": prepared.dataset.row_count,
            "column_count": len(prepared.dataset.columns),
            "dataset_digest": prepared.dataset.digest,
            "pipeline_digest": prepared.pipeline.digest,
            "fingerprint": prepared.fingerprint,
            "stages": [stage.as_dict() for stage in prepared.pipeline.stages],
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    pipeline_path = Path(args.pipeline)
    db_path = Path(args.db)
    log_path = Path(args.log)
    run_id = args.run_id or f"run-{uuid.uuid4().hex[:16]}"
    injection: str | None = None
    try:
        data_path, pipeline_path, injection = _apply_injections(
            args, data_path, pipeline_path
        )
        prepared = load_workspace(data_path, pipeline_path)
    except WorkbenchError as error:
        error = _apply_error_context(error, injection)
        _record_standalone_error(db_path, log_path, error, run_id)
        _print_json({"status": "ERROR", "run_id": run_id, **error.to_dict()})
        return _exit_for_code(error.code)
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        wrapped = WorkbenchError(E_STATE, f"unexpected load failure: {error}")
        _record_standalone_error(db_path, log_path, wrapped, run_id)
        _print_json({"status": "ERROR", "run_id": run_id, **wrapped.to_dict()})
        return _exit_for_code(wrapped.code)

    logger = JsonlLogger(log_path)
    try:
        with Ledger(db_path, logger) as ledger:
            result = Engine(ledger).run(
                prepared,
                run_id=run_id,
                batch_size=args.batch_size,
                normalize_scale=args.normalize_scale,
                crash_after_stage=args.crash_after_stage,
                command={
                    "command": "run",
                    "data": str(data_path),
                    "pipeline": str(pipeline_path),
                    "injection": injection,
                },
            )
            _print_json(_run_result_payload(result))
            return 0
    except WorkbenchError as error:
        error = _apply_error_context(error, injection)
        _print_json({"status": "ERROR", "run_id": run_id, **error.to_dict()})
        return _exit_for_code(error.code)


def cmd_inspect(args: argparse.Namespace) -> int:
    with Ledger(args.db, JsonlLogger(args.log)) as ledger:
        report = RecoveryManager(ledger).inspect(args.run_id)
    _print_json(report)
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    try:
        with Ledger(args.db, JsonlLogger(args.log)) as ledger:
            report = RecoveryManager(ledger).recover(args.run_id)
        _print_json(report)
        return 0
    except WorkbenchError as error:
        _print_json({"status": "ERROR", **error.to_dict()})
        return _exit_for_code(error.code)


def cmd_replay(args: argparse.Namespace) -> int:
    try:
        with Ledger(args.db, JsonlLogger(args.log)) as ledger:
            report = RecoveryManager(ledger).replay(args.run_id)
        _print_json(report)
        return 0
    except WorkbenchError as error:
        _print_json({"status": "ERROR", **error.to_dict()})
        return _exit_for_code(error.code)


def cmd_ledger(args: argparse.Namespace) -> int:
    with Ledger(args.db, JsonlLogger(args.log)) as ledger:
        runs = ledger.list_runs()
        errors = ledger.list_errors(run_id=args.run_id)
        payload = {
            "db": str(Path(args.db).resolve()),
            "counts": ledger.table_counts(),
            "runs": runs,
            "errors": errors,
        }
        if args.run_id:
            payload["stage_runs"] = ledger.latest_stage_runs(args.run_id)
            payload["reports"] = {
                "recovery_inspection": ledger.get_report(args.run_id, "recovery_inspection"),
                "recovery": ledger.get_report(args.run_id, "recovery"),
                "replay": ledger.get_report(args.run_id, "replay"),
            }
    _print_json(payload)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .ui import main as ui_main

        return ui_main(
            db_path=args.db,
            data_path=args.data,
            pipeline_path=args.pipeline,
            log_path=args.log,
        )
    except ModuleNotFoundError as error:
        failure = WorkbenchError(
            E_ENV,
            f"Tkinter is unavailable: {error}",
            location="environment.tkinter",
            details={"hint": "Install the standard CPython Windows distribution with Tcl/Tk."},
        )
        _print_json({"status": "ERROR", **failure.to_dict()})
        return _exit_for_code(failure.code)


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    summary = run_demo(Path(args.out_dir))
    print(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlpipe-workbench",
        description="Deterministic ML training-data pipeline diagnostics workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_demo = sub.add_parser("init-demo", help="materialise deterministic sample JSON")
    init_demo.add_argument("--out-dir", default="data")
    init_demo.add_argument("--force", action="store_true")
    init_demo.set_defaults(func=cmd_init_demo)

    validate = sub.add_parser("validate", help="strong-validate and fingerprint a workspace")
    validate.add_argument("--data", default="data/sample.json")
    validate.add_argument("--pipeline", default="data/pipeline.json")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="execute a deterministic pipeline into the ledger")
    run.add_argument("--data", default="data/sample.json")
    run.add_argument("--pipeline", default="data/pipeline.json")
    run.add_argument("--db", default="workbench.db")
    run.add_argument("--log", default="logs/workbench.jsonl")
    run.add_argument("--run-id")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--normalize-scale", type=int)
    run.add_argument("--crash-after-stage", help="hard-exit after committing this stage")
    run.add_argument("--inject-data-corrupt", action="store_true")
    run.add_argument("--inject-param-bounds", action="store_true")
    run.add_argument("--inject-outlier-ratio", type=int, metavar="PERMILLE")
    run.add_argument("--injection-dir", default="artifacts/injections")
    run.set_defaults(func=cmd_run)

    inspect = sub.add_parser("inspect", help="inspect committed recovery state for a run")
    inspect.add_argument("--db", default="workbench.db")
    inspect.add_argument("--log", default="logs/workbench.jsonl")
    inspect.add_argument("--run-id", required=True)
    inspect.set_defaults(func=cmd_inspect)

    recover = sub.add_parser("recover", help="resume an interrupted run and replay it")
    recover.add_argument("--db", default="workbench.db")
    recover.add_argument("--log", default="logs/workbench.jsonl")
    recover.add_argument("--run-id", required=True)
    recover.set_defaults(func=cmd_recover)

    replay = sub.add_parser("replay", help="recompute and compare every ledger item")
    replay.add_argument("--db", default="workbench.db")
    replay.add_argument("--log", default="logs/workbench.jsonl")
    replay.add_argument("--run-id", required=True)
    replay.set_defaults(func=cmd_replay)

    ledger = sub.add_parser("ledger", help="show ledger counts, runs, and errors")
    ledger.add_argument("--db", default="workbench.db")
    ledger.add_argument("--log", default="logs/workbench.jsonl")
    ledger.add_argument("--run-id")
    ledger.set_defaults(func=cmd_ledger)

    ui = sub.add_parser("ui", help="open the Tkinter diagnostic workbench")
    ui.add_argument("--db", default="workbench.db")
    ui.add_argument("--log", default="logs/workbench.jsonl")
    ui.add_argument("--data", default="data/sample.json")
    ui.add_argument("--pipeline", default="data/pipeline.json")
    ui.set_defaults(func=cmd_ui)

    demo = sub.add_parser("demo", help="run all three repeatable demonstration scenarios")
    demo.add_argument("--out-dir", default="demo_output")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as error:
        _print_json({"status": "ERROR", **error.to_dict()})
        return _exit_for_code(error.code)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
