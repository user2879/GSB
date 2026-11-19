from __future__ import annotations

import argparse
import json
import signal
from typing import Any, Sequence

from .engine import CancellationToken, RunEngine, RunResult
from .errors import BusinessError, ErrorCode
from .persistence import StateStore
from .sample import write_sample
from .schema import load_mesh_file, load_repair_script


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _business_exit_code(error: BusinessError) -> int:
    if error.code in {ErrorCode.E_SCHEMA, ErrorCode.E_INDEX, ErrorCode.E_IO, ErrorCode.E_BUDGET}:
        return 2
    if error.code in {ErrorCode.E_STATE, ErrorCode.E_SNAPSHOT_INVALID}:
        return 3
    return 4


def _result_summary(result: RunResult) -> dict[str, Any]:
    report = result.report
    summary = report.get("summary", {}) if report else {}
    return {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "version_id": result.version_id,
        "report_path": result.report_path,
        "completed_steps": result.completed_steps,
        "total_operators": result.total_operators,
        "resumed_from_step": result.resumed_from_step,
        "verdict": summary.get("fixed_point_verdict", {}).get("status"),
        "first_tolerance_exceedance": summary.get("first_tolerance_exceedance"),
        "business_codes": report.get("business_codes", []),
        "invalid_snapshot_count": result.state_inspection.invalid_snapshot_count,
    }


def _install_signals(token: CancellationToken) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        token.request(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signals(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass


def command_generate(args: argparse.Namespace) -> int:
    mesh_path, script_path = write_sample(args.directory)
    _print_json(
        {
            "ok": True,
            "mesh": str(mesh_path),
            "script": str(script_path),
        }
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    mesh = load_mesh_file(args.mesh)
    script = load_repair_script(args.script)
    engine = RunEngine(mesh, script, args.state, args.output)
    token = CancellationToken()
    previous = _install_signals(token)
    try:
        result = engine.run(token, restart=args.restart)
    finally:
        _restore_signals(previous)
    _print_json(_result_summary(result))
    return result.exit_code


def command_inspect(args: argparse.Namespace) -> int:
    inspection = StateStore(args.state).inspect()
    _print_json({"ok": inspection.invalid_snapshot_count == 0, **inspection.to_dict()})
    return 0 if inspection.invalid_snapshot_count == 0 else 3


def command_gui(args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    gui_main(
        mesh_path=args.mesh,
        script_path=args.script,
        state_dir=args.state,
        output_dir=args.output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh-audit",
        description="Deterministic triangle-mesh repair and fixed-point error audit workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write the deterministic local sample")
    generate.add_argument("--directory", default="samples")
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run", help="run or resume a repair audit")
    run.add_argument("--mesh", required=True)
    run.add_argument("--script", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--output", required=True)
    run.add_argument(
        "--restart",
        action="store_true",
        help="archive the current state run before starting from the input again",
    )
    run.set_defaults(func=command_run)

    inspect = subparsers.add_parser("inspect", help="verify and list persisted snapshots")
    inspect.add_argument("--state", required=True)
    inspect.set_defaults(func=command_inspect)

    gui = subparsers.add_parser("gui", help="open the desktop diagnostic workbench")
    gui.add_argument("--mesh", default="samples/sample_defective.mesh.json")
    gui.add_argument("--script", default="samples/sample_repair_script.json")
    gui.add_argument("--state", default="state/sample")
    gui.add_argument("--output", default="output/sample")
    gui.set_defaults(func=command_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as error:
        _print_json(error.to_dict())
        return _business_exit_code(error)
    except KeyboardInterrupt:
        _print_json(
            {
                "ok": False,
                "exit_code": 130,
                "error": {"code": "E_INTERRUPTED", "message": "interrupted before operator boundary"},
            }
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
