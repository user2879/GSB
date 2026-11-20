from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import to_plain
from .errors import (
    E_ENV,
    E_SPEC,
    EXIT_CODES,
    WorkbenchError,
    append_structured_log,
    log_error,
)
from .service import WorkbenchService
from .spec import write_injected_spec


DEFAULT_SPEC = "data/sample_spec.json"
DEFAULT_STATE_ROOT = ".workbench"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deploy_workbench",
        description="Local multi-environment deployment consistency workbench",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="gui",
        choices=("gui", "run", "replay", "recover", "versions", "show"),
        help="operation to perform (default: gui)",
    )
    parser.add_argument("--spec", default=DEFAULT_SPEC, help="environment declaration JSON")
    parser.add_argument("--state-root", default=DEFAULT_STATE_ROOT, help="ledger, snapshots, logs, injected inputs")
    parser.add_argument("--scale-power", default=None, help="session fixed-point scale power (integer)")
    parser.add_argument("--snapshot", default=None, help="snapshot path for replay")
    parser.add_argument("--run-id", type=int, default=None, help="historical run id for show")
    parser.add_argument("--inject", choices=("env", "bind", "grid"), default=None, help="inject one required failure class")
    parser.add_argument("--inject-env", dest="inject", action="store_const", const="env", help="alias for --inject env")
    parser.add_argument("--inject-bind", dest="inject", action="store_const", const="bind", help="alias for --inject bind")
    parser.add_argument("--inject-grid", dest="inject", action="store_const", const="grid", help="alias for --inject grid")
    parser.add_argument("--max-json", action="store_true", help="emit compact JSON")
    return parser


def _print_json(value: Any, *, compact: bool = False) -> None:
    plain = to_plain(value)
    if compact:
        print(json.dumps(plain, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(plain, ensure_ascii=False, sort_keys=True, indent=2))


def _prepare_inputs(args: argparse.Namespace, state_root: Path) -> tuple[Path, str | None]:
    spec_path = Path(args.spec)
    scale_override: int | str | None = args.scale_power
    if scale_override is not None:
        try:
            scale_override = int(scale_override)
        except (TypeError, ValueError):
            pass
    if args.inject is None:
        return spec_path, scale_override
    injected_dir = state_root / "injected"
    injected_dir.mkdir(parents=True, exist_ok=True)
    if args.inject == "grid":
        destination = injected_dir / "invalid_grid_spec.json"
        write_injected_spec(spec_path, destination, "grid")
        return destination, 99
    destination = injected_dir / f"invalid_{args.inject}_spec.json"
    write_injected_spec(spec_path, destination, args.inject)
    return destination, scale_override


def _open_service(args: argparse.Namespace) -> tuple[WorkbenchService, dict[str, Any]]:
    state_root = Path(args.state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    spec_path, scale_override = _prepare_inputs(args, state_root)
    service = WorkbenchService(spec_path, state_root, scale_power_override=scale_override)
    return service, {
        "input_spec": str(spec_path),
        "injection": args.inject,
        "scale_power_override": scale_override,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    state_root = Path(args.state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    log_path = state_root / "logs" / "workbench.jsonl"
    try:
        if args.command == "gui":
            from .gui import launch_gui

            spec_path, scale_override = _prepare_inputs(args, state_root)
            return launch_gui(
                spec_path=spec_path,
                state_root=state_root,
                scale_power_override=scale_override,
                log_path=log_path,
            )

        service, context = _open_service(args)
        try:
            if args.command == "run":
                result = service.run_all()
            elif args.command == "replay":
                result = service.replay_snapshot(args.snapshot)
            elif args.command == "recover":
                report = service.recover()
                result = {
                    "recovery": report,
                    "current_view": service.get_current_view(),
                }
            elif args.command == "versions":
                result = {
                    "current_versions": service.versions.payload(),
                    "runs": service.list_versions(),
                }
            elif args.command == "show":
                if args.run_id is not None:
                    result = service.get_historical_view(args.run_id)
                else:
                    result = service.get_current_view()
            else:
                raise WorkbenchError(E_SPEC, f"unknown command {args.command!r}", path="$")
        finally:
            service.close()

        result = {"ok": True, "command": args.command, "context": context, **result}
        append_structured_log(
            log_path,
            {
                "level": "INFO",
                "event": "command_completed",
                "command": args.command,
                "injection": args.inject,
                "status": result.get("status", "ok"),
            },
        )
        _print_json(result, compact=args.max_json)
        return 0
    except WorkbenchError as error:
        log_error(log_path, error, command=args.command)
        _print_json(error.to_report(), compact=args.max_json)
        return EXIT_CODES.get(error.code, 1)
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        wrapped = WorkbenchError(E_SPEC, f"unexpected application failure: {type(error).__name__}: {error}", path="$")
        log_error(log_path, wrapped, command=args.command)
        _print_json(wrapped.to_report(), compact=args.max_json)
        return EXIT_CODES[E_SPEC]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
