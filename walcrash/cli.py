"""Command-line entry point for loading, replay, crash drills, and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .engine import ReplayEngine
from .errors import BusinessError, append_diagnostic, utc_now_iso
from .snapshots import create_snapshot, restore_snapshot
from .storage import Storage
from .validation import load_sample
from .verification import verify_replay


ACTION_CHOICES = (
    "load",
    "run",
    "resume",
    "checkpoint",
    "verify",
    "snapshot",
    "restore",
    "status",
    "export-ledger",
    "reset",
)
INJECTION_CHOICES = ("page_size", "order", "snapshot_digest")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walcrash",
        description=(
            "Deterministic WAL crash-recovery diagnostics. Paths passed to "
            "--snapshot-file and --output are resolved against --workspace."
        ),
    )
    parser.add_argument("--version", action="version", version=f"walcrash {__version__}")
    parser.add_argument("--workspace", default="demo_workspace", help="workspace directory (default: demo_workspace)")
    parser.add_argument("--action", choices=ACTION_CHOICES, help="operation to execute")
    parser.add_argument("--input", default="samples/sample.json", help="sample JSON path")
    parser.add_argument(
        "--snapshot-file",
        default="snapshots/session.json",
        help="snapshot path, relative to workspace",
    )
    parser.add_argument(
        "--output",
        default="ledger_export.json",
        help="export path for --action export-ledger, relative to workspace",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="deterministic worker delay used to make the scheduler timeline visible",
    )
    parser.add_argument(
        "--crash-after-batches",
        type=int,
        default=None,
        metavar="N",
        help="deliberately os._exit after N committed batches (crash drill)",
    )
    parser.add_argument("--gui", action="store_true", help="launch the Tkinter diagnostics UI")
    parser.add_argument(
        "--inject",
        choices=INJECTION_CHOICES,
        default=None,
        help="run one explicit business-error injection and exit",
    )
    return parser


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _output_path(storage: Storage, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else storage.workspace / candidate


def _export_ledger(storage: Storage, output: str) -> dict[str, Any]:
    ledger_check = storage.verify_ledger()
    payload = {
        "format_version": 1,
        "exported_at": utc_now_iso(),
        "workspace": str(storage.workspace),
        "status": storage.status(),
        "batches": storage.batches(),
        "tasks": storage.tasks(),
        "checkpoints": storage.checkpoint_rows(),
        "page_versions": storage.page_versions(),
        "ledger_events": storage.ledger_rows(),
        "ledger_verification": ledger_check,
    }
    target = _output_path(storage, output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        display_path = target.relative_to(storage.workspace).as_posix()
    except ValueError:
        display_path = str(target)
    return {"ok": True, "path": display_path, "ledger_verification": ledger_check}


def _run_injection(storage: Storage, mode: str, sample_path: str) -> dict[str, Any]:
    def fixture(name: str) -> str:
        local = Path("samples") / name
        bundled = PROJECT_ROOT / "samples" / name
        return str(bundled if not local.exists() and bundled.exists() else local)

    if mode == "page_size":
        load_sample(fixture("invalid_page_size.json"))
        raise AssertionError("unreachable")
    if mode == "order":
        load_sample(fixture("invalid_order.json"))
        raise AssertionError("unreachable")
    if mode == "snapshot_digest":
        candidate = Path(sample_path)
        if not candidate.exists():
            bundled = PROJECT_ROOT / candidate
            if bundled.exists():
                sample_path = str(bundled)
        if not storage.is_initialized():
            sample = load_sample(sample_path)
            storage.load_fresh(sample)
        if storage.ledger_cursor() == 0:
            ReplayEngine(storage).replay()
        created = create_snapshot(storage, "snapshots/injection_source.json")
        source_path = _output_path(storage, created["path"])
        document = json.loads(source_path.read_text(encoding="utf-8"))
        document["content_digest"] = "0" * 64
        tampered_path = storage.snapshots_dir / "injection_tampered.json"
        tampered_path.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        restore_snapshot(storage, str(tampered_path))
        raise AssertionError("unreachable")
    raise BusinessError("E_STATE", "unknown injection mode", {"mode": mode})


def _dispatch(args: argparse.Namespace, storage: Storage) -> dict[str, Any]:
    if args.action == "load":
        sample = load_sample(args.input)
        storage.load_fresh(sample)
        return {
            "ok": True,
            "action": "load",
            "workspace": str(storage.workspace),
            "source_sample": args.input,
            "pages": len(sample.pages),
            "wal_entries": len(sample.entries),
            "crash_point_lsn": sample.crash_point_lsn,
            "wal_digest": sample.wal_digest,
            "pages_digest": sample.pages_digest,
            "budget_digest": sample.budget_digest,
        }
    if args.action in ("run", "resume"):
        result = ReplayEngine(storage).replay(
            delay_ms=args.delay_ms,
            crash_after_batches=args.crash_after_batches,
        )
        return {"action": args.action, **result}
    if args.action == "checkpoint":
        return {"action": "checkpoint", **storage.save_checkpoint("cli-manual")}
    if args.action == "verify":
        return {"action": "verify", **verify_replay(storage)}
    if args.action == "snapshot":
        return {"action": "snapshot", **create_snapshot(storage, args.snapshot_file)}
    if args.action == "restore":
        return {"action": "restore", **restore_snapshot(storage, args.snapshot_file)}
    if args.action == "status":
        return {"ok": True, "action": "status", **storage.status()}
    if args.action == "export-ledger":
        return {"action": "export-ledger", **_export_ledger(storage, args.output)}
    if args.action == "reset":
        storage.remove_state(remove_snapshots=True)
        return {
            "ok": True,
            "action": "reset",
            "workspace": str(storage.workspace),
            "note": "workspace state and snapshots were removed",
        }
    raise BusinessError("E_STATE", "an action is required", {"hint": "use --action or --gui"})


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    storage = Storage(args.workspace)
    if args.inject:
        try:
            _run_injection(storage, args.inject, args.input)
        except BusinessError as error:
            storage.record_error(error, injection=args.inject)
            try:
                append_diagnostic(
                    storage.logs_dir / "injections.ndjson",
                    {
                        "level": "ERROR",
                        "injection": args.inject,
                        **error.as_dict(),
                    },
                )
            except OSError:
                pass
            _json_print(dict(error.as_dict(), injection=args.inject))
            return error.exit_code
        except Exception as error:  # injector failures must still be structured
            wrapped = BusinessError(
                "E_STATE",
                "injection setup failed before the expected business error",
                {"injection": args.inject, "error": str(error)},
            )
            storage.record_error(wrapped, injection=args.inject)
            _json_print(dict(wrapped.as_dict(), injection=args.inject))
            return wrapped.exit_code

    if args.gui:
        try:
            from .gui import launch_gui

            launch_gui(storage, sample_path=args.input)
            return 0
        except BusinessError as error:
            storage.record_error(error)
            _json_print(error.as_dict())
            return error.exit_code

    if not args.action:
        parser.print_help()
        return 0
    try:
        result = _dispatch(args, storage)
        _json_print(result)
        return 0
    except BusinessError as error:
        storage.record_error(error, action=args.action)
        _json_print(error.as_dict())
        return error.exit_code
    except Exception as exc:
        wrapped = BusinessError(
            "E_STATE",
            "unexpected application failure",
            {"action": args.action, "error": f"{type(exc).__name__}: {exc}"},
        )
        storage.record_error(wrapped, action=args.action)
        _json_print(wrapped.as_dict())
        return wrapped.exit_code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
