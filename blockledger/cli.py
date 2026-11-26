from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .errors import ReportableError, SpecError, VerifyError
from .pipeline import ProjectPaths, execute_run, recover_and_rerun_unknown, write_json_report
from .samplegen import write_samples
from .spec import load_json_path, volume_summary


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False))


def _parse_block(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("block must be VOLUME_ID:BLOCK_INDEX")
    volume_id, raw_index = value.split(":", 1)
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block index must be an integer") from exc
    if not volume_id or index < 0:
        raise argparse.ArgumentTypeError("invalid volume:block pair")
    return volume_id, index


def _load_change(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SpecError("change document must be an object", entry_id=str(path), field="root")
    return raw


def cmd_generate_samples(args: argparse.Namespace) -> int:
    paths = write_samples(args.dir)
    _print_json({"ok": True, "written": [str(path) for path in paths]})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_json_path(args.spec)
    summary = volume_summary(spec)
    payload = {
        "ok": True,
        "code": "OK",
        "format_version": spec["format_version"],
        "volume_count": len(spec["volumes"]),
        "snapshot_count": len(spec["snapshots"]),
        "event_count": len(spec["events"]),
        "budget": spec["budget"],
        "volumes": summary,
    }
    if args.out:
        write_json_report(payload, args.out)
    _print_json(payload)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .engine import replay_spec

    spec = load_json_path(args.spec)
    replay = replay_spec(spec, verify_declared_digests=not args.no_verify_declared)
    payload = {
        "ok": True,
        "fingerprint": replay.fingerprint(),
        "effects": replay.effects,
        "rollback_plans": replay.rollback_plans,
    }
    if args.out:
        write_json_report(payload, args.out)
    _print_json({"ok": True, "fingerprint": payload["fingerprint"], "effects": len(replay.effects), "out": args.out})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_json_path(args.spec)
    paths = ProjectPaths(Path(args.root))
    kill_blocks = set(args.kill_block or [])
    if args.kill_first_selected:
        kill_blocks = {"__FIRST_SELECTED__"}  # type: ignore[arg-type]
    report = execute_run(
        spec,
        ledger_path=args.ledger or paths.ledger_db,
        session_path=args.session or paths.session,
        content_root=args.content_root or paths.content_store,
        live_root=paths.live_blocks,
        work_root=paths.work_root,
        attempt=args.attempt,
        only_rollback=args.only_rollback,
        fault_kill_blocks=kill_blocks,
        corrupt_blocks=set(args.corrupt_block or []),
        crash_after_lease_batch=args.crash_after_lease_batch,
        apply_deletions=not args.skip_deletion_propagation,
        python_executable=sys.executable,
    )
    output = args.report or str(paths.artifacts / "run-report.json")
    write_json_report(report, output)
    _print_json(
        {
            "status": report["status"],
            "report": output,
            "task_count": report["task_count"],
            "counts": report["counts"],
            "corrupt_findings": report["corrupt_findings"][:1],
            "ledger_content_digest": report["ledger_content_digest"],
        }
    )
    return 3 if report["corrupt_findings"] else 0


def cmd_recover(args: argparse.Namespace) -> int:
    spec = load_json_path(args.spec)
    paths = ProjectPaths(Path(args.root))
    report = recover_and_rerun_unknown(
        spec,
        ledger_path=args.ledger or paths.ledger_db,
        session_path=args.session or paths.session,
        content_root=paths.content_store,
        work_root=paths.work_root,
        rerun=args.rerun,
        python_executable=sys.executable,
    )
    output = args.report or str(paths.artifacts / "recovery-report.json")
    write_json_report(report, output)
    _print_json(
        {
            "status": report["status"],
            "report": output,
            "recovered_interrupted": len(report["recovered_interrupted"]),
            "unknown_before_rerun": report["unknown_before_rerun"],
            "rerun_batch_count": report["rerun_batch_count"],
            "counts": report["counts"],
        }
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .verify import verify_ledger

    spec = load_json_path(args.spec)
    paths = ProjectPaths(Path(args.root))
    try:
        result = verify_ledger(spec, args.ledger or paths.ledger_db)
    except VerifyError as exc:
        payload = {"ok": False, **exc.to_dict()}
        if args.out:
            write_json_report(payload, args.out)
        _print_json(payload)
        return 4
    if args.out:
        write_json_report(result, args.out)
    _print_json(result)
    return 0


def cmd_invalidate(args: argparse.Namespace) -> int:
    from .invalidation import analyze_change
    from .ledger import open_readonly_ledger

    spec = load_json_path(args.spec)
    change = _load_change(args.change)
    report = analyze_change(spec, change)
    if args.ledger and Path(args.ledger).exists():
        conn = open_readonly_ledger(args.ledger)
        try:
            affected_rbs = set(report["affected_rollback_ids"])
            running = []
            for row in conn.execute("SELECT batch_id,rollback_id,volume_id,block_set_json,status FROM batches WHERE status='running'"):
                if row["rollback_id"] in affected_rbs:
                    item = dict(row)
                    item["block_set"] = json.loads(item.pop("block_set_json"))
                    running.append(item)
            unknown = [
                dict(row)
                for row in conn.execute(
                    "SELECT rollback_id,volume_id,block_index,batch_id,verification_status,reason "
                    "FROM block_state_latest WHERE verification_status='unknown' ORDER BY volume_id,block_index"
                )
                if row["rollback_id"] in affected_rbs
            ]
            report["ledger_unfinished"] = {
                "running_batches": running,
                "unknown_blocks": unknown,
                "quota_rule": "running leases are terminal-interrupted on restart and their slots are released before rerun",
            }
        finally:
            conn.close()
    if args.out:
        write_json_report(report, args.out)
    _print_json(
        {
            "ok": True,
            "change_id": report["change_id"],
            "affected_snapshots": report["affected_snapshot_ids"],
            "affected_rollbacks": report["affected_rollback_ids"],
            "affected_set_size": len(report["affected_set"]),
            "full_rebuild_match": report["full_rebuild_consistency"]["match"],
            "out": args.out,
        }
    )
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    info: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except Exception as exc:  # noqa: BLE001
        info["numpy_error"] = repr(exc)
    try:
        import tkinter

        display = __import__("os").environ.get("DISPLAY")
        info["tkinter"] = {"version": tkinter.TkVersion, "display": display, "gui_usable": bool(display)}
    except Exception as exc:  # noqa: BLE001
        info["tkinter"] = {"available": False, "error": type(exc).__name__ + ": " + str(exc)}
    _print_json(info)
    return 0


def cmd_gui_check(_args: argparse.Namespace) -> int:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        usable = True
        version = tk.TkVersion
        root.destroy()
        payload = {"gui_check": "ok", "tk_version": version}
    except Exception as exc:  # noqa: BLE001
        payload = {
            "gui_check": "skipped",
            "reason": type(exc).__name__ + ": " + str(exc),
            "note": "core package, ledger and verify commands remain runnable without a display",
        }
    _print_json(payload)
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch

    return launch(args.spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blockledger", description="Deterministic block ledger toolkit")
    parser.add_argument("--root", default=".", help="project root containing ledger/, snapshots/, artifacts/")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-samples", help="generate deterministic JSON samples")
    generate.add_argument("--dir", default="samples")
    generate.set_defaults(func=cmd_generate_samples)

    validate = sub.add_parser("validate", help="load and strongly validate JSON specs")
    validate.add_argument("--spec", default="samples/spec.json")
    validate.add_argument("--out")
    validate.set_defaults(func=cmd_validate)

    replay = sub.add_parser("replay", help="deterministically replay event stream")
    replay.add_argument("--spec", default="samples/spec.json")
    replay.add_argument("--out")
    replay.add_argument("--no-verify-declared", action="store_true")
    replay.set_defaults(func=cmd_replay)

    run = sub.add_parser("run", help="run quota scheduler, subprocess checks and ledger commit")
    run.add_argument("--spec", default="samples/spec.json")
    run.add_argument("--ledger")
    run.add_argument("--session")
    run.add_argument("--content-root")
    run.add_argument("--report")
    run.add_argument("--attempt", type=int, default=1)
    run.add_argument("--only-rollback")
    run.add_argument("--kill-block", type=_parse_block, action="append")
    run.add_argument("--kill-first-selected", action="store_true")
    run.add_argument("--corrupt-block", type=_parse_block, action="append")
    run.add_argument("--crash-after-lease-batch")
    run.add_argument("--skip-deletion-propagation", action="store_true")
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover", help="recover interrupted leases and optionally rerun unknown blocks")
    recover.add_argument("--spec", default="samples/spec.json")
    recover.add_argument("--ledger")
    recover.add_argument("--session")
    recover.add_argument("--report")
    recover.add_argument("--rerun", action="store_true")
    recover.set_defaults(func=cmd_recover)

    verify = sub.add_parser("verify", help="independently replay spec and compare read-only ledger")
    verify.add_argument("--spec", default="samples/spec.json")
    verify.add_argument("--ledger")
    verify.add_argument("--out")
    verify.set_defaults(func=cmd_verify)

    invalidate = sub.add_parser("invalidate", help="analyze deletion/write invalidation subset rebuild")
    invalidate.add_argument("--spec", default="samples/spec.json")
    invalidate.add_argument("--change", required=True)
    invalidate.add_argument("--ledger")
    invalidate.add_argument("--out")
    invalidate.set_defaults(func=cmd_invalidate)

    doctor = sub.add_parser("doctor", help="print runtime dependency status")
    doctor.set_defaults(func=cmd_doctor)

    gui_check = sub.add_parser("gui-check", help="check optional Tkinter GUI")
    gui_check.set_defaults(func=cmd_gui_check)

    gui = sub.add_parser("gui", help="launch minimal Tkinter console")
    gui.add_argument("--spec", default="samples/spec.json")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except VerifyError as exc:
        _print_json({"ok": False, **exc.to_dict()})
        return 4
    except SpecError as exc:
        _print_json({"ok": False, **exc.to_dict()})
        return 2
    except ReportableError as exc:
        _print_json({"ok": False, **exc.to_dict()})
        return 5
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _print_json({"ok": False, "code": "E_RUNTIME", "message": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
