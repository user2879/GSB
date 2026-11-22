"""Command-line entry point for snapdiag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .demo import run_demo
from .doctor import environment_report
from .errors import (
    DiagnosticError,
    E_STATE,
    EXIT_CODES,
)
from .injection import INJECTION_KINDS, InjectionManager
from .manifest import FORMAT_VERSIONS, encode_manifest, load_manifest_file
from .store import JsonEventLogger, Session, SimulatedInterruption
from .util import atomic_write_json
from .view import build_diagnostic_view, render_diagnostic_text, run_gui
from .workflow import initialize_from_spec, prepare_samples, run_declared_session


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _session_from_args(args: argparse.Namespace) -> Session:
    return Session.open(args.session)


def _init(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec:
        session, spec = initialize_from_spec(args.spec, session_root=args.session, reset=args.reset)
        executed: list[dict[str, Any]] = []
        if args.run_steps:
            executed = run_declared_session(session)
        return {
            "ok": True,
            "operation": "init",
            "session": str(session.root),
            "spec": str(Path(args.spec).resolve()),
            "block_size": session.block_size,
            "block_count": session.block_count,
            "snapshot_count": len(session.snapshot_rows()),
            "steps_executed": executed,
        }
    session = Session.initialize(
        args.session,
        block_size=args.block_size,
        block_count=args.block_count,
        image_name=args.image_name,
        seed=args.seed,
        manifest_format=args.manifest_format,
        threshold=args.threshold,
        reset=args.reset,
    )
    return {
        "ok": True,
        "operation": "init",
        "session": str(session.root),
        "block_size": session.block_size,
        "block_count": session.block_count,
        "manifest_format": args.manifest_format,
        "threshold": str(session.threshold),
    }


def _create_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    row = session.create_snapshot(args.snapshot_id, parent_snapshot_id=args.parent)
    return {
        "ok": True,
        "operation": "create-snapshot",
        "snapshot_id": row["snapshot_id"],
        "parent_snapshot_id": row["parent_snapshot_id"],
        "snapshot_version": row["snapshot_version"],
        "checkpoint_sha256": f"sha256:{row['checkpoint_digest']}",
        "state_version": session.state_version,
    }


def _write_block(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    result = session.write_block_hex(args.block, args.hex)
    return {"ok": True, "operation": "write-block", **result}


def _rollback(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    try:
        result = session.rollback(
            args.target,
            threshold=args.threshold,
            path=args.path,
            retry=args.retry,
            failpoint=args.failpoint,
        )
    except SimulatedInterruption as exc:
        return {
            "ok": False,
            "code": E_STATE,
            "message": str(exc),
            "field": "failpoint",
            "requires_rerun": True,
            "recovery": session.recover_session(),
        }
    return {"ok": True, "operation": "rollback", **result.as_dict()}


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    conclusion = session.audit(strict_manifest=args.strict)
    return {"ok": conclusion["ok"], "operation": "audit", **conclusion}


def _recover(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    recovered = session.recover_session()
    return {
        "ok": all(not item["requires_rerun"] for item in recovered),
        "operation": "recover",
        "recovered": recovered,
    }


def _inject(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    manager = InjectionManager(session)
    injection = manager.inject(args.kind, target=args.target, block_index=args.block)
    diagnosis = None if args.no_diagnose else manager.diagnose(args.kind, target=args.target)
    expected = injection.get("expected_business_code")
    observed = diagnosis.get("observed_code") if diagnosis else None
    return {
        "ok": diagnosis is None or observed == expected,
        "operation": "inject",
        "injection": injection,
        "diagnosis": diagnosis,
        "expected_code": expected,
        "observed_code": observed,
    }


def _repair(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    return InjectionManager(session).repair_and_verify(target=args.target)


def _show(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    injection = None
    diagnosis = None
    if args.inject:
        manager = InjectionManager(session)
        injection = manager.inject(args.inject, target=args.target)
        diagnosis = manager.diagnose(args.inject, target=args.target)
    view = build_diagnostic_view(session, target=args.target, threshold=args.threshold)
    text = render_diagnostic_text(view)
    return {
        "ok": True,
        "operation": "show",
        "injection": injection,
        "injection_diagnosis": diagnosis,
        "view": view,
        "text": text,
    }


def _gui(args: argparse.Namespace) -> dict[str, Any]:
    session = Session.open(args.session)
    injection = None
    diagnosis = None
    if args.inject:
        manager = InjectionManager(session)
        injection = manager.inject(args.inject, target=args.target)
        diagnosis = manager.diagnose(args.inject, target=args.target)
    run_gui(session.root, target=args.target, threshold=args.threshold)
    return {
        "ok": True,
        "operation": "gui",
        "session": str(session.root),
        "injection": injection,
        "injection_diagnosis": diagnosis,
    }


def _normalize(args: argparse.Namespace) -> dict[str, Any]:
    resolver = Session.open(args.session) if args.session else None
    model = load_manifest_file(args.file, resolver=resolver)
    document = encode_manifest(model, args.format)
    if args.output:
        atomic_write_json(args.output, document)
    return {
        "ok": True,
        "operation": "normalize-manifest",
        "source_format": model.source_format,
        "filled_fields": list(model.filled_fields),
        "fill_basis": dict(sorted(model.fill_basis.items())),
        "normalized": model.as_dict(),
        "encoded_format": args.format,
        "encoded": document,
        "output": str(args.output) if args.output else None,
    }


def _import_manifest(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    report = session.import_manifest(args.file, reason=args.reason)
    return report


def _reconcile_manifest(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    return session.reconcile_manifest(reason=args.reason)


def _manifest_backups(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    return {
        "ok": True,
        "operation": "manifest-backups",
        "backups": session.list_manifest_backups(),
    }


def _prepare_samples(args: argparse.Namespace) -> dict[str, Any]:
    samples = prepare_samples(args.workspace, reset=not args.keep_existing)
    return {"ok": True, "operation": "prepare-samples", "samples": samples}


def _demo(args: argparse.Namespace) -> dict[str, Any]:
    report = run_demo(args.workspace)
    return report


def _doctor(args: argparse.Namespace) -> dict[str, Any]:
    return environment_report(args.workspace, min_free_mb=args.min_free_mb)


def _diagnose(args: argparse.Namespace) -> dict[str, Any]:
    session = _session_from_args(args)
    manager = InjectionManager(session)
    injection = manager.inject(args.inject, target=args.target)
    diagnosis = manager.diagnose(args.inject, target=args.target)
    return {
        "ok": diagnosis.get("observed_code") == injection.get("expected_business_code"),
        "operation": "diagnose",
        "injection": injection,
        "diagnosis": diagnosis,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snapdiag",
        description="Windows-local block snapshot and rollback diagnostic component",
    )
    parser.add_argument("--version", action="version", version="snapdiag 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize a local session")
    init.add_argument("--session", required=True)
    init.add_argument("--spec")
    init.add_argument("--image-name", default="volume.bin")
    init.add_argument("--block-size", type=int, default=4096)
    init.add_argument("--block-count", type=int, default=32)
    init.add_argument("--seed", type=int, default=20261123)
    init.add_argument("--manifest-format", choices=FORMAT_VERSIONS, default="m3")
    init.add_argument("--threshold", default="0.25")
    init.add_argument("--run-steps", action="store_true")
    init.add_argument("--reset", action="store_true")
    init.set_defaults(handler=_init)

    create = subparsers.add_parser("create-snapshot", help="create a named COW checkpoint")
    create.add_argument("--session", required=True)
    create.add_argument("--snapshot-id", required=True)
    create.add_argument("--parent")
    create.set_defaults(handler=_create_snapshot)

    write = subparsers.add_parser("write-block", help="write one exact hex block")
    write.add_argument("--session", required=True)
    write.add_argument("--block", type=int, required=True)
    write.add_argument("--hex", required=True)
    write.set_defaults(handler=_write_block)

    rollback = subparsers.add_parser("rollback", help="atomically roll back to a snapshot")
    rollback.add_argument("--session", required=True)
    rollback.add_argument("target")
    rollback.add_argument("--threshold")
    rollback.add_argument("--path", choices=("auto", "bitmap", "full"), default="auto")
    rollback.add_argument("--retry", action="store_true")
    rollback.add_argument("--failpoint", choices=("after_prepare", "after_replace"))
    rollback.set_defaults(handler=_rollback)

    audit = subparsers.add_parser("audit", help="run and persist a versioned consistency audit")
    audit.add_argument("--session", required=True)
    audit.add_argument("--strict", action="store_true")
    audit.set_defaults(handler=_audit)

    recover = subparsers.add_parser("recover", help="recover interrupted operations")
    recover.add_argument("--session", required=True)
    recover.set_defaults(handler=_recover)

    inject = subparsers.add_parser("inject", help="inject one supported anomaly and diagnose it")
    inject.add_argument("--session", required=True)
    inject.add_argument("--kind", choices=INJECTION_KINDS, required=True)
    inject.add_argument("--target")
    inject.add_argument("--block", type=int)
    inject.add_argument("--no-diagnose", action="store_true")
    inject.set_defaults(handler=_inject)

    diagnose = subparsers.add_parser("diagnose-injection", help="inject and return the observed business code")
    diagnose.add_argument("--session", required=True)
    diagnose.add_argument("--inject", choices=INJECTION_KINDS, required=True)
    diagnose.add_argument("--target")
    diagnose.set_defaults(handler=_diagnose)

    repair = subparsers.add_parser("repair", help="repair objects/manifest and verify every block")
    repair.add_argument("--session", required=True)
    repair.add_argument("--target")
    repair.set_defaults(handler=_repair)

    show = subparsers.add_parser("show", help="render deterministic diagnostic view")
    show.add_argument("--session", required=True)
    show.add_argument("--target")
    show.add_argument("--threshold")
    show.add_argument("--inject", choices=INJECTION_KINDS)
    show.set_defaults(handler=_show)

    gui = subparsers.add_parser("gui", help="open the Tkinter diagnostic window")
    gui.add_argument("--session", required=True)
    gui.add_argument("--target")
    gui.add_argument("--threshold")
    gui.add_argument("--inject", choices=INJECTION_KINDS)
    gui.set_defaults(handler=_gui)

    normalize = subparsers.add_parser("normalize-manifest", help="normalize m1/m2/m3 with fill metadata")
    normalize.add_argument("--file", required=True)
    normalize.add_argument("--session")
    normalize.add_argument("--output")
    normalize.add_argument("--format", choices=FORMAT_VERSIONS, default="m3")
    normalize.set_defaults(handler=_normalize)

    import_manifest = subparsers.add_parser("import-manifest", help="atomically replace all manifest entries")
    import_manifest.add_argument("--session", required=True)
    import_manifest.add_argument("--file", required=True)
    import_manifest.add_argument("--reason", default="import")
    import_manifest.set_defaults(handler=_import_manifest)

    reconcile = subparsers.add_parser("reconcile-manifest", help="replace stale manifest from canonical DB state")
    reconcile.add_argument("--session", required=True)
    reconcile.add_argument("--reason", default="format-or-digest-mismatch")
    reconcile.set_defaults(handler=_reconcile_manifest)

    backups = subparsers.add_parser("manifest-backups", help="list read-only manifest backups")
    backups.add_argument("--session", required=True)
    backups.set_defaults(handler=_manifest_backups)

    prepare = subparsers.add_parser("prepare-samples", help="generate small/large/high-churn samples")
    prepare.add_argument("--workspace", default=".")
    prepare.add_argument("--keep-existing", action="store_true")
    prepare.set_defaults(handler=_prepare_samples)

    demo = subparsers.add_parser("demo", help="run the full repeatable acceptance demo")
    demo.add_argument("--workspace", default=".")
    demo.set_defaults(handler=_demo)

    doctor = subparsers.add_parser("doctor", help="check Python, sqlite3, Tk, NumPy and local writability")
    doctor.add_argument("--workspace", default=".")
    doctor.add_argument("--min-free-mb", type=int, default=64)
    doctor.set_defaults(handler=_doctor)
    return parser


def _log_error(args: argparse.Namespace, error: DiagnosticError) -> None:
    session_value = getattr(args, "session", None)
    if not session_value:
        return
    try:
        JsonEventLogger(Path(session_value).resolve()).emit_error(
            error,
            operation=str(getattr(args, "command", "cli")),
        )
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        _print_json(result)
        if result.get("ok") is False:
            code = result.get("code")
            return EXIT_CODES.get(code, 1) if isinstance(code, str) else 1
        return 0
    except DiagnosticError as error:
        _log_error(args, error)
        _print_json(error.to_dict())
        return EXIT_CODES.get(error.code, 1)
    except SimulatedInterruption as error:
        _print_json(
            {
                "ok": False,
                "code": E_STATE,
                "message": str(error),
                "field": "failpoint",
                "requires_rerun": True,
            }
        )
        return EXIT_CODES[E_STATE]
    except KeyboardInterrupt:
        _print_json({"ok": False, "code": E_STATE, "message": "interrupted by user"})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
