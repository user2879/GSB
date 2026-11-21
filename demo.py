"""Command line demonstration driver (Windows: `python demo.py <command>`)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lease_middleware.codec import decode_records  # noqa: E402
from lease_middleware.engine import LeaseEngine  # noqa: E402
from lease_middleware.errors import MiddlewareError  # noqa: E402
from lease_middleware import inject  # noqa: E402
from lease_middleware.replay import Replayer  # noqa: E402
from lease_middleware.scenario import build_fresh, import_legacy_leases  # noqa: E402
from lease_middleware.spec import load_spec  # noqa: E402
from lease_middleware.storage import Storage  # noqa: E402
from lease_middleware.upgrade import UpgradeManager  # noqa: E402
from lease_middleware.verifier import verify  # noqa: E402

SAMPLES = ROOT / "samples"
RUNTIME = ROOT / "demo_runtime"
LEGACY = ["leases_r1.json", "leases_r2.json", "leases_r3.json"]


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def runtime_dir(name: str) -> Path:
    path = RUNTIME / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_interop(_args: argparse.Namespace) -> None:
    spec = load_spec(SAMPLES / "config.json")
    raws = json.loads((SAMPLES / "leases_same_lease.json").read_text(encoding="utf-8"))
    records = decode_records(raws, spec)
    views = [{k: v for k, v in r.to_dict().items() if k != "format"} for r in records]
    identical = all(v == views[0] for v in views)
    emit({"command": "interop", "normalized_views": views, "identical": identical,
          "shared_lease_version": views[0]["lease_version"],
          "shared_digest": views[0]["value_digest"]})


def cmd_run(args: argparse.Namespace) -> None:
    out = runtime_dir("run")
    db = out / "ledger.sqlite"
    for f in db.parent.glob("*"):
        f.unlink()
    legacy = [SAMPLES / name for name in LEGACY]
    result = build_fresh(SAMPLES / "config.json", db, legacy, SAMPLES / "changes.json")
    replay = Replayer(Storage(db), load_spec(SAMPLES / "config.json")).run()
    result["replay"] = replay
    result["verification"] = verify(db, SAMPLES / "config.json")
    emit({"command": "run", **result})


def _prepare_legacy_db(out: Path) -> Path:
    db = out / "legacy.sqlite"
    if db.exists():
        db.unlink()
    spec = load_spec(SAMPLES / "config.json")
    storage = Storage(db)
    import_legacy_leases(storage, spec, [SAMPLES / "leases_r1.json",
                                         SAMPLES / "leases_r2.json"])
    storage.close()
    return db


def cmd_upgrade(args: argparse.Namespace) -> None:
    out = runtime_dir("upgrade")
    db = _prepare_legacy_db(out)
    spec = load_spec(SAMPLES / "config.json")
    if args.crash:
        # First pass hard-kills itself after the first durable record upgrade.
        env = dict(os.environ, LM_CRASH_AFTER_FIRST="1")
        proc = subprocess.run(
            [sys.executable, "demo.py", "_upgrade_worker", str(db),
             str(SAMPLES / "config.json")], cwd=ROOT, env=env)
        code = proc.returncode
        checkpoint = json.loads((db.with_suffix(".upgrade.json")).read_text())
        # Resume pass must skip already-upgraded rows and finish the rest.
        storage = Storage(db)
        report = UpgradeManager(storage, spec).run()
        rows = storage.list_lease_rows()
        emit({"command": "upgrade", "forced_termination": True,
              "worker_exit_code": code,
              "checkpoint_before_resume": checkpoint,
              "resume_report": report.to_dict(),
              "all_current_format": all(r["format"] == "r3" for r in rows),
              "all_have_backups": all(r["backup_json"] is not None for r in rows
                                      if r["upgraded"])})
        storage.close()
    else:
        storage = Storage(db)
        report = UpgradeManager(storage, spec).run()
        emit({"command": "upgrade", "forced_termination": False,
              "report": report.to_dict(),
              "rows": [{"lease_id": r["lease_id"], "format": r["format"],
                        "upgraded": bool(r["upgraded"]),
                        "backup_present": r["backup_json"] is not None}
                       for r in storage.list_lease_rows()]})
        storage.close()


def cmd_upgrade_worker(args: argparse.Namespace) -> None:
    spec = load_spec(Path(args.spec))
    storage = Storage(Path(args.db))
    manager = UpgradeManager(storage, spec)
    upgraded_id = manager.upgrade_next()
    storage.close()
    if upgraded_id is not None and os.environ.get("LM_CRASH_AFTER_FIRST") == "1":
        os._exit(137)  # simulate forced termination, no exit report


def cmd_rebuild(_args: argparse.Namespace) -> None:
    out = runtime_dir("rebuild")
    db = out / "ledger.sqlite"
    if db.exists():
        db.unlink()
    spec = load_spec(SAMPLES / "config.json")
    storage = Storage(db)
    import_legacy_leases(storage, spec, [SAMPLES / n for n in LEGACY])
    UpgradeManager(storage, spec).run()
    engine = LeaseEngine(spec, storage)
    engine.rebuild_subscription_view("initial")
    fresh = engine.grant("client-r3", "service.endpoint", config_version=1)
    # renewed but not durably committed -> unknown after rebuild
    engine.renew(fresh.lease_id, 1, commit=False)
    evolved = _evolved_spec(spec)
    before = storage.get_all_cursors()
    diff = engine.rebuild_subscription_view("client_set_evolution", evolved)
    emit({"command": "rebuild", "old_cursors": before,
          "diff": {"added": diff.added, "removed": diff.removed,
                   "upgraded": diff.upgraded,
                   "cursor_changes": diff.cursor_changes,
                   "unknown_renewals": diff.unknown_renewals},
          "view": [dict(r) for r in storage.list_subscription_view()],
          "cursors": storage.get_all_cursors()})
    storage.close()


def _first(storage: Storage, client_id: str) -> str:
    for r in storage.list_lease_rows():
        if r["client_id"] == client_id:
            return r["lease_id"]
    raise RuntimeError("no lease")


def _evolved_spec(spec) -> "Spec":
    data = json.loads(spec.source_path.read_text(encoding="utf-8"))
    # remove r1 client, add a brand new r3 client, and upgrade r2 -> r3
    data["clients"] = [c for c in data["clients"] if c["id"] != "client-r1"]
    for client in data["clients"]:
        if client["id"] == "client-r2":
            client["format_version"] = "r3"
    data["clients"].append(
        {"id": "client-new", "subscribed_keys": ["cache.policy"],
         "lease_duration_ms": 500, "format_version": "r3"})
    evolved_path = RUNTIME / "rebuild" / "evolved_config.json"
    evolved_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return load_spec(evolved_path)


def cmd_inject(args: argparse.Namespace) -> None:
    out = runtime_dir("inject")
    log_path = out / "structured.log"
    if args.kind == "spec":
        bad = inject.inject_bad_spec(SAMPLES / "config.json", out / "bad_config.json")
        try:
            load_spec(bad)
        except MiddlewareError as exc:
            _log_and_emit(log_path, exc)
    elif args.kind == "lease":
        bad = inject.inject_bad_lease(SAMPLES / "leases_r3.json", out / "bad_leases.json")
        spec = load_spec(SAMPLES / "config.json")
        try:
            import_legacy_leases(Storage(out / "ignored.sqlite"), spec, [bad])
        except MiddlewareError as exc:
            _log_and_emit(log_path, exc)
    elif args.kind == "cursor":
        db = out / "cursor.sqlite"
        if db.exists():
            db.unlink()
        build_fresh(SAMPLES / "config.json", db,
                    [SAMPLES / n for n in LEGACY], SAMPLES / "changes.json")
        inject.inject_cursor_ahead(db, "client-r3")
        try:
            verify(db, SAMPLES / "config.json")
        except MiddlewareError as exc:
            _log_and_emit(log_path, exc)
    else:  # pragma: no cover
        raise SystemExit("unknown injection kind")


def _log_and_emit(log_path: Path, exc: MiddlewareError) -> None:
    from lease_middleware.errors import StructuredLogger

    StructuredLogger(log_path).error(exc)
    emit({"captured": {"code": exc.code, "message": exc.message,
                       "field": exc.field_path, "client_id": exc.client_id,
                       "lease_id": exc.lease_id, "context": exc.context},
          "log_file": str(log_path)})


def cmd_verify(args: argparse.Namespace) -> None:
    emit(verify(Path(args.db), Path(args.spec)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config-center lease middleware demo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("interop").set_defaults(func=cmd_interop)
    sub.add_parser("run").set_defaults(func=cmd_run)
    up = sub.add_parser("upgrade")
    up.add_argument("--crash", action="store_true")
    up.set_defaults(func=cmd_upgrade)
    worker = sub.add_parser("_upgrade_worker")
    worker.add_argument("db")
    worker.add_argument("spec")
    worker.set_defaults(func=cmd_upgrade_worker)
    sub.add_parser("rebuild").set_defaults(func=cmd_rebuild)
    inj = sub.add_parser("inject")
    inj.add_argument("kind", choices=["spec", "lease", "cursor"])
    inj.set_defaults(func=cmd_inject)
    ver = sub.add_parser("verify")
    ver.add_argument("db")
    ver.add_argument("spec")
    ver.set_defaults(func=cmd_verify)
    sub.add_parser("dashboard").set_defaults(func=lambda _a: _dashboard())
    return parser


def _dashboard() -> None:
    try:
        from lease_middleware.dashboard import launch_dashboard
        launch_dashboard(RUNTIME / "run" / "ledger.sqlite", SAMPLES / "config.json")
    except Exception as exc:  # Tk missing in headless CI
        emit({"dashboard": "unavailable", "reason": str(exc)})


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except MiddlewareError as exc:
        emit({"unhandled_business_error": {"code": exc.code, "message": exc.message,
                                           "field": exc.field_path}})
        raise SystemExit(2)


if __name__ == "__main__":
    main()
