from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, canonical_json_text, digest_json, load_json, pretty_json_bytes
from .catalog import Catalog, ForcedTermination, scan_catalog
from .config import InteropConfig, load_config
from .errors import InteropError, fail
from .formats import decode_manifest
from .ledger import Ledger, load_ledger
from .logging import StructuredLogger
from .normalize import compare_manifests, compare_upgrade_paths, normalize_manifest
from .samples import FixturePaths, generate_fixtures
from .snapshot import SnapshotStore


def _reconfigure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _relative(path: str | Path, root: str | Path) -> str:
    path_obj = Path(path)
    root_obj = Path(root)
    try:
        return path_obj.resolve().relative_to(root_obj.resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _load_config_ledger(config_path: str | Path | None, ledger_path: str | Path) -> tuple[InteropConfig, Ledger]:
    ledger = load_ledger(ledger_path)
    if config_path is None:
        return ledger.config, ledger
    config = load_config(config_path)
    if digest_json(config.to_dict()) != ledger.config_digest:
        fail(
            "E_CONFIG",
            "config does not match the truth ledger",
            path="config",
            manifest_path=str(config_path),
            details={"field": "config", "ledger_config_digest": ledger.config_digest},
        )
    return config, ledger


def _print_json(value: Any) -> None:
    print(canonical_json_text(value), flush=True)


def _logger_from(args: argparse.Namespace) -> StructuredLogger:
    return StructuredLogger(getattr(args, "log_file", None))


def _fixture_summary(fixtures: FixturePaths) -> dict[str, Any]:
    root = fixtures.workspace
    return {
        "ok": True,
        "operation": "generate",
        "workspace": str(root),
        "config": _relative(fixtures.config, root),
        "ledger": _relative(fixtures.ledger, root),
        "direct": _relative(fixtures.direct, root),
        "historical_v1": _relative(fixtures.historical_v1, root),
        "historical_v2": _relative(fixtures.historical_v2, root),
        "crash_catalog": _relative(fixtures.crash_catalog, root),
        "target_step": fixtures.target_step,
    }


def command_generate(args: argparse.Namespace) -> int:
    fixtures = generate_fixtures(args.workspace)
    _print_json(_fixture_summary(fixtures))
    return 0


def command_open(args: argparse.Namespace) -> int:
    config, ledger = _load_config_ledger(args.config, args.ledger)
    report = Catalog(args.catalog, config, ledger).open(
        upgrade=not args.no_upgrade,
        retry_unknown=args.retry_unknown,
    )
    _print_json(report.to_dict())
    return 0


def command_upgrade(args: argparse.Namespace) -> int:
    config, ledger = _load_config_ledger(args.config, args.ledger)
    report = Catalog(args.catalog, config, ledger).open(
        upgrade=True,
        retry_unknown=args.retry_unknown,
        terminate_after_completed=args.crash_after_completed,
        terminate_after_rewrite=args.crash_after_rewrite,
    )
    _print_json(report.to_dict())
    return 0


def command_normalize(args: argparse.Namespace) -> int:
    _, ledger = _load_config_ledger(None, args.ledger)
    manifest = decode_manifest(args.manifest)
    expected_view = None
    if args.expected_manifest:
        expected_manifest = decode_manifest(args.expected_manifest)
        if expected_manifest.version != 3:
            fail(
                "E_FORMAT",
                "expected manifest must be v3",
                path="format_version",
                manifest_path=args.expected_manifest,
            )
        expected_view = expected_manifest.to_view()
    normalized = normalize_manifest(
        manifest,
        ledger,
        manifest_path=args.manifest,
        expected_view=expected_view,
    )
    report = dict(normalized.report)
    report["expected_manifest"] = args.expected_manifest
    _print_json(report)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    _, ledger = _load_config_ledger(None, args.ledger)
    left = decode_manifest(args.left)
    right = decode_manifest(args.right)
    _print_json(
        compare_manifests(left, right, ledger, left_label=args.left, right_label=args.right)
    )
    return 0


def _snapshot_store(args: argparse.Namespace) -> SnapshotStore:
    return SnapshotStore(args.snapshot_db)


def command_snapshot_save(args: argparse.Namespace) -> int:
    config, ledger = _load_config_ledger(args.config, args.ledger)
    catalog = Catalog(args.catalog, config, ledger)
    with _snapshot_store(args) as store:
        snapshot = store.save(catalog)
    _print_json(
        {
            "ok": True,
            "operation": "snapshot_save",
            "snapshot_id": snapshot.id,
            "format_versions": snapshot.format_versions,
            "content_digest": snapshot.content_digest,
            "normalized_view_digest": snapshot.normalized_view_digest,
            "item_count": len(snapshot.items),
            "readonly": snapshot.readonly,
        }
    )
    return 0


def command_snapshot_restore(args: argparse.Namespace) -> int:
    config, ledger = _load_config_ledger(args.config, args.ledger)
    catalog = Catalog(args.catalog, config, ledger)
    with _snapshot_store(args) as store:
        report = store.restore(
            catalog,
            snapshot_id=args.snapshot_id,
            verify_replay=not args.no_replay,
        )
    _print_json(report)
    return 0


def command_snapshot_replay(args: argparse.Namespace) -> int:
    config, ledger = _load_config_ledger(args.config, args.ledger)
    catalog = Catalog(args.catalog, config, ledger)
    with _snapshot_store(args) as store:
        report = store.replay(catalog, snapshot_id=args.snapshot_id)
    _print_json(report)
    return 0


def _session_open(
    request: dict[str, Any],
    defaults: argparse.Namespace,
) -> tuple[Catalog, dict[str, Any]]:
    catalog_path = request.get("catalog", defaults.catalog)
    config_path = request.get("config", defaults.config)
    ledger_path = request.get("ledger", defaults.ledger)
    if not catalog_path or not ledger_path:
        fail("E_CONFIG", "session open requires catalog and ledger paths", path="catalog")
    config, ledger = _load_config_ledger(config_path, ledger_path)
    catalog = Catalog(catalog_path, config, ledger)
    report = catalog.open(
        upgrade=bool(request.get("upgrade", True)),
        retry_unknown=bool(request.get("retry_unknown", defaults.retry_unknown)),
    )
    return catalog, report.to_dict()


def command_session(args: argparse.Namespace) -> int:
    if not args.catalog or not args.ledger:
        fail("E_CONFIG", "session requires --catalog and --ledger", path="catalog")
    session_config, session_ledger = _load_config_ledger(args.config, args.ledger)
    current_catalog = Catalog(args.catalog, session_config, session_ledger)
    last_upgrade: dict[str, Any] | None = None
    had_error = False
    for line_number, line in enumerate(sys.stdin, start=1):
        text = line.strip()
        if not text:
            continue
        request: Any = None
        try:
            request = json.loads(text)
            if not isinstance(request, dict):
                fail("E_CONFIG", "session request must be a JSON object", path="$")
            command = request.get("command")
            if not command:
                fail("E_CONFIG", "session request is missing command", path="command")
            if command == "quit":
                _print_json({"ok": True, "command": "quit", "request_id": request.get("request_id")})
                break
            if command == "ping":
                response = {"ok": True, "command": "ping"}
            elif command == "scan":
                catalog_path = request.get("catalog", args.catalog)
                records = scan_catalog(catalog_path)
                response = {
                    "ok": True,
                    "command": "scan",
                    "manifests": [
                        {
                            "path": manifest.source_path.name if manifest.source_path else None,
                            "format_version": manifest.version,
                            "global_step": manifest.global_step,
                            "manifest_id": manifest.derived_id(),
                        }
                        for manifest in records
                    ],
                }
            elif command in {"open", "import", "upgrade"}:
                current_catalog, response = _session_open(request, args)
                response = {"command": command, **response}
                last_upgrade = response.get("upgrade")
            elif command == "normalize":
                manifest_path = request.get("manifest")
                if not manifest_path:
                    fail("E_CONFIG", "normalize requires manifest", path="manifest")
                manifest = decode_manifest(manifest_path)
                normalized = normalize_manifest(manifest, current_catalog.ledger, manifest_path=manifest_path)
                response = {"ok": True, "command": "normalize", **normalized.report}
            elif command == "compare":
                left_path = request.get("left")
                right_path = request.get("right")
                if not left_path or not right_path:
                    fail("E_CONFIG", "compare requires left and right", path="left")
                response = {
                    "ok": True,
                    "command": "compare",
                    **compare_manifests(
                        decode_manifest(left_path),
                        decode_manifest(right_path),
                        current_catalog.ledger,
                        left_label=left_path,
                        right_label=right_path,
                    ),
                }
            elif command == "snapshot_save":
                catalog_path = request.get("catalog")
                target_catalog = current_catalog
                if catalog_path:
                    target_catalog = Catalog(catalog_path, current_catalog.config, current_catalog.ledger)
                trace = None
                cursor = None
                if last_upgrade is not None:
                    trace = last_upgrade.get("trace")
                    cursor = last_upgrade.get("completed")
                with SnapshotStore(request.get("snapshot_db", args.snapshot_db)) as store:
                    snapshot = store.save(target_catalog, upgrade_trace=trace, processed_cursor=cursor)
                response = {
                    "ok": True,
                    "command": "snapshot_save",
                    "snapshot_id": snapshot.id,
                    "content_digest": snapshot.content_digest,
                    "normalized_view_digest": snapshot.normalized_view_digest,
                }
            elif command == "snapshot_restore":
                with SnapshotStore(request.get("snapshot_db", args.snapshot_db)) as store:
                    response = {
                        "command": "snapshot_restore",
                        **store.restore(
                            current_catalog,
                            snapshot_id=request.get("snapshot_id"),
                            verify_replay=bool(request.get("verify_replay", True)),
                        ),
                    }
            elif command == "snapshot_replay":
                with SnapshotStore(request.get("snapshot_db", args.snapshot_db)) as store:
                    response = {
                        "command": "snapshot_replay",
                        **store.replay(current_catalog, snapshot_id=request.get("snapshot_id")),
                    }
            elif command == "snapshot_list":
                with SnapshotStore(request.get("snapshot_db", args.snapshot_db)) as store:
                    response = {"ok": True, "command": "snapshot_list", "snapshots": store.list()}
            else:
                fail("E_CONFIG", f"unknown session command {command}", path="command")
        except InteropError as error:
            had_error = True
            _logger_from(args).error(error, operation="session")
            response = {"command": "session", **error.to_dict(), "line": line_number}
        response["request_id"] = request.get("request_id") if isinstance(request, dict) else None
        _print_json(response)
    return 1 if had_error else 0


def _inject_json_field(path: Path, mutate: Any) -> None:
    raw = load_json(path)
    if not isinstance(raw, dict):
        fail("E_INTERNAL", "injection target is not a JSON object", manifest_path=str(path))
    mutate(raw)
    atomic_write_bytes(path, pretty_json_bytes(raw))


def _run_injection(workspace: Path, kind: str) -> int:
    fixtures = generate_fixtures(workspace)
    log_path = fixtures.workspace / "logs" / f"injection-{kind}.jsonl"
    logger = StructuredLogger(log_path)
    operation = f"inject_{kind}"
    try:
        if kind == "integrity":
            target = fixtures.v1_target
            _inject_json_field(
                target,
                lambda raw: raw["shards"][2].__setitem__("parameter_digest", "0" * 64),
            )
            Catalog(fixtures.historical_v1, load_config(fixtures.config), load_ledger(fixtures.ledger)).open()
        elif kind == "chain":
            target = fixtures.direct_target
            _inject_json_field(
                target,
                lambda raw: raw["parent_manifest"].__setitem__("manifest_id", "step-missing"),
            )
            Catalog(fixtures.direct, load_config(fixtures.config), load_ledger(fixtures.ledger)).open()
        elif kind == "upgrade":
            config = load_config(fixtures.config)
            ledger = load_ledger(fixtures.ledger)
            crash_catalog = Catalog(fixtures.crash_catalog, config, ledger)
            crash_catalog.open()
            checkpoint_path = crash_catalog.checkpoint_path
            _inject_json_field(
                checkpoint_path,
                lambda raw: raw["items"][0].__setitem__("source_sha256", "0" * 64),
            )
            crash_catalog.open()
        else:
            fail("E_CONFIG", f"unknown injection kind {kind}", path="injection")
        fail("E_INTERNAL", "injection did not trigger the expected failure", path="injection")
    except InteropError as error:
        logger.error(error, operation=operation)
        payload = error.to_dict()
        payload.update(
            {
                "operation": operation,
                "log_file": str(log_path),
                "injected_input_only": True,
            }
        )
        _print_json(payload)
        return error.exit_code
    return 19


def _relative_report(report: dict[str, Any], root: Path) -> dict[str, Any]:
    result = dict(report)
    for key in ("catalog",):
        if isinstance(result.get(key), str):
            result[key] = _relative(result[key], root)
    if isinstance(result.get("manifests"), list):
        result["manifests"] = [dict(item) for item in result["manifests"]]
    return result


def _run_demo(workspace: Path, injection: str | None = None) -> int:
    if injection is not None:
        return _run_injection(workspace, injection)

    fixtures = generate_fixtures(workspace)
    config = load_config(fixtures.config)
    ledger = load_ledger(fixtures.ledger)
    root = fixtures.workspace
    logger = StructuredLogger(root / "logs" / "demo.jsonl")

    direct_catalog = Catalog(fixtures.direct, config, ledger)
    direct_open = direct_catalog.open()
    direct_normalized = normalize_manifest(
        decode_manifest(fixtures.direct_target), ledger, manifest_path=fixtures.direct_target
    )

    equivalence = compare_upgrade_paths(
        decode_manifest(fixtures.v1_target),
        decode_manifest(fixtures.v2_target),
        decode_manifest(fixtures.direct_target),
        ledger,
    )
    v1_catalog = Catalog(fixtures.historical_v1, config, ledger)
    v1_open = v1_catalog.open()
    v2_catalog = Catalog(fixtures.historical_v2, config, ledger)
    v2_open = v2_catalog.open()

    crash_catalog = Catalog(fixtures.crash_catalog, config, ledger)
    package_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_root) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    crash_command = [
        sys.executable,
        "-m",
        "ckptmeta",
        "upgrade",
        "--catalog",
        str(fixtures.crash_catalog),
        "--config",
        str(fixtures.config),
        "--ledger",
        str(fixtures.ledger),
        "--crash-after-completed",
        "1",
        "--log-file",
        str(root / "logs" / "forced-termination.jsonl"),
    ]
    crashed = subprocess.run(
        crash_command,
        cwd=package_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if crashed.returncode == 0:
        fail("E_INTERNAL", "forced-termination subprocess unexpectedly exited successfully")
    after_crash_versions = [
        {"path": manifest.source_path.name, "format_version": manifest.version}
        for manifest in scan_catalog(fixtures.crash_catalog)
    ]
    resumed = crash_catalog.open()

    snapshot_path = root / "snapshots" / "interop.sqlite3"
    with SnapshotStore(snapshot_path) as store:
        snapshot = store.save(direct_catalog)
        restored = store.restore(direct_catalog)
        replayed = store.replay(direct_catalog)

    all_equal = bool(equivalence["byte_identical"])
    summary = {
        "ok": all_equal and resumed.upgrade is not None,
        "operation": "demo",
        "workspace": str(root),
        "scenario_1_current_direct": {
            "manifest_list": direct_open.to_dict()["manifests"],
            "normalized_step": direct_normalized.report["source_format_version"],
            "view_digest": direct_normalized.view_digest,
            "verified_digest_count": direct_normalized.report["verified_digest_count"],
        },
        "scenario_2_historical_upgrade": {
            "manifest_list_v1": _relative_report(v1_open.to_dict(), root)["manifests"],
            "manifest_list_v2": _relative_report(v2_open.to_dict(), root)["manifests"],
            "upgrade_trace_v1": v1_open.upgrade["trace"] if v1_open.upgrade else [],
            "upgrade_trace_v2": v2_open.upgrade["trace"] if v2_open.upgrade else [],
            "v1_filled_fields": equivalence["filled_fields"]["v1"],
            "v2_filled_fields": equivalence["filled_fields"]["v2"],
            "byte_identical": equivalence["byte_identical"],
            "view_digest": equivalence["view_digest"],
        },
        "scenario_3_forced_termination_resume": {
            "forced_exit_code": crashed.returncode,
            "interrupted_run_exit_report": "unavailable",
            "after_crash_versions": after_crash_versions,
            "resume_status": resumed.upgrade["status"] if resumed.upgrade else None,
            "resume_processed": [item["manifest_path"] for item in resumed.upgrade["processed"]]
            if resumed.upgrade
            else [],
            "resume_skipped_done": resumed.upgrade["skipped_done"] if resumed.upgrade else [],
            "resume_completed": resumed.upgrade["completed"] if resumed.upgrade else 0,
            "previous_run_report": resumed.upgrade["previous_run_report"] if resumed.upgrade else None,
        },
        "scenario_4_snapshot_replay": {
            "snapshot_id": snapshot.id,
            "format_versions": snapshot.format_versions,
            "content_digest": snapshot.content_digest,
            "normalized_view_digest": snapshot.normalized_view_digest,
            "restore_matched": restored["matched"],
            "replay_reproduced": replayed["reproduced"],
            "compared_item_count": replayed["compared_item_count"],
        },
        "summary": {
            "all_normalized_views_byte_identical": equivalence["byte_identical"],
            "upgrade_resume_complete": resumed.upgrade["status"] == "resumed_complete"
            if resumed.upgrade
            else False,
            "snapshot_replay_reproduced": replayed["reproduced"],
            "all_checks_passed": all_equal and bool(resumed.upgrade) and replayed["reproduced"],
        },
    }
    logger.info("demo completed", operation="demo", details={"all_checks_passed": summary["summary"]["all_checks_passed"]})
    _print_json(summary)
    return 0 if summary["summary"]["all_checks_passed"] else 19


def command_demo(args: argparse.Namespace) -> int:
    injection = None
    selected = [name for name in ("integrity", "chain", "upgrade") if getattr(args, f"inject_{name}")]
    if len(selected) > 1:
        fail("E_CONFIG", "only one injection mode may be selected", path="injection")
    if selected:
        injection = selected[0]
    return _run_demo(Path(args.workspace), injection)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ckptmeta", description="Training checkpoint metadata interoperability toolkit")
    parser.add_argument("--log-file", default=None, help="structured JSONL log path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def with_log(subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        subparser.add_argument("--log-file", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        return subparser

    generate = with_log(subparsers.add_parser("generate", help="generate deterministic local fixtures"))
    generate.add_argument("--workspace", required=True)
    generate.set_defaults(handler=command_generate)

    open_parser = with_log(
        subparsers.add_parser(
            "open",
            aliases=["import"],
            help="validate and optionally upgrade a catalog",
        )
    )
    open_parser.add_argument("--catalog", required=True)
    open_parser.add_argument("--config", required=True)
    open_parser.add_argument("--ledger", required=True)
    open_parser.add_argument("--no-upgrade", action="store_true")
    open_parser.add_argument("--retry-unknown", action="store_true")
    open_parser.set_defaults(handler=command_open)

    upgrade = with_log(subparsers.add_parser("upgrade", help="upgrade all historical manifests in a catalog"))
    upgrade.add_argument("--catalog", required=True)
    upgrade.add_argument("--config", required=True)
    upgrade.add_argument("--ledger", required=True)
    upgrade.add_argument("--retry-unknown", action="store_true")
    upgrade.add_argument("--crash-after-completed", type=int, default=None)
    upgrade.add_argument("--crash-after-rewrite", type=int, default=None)
    upgrade.set_defaults(handler=command_upgrade)

    normalize = with_log(subparsers.add_parser("normalize", help="normalize one manifest and compare with the ledger"))
    normalize.add_argument("--manifest", required=True)
    normalize.add_argument("--ledger", required=True)
    normalize.add_argument(
        "--expected-manifest",
        default=None,
        help="optional direct v3 manifest for an E_RESTORE byte comparison",
    )
    normalize.set_defaults(handler=command_normalize)

    compare = with_log(subparsers.add_parser("compare", help="compare two manifests through canonical views"))
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--ledger", required=True)
    compare.set_defaults(handler=command_compare)

    snapshot_save = with_log(subparsers.add_parser("snapshot-save", help="save an interoperability session snapshot"))
    snapshot_save.add_argument("--catalog", required=True)
    snapshot_save.add_argument("--config", required=True)
    snapshot_save.add_argument("--ledger", required=True)
    snapshot_save.add_argument("--snapshot-db", required=True)
    snapshot_save.set_defaults(handler=command_snapshot_save)

    snapshot_restore = with_log(subparsers.add_parser("snapshot-restore", help="restore and replay a snapshot"))
    snapshot_restore.add_argument("--catalog", required=True)
    snapshot_restore.add_argument("--config", required=True)
    snapshot_restore.add_argument("--ledger", required=True)
    snapshot_restore.add_argument("--snapshot-db", required=True)
    snapshot_restore.add_argument("--snapshot-id", type=int, default=None)
    snapshot_restore.add_argument("--no-replay", action="store_true")
    snapshot_restore.set_defaults(handler=command_snapshot_restore)

    snapshot_replay = with_log(subparsers.add_parser("snapshot-replay", help="re-normalize and compare with a snapshot"))
    snapshot_replay.add_argument("--catalog", required=True)
    snapshot_replay.add_argument("--config", required=True)
    snapshot_replay.add_argument("--ledger", required=True)
    snapshot_replay.add_argument("--snapshot-db", required=True)
    snapshot_replay.add_argument("--snapshot-id", type=int, default=None)
    snapshot_replay.set_defaults(handler=command_snapshot_replay)

    session = with_log(subparsers.add_parser("session", help="run the JSONL stdin/stdout interoperability command loop"))
    session.add_argument("--catalog", required=True)
    session.add_argument("--config", default=None)
    session.add_argument("--ledger", required=True)
    session.add_argument("--snapshot-db", required=True)
    session.add_argument("--retry-unknown", action="store_true")
    session.set_defaults(handler=command_session)

    demo = with_log(subparsers.add_parser("demo", help="run the complete deterministic demonstration"))
    demo.add_argument("--workspace", default=".interop-demo")
    injection_group = demo.add_mutually_exclusive_group()
    injection_group.add_argument("--inject-integrity", action="store_true")
    injection_group.add_argument("--inject-chain", action="store_true")
    injection_group.add_argument("--inject-upgrade", action="store_true")
    demo.set_defaults(handler=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = _logger_from(args)
    try:
        return int(args.handler(args))
    except ForcedTermination as termination:
        logger.emit(
            business_code="FORCED_TERMINATION",
            message="process terminated without an exit report",
            operation=args.command,
            level="ERROR",
            details={"phase": termination.phase, "exit_report": "unavailable"},
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(termination.exit_code)
    except InteropError as error:
        logger.error(error, operation=args.command)
        _print_json(error.to_dict())
        return error.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # defensive boundary: never leak an unstructured failure
        wrapped = InteropError("E_INTERNAL", f"unexpected failure: {error}", details={"type": type(error).__name__})
        logger.error(wrapped, operation=args.command)
        _print_json(wrapped.to_dict())
        return wrapped.exit_code
