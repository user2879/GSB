"""Command-line interface and fault-injection entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .distributor import distribute
from .errors import EXIT_CODE_FOR_BIZ, BizError, ErrorCode
from .logging_setup import StructuredLogger, read_log_records
from .manifest import load_manifest, manifest_to_json
from .planner import build_plan, plan_fingerprint
from .replay import replay_latest
from .runtime import NodePool, check_ports_free, load_node_declaration
from .session import SessionStore
from .snapshot import build_snapshot, load_snapshot, restore_snapshot, save_snapshot
from .upgrade import read_upgrade_trail, resolve_manifest_path, upgrade_manifest


def _print(payload: dict | list) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def _runtime_layout(workdir: Path) -> dict[str, Path]:
    for name in ("db", "snapshots", "checkpoints", "backups", "logs"):
        (workdir / name).mkdir(parents=True, exist_ok=True)
    return {
        "db": workdir / "db" / "sessions.db",
        "snapshots": workdir / "snapshots",
        "checkpoints": workdir / "checkpoints",
        "backups": workdir / "backups",
        "logs": workdir / "logs" / "events.jsonl",
    }


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _print(
        {
            "ok": True,
            "source": args.manifest,
            "format_version": manifest["format_version"],
            "entry_count": len(manifest["entries"]),
        }
    )
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    layout = _runtime_layout(Path(args.workdir))
    logger = StructuredLogger(layout["logs"])
    nodes = [item["id"] for item in load_node_declaration(args.nodes)]
    report = upgrade_manifest(
        args.manifest,
        nodes,
        Path(args.workdir),
        limit=args.limit,
        logger=logger,
    )
    _print(report)
    return 0


def cmd_trail(args: argparse.Namespace) -> int:
    manifest_path = resolve_manifest_path(args.manifest)
    trail = Path(args.workdir) / "checkpoints" / f"{manifest_path.stem}.upgrade-trail.jsonl"
    _print({"trail": read_upgrade_trail(trail)})
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    node_ids = [item["id"] for item in load_node_declaration(args.nodes)]
    plan = build_plan(manifest, node_ids)
    _print({"plan": plan, "fingerprint": plan_fingerprint(plan)})
    return 0


def cmd_distribute(args: argparse.Namespace) -> int:
    layout = _runtime_layout(Path(args.workdir))
    logger = StructuredLogger(layout["logs"])
    manifest = load_manifest(args.manifest)
    node_specs = load_node_declaration(args.nodes)
    node_ids = [item["id"] for item in node_specs]
    plan = build_plan(manifest, node_ids)
    with NodePool(node_specs, logger=logger) as pool:
        with SessionStore(layout["db"]) as store:
            report = distribute(
                plan,
                node_specs,
                store,
                logger=logger,
                max_concurrency=args.max_concurrency,
            )
    _print(report)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    layout = _runtime_layout(Path(args.workdir))
    report = replay_latest(layout["db"], args.manifest, args.nodes)
    _print(report)
    return 0


def cmd_snapshot_save(args: argparse.Namespace) -> int:
    layout = _runtime_layout(Path(args.workdir))
    manifest = load_manifest(args.manifest)
    node_ids = [item["id"] for item in load_node_declaration(args.nodes)]
    plan = build_plan(manifest, node_ids)
    with SessionStore(layout["db"]) as store:
        committed = store.latest_committed_version()
        if committed is None:
            raise BizError(
                ErrorCode.E_STATE,
                "no committed session exists; run distribute first",
            )
        session_id = committed["session_id"]
        snapshot = build_snapshot(manifest, committed["plan"], store, session_id)
    target = layout["snapshots"] / f"snapshot-{manifest['format_version']}.json"
    save_snapshot(snapshot, target)
    _print({"ok": True, "snapshot": str(target), "manifest_version": manifest["format_version"]})
    return 0


def cmd_snapshot_restore(args: argparse.Namespace) -> int:
    layout = _runtime_layout(Path(args.workdir))
    logger = StructuredLogger(layout["logs"])
    current_manifest = load_manifest(args.manifest)
    archive_dir = layout["snapshots"] / "archive"
    with SessionStore(layout["db"]) as store:
        report = restore_snapshot(
            args.snapshot,
            current_manifest,
            store,
            session_id=f"restored-{Path(args.snapshot).stem}",
            archive_dir=archive_dir,
            logger=logger,
        )
    _print(report)
    return 0


def cmd_inject_corrupt(args: argparse.Namespace) -> int:
    """Inject corruption into an input file only (implementation untouched)."""
    layout = _runtime_layout(Path(args.workdir))
    logger = StructuredLogger(layout["logs"])
    source = Path(args.manifest)
    corrupted = Path(args.output) if args.output else source.with_suffix(".corrupt.json")
    text = source.read_text(encoding="utf-8")
    if args.kind == "duplicate":
        # Duplicate an entry key to trigger duplicate-key rejection.
        manifest = json.loads(text)
        manifest["entries"].append(dict(manifest["entries"][0]))
        corrupted.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif args.kind == "truncate":
        corrupted.write_text(text[: max(1, len(text) // 2)], encoding="utf-8")
    else:
        corrupted.write_text(text.replace('"format_version": 3', '"format_version": 9'), encoding="utf-8")
    logger.biz_error(
        "inject.corrupt_manifest",
        ErrorCode.E_FORMAT,
        f"corrupted manifest prepared: {args.kind}",
        path=str(corrupted),
    )
    # Validate immediately so the E_FORMAT chain is observable.
    load_manifest(corrupted)
    _print({"ok": True, "corrupted_manifest": str(corrupted)})
    return 0


def cmd_inject_port(args: argparse.Namespace) -> int:
    """Occupy the declared port, then run the platform port check -> E_PORT."""
    layout = _runtime_layout(Path(args.workdir))
    logger = StructuredLogger(layout["logs"])
    nodes = load_node_declaration(args.nodes)
    import socket

    holders = []
    for node in nodes:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((node["host"], node["port"]))
        except OSError as exc:
            sock.close()
            raise BizError(
                ErrorCode.E_PORT,
                f"cannot occupy port {node['port']}: {exc}",
                {"node": node["id"], "port": node["port"]},
            ) from exc
        sock.listen(1)
        holders.append(sock)
    logger.log("inject.port_occupied", nodes=[node["id"] for node in nodes])
    try:
        check_ports_free(nodes)
    finally:
        for sock in holders:
            sock.close()
    return 0


def cmd_inject_snapshot(args: argparse.Namespace) -> int:
    """Build a snapshot whose manifest version conflicts with the current one."""
    layout = _runtime_layout(Path(args.workdir))
    logger = StructuredLogger(layout["logs"])
    snapshot = load_snapshot(args.snapshot)
    current_manifest = load_manifest(args.manifest)
    # Change only the input snapshot data, never implementation code.
    conflicting_version = (
        current_manifest["format_version"] + 1
        if snapshot["manifest_version"] == current_manifest["format_version"]
        else snapshot["manifest_version"]
    )
    snapshot["manifest_version"] = conflicting_version
    target = layout["snapshots"] / f"conflicting-v{conflicting_version}.json"
    save_snapshot(snapshot, target)
    logger.log(
        "inject.snapshot_conflict_prepared",
        snapshot=str(target),
        snapshot_version=conflicting_version,
        current_version=current_manifest["format_version"],
    )
    archive_dir = layout["snapshots"] / "archive"
    with SessionStore(layout["db"]) as store:
        restore_snapshot(
            target,
            current_manifest,
            store,
            session_id="restored-conflict",
            archive_dir=archive_dir,
            logger=logger,
        )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    layout = _runtime_layout(Path(args.workdir))
    records = read_log_records(layout["logs"])
    if args.code:
        records = [item for item in records if item.get("error_code") == args.code]
    _print(records)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfgdist",
        description="Cross-component configuration distribution and recovery platform",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func):
        item = sub.add_parser(name)
        item.set_defaults(func=func)
        return item

    validate = add("validate", cmd_validate)
    validate.add_argument("manifest")

    upgrade = add("upgrade", cmd_upgrade)
    upgrade.add_argument("manifest")
    upgrade.add_argument("--nodes", required=True)
    upgrade.add_argument("--workdir", default="run")
    upgrade.add_argument("--limit", type=int, default=None)

    trail = add("trail", cmd_trail)
    trail.add_argument("manifest")
    trail.add_argument("--workdir", default="run")

    plan = add("plan", cmd_plan)
    plan.add_argument("manifest")
    plan.add_argument("--nodes", required=True)

    distribute_cmd = add("distribute", cmd_distribute)
    distribute_cmd.add_argument("manifest")
    distribute_cmd.add_argument("--nodes", required=True)
    distribute_cmd.add_argument("--workdir", default="run")
    distribute_cmd.add_argument("--max-concurrency", type=int, default=4)

    replay = add("replay", cmd_replay)
    replay.add_argument("manifest")
    replay.add_argument("--nodes", required=True)
    replay.add_argument("--workdir", default="run")

    snap_save = add("snapshot-save", cmd_snapshot_save)
    snap_save.add_argument("manifest")
    snap_save.add_argument("--nodes", required=True)
    snap_save.add_argument("--workdir", default="run")

    snap_restore = add("snapshot-restore", cmd_snapshot_restore)
    snap_restore.add_argument("snapshot")
    snap_restore.add_argument("--manifest", required=True)
    snap_restore.add_argument("--workdir", default="run")

    corrupt = add("inject-corrupt", cmd_inject_corrupt)
    corrupt.add_argument("manifest")
    corrupt.add_argument("--kind", choices=["duplicate", "truncate", "unknown-version"],
                         default="duplicate")
    corrupt.add_argument("--output")
    corrupt.add_argument("--workdir", default="run")

    port = add("inject-port", cmd_inject_port)
    port.add_argument("--nodes", required=True)
    port.add_argument("--workdir", default="run")

    snap_inject = add("inject-snapshot", cmd_inject_snapshot)
    snap_inject.add_argument("snapshot")
    snap_inject.add_argument("--manifest", required=True)
    snap_inject.add_argument("--workdir", default="run")

    logs = add("logs", cmd_logs)
    logs.add_argument("--workdir", default="run")
    logs.add_argument("--code")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BizError as exc:
        logger = StructuredLogger(
            Path(getattr(args, "workdir", "run")) / "logs" / "events.jsonl"
        )
        logger.biz_error(
            f"cli.{getattr(args, 'command', 'unknown')}",
            exc.code,
            exc.message,
            **exc.details,
        )
        _print(exc.to_dict())
        return EXIT_CODE_FOR_BIZ[exc.code]


if __name__ == "__main__":
    raise SystemExit(main())
