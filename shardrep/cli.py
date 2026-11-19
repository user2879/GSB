"""Command line interface for validation, injection, scheduling and demos."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .audit import JsonlReport, StructuredLogger
from .demo import run_demo
from .errors import (
    IntegrityError,
    ShardRepairError,
    SnapshotError,
    SpecValidationError,
)
from .ledger import Ledger
from .packing import PackStore, build_pack_tasks, compare_artifact_sets
from .propagation import propagate_invalidation
from .registry import ReplicaRegistry
from .repair import apply_and_report, block_replica_digest
from .sample import generate_sample
from .scheduler import PackScheduler
from .snapshot import SnapshotManager, tamper_snapshot_copy
from .spec import load_spec, spec_manifest_digest


def _common_paths(args: argparse.Namespace, spec_path: Path) -> tuple[Path, Path]:
    root = spec_path.parent.parent if spec_path.parent.name == "samples" else Path.cwd()
    log_dir = Path(getattr(args, "log_dir", None) or root / "run" / "logs")
    db_path = Path(getattr(args, "db", None) or root / "run" / "shardrep.sqlite3")
    log_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return log_dir / "events.jsonl", db_path


def _runtime(args: argparse.Namespace, spec_path: Path) -> tuple[Any, Any, Any]:
    jsonl, db_path = _common_paths(args, spec_path)
    ledger = Ledger(db_path)
    logger = StructuredLogger(jsonl, ledger)
    spec = load_spec(spec_path)
    return spec, ledger, logger


def _emit(value: Any, *, output: str | None = None) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    sys.stdout.write(text)
    if output:
        JsonlReport.write(Path(output), value)


def _error(exc: ShardRepairError) -> int:
    sys.stderr.write(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return 2


def cmd_init_sample(args: argparse.Namespace) -> int:
    path = generate_sample(args.root, force=not args.keep)
    _emit({"code": "E_SAMPLE_OK", "spec": str(path)})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    jsonl, db_path = _common_paths(args, spec_path)
    ledger = Ledger(db_path)
    logger = StructuredLogger(jsonl, ledger)
    try:
        if args.anomaly == "cycle":
            raw = json.loads(spec_path.read_text(encoding="utf-8-sig"))
            raw.setdefault("relations", []).append({"kind": "mirror", "source": "s7", "target": "s1"})
            spec = load_spec(spec_path, raw_override=raw)
        else:
            spec = load_spec(spec_path)
        registry = ReplicaRegistry.load(spec, logger=logger)
    except SpecValidationError as exc:
        cause = exc.cause_code or exc.details.get("cause_code")
        if cause == "E_CYCLE":
            version = raw.get("group_version") if "raw" in locals() else None
            logger.emit("E_CYCLE", shard_id=exc.shard_id, version=version, field_path=exc.field_path)
        _emit(
            {
                "code": exc.code,
                "business_code": cause or exc.code,
                "error": exc.to_dict(),
                "integrity_issues": [],
            }
        )
        return 2
    result = {
        "code": "E_SPEC_OK",
        "group_id": spec.group_id,
        "group_version": spec.group_version,
        "mirror_graph_version": spec.mirror_graph_version,
        "node_count": len(spec.nodes),
        "logical_shard_count": len(spec.logical_shards),
        "physical_replica_count": len(registry.records),
        "topological_order": list(spec.topo_order),
        "integrity_issues": registry.integrity_issues,
        "registry": registry.summary(),
    }
    _emit(result)
    return 3 if registry.integrity_issues else 0


def cmd_inject(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec, _ledger, logger = _runtime(args, spec_path)
    registry = ReplicaRegistry.load(spec, logger=logger)
    try:
        record = registry.inject(
            shard_id=args.shard_id,
            node_id=args.node_id,
            kind=args.kind,
            offset=args.offset,
            keep_bytes=args.keep_bytes,
            replacement_hex=args.replacement_hex,
            sequence=args.sequence,
        )
    except (KeyError, ValueError) as exc:
        _emit({"code": "E_INJECT", "message": str(exc)})
        return 2
    _emit(
        {
            "code": "E_INTEGRITY",
            "injection": record.as_dict(),
            "registry": registry.summary(),
            "integrity_issues": registry.integrity_issues,
        },
        output=args.output,
    )
    return 0


def _parse_injections(values: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            if "=" not in value:
                raise ValueError(f"--inject[{index}] must be JSON or SHARD_ID=KIND")
            shard_id, kind = value.rsplit("=", 1)
            parsed = {"shard_id": shard_id, "kind": kind}
        if not isinstance(parsed, dict) or "shard_id" not in parsed or "kind" not in parsed:
            raise ValueError(f"--inject[{index}] must contain shard_id and kind")
        parsed.setdefault("sequence", index)
        result.append(parsed)
    return result


def cmd_run(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec, ledger, logger = _runtime(args, spec_path)
    registry = ReplicaRegistry.load(spec, logger=logger)
    try:
        injections = _parse_injections(args.inject)
    except ValueError as exc:
        _emit({"code": "E_ARGS", "message": str(exc)})
        return 2
    applied = []
    if args.resume:
        # The damaged bytes are already on disk. Resume derives the same seeds
        # without mutating the shards a second time.
        seeds = sorted(
            {
                spec.shard_by_id[value["shard_id"]].logical_id
                for value in injections
                if value["shard_id"] in spec.shard_by_id
            }
        )
    else:
        for index, value in enumerate(injections):
            applied.append(
                registry.inject(
                    shard_id=value["shard_id"],
                    node_id=value.get("node_id"),
                    kind=value["kind"],
                    offset=int(value.get("offset", 0)),
                    keep_bytes=value.get("keep_bytes"),
                    replacement_hex=value.get("replacement_hex"),
                    sequence=index,
                )
            )
        seeds = [record.logical_id for record in applied]
    propagation = propagate_invalidation(spec, seeds)
    registry.mark_affected(propagation.affected)
    quotas = {node.node_id: node.pack_quota for node in spec.nodes}
    tasks = build_pack_tasks(
        spec,
        spec.topo_order if args.full else propagation.affected,
        timeout_ms=args.timeout_ms,
        simulate_ms=args.simulate_ms,
        max_retries=args.max_retries,
        quotas=quotas,
    )
    recovery = None
    if args.resume:
        recovery = ledger.recover(args.run_id, logger=logger)
    store = PackStore(Path(args.pack_dir).resolve(), spec)
    schedule = PackScheduler(
        store=store,
        global_slots=args.global_slots,
        node_quotas=quotas,
        logger=logger,
        ledger=ledger,
        run_id=args.run_id,
        crash_after_commits=args.crash_after_commits,
    ).run(tasks, resume_completed=args.resume)
    repair_report = None
    if args.repair:
        repair_report = apply_and_report(spec, registry, logger=logger)
        registry.clear_pending()
    result = {
        "code": (
            "E_RUN_OK"
            if all(entry.outcome in {"success", "recovered"} for entry in schedule.entries)
            else "E_RUN_INCOMPLETE"
        ),
        "run_id": args.run_id,
        "injections": [record.as_dict() for record in applied],
        "resume_seeds": list(seeds) if args.resume else [],
        "propagation": propagation.as_dict(),
        "tasks": [task.as_dict() for task in tasks],
        "schedule": schedule.as_dict(),
        "repair_report": repair_report,
        "registry": registry.summary(),
        "recovery": recovery,
        "pack_manifest": store.manifest(),
    }
    _emit(result, output=args.output)
    return 0 if result["code"] == "E_RUN_OK" else 1


def cmd_verify(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec = load_spec(spec_path)
    registry = ReplicaRegistry.load(spec)
    incremental = PackStore(Path(args.incremental_pack).resolve(), spec)
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    full = PackStore(work_dir, spec)
    full.repack_all_direct()
    digest = block_replica_digest(spec, registry)
    try:
        comparison = compare_artifact_sets(
            incremental.manifest(),
            full.manifest(),
            incremental_registry_digest=digest,
            full_registry_digest=digest,
        )
    except ShardRepairError as exc:
        if exc.code != "E_INCR":
            raise
        _emit({"code": "E_INCR", "error": exc.to_dict()})
        return 2
    result = {
        "code": "E_INCR_OK",
        "comparison": comparison,
        "block_replica_digest": digest,
    }
    _emit(result, output=args.output)
    return 0


def cmd_snapshot_restore(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec, _ledger, logger = _runtime(args, spec_path)
    registry = ReplicaRegistry.load(spec, logger=logger)
    manager = SnapshotManager(Path(args.snapshot).resolve())
    try:
        result = manager.restore(
            spec=spec,
            registry=registry,
            state_path=Path(args.state).resolve(),
            logger=logger,
        )
    except SnapshotError as exc:
        logger.emit("E_SNAPSHOT", shard_id=exc.shard_id, version=exc.version, error=exc.message)
        return _error(exc)
    _emit(result, output=args.output)
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    ledger = Ledger(db_path)
    jsonl = Path(args.log_dir).resolve() / "events.jsonl"
    logger = StructuredLogger(jsonl, ledger)
    report = ledger.recover(args.run_id, logger=logger)
    _emit(report, output=args.output)
    return 0


def cmd_anomaly(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    jsonl, db_path = _common_paths(args, spec_path)
    ledger = Ledger(db_path)
    logger = StructuredLogger(jsonl, ledger)
    if args.kind == "cycle":
        try:
            raw = json.loads(spec_path.read_text(encoding="utf-8-sig"))
            raw.setdefault("relations", []).append({"kind": "mirror", "source": "s7", "target": "s1"})
            load_spec(spec_path, raw_override=raw)
        except SpecValidationError as exc:
            logger.emit(
                "E_CYCLE",
                shard_id=exc.shard_id,
                version=raw.get("group_version"),
                field_path=exc.field_path,
                message=exc.message,
            )
            _emit({"code": "E_CYCLE", "spec_error": exc.to_dict()})
            return 2
        _emit({"code": "E_NO_ANOMALY"})
        return 1
    if args.kind == "integrity":
        spec = load_spec(spec_path)
        registry = ReplicaRegistry.load(spec, logger=logger)
        shard_id = args.shard_id or sorted(spec.shard_by_id)[0]
        record = registry.inject(shard_id=shard_id, kind="flip", sequence=0)
        error = IntegrityError(
            "injected content digest mismatch",
            shard_id=shard_id,
            field_path="sha256",
            version=spec.group_version,
            details={"injection": record.as_dict(), "issues": registry.integrity_issues},
        )
        _emit({"code": "E_INTEGRITY", "error": error.to_dict(), "injection": record.as_dict()})
        return 2
    if args.kind == "snapshot":
        if not args.snapshot:
            _emit({"code": "E_ARGS", "message": "--snapshot is required for snapshot anomaly"})
            return 2
        source = Path(args.snapshot).resolve()
        target = source.with_name(source.stem + ".tampered.json")
        tamper_snapshot_copy(source, target)
        manager = SnapshotManager(target)
        try:
            manager.load(verify=True)
        except SnapshotError as exc:
            logger.emit("E_SNAPSHOT", shard_id=exc.shard_id, version=exc.version, error=exc.message)
            _emit({"code": "E_SNAPSHOT", "error": exc.to_dict(), "tampered_copy": str(target)})
            return 2
        _emit({"code": "E_NO_ANOMALY"})
        return 1
    _emit({"code": "E_ARGS", "message": "unknown anomaly kind"})
    return 2


def cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(
        args.root,
        simulate_ms=args.simulate_ms,
        timeout_ms=args.timeout_ms,
        max_retries=args.max_retries,
    )
    summary = {
        "code": result["code"],
        "stable_digest": result["stable_digest"],
        "stage1_repair_counts": result["stage1_full_repair"]["repair_counts"],
        "stage2_affected": result["stage2_incremental"]["affected"],
        "stage2_packed_shards": result["stage2_incremental"]["packed_shards"],
        "stage2_repair_counts": result["stage2_incremental"]["repair_counts"],
        "incremental_vs_full_ok": result["stage2_incremental"]["incremental_vs_full"]["ok"],
        "snapshot_restore_replaced": result["snapshot_restore"]["restore"]["replaced"],
        "replay_ok": result["replay"]["ok"],
        "report": str(Path(args.root).resolve() / "demo-run" / "demo_report.json"),
        "log": result["artifacts"]["log"],
        "sqlite": result["artifacts"]["sqlite"],
    }
    _emit(summary, output=args.output)
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    launch_gui(args.spec or "samples/spec.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shardrep", description="Local shard replica repair middleware")
    sub = parser.add_subparsers(dest="command", required=True)

    init_sample = sub.add_parser("init-sample", help="generate the deterministic local sample")
    init_sample.add_argument("--root", default=".")
    init_sample.add_argument("--keep", action="store_true", help="keep an existing sample")
    init_sample.set_defaults(func=cmd_init_sample)

    validate = sub.add_parser("validate", help="strictly validate a shard group spec")
    validate.add_argument("--spec", default="samples/spec.json")
    validate.add_argument("--anomaly", choices=["none", "cycle"], default="none")
    validate.add_argument("--log-dir")
    validate.add_argument("--db")
    validate.set_defaults(func=cmd_validate)

    inject = sub.add_parser("inject", help="inject file damage by physical shard id")
    inject.add_argument("--spec", default="samples/spec.json")
    inject.add_argument("--shard-id", required=True)
    inject.add_argument("--node-id")
    inject.add_argument("--kind", required=True, choices=["truncate", "flip", "replace", "delete"])
    inject.add_argument("--offset", type=int, default=0)
    inject.add_argument("--keep-bytes", type=int)
    inject.add_argument("--replacement-hex")
    inject.add_argument("--sequence", type=int, default=0)
    inject.add_argument("--output")
    inject.add_argument("--log-dir")
    inject.add_argument("--db")
    inject.set_defaults(func=cmd_inject)

    run = sub.add_parser("run", help="propagate invalidation and run budgeted repacking")
    run.add_argument("--spec", default="samples/spec.json")
    run.add_argument("--pack-dir", default="run/packs")
    run.add_argument("--global-slots", type=int, default=2)
    run.add_argument("--timeout-ms", type=int, default=1500)
    run.add_argument("--simulate-ms", type=int, default=0)
    run.add_argument("--max-retries", type=int, default=1)
    run.add_argument("--inject", action="append", default=[], metavar="SHARD_ID=KIND|JSON")
    run.add_argument("--full", action="store_true", help="repack all shards instead of the affected set")
    run.add_argument("--repair", action="store_true")
    run.add_argument("--run-id", default="run-0001")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--crash-after-commits", type=int)
    run.add_argument("--output")
    run.add_argument("--log-dir")
    run.add_argument("--db")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="compare incremental artifacts with a full rebuild")
    verify.add_argument("--spec", default="samples/spec.json")
    verify.add_argument("--incremental-pack", default="demo-run/packs")
    verify.add_argument("--work-dir", default="run/verify-full")
    verify.add_argument("--output")
    verify.set_defaults(func=cmd_verify)

    restore = sub.add_parser("snapshot-restore", help="restore a snapshot as a whole state replacement")
    restore.add_argument("--spec", default="samples/spec.json")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--state", required=True)
    restore.add_argument("--output")
    restore.add_argument("--log-dir")
    restore.add_argument("--db")
    restore.set_defaults(func=cmd_snapshot_restore)

    recover = sub.add_parser("recover", help="recover only committed SQLite task records")
    recover.add_argument("--db", required=True)
    recover.add_argument("--run-id", required=True)
    recover.add_argument("--log-dir", default="run/logs")
    recover.add_argument("--output")
    recover.set_defaults(func=cmd_recover)

    anomaly = sub.add_parser("anomaly", help="exercise cycle, integrity or snapshot anomaly paths")
    anomaly.add_argument("--kind", required=True, choices=["cycle", "integrity", "snapshot"])
    anomaly.add_argument("--spec", default="samples/spec.json")
    anomaly.add_argument("--snapshot")
    anomaly.add_argument("--shard-id")
    anomaly.add_argument("--log-dir")
    anomaly.add_argument("--db")
    anomaly.set_defaults(func=cmd_anomaly)

    demo = sub.add_parser("demo", help="run the complete repeatable demonstration")
    demo.add_argument("--root", default=".")
    demo.add_argument("--simulate-ms", type=int, default=12)
    demo.add_argument("--timeout-ms", type=int, default=1500)
    demo.add_argument("--max-retries", type=int, default=1)
    demo.add_argument("--output")
    demo.set_defaults(func=cmd_demo)

    gui = sub.add_parser("gui", help="open the local Tkinter diagnostic interface")
    gui.add_argument("--spec", default="samples/spec.json")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ShardRepairError as exc:
        return _error(exc)
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(json.dumps({"code": "E_RUNTIME", "message": str(exc)}, ensure_ascii=False) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
