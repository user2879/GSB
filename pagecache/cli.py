from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .aggregate import FixedPointAggregator
from .cache import PageCache
from .canonical import canonical_bytes, sha256_hex
from .errors import EngineError, ErrorCode, error_json, write_structured_log
from .injection import corrupt_segment_checksum_copy, trigger_zero_budget, truncate_manifest_copy
from .manifest import read_manifest
from .model import Manifest
from .pagefile import read_page_file
from .replay import replay_from_snapshot
from .segment import read_segment_file
from .session import run_prefix, run_reference
from .snapshot import load_snapshot, make_snapshot, restore_snapshot, save_snapshot
from .upgrade import SimulatedUpgradeCrash, upgrade_manifest
from .verifier import verify_artifacts
from samples.generate import generate_samples


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = ROOT / "samples" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pagecache", description="Deterministic page cache, spill, aggregation, snapshot and verification tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--out", default=str(SAMPLE_DATA))
    generate.add_argument("--seed", type=int, default=20261118)

    upgrade = subparsers.add_parser("upgrade", help="upgrade a manifest to v3 with resumable checkpoints")
    upgrade.add_argument("--manifest", required=True)
    upgrade.add_argument("--pages-dir", required=True)
    upgrade.add_argument("--overflow-dir", required=True)
    upgrade.add_argument("--crash-after-checkpoint", type=int, default=None)
    upgrade.add_argument("--log", default=None)

    run = subparsers.add_parser("run", help="scan pages, spill deterministically, and aggregate")
    _add_engine_arguments(run)
    run.add_argument("--order", default=None, help="comma-separated page IDs; defaults to samples/data metadata")
    run.add_argument("--group-column", default="group")
    run.add_argument("--value-column", default="value")
    run.add_argument("--output", required=True)
    run.add_argument("--events", default=None)
    run.add_argument("--snapshot", default=None)
    run.add_argument("--snapshot-at", type=int, default=None)
    run.add_argument("--log", default=None)

    replay = subparsers.add_parser("replay", help="restore a snapshot and replay its reference trace")
    _add_engine_arguments(replay)
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--group-column", default="group")
    replay.add_argument("--value-column", default="value")
    replay.add_argument("--log", default=None)

    restore = subparsers.add_parser("restore", help="wholesale-replace cache and aggregation state from a snapshot")
    _add_engine_arguments(restore)
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--group-column", default="group")
    restore.add_argument("--value-column", default="value")
    restore.add_argument("--log", default=None)

    verify = subparsers.add_parser("verify", help="independently validate snapshot, page state, output and error bounds")
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--pages-dir", required=True)
    verify.add_argument("--overflow-dir", required=True)
    verify.add_argument("--report", default=None)
    verify.add_argument("--log", default=None)

    inject = subparsers.add_parser("inject", help="run one of the three required anomaly injections")
    inject.add_argument("--kind", required=True, choices=("manifest_tail_truncate", "segment_checksum_mismatch", "memory_budget_zero"))
    inject.add_argument("--manifest", required=True)
    inject.add_argument("--pages-dir", required=True)
    inject.add_argument("--overflow-dir", required=True)
    inject.add_argument("--page-size", type=int, default=512)
    inject.add_argument("--work-dir", required=True)
    inject.add_argument("--log", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_samples(args.out, seed=args.seed)
        elif args.command == "upgrade":
            result = _command_upgrade(args)
        elif args.command == "run":
            result = _command_run(args)
        elif args.command == "replay":
            result = _command_replay(args)
        elif args.command == "restore":
            result = _command_restore(args)
        elif args.command == "verify":
            result = _command_verify(args)
        elif args.command == "inject":
            result = _command_inject(args)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result.get("status") != "FAIL" else 1
    except SimulatedUpgradeCrash as exc:
        payload = {
            "status": "interrupted",
            "event": "simulated_crash_after_checkpoint",
            "completed_records": exc.completed_records,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 75
    except EngineError as exc:
        if getattr(args, "log", None):
            write_structured_log(args.log, args.command, exc)
        print(error_json(exc))
        return 2
    except (OSError, ValueError) as exc:
        error = EngineError(ErrorCode.E_FORMAT, str(exc), path="$cli")
        if getattr(args, "log", None):
            write_structured_log(args.log, args.command, error)
        print(error_json(error))
        return 2


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pages-dir", required=True)
    parser.add_argument("--overflow-dir", required=True)
    parser.add_argument("--budget-bytes", required=True, type=int)
    parser.add_argument("--page-size", type=int, default=512)


def _command_upgrade(args: argparse.Namespace) -> dict[str, Any]:
    result = upgrade_manifest(
        args.manifest,
        pages_dir=args.pages_dir,
        overflow_dir=args.overflow_dir,
        crash_after_checkpoint=args.crash_after_checkpoint,
    )
    payload = {"status": "ok", **result.to_dict()}
    if args.log:
        write_structured_log(args.log, "upgrade", None, result=payload)
    return payload


def _command_run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_manifest(args.manifest)
    order = _load_order(args.order)
    aggregator = FixedPointAggregator(args.group_column, args.value_column)

    snapshot_position: int | None = None
    if args.snapshot:
        reference_cache = PageCache(
            manifest=manifest,
            pages_dir=args.pages_dir,
            overflow_dir=args.overflow_dir,
            budget_bytes=args.budget_bytes,
            page_size=args.page_size,
        )
        reference_aggregator = FixedPointAggregator(args.group_column, args.value_column)
        reference = run_reference(cache=reference_cache, aggregator=reference_aggregator, scan_order=order)
        position = args.snapshot_at if args.snapshot_at is not None else len(order) // 2
        snapshot_position = position
        cache = PageCache(
            manifest=manifest,
            pages_dir=args.pages_dir,
            overflow_dir=args.overflow_dir,
            budget_bytes=args.budget_bytes,
            page_size=args.page_size,
        )
        aggregator = FixedPointAggregator(args.group_column, args.value_column)
        run_prefix(cache=cache, aggregator=aggregator, scan_order=order, position=position)
        snapshot = make_snapshot(
            cache=cache,
            aggregator=aggregator,
            scan_order=order,
            position=position,
            reference_trace=reference["trace"],
            reference_final_output=reference["final_output"],
        )
        save_snapshot(args.snapshot, snapshot)
        for page_id in order[position:]:
            aggregator.consume_page(cache.get_page(page_id))
    else:
        cache = PageCache(
            manifest=manifest,
            pages_dir=args.pages_dir,
            overflow_dir=args.overflow_dir,
            budget_bytes=args.budget_bytes,
            page_size=args.page_size,
        )
        reference = run_reference(cache=cache, aggregator=aggregator, scan_order=order)
    output_bytes = aggregator.write_output(args.output)
    output = aggregator.output()
    cache_state = cache.state()
    if args.events:
        Path(args.events).parent.mkdir(parents=True, exist_ok=True)
        Path(args.events).write_bytes(canonical_bytes(cache_state["events"]))
    result = {
        "status": "ok",
        "budget_bytes": args.budget_bytes,
        "page_size": args.page_size,
        "capacity_frames": cache.capacity,
        "scan_order": order,
        "overflow_sequence": cache_state["overflow_sequence"],
        "swap_in_sequence": cache_state["swap_in_sequence"],
        "swap_in_count": cache_state["swap_in_count"],
        "eviction_count": cache_state["eviction_count"],
        "page_load_count": cache_state["page_load_count"],
        "resident_pages": [row["page_id"] for row in cache_state["resident_pages"]],
        "overflow_pages": [row["page_id"] for row in cache_state["overflow_pages"]],
        "output_path": str(Path(args.output)),
        "output_sha256": sha256_hex(output_bytes),
        "output": output,
        "snapshot_path": args.snapshot,
        "snapshot_at": snapshot_position,
    }
    if args.snapshot:
        result["reference_output_sha256"] = reference["final_output_sha256"]
        result["matches_reference"] = output_bytes == canonical_bytes(reference["final_output"])
    if args.log:
        write_structured_log(args.log, "run", None, result={key: value for key, value in result.items() if key != "output"})
    return result


def _command_replay(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_manifest(args.manifest)
    cache = PageCache(
        manifest=manifest,
        pages_dir=args.pages_dir,
        overflow_dir=args.overflow_dir,
        budget_bytes=args.budget_bytes,
        page_size=args.page_size,
    )
    aggregator = FixedPointAggregator(args.group_column, args.value_column)
    result = replay_from_snapshot(args.snapshot, cache=cache, aggregator=aggregator)
    result = {"status": "ok", **result}
    if args.log:
        write_structured_log(args.log, "replay", None, result=result)
    return result


def _command_restore(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_manifest(args.manifest)
    cache = PageCache(
        manifest=manifest,
        pages_dir=args.pages_dir,
        overflow_dir=args.overflow_dir,
        budget_bytes=args.budget_bytes,
        page_size=args.page_size,
    )
    aggregator = FixedPointAggregator(args.group_column, args.value_column)
    snapshot = load_snapshot(args.snapshot)
    report = restore_snapshot(snapshot, cache=cache, aggregator=aggregator, exact_disk=False)
    cache_state = cache.state()
    result = {
        "status": "ok",
        **report,
        "resident_pages": [row["page_id"] for row in cache_state["resident_pages"]],
        "overflow_pages": [row["page_id"] for row in cache_state["overflow_pages"]],
        "aggregation_groups": aggregator.state()["groups"],
    }
    if args.log:
        write_structured_log(args.log, "restore", None, result=result)
    return result


def _command_verify(args: argparse.Namespace) -> dict[str, Any]:
    report = verify_artifacts(
        snapshot_path=args.snapshot,
        output_path=args.output,
        manifest_path=args.manifest,
        pages_dir=args.pages_dir,
        overflow_dir=args.overflow_dir,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_bytes(canonical_bytes(report))
    if args.log:
        write_structured_log(args.log, "verify", None if report["status"] == "PASS" else EngineError(ErrorCode.E_REPLAY, "verification failed", path="$verify"), report=report)
    return report


def _command_inject(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.kind == "manifest_tail_truncate":
            target = truncate_manifest_copy(args.manifest, work_dir / "truncated.manifest")
            read_manifest(target)
        elif args.kind == "segment_checksum_mismatch":
            target = corrupt_segment_checksum_copy(args.overflow_dir, work_dir / "corrupt.seg")
            read_segment_file(target)
        elif args.kind == "memory_budget_zero":
            trigger_zero_budget(
                manifest_path=args.manifest,
                pages_dir=args.pages_dir,
                overflow_dir=args.overflow_dir,
                page_size=args.page_size,
            )
        else:
            raise AssertionError(args.kind)
    except EngineError as exc:
        expected = {
            "manifest_tail_truncate": ErrorCode.E_FORMAT,
            "segment_checksum_mismatch": ErrorCode.E_INTEGRITY,
            "memory_budget_zero": ErrorCode.E_BUDGET,
        }[args.kind]
        write_structured_log(args.log, "injection", exc, injection=args.kind)
        if exc.code is not expected:
            raise
        return {
            "status": "reported",
            "injection": args.kind,
            "expected_code": expected.value,
            "actual_code": exc.code.value,
            "error": exc.to_dict(),
            "log_path": str(Path(args.log)),
        }
    raise EngineError(ErrorCode.E_FORMAT, "injection did not trigger the expected failure", path="$injection")


def _load_order(value: str | None) -> list[str]:
    if value:
        order = [item.strip() for item in value.split(",") if item.strip()]
    else:
        meta_path = SAMPLE_DATA / "sample_meta.json"
        if not meta_path.exists():
            raise EngineError(ErrorCode.E_FORMAT, "--order is required when sample metadata is absent", path="$cli.order")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        order = [str(item) for item in meta["scan_order"]]
    if not order:
        raise EngineError(ErrorCode.E_FORMAT, "scan order is empty", path="$cli.order")
    return order


if __name__ == "__main__":
    raise SystemExit(main())
