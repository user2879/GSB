from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import ReconError
from .incremental import (
    authoritative_conclusion,
    edit_document,
    edit_shard_size,
    edit_threshold,
    edit_weight,
    initialize_fast_view,
    read_fast_once,
    verify_fast_matches_authoritative,
)
from .ledger import Ledger
from .repository import (
    open_initialized_ledger,
    persist_current_view,
    process_pending_batches,
    record_fast_read,
    replay_ledger,
)
from .scheduler import recompute_shards_tcp
from .specs import load_bundle
from .synthetic import generate_workspace
from .workspace import Workspace


def print_json(payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))


def error_exit(exc: ReconError) -> int:
    payload = exc.to_dict()
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False), file=sys.stderr)
    codes = {"E_SPEC": 2, "E_DIVERGE": 3, "E_REPLAY": 4, "E_PORT": 5, "E_TIMEOUT": 6, "E_FIELD": 7}
    return codes.get(exc.code, 1)


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace(args.root)


def cmd_generate(args: argparse.Namespace) -> int:
    generate_workspace(args.root)
    print_json({"generated": True, "root": str(Path(args.root).resolve())})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.root)
    print_json({"valid": True, "domains": bundle.domain_ids, "batches": len(bundle.batches), "docs": sum(len(b["docs"]) for b in bundle.batches)})
    return 0


def cmd_init_cache(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    view = initialize_fast_view(ws, use_tcp=args.tcp)
    print_json({"initialized": True, "config_version": view["config_version"], "digest": view["cache_digest"], "tcp": args.tcp})
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    result = read_fast_once(ws)
    summary = {
        "config_version": result["config_version"],
        "cache_digest": result["cache_digest"],
        "elapsed_ms": result["elapsed_ms"],
        "bytes_read": result["bytes_read"],
        "shards": [
            {
                "shard_id": shard["shard_id"],
                "doc_count": shard["doc_count"],
                "kept_count": shard["kept_count"],
                "dropped_count": shard["dropped_count"],
                "domains": shard["domains"],
            }
            for shard in result["view"]["conclusion"]["shards"]
        ],
        "ratio": result["view"]["conclusion"]["ratio_normalization"]["domains"],
        "last_invalidation": result["view"].get("last_invalidation"),
    }
    if args.record:
        ledger = open_initialized_ledger(ws)
        read_record = record_fast_read(ws, ledger)
        summary["ledger_read_id"] = read_record["read_id"]
    print_json(summary)
    return 0


def cmd_authoritative(args: argparse.Namespace) -> int:
    conclusion = authoritative_conclusion(_workspace(args))
    print_json({"digest": conclusion["conclusion_digest"], "path": str(_workspace(args).authoritative_path)})
    return 0


def cmd_verify_view(args: argparse.Namespace) -> int:
    print_json(verify_fast_matches_authoritative(_workspace(args)))
    return 0


def cmd_edit_weight(args: argparse.Namespace) -> int:
    propagation = edit_weight(_workspace(args), args.domain, args.weight, use_tcp=args.tcp)
    print_json(_propagation_summary(propagation))
    return 0


def cmd_edit_threshold(args: argparse.Namespace) -> int:
    propagation = edit_threshold(_workspace(args), args.threshold, use_tcp=args.tcp)
    print_json(_propagation_summary(propagation))
    return 0


def cmd_edit_shard_size(args: argparse.Namespace) -> int:
    propagation = edit_shard_size(_workspace(args), args.shard_size, use_tcp=args.tcp)
    print_json(_propagation_summary(propagation))
    return 0


def cmd_edit_doc(args: argparse.Namespace) -> int:
    counts = json.loads(args.term_counts) if args.term_counts else None
    bitmap = json.loads(args.missing_bitmap) if args.missing_bitmap else None
    propagation = edit_document(_workspace(args), args.doc_id, term_counts=counts, missing_bitmap=bitmap, use_tcp=args.tcp)
    print_json(_propagation_summary(propagation))
    return 0


def _propagation_summary(propagation: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_version": propagation["config_version"],
        "affected_domains": propagation["affected_domains"],
        "affected_shards": propagation["affected_shards"],
        "removed_shards": propagation["removed_shards"],
        "invalidated_shards": propagation.get("invalidated_shards", propagation["affected_shards"]),
        "affected_ratio_domains": propagation["affected_ratio_domains"],
        "preserved_shards": propagation["preserved_shards"],
        "preserved_ratio_domains": propagation["preserved_ratio_domains"],
        "step_count": len(propagation["steps"]),
        "propagation_digest": propagation.get("propagation_digest"),
    }


def cmd_ledger_init(args: argparse.Namespace) -> int:
    ledger = open_initialized_ledger(_workspace(args))
    print_json({"ledger": str(ledger.path), "user_version": ledger.user_version()})
    return 0


def cmd_process_batches(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    ledger = open_initialized_ledger(ws)
    processed = process_pending_batches(ws, ledger, crash_batch_id=args.crash_batch, marker_path=Path(args.marker) if args.marker else None)
    print_json({"processed": processed, "pending": len(ledger.pending_batches())})
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    ledger = Ledger(_workspace(args).ledger_path)
    unknown = ledger.recover_unknown()
    print_json({"unknown_batches": unknown})
    return 0


def cmd_persist_view(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    ledger = open_initialized_ledger(ws)
    print_json(persist_current_view(ws, ledger))
    return 0


def cmd_record_read(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    ledger = open_initialized_ledger(ws)
    record = record_fast_read(ws, ledger)
    print_json({"read_id": record["read_id"], "elapsed_ms": record["elapsed_ms"], "bytes_read": record["bytes_read"]})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    ledger = Ledger(ws.ledger_path)
    print_json(replay_ledger(ws, ledger, config_version=args.config_version))
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    ledger = Ledger(ws.ledger_path)
    version = args.config_version or ledger.latest_config_version_with_items() or ws.load_config()["config_version"]
    metadata = ledger.create_snapshot(ws.snapshot_dir, int(version))
    print_json(metadata)
    return 0


def cmd_snapshots(args: argparse.Namespace) -> int:
    ledger = Ledger(_workspace(args).ledger_path)
    print_json({"snapshots": ledger.list_snapshots()})
    return 0


def cmd_tcp_recompute(args: argparse.Namespace) -> int:
    bundle = _workspace(args).load_bundle()
    result = recompute_shards_tcp(bundle)
    scheduling = result["scheduling"]
    print_json(
        {
            "shards": len(result["shard_results"]),
            "tasks": scheduling["task_count"],
            "peak_global_points": scheduling["peak_global_points"],
            "peak_domain_points": scheduling["peak_domain_points"],
            "wait_records": len(scheduling["wait_records"]),
            "invariant_violations": scheduling["invariant_violations"],
        }
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import run_tk_app

    return run_tk_app(_workspace(args), smoke_ms=args.smoke_ms)


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    report = run_demo(Path(args.root), keep_server=not args.no_keep_server)
    print_json(report)
    return 0 if report.get("success") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reconflow", description="Deterministic stream reconciliation demo")
    parser.add_argument("--root", default="artifacts/demo", help="workspace root")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate").set_defaults(func=cmd_generate)
    sub.add_parser("validate").set_defaults(func=cmd_validate)

    p = sub.add_parser("init-cache")
    p.add_argument("--tcp", action="store_true")
    p.set_defaults(func=cmd_init_cache)

    p = sub.add_parser("view")
    p.add_argument("--record", action="store_true")
    p.set_defaults(func=cmd_view)

    sub.add_parser("authoritative").set_defaults(func=cmd_authoritative)
    sub.add_parser("verify-view").set_defaults(func=cmd_verify_view)

    p = sub.add_parser("edit-weight")
    p.add_argument("domain")
    p.add_argument("weight", type=int)
    p.add_argument("--tcp", action="store_true")
    p.set_defaults(func=cmd_edit_weight)

    p = sub.add_parser("edit-threshold")
    p.add_argument("threshold", type=int)
    p.add_argument("--tcp", action="store_true")
    p.set_defaults(func=cmd_edit_threshold)

    p = sub.add_parser("edit-shard-size")
    p.add_argument("shard_size", type=int)
    p.add_argument("--tcp", action="store_true")
    p.set_defaults(func=cmd_edit_shard_size)

    p = sub.add_parser("edit-doc")
    p.add_argument("doc_id")
    p.add_argument("--term-counts")
    p.add_argument("--missing-bitmap")
    p.add_argument("--tcp", action="store_true")
    p.set_defaults(func=cmd_edit_doc)

    sub.add_parser("ledger-init").set_defaults(func=cmd_ledger_init)

    p = sub.add_parser("process-batches")
    p.add_argument("--crash-batch")
    p.add_argument("--marker")
    p.set_defaults(func=cmd_process_batches)

    sub.add_parser("recover").set_defaults(func=cmd_recover)
    sub.add_parser("persist-view").set_defaults(func=cmd_persist_view)
    sub.add_parser("record-read").set_defaults(func=cmd_record_read)

    p = sub.add_parser("replay")
    p.add_argument("--config-version", type=int)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("snapshot")
    p.add_argument("--config-version", type=int)
    p.set_defaults(func=cmd_snapshot)
    sub.add_parser("snapshots").set_defaults(func=cmd_snapshots)
    sub.add_parser("tcp-recompute").set_defaults(func=cmd_tcp_recompute)

    p = sub.add_parser("ui")
    p.add_argument("--smoke-ms", type=int, default=0)
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("demo")
    p.add_argument("--no-keep-server", action="store_true")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--root" in raw_args[1:]:
        index = raw_args.index("--root", 1)
        if index + 1 < len(raw_args):
            root_value = raw_args.pop(index + 1)
            raw_args.pop(index)
            raw_args = ["--root", root_value] + raw_args
    for index, arg in enumerate(list(raw_args)):
        if arg.startswith("--root=") and index > 0:
            root_value = arg.split("=", 1)[1]
            raw_args.pop(index)
            raw_args = ["--root", root_value] + raw_args
            break
    parser = build_parser()
    args = parser.parse_args(raw_args)
    try:
        return int(args.func(args))
    except ReconError as exc:
        return error_exit(exc)
    except FileNotFoundError as exc:
        print(json.dumps({"code": "E_SPEC", "message": f"file not found: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
