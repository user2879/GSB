"""Command line entry point for validation, backfill and report comparison."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

from .config_loader import (
    feature_fingerprint,
    load_feature_file,
    load_plan,
    parse_config,
    plan_fingerprint,
    validate_feature_graph,
)
from .data_loader import dataset_fingerprint, load_samples, rows_by_entity
from .engine import (
    BackfillEngine,
    RunOptions,
    compare_reports,
    execution_comparison,
)
from .errors import (
    AppError,
    E_CONFIG_INVALID,
    E_FEATURE_DAG_INVALID,
    E_PLAN_INVALID,
    sorted_failures,
)
from .fingerprints import canonical_json
from .injections import load_injections
from .clock import MonotonicClock
from .leases import LeaseManager
from .models import AppConfig, FeatureDef
from .plan import materialize_plan_state


def _load_all(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "config": None,
        "config_failures": [],
        "features": {},
        "raw_fields": [],
        "feature_failures": [],
        "rows": [],
        "row_failures": [],
        "plan": [],
        "plan_failures": [],
        "injections": [],
        "injection_failures": [],
    }
    try:
        config = parse_config(args.config)
    except AppError as exc:
        result["config_failures"] = [exc.to_failure()]
        return result
    result["config"] = config
    features, declared_fields, feature_failures = load_feature_file(args.features)
    result["features"] = features
    result["feature_failures"] = feature_failures
    raw_fields = set(declared_fields)
    if not raw_fields:
        raw_fields.update(
            feature.source_field
            for feature in features.values()
            if feature.source_field is not None
        )
    result["raw_fields"] = sorted(raw_fields)
    rows, row_failures = load_samples(
        args.samples,
        config=config,
        raw_fields=sorted(raw_fields),
    )
    result["rows"] = rows
    result["row_failures"] = row_failures
    plan, plan_failures = load_plan(args.plan)
    result["plan"] = plan
    result["plan_failures"] = plan_failures
    injections_path = getattr(args, "injections", None)
    if injections_path and getattr(args, "enable_injections", True):
        injections, injection_failures = load_injections(injections_path)
    else:
        injections, injection_failures = [], []
    result["injections"] = injections
    result["injection_failures"] = injection_failures
    return result


def _all_failures(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted_failures(
        loaded["config_failures"]
        + loaded["feature_failures"]
        + loaded["row_failures"]
        + loaded["plan_failures"]
        + loaded["injection_failures"]
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def command_validate(args: argparse.Namespace) -> int:
    loaded = _load_all(args)
    failures = _all_failures(loaded)
    config: AppConfig | None = loaded["config"]
    features: dict[str, FeatureDef] = loaded["features"]
    order: list[str] = []
    if features:
        try:
            order = validate_feature_graph(
                features,
                raw_fields=set(loaded["raw_fields"]),
            )
        except AppError as exc:
            if exc.to_failure() not in failures:
                failures.append(exc.to_failure())
    report = {
        "status": "valid" if not failures else "invalid",
        "failures": sorted_failures(failures),
        "row_count": len(loaded["rows"]),
        "entity_count": len(rows_by_entity(loaded["rows"])),
        "feature_order": order,
        "config_hash": config.fingerprint() if config else None,
        "dataset_fingerprint": dataset_fingerprint(loaded["rows"]),
        "feature_fingerprint": feature_fingerprint(features) if features else None,
        "plan_hash": plan_fingerprint(loaded["plan"]),
        "injection_count": len(loaded["injections"]),
    }
    _print_json(report)
    if getattr(args, "report", None):
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(canonical_json(report) + "\n", encoding="utf-8")
    return 0 if not failures else 1


def command_run(args: argparse.Namespace) -> int:
    loaded = _load_all(args)
    failures = _all_failures(loaded)
    blocking_codes = {E_CONFIG_INVALID, E_FEATURE_DAG_INVALID, E_PLAN_INVALID}
    blocking = [failure for failure in failures if failure.get("code") in blocking_codes]
    if blocking or loaded["config"] is None:
        report = {
            "status": "rejected",
            "failures": failures,
            "blocking_codes": sorted({failure.get("code") for failure in blocking}),
        }
        _print_json(report)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(canonical_json(report) + "\n", encoding="utf-8")
        return 2
    engine = BackfillEngine(
        base_rows=loaded["rows"],
        base_features=loaded["features"],
        raw_fields=loaded["raw_fields"],
        config=loaded["config"],
        plan=loaded["plan"],
        base_failures=loaded["row_failures"],
        injections=loaded["injections"],
        snapshot_root=args.snapshot_dir or loaded["config"].snapshot_dir,
    )
    try:
        report = engine.run(
            RunOptions(
                resume=args.resume,
                crash_before_commit_batch=args.crash_before_commit_batch,
                report_path=args.report,
                enable_injections=not args.no_injections,
            )
        )
    except AppError as exc:
        report = {
            "status": "rejected",
            "failures": [exc.to_failure()],
            "snapshot_root": args.snapshot_dir or loaded["config"].snapshot_dir,
        }
        _print_json(report)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(canonical_json(report) + "\n", encoding="utf-8")
        return 2
    summary = {
        "status": report["status"],
        "run": report["run"],
        "entity_count": len(report["entities"]),
        "completed_entity_count": len(report["completed_entities"]),
        "pending": report["pending"],
        "failure_count": len(report["failures"]),
        "failure_codes": sorted({item.get("code") for item in report["failures"]}),
        "injection_hits": report["injection_hits"],
        "consumed_points_by_batch": report["consumed_points_by_batch"],
        "report_path": args.report,
    }
    _print_json(summary)
    if report["status"] == "completed":
        return 0
    return 3


def command_compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    result = compare_reports(left, right)
    if args.execution:
        result["execution_equal"] = execution_comparison(left) == execution_comparison(right)
    _print_json(result)
    if not result["equal"]:
        return 1
    if args.execution and not result.get("execution_equal", False):
        return 1
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    config = parse_config(args.config)
    features, declared_fields, feature_failures = load_feature_file(args.features)
    if feature_failures:
        _print_json({"status": "invalid", "failures": feature_failures})
        return 1
    raw_fields = set(declared_fields) or {
        feature.source_field
        for feature in features.values()
        if feature.source_field is not None
    }
    rows, row_failures = load_samples(args.samples, config=config, raw_fields=raw_fields)
    plan, plan_failures = load_plan(args.plan)
    if row_failures or plan_failures:
        _print_json(
            {"status": "invalid", "failures": row_failures + plan_failures}
        )
        return 1
    final_rows, final_features, failures = materialize_plan_state(
        rows,
        features,
        plan,
        config=config,
        raw_fields=raw_fields,
    )
    if failures:
        _print_json({"status": "invalid", "failures": failures})
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(final_rows, key=lambda item: (item.entity_id, item.timestamp_ms, item.row_id)):
            handle.write(canonical_json(row.to_dict()) + "\n")
    feature_payload = {
        "raw_fields": sorted(raw_fields),
        "features": [final_features[name].to_dict() for name in sorted(final_features)],
    }
    (out / "features.json").write_text(
        json.dumps(feature_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_json(
        {
            "status": "materialized",
            "out_dir": str(out),
            "row_count": len(final_rows),
            "entity_count": len(rows_by_entity(final_rows)),
            "feature_count": len(final_features),
            "dataset_fingerprint": dataset_fingerprint(final_rows),
        }
    )
    return 0


def command_lease_demo(args: argparse.Namespace) -> int:
    manager = LeaseManager(
        clock=MonotonicClock("monotonic_ns"),
        ttl_ms=args.ttl_ms,
        renew_interval_ms=max(1, args.ttl_ms // 4),
        shard_count=1,
        wait_timeout_ms=args.wait_timeout_ms,
    )
    first = manager.acquire(args.entity, owner="worker-A", version="demo-v1")
    started_ns = manager.clock.now_ns()
    outcome: dict[str, Any] = {}

    def contender() -> None:
        try:
            outcome["lease"] = manager.acquire(
                args.entity, owner="worker-B", version="demo-v1"
            )
            outcome["finished_ns"] = manager.clock.now_ns()
        except AppError as exc:
            outcome["error"] = exc.to_failure()

    thread = threading.Thread(target=contender, name="lease-contender", daemon=True)
    thread.start()
    thread.join(timeout=args.wait_timeout_ms / 1000 + 1)
    result = {
        "status": "reclaimed" if "lease" in outcome else "failed",
        "first_lease": first.to_dict(),
        "contender": outcome.get("lease").to_dict() if "lease" in outcome else outcome.get("error"),
        "elapsed_ms": round((int(outcome.get("finished_ns", manager.clock.now_ns())) - started_ns) / 1e6, 3),
        "conflict_observed": any(
            event.get("event") == "lease_conflict" for event in manager.events
        ),
        "events": manager.events,
    }
    _print_json(result)
    return 0 if result["status"] == "reclaimed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-backfill",
        description="Deterministic offline feature backfill with quota, leases and atomic snapshots",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_inputs(target: argparse.ArgumentParser, *, plan: bool = True) -> None:
        target.add_argument("--samples", required=True, help="sample JSONL path")
        target.add_argument("--features", required=True, help="feature definition JSON path")
        target.add_argument("--config", required=True, help="backfill YAML config path")
        if plan:
            target.add_argument("--plan", required=True, help="backfill plan JSON path")

    validate = sub.add_parser("validate", help="validate all input artefacts")
    add_inputs(validate)
    validate.add_argument("--injections", help="optional injection manifest")
    validate.add_argument("--report", help="write validation report JSON")
    validate.set_defaults(func=command_validate, enable_injections=True)

    run = sub.add_parser("run", help="run or resume backfill")
    add_inputs(run)
    run.add_argument("--snapshot-dir", help="override snapshot root")
    run.add_argument("--injections", help="optional injection manifest")
    run.add_argument("--resume", action="store_true", help="resume from committed CURRENT snapshot")
    run.add_argument(
        "--crash-before-commit-batch",
        type=int,
        help="test hook: exit 137 after staging this batch but before CURRENT replacement",
    )
    run.add_argument("--no-injections", action="store_true")
    run.add_argument("--report", help="write full JSON report")
    run.set_defaults(func=command_run, enable_injections=True)

    compare = sub.add_parser("compare", help="compare two full reports")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument(
        "--execution",
        action="store_true",
        help="also compare quota allocation and batch execution sequences",
    )
    compare.set_defaults(func=command_compare)

    materialize = sub.add_parser(
        "materialize", help="write the final sample/feature state after the plan"
    )
    add_inputs(materialize)
    materialize.add_argument("--out-dir", required=True)
    materialize.set_defaults(func=command_materialize)

    lease_demo = sub.add_parser(
        "lease-demo",
        help="demonstrate real monotonic lease expiry, conflict wait and reclaim",
    )
    lease_demo.add_argument("--entity", default="E-LEASE-DEMO")
    lease_demo.add_argument("--ttl-ms", type=int, default=60)
    lease_demo.add_argument("--wait-timeout-ms", type=int, default=3000)
    lease_demo.set_defaults(func=command_lease_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AppError as exc:
        _print_json({"status": "rejected", "failures": [exc.to_failure()]})
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
