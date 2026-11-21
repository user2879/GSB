"""Command-line interface for repeatable local demos and verification."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .core.engine import Engine, report_to_dict
from .core.errors import GraphDiagError
from .core.fixedpoint import format_micro
from .core.traversal import (
    CONFIRMED_OVER,
    ERROR_CROSS,
    UNREACHABLE,
    WITHIN,
)
from .sample_data import generate

LABEL = {
    WITHIN: "WITHIN",
    CONFIRMED_OVER: "CONFIRMED-OVER",
    ERROR_CROSS: "INDETERMINATE-ERROR-CROSS",
    UNREACHABLE: "UNREACHABLE",
}


def _print_report(engine: Engine) -> None:
    view = engine.current_view
    assert view is not None
    print(
        f"versions graph=v{view.graph_version} queries=v{view.query_version} "
        f"scale=1e{view.scale_exponent}"
    )
    for schedule in view.report.ordered():
        result = schedule.result
        print(
            f"\n[{result.query_id}] source={result.source} "
            f"start_ms={schedule.start_monotonic_ms} end_ms={schedule.end_monotonic_ms} "
            f"visited={schedule.visited_nodes} pruned={schedule.pruned_branches} "
            f"budget={schedule.budget_consumed} exhausted={schedule.budget_exhausted} "
            f"wait_ms={schedule.wait_duration_ms}"
        )
        print(f"  visit sequence: {result.visited_sequence}")
        for target, tr in sorted(result.targets.items()):
            if not tr.reached:
                print(f"  target {target:>3}: {LABEL[tr.classification]}")
                continue
            print(
                f"  target {target:>3}: dist_fp={tr.dist_fp} "
                f"dist={format_micro(tr.dist_micro)} "
                f"err_exact_micro={tr.exact_bound_micro} "
                f"err_cons_ulp={tr.conservative_bound_ulp} "
                f"cap={format_micro(tr.cap_micro)} "
                f"verdict={LABEL[tr.classification]} "
                f"error_affected={tr.error_affected} "
                f"dominant_edge={tr.dominant_edge} "
                f"dominant_remainder_micro={tr.dominant_remainder_micro}"
            )
            print(f"      path={list(tr.path_nodes)}")
            for term in tr.terms:
                print(
                    f"      edge#{term.edge_index} {term.source}->{term.target} "
                    f"w_micro={term.weight_micro} q_fp={term.quantized_fp} "
                    f"remainder_micro={term.edge_remainder_micro} "
                    f"running_cons_ulp={term.conservative_bound_ulp} "
                    f"running_exact_micro={term.exact_bound_micro}"
                )
        for prune in result.pruned:
            print(
                f"  prune {prune.source}->{prune.target} reason={prune.reason} "
                f"candidate_fp={prune.candidate_fp} cap_fp={prune.cap_fp}"
            )


def _print_changes(engine: Engine) -> None:
    changes = engine.diff_views()
    if not changes:
        print("\nno changes versus previous whole view")
        return
    print(f"\nwhole-view replacement: {len(changes)} changed target(s)")
    for change in changes:
        print(
            f"  {change.query_id} target={change.target} "
            f"dist {change.old_dist_micro} -> {change.new_dist_micro} "
            f"delta_micro={change.distance_delta_micro} "
            f"verdict {change.old_classification} -> {change.new_classification}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphdiag",
        description="Fixed-point graph traversal reachability & error diagnostics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gen-samples", help="regenerate deterministic sample JSON files")

    run_p = sub.add_parser("run", help="load a declaration and run all queries")
    run_p.add_argument("declaration")
    run_p.add_argument("--db", default="graphdiag.db")
    run_p.add_argument("--budget", type=int, default=10_000)
    run_p.add_argument("--workers", type=int, default=2)
    run_p.add_argument("--error-mode", choices=["exact", "conservative"], default="exact")
    run_p.add_argument("--wait-timeout", type=float, default=5.0)
    run_p.add_argument("--json-out", help="optional path to dump the full report JSON")

    snap_p = sub.add_parser("snapshot", help="run then persist a session snapshot")
    snap_p.add_argument("declaration")
    snap_p.add_argument("--db", default="graphdiag.db")
    snap_p.add_argument("--snapshot-id", default="snapshot-latest")
    snap_p.add_argument("--budget", type=int, default=10_000)
    snap_p.add_argument("--workers", type=int, default=2)
    snap_p.add_argument("--wait-timeout", type=float, default=5.0)
    snap_p.add_argument("--error-mode", choices=["exact", "conservative"], default="exact")
    snap_p.add_argument("--json-out", help="optional path to dump the full report JSON")

    restore_p = sub.add_parser("restore", help="restore a snapshot and replay-verify it")
    restore_p.add_argument("snapshot_id")
    restore_p.add_argument("--db", default="graphdiag.db")
    restore_p.add_argument("--error-mode", choices=["exact", "conservative"], default=None)

    sub.add_parser("list-snapshots", help="list snapshots in the database").add_argument(
        "--db", default="graphdiag.db"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "gen-samples":
            paths = generate()
            for name, path in paths.items():
                print(f"generated {name}: {path}")
            return 0

        if args.command == "list-snapshots":
            engine = Engine(args.db)
            for row in engine.store.list_snapshots():
                print(row)
            engine.store.close()
            return 0

        if args.command == "restore":
            engine = Engine(args.db, error_mode=args.error_mode or "exact")
            outcome = engine.restore_snapshot(args.snapshot_id)
            print(json.dumps(outcome, indent=2))
            engine.store.close()
            return 0

        engine = Engine(args.db, error_mode=args.error_mode)
        info = engine.load(args.declaration)
        print(json.dumps(info, indent=2))
        engine.run(
            budget_limit=args.budget,
            max_workers=args.workers,
            wait_timeout=args.wait_timeout,
        )
        _print_report(engine)
        _print_changes(engine)

        if args.command == "snapshot":
            digest = engine.save_snapshot(args.snapshot_id)
            print(f"\nsnapshot {args.snapshot_id} saved digest={digest}")
        if getattr(args, 'json_out', None):
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(
                    report_to_dict(engine.current_view.report),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"report JSON written: {args.json_out}")
        engine.store.close()
        return 0
    except GraphDiagError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        if exc.detail:
            print(json.dumps(exc.detail, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
