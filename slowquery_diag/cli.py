"""Command-line entry point for the slow-query diagnostic engine."""

from __future__ import annotations

import argparse
import os
import sys

from .engine import DiagnosticEngine
from .errors import DiagError
from .models import canonical_json


def _serialize(result) -> dict:
    report = result.report
    return {
        "scheme_version": result.scheme_version,
        "verdicts": [
            {
                "query_id": query_id,
                "latency_winner": result.verdicts[query_id].latency.winner_source,
                "latency_adopted_us": result.verdicts[query_id].latency.adopted_value,
                "latency_views": [
                    view.__dict__
                    for view in result.verdicts[query_id].latency.views
                ],
                "rows_winner": result.verdicts[query_id].rows.winner_source,
                "rows_adopted": result.verdicts[query_id].rows.adopted_value,
                "rows_views": [
                    view.__dict__
                    for view in result.verdicts[query_id].rows.views
                ],
            }
            for query_id in sorted(result.verdicts)
        ],
        "error_bounds": [
            {
                "metric": bound.metric,
                "query_id": bound.query_id,
                "total_raw": bound.total_raw,
                "total_ticks": bound.total_ticks,
                "dominant_segment": bound.dominant_segment,
                "contributions": [item.__dict__ for item in bound.contributions],
            }
            for bound in sorted(
                result.error_bounds.values(),
                key=lambda item: (item.metric, item.query_id),
            )
        ],
        "diagnosis": {
            "scale_exponent": report.scale_exponent,
            "session_ratios": report.session_ratios,
            "evaluations": [item.__dict__ for item in report.evaluations],
            "attributions": [item.__dict__ for item in report.attributions],
        },
        "replay": [
            {
                "candidate_id": item.job.candidate_id,
                "source_id": item.job.source_id,
                "ok": item.ok,
                "ack": item.ack,
                "error_code": item.error_code,
                "wait_ms": item.job.wait_end_ms - item.job.wait_start_ms,
            }
            for item in result.replay_results
        ],
        "view_replacement": (
            result.view_replacement.__dict__
            if result.view_replacement is not None else None
        ),
        "stale_entries": [
            {
                "entry_key": list(entry.entry_key),
                "stale_scheme_version": entry.stale_scheme_version,
                "reason": entry.reason,
                "related_sources": list(entry.related_sources),
                "related_candidates": list(entry.related_candidates),
            }
            for entry in getattr(result, "stale_entries", [])
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slowquery-diag",
        description="Deterministic multi-source slow-query diagnostics",
    )
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--scale-exponent", type=int, default=1,
                        choices=(0, 1, 2, 3))
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--catalog")
    parser.add_argument("--global-concurrency", type=int, default=2)
    parser.add_argument("--per-source-quota", type=int, default=1)
    parser.add_argument("--socket-timeout-ms", type=int, default=2000)
    parser.add_argument("--output", help="write canonical JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = DiagnosticEngine(
        os.path.abspath(args.ledger),
        scale_exponent=args.scale_exponent,
    )
    try:
        replay_config = None
        if args.replay:
            catalog_path = args.catalog or os.path.join(
                os.path.dirname(os.path.abspath(args.ledger)),
                "replay_catalog.json",
            )
            # The catalog only depends on the declaration; build it before
            # the single engine run so the scheme version advances once.
            from .config import load_declaration
            bundle = load_declaration(os.path.abspath(args.declaration))
            engine.build_replay_catalog(bundle, catalog_path)
            replay_config = {
                "catalog_path": catalog_path,
                "global_concurrency": args.global_concurrency,
                "per_source_quota": {
                    source.source_id: args.per_source_quota
                    for source in bundle.sources
                },
                "socket_timeout_ms": args.socket_timeout_ms,
            }
        result = engine.run(
            os.path.abspath(args.declaration),
            replay=args.replay, replay_config=replay_config,
        )
        payload = _serialize(result)
        rendered = canonical_json(payload).decode("utf-8")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(rendered)
        else:
            print(rendered)
        return 0
    except DiagError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
