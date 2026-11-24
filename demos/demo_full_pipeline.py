"""End-to-end happy-path demo on deterministic local samples.

Stages: declaration validation -> multi-version decode -> arbitration ->
fixed-point error bounds -> stable diagnosis -> quota-bounded loopback
replay -> SQLite ledger commit.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slowquery_diag.engine import DiagnosticEngine  # noqa: E402
from slowquery_diag.models import canonical_json  # noqa: E402

WORKDIR = os.path.join(ROOT, "workdir")


def main() -> int:
    os.makedirs(WORKDIR, exist_ok=True)
    declaration = os.path.join(ROOT, "examples", "declaration.json")
    ledger_path = os.path.join(WORKDIR, "demo_ledger.db")
    catalog_path = os.path.join(WORKDIR, "replay_catalog.json")
    if os.path.exists(ledger_path):
        os.remove(ledger_path)

    engine = DiagnosticEngine(ledger_path, scale_exponent=1)
    result = engine.run(declaration)
    engine.build_replay_catalog(result.bundle, catalog_path)
    config = {
        "catalog_path": catalog_path,
        "global_concurrency": 2,
        "per_source_quota": {
            source.source_id: 1 for source in result.bundle.sources
        },
        "socket_timeout_ms": 2000,
    }
    result = engine.run(declaration, replay=True, replay_config=config)

    print("=== merged verdicts (deterministic order) ===")
    for query_id in sorted(result.verdicts):
        verdict = result.verdicts[query_id]
        print(
            f"query={query_id} latency_winner={verdict.latency.winner_source}"
            f" adopted_us={verdict.latency.adopted_value}"
            f" rows_winner={verdict.rows.winner_source}"
            f" rows={verdict.rows.adopted_value}"
        )
        for view in verdict.latency.views:
            flag = "ADOPT" if view.adopted else f"reject:{view.reject_reason}"
            print(f"   {view.source_id:20s} v{view.format_version} {flag}")

    print("\n=== fixed-point error bounds (three-segment split) ===")
    for bound in sorted(result.error_bounds.values(),
                        key=lambda item: (item.metric, item.query_id)):
        parts = ", ".join(
            f"{c.name}={c.raw_units}raw/{c.ticks}ticks"
            + ("*" if c.dominant else "")
            for c in bound.contributions
        )
        print(f"{bound.metric:13s} q={bound.query_id} total="
              f"{bound.total_raw}raw/{bound.total_ticks}ticks [{parts}]")

    print("\n=== stable diagnosis & cause attribution ===")
    for evaluation in result.report.evaluations:
        print(
            f"{evaluation.candidate_id:28s} base="
            f"{evaluation.baseline_latency_ticks}ticks gain="
            f"{evaluation.expected_gain_ticks}ticks affected="
            f"{evaluation.error_affected} note={evaluation.decision_note}"
        )
    print("session ratios:", result.report.session_ratios)

    print("\n=== loopback replay ===")
    for item in result.replay_results:
        wait = item.job.wait_end_ms - item.job.wait_start_ms
        print(
            f"{item.job.source_id:20s} {item.job.candidate_id:28s} ok="
            f"{item.ok} ack_latency="
            f"{item.ack['est_latency_us'] if item.ok else item.error_code}"
            f" queued_wait_ms={wait}"
        )
    print("scheduler stats:", engine._last_scheduler_stats)
    print("ledger digest:", engine.ledger.current_digest())
    engine.close()

    report_path = os.path.join(WORKDIR, "demo_report.json")
    from slowquery_diag.cli import _serialize
    with open(report_path, "wb") as handle:
        handle.write(canonical_json(_serialize(result)))
    print(f"\ncanonical report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
