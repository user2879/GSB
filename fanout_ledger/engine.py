"""Session orchestration: run fan-out, aggregate, judge SLO, snapshot."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Scene
from .ledger import Ledger, canonical_json
from .logging_util import StructuredLogger
from .metrics import aggregate, canonical_judgment, judge_slo
from .scheduler import FanoutScheduler, RuntimeCounters
from .snapshot import write_snapshot


@dataclass
class SessionResult:
    session_id: str
    counters: RuntimeCounters
    metrics: dict[str, Any]
    judgment: dict[str, Any]
    final_snapshot: Path
    retry_exhaustions: list[Any]


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def run_session(scene: Scene, run_dir: str | Path, session_id: str | None = None,
                snapshot_every: int = 0,
                progress_cb: Callable[[RuntimeCounters], None] | None = None,
                logger: StructuredLogger | None = None) -> SessionResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "snapshots"
    db_path = run_dir / "ledger.sqlite3"
    session_id = session_id or new_session_id()

    ledger = Ledger(db_path, session_id)
    try:
        scheduler = FanoutScheduler(
            scene, ledger, snapshot_dir=str(snapshot_dir),
            snapshot_every=snapshot_every, progress_cb=progress_cb,
        )
        counters = scheduler.run()
        records = ledger.fetch_attempts()
        metrics = aggregate(records, scene)
        judgment = judge_slo(metrics, scene)
        ledger.save_judgment(canonical_judgment(judgment),
                             judged_at_ms=scheduler.counters.trajectory[-1]["ended_ms"]
                             if scheduler.counters.trajectory else 0)
        snapshot = scheduler.snapshot("completed", metrics=metrics, judgment=judgment)
        final_path = write_snapshot(snapshot, snapshot_dir,
                                    name=f"snapshot-{session_id}-final.json")
        if logger:
            logger.info(
                "session_completed",
                session_id=session_id,
                passed=judgment["passed"],
                attempts=metrics["global"]["samples"],
                success_rate_scaled=metrics["global"]["success_rate_scaled"],
                p95_latency_ms=metrics["global"]["p95_latency_ms"],
                snapshot=str(final_path),
            )
        return SessionResult(
            session_id=session_id, counters=counters, metrics=metrics,
            judgment=judgment, final_snapshot=final_path,
            retry_exhaustions=ledger.fetch_retry_exhaustions(),
        )
    finally:
        ledger.close()


def render_report(result: SessionResult, scene: Scene) -> str:
    """Deterministic, stable-order human-readable summary report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"FAN-OUT SESSION REPORT  session={result.session_id}")
    lines.append(f"scene_version={scene.scene_version}  payload_digest={scene.payload_digest[:16]}...")
    g = result.metrics["global"]
    lines.append(
        f"GLOBAL deliveries={g['samples']} successful_deliveries={g['success_count']} "
        f"success_rate={g['success_rate_scaled']}/{g['success_rate_scale']} "
        f"p95_ms={g['p95_latency_ms']} throughput_kbps="
        f"{g['throughput_kbps_scaled']}/{g['throughput_scale']} "
        f"total_attempts={g['attempt_samples']}"
    )
    counts = g["outcome_counts"]
    lines.append(
        "DELIVERY_OUTCOMES " + " ".join(f"{name}={counts[name]}" for name in sorted(counts))
    )
    attempt_counts = g["attempt_outcome_counts"]
    lines.append(
        "ATTEMPT_OUTCOMES " + " ".join(f"{name}={attempt_counts[name]}"
                                       for name in sorted(attempt_counts))
    )
    lines.append("PER-NODE (scene order)")
    for block, node in zip(result.metrics["nodes"], scene.nodes):
        lines.append(
            f"  {node.node_id:<10} deliveries={block['samples']:<3} "
            f"attempts={block['attempt_samples']:<3} "
            f"success_rate={block['success_rate_scaled']:<6} "
            f"p95_ms={block['p95_latency_ms']:<6} "
            f"kbps={block['throughput_kbps_scaled']:<7} "
            f"delivery_outcomes=" + ",".join(
                f"{k}:{v}" for k, v in sorted(block["outcome_counts"].items()) if v)
        )
    lines.append(
        f"SLO passed={result.judgment['passed']} "
        f"thresholds(success%={scene.slo.success_rate_percent}, "
        f"p95<={scene.slo.p95_latency_ms}ms)"
    )
    if result.judgment["reasons"]:
        for reason in result.judgment["reasons"]:
            lines.append("  SLO_FAIL " + canonical_json(reason))
    else:
        lines.append("  SLO_FAIL none")
    if result.retry_exhaustions:
        for row in result.retry_exhaustions:
            lines.append(
                f"  E_RETRY node={row['node_id']} delivery={row['delivery_seq']} "
                f"attempts_used={row['attempts_used']} "
                f"backoff_ms={row['total_backoff_ms']} last={row['last_outcome']}"
            )
    lines.append(f"SNAPSHOT {result.final_snapshot}")
    lines.append("=" * 72)
    return "\n".join(lines)
