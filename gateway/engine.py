"""Run orchestration: validate -> fingerprint -> route -> strategy -> plan ->
schedule -> commit -> statistics. Recovery/replay lives in replay.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .batching import StrategyDecision, build_plan, select_strategy
from .fingerprint import fingerprints
from .forward import forward_scores, summarize
from .ledger import Ledger, config_hash
from .loader import load_declaration
from .models import BatchResult, Declaration, RoutedRequest
from .routing import route_requests
from .scheduler import Scheduler
from .stats import batch_hit_stats, canary_distribution


@dataclass
class RunResult:
    decl: Declaration
    fingerprints: Dict[str, str]
    routed: List[RoutedRequest]
    decision: StrategyDecision
    batches: list
    results: List[BatchResult]
    config_version: int
    config_hash: str
    stale_versions: List[int]
    canary_distribution: dict
    batch_stats: dict
    peak_running: int
    backend_peak: Dict[str, int]
    skipped_committed: List[str] = field(default_factory=list)


def _batch_summaries(decl: Declaration, result: BatchResult) -> Dict[int, int]:
    inputs = tuple(decl.requests[seq].inputs for seq in result.batch.request_seqs)
    scores = forward_scores(
        {b.model_id: b.params for b in decl.backends}[result.batch.backend_id],
        inputs)
    return summarize(scores)


def run(backends_path: str, rules_path: str, requests_path: str,
        ledger: Ledger, global_limit: int = 4, forward_repeat: int = 1,
        timeout_ms: Optional[int] = None,
        crash_after_commits: Optional[int] = None) -> RunResult:
    decl = load_declaration(backends_path, rules_path, requests_path)
    fps = fingerprints(decl)
    routed = route_requests(decl)
    decision = select_strategy(decl, routed)
    batches = build_plan(decl, routed, decision.strategy)

    chash = config_hash(decl, decision.strategy)
    version = ledger.ensure_version(chash)
    stale = [s["config_version"] for s in ledger.snapshots()
             if s["config_version"] != version]

    routed_by_seq: Dict[int, RoutedRequest] = {r.request.seq: r for r in routed}
    committed = set(ledger.committed_batch_ids(version))
    todo = [b for b in batches if b.batch_id not in committed]
    skipped = sorted(committed)

    def on_committed(result: BatchResult) -> None:
        ledger.commit_batch(version, result, routed_by_seq,
                            _batch_summaries(decl, result))

    scheduler = Scheduler(decl, global_limit, forward_repeat, timeout_ms,
                          on_batch_committed=on_committed,
                          crash_after_commits=crash_after_commits)
    results = scheduler.run(todo, fps)

    # Current view is built wholesale from the CURRENT version only; rows
    # from stale versions never merge into the report.
    rows = ledger.ledger_rows(version)
    return RunResult(
        decl=decl, fingerprints=fps, routed=routed, decision=decision,
        batches=batches, results=results, config_version=version,
        config_hash=chash, stale_versions=stale,
        canary_distribution=canary_distribution(rows),
        batch_stats=batch_hit_stats(rows),
        peak_running=scheduler.peak_running,
        backend_peak=scheduler.backend_peak,
        skipped_committed=skipped,
    )
