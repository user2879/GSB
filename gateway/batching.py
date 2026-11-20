"""Deterministic batch planning with scale-adaptive strategy selection.

Strategies (all deterministic, derived only from batch size params, arrival
sequence numbers and backend identifiers):

- ``arrival_fill``  : per backend, sort by arrival seq, fill batches in order.
- ``client_cluster``: per backend, sort by (client_key, seq) so one client's
                      requests cluster into the same batches.
- ``quota_spread``  : per backend, batch size scaled by the backend's quota
                      share, evening out batch counts across backends.

Selection is a deterministic function of scale parameters (request count,
batch size, backend count, quota spread). When the estimated work of the two
best candidate strategies differs by less than STRATEGY_TIE_THRESHOLD batches,
the lexicographically smallest strategy name wins (stable tie-break, never
dependent on iteration order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import (STRATEGY_TIE_THRESHOLD, Batch, Declaration, RoutedRequest)

ARRIVAL_FILL = "arrival_fill"
CLIENT_CLUSTER = "client_cluster"
QUOTA_SPREAD = "quota_spread"
STRATEGIES = (ARRIVAL_FILL, CLIENT_CLUSTER, QUOTA_SPREAD)


@dataclass(frozen=True)
class StrategyDecision:
    strategy: str
    reason: str
    scale: dict          # key scale parameters
    estimates: dict      # strategy -> estimated work (batches)
    tie_threshold: int
    tie_broken: bool


def _chunk(seqs: Tuple[int, ...], size: int) -> List[Tuple[int, ...]]:
    return [tuple(seqs[i:i + size]) for i in range(0, len(seqs), size)]


def _quota_scaled_sizes(decl: Declaration, rule_batch_size: int) -> Dict[str, int]:
    quotas = {b.model_id: b.max_concurrency for b in decl.backends}
    max_quota = max(quotas.values())
    return {
        b.model_id: max(1, round(rule_batch_size * quotas[b.model_id] / max_quota))
        for b in decl.backends
    }


def build_plan(decl: Declaration, routed: List[RoutedRequest],
               strategy: str) -> List[Batch]:
    """Build the deterministic batch plan for the chosen strategy."""
    rule_bs = {r.rule_id: r.batch_size for r in decl.rules}
    by_backend: Dict[str, List[RoutedRequest]] = {}
    for rr in routed:
        by_backend.setdefault(rr.backend_id, []).append(rr)

    sizes = None
    if strategy == QUOTA_SPREAD:
        # quota spread uses the max rule batch size as the base scale
        base = max(rule_bs.values())
        sizes = _quota_scaled_sizes(decl, base)

    batches: List[Batch] = []
    for backend_id in sorted(by_backend):
        group = by_backend[backend_id]
        if strategy == CLIENT_CLUSTER:
            group = sorted(group, key=lambda r: (r.request.client_key,
                                                 r.request.seq))
        else:
            group = sorted(group, key=lambda r: r.request.seq)
        size = sizes[backend_id] if sizes else rule_bs[group[0].rule_id]
        seqs = tuple(r.request.seq for r in group)
        for index, chunk in enumerate(_chunk(seqs, size)):
            batches.append(Batch(
                batch_id=f"{backend_id}#{index:04d}",
                backend_id=backend_id,
                rule_id=group[0].rule_id,
                batch_index=index,
                request_seqs=chunk,
            ))
    # Canonical execution order: by first request seq, then batch_id.
    batches.sort(key=lambda b: (b.request_seqs[0], b.batch_id))
    return batches


def estimate_work(decl: Declaration, routed: List[RoutedRequest],
                  strategy: str) -> int:
    """Estimated work in batch units: number of batches produced."""
    return len(build_plan(decl, routed, strategy))


def select_strategy(decl: Declaration,
                    routed: List[RoutedRequest]) -> StrategyDecision:
    n = len(routed)
    k = len(decl.backends)
    quotas = [b.max_concurrency for b in decl.backends]
    q_min, q_max = min(quotas), max(quotas)
    clients = {r.request.client_key for r in routed}
    avg_per_client = n / max(1, len(clients))
    base_bs = max(r.batch_size for r in decl.rules)

    scale = {
        "requests": n,
        "backends": k,
        "max_batch_size": base_bs,
        "quota_min": q_min,
        "quota_max": q_max,
        "avg_requests_per_client": round(avg_per_client, 3),
    }

    # Deterministic heuristic preference from scale parameters.
    if k >= 3 and q_max > 2 * q_min:
        preferred, reason = QUOTA_SPREAD, (
            f"{k} backends with quota spread {q_min}..{q_max} (>2x): "
            "spread batches by quota share")
    elif avg_per_client >= 2 * base_bs:
        preferred, reason = CLIENT_CLUSTER, (
            f"avg {avg_per_client:.1f} requests/client >= 2x batch size "
            f"{base_bs}: cluster by client key")
    else:
        preferred, reason = ARRIVAL_FILL, (
            f"{n} requests, batch size {base_bs}: fill batches in arrival order")

    estimates = {s: estimate_work(decl, routed, s) for s in STRATEGIES}
    best_work = min(estimates.values())
    # Candidates within the explicit tie threshold of the preferred strategy.
    pref_work = estimates[preferred]
    tied = sorted(s for s in STRATEGIES
                  if abs(estimates[s] - pref_work) <= STRATEGY_TIE_THRESHOLD)
    tie_broken = len(tied) > 1
    chosen = tied[0] if tie_broken else preferred
    if tie_broken and chosen != preferred:
        reason += (f"; tie within threshold {STRATEGY_TIE_THRESHOLD} among "
                   f"{tied} (work={estimates}), stable tie-break picks "
                   f"{chosen!r}")
    elif best_work < pref_work - STRATEGY_TIE_THRESHOLD:
        # Preferred heuristic is clearly worse than the best estimate: still
        # keep the heuristic (documented trade-off) but record the gap.
        reason += (f"; note: {min(estimates, key=estimates.get)} would save "
                   f"{pref_work - best_work} batches, kept for determinism")
    return StrategyDecision(chosen, reason, scale, estimates,
                            STRATEGY_TIE_THRESHOLD, tie_broken)
