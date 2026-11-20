"""Per-query processing pipeline with stage-level exception isolation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import PipelineConfig, fingerprint
from .entity import EntityMatcher
from .errors import E_PIPELINE_STAGE
from .intent import IntentScorer
from .normalization import normalize_query
from .rewrite import RewriteEngine
from .strategy import StrategyDecision


STAGE_ORDER = ("normalize", "entity", "intent", "rewrite")


def sort_key(query_id: str, stage: str) -> str:
    """Stable ordering key derived from query id and pipeline stage."""
    return f"{query_id}::{stage}"


@dataclass
class QueryResult:
    query_id: str
    original_query: str
    sort_key: str
    query_fingerprint: str
    ok: bool
    normalized: Optional[dict] = None
    entity: Optional[dict] = None
    intent: Optional[dict] = None
    rewrite: Optional[dict] = None
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    failures: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "sort_key": self.sort_key,
            "query_fingerprint": self.query_fingerprint,
            "ok": self.ok,
            "normalized": self.normalized,
            "entity": self.entity,
            "intent": self.intent,
            "rewrite": self.rewrite,
            "stage_timings_ms": self.stage_timings_ms,
            "total_latency_ms": self.total_latency_ms,
            "failures": self.failures,
        }


class QueryPipeline:
    def __init__(self, config: PipelineConfig, matcher: EntityMatcher,
                 scorer: IntentScorer, rewriter: RewriteEngine,
                 decision: StrategyDecision):
        self.config = config
        self.matcher = matcher
        self.scorer = scorer
        self.rewriter = rewriter
        self.decision = decision

    def process(self, query_id: str, raw_query: str,
                context: dict, labels: dict) -> QueryResult:
        started = time.perf_counter()
        result = QueryResult(
            query_id=query_id,
            original_query=raw_query,
            sort_key=sort_key(query_id, "result"),
            query_fingerprint=fingerprint(
                {"id": query_id, "query": raw_query,
                 "context": context, "labels": labels}),
            ok=True,
        )

        try:
            t0 = time.perf_counter()
            norm = normalize_query(raw_query)
            result.stage_timings_ms["normalize"] = round(
                (time.perf_counter() - t0) * 1000.0, 6)
            result.normalized = norm.to_dict()
        except Exception as exc:
            result.ok = False
            result.failures.append({
                "query_id": query_id,
                "stage": "normalize",
                "code": getattr(exc, "code", E_PIPELINE_STAGE),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "sort_key": sort_key(query_id, "normalize"),
            })
            result.total_latency_ms = round((time.perf_counter() - started) * 1000.0, 6)
            return result

        text = result.normalized["normalized"]

        try:
            t0 = time.perf_counter()
            entity_result = self.matcher.match(text, self.decision.entity_mode)
            result.stage_timings_ms["entity"] = round(
                (time.perf_counter() - t0) * 1000.0, 6)
            result.entity = {
                "mode": entity_result.mode,
                "slots": entity_result.slots,
                "suppressed": entity_result.suppressed,
                "resolutions": entity_result.resolutions,
                "match_checks": entity_result.match_checks,
                "candidate_count": entity_result.candidate_count,
            }
        except Exception as exc:
            result.ok = False
            result.failures.append(self._stage_failure(query_id, "entity", exc))

        try:
            t0 = time.perf_counter()
            intent_result = self.scorer.score(text, self.decision.intent_mode)
            result.stage_timings_ms["intent"] = round(
                (time.perf_counter() - t0) * 1000.0, 6)
            result.intent = {
                "primary_intent": intent_result.primary_intent,
                "raw_scores": intent_result.raw_scores,
                "normalized_scores": intent_result.normalized_scores,
                "contributions": intent_result.contributions,
                "rule_bonuses": intent_result.rule_bonuses,
                "hit_checks": intent_result.hit_checks,
                "mode": intent_result.mode,
            }
        except Exception as exc:
            result.ok = False
            result.failures.append(self._stage_failure(query_id, "intent", exc))

        try:
            t0 = time.perf_counter()
            rewrite_outcome = self.rewriter.generate(
                text, prune=self.decision.entity_mode != "exhaustive")
            result.stage_timings_ms["rewrite"] = round(
                (time.perf_counter() - t0) * 1000.0, 6)
            result.rewrite = {
                "suggestions": rewrite_outcome.suggestions,
                "full_candidates": rewrite_outcome.full_candidates,
                "pruning_diagnosis": rewrite_outcome.pruning_diagnosis,
                "full_rule_checks": rewrite_outcome.full_rule_checks,
                "pruned_rule_checks": rewrite_outcome.pruned_rule_checks,
            }
        except Exception as exc:
            result.ok = False
            result.failures.append(self._stage_failure(query_id, "rewrite", exc))

        result.total_latency_ms = round((time.perf_counter() - started) * 1000.0, 6)
        return result

    @staticmethod
    def _stage_failure(query_id: str, stage: str, exc: Exception) -> dict:
        return {
            "query_id": query_id,
            "stage": stage,
            "code": getattr(exc, "code", E_PIPELINE_STAGE),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "sort_key": sort_key(query_id, stage),
        }
