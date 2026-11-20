"""Scale-adaptive strategy selector.

The selector enumerates a fixed preset strategy table, estimates each
candidate's per-query work from concrete scale metrics, and picks the
cheapest strategy whose estimated latency stays within the configured
budget. Every input metric, cost estimate, final choice and reason is
recorded for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import PipelineConfig, fingerprint

# Rough but explicit unit costs, chosen to be conservative on a laptop.
_ENTITY_CHECK_NS = 220.0
_VOCAB_CHECK_NS = 180.0
_RULE_CHECK_NS = 150.0
_CACHE_LOOKUP_NS = 300.0
# Building/consulting an index has a fixed per-query bookkeeping cost that
# only pays off once the dictionary is large enough.
_INDEX_BUILD_NS = 90_000.0


@dataclass
class StrategyDecision:
    entity_mode: str
    intent_mode: str
    cache_enabled: bool
    estimated_latency_ms: float
    estimated_costs: List[dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    budget_ms: float = 0.0
    reason: str = ""
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_mode": self.entity_mode,
            "intent_mode": self.intent_mode,
            "cache_enabled": self.cache_enabled,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_costs": self.estimated_costs,
            "metrics": self.metrics,
            "budget_ms": self.budget_ms,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
        }


class StrategySelector:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def choose(self, *, query_count: int, vocab_size: int,
               dict_size: int, rule_count: int,
               avg_query_len: float) -> StrategyDecision:
        thresholds = self.config.strategy_thresholds
        # Expected distinct first characters / prefixes in one query. The
        # indexes only probe entries keyed by characters actually observed, so
        # expected work scales with alphabet coverage rather than table size.
        distinct_chars = min(avg_query_len, 64.0)
        expected_prefixes = max(distinct_chars - 1.0, 1.0)
        # Effective alphabet size: grows gently with table size and saturates
        # at a conservative CJK+ASCII ceiling.
        alphabet_first = min(2_048.0, 64.0 + dict_size ** 0.5)
        # Two-character prefixes spread entries over a much larger key space,
        # so the prefix-group index prunes harder once dictionaries grow.
        alphabet_prefix = min(8_000.0, 24.0 + dict_size * 0.012)

        def indexed_cost(table_size: float, observed: float,
                         alphabet: float) -> float:
            if table_size <= 0:
                return 0.0
            bucket = table_size / max(alphabet, 1.0)
            return min(table_size, observed * bucket)

        entity_paths = {
            "exhaustive": float(dict_size),
            "inverted": indexed_cost(float(dict_size), distinct_chars,
                                     alphabet_first)
                + _INDEX_BUILD_NS / _ENTITY_CHECK_NS,
            "prefix_group": indexed_cost(float(dict_size), expected_prefixes,
                                         alphabet_prefix)
                + 1.6 * _INDEX_BUILD_NS / _ENTITY_CHECK_NS,
        }
        intent_paths = {
            "full_scan": float(vocab_size),
            "first_char_pruned": indexed_cost(
                float(vocab_size), distinct_chars, alphabet_first)
                + _INDEX_BUILD_NS / _VOCAB_CHECK_NS,
        }

        estimates: List[dict] = []
        viable: List[dict] = []
        for entity_mode, entity_checks in entity_paths.items():
            for intent_mode, intent_checks in intent_paths.items():
                for cache_enabled in (False, True):
                    work_ns = (
                        entity_checks * _ENTITY_CHECK_NS
                        + intent_checks * _VOCAB_CHECK_NS
                        + rule_count * _RULE_CHECK_NS
                        + (_CACHE_LOOKUP_NS if cache_enabled else 0.0)
                    )
                    latency_ms = round(work_ns / 1_000_000.0, 6)
                    record = {
                        "entity_mode": entity_mode,
                        "intent_mode": intent_mode,
                        "cache_enabled": cache_enabled,
                        "estimated_entity_checks": round(entity_checks, 3),
                        "estimated_intent_checks": round(intent_checks, 3),
                        "estimated_latency_ms": latency_ms,
                    }
                    estimates.append(record)
                    if latency_ms <= self.config.latency_budget_ms:
                        viable.append(record)

        if viable:
            best = min(viable, key=lambda item: (
                item["estimated_latency_ms"],
                item["entity_mode"], item["intent_mode"],
                item["cache_enabled"]))
            reason = "minimum estimated latency within budget"
        else:
            best = min(estimates, key=lambda item: (
                item["estimated_latency_ms"],
                item["entity_mode"], item["intent_mode"],
                item["cache_enabled"]))
            reason = "no strategy fits budget; selected cheapest available"

        # Threshold guardrails document the scale regime explicitly.
        if dict_size >= thresholds["prefix_dict_entries"]:
            guardrail = "large-dictionary regime"
        elif dict_size >= thresholds["inverted_dict_entries"]:
            guardrail = "medium-dictionary regime"
        else:
            guardrail = "small-dictionary regime"
        cache_by_scale = query_count >= thresholds["cache_queries"]
        if cache_by_scale and not best["cache_enabled"]:
            # Prefer the cached equivalent when scale demands reuse.
            cached = next(
                (item for item in viable
                 if item["entity_mode"] == best["entity_mode"]
                 and item["intent_mode"] == best["intent_mode"]
                 and item["cache_enabled"]),
                None,
            )
            if cached is not None:
                best = cached
                reason = f"{reason}; cache enabled at query scale"

        metrics = {
            "query_count": float(query_count),
            "vocab_size": float(vocab_size),
            "dict_size": float(dict_size),
            "rule_count": float(rule_count),
            "avg_query_len": round(float(avg_query_len), 6),
            "expected_distinct_chars": round(distinct_chars, 3),
        }
        decision = StrategyDecision(
            entity_mode=best["entity_mode"],
            intent_mode=best["intent_mode"],
            cache_enabled=best["cache_enabled"],
            estimated_latency_ms=best["estimated_latency_ms"],
            estimated_costs=estimates,
            metrics=metrics,
            budget_ms=self.config.latency_budget_ms,
            reason=f"{guardrail}; {reason}",
        )
        # The cache scope must only depend on the algorithmic choice, not on
        # audit metrics such as average query length; otherwise a single
        # replaced sample would invalidate the whole tier.
        decision.fingerprint = fingerprint({
            "entity_mode": decision.entity_mode,
            "intent_mode": decision.intent_mode,
            "cache_enabled": decision.cache_enabled,
        })
        return decision
