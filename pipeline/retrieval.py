from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional

from .config import sha256_json
from .data import Query, TranslationEntry, TranslationResolver
from .errors import (
    MissingTranslationInjectionError,
    NonFiniteConfidenceInjectionError,
    QueryBudgetExhaustedError,
    RecallLimitError,
)
from .index import BM25Index
from .injections import Injection
from .normalization import normalize_text, quantize, token_script_language


def _other_language(language: str) -> str:
    return "en" if language == "zh" else "zh"


def _predicted_grade(score: float, thresholds: list[dict]) -> int:
    result = 0
    for threshold in thresholds:
        if score >= float(threshold["min_score"]):
            result = int(threshold["grade"])
        else:
            break
    return result


def _round(value: float) -> float:
    return round(float(value), 12)


def _retrieval_injection(injections: Iterable[Injection]) -> Optional[Injection]:
    ordered = sorted(injections, key=lambda item: item.injection_id)
    return ordered[0] if ordered else None


def _validate_confidence(confidence: float, *, injection_id: Optional[str] = None) -> None:
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise NonFiniteConfidenceInjectionError(
            "alignment confidence is not finite and within [0, 1]",
            injection_id=injection_id,
        )


def _raise_retrieval_injection(
    query: Query,
    injection: Injection,
    resolver: TranslationResolver,
    config: dict,
) -> None:
    if injection.kind == "recall_limit_zero":
        raise RecallLimitError(
            f"injected recall limit 0 for query {query.query_id}",
            injection_id=injection.injection_id,
        )
    if injection.kind == "nonfinite_confidence":
        requested_term = injection.params.get("source_term")
        candidates: list[TranslationEntry] = []
        if requested_term:
            for lang in config["index_languages"]:
                candidates = resolver.lookup(lang, normalize_text(str(requested_term), config["normalization"]))
                if candidates:
                    break
        if not candidates:
            for term in sorted(set(query.tokens)):
                lang = token_script_language(term)
                if lang:
                    candidates.extend(resolver.lookup(lang, term))
                if candidates:
                    break
        if not candidates and resolver.entries:
            candidates = [resolver.entries[0]]
        if not candidates:
            raise NonFiniteConfidenceInjectionError(
                f"injected non-finite confidence had no translation target for query {query.query_id}",
                injection_id=injection.injection_id,
            )
        entry = candidates[0]
        injected = TranslationEntry(
            source_lang=entry.source_lang,
            source_term=entry.source_term,
            target_lang=entry.target_lang,
            target_term=entry.target_term,
            confidence=float("nan"),
        )
        _validate_confidence(injected.confidence, injection_id=injection.injection_id)
        raise AssertionError("unreachable")
    if injection.kind == "translation_term_missing":
        requested_term = injection.params.get("term", "__injected_missing_term__")
        source_lang = str(injection.params.get("source_lang", "en"))
        if source_lang not in config["index_languages"]:
            source_lang = "en"
        missing_term = normalize_text(str(requested_term), config["normalization"])
        if resolver.lookup(source_lang, missing_term):
            missing_term = f"{missing_term}__injected_missing__"
        raise MissingTranslationInjectionError(
            f"injected term {missing_term!r} is absent from the translation table ({source_lang})",
            injection_id=injection.injection_id,
        )
    raise RuntimeError(f"unhandled retrieval injection kind: {injection.kind}")


def retrieval_budget_plan(queries: Iterable[Query], budget_config: dict) -> list[dict]:
    global_budget = int(budget_config["global_retrieval_points"])
    per_query_quota = int(budget_config["per_query_quota"])
    remaining = global_budget
    plan = []
    for query in sorted(queries, key=lambda item: item.query_id):
        if remaining <= 0:
            allocation = 0
            started = False
            code = "E_BUDGET_EXHAUSTED"
        else:
            allocation = min(per_query_quota, remaining)
            started = True
            code = None
        if started:
            remaining -= allocation
        plan.append(
            {
                "query_id": query.query_id,
                "allocation": allocation,
                "started": started,
                "pre_code": code,
                "remaining_allocatable_after": remaining,
            }
        )
    return plan


def budget_exhausted_result(query: Query, lid: dict, plan_item: dict, config: dict, reason: str) -> dict:
    return {
        "query_id": query.query_id,
        "task_key": f"{query.query_id}|query|retrieval_scoring",
        "query_text": query.text,
        "declared_language": query.declared_language,
        "detected_language": lid["detected"],
        "language_used": lid["used"],
        "language_evidence": lid["evidence"],
        "status": "budget_exhausted",
        "code": plan_item["pre_code"],
        "allocation": plan_item["allocation"],
        "consumed_points": 0,
        "candidate_count": 0,
        "scored_candidate_count": 0,
        "recall_limit": int(config["recall_limit"]),
        "results": [],
        "translation_expansions": [],
        "missing_translation_terms": [],
        "diagnostics": {"reason": reason},
        "fixed_point": None,
    }


def _term_plan_sort_key(item: dict) -> tuple:
    return (
        0 if item["role"] == "mono" else 1,
        item["source_language"],
        item["target_language"],
        item["term"],
        item["expanded_term"],
    )


def execute_query(
    query: Query,
    *,
    lid: dict,
    index: BM25Index,
    resolver: TranslationResolver,
    config: dict,
    allocation: int,
    injections: Iterable[Injection],
    document_metadata: dict[str, dict],
) -> dict:
    injection = _retrieval_injection(injections)
    if injection is not None:
        _raise_retrieval_injection(query, injection, resolver, config)

    recall_limit = int(config["recall_limit"])
    if recall_limit <= 0:
        raise RecallLimitError(f"recall limit must be positive for query {query.query_id}")

    bridge_weight = float(config["translation"]["bridge_weight"])
    query_counts = Counter(query.tokens)
    term_plans: list[dict] = []
    expansions: list[dict] = []
    missing_terms: list[dict] = []
    candidate_ids: set[str] = set()

    for term in sorted(query_counts, key=lambda item: (token_script_language(item) or "", item)):
        query_tf = int(query_counts[term])
        source_language = token_script_language(term)
        if source_language not in index.languages:
            missing_terms.append(
                {"term": term, "source_language": source_language, "query_tf": query_tf, "reason": "non_indexable_script"}
            )
            continue
        native_postings = index.posting_map(source_language, term)
        if native_postings:
            term_plans.append(
                {
                    "role": "mono",
                    "term": term,
                    "source_language": source_language,
                    "target_language": source_language,
                    "expanded_term": term,
                    "query_tf": query_tf,
                    "confidence": 1.0,
                    "bridge_weight": 1.0,
                    "postings": native_postings,
                }
            )
            candidate_ids.update(native_postings)
            continue

        target_language = _other_language(source_language)
        candidates = resolver.lookup(source_language, term, [target_language])
        if not candidates:
            missing_terms.append(
                {"term": term, "source_language": source_language, "query_tf": query_tf, "reason": "no_translation"}
            )
            continue
        chosen = candidates[0]
        suppressed = candidates[1:]
        target_postings = index.posting_map(chosen.target_lang, chosen.target_term)
        expansion = {
            "source_term": term,
            "source_language": source_language,
            "query_tf": query_tf,
            "chosen": {
                "target_term": chosen.target_term,
                "target_language": chosen.target_lang,
                "confidence": _round(chosen.confidence),
            },
            "suppressed": [
                {
                    "target_term": item.target_term,
                    "target_language": item.target_lang,
                    "confidence": _round(item.confidence),
                }
                for item in suppressed
            ],
            "document_hits": len(target_postings),
            "weighted_score_sum": 0.0,
        }
        expansions.append(expansion)
        term_plans.append(
            {
                "role": "bridge",
                "term": term,
                "source_language": source_language,
                "target_language": chosen.target_lang,
                "expanded_term": chosen.target_term,
                "query_tf": query_tf,
                "confidence": chosen.confidence,
                "bridge_weight": bridge_weight,
                "postings": target_postings,
                "expansion": expansion,
            }
        )
        candidate_ids.update(target_postings)

    term_plans.sort(key=_term_plan_sort_key)
    ordered_candidate_ids = sorted(candidate_ids)
    fixed = config["fixed_point"]
    fusion = config["fusion"]
    temperature = float(fusion["temperature"])
    mono_weight = float(fusion["mono_weight"])
    cross_weight = float(fusion["cross_weight"])
    weight_total = mono_weight + cross_weight
    scale = int(fixed["scale"])
    scored = []
    consumed_points = 0

    for doc_id in ordered_candidate_ids:
        if consumed_points >= allocation:
            error = QueryBudgetExhaustedError(
                f"query {query.query_id} attempted to score candidate {consumed_points + 1} above allocation {allocation}",
            )
            error.candidate_count = len(ordered_candidate_ids)
            error.consumed_points = consumed_points
            raise error

        term_contributions = []
        mono_raw = 0.0
        bridge_raw = 0.0
        for plan in term_plans:
            if doc_id not in plan["postings"]:
                continue
            base = index.score(plan["target_language"], plan["expanded_term"], doc_id) * plan["query_tf"]
            if plan["role"] == "mono":
                contribution = base
                stable_prefix = "000_mono"
            else:
                contribution = base * plan["confidence"] * plan["bridge_weight"]
                stable_prefix = "001_bridge"
                bridge_raw += contribution
                plan["expansion"]["weighted_score_sum"] += contribution
            if plan["role"] == "mono":
                mono_raw += contribution
            term_contributions.append(
                {
                    "role": plan["role"],
                    "term": plan["term"],
                    "source_language": plan["source_language"],
                    "target_language": plan["target_language"],
                    "expanded_term": plan["expanded_term"],
                    "query_tf": plan["query_tf"],
                    "bm25": _round(base),
                    "confidence": _round(plan["confidence"]),
                    "bridge_weight": _round(plan["bridge_weight"]),
                    "contribution": _round(contribution),
                    "stable_key": (
                        f"{query.query_id}|{doc_id}|term|{stable_prefix}|{plan['source_language']}|"
                        f"{plan['target_language']}|{plan['term']}|{plan['expanded_term']}"
                    ),
                }
            )

        mono_norm = mono_raw / (mono_raw + temperature) if mono_raw > 0.0 else 0.0
        bridge_norm = bridge_raw / (bridge_raw + temperature) if bridge_raw > 0.0 else 0.0
        fused_norm = (mono_weight * mono_norm + cross_weight * bridge_norm) / weight_total
        score_fixed = quantize(fused_norm, fixed)
        mono_fixed = quantize(mono_raw, fixed)
        bridge_fixed = quantize(bridge_raw, fixed)
        normalized_score = score_fixed / scale
        metadata = document_metadata.get(doc_id, {})
        scored.append(
            {
                "doc_id": doc_id,
                "candidate_key": f"{query.query_id}|{doc_id}|retrieval_scoring",
                "language": metadata.get("language", ""),
                "title": metadata.get("title", ""),
                "score_fixed": score_fixed,
                "score": score_fixed / scale,
                "mono_score": _round(mono_raw),
                "mono_fixed": mono_fixed,
                "bridge_score": _round(bridge_raw),
                "bridge_fixed": bridge_fixed,
                "predicted_grade": _predicted_grade(normalized_score, config["relevance_thresholds"]),
                "term_breakdown": term_contributions,
            }
        )
        consumed_points += 1

    scored.sort(key=lambda item: (-item["score_fixed"], -item["mono_fixed"], item["doc_id"]))
    selected = scored[:recall_limit]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        item["fixed_width"] = int(fixed["bits"])
        item["fixed_scale"] = scale

    for expansion in expansions:
        expansion["weighted_score_sum"] = _round(expansion["weighted_score_sum"])
    expansions.sort(key=lambda item: (item["source_language"], item["source_term"]))
    missing_terms.sort(key=lambda item: (item["source_language"] or "", item["term"]))
    return {
        "query_id": query.query_id,
        "task_key": f"{query.query_id}|query|retrieval_scoring",
        "query_text": query.text,
        "declared_language": query.declared_language,
        "detected_language": lid["detected"],
        "language_used": lid["used"],
        "language_evidence": lid["evidence"],
        "status": "ok",
        "code": None,
        "allocation": allocation,
        "consumed_points": consumed_points,
        "candidate_count": len(ordered_candidate_ids),
        "scored_candidate_count": consumed_points,
        "recall_limit": recall_limit,
        "results": selected,
        "translation_expansions": expansions,
        "missing_translation_terms": missing_terms,
        "diagnostics": {
            "bridge_weight": _round(bridge_weight),
            "fusion_temperature": _round(temperature),
            "mono_weight": _round(mono_weight),
            "cross_weight": _round(cross_weight),
            "candidate_count_within_allocation": len(ordered_candidate_ids) <= allocation,
        },
        "fixed_point": {
            "bits": int(fixed["bits"]),
            "scale": scale,
            "max_integer": (1 << (int(fixed["bits"]) - 1)) - 1,
        },
    }


def query_plan_digest(plan: list[dict]) -> str:
    return sha256_json(plan)

