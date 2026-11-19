from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Callable

import numpy as np

from .config import Config
from .loaders import Document, Query
from .postings import FIELDS, LogicalIndex
from .util import round_half_up_decimal, round_half_up_fraction, sha256_json


@dataclass(frozen=True)
class SearchOutcome:
    results: tuple[dict, ...]
    diagnostics: dict
    reused_query_ids: tuple[str, ...]
    rebuilt_query_ids: tuple[str, ...]


@dataclass(frozen=True)
class _DocScoringData:
    document: Document
    title_unique: frozenset[str]
    body_counts: dict[str, int]


def _cooc_ratio(dot: int, query_norm_sq: int, doc_norm_sq: int) -> Decimal:
    if dot <= 0 or query_norm_sq <= 0 or doc_norm_sq <= 0:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 70
        return Decimal(dot) / Decimal(query_norm_sq * doc_norm_sq).sqrt()


def _cooc_ratio_float(dot: int, query_norm_sq: int, doc_norm_sq: int) -> np.float64:
    if dot <= 0 or query_norm_sq <= 0 or doc_norm_sq <= 0:
        return np.float64(0.0)
    return np.float64(dot) / np.sqrt(np.float64(query_norm_sq) * np.float64(doc_norm_sq))


def _fixed_contribution(weight: Fraction, raw: Decimal, temperature: Fraction, scale: int) -> int:
    with localcontext() as context:
        context.prec = 80
        value = (
            Decimal(weight.numerator)
            * raw
            * Decimal(scale)
            / (Decimal(weight.denominator) * Decimal(temperature.numerator) / Decimal(temperature.denominator))
        )
    return round_half_up_decimal(value)


def _lexical_contribution(weight: Fraction, overlap: int, denominator: int, temperature: Fraction, scale: int) -> int:
    if denominator <= 0:
        return 0
    value = weight * Fraction(overlap, denominator) * scale / temperature
    return round_half_up_fraction(value)


def _query_reuse_signature(
    query: Query,
    logical: LogicalIndex,
    config: Config,
) -> str:
    terms = sorted(set(query.terms))
    relevant = {
        field: {
            term: [
                [posting.doc_id, posting.tf]
                for posting in logical.postings[field].get(term, ())
            ]
            for term in terms
        }
        for field in FIELDS
    }
    return sha256_json(
        {
            "schema": "cm-fusion-query-result-v1",
            "query_id": query.query_id,
            "terms": list(query.terms),
            "stream_version": query.stream_version,
            "scoring_hash": config.scoring_hash(),
            "relevant_postings": relevant,
        }
    )


def _doc_data(documents: tuple[Document, ...]) -> list[_DocScoringData]:
    result = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        result.append(
            _DocScoringData(
                document=document,
                title_unique=frozenset(document.title_terms),
                body_counts=dict(Counter(document.body_terms)),
            )
        )
    return result


def _score_query(query: Query, docs: list[_DocScoringData], config: Config, reuse_signature: str) -> dict:
    query_counter = Counter(query.terms)
    unique_terms = sorted(query_counter)
    query_norm_sq = sum(value * value for value in query_counter.values())
    scale = config.fixed_scale
    temperature = config.fusion_temperature
    over_tolerance: list[str] = []
    candidates: list[dict] = []

    for doc in docs:
        body_counts = doc.body_counts
        dot = sum(query_counter[term] * body_counts.get(term, 0) for term in unique_terms)
        doc_norm_sq = sum(body_counts.get(term, 0) ** 2 for term in unique_terms)
        overlap_terms = [term for term in unique_terms if term in doc.title_unique]
        overlap = len(overlap_terms)
        if dot == 0 and overlap == 0:
            continue
        lexical_raw = Decimal(overlap) / Decimal(len(unique_terms))
        lexical_raw_float = float(overlap) / float(len(unique_terms))
        cooc_raw = _cooc_ratio(dot, query_norm_sq, doc_norm_sq)
        cooc_raw_float = _cooc_ratio_float(dot, query_norm_sq, doc_norm_sq)

        lexical_fixed = round_half_up_decimal(lexical_raw * Decimal(scale))
        cooc_fixed = round_half_up_decimal(cooc_raw * Decimal(scale))
        lexical_contribution = _lexical_contribution(
            config.lexical_weight, overlap, len(unique_terms), temperature, scale
        )
        cooc_contribution = _fixed_contribution(
            config.cooccurrence_weight, cooc_raw, temperature, scale
        )
        fused_fixed = lexical_contribution + cooc_contribution
        fused_float = (
            float(config.lexical_weight) * lexical_raw_float / float(temperature)
            + float(config.cooccurrence_weight) * float(cooc_raw_float) / float(temperature)
        )
        error_units = abs(float(fused_fixed) - fused_float * float(scale))
        if error_units > float(config.fixed_tolerance_units):
            over_tolerance.append(doc.document.doc_id)
        candidates.append(
            {
                "doc_id": doc.document.doc_id,
                "shard": doc.document.shard,
                "fused_fixed": fused_fixed,
                "float64_reference": fused_float,
                "abs_error_units": error_units,
                "lexical": {
                    "overlap_count": overlap,
                    "overlap_terms": overlap_terms,
                    "query_unique_term_count": len(unique_terms),
                    "raw_fixed": lexical_fixed,
                    "raw_float": lexical_raw_float,
                    "weight": float(config.lexical_weight),
                    "contribution_fixed": lexical_contribution,
                },
                "cooccurrence": {
                    "dot": dot,
                    "query_norm_sq": query_norm_sq,
                    "doc_norm_sq": doc_norm_sq,
                    "raw_fixed": cooc_fixed,
                    "raw_float": float(cooc_raw_float),
                    "weight": float(config.cooccurrence_weight),
                    "contribution_fixed": cooc_contribution,
                },
            }
        )

    candidates.sort(key=lambda item: (-item["fused_fixed"], item["doc_id"]))
    float_order = sorted(candidates, key=lambda item: (-item["float64_reference"], item["doc_id"]))
    selected = candidates[: config.top_k]
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    max_error = max((item["abs_error_units"] for item in candidates), default=0.0)
    fixed_ids = [item["doc_id"] for item in selected]
    float_ids = [item["doc_id"] for item in float_order[: config.top_k]]
    return {
        "query_id": query.query_id,
        "stream_version": query.stream_version,
        "query_terms": list(query.terms),
        "reuse_signature": reuse_signature,
        "candidate_count": len(candidates),
        "candidates": selected,
        "max_fixed_error_units": max_error,
        "over_tolerance_docs": sorted(over_tolerance),
        "float_topk_rank_agreement": fixed_ids == float_ids,
        "scoring_hash": config.scoring_hash(),
    }


def execute_search(
    documents: tuple[Document, ...],
    queries: tuple[Query, ...],
    logical: LogicalIndex,
    config: Config,
    previous_results: list[dict] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> SearchOutcome:
    docs = _doc_data(documents)
    prior = {item.get("query_id"): item for item in (previous_results or [])}
    signatures = {query.query_id: _query_reuse_signature(query, logical, config) for query in queries}

    def run_one(index: int, query: Query) -> tuple[int, dict, bool, int, int, int, float, str]:
        if checkpoint is not None:
            checkpoint()
        started = time.perf_counter_ns()
        signature = signatures[query.query_id]
        prior_item = prior.get(query.query_id)
        reused = bool(prior_item and prior_item.get("reuse_signature") == signature and prior_item.get("scoring_hash") == config.scoring_hash())
        if reused:
            result = prior_item
            reason = "reuse_signature_match"
        else:
            result = _score_query(query, docs, config, signature)
            if prior_item is None:
                reason = "no_previous_result"
            elif prior_item.get("scoring_hash") != config.scoring_hash():
                reason = "scoring_config_changed"
            else:
                reason = "relevant_postings_changed"
        logical_cost = len(docs) * max(1, len(query.terms))
        elapsed_ns = time.perf_counter_ns() - started
        return index, result, reused, logical_cost, len(result.get("candidates", [])), query.line_no, elapsed_ns / 1_000_000.0, reason

    tasks = list(enumerate(queries))
    observed: list[tuple[int, dict, bool, int, int, int, float, str]] = []
    if config.concurrency == 1 or len(tasks) <= 1:
        for index, query in tasks:
            observed.append(run_one(index, query))
    else:
        with ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="cm-query") as executor:
            futures = [executor.submit(run_one, index, query) for index, query in tasks]
            observed = [future.result() for future in futures]

    observed.sort(key=lambda item: item[0])
    results = tuple(item[1] for item in observed)
    reuse_ids = tuple(item[1]["query_id"] for item in observed if item[2])
    rebuilt_ids = tuple(item[1]["query_id"] for item in observed if not item[2])
    timings = [
        {
            "order": item[0],
            "query_id": item[1]["query_id"],
            "logical_cost": item[3],
            "deterministic_time_ms": item[3] / 1000.0,
            "candidate_count": item[4],
        }
        for item in observed
    ]
    elapsed_observations = [
        {
            "order": item[0],
            "query_id": item[1]["query_id"],
            "observed_wall_time_ms": item[6],
        }
        for item in observed
    ]
    query_invalidation = {
        item[1]["query_id"]: {
            "action": "reused" if item[2] else "rebuilt",
            "reason": item[7],
        }
        for item in observed
    }
    completed = len(results)
    result_count = sum(len(result.get("candidates", [])) for result in results)
    max_error = max((result.get("max_fixed_error_units", 0.0) for result in results), default=0.0)
    over_docs = sorted({doc_id for result in results for doc_id in result.get("over_tolerance_docs", [])})
    diagnostics = {
        "schema_version": 1,
        "scoring_hash": config.scoring_hash(),
        "query_count": len(queries),
        "summary": {
            "completed_queries": completed,
            "result_count": result_count,
            "progress": 1.0 if completed == len(queries) else (completed / len(queries) if queries else 1.0),
        },
        "query_reuse": {
            "reused_query_ids": list(reuse_ids),
            "rebuilt_query_ids": list(rebuilt_ids),
            "reused_query_count": len(reuse_ids),
            "rebuilt_query_count": len(rebuilt_ids),
        },
        "query_invalidation": query_invalidation,
        "query_timings": timings,
        "query_elapsed_observations": elapsed_observations,
        "timing_note": "query_timings is reproducible; query_elapsed_observations is wall-clock-only",
        "max_fixed_error_units": max_error,
        "over_tolerance_doc_count": len(over_docs),
        "over_tolerance_docs": over_docs,
    }
    return SearchOutcome(results, diagnostics, reuse_ids, rebuilt_ids)
