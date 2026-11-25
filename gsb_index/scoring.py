"""Fixed-point scoring and shard-local candidate recall."""

from __future__ import annotations

from math import gcd
from typing import Any, Iterable, Mapping

from .models import Query, ScoringProfile


def _saturate(value: int, bit_width: int) -> int:
    maximum = (1 << bit_width) - 1
    if value < 0:
        return 0
    return maximum if value > maximum else value


def fixed_log2_fraction(numerator: int, denominator: int, fraction_bits: int) -> int:
    """Return floor(log2(numerator/denominator) * 2^fraction_bits).

    This is an integer repeated-squaring implementation.  Floating point is not
    used anywhere in the scoring path.
    """

    if numerator <= 0 or denominator <= 0:
        raise ValueError("log2 arguments must be positive")
    if numerator == denominator:
        return 0
    if numerator < denominator:
        return -fixed_log2_fraction(denominator, numerator, fraction_bits)
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent > 0:
        denominator <<= exponent
        if denominator > numerator:
            denominator >>= 1
            exponent -= 1
    else:
        exponent = 0
    result = exponent << fraction_bits
    for bit in range(fraction_bits):
        numerator *= numerator
        denominator *= denominator
        common = gcd(numerator, denominator)
        numerator //= common
        denominator //= common
        if numerator >= (denominator << 1):
            result += 1 << (fraction_bits - bit - 1)
            denominator <<= 1
            common = gcd(numerator, denominator)
            numerator //= common
            denominator //= common
    return result


def fixed_idf(total_docs: int, document_frequency: int, fraction_bits: int) -> int:
    # log2((N+1)/(df+1)) plus one guarantees a positive integer contribution
    # when every document contains the term.
    return (1 << fraction_bits) + fixed_log2_fraction(total_docs + 1, document_frequency + 1, fraction_bits)


def fixed_term_frequency(tf: int, saturation_k: int, fraction_bits: int) -> int:
    if tf <= 0:
        return 0
    return (tf << fraction_bits) // (tf + saturation_k)


def score_document(
    term_stats: Mapping[str, tuple[int, int] | Mapping[str, int]],
    query: Query,
    total_docs: int,
    profile: ScoringProfile,
) -> dict[str, Any]:
    title_weight = profile.title_weight
    body_weight = profile.body_weight
    fraction_bits = profile.fraction_bits
    contributions: list[dict[str, Any]] = []
    running = 0
    for term in query.terms:
        values = term_stats.get(term)
        if values is None:
            contributions.append(
                {
                    "term": term,
                    "title_tf": 0,
                    "body_tf": 0,
                    "df": 0,
                    "title_tf_fp": 0,
                    "body_tf_fp": 0,
                    "idf_fp": 0,
                    "term_score": 0,
                    "running_score": running,
                }
            )
            continue
        if isinstance(values, Mapping):
            title_tf = int(values.get("title_tf", 0))
            body_tf = int(values.get("body_tf", 0))
            df = int(values.get("df", 0))
        else:
            title_tf, body_tf = values
            df = 1
        title_tf_fp = _saturate(
            fixed_term_frequency(title_tf, profile.tf_saturation_k, fraction_bits), profile.bit_width
        )
        body_tf_fp = _saturate(
            fixed_term_frequency(body_tf, profile.tf_saturation_k, fraction_bits), profile.bit_width
        )
        idf_fp = _saturate(fixed_idf(total_docs, df, fraction_bits), profile.bit_width)
        weighted = _saturate(title_tf_fp * title_weight + body_tf_fp * body_weight, profile.bit_width)
        term_score = _saturate((weighted * idf_fp) >> fraction_bits, profile.bit_width)
        running = _saturate(running + term_score, profile.bit_width)
        contributions.append(
            {
                "term": term,
                "title_tf": title_tf,
                "body_tf": body_tf,
                "df": df,
                "title_tf_fp": title_tf_fp,
                "body_tf_fp": body_tf_fp,
                "idf_fp": idf_fp,
                "term_score": term_score,
                "running_score": running,
            }
        )
    raw_score = running
    step = profile.quantum_step
    quantized = _saturate(((raw_score + step // 2) // step) * step, profile.bit_width)
    return {
        "raw_score": raw_score,
        "quantized_score": quantized,
        "score_components": contributions,
    }


def _summary_documents(summary: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
    documents: dict[str, dict[str, Any]] = {}
    for bucket in summary.get("buckets", []):
        for term_item in bucket.get("terms", []):
            term = term_item["term"]
            for posting in term_item.get("postings", []):
                doc_id = posting["doc_id"]
                entry = documents.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "version": int(posting["version"]),
                        "terms": {},
                    },
                )
                if int(posting["version"]) > entry["version"]:
                    entry["version"] = int(posting["version"])
                entry["terms"][term] = {
                    "title_tf": int(posting.get("title_tf", 0)),
                    "body_tf": int(posting.get("body_tf", 0)),
                    "df": int(term_item.get("df", 0)),
                }
    total_docs = int(summary.get("doc_count", len(documents)))
    return documents, total_docs


def recall_shard(
    summary: Mapping[str, Any],
    query: Query,
    profile: ScoringProfile,
) -> dict[str, Any]:
    documents, total_docs = _summary_documents(summary)
    candidates: list[dict[str, Any]] = []
    query_terms = set(query.terms)
    for doc_id, entry in documents.items():
        if not query_terms.intersection(entry["terms"]):
            continue
        scored = score_document(entry["terms"], query, total_docs, profile)
        candidates.append(
            {
                "doc_id": doc_id,
                "version": entry["version"],
                "shard_id": summary.get("shard_id"),
                "raw_score": scored["raw_score"],
                "quantized_score": scored["quantized_score"],
                "score_components": scored["score_components"],
                "boundary": False,
            }
        )
    candidates.sort(key=lambda item: (-item["quantized_score"], item["doc_id"], item["version"]))
    if not candidates:
        return {
            "query_id": query.query_id,
            "shard_id": summary.get("shard_id"),
            "ranked": [],
            "selected": [],
            "cutoff_score": None,
            "boundary_doc_ids": [],
            "fixed_point": profile.to_dict(),
        }
    selected_count = min(query.top_k, len(candidates))
    cutoff_raw = candidates[selected_count - 1]["raw_score"]
    cutoff_quantized = candidates[selected_count - 1]["quantized_score"]
    half_step_numerator = profile.quantum_step
    for candidate in candidates:
        if abs(candidate["raw_score"] - cutoff_raw) * 2 <= half_step_numerator:
            candidate["boundary"] = True
        elif candidate["quantized_score"] == cutoff_quantized:
            candidate["boundary"] = True
    selected = candidates[:selected_count]
    boundary_doc_ids = [
        item["doc_id"]
        for item in candidates
        if item["boundary"]
    ]
    return {
        "query_id": query.query_id,
        "shard_id": summary.get("shard_id"),
        "ranked": candidates,
        "selected": selected,
        "cutoff_score": cutoff_quantized,
        "boundary_doc_ids": boundary_doc_ids,
        "fixed_point": profile.to_dict(),
    }


def merge_shard_recalls(query: Query, shard_recalls: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for shard_result in shard_recalls:
        for candidate in shard_result.get("ranked", []):
            candidates.append(dict(candidate))
    candidates.sort(key=lambda item: (-item["quantized_score"], item["doc_id"], item["version"]))
    if not candidates:
        return {
            "query_id": query.query_id,
            "candidates": [],
            "selected": [],
            "boundary_doc_ids": [],
            "shards_considered": sorted(result.get("shard_id") for result in shard_recalls),
        }
    selected_count = min(query.top_k, len(candidates))
    selected = candidates[:selected_count]
    cutoff_quantized = selected[-1]["quantized_score"]
    # Quantum boundaries are local decisions; preserve marks from shard recalls.
    for candidate in candidates:
        if candidate["quantized_score"] == cutoff_quantized:
            candidate["boundary"] = True
    boundary = [item["doc_id"] for item in candidates if item["boundary"]]
    return {
        "query_id": query.query_id,
        "candidates": candidates,
        "selected": selected,
        "cutoff_score": cutoff_quantized,
        "boundary_doc_ids": boundary,
        "shards_considered": sorted(result.get("shard_id") for result in shard_recalls),
    }


def recall_all(
    summaries: Iterable[Mapping[str, Any]],
    queries: Iterable[Query],
    profile: ScoringProfile,
) -> list[dict[str, Any]]:
    summary_list = sorted(summaries, key=lambda summary: str(summary.get("shard_id")))
    by_query: list[dict[str, Any]] = []
    for query in queries:
        local = [recall_shard(summary, query, profile) for summary in summary_list]
        by_query.append(merge_shard_recalls(query, local))
    return by_query
