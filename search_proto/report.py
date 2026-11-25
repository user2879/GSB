from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import SearchEngine
from .util import canonical_json, write_json, write_jsonl


def engine_artifacts(engine: SearchEngine) -> dict[str, Any]:
    index = {
        "schema": "inverted-index-v1",
        "index_fingerprint": engine.index_fingerprint,
        "content_fingerprint": engine.content_fingerprint,
        "config_fingerprint": engine.config.semantic_fingerprint,
        "stopword_fingerprint": engine.stopwords.content_fingerprint,
        "stopword_version": engine.stopwords.effective_version,
        "document_count": len(engine.documents),
        "documents": [
            engine.document_stats[doc_id]
            for doc_id in sorted(engine.document_stats)
        ],
        "postings": [
            {
                "term": term,
                "documents": [
                    engine.postings[term][doc_id]
                    for doc_id in sorted(engine.postings[term])
                ],
            }
            for term in sorted(engine.postings)
        ],
        "dependencies": engine.dependencies(),
    }
    term_stats = {
        "schema": "term-stats-v1",
        "index_fingerprint": engine.index_fingerprint,
        "terms": [engine.term_stats[term] for term in sorted(engine.term_stats)],
    }
    scores = {
        "schema": "score-ledger-v1",
        "index_fingerprint": engine.index_fingerprint,
        "config_hash": engine.config.full_fingerprint,
        "fixed": engine.config.fixed.to_dict(),
        "scores": engine.score_records(),
    }
    ranking = engine.ranking_index()
    failures = [failure.to_dict() for failure in engine.sorted_failures()]
    cache_ledger = engine.cache_ledger_entries()
    report = engine.report()
    report["artifact_fingerprints"] = {
        "index": canonical_json(index),
        "term_stats": canonical_json(term_stats),
        "scores": canonical_json(scores),
        "ranking": canonical_json(ranking),
        "failures": canonical_json(failures),
        "cache_ledger": canonical_json(cache_ledger),
    }
    from .util import sha256_json

    report["artifact_fingerprints"] = {
        key: sha256_json(value)
        for key, value in {
            "index": index,
            "term_stats": term_stats,
            "scores": scores,
            "ranking": ranking,
            "failures": failures,
            "cache_ledger": cache_ledger,
        }.items()
    }
    return {
        "index": index,
        "term_stats": term_stats,
        "scores": scores,
        "ranking": ranking,
        "failures": failures,
        "cache_ledger": cache_ledger,
        "report": report,
    }


def write_run_outputs(engine: SearchEngine, out_dir: Path) -> dict[str, Any]:
    artifacts = engine_artifacts(engine)
    write_json(out_dir / "index.json", artifacts["index"])
    write_json(out_dir / "term_stats.json", artifacts["term_stats"])
    write_json(out_dir / "scores.json", artifacts["scores"])
    write_json(out_dir / "ranking.json", artifacts["ranking"])
    write_json(out_dir / "failures.json", artifacts["failures"])
    write_jsonl(out_dir / "cache_ledger.jsonl", artifacts["cache_ledger"])
    write_json(out_dir / "report.json", artifacts["report"])
    return artifacts


def comparison_projection(engine: SearchEngine) -> dict[str, Any]:
    current_scores = {
        (record["query_id"], record["doc_id"]) for record in engine.score_records()
    }
    active_ledger = [
        {
            "scope": entry["scope"],
            "entity": entry["entity"],
            "cache_key": entry.get("cache_key"),
            "result_fingerprint": entry.get("result_fingerprint"),
        }
        for entry in engine.cache_ledger_entries(active_only=True)
        if not (
            entry["scope"] == "query_document_score"
            and tuple(entry["entity"].split("|", 1)) not in current_scores
        )
        if not (
            entry["scope"] == "query_ranking"
            and entry["entity"] not in engine.completed_query_ids
        )
    ]
    return {
        "index_fingerprint": engine.index_fingerprint,
        "content_fingerprint": engine.content_fingerprint,
        "config_semantic_fingerprint": engine.config.semantic_fingerprint,
        "stopword_version": engine.stopwords.effective_version,
        "dependencies": engine.dependencies(),
        "scores": engine.score_records(),
        "rankings": [
            engine.query_results[query.query_id]
            for query in engine.ordered_queries
            if query.query_id in engine.query_results
        ],
        "failures": [failure.to_dict() for failure in engine.sorted_failures()],
        "active_cache_evidence": active_ledger,
        "completed_query_ids": list(engine.completed_query_ids),
        "aborted_query_ids": list(engine.aborted_query_ids),
        "pending_query_ids": list(engine.pending_query_ids),
    }


def compare_engines(
    incremental: SearchEngine, full: SearchEngine
) -> dict[str, Any]:
    from .util import sha256_json

    incremental_projection = comparison_projection(incremental)
    full_projection = comparison_projection(full)
    fields: dict[str, bool] = {}
    for key in incremental_projection:
        fields[key] = incremental_projection[key] == full_projection.get(key)
    return {
        "equal": all(fields.values()),
        "fields": {key: fields[key] for key in sorted(fields)},
        "incremental_projection_fingerprint": sha256_json(incremental_projection),
        "full_projection_fingerprint": sha256_json(full_projection),
        "incremental_failures": len(incremental.sorted_failures()),
        "full_failures": len(full.sorted_failures()),
        "incremental_reused_entries": sum(
            1
            for entry in incremental.cache_ledger_entries()
            if entry["decision"] == "reused"
        ),
        "incremental_rebuilt_entries": sum(
            1
            for entry in incremental.cache_ledger_entries()
            if entry["decision"] == "rebuilt"
        ),
        "incremental_invalidated_entries": sum(
            1
            for entry in incremental.cache_ledger_entries()
            if entry["decision"] == "invalidated"
        ),
    }
