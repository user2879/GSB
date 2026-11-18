"""Per-turn three-stage pipeline with cache lookup and snapshot persistence."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .engine_keys import (
    citation_key,
    extraction_key,
    retrieval_key,
    snippets_fingerprint,
)
from .extraction import (
    CitationResult,
    ExtractionResult,
    Snippet,
    assemble_citations,
    extract_snippets,
)
from .models import QueryTurn, RetrievalConfig, turn_sort_key
from .retrieval import BM25Index, Candidate, RetrievalResult
from .rules import RuleTable


def _candidate_from_dict(data: dict) -> Candidate:
    return Candidate(
        doc_id=data["doc_id"],
        score=float(data["score"]),
        matched_terms=tuple(data["matched_terms"]),
    )


def _snippet_from_dict(data: dict) -> Snippet:
    return Snippet(
        doc_id=data["doc_id"],
        source=data["source"],
        sentence_index=int(data["sentence_index"]),
        text=data["text"],
        score=float(data["score"]),
        doc_score=float(data["doc_score"]),
        backfilled=bool(data["backfilled"]),
        backfill_reason=data["backfill_reason"],
    )


def retrieval_from_dict(key: str, data: dict) -> RetrievalResult:
    return RetrievalResult(
        key=key,
        effective_query=data["effective_query"],
        query_terms=tuple(data["query_terms"]),
        candidates=tuple(
            _candidate_from_dict(item) for item in data["candidates"]
        ),
        doc_set=tuple(data["doc_set"]),
    )


def extraction_from_dict(key: str, data: dict) -> ExtractionResult:
    return ExtractionResult(
        key=key,
        snippets=tuple(
            _snippet_from_dict(item) for item in data["snippets"]
        ),
        referenced_doc_set=tuple(data["referenced_doc_set"]),
    )


def citation_from_dict(key: str, data: dict) -> CitationResult:
    failure = None
    if data.get("failure"):
        from .models import Failure

        item = data["failure"]
        failure = Failure(
            code=item["code"],
            scope=item["scope"],
            message=item["message"],
            sort_key=item["sort_key"],
        )
    return CitationResult(
        key=key,
        answer=data["answer"],
        answer_available=bool(data["answer_available"]),
        citations=tuple(
            _snippet_from_dict(item) for item in data["citations"]
        ),
        backfilled=tuple(data["backfilled"]),
        referenced_doc_set=tuple(data["referenced_doc_set"]),
        failure=failure,
    )


@dataclass
class TurnArtifacts:
    retrieval: RetrievalResult | None = None
    extraction: ExtractionResult | None = None
    citation: CitationResult | None = None


@dataclass
class PipelineStats:
    cache_hits: dict[str, int] = field(
        default_factory=lambda: {
            "retrieval": 0,
            "extraction": 0,
            "citation": 0,
        }
    )
    cache_misses: dict[str, int] = field(
        default_factory=lambda: {
            "retrieval": 0,
            "extraction": 0,
            "citation": 0,
        }
    )
    invalidations: dict[str, int] = field(
        default_factory=lambda: {
            "retrieval": 0,
            "extraction": 0,
            "citation": 0,
        }
    )
    stage_skips: dict[str, int] = field(
        default_factory=lambda: {
            "retrieval": 0,
            "extraction": 0,
            "citation": 0,
        }
    )
    retrievals_executed: int = 0


class TurnPipeline:
    """Owns stage caches and runs one turn through the three stages."""

    def __init__(
        self,
        store,
        stats: PipelineStats,
        lock: threading.RLock,
    ):
        self.store = store
        self.stats = stats
        self.lock = lock
        self.memory: dict[str, dict[str, object]] = {
            "retrieval": {},
            "extraction": {},
            "citation": {},
        }

    def _load_or_none(self, stage: str, key: str):
        cached = self.memory[stage].get(key)
        if cached is not None:
            return cached
        raw = self.store.read_stage(stage, key)
        if raw is None:
            return None
        if stage == "retrieval":
            obj = retrieval_from_dict(key, raw)
        elif stage == "extraction":
            obj = extraction_from_dict(key, raw)
        else:
            obj = citation_from_dict(key, raw)
        self.memory[stage][key] = obj
        return obj

    def _store(self, stage: str, key: str, payload: dict, obj) -> None:
        self.store.write_stage(stage, key, payload)
        self.memory[stage][key] = obj

    def run(
        self,
        index: BM25Index,
        config: RetrievalConfig,
        rules: RuleTable,
        template_name: str,
        turn: QueryTurn,
        effective_query: str,
        prior_artifacts: TurnArtifacts | None,
        stale_stages: set[str],
    ) -> tuple[TurnArtifacts, list]:
        scope = turn_sort_key(turn.session_id, turn.turn_no)
        artifacts = TurnArtifacts(
            retrieval=(
                prior_artifacts.retrieval if prior_artifacts is not None else None
            ),
            extraction=(
                prior_artifacts.extraction if prior_artifacts is not None else None
            ),
            citation=(
                prior_artifacts.citation if prior_artifacts is not None else None
            ),
        )
        failures: list = []
        r_key = retrieval_key(
            effective_query=effective_query,
            config=config,
            rules=rules,
            index_fingerprint=index.content_fingerprint,
        )
        with self.lock:
            if "retrieval" not in stale_stages and artifacts.retrieval is not None:
                self.stats.stage_skips["retrieval"] += 1
                retrieval = artifacts.retrieval
            else:
                cached = self._load_or_none("retrieval", r_key)
                if cached is not None:
                    self.stats.cache_hits["retrieval"] += 1
                    retrieval = cached
                else:
                    self.stats.cache_misses["retrieval"] += 1
                    if prior_artifacts is not None and prior_artifacts.retrieval:
                        self.stats.invalidations["retrieval"] += 1
                    retrieval = index.search(
                        r_key, effective_query, config.recall_top_k
                    )
                    self.stats.retrievals_executed += 1
                    self._store(
                        "retrieval", r_key, retrieval.to_dict(), retrieval
                    )
        artifacts.retrieval = retrieval

        e_key = extraction_key(
            parent_key=r_key,
            retrieval_doc_set=retrieval.doc_set,
            query_terms=retrieval.query_terms,
            config=config,
            rules=rules,
        )
        with self.lock:
            if "extraction" not in stale_stages and artifacts.extraction is not None:
                self.stats.stage_skips["extraction"] += 1
                extraction = artifacts.extraction
            else:
                cached = self._load_or_none("extraction", e_key)
                if cached is not None:
                    self.stats.cache_hits["extraction"] += 1
                    extraction = cached
                else:
                    self.stats.cache_misses["extraction"] += 1
                    if prior_artifacts is not None and prior_artifacts.extraction:
                        self.stats.invalidations["extraction"] += 1
                    extraction = extract_snippets(
                        e_key,
                        index,
                        retrieval,
                        config,
                        rules,
                    )
                    self._store(
                        "extraction", e_key, extraction.to_dict(), extraction
                    )
        artifacts.extraction = extraction

        c_key = citation_key(
            parent_key=e_key,
            snippet_fingerprint=snippets_fingerprint(extraction.snippets),
            config=config,
            rules=rules,
            template_name=template_name,
        )
        with self.lock:
            if "citation" not in stale_stages and artifacts.citation is not None:
                self.stats.stage_skips["citation"] += 1
                citation = artifacts.citation
            else:
                cached = self._load_or_none("citation", c_key)
                if cached is not None:
                    self.stats.cache_hits["citation"] += 1
                    citation = cached
                else:
                    self.stats.cache_misses["citation"] += 1
                    if prior_artifacts is not None and prior_artifacts.citation:
                        self.stats.invalidations["citation"] += 1
                    citation = assemble_citations(
                        c_key,
                        extraction,
                        retrieval,
                        index,
                        config,
                        rules,
                        template_name,
                        scope,
                    )
                    self._store("citation", c_key, citation.to_dict(), citation)
                    if citation.failure is not None:
                        failures.append(citation.failure)
        artifacts.citation = citation
        return artifacts, failures
