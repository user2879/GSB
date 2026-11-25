from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from .config import PrototypeConfig
from .errors import (
    E_INDEX_INPUT,
    E_POINT_BUDGET_PENDING,
    E_QUERY_BUDGET_EXCEEDED,
    Failure,
)
from .index import (
    ConcurrencyProbe,
    compute_document_stats,
    content_fingerprint,
    document_posting_terms,
    document_stats_cache_key,
    index_fingerprint,
    posting_record_from_stats,
    term_stats_from_postings,
)
from .loader import EditCommand
from .model import FIELD_ORDER, ControlledVocabulary, Document, QuerySpec, StopwordTable
from .ranking import rank_query, ranking_cache_key
from .scoring import query_document_score_cache_key, score_document
from .util import sha256_json


def _result_fingerprint(value: Any) -> str:
    return sha256_json(value)


def _term_stats_cache_key(term: str, postings: dict[str, dict[str, Any]]) -> str:
    return sha256_json(
        {
            "term": term,
            "postings": [
                postings[doc_id] for doc_id in sorted(postings)
            ],
        }
    )


class SearchEngine:
    """Incremental deterministic index, scoring, ranking and scheduler."""

    def __init__(
        self,
        config: PrototypeConfig,
        stopwords: StopwordTable,
        vocabulary: ControlledVocabulary,
        documents: Iterable[Document],
        queries: Iterable[QuerySpec],
        *,
        failures: Iterable[Failure] = (),
        max_workers_override: int | None = None,
    ):
        self.config = config
        self.stopwords = stopwords
        self.vocabulary = vocabulary
        self.documents: dict[str, Document] = {
            document.doc_id: document for document in documents
        }
        self.queries: dict[str, QuerySpec] = {
            query.query_id: query for query in queries
        }
        self.failures: list[Failure] = list(failures)
        self.max_workers_override = max_workers_override

        self.document_stats: dict[str, dict[str, Any]] = {}
        self.postings: dict[str, dict[str, dict[str, Any]]] = {}
        self.term_stats: dict[str, dict[str, Any]] = {}
        self.config_fingerprint = config.semantic_fingerprint
        self.stopword_fingerprint = stopwords.content_fingerprint
        self.content_fingerprint = content_fingerprint(())
        self.index_fingerprint = index_fingerprint(
            config, stopwords, self.content_fingerprint
        )

        self._document_cache: dict[str, dict[str, Any]] = {}
        self._term_cache: dict[str, dict[str, Any]] = {}
        self._score_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._ranking_cache: dict[str, dict[str, Any]] = {}
        self._cache_ledger: dict[tuple[str, str], dict[str, Any]] = {}
        self._changed_doc_ids: set[str] = set(self.documents)
        self._deleted_doc_ids: set[str] = set()
        self._queries_dirty = False
        self._edit_count = 0

        self.query_progress: dict[str, dict[str, Any]] = {}
        self.query_results: dict[str, dict[str, Any]] = {}
        self.completed_query_ids: list[str] = []
        self.aborted_query_ids: list[str] = []
        self.pending_query_ids: list[str] = []
        self.batch_index = 0
        self.batch_consumed_points: dict[int, int] = {}
        self.total_consumed_points = 0
        self.global_version = 0
        self.concurrency_observations: list[dict[str, Any]] = []
        self.materialization_count = 0
        self.last_materialization: dict[str, Any] = {}
        self.input_fingerprint = ""
        self.scheduler_stalled = False

    @property
    def ordered_documents(self) -> list[Document]:
        return [self.documents[key] for key in sorted(self.documents)]

    @property
    def ordered_queries(self) -> list[QuerySpec]:
        return [self.queries[key] for key in sorted(self.queries)]

    @property
    def workers(self) -> int:
        if self.max_workers_override is not None:
            return max(1, min(self.max_workers_override, self.config.max_workers))
        return max(1, self.config.max_workers)

    def _record_decision(
        self,
        scope: str,
        entity: str,
        decision: str,
        cache_key: str | None,
        previous_cache_key: str | None,
        basis: str,
        result: Any = None,
        **extra: Any,
    ) -> None:
        entry: dict[str, Any] = {
            "scope": scope,
            "entity": entity,
            "decision": decision,
            "cache_key": cache_key,
            "previous_cache_key": previous_cache_key,
            "basis": basis,
            "active": decision not in {"invalidated", "removed"},
        }
        if result is not None:
            entry["result_fingerprint"] = _result_fingerprint(result)
        entry.update(extra)
        self._cache_ledger[(scope, entity)] = entry

    def _run_parallel(
        self,
        tasks: list[tuple[str, Any]],
        function: Any,
        *,
        stage: str,
    ) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
        workers = min(self.workers, max(1, len(tasks)))
        probe = ConcurrencyProbe(maximum_allowed=self.config.max_workers)
        if workers <= 1 or len(tasks) <= 1:
            results = [
                (key, probe.run(key, lambda payload=payload: function(payload)))
                for key, payload in tasks
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix=f"search-{stage}"
            ) as pool:
                futures = [
                    pool.submit(
                        probe.run,
                        key,
                        lambda payload=payload: function(payload),
                    )
                    for key, payload in tasks
                ]
                results = [
                    (tasks[index][0], future.result())
                    for index, future in enumerate(futures)
                ]
        observation = probe.to_dict()
        observation["stage"] = stage
        self.concurrency_observations.append(observation)
        return results, observation

    def _compute_changed_document_stats(self) -> tuple[set[str], set[str]]:
        affected_terms: set[str] = set()
        affected_doc_ids = set(self._changed_doc_ids) | set(self._deleted_doc_ids)
        for doc_id in sorted(affected_doc_ids):
            old_stats = self.document_stats.get(doc_id)
            if old_stats is not None:
                affected_terms.update(document_posting_terms(old_stats))

        for doc_id in sorted(affected_doc_ids):
            for term in document_posting_terms(self.document_stats.get(doc_id, {})):
                term_postings = self.postings.get(term)
                if term_postings is not None:
                    term_postings.pop(doc_id, None)
                    if not term_postings:
                        self.postings.pop(term, None)

        for doc_id in sorted(self._deleted_doc_ids):
            self.document_stats.pop(doc_id, None)
            old_entry = self._document_cache.pop(doc_id, None)
            if old_entry is not None:
                self._record_decision(
                    "document_field_stats",
                    doc_id,
                    "invalidated",
                    None,
                    old_entry.get("key"),
                    "document_deleted",
                )

        tasks: list[tuple[str, Document]] = []
        cached_keys: dict[str, str] = {}
        reused_doc_ids: set[str] = set()
        for doc_id in sorted(self._changed_doc_ids):
            document = self.documents.get(doc_id)
            if document is None:
                continue
            key = document_stats_cache_key(document, self.config, self.stopwords)
            previous = self._document_cache.get(doc_id)
            if previous is not None and previous.get("key") == key:
                previous_stats = self.document_stats.get(doc_id)
                if previous_stats is not None:
                    cached_keys[doc_id] = key
                    reused_doc_ids.add(doc_id)
                    self._record_decision(
                        "document_field_stats",
                        doc_id,
                        "reused",
                        key,
                        previous.get("key"),
                        "content_and_stopword_dependency_unchanged",
                        previous_stats,
                    )
                    continue
            tasks.append((f"document:{doc_id}:field_stats", document))

        def compute(document: Document) -> dict[str, Any]:
            return compute_document_stats(document, self.config, self.stopwords)

        results, _ = self._run_parallel(tasks, compute, stage="document_field_stats")
        for (_, document), (_, stats) in zip(tasks, results):
            doc_id = document.doc_id
            key = document_stats_cache_key(document, self.config, self.stopwords)
            previous = self._document_cache.get(doc_id)
            self.document_stats[doc_id] = stats
            self._document_cache[doc_id] = {"key": key, "stats": stats}
            affected_terms.update(document_posting_terms(stats))
            basis = "blank_state" if previous is None else "content_or_stopword_dependency_changed"
            self._record_decision(
                "document_field_stats",
                doc_id,
                "rebuilt",
                key,
                None if previous is None else previous.get("key"),
                basis,
                stats,
            )

        for doc_id in sorted(self._changed_doc_ids):
            if doc_id in reused_doc_ids and doc_id not in self.document_stats:
                stats = self._document_cache[doc_id]["stats"]
                self.document_stats[doc_id] = stats

        for doc_id in sorted(self._changed_doc_ids):
            stats = self.document_stats.get(doc_id)
            if stats is None:
                continue
            for term in document_posting_terms(stats):
                self.postings.setdefault(term, {})[doc_id] = posting_record_from_stats(
                    stats, term
                )
                affected_terms.add(term)
        return affected_terms, affected_doc_ids

    def _refresh_affected_terms(self, affected_terms: set[str]) -> None:
        for term in sorted(affected_terms):
            term_postings = self.postings.get(term)
            previous = self._term_cache.get(term)
            if not term_postings:
                self.term_stats.pop(term, None)
                self._term_cache.pop(term, None)
                if previous is not None:
                    self._record_decision(
                        "term_stats",
                        term,
                        "invalidated",
                        None,
                        previous.get("key"),
                        "term_has_no_documents",
                    )
                continue
            key = _term_stats_cache_key(term, term_postings)
            if previous is not None and previous.get("key") == key:
                stats = previous.get("stats")
                if stats is None:
                    stats = term_stats_from_postings(term, term_postings)
                self.term_stats[term] = stats
                self._record_decision(
                    "term_stats",
                    term,
                    "reused",
                    key,
                    previous.get("key"),
                    "posting_dependency_unchanged",
                    stats,
                )
                continue
            stats = term_stats_from_postings(term, term_postings)
            self.term_stats[term] = stats
            self._term_cache[term] = {"key": key, "stats": stats}
            self._record_decision(
                "term_stats",
                term,
                "rebuilt",
                key,
                None if previous is None else previous.get("key"),
                "blank_state" if previous is None else "posting_dependency_changed",
                stats,
            )

    def _record_unaffected_reuse(
        self,
        affected_terms: set[str],
        affected_doc_ids: set[str],
    ) -> None:
        for doc_id in sorted(self.documents):
            if doc_id in affected_doc_ids:
                continue
            entry = self._document_cache.get(doc_id)
            stats = self.document_stats.get(doc_id)
            if entry is None or stats is None:
                continue
            current_key = document_stats_cache_key(
                self.documents[doc_id], self.config, self.stopwords
            )
            previous_key = entry.get("key")
            entry["key"] = current_key
            self._record_decision(
                "document_field_stats",
                doc_id,
                "reused",
                current_key,
                previous_key,
                "document_exclusions_and_field_config_unchanged",
                stats,
            )
        for term in sorted(self.term_stats):
            if term in affected_terms:
                continue
            entry = self._term_cache.get(term)
            stats = self.term_stats.get(term)
            if entry is None or stats is None:
                continue
            current_key = _term_stats_cache_key(term, self.postings.get(term, {}))
            previous_key = entry.get("key")
            entry["key"] = current_key
            self._record_decision(
                "term_stats",
                term,
                "reused",
                current_key,
                previous_key,
                "posting_dependency_unchanged",
                stats,
            )

    def materialize(self) -> None:
        affected_terms, affected_doc_ids = self._compute_changed_document_stats()
        self._refresh_affected_terms(affected_terms)
        self._record_unaffected_reuse(affected_terms, affected_doc_ids)
        self.content_fingerprint = content_fingerprint(self.ordered_documents)
        self.config_fingerprint = self.config.semantic_fingerprint
        self.stopword_fingerprint = self.stopwords.content_fingerprint
        self.index_fingerprint = index_fingerprint(
            self.config, self.stopwords, self.content_fingerprint
        )
        self.materialization_count += 1
        self.last_materialization = {
            "changed_doc_ids": sorted(self._changed_doc_ids),
            "deleted_doc_ids": sorted(self._deleted_doc_ids),
            "affected_terms": sorted(affected_terms),
            "affected_doc_ids": sorted(affected_doc_ids),
            "materialization": self.materialization_count,
        }
        self._changed_doc_ids.clear()
        self._deleted_doc_ids.clear()

        if self._queries_dirty and (
            self.completed_query_ids or self.query_progress or self.pending_query_ids
        ):
            self._reset_query_execution_for_revalidation()
        self._queries_dirty = False

    def _reset_query_execution_for_revalidation(self) -> None:
        self.query_progress = {}
        self.query_results = {}
        self.completed_query_ids = []
        self.aborted_query_ids = []
        self.pending_query_ids = []
        self.batch_index = 0
        self.batch_consumed_points = {}
        self.total_consumed_points = 0

    def apply_edit(self, command: EditCommand) -> bool:
        if self._edit_count == 0:
            self.failures = [
                failure
                for failure in self.failures
                if failure.code not in {E_POINT_BUDGET_PENDING, E_QUERY_BUDGET_EXCEEDED}
            ]
        op = command.op
        payload = command.payload
        if op == "append_doc":
            document: Document = payload["doc"]
            if document.doc_id in self.documents:
                self.failures.append(
                    Failure(
                        code=E_INDEX_INPUT,
                        source="edits",
                        line=command.line,
                        entity_id=document.doc_id,
                        message="append_doc 的文档标识已存在",
                        detail_code="EDIT_APPEND_DUPLICATE",
                    )
                )
                return False
            self.documents[document.doc_id] = document
            self._changed_doc_ids.add(document.doc_id)
        elif op == "replace_doc":
            document = payload["doc"]
            if document.doc_id not in self.documents:
                self.failures.append(
                    Failure(
                        code=E_INDEX_INPUT,
                        source="edits",
                        line=command.line,
                        entity_id=document.doc_id,
                        message="replace_doc 的目标文档不存在",
                        detail_code="EDIT_REPLACE_MISSING",
                    )
                )
                return False
            self.documents[document.doc_id] = document
            self._changed_doc_ids.add(document.doc_id)
        elif op == "delete_doc":
            doc_id = payload["doc_id"]
            if doc_id not in self.documents:
                self.failures.append(
                    Failure(
                        code=E_INDEX_INPUT,
                        source="edits",
                        line=command.line,
                        entity_id=doc_id,
                        message="delete_doc 的目标文档不存在",
                        detail_code="EDIT_DELETE_MISSING",
                    )
                )
                return False
            del self.documents[doc_id]
            self._changed_doc_ids.discard(doc_id)
            self._deleted_doc_ids.add(doc_id)
            for key in sorted(self._score_cache):
                if key[1] != doc_id:
                    continue
                previous = self._score_cache.pop(key)
                self._record_decision(
                    "query_document_score",
                    f"{key[0]}|{key[1]}",
                    "invalidated",
                    None,
                    previous.get("key"),
                    "document_deleted",
                    query_id=key[0],
                    doc_id=key[1],
                )
        elif op == "set_field_weights":
            self.config = self.config.with_field_weights(payload["weights"])
        elif op == "set_bm25":
            self.config = self.config.with_bm25(payload["k1"], payload["b"])
        elif op == "set_fixed":
            self.config = self.config.with_fixed(
                payload["decimal_places"], payload["quantum"]
            )
        elif op == "set_top_k":
            self.config = self.config.with_top_k(payload["top_k"])
        elif op in {"add_stopword", "remove_stopword"}:
            field_name = payload["field"]
            term = payload["term"]
            old_stopwords = self.stopwords
            self.stopwords = (
                old_stopwords.add(field_name, term)
                if op == "add_stopword"
                else old_stopwords.remove(field_name, term)
            )
            fields = list(FIELD_ORDER) if field_name == "*" else [field_name]
            for document in self.documents.values():
                if any(term in document.fields.get(name, ()) for name in fields):
                    self._changed_doc_ids.add(document.doc_id)
        else:
            self.failures.append(
                Failure(
                    code=E_INDEX_INPUT,
                    source="edits",
                    line=command.line,
                    entity_id="",
                    message=f"未处理的编辑命令: {op}",
                    detail_code="EDIT_UNHANDLED",
                )
            )
            return False
        self._edit_count += 1
        self._queries_dirty = True
        return True

    def apply_edits(self, commands: Iterable[EditCommand]) -> None:
        for command in commands:
            self.apply_edit(command)

    def _initialize_query_progress(self) -> None:
        ordered_doc_ids = sorted(self.documents)
        for query in self.ordered_queries:
            self.query_progress.setdefault(
                query.query_id,
                {
                    "remaining_doc_ids": list(ordered_doc_ids),
                    "consumed_points": 0,
                    "scores": {},
                    "score_cache_keys": {},
                },
            )

    def _score_query_documents(
        self,
        query: QuerySpec,
        doc_ids: list[str],
    ) -> dict[str, Any]:
        tasks: list[tuple[str, str]] = []
        for doc_id in sorted(doc_ids):
            tasks.append((f"query:{query.query_id}:score:{doc_id}", doc_id))

        def compute(doc_id: str) -> dict[str, Any]:
            return score_document(
                query,
                self.document_stats[doc_id],
                self.term_stats,
                self.config,
                len(self.documents),
            )

        results, _ = self._run_parallel(tasks, compute, stage="query_scoring")
        by_doc: dict[str, Any] = {}
        for (_, doc_id), (_, record) in zip(tasks, results):
            by_doc[doc_id] = record
        return by_doc

    def _get_or_compute_scores(
        self,
        query: QuerySpec,
        doc_ids: list[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        progress = self.query_progress[query.query_id]
        records: dict[str, dict[str, Any]] = {}
        keys: dict[str, str] = {}
        missing: list[str] = []
        for doc_id in sorted(doc_ids):
            document_stats = self.document_stats[doc_id]
            key = query_document_score_cache_key(
                query, document_stats, self.term_stats, self.config
            )
            previous = self._score_cache.get((query.query_id, doc_id))
            if previous is not None and previous.get("key") == key:
                records[doc_id] = previous["record"]
                keys[doc_id] = key
                self._record_decision(
                    "query_document_score",
                    f"{query.query_id}|{doc_id}",
                    "reused",
                    key,
                    previous.get("key"),
                    "query_document_dependency_unchanged",
                    previous["record"],
                    query_id=query.query_id,
                    doc_id=doc_id,
                )
            else:
                missing.append(doc_id)

        if missing:
            computed = self._score_query_documents(query, missing)
            for doc_id in missing:
                record = computed[doc_id]
                key = query_document_score_cache_key(
                    query,
                    self.document_stats[doc_id],
                    self.term_stats,
                    self.config,
                )
                previous = self._score_cache.get((query.query_id, doc_id))
                self._score_cache[(query.query_id, doc_id)] = {
                    "key": key,
                    "record": record,
                }
                records[doc_id] = record
                keys[doc_id] = key
                self._record_decision(
                    "query_document_score",
                    f"{query.query_id}|{doc_id}",
                    "rebuilt",
                    key,
                    None if previous is None else previous.get("key"),
                    "blank_state" if previous is None else "query_document_dependency_changed",
                    record,
                    query_id=query.query_id,
                    doc_id=doc_id,
                )
        progress["scores"].update(records)
        progress["score_cache_keys"].update(keys)
        return records, keys

    def _finalize_query(self, query: QuerySpec) -> None:
        progress = self.query_progress[query.query_id]
        scores = [progress["scores"][doc_id] for doc_id in sorted(progress["scores"])]
        key = ranking_cache_key(
            query,
            scores,
            self.config,
            progress["score_cache_keys"],
        )
        previous = self._ranking_cache.get(query.query_id)
        if previous is not None and previous.get("key") == key:
            ranking = previous["result"]
            decision = "reused"
            basis = "ranking_dependency_unchanged"
        else:
            ranking = rank_query(query, scores, self.config)
            self._ranking_cache[query.query_id] = {"key": key, "result": ranking}
            decision = "rebuilt"
            basis = "blank_state" if previous is None else "ranking_dependency_changed"
        self.query_results[query.query_id] = ranking
        self._record_decision(
            "query_ranking",
            query.query_id,
            decision,
            key,
            None if previous is None else previous.get("key"),
            basis,
            ranking,
            query_id=query.query_id,
        )
        if query.query_id not in self.completed_query_ids:
            self.completed_query_ids.append(query.query_id)
            self.completed_query_ids.sort()
        self.pending_query_ids = [
            qid for qid in self.pending_query_ids if qid != query.query_id
        ]

    def _abort_query(
        self,
        query: QuerySpec,
        *,
        batch_index: int,
        consumed: int,
        quota: int,
    ) -> None:
        if query.query_id not in self.aborted_query_ids:
            self.aborted_query_ids.append(query.query_id)
            self.aborted_query_ids.sort()
        self.pending_query_ids = [
            qid for qid in self.pending_query_ids if qid != query.query_id
        ]
        self.failures.append(
            Failure(
                code=E_QUERY_BUDGET_EXCEEDED,
                source="scheduler",
                line=batch_index,
                entity_id=query.query_id,
                message="查询累计打分点数达到单查询配额，查询中止",
                detail_code="QUERY_QUOTA_EXHAUSTED",
                context={
                    "consumed_points": consumed,
                    "per_query_point_quota": quota,
                    "remaining_documents": len(
                        self.query_progress[query.query_id]["remaining_doc_ids"]
                    ),
                },
            )
        )

    def _mark_pending(
        self,
        query: QuerySpec,
        *,
        batch_index: int,
        consumed: int,
        batch_budget: int,
    ) -> None:
        if query.query_id not in self.pending_query_ids:
            self.pending_query_ids.append(query.query_id)
            self.pending_query_ids.sort()
        self.failures.append(
            Failure(
                code=E_POINT_BUDGET_PENDING,
                source="scheduler",
                line=batch_index,
                entity_id=query.query_id,
                message="本批打分点数耗尽，查询保留至下一批优先处理",
                detail_code="BATCH_BUDGET_EXHAUSTED",
                context={
                    "consumed_points": consumed,
                    "batch_point_budget": batch_budget,
                    "remaining_documents": len(
                        self.query_progress[query.query_id]["remaining_doc_ids"]
                    ),
                },
            )
        )

    def run_batches(
        self,
        *,
        max_batches: int | None = None,
        snapshot_callback: Any = None,
        crash_after_batch: int | None = None,
    ) -> None:
        self._initialize_query_progress()
        self.scheduler_stalled = False
        batches_run = 0
        while True:
            unfinished = [
                query.query_id
                for query in self.ordered_queries
                if query.query_id not in self.completed_query_ids
                and query.query_id not in self.aborted_query_ids
            ]
            if not unfinished:
                break
            if max_batches is not None and batches_run >= max_batches:
                break

            batch_index = self.batch_index
            remaining_budget = self.config.batch_point_budget
            consumed_this_batch = 0
            order = sorted(set(self.pending_query_ids)) + [
                qid for qid in sorted(unfinished) if qid not in self.pending_query_ids
            ]
            processed: set[str] = set()
            budget_exhausted = False
            terminal_events = 0
            for query_id in order:
                query = self.queries[query_id]
                if query_id in self.completed_query_ids or query_id in self.aborted_query_ids:
                    continue
                progress = self.query_progress[query_id]
                term_count = max(1, len(query.sorted_unique_terms()))
                quota = self.config.per_query_point_quota

                while progress["remaining_doc_ids"]:
                    if remaining_budget <= 0:
                        budget_exhausted = True
                        break
                    if progress["consumed_points"] >= quota:
                        break
                    docs_by_quota = (
                        quota - progress["consumed_points"]
                    ) // term_count
                    docs_by_budget = remaining_budget // term_count
                    chunk_size = min(
                        docs_by_quota,
                        docs_by_budget,
                        len(progress["remaining_doc_ids"]),
                    )
                    if chunk_size <= 0:
                        break
                    chunk = progress["remaining_doc_ids"][:chunk_size]
                    self._get_or_compute_scores(query, chunk)
                    points = chunk_size * term_count
                    progress["remaining_doc_ids"] = progress[
                        "remaining_doc_ids"
                    ][chunk_size:]
                    progress["consumed_points"] += points
                    remaining_budget -= points
                    consumed_this_batch += points
                    self.total_consumed_points += points

                processed.add(query_id)
                if not progress["remaining_doc_ids"]:
                    self._finalize_query(query)
                    terminal_events += 1
                elif (
                    progress["consumed_points"] >= quota
                    or quota - progress["consumed_points"] < term_count
                ):
                    self._abort_query(
                        query,
                        batch_index=batch_index,
                        consumed=progress["consumed_points"],
                        quota=quota,
                    )
                    terminal_events += 1
                else:
                    self._mark_pending(
                        query,
                        batch_index=batch_index,
                        consumed=progress["consumed_points"],
                        batch_budget=self.config.batch_point_budget,
                    )
                    budget_exhausted = True

                if budget_exhausted:
                    break

            unprocessed = [
                qid
                for qid in order
                if qid not in processed
                and qid not in self.completed_query_ids
                and qid not in self.aborted_query_ids
                and qid not in self.pending_query_ids
            ]
            for query_id in unprocessed:
                query = self.queries[query_id]
                progress = self.query_progress[query_id]
                self._mark_pending(
                    query,
                    batch_index=batch_index,
                    consumed=progress["consumed_points"],
                    batch_budget=self.config.batch_point_budget,
                )

            self.batch_consumed_points[batch_index] = consumed_this_batch
            self.batch_index += 1
            batches_run += 1
            self.global_version += 1
            if snapshot_callback is not None:
                snapshot_callback(self)
            if crash_after_batch is not None and batches_run >= crash_after_batch:
                os._exit(137)
            if consumed_this_batch == 0 and terminal_events == 0:
                self.scheduler_stalled = True
                break

    def dependencies(self) -> dict[str, Any]:
        queries_to_terms = {
            query.query_id: list(query.sorted_unique_terms())
            for query in self.ordered_queries
        }
        terms_to_queries: dict[str, list[str]] = {}
        for query_id, terms in queries_to_terms.items():
            for term in terms:
                terms_to_queries.setdefault(term, []).append(query_id)
        terms_to_documents = {
            term: sorted(doc_ids) for term, doc_ids in sorted(self.postings.items())
        }
        return {
            "query_to_terms": queries_to_terms,
            "term_to_queries": {
                term: sorted(query_ids)
                for term, query_ids in sorted(terms_to_queries.items())
            },
            "term_to_documents": terms_to_documents,
            "document_to_terms": {
                doc_id: document_posting_terms(self.document_stats[doc_id])
                for doc_id in sorted(self.document_stats)
            },
        }

    @property
    def completed_query_cursor(self) -> str | None:
        return self.completed_query_ids[-1] if self.completed_query_ids else None

    def sorted_failures(self) -> list[Failure]:
        unique: dict[str, Failure] = {}
        for failure in self.failures:
            unique[sha256_json(failure.to_dict())] = failure
        return sorted(unique.values(), key=Failure.sort_key)

    def business_code_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self.sorted_failures():
            counts[failure.code] = counts.get(failure.code, 0) + 1
        return {key: counts[key] for key in sorted(counts)}

    def cache_ledger_entries(
        self, *, active_only: bool = False
    ) -> list[dict[str, Any]]:
        entries = [
            self._cache_ledger[key]
            for key in sorted(self._cache_ledger)
            if not active_only or self._cache_ledger[key].get("active", False)
        ]
        return entries

    def cache_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for entry in self._cache_ledger.values():
            scope = entry["scope"]
            scope_counts = counts.setdefault(scope, {})
            decision = entry["decision"]
            scope_counts[decision] = scope_counts.get(decision, 0) + 1
        return {
            scope: {key: counts[scope][key] for key in sorted(counts[scope])}
            for scope in sorted(counts)
        }

    def query_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for query in self.ordered_queries:
            ranking = self.query_results.get(query.query_id)
            progress = self.query_progress.get(query.query_id)
            if ranking is not None:
                summaries.append(
                    {
                        "query_id": query.query_id,
                        "status": "completed",
                        "top_k": ranking["top_k"],
                        "boundary_document_count": len(
                            ranking["boundary"]["documents"]
                        ),
                        "order_flip_count": ranking["order_flip"][
                            "pair_flip_count"
                        ],
                        "top_k_symmetric_difference": ranking["order_flip"][
                            "top_k_symmetric_difference"
                        ],
                    }
                )
            elif query.query_id in self.aborted_query_ids:
                summaries.append(
                    {
                        "query_id": query.query_id,
                        "status": "aborted_budget",
                        "top_k": [],
                        "boundary_document_count": 0,
                        "order_flip_count": 0,
                        "consumed_points": 0
                        if progress is None
                        else progress["consumed_points"],
                    }
                )
            else:
                summaries.append(
                    {
                        "query_id": query.query_id,
                        "status": "pending",
                        "top_k": [],
                        "boundary_document_count": 0,
                        "order_flip_count": 0,
                        "consumed_points": 0
                        if progress is None
                        else progress["consumed_points"],
                        "remaining_documents": len(
                            sorted(self.documents)
                            if progress is None
                            else progress["remaining_doc_ids"]
                        ),
                    }
                )
        return summaries

    def report(self) -> dict[str, Any]:
        summaries = self.query_summaries()
        ledger = self.cache_ledger_entries()
        active_ledger = self.cache_ledger_entries(active_only=True)
        concurrency_max = max(
            (
                int(observation.get("max_active_observed", 0))
                for observation in self.concurrency_observations
            ),
            default=0,
        )
        return {
            "schema": "search-prototype-report-v1",
            "global_version": self.global_version,
            "config_hash": self.config.full_fingerprint,
            "semantic_config_hash": self.config.semantic_fingerprint,
            "stopword_version": self.stopwords.effective_version,
            "index_fingerprint": self.index_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "document_count": len(self.documents),
            "query_count": len(self.queries),
            "completed_query_cursor": self.completed_query_cursor,
            "completed_query_ids": list(self.completed_query_ids),
            "aborted_query_ids": list(self.aborted_query_ids),
            "pending_query_ids": list(self.pending_query_ids),
            "batch_index": self.batch_index,
            "batch_consumed_points": {
                str(key): self.batch_consumed_points[key]
                for key in sorted(self.batch_consumed_points)
            },
            "total_consumed_points": self.total_consumed_points,
            "business_code_counts": self.business_code_counts(),
            "queries": summaries,
            "boundary_document_count": sum(
                summary["boundary_document_count"] for summary in summaries
            ),
            "order_flip_count": sum(
                summary["order_flip_count"] for summary in summaries
            ),
            "cache_decisions": self.cache_counts(),
            "cache_entry_counts": {
                "all": len(ledger),
                "active": len(active_ledger),
                "reused": sum(1 for item in ledger if item["decision"] == "reused"),
                "rebuilt": sum(1 for item in ledger if item["decision"] == "rebuilt"),
                "invalidated": sum(
                    1 for item in ledger if item["decision"] == "invalidated"
                ),
            },
            "concurrency": {
                "configured_max_workers": self.config.max_workers,
                "effective_workers": self.workers,
                "max_active_observed": concurrency_max,
                "observations": self.concurrency_observations,
            },
            "last_materialization": self.last_materialization,
            "scheduler_stalled": self.scheduler_stalled,
            "scheduler_stall_reason": (
                "batch_point_budget_smaller_than_remaining_query_work"
                if self.scheduler_stalled
                else None
            ),
        }

    def ranking_index(self) -> dict[str, Any]:
        return {
            "schema": "query-ranking-v1",
            "index_fingerprint": self.index_fingerprint,
            "config_hash": self.config.full_fingerprint,
            "stopword_version": self.stopwords.effective_version,
            "queries": [
                self.query_results[query.query_id]
                for query in self.ordered_queries
                if query.query_id in self.query_results
            ],
            "pending_query_ids": list(self.pending_query_ids),
            "aborted_query_ids": list(self.aborted_query_ids),
            "dependencies": self.dependencies(),
        }

    def score_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for query in self.ordered_queries:
            progress = self.query_progress.get(query.query_id)
            if progress is None:
                continue
            for doc_id in sorted(progress["scores"]):
                records.append(progress["scores"][doc_id])
        return records

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "search-state-v1",
            "config": self.config.to_dict(),
            "stopwords": self.stopwords.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "documents": [document.to_dict() for document in self.ordered_documents],
            "queries": [query.to_dict() for query in self.ordered_queries],
            "failures": [failure.to_dict() for failure in self.sorted_failures()],
            "document_stats": [
                self.document_stats[doc_id] for doc_id in sorted(self.document_stats)
            ],
            "postings": [
                {
                    "term": term,
                    "documents": [
                        self.postings[term][doc_id]
                        for doc_id in sorted(self.postings[term])
                    ],
                }
                for term in sorted(self.postings)
            ],
            "term_stats": [
                self.term_stats[term] for term in sorted(self.term_stats)
            ],
            "document_cache": {
                doc_id: {"key": entry["key"]}
                for doc_id, entry in sorted(self._document_cache.items())
            },
            "term_cache": {
                term: {"key": entry["key"]}
                for term, entry in sorted(self._term_cache.items())
            },
            "score_cache": [
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "key": self._score_cache[(query_id, doc_id)]["key"],
                    "record": self._score_cache[(query_id, doc_id)]["record"],
                }
                for query_id, doc_id in sorted(self._score_cache)
            ],
            "ranking_cache": [
                {
                    "query_id": query_id,
                    "key": self._ranking_cache[query_id]["key"],
                    "result": self._ranking_cache[query_id]["result"],
                }
                for query_id in sorted(self._ranking_cache)
            ],
            "cache_ledger": self.cache_ledger_entries(),
            "query_progress": [
                {"query_id": query_id, **self.query_progress[query_id]}
                for query_id in sorted(self.query_progress)
            ],
            "query_results": [
                {"query_id": query_id, "result": self.query_results[query_id]}
                for query_id in sorted(self.query_results)
            ],
            "completed_query_ids": list(self.completed_query_ids),
            "aborted_query_ids": list(self.aborted_query_ids),
            "pending_query_ids": list(self.pending_query_ids),
            "batch_index": self.batch_index,
            "batch_consumed_points": {
                str(key): value for key, value in sorted(self.batch_consumed_points.items())
            },
            "total_consumed_points": self.total_consumed_points,
            "global_version": self.global_version,
            "edit_count": self._edit_count,
            "concurrency_observations": self.concurrency_observations,
            "materialization_count": self.materialization_count,
            "last_materialization": self.last_materialization,
            "content_fingerprint": self.content_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "scheduler_stalled": self.scheduler_stalled,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SearchEngine":
        from .config import validate_config

        engine = cls(
            validate_config(state["config"]),
            StopwordTable.from_dict(state["stopwords"]),
            ControlledVocabulary.from_dict(state["vocabulary"]),
            [Document.from_dict(value) for value in state.get("documents", [])],
            [QuerySpec.from_dict(value) for value in state.get("queries", [])],
        )
        engine.failures = [
            Failure(
                code=value["code"],
                source=value["source"],
                line=int(value["line"]),
                entity_id=value["entity_id"],
                message=value["message"],
                detail_code=value.get("detail_code", ""),
                context=value.get("context"),
            )
            for value in state.get("failures", [])
        ]
        engine.document_stats = {
            value["doc_id"]: value for value in state.get("document_stats", [])
        }
        engine.postings = {
            item["term"]: {
                posting["doc_id"]: posting for posting in item["documents"]
            }
            for item in state.get("postings", [])
        }
        engine.term_stats = {
            value["term"]: value for value in state.get("term_stats", [])
        }
        for doc_id, value in state.get("document_cache", {}).items():
            stats = engine.document_stats.get(doc_id)
            if stats is not None:
                engine._document_cache[doc_id] = {"key": value["key"], "stats": stats}
        for term, value in state.get("term_cache", {}).items():
            stats = engine.term_stats.get(term)
            if stats is not None:
                engine._term_cache[term] = {"key": value["key"], "stats": stats}
        for item in state.get("score_cache", []):
            engine._score_cache[(item["query_id"], item["doc_id"])] = {
                "key": item["key"],
                "record": item["record"],
            }
        for item in state.get("ranking_cache", []):
            engine._ranking_cache[item["query_id"]] = {
                "key": item["key"],
                "result": item["result"],
            }
        engine._cache_ledger = {
            (item["scope"], item["entity"]): item
            for item in state.get("cache_ledger", [])
        }
        engine.query_progress = {
            item["query_id"]: {
                "remaining_doc_ids": item["remaining_doc_ids"],
                "consumed_points": item["consumed_points"],
                "scores": item["scores"],
                "score_cache_keys": item["score_cache_keys"],
            }
            for item in state.get("query_progress", [])
        }
        engine.query_results = {
            item["query_id"]: item["result"]
            for item in state.get("query_results", [])
        }
        engine.completed_query_ids = list(state.get("completed_query_ids", []))
        engine.aborted_query_ids = list(state.get("aborted_query_ids", []))
        engine.pending_query_ids = list(state.get("pending_query_ids", []))
        engine.batch_index = int(state.get("batch_index", 0))
        engine.batch_consumed_points = {
            int(key): int(value)
            for key, value in state.get("batch_consumed_points", {}).items()
        }
        engine.total_consumed_points = int(state.get("total_consumed_points", 0))
        engine.global_version = int(state.get("global_version", 0))
        engine._edit_count = int(state.get("edit_count", 0))
        engine.concurrency_observations = list(
            state.get("concurrency_observations", [])
        )
        engine.materialization_count = int(state.get("materialization_count", 0))
        engine.last_materialization = dict(state.get("last_materialization", {}))
        engine.input_fingerprint = state.get("input_fingerprint", "")
        engine.scheduler_stalled = bool(state.get("scheduler_stalled", False))
        engine.content_fingerprint = state.get(
            "content_fingerprint", content_fingerprint(engine.ordered_documents)
        )
        engine.index_fingerprint = state.get(
            "index_fingerprint",
            index_fingerprint(
                engine.config, engine.stopwords, engine.content_fingerprint
            ),
        )
        engine._changed_doc_ids.clear()
        engine._deleted_doc_ids.clear()
        engine._queries_dirty = False
        return engine
