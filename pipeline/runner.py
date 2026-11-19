from __future__ import annotations

import copy
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from .config import load_config, sha256_json, with_overrides
from .data import (
    Document,
    NormalizedState,
    Qrel,
    Query,
    TranslationResolver,
    apply_normalization_batch,
    input_hashes,
    normalize_raw_item,
    read_raw_sources,
    sorted_failures,
    state_counts,
    valid_translation_hash,
)
from .errors import Failure as FailureRecord, PipelineError, SnapshotMismatchError
from .evaluation import aggregate_metrics, evaluate_query_result
from .index import BM25Index
from .injections import Injection, load_injections
from .normalization import detect_language
from .retrieval import budget_exhausted_result, execute_query, query_plan_digest, retrieval_budget_plan
from .snapshot import SnapshotStore, atomic_write_json, utc_now


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _preview_translation_hash(raw_items: list, config: dict) -> str:
    preview = NormalizedState()
    for item in raw_items:
        if item.source == "translations":
            normalize_raw_item(preview, item, config)
    return valid_translation_hash(preview)


class ConcurrencyGauge:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class PipelineRunner:
    def __init__(
        self,
        *,
        sample_dir: str | Path,
        state_dir: str | Path,
        report_path: str | Path,
        config: dict,
        config_hash: str,
        injections: list[Injection],
        injection_hash: str,
        fresh: bool = False,
        stop_after_batches: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
        stop_reason: str = "cooperative_signal",
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.state_dir = Path(state_dir)
        self.report_path = Path(report_path)
        self.config = config
        self.config_hash = config_hash
        self.injections = sorted(injections, key=lambda item: (item.query_id, item.stage, item.injection_id))
        self.injection_hash = injection_hash
        self.fresh = fresh
        self.stop_after_batches = stop_after_batches
        self.stop_event = stop_event or threading.Event()
        self.stop_reason = stop_reason
        self.interrupted = False
        self.current_stage: Optional[str] = None
        self.commits_this_run = 0
        self.stage_timings: dict[str, dict] = {}
        self.max_active_workers = 0
        self.started_at = utc_now()
        self.started_perf = time.perf_counter()
        self.raw_items: list = []
        self.store: Optional[SnapshotStore] = None
        self._failure_records: dict[tuple, FailureRecord] = {}

    @classmethod
    def from_sample(
        cls,
        *,
        sample_dir: str | Path,
        state_dir: str | Path,
        report_path: str | Path,
        config_path: Optional[str | Path] = None,
        injections_path: Optional[str | Path] = None,
        fresh: bool = False,
        stop_after_batches: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
        stop_reason: str = "cooperative_signal",
        max_workers: Optional[int] = None,
        global_budget: Optional[int] = None,
        per_query_quota: Optional[int] = None,
    ) -> "PipelineRunner":
        sample_dir = Path(sample_dir)
        config_path = Path(config_path) if config_path else sample_dir / "config.json"
        config, config_hash = load_config(config_path)
        if any(value is not None for value in (max_workers, global_budget, per_query_quota)):
            config, config_hash = with_overrides(
                config,
                max_workers=max_workers,
                global_budget=global_budget,
                per_query_quota=per_query_quota,
            )
        injections, injection_hash = load_injections(injections_path)
        return cls(
            sample_dir=sample_dir,
            state_dir=state_dir,
            report_path=report_path,
            config=config,
            config_hash=config_hash,
            injections=injections,
            injection_hash=injection_hash,
            fresh=fresh,
            stop_after_batches=stop_after_batches,
            stop_event=stop_event,
            stop_reason=stop_reason,
        )

    def _initialize(self) -> None:
        self.raw_items = read_raw_sources(self.sample_dir)
        translation_hash = _preview_translation_hash(self.raw_items, self.config)
        identity = {
            "pipeline_version": self.config["pipeline_version"],
            "config_hash": self.config_hash,
            "translation_hash": translation_hash,
            "injection_hash": self.injection_hash,
            "input_hashes": input_hashes(self.sample_dir),
        }
        if self.fresh:
            self.store = SnapshotStore.reset(self.state_dir, identity)
        else:
            self.store = SnapshotStore(self.state_dir, identity)
            self.store.check_compatible()

    def _after_commit(self) -> None:
        self.commits_this_run += 1
        if self.stop_after_batches is not None and self.commits_this_run >= self.stop_after_batches:
            self.interrupted = True
            self.stop_reason = "stop_after_batches_test_hook"
        if self.stop_event.is_set():
            self.interrupted = True

    def _stage_start(self, stage: str) -> float:
        self.current_stage = stage
        return time.perf_counter()

    def _stage_finish(self, stage: str, started: float, *, expected: int, resumed: bool) -> None:
        assert self.store is not None
        completed = self.store.completed_batches(stage)
        self.stage_timings[stage] = {
            "seconds": round(time.perf_counter() - started, 9),
            "expected_batches": expected,
            "completed_batches": sorted(completed),
            "resumed": resumed,
            "complete": completed >= set(range(1, expected + 1)),
        }
        self.current_stage = None

    def _normalize_stage(self) -> NormalizedState:
        assert self.store is not None
        started = self._stage_start("normalize")
        batch_size = int(self.config["snapshot"]["batch_size"])
        batches = _chunks(self.raw_items, batch_size)
        expected = len(batches)
        completed = self.store.completed_batches("normalize")
        if expected == 0:
            self.store.mark_empty_stage("normalize")
            state = NormalizedState()
            self._stage_finish("normalize", started, expected=0, resumed=True)
            return state
        loaded = self.store.load_stage_state("normalize") if completed else None
        state = NormalizedState.from_dict(loaded) if loaded else NormalizedState()
        resumed = bool(completed)
        for batch_no, batch in enumerate(batches, start=1):
            if batch_no in completed:
                continue
            if self.interrupted:
                break
            apply_normalization_batch(state, batch, self.config)
            self.store.commit_batch("normalize", batch_no, state.to_dict())
            self._after_commit()
            if self.interrupted:
                break
        if not self.interrupted and self.store.stage_is_complete("normalize", expected):
            latest = self.store.load_stage_state("normalize")
            state = NormalizedState.from_dict(latest) if latest else state
        self._stage_finish("normalize", started, expected=expected, resumed=resumed)
        return state

    def _query_language(self, query: Query, resolver: TranslationResolver) -> dict:
        return detect_language(query.tokens, declared=query.declared_language, config=self.config, translation_resolver=resolver)

    def _document_language(self, document: Document, resolver: TranslationResolver) -> dict:
        return detect_language(document.tokens, declared=document.declared_language, config=self.config, translation_resolver=resolver)

    def _index_stage(self, state: NormalizedState, resolver: TranslationResolver) -> tuple[BM25Index, dict[str, dict]]:
        assert self.store is not None
        started = self._stage_start("index_build")
        batch_size = int(self.config["snapshot"]["batch_size"])
        documents = sorted(state.documents, key=lambda item: item.doc_id)
        batches = _chunks(documents, batch_size)
        expected = len(batches)
        doc_lid = {document.doc_id: self._document_language(document, resolver) for document in documents}
        doc_index_languages: dict[str, Any] = {}
        loaded = self.store.load_stage_state("index_build") if self.store.completed_batches("index_build") else None
        if loaded:
            index = BM25Index.from_dict(loaded["index"])
            doc_index_languages = loaded.get("document_index_languages", {})
            indexed_ids = set(loaded.get("indexed_documents", []))
        else:
            index = BM25Index(
                k1=float(self.config["bm25"]["k1"]),
                b=float(self.config["bm25"]["b"]),
                title_repeat=int(self.config["bm25"]["title_repeat"]),
                languages=self.config["index_languages"],
            )
            indexed_ids = set()
        if expected == 0:
            self.store.mark_empty_stage("index_build")
            self._stage_finish("index_build", started, expected=0, resumed=True)
            return index, doc_lid
        completed = self.store.completed_batches("index_build")
        resumed = bool(completed)
        for batch_no, batch in enumerate(batches, start=1):
            if batch_no in completed:
                continue
            if self.interrupted:
                break
            for document in batch:
                lid = doc_lid[document.doc_id]
                used = lid["used"]
                if used in self.config["index_languages"]:
                    index.add_document(document, used)
                    doc_index_languages[document.doc_id] = [used]
                else:
                    script_languages = sorted(
                        set(
                            language
                            for language in self.config["index_languages"]
                            if any(_token_language(token) == language for token in document.tokens)
                        )
                    )
                    index.add_document(document, *script_languages)
                    doc_index_languages[document.doc_id] = script_languages
                indexed_ids.add(document.doc_id)
            payload = {
                "index": index.to_dict(),
                "indexed_documents": sorted(indexed_ids),
                "document_index_languages": {
                    doc_id: doc_index_languages[doc_id] for doc_id in sorted(doc_index_languages)
                },
            }
            self.store.commit_batch("index_build", batch_no, payload)
            self._after_commit()
            if self.interrupted:
                break
        if not self.interrupted:
            latest = self.store.load_stage_state("index_build")
            if latest:
                index = BM25Index.from_dict(latest["index"])
                doc_index_languages = latest.get("document_index_languages", {})
        self._stage_finish("index_build", started, expected=expected, resumed=resumed)
        return index, doc_lid

    def _error_result(
        self,
        query: Query,
        lid: dict,
        allocation: int,
        exc: BaseException,
    ) -> tuple[dict, FailureRecord]:
        stage = getattr(exc, "stage", "retrieval_scoring")
        code = getattr(exc, "code", "E_QUERY_FAILED")
        injection_id = getattr(exc, "injection_id", None)
        if not isinstance(exc, PipelineError) and not injection_id:
            injection_id = None
        failure = FailureRecord(
            identifier=query.query_id,
            stage=stage,
            code=code,
            exception_type=type(exc).__name__,
            message=str(exc),
            source="queries",
            injection_id=injection_id,
        )
        consumed = int(getattr(exc, "consumed_points", 0))
        candidate_count = int(getattr(exc, "candidate_count", 0))
        result = {
            "query_id": query.query_id,
            "task_key": f"{query.query_id}|query|retrieval_scoring",
            "query_text": query.text,
            "declared_language": query.declared_language,
            "detected_language": lid["detected"],
            "language_used": lid["used"],
            "language_evidence": lid["evidence"],
            "status": "error",
            "code": code,
            "allocation": allocation,
            "consumed_points": consumed,
            "candidate_count": candidate_count,
            "scored_candidate_count": consumed,
            "recall_limit": int(self.config["recall_limit"]),
            "results": [],
            "translation_expansions": [],
            "missing_translation_terms": [],
            "diagnostics": {"error_type": type(exc).__name__, "message": str(exc)},
            "fixed_point": {
                "bits": int(self.config["fixed_point"]["bits"]),
                "scale": int(self.config["fixed_point"]["scale"]),
                "max_integer": (1 << (int(self.config["fixed_point"]["bits"]) - 1)) - 1,
            },
            "failure": failure.to_dict(),
        }
        return result, failure

    def _retrieval_stage(
        self,
        state: NormalizedState,
        index: BM25Index,
        doc_lid: dict[str, dict],
        resolver: TranslationResolver,
    ) -> tuple[dict[str, dict], list[dict]]:
        assert self.store is not None
        started = self._stage_start("retrieval_scoring")
        queries = sorted(state.queries, key=lambda item: item.query_id)
        query_lid = {query.query_id: self._query_language(query, resolver) for query in queries}
        plan = retrieval_budget_plan(queries, self.config["budget"])
        plan_by_id = {item["query_id"]: item for item in plan}
        plan_digest = query_plan_digest(plan)
        batch_size = int(self.config["snapshot"]["batch_size"])
        batches = _chunks(queries, batch_size)
        expected = len(batches)
        loaded = self.store.load_stage_state("retrieval_scoring") if self.store.completed_batches("retrieval_scoring") else None
        if loaded:
            if loaded.get("plan_digest") != plan_digest:
                error = SnapshotMismatchError("retrieval budget plan digest differs from the persisted snapshot")
                raise error
            results = dict(loaded.get("results", {}))
        else:
            results = {}
        if expected == 0:
            self.store.mark_empty_stage("retrieval_scoring")
            self._stage_finish("retrieval_scoring", started, expected=0, resumed=True)
            return results, plan

        doc_metadata = {
            document.doc_id: {
                "title": document.title,
                "language": doc_lid[document.doc_id]["used"],
                "declared_language": document.declared_language,
                "detected_language": doc_lid[document.doc_id]["detected"],
            }
            for document in state.documents
        }
        completed = self.store.completed_batches("retrieval_scoring")
        resumed = bool(completed)
        gauge = ConcurrencyGauge()
        max_workers = int(self.config["threads"]["max_workers"])
        query_by_id = {query.query_id: query for query in queries}
        injected_by_query: dict[tuple[str, str], list[Injection]] = defaultdict(list)
        for injection in self.injections:
            if injection.enabled:
                injected_by_query[(injection.query_id, injection.stage)].append(injection)

        for batch_no, batch in enumerate(batches, start=1):
            if batch_no in completed:
                continue
            if self.interrupted:
                break
            tasks: list[Query] = []
            for query in batch:
                if query.query_id in results:
                    continue
                plan_item = plan_by_id[query.query_id]
                if not plan_item["started"]:
                    results[query.query_id] = budget_exhausted_result(
                        query,
                        query_lid[query.query_id],
                        plan_item,
                        self.config,
                        "global retrieval budget already exhausted before this query in stable query-ID order",
                    )
                else:
                    tasks.append(query)

            start_barrier: Optional[threading.Barrier] = None

            def run_one(query: Query) -> tuple[dict, Optional[FailureRecord]]:
                gauge.enter()
                try:
                    if start_barrier is not None:
                        start_barrier.wait()
                    try:
                        result = execute_query(
                            query,
                            lid=query_lid[query.query_id],
                            index=index,
                            resolver=resolver,
                            config=self.config,
                            allocation=int(plan_by_id[query.query_id]["allocation"]),
                            injections=injected_by_query.get((query.query_id, "retrieval_scoring"), []),
                            document_metadata=doc_metadata,
                        )
                        return result, None
                    except Exception as exc:
                        return self._error_result(
                            query,
                            query_lid[query.query_id],
                            int(plan_by_id[query.query_id]["allocation"]),
                            exc,
                        )
                finally:
                    gauge.leave()

            if tasks:
                workers = min(max_workers, len(tasks))
                if workers > 1:
                    start_barrier = threading.Barrier(workers)
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retrieval") as executor:
                    futures = [executor.submit(run_one, query) for query in sorted(tasks, key=lambda item: item.query_id)]
                    for query, future in zip(sorted(tasks, key=lambda item: item.query_id), futures):
                        result, failure = future.result()
                        results[query.query_id] = result
                        if failure is not None:
                            self._remember_failure(failure)

            payload = {
                "plan": plan,
                "plan_digest": plan_digest,
                "results": {query_id: results[query_id] for query_id in sorted(results)},
            }
            self.store.commit_batch("retrieval_scoring", batch_no, payload)
            self.max_active_workers = max(self.max_active_workers, gauge.maximum)
            self._after_commit()
            if self.interrupted:
                break

        self.max_active_workers = max(self.max_active_workers, gauge.maximum)
        self._stage_finish("retrieval_scoring", started, expected=expected, resumed=resumed)
        return results, plan

    def _evaluation_stage(self, state: NormalizedState, retrieval_results: dict[str, dict]) -> dict[str, dict]:
        assert self.store is not None
        started = self._stage_start("evaluation")
        queries = sorted(state.queries, key=lambda item: item.query_id)
        batch_size = int(self.config["snapshot"]["batch_size"])
        batches = _chunks(queries, batch_size)
        expected = len(batches)
        loaded = self.store.load_stage_state("evaluation") if self.store.completed_batches("evaluation") else None
        evaluations = dict(loaded.get("per_query", {})) if loaded else {}
        if expected == 0:
            self.store.mark_empty_stage("evaluation")
            self._stage_finish("evaluation", started, expected=0, resumed=True)
            return evaluations
        qrels_by_query: dict[str, list[Qrel]] = defaultdict(list)
        for qrel in state.qrels:
            qrels_by_query[qrel.query_id].append(qrel)
        injection_by_query: dict[str, list[Injection]] = defaultdict(list)
        for injection in self.injections:
            if injection.enabled and injection.stage == "evaluation":
                injection_by_query[injection.query_id].append(injection)
        completed = self.store.completed_batches("evaluation")
        resumed = bool(completed)
        for batch_no, batch in enumerate(batches, start=1):
            if batch_no in completed:
                continue
            if self.interrupted:
                break
            for query in batch:
                if query.query_id in evaluations:
                    continue
                retrieval = retrieval_results.get(query.query_id)
                if retrieval is None:
                    evaluations[query.query_id] = {
                        "query_id": query.query_id,
                        "status": "skipped",
                        "code": "E_PIPELINE_INTERRUPTED",
                        "metrics": {},
                        "reason": "retrieval result was not committed before the cooperative stop",
                    }
                    continue
                if retrieval.get("status") != "ok":
                    evaluations[query.query_id] = {
                        "query_id": query.query_id,
                        "status": "skipped",
                        "code": retrieval.get("code"),
                        "metrics": {},
                        "reason": "retrieval did not complete successfully",
                    }
                    continue
                try:
                    evaluations[query.query_id] = evaluate_query_result(
                        retrieval,
                        qrels_by_query.get(query.query_id, []),
                        self.config,
                        injection_by_query.get(query.query_id, []),
                    )
                except Exception as exc:
                    stage = getattr(exc, "stage", "evaluation")
                    code = getattr(exc, "code", "E_EVALUATION_FAILED")
                    failure = FailureRecord(
                        identifier=query.query_id,
                        stage=stage,
                        code=code,
                        exception_type=type(exc).__name__,
                        message=str(exc),
                        source="qrels",
                        injection_id=getattr(exc, "injection_id", None),
                    )
                    self._remember_failure(failure)
                    evaluations[query.query_id] = {
                        "query_id": query.query_id,
                        "status": "error",
                        "code": code,
                        "metrics": {},
                        "failure": failure.to_dict(),
                        "reason": str(exc),
                    }
            payload = {"per_query": {query_id: evaluations[query_id] for query_id in sorted(evaluations)}}
            self.store.commit_batch("evaluation", batch_no, payload)
            self._after_commit()
            if self.interrupted:
                break
        self._stage_finish("evaluation", started, expected=expected, resumed=resumed)
        return evaluations

    def _remember_failure(self, failure: FailureRecord) -> None:
        key = (
            failure.stage,
            failure.identifier,
            failure.code,
            failure.exception_type,
            failure.message,
            failure.injection_id or "",
        )
        self._failure_records[key] = failure

    def _build_report(
        self,
        *,
        state: NormalizedState,
        resolver: TranslationResolver,
        index: BM25Index,
        doc_lid: dict[str, dict],
        retrieval_results: dict[str, dict],
        budget_plan: list[dict],
        evaluations: dict[str, dict],
        fatal_errors: Optional[list[dict]] = None,
    ) -> dict:
        assert self.store is not None
        status = "interrupted" if self.interrupted else "complete"
        exit_code = 0
        queries = sorted(state.queries, key=lambda item: item.query_id)
        plan_by_id = {item["query_id"]: item for item in budget_plan}
        query_reports = []
        for position, query in enumerate(queries, start=1):
            lid = self._query_language(query, resolver)
            result = retrieval_results.get(query.query_id)
            if result is None:
                plan_item = plan_by_id.get(query.query_id, {"allocation": 0, "pre_code": "E_BUDGET_EXHAUSTED"})
                result = {
                    "query_id": query.query_id,
                    "task_key": f"{query.query_id}|query|retrieval_scoring",
                    "query_text": query.text,
                    "declared_language": query.declared_language,
                    "detected_language": lid["detected"],
                    "language_used": lid["used"],
                    "language_evidence": lid["evidence"],
                    "status": "not_processed",
                    "code": "E_PIPELINE_INTERRUPTED",
                    "allocation": int(plan_item.get("allocation", 0)),
                    "consumed_points": 0,
                    "candidate_count": 0,
                    "scored_candidate_count": 0,
                    "recall_limit": int(self.config["recall_limit"]),
                    "results": [],
                    "translation_expansions": [],
                    "missing_translation_terms": [],
                    "diagnostics": {"reason": "cooperative stop occurred before this query batch committed"},
                    "fixed_point": None,
                }
            result = dict(result)
            result["position"] = position
            result["evaluation"] = evaluations.get(query.query_id)
            query_reports.append(result)
            failure = result.get("failure")
            if isinstance(failure, dict):
                self._remember_failure(FailureRecord(**failure))

        for evaluation in evaluations.values():
            failure = evaluation.get("failure")
            if isinstance(failure, dict):
                self._remember_failure(FailureRecord(**failure))
        for failure in state.failures:
            self._remember_failure(failure)
        failures = sorted_failures(self._failure_records.values())

        budget_sequence = []
        for item in budget_plan:
            result = retrieval_results.get(item["query_id"], {})
            budget_sequence.append(
                {
                    "query_id": item["query_id"],
                    "allocation": int(item["allocation"]),
                    "started": bool(item["started"]),
                    "status": result.get("status", "not_processed"),
                    "code": result.get("code", item.get("pre_code")),
                    "candidate_count": int(result.get("candidate_count", 0)),
                    "consumed_points": int(result.get("consumed_points", 0)),
                    "remaining_allocatable_after": int(item["remaining_allocatable_after"]),
                }
            )
        global_budget = int(self.config["budget"]["global_retrieval_points"])
        allocated_total = sum(item["allocation"] for item in budget_sequence)
        consumed_total = sum(item["consumed_points"] for item in budget_sequence)
        budget_summary = {
            "global_budget": global_budget,
            "per_query_quota": int(self.config["budget"]["per_query_quota"]),
            "allocated_total": allocated_total,
            "consumed_total": consumed_total,
            "remaining_after_allocation": global_budget - allocated_total,
            "remaining_after_consumption": global_budget - consumed_total,
            "unstarted_queries": sum(1 for item in budget_sequence if not item["started"]),
            "query_budget_exhausted": sum(1 for item in budget_sequence if item["code"] == "E_QUERY_BUDGET_EXHAUSTED"),
            "budget_exhausted": sum(1 for item in budget_sequence if item["code"] == "E_BUDGET_EXHAUSTED"),
            "sequence": budget_sequence,
        }

        requested_metrics = list(self.config["evaluation"]["metrics"])
        evaluation_summary = aggregate_metrics(evaluations.values(), requested_metrics)
        evaluation_summary["per_query"] = [evaluations[query.query_id] for query in queries if query.query_id in evaluations]

        document_reports = []
        for document in sorted(state.documents, key=lambda item: item.doc_id):
            lid = doc_lid.get(document.doc_id) or self._document_language(document, resolver)
            document_reports.append(
                {
                    "doc_id": document.doc_id,
                    "declared_language": document.declared_language,
                    "detected_language": lid["detected"],
                    "language_used": lid["used"],
                    "evidence": lid["evidence"],
                }
            )
        declared_counts = Counter(item["declared_language"] for item in document_reports)
        detected_counts = Counter(item["detected_language"] for item in document_reports)
        query_detected_counts = Counter(item["detected_language"] for item in query_reports)

        injection_hits: dict[str, FailureRecord] = {}
        for failure in failures:
            if failure.injection_id and failure.injection_id not in injection_hits:
                injection_hits[failure.injection_id] = failure
        injection_reports = []
        for injection in self.injections:
            failure = injection_hits.get(injection.injection_id)
            injection_reports.append(
                {
                    **injection.to_dict(),
                    "hit": failure is not None,
                    "hit_stage": failure.stage if failure else None,
                    "failure": failure.to_dict() if failure else None,
                }
            )

        dataset = {
            "counts": state_counts(state),
            "input_hashes": input_hashes(self.sample_dir),
            "translation_hash": resolver.hash(),
            "declared_language_counts": dict(sorted(declared_counts.items())),
            "detected_language_counts": dict(sorted(detected_counts.items())),
            "query_detected_language_counts": dict(sorted(query_detected_counts.items())),
        }
        language_detection = {
            "documents": document_reports,
            "queries": [
                {
                    "query_id": item["query_id"],
                    "declared_language": item["declared_language"],
                    "detected_language": item["detected_language"],
                    "language_used": item["language_used"],
                    "evidence": item["language_evidence"],
                }
                for item in query_reports
            ],
        }
        report_snapshot = self.store.snapshot_summary()
        canonical = {
            "config": _deterministic_config(self.config),
            "dataset": dataset,
            "language_detection": language_detection,
            "index": index.summary(),
            "budget": budget_summary,
            "queries": query_reports,
            "evaluation": evaluation_summary,
            "failures": [item.to_dict() for item in failures],
            "injections": injection_reports,
            "snapshot": {
                "version": report_snapshot["version"],
                "completed_batches": report_snapshot["completed_batches"],
                "stage_artifacts": {
                    stage: {
                        "version": artifact.get("version"),
                        "last_batch": artifact.get("last_batch"),
                        "global_version": artifact.get("global_version"),
                    }
                    for stage, artifact in sorted(report_snapshot["stage_artifacts"].items())
                },
            },
        }
        canonical_hash = sha256_json(canonical)
        finished_at = utc_now()
        return {
            "schema_version": "1.0",
            "status": status,
            "run": {
                "started_at": self.started_at,
                "finished_at": finished_at,
                "duration_seconds": round(time.perf_counter() - self.started_perf, 9),
                "exit_code": exit_code,
                "fresh": self.fresh,
                "stop_reason": self.stop_reason if self.interrupted else None,
                "commits_this_run": self.commits_this_run,
                "configured_max_workers": int(self.config["threads"]["max_workers"]),
                "observed_max_active_workers": self.max_active_workers,
                "report_path": str(self.report_path),
                "state_dir": str(self.state_dir),
            },
            "fatal_errors": fatal_errors or [],
            "config": self.config,
            "config_hash": self.config_hash,
            "dataset": dataset,
            "language_detection": language_detection,
            "index": index.summary(),
            "budget": budget_summary,
            "queries": query_reports,
            "evaluation": evaluation_summary,
            "failures": [item.to_dict() for item in failures],
            "injections": injection_reports,
            "snapshot": report_snapshot,
            "stage_timings": self.stage_timings,
            "determinism": {
                "sha256": canonical_hash,
                "excluded_from_hash": [
                    "run timestamps/durations",
                    "stage_timings",
                    "observed_max_active_workers",
                    "threads.max_workers",
                    "snapshot identity hashes and artifact absolute locations",
                ],
            },
        }

    def _fatal_report(self, exc: BaseException) -> dict:
        code = getattr(exc, "code", "E_PIPELINE")
        stage = getattr(exc, "stage", "pipeline")
        details = getattr(exc, "details", None)
        return {
            "schema_version": "1.0",
            "status": "error",
            "run": {
                "started_at": self.started_at,
                "finished_at": utc_now(),
                "duration_seconds": round(time.perf_counter() - self.started_perf, 9),
                "exit_code": 2,
                "fresh": self.fresh,
            },
            "fatal_errors": [
                {
                    "code": code,
                    "stage": stage,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "details": details,
                }
            ],
            "config_hash": self.config_hash,
            "injection_hash": self.injection_hash,
            "snapshot": self.store.snapshot_summary() if self.store else None,
        }

    def run(self) -> dict:
        self._failure_records = {}
        try:
            self._initialize()
        except Exception as exc:
            report = self._fatal_report(exc)
            atomic_write_json(self.report_path, report)
            return report

        assert self.store is not None
        try:
            state = self._normalize_stage()
            resolver = TranslationResolver(state.translations)
            index, doc_lid = self._index_stage(state, resolver)
            if self.interrupted:
                retrieval_results: dict[str, dict] = {}
                budget_plan = retrieval_budget_plan(sorted(state.queries, key=lambda item: item.query_id), self.config["budget"])
                evaluations: dict[str, dict] = {}
            else:
                retrieval_results, budget_plan = self._retrieval_stage(state, index, doc_lid, resolver)
                if self.interrupted:
                    evaluations = {}
                else:
                    evaluations = self._evaluation_stage(state, retrieval_results)
            report = self._build_report(
                state=state,
                resolver=resolver,
                index=index,
                doc_lid=doc_lid,
                retrieval_results=retrieval_results,
                budget_plan=budget_plan,
                evaluations=evaluations,
            )
        except Exception as exc:
            report = self._fatal_report(exc)
        atomic_write_json(self.report_path, report)
        return report


def _token_language(token: str) -> Optional[str]:
    from .normalization import token_script_language

    return token_script_language(token)


def _deterministic_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.get("threads", {}).pop("max_workers", None)
    return value
