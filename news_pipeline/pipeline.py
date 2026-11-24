"""End-to-end deterministic pipeline orchestration."""

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

from .cache import FingerprintCache, sentence_dependencies
from .errors import E_SENTENCE_BUDGET_PENDING
from .hashing import combine
from .rewrite import rewrite_sentence
from .scoring import score_document
from .selection import select_sentences
from .segmentation import assign_sentence_ids, build_vocabulary, segment_document
from .snapshot import STAGES, SnapshotStore


class GracefulStop:
    """Processes the current document, then commits and exits."""

    def __init__(self):
        self.requested = False
        self._lock = threading.Lock()

    def request(self, *_args):
        with self._lock:
            self.requested = True

    def install(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, self.request)
            except (ValueError, OSError):
                pass


class ConcurrencyGuard:
    """Hard cap on simultaneously executing document tasks."""

    def __init__(self, limit):
        self.semaphore = threading.BoundedSemaphore(max(1, limit))
        self.peak = 0
        self._lock = threading.Lock()
        self._active = 0

    def __enter__(self):
        self.semaphore.acquire()
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        return self

    def __exit__(self, exc_type, exc, traceback):
        with self._lock:
            self._active -= 1
        self.semaphore.release()
        return False


def _stage_artifact(store, stage, doc_id):
    payload = store.read_stage(stage)
    if payload is None:
        return None
    return payload["artifacts"].get(doc_id)


def process_document(document, vocabulary, config, table, cache):
    """Run all five stages for one document with sentence-level caching."""
    q_format = config.q_format
    doc_id = document["doc_id"]
    reused = {"segmentation": 0, "scoring": 0, "rewrite": 0}

    # Stage 1: segmentation (artifact key derives from document content).
    seg_key = combine("segmentation", document["content_fingerprint"])
    cached_seg = cache.get("segmentation", seg_key)
    if cached_seg is not None:
        sentences = cached_seg["artifact"]["sentences"]
        reused["segmentation"] = len(sentences)
    else:
        sentences = assign_sentence_ids(segment_document(document))
        cache.put(
            "segmentation", seg_key,
            {"doc_id": doc_id, "sentences": sentences},
            artifact_id=seg_key,
        )

    # Stage 2: scoring, one cache entry per sentence.
    score_records = []
    fresh_scores = score_document(sentences, document, vocabulary, config, q_format)
    for fresh in fresh_scores:
        cached = cache.get("scoring", fresh["cache_key"])
        if cached is not None:
            score_records.append(cached["artifact"])
            reused["scoring"] += 1
        else:
            cache.put("scoring", fresh["cache_key"], fresh, artifact_id=fresh["sentence_id"],
                      dependencies=[{"type": "sentence", "id": fresh["sentence_id"]}])
            score_records.append(fresh)

    # Stage 3: coverage-constrained selection and boundary decisions.
    selection = select_sentences(score_records, document, config)
    cache.put("selection", selection["cache_key"], selection, artifact_id=doc_id)

    # Stage 4: rewriting, one cache entry per selected sentence.
    rewrite_records = []
    selected_by_id = {record["sentence_id"]: record for record in score_records}
    for sentence_id in selection["selected_sentence_ids"]:
        score_record = selected_by_id[sentence_id]
        rewrite_record = rewrite_sentence(score_record["text"], table, config.style_template_id)
        rewrite_record["sentence_id"] = sentence_id
        cached = cache.get("rewrite", rewrite_record["cache_key"])
        if cached is not None:
            rewrite_records.append(cached["artifact"])
            reused["rewrite"] += 1
        else:
            deps = sentence_dependencies(score_record, rewrite_record)
            cache.put("rewrite", rewrite_record["cache_key"], rewrite_record,
                      artifact_id=sentence_id, dependencies=[deps])
            rewrite_records.append(rewrite_record)

    # Stage 5: summary rollup.
    ordered_rewrites = sorted(rewrite_records, key=lambda item: item["sentence_id"])
    summary_text = "".join(item["rewritten_text"] for item in ordered_rewrites)
    summary = {
        "doc_id": doc_id,
        "summary_text": summary_text,
        "sentence_count": len(ordered_rewrites),
        "actual_entity_coverage": selection["actual_entity_coverage"],
        "boundary_sentence_count": len(selection["boundary_sentences"]),
        "touch_flags": selection["touch_flags"],
        "rewrite_records": ordered_rewrites,
        "cache_reuse": reused,
    }
    summary_key = combine(
        "summary",
        selection["cache_key"],
        [item["cache_key"] for item in ordered_rewrites],
        table.fingerprint,
    )
    summary["cache_key"] = summary_key
    cache.put("summary", summary_key, summary, artifact_id=doc_id)

    return {
        "doc_id": doc_id,
        "segmentation": {"doc_id": doc_id, "sentences": sentences},
        "scores": score_records,
        "selection": selection,
        "rewrites": ordered_rewrites,
        "summary": summary,
        "reused": reused,
        "sentence_total": len(sentences),
    }


def _tier_diagnostics(results, config):
    """First stage whose per-stage accumulated error exceeds its tolerance."""
    scale = float(config.q_format.scale)
    max_score_error = 0.0
    for result in results:
        for record in result["scores"]:
            dequantized = record["fixed_score"] / scale
            max_score_error = max(max_score_error, abs(dequantized - record["reference_score"]))
    selection_divergence = sum(len(result["selection"]["boundary_sentences"]) for result in results)
    stage_errors = {
        "segmentation": 0.0,
        "scoring": max_score_error,
        "selection": float(selection_divergence),
        "rewrite": 0.0,
        "summary": 0.0,
    }
    tolerances = {
        "segmentation": 0.0,
        "scoring": 1.0 / scale,
        "selection": 0.0,
        "rewrite": 0.0,
        "summary": 0.0,
    }
    first_breach = None
    for stage in STAGES:
        if stage_errors[stage] > tolerances[stage] + 1e-15:
            first_breach = stage
            break
    return {
        "first_stage_exceeding_tolerance": first_breach,
        "stage_errors": stage_errors,
        "stage_tolerances": tolerances,
    }


def run_pipeline(documents, config, table, snapshot_dir, stop_controller=None, code_counter=None,
                 after_doc_callback=None):
    """Execute the pipeline honoring budget batches and worker concurrency."""
    stop_controller = stop_controller or GracefulStop()
    code_counter = code_counter if code_counter is not None else {}
    store = SnapshotStore(snapshot_dir)
    compatible, manifest = store.is_compatible(config, table)
    stale_events = list(store.stale_events)
    if not compatible:
        if manifest is not None:
            code_counter["E_SNAPSHOT_STALE"] = code_counter.get("E_SNAPSHOT_STALE", 0) + 1
        store.reset_for_rebuild()

    cache = FingerprintCache(os.path.join(store.root, "cache.json"))
    vocabulary = build_vocabulary(documents)
    pending = list(documents)
    all_results = {}
    budget_pending = []
    report_peak = 0
    global_version = (manifest or {}).get("global_version", 0) + 1 if not compatible else \
        (manifest or {}).get("global_version", 1)

    pending_marks = set()
    while pending:
        # Stable consumption order: doc_id order, sentence offsets implicit.
        pending.sort(key=lambda item: item["doc_id"])
        batch = []
        budget = config.sentence_budget
        deferred = []
        for document in pending:
            estimated = len(segment_document(document))
            if estimated > budget:
                deferred.append(document)
                if document["doc_id"] not in pending_marks:
                    pending_marks.add(document["doc_id"])
                    code_counter[E_SENTENCE_BUDGET_PENDING] = code_counter.get(E_SENTENCE_BUDGET_PENDING, 0) + 1
                continue
            batch.append(document)
            budget -= estimated
        if not batch and deferred:
            # A document larger than a full batch gets a dedicated batch so
            # the scheduler can never starve; the pending code is retained.
            head = deferred.pop(0)
            batch.append(head)
        pending = deferred

        ordered_doc_ids = [document["doc_id"] for document in sorted(batch, key=lambda item: item["doc_id"])]
        batch_by_id = {document["doc_id"]: document for document in batch}
        workers = max(1, min(config.workers, len(ordered_doc_ids) or 1))
        guard = ConcurrencyGuard(workers)

        def _run(doc_id):
            with guard:
                return process_document(batch_by_id[doc_id], vocabulary, config, table, cache)

        if workers == 1:
            batch_results = [_run(doc_id) for doc_id in ordered_doc_ids]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {doc_id: executor.submit(_run, doc_id) for doc_id in ordered_doc_ids}
            batch_results = [futures[doc_id].result() for doc_id in ordered_doc_ids]

        for result in batch_results:
            all_results[result["doc_id"]] = result
            if after_doc_callback is not None:
                after_doc_callback(len(all_results), result["doc_id"])

        completed_cursor = sorted(all_results.keys())
        stage_buckets = {stage: {} for stage in STAGES}
        for result in batch_results:
            stage_buckets["segmentation"][result["doc_id"]] = result["segmentation"]
            stage_buckets["scoring"][result["doc_id"]] = result["scores"]
            stage_buckets["selection"][result["doc_id"]] = result["selection"]
            stage_buckets["rewrite"][result["doc_id"]] = result["rewrites"]
            stage_buckets["summary"][result["doc_id"]] = result["summary"]
        for stage in STAGES:
            store.commit_stage(
                stage, stage_buckets[stage],
                {
                    "config_fingerprint": config.fingerprint,
                    "synonym_table_version": table.version,
                    "table_fingerprint": table.fingerprint,
                    "completed_doc_cursor": completed_cursor,
                },
                global_version,
            )
        cache_tmp = os.path.join(store.tmp_root, "cache.json")
        cache.save(cache_tmp)
        os.replace(cache_tmp, os.path.join(store.root, "cache.json"))
        budget_pending = list(deferred)
        report_peak = guard.peak
        if stop_controller.requested:
            break

    ordered_results = [all_results[doc["doc_id"]] for doc in documents if doc["doc_id"] in all_results]
    diagnostics = _tier_diagnostics(ordered_results, config)
    report = _build_report(ordered_results, config, diagnostics, code_counter, stale_events,
                           cache, budget_pending, store, report_peak if ordered_results else 0)
    return report, ordered_results, store, cache


def _build_report(results, config, diagnostics, code_counter, stale_events, cache, budget_pending,
                  store, peak_concurrency):
    total_selected = sum(len(result["selection"]["selected_sentence_ids"]) for result in results)
    total_boundary = sum(len(result["selection"]["boundary_sentences"]) for result in results)
    reuse = {"segmentation": 0, "scoring": 0, "rewrite": 0}
    for result in results:
        for key in reuse:
            reuse[key] += result["reused"][key]
    report = {
        "run": {
            "config_fingerprint": config.fingerprint,
            "synonym_table_version": table_version_safe(store),
            "document_count": len(results),
            "workers_configured": config.workers,
            "peak_concurrency": peak_concurrency,
            "sentence_budget_per_batch": config.sentence_budget,
        },
        "documents": [
            {
                "doc_id": result["doc_id"],
                "selected_sentence_count": len(result["selection"]["selected_sentence_ids"]),
                "actual_entity_coverage": result["selection"]["actual_entity_coverage"],
                "boundary_sentence_count": len(result["selection"]["boundary_sentences"]),
                "touch_flags": result["selection"]["touch_flags"],
                "summary_text": result["summary"]["summary_text"],
                "reused_artifacts": result["reused"],
            }
            for result in results
        ],
        "totals": {
            "selected_sentences": total_selected,
            "boundary_sentences": total_boundary,
            "cache_reuse": reuse,
            "cache_entries": cache.stats(),
        },
        "budget": {
            "pending_doc_ids": sorted({doc["doc_id"] for doc in budget_pending}),
        },
        "tier_diagnostics": diagnostics,
        "business_codes": dict(sorted(code_counter.items())),
        "snapshot_stale_events": stale_events,
        "failures": [],
    }
    return report


def table_version_safe(store):
    manifest = store.load_manifest() or {}
    return manifest.get("synonym_table_version")
