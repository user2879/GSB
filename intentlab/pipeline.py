"""End-to-end deterministic diagnostic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cache import CacheManager, PreviousArtifacts
from .diagnostics import diagnose_query, float_as_score_result
from .errors import Codes, IssueCollector
from .fingerprints import (
    query_content_fingerprint,
    rule_view_fingerprint,
    state_fingerprint,
    trigger_entry_fingerprint,
)
from .matching import match_point_cost, match_query
from .merging import QueryScorePair, build_segment, partition_query_pairs, segment_key_for, segment_from_dict
from .models import (
    MatchResult,
    QueryConclusion,
    ScoreResult,
    Segment,
    TriggerEntry,
)
from .report import write_report
from .rules import compose_effective_rules
from .scheduler import RecomputeTask, execute_tasks, pending_entries, select_tasks
from .scoring import high_width_reference, score_query, score_query_float64
from .snapshot import (
    CurrentReference,
    SnapshotError,
    SnapshotHandle,
    SnapshotStore,
    config_hash,
)
from .validation import load_inputs


@dataclass(frozen=True)
class PipelinePaths:
    config_path: str | Path
    vocabulary_path: str | Path
    queries_path: str | Path
    rules_path: str | Path
    edits_path: str | Path | None
    snapshot_root: str | Path
    report_root: str | Path


@dataclass(frozen=True)
class PipelineResult:
    report: Mapping[str, Any]
    json_path: Path
    markdown_path: Path
    snapshot_reference: CurrentReference | None


def _entry_from_dict(data: Mapping[str, Any]) -> TriggerEntry:
    return TriggerEntry(
        trigger_id=str(data["trigger_id"]),
        terms=tuple(str(item) for item in data.get("terms", [])),
        intent_id=str(data["intent_id"]),
        weight=int(data["weight"]),
        slot_pattern=tuple(str(item) for item in data.get("slot_pattern", [])),
        source_revision=str(data["source_revision"]),
        entry_order=int(data["entry_order"]),
        provenance=tuple(dict(item) for item in data.get("provenance", [])),
    )


def _load_previous_artifacts(
    store: SnapshotStore,
    handle: SnapshotHandle | None,
    issues: IssueCollector,
) -> PreviousArtifacts | None:
    if handle is None:
        return None
    try:
        state_queries = store.read_artifact(handle, "state/queries.json").get("queries", {})
        matches_index = store.read_artifact(handle, "matches/index.json")
        scores_index = store.read_artifact(handle, "scores/index.json")
        segments_index = store.read_artifact(handle, "segments/index.json")
        rule_view = store.read_artifact(handle, "rules/effective_view.json")
        matches = {str(qid): store.read_artifact(handle, str(path)) for qid, path in matches_index.items()}
        scores = {str(qid): store.read_artifact(handle, str(path)) for qid, path in scores_index.items()}
        segments = {str(key): store.read_artifact(handle, str(path)) for key, path in segments_index.items()}
        entries = tuple(_entry_from_dict(item) for item in rule_view.get("entries", []))
        return PreviousArtifacts(
            generation=handle.generation,
            queries=state_queries,
            matches=matches,
            scores=scores,
            segments=segments,
            rule_entries=entries,
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        issues.add(
            Codes.SNAPSHOT_INTEGRITY,
            f"committed snapshot artifacts are unreadable: {exc}",
            scope="snapshot",
            entity_id=handle.generation,
            retained=False,
        )
        return None


def _input_hashes(paths: PipelinePaths) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, path in (
        ("config", paths.config_path),
        ("vocabulary", paths.vocabulary_path),
        ("queries", paths.queries_path),
        ("rules", paths.rules_path),
        ("edits", paths.edits_path),
    ):
        if path is None:
            continue
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        result[label] = digest
    return result


def _revision_counters(
    counts: Mapping[str, Mapping[str, int]],
    lineage: Iterable[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for revision_id in sorted(set(lineage) | set(counts)):
        source = counts.get(revision_id, {})
        result[revision_id] = {
            "match": int(source.get("match", 0)),
            "score": int(source.get("score", 0)),
            "merge": int(source.get("merge", 0)),
        }
    return result


def _query_report_item(conclusion: QueryConclusion) -> dict[str, Any]:
    return {
        "query_id": conclusion.query.query_id,
        "text": conclusion.query.text,
        "terms": list(conclusion.query.terms),
        "session_id": conclusion.query.session_id,
        "turn": conclusion.query.turn,
        "source": conclusion.query.source,
        "ground_truth_intent": conclusion.query.ground_truth_intent,
        "candidates": list(conclusion.query.candidates),
        "match": conclusion.match.to_dict(),
        "score": conclusion.score.to_dict(),
        "match_reused": conclusion.match_reused,
        "score_reused": conclusion.score_reused,
        "segment_id": conclusion.segment_id,
        "diagnostics": conclusion.diagnostics,
    }


def _combine_business_counts(
    issues: IssueCollector,
    pending: Iterable[Mapping[str, str]],
    failures: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    counts = issues.counts()
    for item in pending:
        code = str(item.get("code", ""))
        counts[code] = counts.get(code, 0) + 1
    for item in failures:
        code = str(item.get("code", ""))
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def run_pipeline(paths: PipelinePaths) -> PipelineResult:
    issues = IssueCollector()
    input_hashes_at_start = _input_hashes(paths)
    loaded, load_issues = load_inputs(
        config_path=paths.config_path,
        vocabulary_path=paths.vocabulary_path,
        queries_path=paths.queries_path,
        rules_path=paths.rules_path,
        edits_path=paths.edits_path,
    )
    issues.extend(load_issues)
    config = loaded.config
    view = compose_effective_rules(loaded.revisions, loaded.active_revision_ids, issues)
    query_map = {query.query_id: query for query in loaded.queries}
    current_state_token = state_fingerprint(
        queries=loaded.queries,
        revisions=loaded.revisions,
        active_revision_ids=loaded.active_revision_ids,
        config=config,
    )
    current_config_hash = config_hash(config.to_dict())

    store = SnapshotStore(paths.snapshot_root)
    previous_handle: SnapshotHandle | None = None
    snapshot_load_failed = False
    try:
        previous_handle = store.load_current()
    except SnapshotError as exc:
        snapshot_load_failed = True
        issues.add(
            exc.code,
            str(exc),
            scope="snapshot",
            retained=False,
        )
    previous = _load_previous_artifacts(store, previous_handle, issues)
    recovery = store.recovery_info(previous_handle)
    expected_reference = previous_handle.reference if previous_handle is not None else None

    previous_pending: set[str] = set()
    if previous_handle is not None:
        try:
            for item in store.read_artifact(previous_handle, "pending.json"):
                if isinstance(item, dict) and item.get("query_id"):
                    previous_pending.add(str(item["query_id"]))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            issues.add(
                Codes.SNAPSHOT_INTEGRITY,
                f"previously committed pending queue is unreadable: {exc}",
                scope="snapshot",
                entity_id=previous_handle.generation,
                retained=False,
            )

    cache_manager = CacheManager(
        previous=previous,
        current_view=view,
        queries=loaded.queries,
        config=config,
    )
    cache_plan = cache_manager.plan()

    immediate: dict[str, tuple[MatchResult, ScoreResult, bool, bool]] = {}
    tasks: list[RecomputeTask] = []
    for query in loaded.queries:
        decision = cache_plan.decisions[query.query_id]
        if decision.match is not None and decision.score is not None:
            immediate[query.query_id] = (decision.match, decision.score, True, True)
            continue
        requires_match = decision.match is None
        tasks.append(
            RecomputeTask(
                query_id=query.query_id,
                source=query.source,
                cost=match_point_cost(query, view) if requires_match else 0,
                requires_match=requires_match,
                requires_score=True,
                pending=query.query_id in previous_pending,
                priority=query.priority,
                session_id=query.session_id,
                turn=query.turn,
            )
        )
    schedule = select_tasks(
        tasks,
        match_budget_points=config.match_budget_points,
        source_quotas=config.source_quotas,
        batch_query_limit=config.batch_query_limit,
        default_source_quota=config.batch_query_limit,
    )

    def worker(task: RecomputeTask) -> tuple[MatchResult, ScoreResult, bool, bool]:
        from .models import RuleView

        query = query_map[task.query_id]
        decision = cache_plan.decisions[task.query_id]
        match = decision.match
        if match is None:
            local_view = RuleView(
                entries=view.entries,
                active_revision_ids=view.active_revision_ids,
                lineage_revision_ids=view.lineage_revision_ids,
                chain_version=view.chain_version,
            )
            match = match_query(query, local_view)
        score = decision.score
        if score is None:
            score = score_query(query, match, view.entries, config)
        return match, score, decision.match is not None, decision.score is not None

    computed, task_failures = execute_tasks(schedule.selected, worker, workers=config.workers)
    processed: dict[str, tuple[MatchResult, ScoreResult, bool, bool]] = dict(immediate)
    processed.update(computed)
    deferred = list(schedule.deferred)
    pending = list(pending_entries(deferred, tasks=tasks, failures=task_failures))

    pairs = [
        QueryScorePair(query=query_map[query_id], score=result[1])
        for query_id, result in sorted(processed.items())
        if query_id in query_map
    ]
    groups = partition_query_pairs(pairs, config)
    previous_segments = previous.segments if previous else {}
    segments: list[Segment] = []
    segment_reuse_ledger: list[dict[str, Any]] = []
    merge_invalidations = 0
    merge_revision_ids: set[str] = set()
    if previous is not None:
        merge_revision_ids = {
            cache_manager.trigger_revision.get(trigger_id, "")
            for trigger_id in cache_plan.changed_trigger_ids
        }
        merge_revision_ids.discard("")
    for group in groups:
        _, key = segment_key_for(group, config)
        cached_data = previous_segments.get(key)
        cached: Segment | None = None
        if cached_data is not None:
            try:
                candidate = segment_from_dict(cached_data)
                expected_query_ids = tuple(pair.query.query_id for pair in group)
                if (
                    candidate.cache_key == key
                    and candidate.query_ids == expected_query_ids
                    and candidate.intent == group[0].score.predicted_intent
                ):
                    cached = candidate
            except (KeyError, TypeError, ValueError):
                cached = None
        if cached is not None:
            segment = cached
            segment_reuse_ledger.append(
                {
                    "stage": "merge",
                    "segment_id": segment.segment_id,
                    "cache_key": key,
                    "basis": "member_score_keys_and_query_fingerprints_same",
                    "source_generation": previous.generation if previous else None,
                    "dependencies": list(segment.query_ids),
                }
            )
        else:
            segment = build_segment(group, config, cache_key=key)
            if previous is not None:
                merge_invalidations += 1
        segments.append(segment)
    segments.sort(key=lambda item: (item.session_id, item.start_turn, item.segment_id))

    segment_by_query: dict[str, str] = {}
    for segment in segments:
        for query_id in segment.query_ids:
            segment_by_query[query_id] = segment.segment_id

    reference_pairs: list[QueryScorePair] = []
    float_results: dict[str, Any] = {}
    high_results: dict[str, ScoreResult] = {}
    for query in loaded.queries:
        if query.query_id not in processed:
            continue
        match, fixed_score, _, _ = processed[query.query_id]
        float_score = score_query_float64(query, match, view.entries, config)
        float_results[query.query_id] = float_score
        high_results[query.query_id] = high_width_reference(query, match, view.entries, config)
        reference_pairs.append(
            QueryScorePair(
                query=query,
                score=float_as_score_result(fixed_score, float_score, config),
            )
        )
    reference_segments = tuple(
        build_segment(group, config) for group in partition_query_pairs(reference_pairs, config)
    )

    def segment_for_query(segments_to_search: Iterable[Segment], query_id: str) -> Segment | None:
        for segment in segments_to_search:
            if query_id in segment.query_ids:
                return segment
        return None

    conclusions: dict[str, QueryConclusion] = {}
    for query in loaded.queries:
        if query.query_id not in processed:
            continue
        match, score, match_reused, score_reused = processed[query.query_id]
        diagnostics = diagnose_query(
            query=query,
            match=match,
            fixed=score,
            float_reference=float_results[query.query_id],
            high_width=high_results[query.query_id],
            entries=view.entries,
            config=config,
            fixed_segment=segment_for_query(segments, query.query_id),
            reference_segment=segment_for_query(reference_segments, query.query_id),
        )
        conclusions[query.query_id] = QueryConclusion(
            query=query,
            match=match,
            score=score,
            match_reused=match_reused,
            score_reused=score_reused,
            segment_id=segment_by_query[query.query_id],
            diagnostics=diagnostics,
        )

    invalidation_counts = {
        revision_id: dict(stages)
        for revision_id, stages in cache_plan.invalidation_counts.items()
    }
    if merge_invalidations:
        targets = merge_revision_ids or {"merge-conclusion"}
        for revision_id in targets:
            bucket = invalidation_counts.setdefault(revision_id, {"match": 0, "score": 0, "merge": 0})
            bucket["merge"] = bucket.get("merge", 0) + merge_invalidations
    invalidation_by_revision = _revision_counters(invalidation_counts, view.lineage_revision_ids)

    invalidations: list[dict[str, Any]] = []
    for query_id in sorted(cache_plan.decisions):
        decision = cache_plan.decisions[query_id]
        if decision.match is None and decision.match_invalidated_by:
            invalidations.append(
                {
                    "stage": "match",
                    "query_id": query_id,
                    "changed_trigger_ids": list(decision.match_invalidated_by),
                    "basis": decision.match_basis,
                }
            )
        if decision.score is None and decision.score_invalidated_by:
            invalidations.append(
                {
                    "stage": "score",
                    "query_id": query_id,
                    "changed_trigger_ids": list(decision.score_invalidated_by),
                    "basis": decision.score_basis,
                }
            )
    reuse_ledger = list(cache_plan.ledger) + segment_reuse_ledger
    reuse_hits = len(reuse_ledger)
    query_items = [_query_report_item(conclusions[qid]) for qid in sorted(conclusions)]
    boundary_intents = sum(1 for conclusion in conclusions.values() if conclusion.score.boundary)
    pending_sorted = sorted(pending, key=lambda item: (str(item.get("query_id", "")), str(item.get("code", ""))))
    failures_sorted = sorted(task_failures, key=lambda item: (str(item.get("code", "")), str(item.get("query_id", ""))))
    issue_dicts = issues.to_dicts()
    business_code_counts = _combine_business_counts(issues, pending_sorted, failures_sorted)

    report_core: dict[str, Any] = {
        "schema_version": 1,
        "state_token": current_state_token,
        "config_hash": current_config_hash,
        "global_version": config.global_version,
        "summary": {
            "queries_loaded": len(loaded.queries),
            "queries_processed": len(conclusions),
            "queries_pending": len(pending_sorted),
            "segments": len(segments),
            "boundary_intents": boundary_intents,
            "reuse_hits": reuse_hits,
            "match_reuse_hits": sum(1 for item in query_items if item["match_reused"]),
            "score_reuse_hits": sum(1 for item in query_items if item["score_reused"]),
            "merge_reuse_hits": len(segment_reuse_ledger),
            "business_code_counts": business_code_counts,
            "budget_points_reserved": schedule.reserved_points,
            "source_usage": dict(schedule.source_usage),
        },
        "rules": {
            "cache_key": rule_view_fingerprint(view),
            "active_revision_ids": list(view.active_revision_ids),
            "lineage_revision_ids": list(view.lineage_revision_ids),
            "chain_version": view.chain_version,
            "trigger_count": len(view.entries),
            "triggers": [
                {
                    "trigger_id": entry.trigger_id,
                    "terms": list(entry.terms),
                    "intent_id": entry.intent_id,
                    "weight": entry.weight,
                    "slot_pattern": list(entry.slot_pattern),
                    "source_revision": entry.source_revision,
                    "entry_order": entry.entry_order,
                    "fingerprint": trigger_entry_fingerprint(entry),
                    "provenance": list(entry.provenance),
                }
                for entry in view.entries
            ],
        },
        "invalidation_by_revision": invalidation_by_revision,
        "queries": query_items,
        "segments": [segment.to_dict() for segment in segments],
        "pending": pending_sorted,
        "failures": failures_sorted,
        "issues": issue_dicts,
        "ledger": {
            "reuse": reuse_ledger,
            "invalidations": invalidations,
            "changed_trigger_ids": list(cache_plan.changed_trigger_ids),
            "match_changed_trigger_ids": list(cache_plan.match_changed_trigger_ids),
        },
        "recovery": recovery,
        "snapshot": {"generation": None, "committed": False},
    }

    committed_reference: CurrentReference | None = None
    allow_commit = not snapshot_load_failed
    if allow_commit and _input_hashes(paths) != input_hashes_at_start:
        issues.add(
            Codes.SNAPSHOT_CONCURRENT_COMMIT,
            "input files changed while the run was in progress; snapshot commit refused",
            scope="snapshot",
            retained=False,
        )
        allow_commit = False
    if allow_commit:
        stager = store.new_stager(
            state_token=current_state_token,
            config_hash=current_config_hash,
            global_version=config.global_version,
            expected_reference=expected_reference,
        )
        stager.write_json(
            "state/queries.json",
            {
                "queries": {
                    query.query_id: {
                        "fingerprint": query_content_fingerprint(query),
                        "query": query.to_dict(),
                    }
                    for query in loaded.queries
                }
            },
        )
        stager.write_json("state/config.json", config.to_dict())
        stager.write_json("state/input_issues.json", issue_dicts)
        stager.write_json("rules/effective_view.json", view.to_dict())
        stager.finish_stage(
            "rules",
            {
                "trigger_count": len(view.entries),
                "active_revision_ids": list(view.active_revision_ids),
            },
        )

        match_index: dict[str, str] = {}
        for query_id in sorted(conclusions):
            relative = f"matches/{query_id}.json"
            stager.write_json(relative, conclusions[query_id].match.to_dict())
            match_index[query_id] = relative
        stager.write_json("matches/index.json", match_index)
        stager.finish_stage(
            "match",
            {
                "artifacts": len(match_index),
                "reuse_hits": sum(1 for item in query_items if item["match_reused"]),
            },
        )

        score_index: dict[str, str] = {}
        for query_id in sorted(conclusions):
            relative = f"scores/{query_id}.json"
            stager.write_json(relative, conclusions[query_id].score.to_dict())
            score_index[query_id] = relative
        stager.write_json("scores/index.json", score_index)
        stager.finish_stage(
            "score",
            {
                "artifacts": len(score_index),
                "reuse_hits": sum(1 for item in query_items if item["score_reused"]),
            },
        )

        segment_index: dict[str, str] = {}
        for segment in segments:
            relative = f"segments/{segment.cache_key}.json"
            stager.write_json(relative, segment.to_dict())
            segment_index[segment.cache_key] = relative
        stager.write_json("segments/index.json", segment_index)
        stager.write_json("results/query_conclusions.json", query_items)
        stager.write_json("pending.json", pending_sorted)
        stager.write_json("failures.json", failures_sorted)
        stager.write_json(
            "ledger/cache.json",
            {
                "reuse": reuse_ledger,
                "invalidations": invalidations,
                "invalidation_by_revision": invalidation_by_revision,
            },
        )
        stager.write_json(
            "report_core.json",
            {
                "summary": report_core["summary"],
                "rules": report_core["rules"],
                "recovery": recovery,
            },
        )
        stager.finish_stage(
            "merge",
            {
                "segments": len(segments),
                "reuse_hits": len(segment_reuse_ledger),
                "boundary_intents": boundary_intents,
            },
        )
        try:
            committed_reference = stager.commit()
        except SnapshotError as exc:
            issues.add(
                exc.code,
                str(exc),
                scope="snapshot",
                retained=False,
            )
            issue_dicts = issues.to_dicts()
            business_code_counts = _combine_business_counts(issues, pending_sorted, failures_sorted)
            report_core["issues"] = issue_dicts
            report_core["summary"]["business_code_counts"] = business_code_counts

    report = dict(report_core)
    report["issues"] = issue_dicts
    report["summary"]["business_code_counts"] = business_code_counts
    if committed_reference is not None:
        report["snapshot"] = {
            "generation": committed_reference.generation,
            "committed": True,
            "state_token": committed_reference.state_token,
            "config_hash": committed_reference.config_hash,
            "global_version": committed_reference.global_version,
            "manifest_sha256": committed_reference.manifest_sha256,
        }
        report["run_id"] = committed_reference.generation
    else:
        report["snapshot"] = {
            "generation": None,
            "committed": False,
            "state_token": current_state_token,
            "config_hash": current_config_hash,
            "global_version": config.global_version,
        }
        report["run_id"] = "uncommitted-" + current_state_token[:12]

    json_path, markdown_path, _ = write_report(report, paths.report_root)
    return PipelineResult(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        snapshot_reference=committed_reference,
    )
