"""Deterministic final JSON report assembly from committed stage artifacts."""

from __future__ import annotations

from typing import Any, Iterable

from .diagnostics import boundary_triplets_sort_key, score_histogram
from .fusion import triple_sort_key
from .loader import STAGE_ORDER, Injection, ModelPackage, Sentence, WorkbenchConfig
from .snapshot import SnapshotState


REPORT_SCHEMA_VERSION = 1


def _record_map(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["sentence_id"]: record for record in records}


def _sorted_failures(failures: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for failure in failures:
        key = (
            failure["sentence_id"],
            STAGE_ORDER.index(failure["stage"])
            if failure["stage"] in STAGE_ORDER
            else len(STAGE_ORDER),
            failure["code"],
            failure["exception_type"],
            failure["message"],
            tuple(failure.get("injection_ids", ())),
        )
        unique[key] = failure
    return [unique[key] for key in sorted(unique)]


def _public_triple(triple: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "sentence_id",
        "subject",
        "predicate",
        "object",
        "subject_span",
        "object_span",
        "subject_type",
        "object_type",
        "entity_score",
        "relation_score",
        "fusion_score",
        "confidence_permille",
        "threshold",
        "boundary_sample",
        "accepted",
        "score_source",
        "trigger",
        "suppressed",
        "injected_threshold_equal",
    )
    return {field: triple.get(field) for field in fields}


def build_report(
    config: WorkbenchConfig,
    model: ModelPackage,
    injections: tuple[Injection, ...],
    sentences: dict[str, Sentence],
    state: SnapshotState,
    *,
    initial_version: int | None,
    initial_stages: list[str],
    resumed: bool = False,
) -> dict[str, Any]:
    load_artifact = state.artifact("load") or {}
    entity_records = _record_map(state.records("entity"))
    relation_records = _record_map(state.records("relation"))
    fusion_records = _record_map(state.records("fusion"))
    diagnosis_records = _record_map(state.records("diagnosis"))

    failures: list[dict[str, Any]] = []
    injection_events: list[dict[str, Any]] = list(load_artifact.get("injection_events", []))
    for stage in ("load", "entity", "relation", "fusion", "diagnosis"):
        artifact = state.artifact(stage) or {}
        injection_events.extend(artifact.get("injection_events", []))
        for record in artifact.get("records", []):
            injection_events.extend(record.get("injection_events", []))
            if record.get("status") == "error" and record.get("failure"):
                failures.append(record["failure"])

    all_triples: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    reference_triples: list[dict[str, Any]] = []
    reference_candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    boundary_sensitive: list[dict[str, Any]] = []
    sentence_reports: list[dict[str, Any]] = []

    for sentence_id in sorted(sentences):
        entity_record = entity_records.get(sentence_id)
        relation_record = relation_records.get(sentence_id)
        fusion_record = fusion_records.get(sentence_id)
        diagnosis_record = diagnosis_records.get(sentence_id)
        item: dict[str, Any] = {"sentence_id": sentence_id}
        if entity_record is not None:
            item["entity_status"] = entity_record.get("status")
            if entity_record.get("status") == "ok":
                item["entity_recognition"] = entity_record["data"]
            else:
                item["entity_failure"] = entity_record.get("failure")
        if relation_record is not None:
            item["relation_status"] = relation_record.get("status")
            if relation_record.get("status") == "ok":
                item["relations"] = relation_record["data"]
                item["relation_candidates"] = relation_record["data"]["relations"]
            else:
                item["relation_failure"] = relation_record.get("failure")
        if fusion_record is not None:
            item["fusion_status"] = fusion_record.get("status")
            if fusion_record.get("status") == "ok":
                item["fusion"] = fusion_record["data"]
                item["triples"] = [
                    _public_triple(triple) for triple in fusion_record["data"]["triples"]
                ]
                all_triples.extend(fusion_record["data"]["triples"])
                all_candidates.extend(fusion_record["data"]["candidates"])
            else:
                item["fusion_failure"] = fusion_record.get("failure")
        if diagnosis_record is not None:
            item["diagnosis_status"] = diagnosis_record.get("status")
            if diagnosis_record.get("status") == "ok":
                diagnostic = diagnosis_record["data"]["diagnostic"]
                reference = diagnosis_record["data"]["reference"]
                item["diagnostic"] = diagnostic
                item["reference"] = reference
                diagnostics.append(diagnostic)
                boundary_sensitive.extend(diagnostic["boundary_sensitive_triples"])
                reference_triples.extend(reference["fusion"]["triples"])
                reference_candidates.extend(reference["fusion"]["candidates"])
            else:
                item["diagnosis_failure"] = diagnosis_record.get("failure")
        sentence_reports.append(item)

    all_triples.sort(key=triple_sort_key)
    reference_triples.sort(key=triple_sort_key)
    diagnostics.sort(key=lambda item: item["sentence_id"])
    boundary_sensitive.sort(key=boundary_triplets_sort_key)
    failures = _sorted_failures(failures)

    event_by_id: dict[str, dict[str, Any]] = {}
    for event in sorted(
        injection_events,
        key=lambda item: (
            item.get("injection_id", ""),
            item.get("sentence_id", ""),
            item.get("stage", ""),
        ),
    ):
        event_by_id.setdefault(event["injection_id"], event)
    injection_hits: list[dict[str, Any]] = []
    for injection in sorted(injections, key=lambda item: item.injection_id):
        event = event_by_id.get(injection.injection_id)
        if event is None:
            injection_hits.append(
                {
                    "injection_id": injection.injection_id,
                    "sentence_id": injection.sentence_id,
                    "stage": injection.stage,
                    "kind": injection.kind,
                    "applied": False,
                    "hit": False,
                    "reason": "stage_not_reached_or_condition_not_triggered",
                }
            )
        else:
            hit = {
                **event,
                "hit": bool(event.get("applied")),
            }
            hit.pop("failure_code", None)
            injection_hits.append(hit)

    configured_histogram = {
        "candidates": score_histogram(all_candidates),
        "output_triples": score_histogram(all_triples),
    }
    reference_histogram = {
        "candidates": score_histogram(reference_candidates),
        "output_triples": score_histogram(reference_triples),
    }
    failed_sentence_ids = sorted({failure["sentence_id"] for failure in failures})
    accepted_sentence_count = sum(
        1
        for record in load_artifact.get("records", [])
        if record.get("status") == "ok"
    )
    entity_count = sum(
        len(record["data"].get("entities", []))
        for record in entity_records.values()
        if record.get("status") == "ok"
    )
    relation_candidate_count = sum(
        len(record["data"].get("relations", []))
        for record in relation_records.values()
        if record.get("status") == "ok"
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed",
        "config_hash": config.config_hash,
        "model_hash": model.model_hash,
        "input_hash": state.manifest.get("input_hash"),
        "precisions": {
            "configured": {
                "decimals": config.fixed.decimals,
                "quantization_decimals": config.fixed.quantization_decimals,
            },
            "reference": {
                "decimals": config.fixed.reference().decimals,
                "quantization_decimals": config.fixed.reference().quantization_decimals,
            },
        },
        "snapshot": {
            "version": state.version,
            "config_hash": state.manifest.get("config_hash"),
            "model_hash": state.manifest.get("model_hash"),
            "input_hash": state.manifest.get("input_hash"),
            "stages_done": list(state.stages_done),
            "stage_versions": {
                stage: meta["version"]
                for stage, meta in sorted(
                    state.manifest.get("artifacts", {}).items(),
                    key=lambda item: STAGE_ORDER.index(item[0]),
                )
            },
            "completed_sentences": list(
                state.manifest.get("completed_sentences", [])
            ),
        },
        "execution": {
            "workers": config.workers,
            "snapshot_commit_interval": config.snapshot_commit_interval,
            "resumed": resumed,
            "resumed_from_version": initial_version if resumed else None,
            "initial_stages_done": initial_stages if resumed else [],
        },
        "statistics": {
            "input_sentences": accepted_sentence_count + len(
                [failure for failure in failures if failure["stage"] == "load"]
            ),
            "accepted_sentences": accepted_sentence_count,
            "failed_sentences": len(failed_sentence_ids),
            "recognized_entities": entity_count,
            "relation_candidates": relation_candidate_count,
            "candidate_triplets": len(all_candidates),
            "output_triplets": len(all_triples),
            "rejected_triplets": len(all_candidates) - len(all_triples),
            "boundary_sensitive_triplets": len(boundary_sensitive),
            "injections_hit": sum(1 for item in injection_hits if item["hit"]),
            "failures": len(failures),
        },
        "triples": [_public_triple(triple) for triple in all_triples],
        "score_histograms": {
            "configured": configured_histogram,
            "reference": reference_histogram,
        },
        "boundary_sensitive_triples": boundary_sensitive,
        "injection_hits": injection_hits,
        "diagnostics": diagnostics,
        "failures": failures,
        "sentences": sentence_reports,
    }
    return report
