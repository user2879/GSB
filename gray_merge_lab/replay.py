from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_json, first_difference
from .errors import LabError
from .fixedpoint import BatchFixedStats
from .ledger import Ledger
from .merge import merge_receipts
from .routing import plan_batches, route_sample
from .snapshot import build_payload
from .specs import RuleSpec, SampleSpec


def _record_samples(routing_input: dict[str, Any]) -> list[SampleSpec]:
    return [SampleSpec(**item) for item in routing_input["samples"]]


def _record_rule(routing_input: dict[str, Any]) -> RuleSpec:
    return RuleSpec(**routing_input["rule"])


def replay_batch(record: dict[str, Any]) -> dict[str, Any]:
    rule = _record_rule(record["routing_input"])
    samples = _record_samples(record["routing_input"])
    routed = [route_sample(sample, rule) for sample in samples]
    destinations = {item.destination for item in routed}
    buckets = [item.bucket for item in routed]
    if len(destinations) != 1 or destinations.pop() != record["destination"]:
        raise LabError(
            "E_REPLAY",
            "routing destination differs from ledger record",
            field="destination",
            batch_id=record["batch_id"],
        )
    if buckets != list(record["routing_input"]["buckets"]):
        raise LabError(
            "E_REPLAY",
            "routing buckets differ from ledger record",
            field="buckets",
            batch_id=record["batch_id"],
        )
    plans = plan_batches(samples, [rule])
    if len(plans) != 1 or plans[0].batch_id != record["batch_id"]:
        raise LabError(
            "E_REPLAY",
            "recomputed batch identity differs from ledger record",
            field="batch_id",
            batch_id=record["batch_id"],
        )

    merge_input = record["merge_input"]
    conclusion = merge_receipts(
        merge_input["receipts"],
        expected_sample_ids=merge_input["expected_sample_ids"],
        param_versions=merge_input.get("param_versions", {}),
        rule_id=merge_input.get("rule_id"),
    )
    stored_conclusion = record["conclusion"]
    if conclusion != stored_conclusion:
        path, expected, actual = first_difference(stored_conclusion, conclusion, "conclusion")
        raise LabError(
            "E_REPLAY",
            "merge conclusion differs from ledger record",
            field=path,
            batch_id=record["batch_id"],
            details={"stored": expected, "recomputed": actual},
        )

    raw_duration = int(record["raw_duration_ms"])
    raw_samples = int(record["raw_sample_count"])
    scale_exp = int(record["scale_exp"])
    fixed = BatchFixedStats(record["batch_id"], raw_duration, raw_samples, scale_exp)
    authoritative_fixed = fixed.to_dict()
    if record["duration_q"] != fixed.duration_q:
        raise LabError(
            "E_REPLAY",
            "fixed duration differs from raw integer conversion",
            field="duration_q",
            batch_id=record["batch_id"],
        )
    if record["samples_q"] != fixed.samples_q:
        raise LabError(
            "E_REPLAY",
            "fixed sample count differs from raw integer conversion",
            field="samples_q",
            batch_id=record["batch_id"],
        )
    if record.get("fixed_stats") != authoritative_fixed:
        path, expected, actual = first_difference(
            record.get("fixed_stats"), authoritative_fixed, "fixed_stats"
        )
        raise LabError(
            "E_REPLAY",
            "fixed-point statistics differ from authoritative conversion",
            field=path,
            batch_id=record["batch_id"],
            details={"stored": expected, "recomputed": actual},
        )
    return {
        "batch_id": record["batch_id"],
        "batch_index": record["batch_index"],
        "rule_id": record["rule_id"],
        "destination": record["destination"],
        "bucket": buckets[0],
        "sample_ids": record["sample_ids"],
        "conclusion": conclusion,
        "fixed_stats": authoritative_fixed,
    }


def _preview_from_replay(record: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    fixed = replayed["fixed_stats"]
    conclusion = replayed["conclusion"]
    return {
        "batch_id": record["batch_id"],
        "batch_index": record["batch_index"],
        "rule_id": record["rule_id"],
        "destination": record["destination"],
        "client_key": record["routing_input"]["samples"][0]["client_key"],
        "client_bucket": replayed["bucket"],
        "canary_weight_bp": record["routing_input"]["rule"]["canary_weight_bp"],
        "sample_ids": record["sample_ids"],
        "raw_duration_ms": fixed["raw_duration_ms"],
        "raw_sample_count": fixed["raw_samples"],
        "duration_q": fixed["duration_q"],
        "samples_q": fixed["samples_q"],
        "avg_duration_q": fixed["duration_q"],
        "avg_samples_q": fixed["samples_q"],
        "scale_exp": fixed["scale_exp"],
        "truncation": fixed["truncation"],
        "single_value_error_bound_original": fixed["single_value_error_original"],
        "output_digest": conclusion["output_digest"],
        "backend_param_digest": conclusion["backend_param_digest"],
        "backend_param_version": conclusion["backend_param_version"],
        "conclusion_digest": conclusion["conclusion_digest"],
        "sampling_error_affected": False,
    }


def authoritative_recompute(
    ledger: Ledger,
    snapshot_payload: dict[str, Any],
) -> dict[str, Any]:
    ledger.verify_integrity()
    records = ledger.batches()
    replayed: list[dict[str, Any]] = []
    expected_previews: list[dict[str, Any]] = []
    for record in records:
        result = replay_batch(record)
        replayed.append(result)
        expected_previews.append(_preview_from_replay(record, result))

    actual_previews = {
        item.get("batch_id"): item for item in snapshot_payload.get("preview_batches", [])
    }
    for expected in expected_previews:
        batch_id = expected["batch_id"]
        actual = actual_previews.get(batch_id)
        if actual is None:
            raise LabError(
                "E_PREVIEW",
                "snapshot is missing a batch shown by authoritative replay",
                field="preview_batches.batch_id",
                batch_id=batch_id,
            )
        if actual != expected:
            path, stored, recomputed = first_difference(actual, expected, "preview")
            raise LabError(
                "E_PREVIEW",
                "quick preview differs from authoritative recompute",
                field=path,
                batch_id=batch_id,
                details={"preview": stored, "authoritative": recomputed},
            )
    extra_ids = sorted(set(actual_previews) - {item["batch_id"] for item in expected_previews})
    if extra_ids:
        raise LabError(
            "E_PREVIEW",
            "snapshot contains batches absent from the ledger",
            field="preview_batches.batch_id",
            batch_id=extra_ids[0],
        )

    expected_conclusions = [item["conclusion"] for item in replayed]
    actual_conclusions = snapshot_payload.get("conclusions", [])
    if actual_conclusions != expected_conclusions:
        mismatch_index = None
        for index in range(max(len(actual_conclusions), len(expected_conclusions))):
            actual = actual_conclusions[index] if index < len(actual_conclusions) else "<missing>"
            expected = expected_conclusions[index] if index < len(expected_conclusions) else "<missing>"
            if actual != expected:
                mismatch_index = index
                break
        assert mismatch_index is not None
        expected = expected_conclusions[mismatch_index] if mismatch_index < len(expected_conclusions) else {}
        actual = actual_conclusions[mismatch_index] if mismatch_index < len(actual_conclusions) else {}
        batch_id = expected.get("batch_id") or actual.get("batch_id") or (records[mismatch_index]["batch_id"] if mismatch_index < len(records) else None)
        path = f"conclusions[{mismatch_index}]"
        if isinstance(actual, dict) and isinstance(expected, dict):
            path, stored, recomputed = first_difference(actual, expected, path)
        else:
            stored, recomputed = actual, expected
        raise LabError(
            "E_PREVIEW",
            "snapshot merge conclusion differs from authoritative recompute",
            field=path,
            batch_id=batch_id,
            details={"preview": stored, "authoritative": recomputed},
        )

    expected_progress = ledger.cursor()
    actual_progress = snapshot_payload.get("progress", {})
    if actual_progress != expected_progress:
        batch_id = records[0]["batch_id"] if records else None
        path, stored, recomputed = first_difference(actual_progress, expected_progress, "progress")
        raise LabError(
            "E_PREVIEW",
            "snapshot progress differs from authoritative ledger cursor",
            field=path,
            batch_id=batch_id,
            details={"preview": stored, "authoritative": recomputed},
        )

    authoritative_payload = build_payload(ledger, int(snapshot_payload.get("scale_exp", 0)))
    actual_stats = snapshot_payload.get("rule_stats", {})
    expected_stats = authoritative_payload["rule_stats"]
    if actual_stats != expected_stats:
        path, stored, recomputed = first_difference(actual_stats, expected_stats, "rule_stats")
        rule_id = path.split(".", 2)[1] if path.startswith("rule_stats.") else None
        first_batch = next(
            (record["batch_id"] for record in records if rule_id is None or record["rule_id"] == rule_id),
            None,
        )
        raise LabError(
            "E_PREVIEW",
            "rule statistics differ from authoritative recompute",
            field=path,
            batch_id=first_batch,
            details={"preview": stored, "authoritative": recomputed},
        )
    return {
        "ok": True,
        "completed_batches": len(records),
        "ledger_content_digest": ledger.content_digest(),
        "fixed_scale_exp": authoritative_payload["scale_exp"],
        "batches": replayed,
    }


def replay_remaining(
    ledger: Ledger,
    snapshot_payload: dict[str, Any],
) -> dict[str, Any]:
    ledger.verify_integrity()
    records = ledger.batches()
    progress = snapshot_payload.get("progress", {})
    completed = int(progress.get("completed_batches", 0))
    if completed < 0 or completed > len(records):
        raise LabError(
            "E_REPLAY",
            "snapshot progress is outside the ledger batch range",
            field="progress.completed_batches",
            details={"completed": completed, "ledger_batches": len(records)},
        )
    replayed: list[dict[str, Any]] = []
    for record in records[completed:]:
        try:
            result = replay_batch(record)
        except LabError as exc:
            if exc.code == "E_REPLAY":
                raise
            raise LabError(
                "E_REPLAY",
                exc.message,
                field=exc.field,
                batch_id=record.get("batch_id"),
                details=exc.details,
            ) from exc
        replayed.append(result)
    updated_payload = build_payload(ledger, int(snapshot_payload.get("scale_exp", 0)))
    return {
        "ok": True,
        "resumed_after_batch_index": completed,
        "replayed_batches": len(replayed),
        "batch_ids": [item["batch_id"] for item in replayed],
        "updated_payload": updated_payload,
    }
