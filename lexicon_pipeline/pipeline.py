"""End-to-end deterministic pipeline and canonical comparison views."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .cache import (
    CacheStore,
    build_manifest,
    final_annotation_key,
    path_total_key,
    plan_impact,
    segment_fragment_key,
    serialization_key,
    tag_candidate_key,
)
from .formats import jv1_to_jv2, jv2_to_jv1, parse_jv1, write_jv2_document
from .hashing import fingerprint
from .scheduler import run_batches_until_complete
from .segmentation import build_index, rule_package_fingerprint, segment_document
from .tagging import normalize_transition_weights, tag_document


def run_state_once(
    state: dict[str, Any],
    cache_path: str | Path | None = None,
    previous_manifest: dict[str, Any] | None = None,
    execution_mode: str = "concurrent",
    run_id: str = "run",
    command_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(state["config"])
    if execution_mode == "serial":
        config["concurrency"] = 1
    exchange_version = state.get("exchange_version", config.get("exchange_version", "jv2"))
    lexicon = state["lexicon"]
    documents_document = state["documents"]
    documents = documents_document["documents"]
    weights_document = state["weights"]
    rules_document = state["rules"]

    index = build_index(lexicon)
    transitions = normalize_transition_weights(weights_document, index)
    rule_fp = rule_package_fingerprint(rules_document)
    cache = CacheStore(cache_path) if cache_path else CacheStore()
    cache_events: list[dict[str, Any]] = []

    def process_document(task: dict[str, Any]) -> dict[str, Any]:
        doc = next(item for item in documents if item["doc_id"] == task["doc_id"])
        segmentation = segment_document(doc, index, rules_document, lexicon.get("revision", 0), rule_fp)
        tagging = tag_document(segmentation, index, transitions, config)
        events: list[dict[str, Any]] = []
        previous_path_key: str | None = None

        for position, fragment in enumerate(segmentation["fragments"]):
            tagged = tagging["fragments"][position]
            seg_key = segment_fragment_key(
                fragment,
                index,
                segmentation["effective_rule_fingerprint"],
                rule_fp,
            )
            _, hit = cache.get_or_set(
                seg_key,
                lambda f=fragment: {"fragment": _json_safe_fragment(f)},
            )
            events.append(_cache_event(run_id, segmentation, fragment, "segment", seg_key, hit))

            scoped_weights = {
                name: transitions["named"].get(name, transitions["default"])
                for name in tagged["weight_refs"]
            }
            tag_key = tag_candidate_key(seg_key, tagged, config, scoped_weights)
            _, hit = cache.get_or_set(
                tag_key,
                lambda t=tagged: {"tagged_fragment": _json_safe_tagged_fragment(t)},
            )
            events.append(_cache_event(run_id, segmentation, fragment, "tag_candidates", tag_key, hit))

            p_key = path_total_key(tag_key, previous_path_key, config)
            dp_artifact = {
                "dp_by_code": tagging["_dp_steps"][position],
                "backpointers": tagging["_backpointers"][position],
                "selected_prefix_total_int": tagged["selected_prefix_total_int"],
            }
            _, hit = cache.get_or_set(p_key, lambda value=dp_artifact: value)
            events.append(_cache_event(run_id, segmentation, fragment, "path_total", p_key, hit))
            previous_path_key = p_key

            annotation_key_value = final_annotation_key(p_key, tagged["selected_code"], exchange_version)
            _, hit = cache.get_or_set(
                annotation_key_value,
                lambda t=tagged: {
                    "selected_tag": t["selected_tag"],
                    "selected_code": t["selected_code"],
                    "annotation_key": annotation_key_value,
                },
            )
            events.append(_cache_event(run_id, segmentation, fragment, "annotation", annotation_key_value, hit))

        return {
            "doc_id": doc["doc_id"],
            "doc_seq": doc["doc_seq"],
            "task": task,
            "segmentation": segmentation,
            "tagging": tagging,
            "cache_events": events,
        }

    scheduled = run_batches_until_complete(documents, index, process_document, config)
    doc_results = scheduled["results"]
    for result in doc_results:
        cache_events.extend(result["cache_events"])

    segmentations = [result["segmentation"] for result in doc_results]
    taggings = [_public_tagging(result["tagging"]) for result in doc_results]
    segmentations.sort(key=lambda item: item["doc_seq"])
    taggings.sort(key=lambda item: item["doc_seq"])

    manifest = build_manifest(
        lexicon,
        index,
        weights_document,
        transitions,
        rules_document,
        rule_fp,
        config,
        segmentations,
        taggings,
        exchange_version,
    )

    all_receipts: list[dict[str, Any]] = []
    for tagging in taggings:
        all_receipts.extend(tagging["receipts"])
    all_receipts.sort(key=lambda item: (item["doc_seq"], item["fragment_seq"]))
    jv2_document = write_jv2_document(all_receipts, config["fixed_point"]["rounding_mode"])

    pos_codes = index["pos_to_code"]
    downgrade = jv2_to_jv1(all_receipts, pos_codes, config["jv1_widths"])
    jv1_text = downgrade.text or ""
    jv1_records = downgrade.records
    sidecar = downgrade.sidecar or {"records": {}, "downgrade_entries": []}

    parsed_jv1 = parse_jv1(jv1_text, config["jv1_widths"])
    upgraded = jv1_to_jv2(parsed_jv1.records, index["code_to_pos"], sidecar)
    roundtrip_issues = parsed_jv1.issues + downgrade.issues + upgraded.issues
    roundtrip_matches = _compare_roundtrip(all_receipts, upgraded.records)

    jv1_by_key = {record["key"]: record for record in jv1_records}
    jv2_by_key = {f"{r['doc_seq']:04d}:{r['fragment_seq']:04d}": r for r in all_receipts}
    for fragment_manifest in manifest["fragments"]:
        key = f"{fragment_manifest['doc_seq']:04d}:{fragment_manifest['fragment_seq']:04d}"
        jv1_key = fragment_manifest["keys"]["jv1"]
        _, hit = cache.get_or_set(jv1_key, lambda key=key: {"line": jv1_by_key.get(key, {}).get("line")})
        cache_events.append(
            {
                "run_id": run_id,
                "doc_id": fragment_manifest["doc_id"],
                "doc_seq": fragment_manifest["doc_seq"],
                "fragment_id": fragment_manifest["fragment_id"],
                "fragment_seq": fragment_manifest["fragment_seq"],
                "stage": "serialization_jv1",
                "cache_key": jv1_key,
                "hit": hit,
            }
        )
        jv2_key = fragment_manifest["keys"]["jv2"]
        _, hit = cache.get_or_set(jv2_key, lambda key=key: {"receipt": jv2_by_key.get(key)})
        cache_events.append(
            {
                "run_id": run_id,
                "doc_id": fragment_manifest["doc_id"],
                "doc_seq": fragment_manifest["doc_seq"],
                "fragment_id": fragment_manifest["fragment_id"],
                "fragment_seq": fragment_manifest["fragment_seq"],
                "stage": "serialization_jv2",
                "cache_key": jv2_key,
                "hit": hit,
            }
        )

    impact = plan_impact(previous_manifest, manifest, documents, lexicon)
    cache.save()

    boundary_fragments: list[dict[str, Any]] = []
    for tagging in taggings:
        boundary_fragments.extend(tagging["boundary_fragments"])
    boundary_fragments.sort(key=lambda item: (item["doc_seq"] if "doc_seq" in item else 0, item["fragment_seq"]))

    failures = list(command_failures or []) + list(scheduled["failures"])
    for issue in roundtrip_issues:
        failures.append({"code": issue.code, "message": issue.message, "line": issue.line, "field": issue.field, "path": issue.path})
    failures = _stable_failures(failures)

    ledger_entries = _build_ledger_entries(
        run_id,
        scheduled,
        cache_events,
        impact,
        sidecar["downgrade_entries"],
        boundary_fragments,
        failures,
        command_failures or [],
    )

    return {
        "run_id": run_id,
        "execution_mode": execution_mode,
        "config": _public_config(config),
        "exchange_version": exchange_version,
        "lexicon_revision": lexicon.get("revision", 0),
        "rule_package_fingerprint": rule_fp,
        "scheduler": {
            "batches": scheduled["batches"],
            "pending": scheduled["pending"],
            "failures": scheduled["failures"],
            "pool_metrics": scheduled["pool_metrics"],
            "cumulative_consumed_budget": scheduled["cumulative_consumed_budget"],
        },
        "doc_results": [
            {
                "doc_id": item["doc_id"],
                "doc_seq": item["doc_seq"],
                "segmentation": _json_safe(item["segmentation"]),
                "tagging": taggings[[t["doc_id"] for t in taggings].index(item["doc_id"])],
            }
            for item in doc_results
        ],
        "segmentations": segmentations,
        "taggings": taggings,
        "manifest": manifest,
        "impact": impact,
        "cache_events": sorted(cache_events, key=_cache_event_sort_key),
        "jv1": {"format": "jv1", "text": jv1_text, "records": jv1_records},
        "jv2": jv2_document,
        "sidecar": sidecar,
        "roundtrip": {"matches": roundtrip_matches, "issue_count": len(roundtrip_issues)},
        "boundary_fragments": boundary_fragments,
        "failures": failures,
        "ledger_entries": ledger_entries,
    }


def _public_tagging(tagging: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tagging.items() if not key.startswith("_")}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _json_safe_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(fragment)


def _json_safe_tagged_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(fragment)


def _cache_event(run_id, segmentation, fragment, stage, key, hit):
    return {
        "run_id": run_id,
        "doc_id": segmentation["doc_id"],
        "doc_seq": segmentation["doc_seq"],
        "fragment_id": fragment["fragment_id"],
        "fragment_seq": fragment["fragment_seq"],
        "stage": stage,
        "cache_key": key,
        "hit": bool(hit),
    }


def _cache_event_sort_key(event: dict[str, Any]) -> tuple:
    stage_order = {"segment": 0, "tag_candidates": 1, "path_total": 2, "annotation": 3, "serialization_jv1": 4, "serialization_jv2": 5}
    return (event["doc_seq"], event["fragment_seq"], stage_order.get(event["stage"], 9), event["stage"])


def _compare_roundtrip(original: list[dict[str, Any]], restored: list[dict[str, Any]]) -> dict[str, Any]:
    original_by_key = {f"{r['doc_seq']:04d}:{r['fragment_seq']:04d}": r for r in original}
    restored_by_key = {r["key"]: r for r in restored}
    mismatches: list[dict[str, Any]] = []
    for key, source in sorted(original_by_key.items()):
        target = restored_by_key.get(key)
        if target is None:
            mismatches.append({"key": key, "field": "receipt", "reason": "missing"})
            continue
        for field in ("score", "pos", "rounding_mode"):
            if source.get(field) != target.get(field):
                mismatches.append({"key": key, "field": field, "source": source.get(field), "restored": target.get(field)})
    return {"checked": len(original_by_key), "mismatch_count": len(mismatches), "mismatches": mismatches, "ok": not mismatches}


def _stable_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for failure in failures:
        key = fingerprint(
            {
                "seq": failure.get("seq"),
                "doc_id": failure.get("doc_id"),
                "code": failure.get("code"),
                "field": failure.get("field"),
                "line": failure.get("line"),
                "message": failure.get("message"),
            },
            32,
        )
        unique[key] = failure
    return sorted(
        unique.values(),
        key=lambda item: (
            item.get("seq") if isinstance(item.get("seq"), int) else 10**9,
            item.get("doc_seq") if isinstance(item.get("doc_seq"), int) else 10**9,
            item.get("line") if isinstance(item.get("line"), int) else 10**9,
            str(item.get("code", "")),
            str(item.get("field", "")),
            str(item.get("message", "")),
        ),
    )


def _build_ledger_entries(run_id, scheduled, cache_events, impact, downgrades, boundaries, failures, command_failures):
    entries: list[dict[str, Any]] = []
    entries.append({"entry_type": "stage", "run_id": run_id, "stage": "load_validate", "status": "ok"})
    for failure in command_failures:
        entries.append({"entry_type": "command_failure", "run_id": run_id, **failure})
    for batch in scheduled["batches"]:
        for item in batch["pending"]:
            entries.append({"entry_type": "scheduling", "run_id": run_id, **item})
    for item in scheduled["failures"]:
        entries.append({"entry_type": "failure", "run_id": run_id, **item})
    for event in sorted(cache_events, key=_cache_event_sort_key):
        entries.append({"entry_type": "cache_hit" if event["hit"] else "cache_miss", **event})
    for item in impact.get("invalidations", []):
        entries.append({"entry_type": "invalidation", "run_id": run_id, **item})
    for item in impact.get("reuses", []):
        entries.append({"entry_type": "reuse_plan", "run_id": run_id, **item})
    for item in downgrades:
        entries.append({"entry_type": "downgrade", "run_id": run_id, **item})
    for item in boundaries:
        entries.append({"entry_type": "boundary", "run_id": run_id, **item})
    scheduled_failure_keys = {fingerprint(item, 32) for item in scheduled["failures"]}
    command_failure_keys = {fingerprint(item, 32) for item in command_failures}
    for failure in failures:
        if fingerprint(failure, 32) not in scheduled_failure_keys | command_failure_keys:
            entries.append({"entry_type": "failure", "run_id": run_id, **failure})
    return entries


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "concurrency": config["concurrency"],
        "config_hash": config["config_hash"],
        "effective_fraction_bits": config["effective_fraction_bits"],
        "fixed_point_scale": config["fixed_point_scale"],
        "high_precision_bits": config["high_precision_bits"],
        "high_precision_scale": config["high_precision_scale"],
        "quantization_tier": config["fixed_point"]["quantization_tier"],
        "match_budget_points": config["match_budget_points"],
        "doc_quota_points": config["doc_quota_points"],
        "jv1_widths": config["jv1_widths"],
    }


def comparison_view(result: dict[str, Any]) -> dict[str, Any]:
    """Field ordering and volatile execution metadata are canonicalized."""

    config_view = {key: value for key, value in result["config"].items() if key != "concurrency"}
    return {
        "lexicon_revision": result["lexicon_revision"],
        "config": config_view,
        "exchange_version": result["exchange_version"],
        "documents": [
            {
                "doc_id": doc["doc_id"],
                "doc_seq": doc["doc_seq"],
                "fragments": [
                    {key: fragment.get(key) for key in (
                        "fragment_seq",
                        "start",
                        "end",
                        "surface",
                        "entry_id",
                        "entry_refs",
                        "rule_ids",
                        "origin",
                    )}
                    for fragment in doc["segmentation"]["fragments"]
                ],
                "suppressed": doc["segmentation"]["suppressed"],
                "rule_decisions": doc["segmentation"]["rule_decisions"],
                "tags": doc["tagging"]["tags"],
                "selected_codes": doc["tagging"]["selected_codes"],
                "path_total_int": doc["tagging"]["path_total_int"],
                "path_total_float": doc["tagging"]["path_total_float"],
                "fragments": [
                    {
                        "fragment_seq": fragment["fragment_seq"],
                        "selected_tag": fragment["selected_tag"],
                        "selected_code": fragment["selected_code"],
                        "candidate_scores": fragment["candidate_scores"],
                        "weight_refs": fragment["weight_refs"],
                    }
                    for fragment in doc["tagging"]["fragments"]
                ],
            }
            for doc in sorted(result["doc_results"], key=lambda item: item["doc_seq"])
        ],
        "boundary_fragments": result["boundary_fragments"],
        "scheduler_plan": {
            "batches": result["scheduler"]["batches"],
            "pending": result["scheduler"]["pending"],
            "failures": result["scheduler"]["failures"],
            "cumulative_consumed_budget": result["scheduler"]["cumulative_consumed_budget"],
        },
        "jv1_text": result["jv1"]["text"],
        "jv2": result["jv2"],
        "sidecar_records": result["sidecar"]["records"],
        "downgrade_entries": result["sidecar"]["downgrade_entries"],
        "failures": result["failures"],
        "impact": result["impact"],
        "manifest_keys": [fragment["keys"] for fragment in result["manifest"]["fragments"]],
    }


def json_dumps(value: Any, sort_keys: bool = False) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, default=str)
