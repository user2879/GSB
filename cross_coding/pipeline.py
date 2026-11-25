from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .coding import (
    adjacent_quant_levels,
    adjacent_widths,
    cardinality_decision,
    effective_combo_fingerprint,
    encode_with_profile,
    point_plan as build_point_plan,
    profile_for_quant_levels,
    profile_for_width,
    reference_width,
)
from .combos import enumerate_combinations, task_sort_key
from .constants import CODE_POINT_BUDGET_PENDING
from .hashing import fingerprint
from .store import SnapshotStore, cache_key, load_previous_index


def load_previous_pending_keys(store: SnapshotStore | None) -> list[str]:
    if store is None:
        return []
    manifest = store.current_manifest()
    if not manifest:
        return []
    info = (manifest.get("stages") or {}).get("pending_queue") or {}
    relative = info.get("path")
    if not relative:
        return []
    path = Path(relative)
    if not path.is_absolute():
        snapshot_dir = manifest.get("snapshot_dir")
        base = store.root / snapshot_dir if snapshot_dir else store.root
        path = base / relative
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [item["canonical_key"] for item in payload.get("pending_queue", []) if item.get("canonical_key")]


def current_profile(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoding_width_bits": int(policy["encoding_width_bits"]),
        "fixed_point_bits": int(policy["fixed_point_bits"]),
        "quantization_levels": int(policy["quantization_levels"]),
    }


def stage_strategy_fingerprint(stage: str, policy: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"stage": stage, "version": 1}
    if stage == "enumeration":
        payload.update(
            {
                "combination_enumeration": policy.get("combination_enumeration", "connected_subsets"),
            }
        )
    elif stage == "coding":
        payload.update({"hash": "blake2b-256-mod-width", "missing_encoding": "stable-sentinel-v1"})
    elif stage == "cardinality":
        payload.update({"reference_width": "next_allowed_or_double", "adjacent_profiles": True})
    elif stage == "adjudication":
        payload.update(
            {
                "cardinality_budget": policy["cardinality_budget"],
                "over_cardinality_strategy": policy["over_cardinality_strategy"],
                "encoding_point_budget": policy["encoding_point_budget"],
            }
        )
    if extra:
        payload.update(extra)
    return fingerprint("strategy", payload)


def _policy_changes(old_policy: dict[str, Any] | None, policy: dict[str, Any]) -> list[str]:
    if not old_policy:
        return []
    fields = (
        "candidate_pairs",
        "max_order",
        "cardinality_budget",
        "encoding_point_budget",
        "encoding_width_bits",
        "fixed_point_bits",
        "quantization_levels",
        "over_cardinality_strategy",
        "combination_enumeration",
    )
    changes = []
    for field in fields:
        if old_policy.get(field) != policy.get(field):
            changes.append(f"policy_parameter_changed:{field}")
    return changes


def invalidation_reasons(
    combo: dict[str, Any],
    previous_index: dict[str, Any],
    previous_manifest: dict[str, Any] | None,
    policy: dict[str, Any],
    *,
    stage: str,
) -> list[str]:
    old = previous_index.get(combo["canonical_key"])
    if not old:
        return ["new_combination_or_previous_snapshot_absent"]

    reasons: list[str] = []
    old_columns = old.get("column_fingerprints", {})
    for name, current_fp in combo["column_fingerprints"].items():
        if old_columns.get(name) != current_fp:
            reasons.append(f"column_content_changed:{name}")

    old_pairs = old.get("reusable_pairs", [])
    current_pairs = combo.get("reusable_pairs", [])
    if sorted(map(tuple, old_pairs)) != sorted(map(tuple, current_pairs)):
        reasons.append("candidate_pair_membership_changed")

    old_pair_fps = old.get("reusable_pair_fingerprints", {})
    for pair_key, pair_fp in combo.get("reusable_pair_fingerprints", {}).items():
        if old_pair_fps.get(pair_key) != pair_fp and "candidate_pair_content_changed" not in reasons:
            reasons.append("candidate_pair_content_changed")

    old_manifest_profile = (previous_manifest or {}).get("width_profile") or {}
    profile = current_profile(policy)
    if old_manifest_profile.get("encoding_width_bits") != profile["encoding_width_bits"]:
        reasons.append("encoding_width_changed")
    if old_manifest_profile.get("quantization_levels") != profile["quantization_levels"]:
        reasons.append("quantization_level_changed")
    if old_manifest_profile.get("fixed_point_bits") != profile["fixed_point_bits"]:
        reasons.append("fixed_point_width_changed")

    if stage == "adjudication":
        old_policy = (previous_manifest or {}).get("normalized_policy")
        reasons.extend(_policy_changes(old_policy, policy))

    if not reasons and old.get("content_fingerprint") != combo["content_fingerprint"]:
        reasons.append("combination_content_fingerprint_changed")
    if not reasons and old.get("combo_fingerprint") != combo["combo_fingerprint"]:
        reasons.append("combination_fingerprint_changed")
    return reasons or ["content_addressed_key_miss"]


def _ledger_entry(
    stage: str,
    combo: dict[str, Any],
    key: str,
    status: str,
    basis: list[str],
    artifact_digest: str | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "canonical_key": combo["canonical_key"],
        "columns": combo["columns"],
        "combo_fingerprint": combo["combo_fingerprint"],
        "cache_key": key,
        "result": "reuse_hit" if status == "hit" else status,
        "basis": basis if status != "hit" else ["cache_key_and_digest_match"],
        "artifact_digest": artifact_digest,
        "stable_sort_key": task_sort_key(combo["combo_fingerprint"], stage),
    }


def build_cardinality_diagnostic(combo: dict[str, Any], table: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    profile = current_profile(policy)
    encoding = encode_with_profile(combo, table, profile)
    ref_width = reference_width(int(policy["encoding_width_bits"]))
    reference_encoding = encode_with_profile(combo, table, profile_for_width(policy, ref_width))
    plus_one_width = int(policy["encoding_width_bits"]) + 1
    plus_one_encoding = encode_with_profile(
        combo, table, profile_for_width(policy, plus_one_width)
    )

    adjacent: dict[str, Any] = {}
    for label, width in adjacent_widths(int(policy["encoding_width_bits"])):
        adjacent[label] = encode_with_profile(combo, table, profile_for_width(policy, width))
    for label, levels in adjacent_quant_levels(int(policy["quantization_levels"])):
        adjacent[label] = encode_with_profile(combo, table, profile_for_quant_levels(policy, levels))

    return {
        "current": _encoding_summary(encoding),
        "reference_high_width": {
            "width_bits": ref_width,
            **_encoding_summary(reference_encoding),
        },
        "higher_width_reference": {
            "width_bits": ref_width,
            **_encoding_summary(reference_encoding),
        },
        "reference_plus_one_bit": {
            "width_bits": plus_one_width,
            **_encoding_summary(plus_one_encoding),
        },
        "adjacent_profiles": {
            label: {
                "profile": item["profile"],
                "combo_fingerprint": item["combo_fingerprint"],
                **_encoding_summary(item),
                "codes": item["codes"],
            }
            for label, item in adjacent.items()
        },
        "dictionary_digest": encoding["dictionary_digest"],
    }


def _encoding_summary(encoding: dict[str, Any]) -> dict[str, Any]:
    return {
        "combo_fingerprint": encoding["combo_fingerprint"],
        "profile": encoding["profile"],
        "codes": encoding["codes"],
        "code_counts": encoding["code_counts"],
        "total_count": encoding["total_count"],
        "unique_count": encoding["unique_count"],
        "collision_count": encoding["collision_count"],
        "collision_rate": encoding["collision_rate"],
        "collision_pair_count": encoding["collision_pair_count"],
        "colliding_code_count": encoding["colliding_code_count"],
        "width_capacity": encoding["width_capacity"],
        "dictionary_digest": encoding["dictionary_digest"],
    }


def build_flip_attribution(
    combo: dict[str, Any],
    encoding: dict[str, Any],
    decision: dict[str, Any],
    cardinality_artifact: dict[str, Any],
    policy: dict[str, Any],
    point_plans_by_width: dict[int, dict[str, Any]],
    rank: int,
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    current_point_plan = point_plans_by_width[int(policy["encoding_width_bits"])]
    current_point_record = next(
        record for record in current_point_plan["records"].values() if record["rank"] == rank
    )
    current_pending_ranks = [
        record["rank"]
        for record in current_point_plan["records"].values()
        if record["point_status"] == "pending"
    ]

    for label, adjacent in cardinality_artifact.get("adjacent_profiles", {}).items():
        profile = adjacent["profile"]
        adjacent_decision = cardinality_decision(adjacent, policy)
        width = int(profile["encoding_width_bits"])
        adjacent_point_plan = point_plans_by_width.get(width)
        adjacent_point_record = None
        point_order_change = False
        if adjacent_point_plan is not None:
            adjacent_point_record = next(
                record for record in adjacent_point_plan["records"].values() if record["rank"] == rank
            )
            point_order_change = (
                adjacent_point_record["point_status"] != current_point_record["point_status"]
                or [
                    record["rank"]
                    for record in adjacent_point_plan["records"].values()
                    if record["point_status"] == "pending"
                ]
                != current_pending_ranks
            )

        salt_fingerprint_change = adjacent["combo_fingerprint"] != encoding["combo_fingerprint"]
        integer_change = adjacent["codes"] != encoding["codes"]
        unique_change = adjacent["unique_count"] != encoding["unique_count"]
        collision_change = adjacent["collision_count"] != encoding["collision_count"]
        hash_salt_diagnostic_change = (
            integer_change or unique_change or collision_change
        )
        bucket_change = (
            adjacent_decision["bucket_count"] != decision["bucket_count"]
            or adjacent_decision["action"] != decision["action"]
        )
        verdict_change = (
            adjacent_decision["verdict"] != decision["verdict"]
            or point_order_change
        )
        sources: list[str] = []
        if hash_salt_diagnostic_change:
            sources.append("hash_salt_change")
        if bucket_change:
            sources.append("truncation_bucket_count_change")
        if point_order_change:
            sources.append("point_budget_ruling_order_change")
        if not sources and not verdict_change:
            sources.append("no_diagnostic_flip")

        scenarios.append(
            {
                "scenario": label,
                "profile": profile,
                "combo_fingerprint": adjacent["combo_fingerprint"],
                "salt_fingerprint_change": salt_fingerprint_change,
                "hash_salt_change": hash_salt_diagnostic_change,
                "encoded_integer_change": integer_change,
                "unique_count_change": unique_change,
                "collision_count_change": collision_change,
                "truncation_bucket_count_change": bucket_change,
                "point_budget_order_change": point_order_change,
                "verdict_change": verdict_change,
                "difference_sources": sources,
                "current": {
                    "unique_count": encoding["unique_count"],
                    "collision_count": encoding["collision_count"],
                    "bucket_count": decision["bucket_count"],
                    "verdict": decision["verdict"],
                    "point_status": current_point_record["point_status"],
                },
                "adjacent": {
                    "unique_count": adjacent["unique_count"],
                    "collision_count": adjacent["collision_count"],
                    "bucket_count": adjacent_decision["bucket_count"],
                    "verdict": adjacent_decision["verdict"],
                    "point_status": adjacent_point_record["point_status"] if adjacent_point_record else None,
                },
            }
        )

    changed = [item for item in scenarios if item["difference_sources"] != ["no_diagnostic_flip"]]
    source_counts: dict[str, int] = {}
    for scenario in changed:
        for source in scenario["difference_sources"]:
            source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "current_profile": current_profile(policy),
        "reference_high_width": cardinality_artifact.get("reference_high_width"),
        "scenarios": scenarios,
        "changed_scenario_count": len(changed),
        "source_counts": source_counts,
    }


def numeric_quantization_report(table: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fixed_bits = int(policy["fixed_point_bits"])
    levels = int(policy["quantization_levels"])
    scale = 1 << fixed_bits
    for name in table["order"]:
        column = table["columns"][name]
        if column["type"] != "numeric":
            continue
        low = float(column.get("min", 0))
        high = float(column.get("max", 100))
        span = max(1e-12, high - low)
        rows = []
        bins: dict[int, int] = {}
        fixed_digests = []
        for index, value in enumerate(column["values"]):
            if value is None:
                fixed_int = None
                bin_index = None
            else:
                fixed_int = int(round(float(value) * scale))
                bin_index = min(levels - 1, max(0, int((float(value) - low) / span * levels)))
                bins[bin_index] = bins.get(bin_index, 0) + 1
                fixed_digests.append(fixed_int)
            if index < 12:
                rows.append({"row": index, "raw_value": value, "fixed_point_integer": fixed_int, "quantile_bin": bin_index})
        result[name] = {
            "fixed_point_bits": fixed_bits,
            "quantization_levels": levels,
            "scale": scale,
            "sample_rows": rows,
            "bin_counts": {str(key): bins[key] for key in sorted(bins)},
            "quantization_digest": fingerprint(
                "numeric-quantization",
                {"fixed_bits": fixed_bits, "levels": levels, "integers": fixed_digests, "bins": bins},
            ),
        }
    return result


def run_pipeline(
    bundle: dict[str, Any],
    state_dir: str | None = None,
    *,
    parallel: bool = True,
    command_cursor: int | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    policy = copy.deepcopy(bundle["policy"])
    table = bundle["table"]
    profile = current_profile(policy)
    store = SnapshotStore(state_dir) if state_dir else None
    recovery = store.recover() if store else {"status": "in_memory", "global_version": 0}
    previous_manifest = store.current_manifest() if store else None
    previous_index = load_previous_index(store)
    previous_pending_keys = load_previous_pending_keys(store)
    expected_version = int(previous_manifest["global_version"]) if previous_manifest else 0
    if command_cursor is None:
        command_cursor = int((previous_manifest or {}).get("completed_command_cursor", 0)) + 1

    combinations_list = enumerate_combinations(policy, table, encoding_profile=profile)
    current_width = int(policy["encoding_width_bits"])
    point_budget = int(policy["encoding_point_budget"])
    point_plan = build_point_plan(combinations_list, table, point_budget, current_width)
    point_plans_by_width: dict[int, dict[str, Any]] = {current_width: point_plan}
    for _label, width in adjacent_widths(current_width):
        if width not in point_plans_by_width:
            point_plans_by_width[width] = build_point_plan(
                combinations_list, table, point_budget, width
            )

    enum_strategy_fp = stage_strategy_fingerprint("enumeration", policy)
    coding_strategy_fp = stage_strategy_fingerprint("coding", policy)
    cardinality_strategy_fp = stage_strategy_fingerprint("cardinality", policy)
    point_plan_fp = fingerprint("point-plan", point_plan["records"])
    adjudication_strategy_fp = stage_strategy_fingerprint(
        "adjudication",
        policy,
        {
            "point_plan_fingerprint": point_plan_fp,
            "adjacent_point_plan_fingerprints": {
                str(width): fingerprint("point-plan", plan["records"])
                for width, plan in point_plans_by_width.items()
            },
        },
    )

    concurrency_state = {"active": 0, "max_observed": 0, "lock": Lock()}

    def process_combo(rank_and_combo: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        rank, combo = rank_and_combo
        with concurrency_state["lock"]:
            concurrency_state["active"] += 1
            concurrency_state["max_observed"] = max(
                concurrency_state["max_observed"], concurrency_state["active"]
            )
        try:
            return _process_one_combo(
                rank=rank,
                combo=combo,
                bundle=bundle,
                policy=policy,
                table=table,
                store=store,
                previous_index=previous_index,
                previous_manifest=previous_manifest,
                profile=profile,
                point_plan=point_plan,
                point_plans_by_width=point_plans_by_width,
                strategy_fps={
                    "enumeration": enum_strategy_fp,
                    "coding": coding_strategy_fp,
                    "cardinality": cardinality_strategy_fp,
                    "adjudication": adjudication_strategy_fp,
                },
            )
        finally:
            with concurrency_state["lock"]:
                concurrency_state["active"] -= 1

    ranked = list(enumerate(combinations_list))
    results_by_rank: dict[int, dict[str, Any]] = {}
    configured_workers = int(policy["max_workers"])
    pending_priority = {key: index for index, key in enumerate(previous_pending_keys)}

    def submission_priority(item: tuple[int, dict[str, Any]]):
        rank, combo = item
        previous_pending_rank = pending_priority.get(combo["canonical_key"])
        if previous_pending_rank is None:
            return (1, rank, task_sort_key(combo["combo_fingerprint"], "coding"))
        return (0, previous_pending_rank, task_sort_key(combo["combo_fingerprint"], "coding"))

    submission_order = sorted(ranked, key=submission_priority)
    scheduler_order = [combo["canonical_key"] for _rank, combo in submission_order]
    if parallel and ranked:
        with ThreadPoolExecutor(max_workers=configured_workers) as executor:
            future_map = {
                executor.submit(process_combo, item): item[0] for item in submission_order
            }
            for future in as_completed(future_map):
                rank = future_map[future]
                results_by_rank[rank] = future.result()
        execution_mode = "thread_pool"
    else:
        for item in submission_order:
            results_by_rank[item[0]] = process_combo(item)
        execution_mode = "serial"

    combo_views = [results_by_rank[index]["view"] for index in range(len(ranked))]
    ledger_entries: list[dict[str, Any]] = []
    coding_artifacts: dict[str, Any] = {}
    cardinality_artifacts: dict[str, Any] = {}
    adjudication_artifacts: list[dict[str, Any]] = []

    for index in range(len(ranked)):
        result = results_by_rank[index]
        ledger_entries.extend(result["ledger_entries"])
        if result.get("coding_artifact"):
            coding_artifacts[result["combo"]["canonical_key"]] = result["coding_artifact"]
        if result.get("cardinality_artifact"):
            cardinality_artifacts[result["combo"]["canonical_key"]] = result["cardinality_artifact"]
        if result.get("adjudication_artifact"):
            adjudication_artifacts.append(result["adjudication_artifact"])

    removed_entries = _removed_combo_entries(previous_index, combinations_list)
    ledger_entries = removed_entries + ledger_entries

    pending_queue = [
        {
            "rank": view["point"]["rank"],
            "canonical_key": view["canonical_key"],
            "columns": view["columns"],
            "combo_fingerprint": view["combo_fingerprint"],
            "code": CODE_POINT_BUDGET_PENDING,
            "stage": "adjudication",
            "message": (
                f"encoding point cost {view['point']['point_cost']} exceeds remaining budget "
                f"at stable combination rank {view['point']['rank']}"
            ),
            "point_cost": view["point"]["point_cost"],
            "cumulative_before": view["point"]["cumulative_before"],
            "point_budget": point_budget,
        }
        for view in combo_views
        if view["point"]["point_status"] == "pending"
    ]

    reuse_summary = _reuse_summary(ledger_entries)
    code_counts: dict[str, int] = {}
    for item in pending_queue:
        code_counts[CODE_POINT_BUDGET_PENDING] = code_counts.get(CODE_POINT_BUDGET_PENDING, 0) + 1

    report: dict[str, Any] = {
        "ok": True,
        "status": "completed_with_pending" if pending_queue else "completed",
        "execution": {
            "mode": execution_mode,
            "configured_max_workers": configured_workers,
            "thread_pool_max_workers": min(configured_workers, max(1, len(ranked))),
            "observed_max_concurrency": concurrency_state["max_observed"],
            "previous_pending_priority": previous_pending_keys,
            "scheduler_submission_order": scheduler_order,
        },
        "config_paths": {
            "feature": bundle.get("feature_path"),
            "policy": bundle.get("policy_path"),
        },
        "config_fingerprint": bundle["config_fingerprint"],
        "policy": policy,
        "width_profile": profile,
        "table": _public_table(table),
        "numeric_quantization": numeric_quantization_report(table, policy),
        "enumeration_order": [combo["canonical_key"] for combo in combinations_list],
        "combinations": combo_views,
        "pending_queue": pending_queue,
        "failures": [],
        "failure_list": [],
        "business_code_counts": code_counts,
        "ledger": {
            "entries": ledger_entries,
            "summary": reuse_summary,
        },
        "reuse_summary": reuse_summary,
        "point_plan": point_plan,
        "recovery": recovery,
        "command_cursor": command_cursor,
    }

    if store and commit:
        enum_payload = {
            "combinations": [_public_enum_item(combo) for combo in combinations_list]
        }
        coding_payload = {"artifacts": coding_artifacts}
        cardinality_payload = {"artifacts": cardinality_artifacts}
        pending_payload = {"pending_queue": pending_queue}
        ledger_payload = {"entries": ledger_entries, "summary": reuse_summary}
        manifest = store.commit(
            stage_payloads={
                "enumeration": enum_payload,
                "coding": coding_payload,
                "cardinality": cardinality_payload,
                "adjudication": {"artifacts": adjudication_artifacts, "views": combo_views},
                "pending_queue": pending_payload,
                "ledger": ledger_payload,
            },
            policy_hash=policy["policy_fingerprint"],
            width_profile=profile,
            command_cursor=command_cursor,
            expected_version=expected_version,
            ledger_entries=ledger_entries,
            extra_manifest={
                "status": report["status"],
                "normalized_policy": policy,
                "table_fingerprint": table["table_fingerprint"],
                "completed_command_cursor": command_cursor,
                "snapshot_commit_interval": policy.get("snapshot_commit_interval"),
                "business_code_counts": code_counts,
            },
        )
        report["manifest"] = {
            key: value
            for key, value in manifest.items()
            if key not in ("normalized_policy",)
        }
    else:
        report["manifest"] = None
    report["comparison_view"] = comparison_view(report)
    return report


def _process_one_combo(
    *,
    rank: int,
    combo: dict[str, Any],
    bundle: dict[str, Any],
    policy: dict[str, Any],
    table: dict[str, Any],
    store: SnapshotStore | None,
    previous_index: dict[str, Any],
    previous_manifest: dict[str, Any] | None,
    profile: dict[str, Any],
    point_plan: dict[str, Any],
    point_plans_by_width: dict[int, dict[str, Any]],
    strategy_fps: dict[str, str],
) -> dict[str, Any]:
    ledger_entries: list[dict[str, Any]] = []
    keys = {
        stage: cache_key(stage, combo["combo_fingerprint"], strategy_fp, profile)
        for stage, strategy_fp in strategy_fps.items()
    }

    enum_reasons = invalidation_reasons(
        combo, previous_index, previous_manifest, policy, stage="enumeration"
    )
    enum_artifact = {"combo": _public_enum_item(combo)}
    cached_enum = store.artifacts.get("enumeration", keys["enumeration"]) if store else None
    if cached_enum is not None:
        ledger_entries.append(_ledger_entry("enumeration", combo, keys["enumeration"], "hit", []))
    else:
        if store:
            store.artifacts.put("enumeration", keys["enumeration"], enum_artifact)
        ledger_entries.append(
            _ledger_entry(
                "enumeration",
                combo,
                keys["enumeration"],
                "recomputed",
                enum_reasons,
                fingerprint("enum-artifact", enum_artifact),
            )
        )

    point_record = next(record for record in point_plan["records"].values() if record["rank"] == rank)
    view: dict[str, Any] = {
        "rank": rank,
        "canonical_key": combo["canonical_key"],
        "columns": combo["columns"],
        "order": combo["order"],
        "content_fingerprint": combo["content_fingerprint"],
        "combo_fingerprint": combo["combo_fingerprint"],
        "fingerprint": combo["fingerprint"],
        "column_fingerprints": combo["column_fingerprints"],
        "reusable_pairs": combo["reusable_pairs"],
        "reusable_pair_fingerprints": combo["reusable_pair_fingerprints"],
        "reusable_pair_salts": combo["reusable_pair_salts"],
        "task_sort_keys": {
            stage: task_sort_key(combo["combo_fingerprint"], stage)
            for stage in ("enumeration", "coding", "cardinality", "adjudication")
        },
        "cache_keys": keys,
        "point": point_record,
        "point_status": point_record["point_status"],
        "status": "pending" if point_record["point_status"] == "pending" else "processing",
        "encoding": None,
        "cardinality_diagnostic": None,
        "cardinality_ruling": None,
        "budget_decision": None,
        "flip_attribution": None,
    }

    if point_record["point_status"] == "pending":
        pending_decision = {
            "verdict": "POINT_BUDGET_PENDING",
            "code": CODE_POINT_BUDGET_PENDING,
            "point_status": "pending",
            "cardinality_status": "not_evaluated_before_point_cutoff",
        }
        view["budget_decision"] = pending_decision
        view["status"] = "pending"
        adjudication_artifact = {
            "decision": pending_decision,
            "rank": rank,
            "canonical_key": combo["canonical_key"],
            "combo_fingerprint": combo["combo_fingerprint"],
            "content_fingerprint": combo["content_fingerprint"],
            "column_fingerprints": combo["column_fingerprints"],
        }
        if store:
            store.artifacts.put("adjudication", keys["adjudication"], adjudication_artifact)
        ledger_entries.append(
            _ledger_entry(
                "coding",
                combo,
                keys["coding"],
                "pending",
                ["not_processed_due_to_E_POINT_BUDGET_PENDING"],
            )
        )
        ledger_entries.append(
            _ledger_entry(
                "cardinality",
                combo,
                keys["cardinality"],
                "pending",
                ["not_processed_due_to_E_POINT_BUDGET_PENDING"],
            )
        )
        ledger_entries.append(
            _ledger_entry(
                "adjudication",
                combo,
                keys["adjudication"],
                "recomputed",
                ["stable_point_budget_cutoff"],
                fingerprint("adjudication", adjudication_artifact),
            )
        )
        return {
            "view": view,
            "ledger_entries": ledger_entries,
            "coding_artifact": None,
            "cardinality_artifact": None,
            "adjudication_artifact": adjudication_artifact,
            "combo": combo,
        }

    coding_reasons = invalidation_reasons(
        combo, previous_index, previous_manifest, policy, stage="coding"
    )
    coding_artifact = store.artifacts.get("coding", keys["coding"]) if store else None
    if coding_artifact is not None:
        ledger_entries.append(_ledger_entry("coding", combo, keys["coding"], "hit", []))
    else:
        encoding = encode_with_profile(combo, table, profile)
        coding_artifact = _encoding_summary(encoding)
        coding_artifact["value_dictionary"] = encoding["value_dictionary"]
        if store:
            store.artifacts.put("coding", keys["coding"], coding_artifact)
        ledger_entries.append(
            _ledger_entry(
                "coding",
                combo,
                keys["coding"],
                "recomputed",
                coding_reasons,
                coding_artifact["dictionary_digest"],
            )
        )

    cardinality_artifact = store.artifacts.get("cardinality", keys["cardinality"]) if store else None
    if cardinality_artifact is not None:
        ledger_entries.append(_ledger_entry("cardinality", combo, keys["cardinality"], "hit", []))
    else:
        encoding_for_diag = encode_with_profile(combo, table, profile)
        cardinality_artifact = build_cardinality_diagnostic(combo, table, policy)
        if store:
            store.artifacts.put("cardinality", keys["cardinality"], cardinality_artifact)
        ledger_entries.append(
            _ledger_entry(
                "cardinality",
                combo,
                keys["cardinality"],
                "recomputed",
                coding_reasons,
                cardinality_artifact["dictionary_digest"],
            )
        )

    view["cardinality_diagnostic"] = {
        key: value
        for key, value in cardinality_artifact.items()
        if key != "adjacent_profiles"
    }
    view["encoding"] = {
        key: value for key, value in coding_artifact.items() if key != "value_dictionary"
    }

    cached_adjudication = (
        store.artifacts.get("adjudication", keys["adjudication"]) if store else None
    )
    if cached_adjudication is not None:
        ledger_entries.append(_ledger_entry("adjudication", combo, keys["adjudication"], "hit", []))
        adjudication_artifact = cached_adjudication
        budget_decision = cached_adjudication["decision"]
        flip = cached_adjudication.get("flip_attribution")
        view["flip_attribution"] = flip
        view["cardinality_ruling"] = budget_decision
        view["budget_decision"] = budget_decision
        view["status"] = budget_decision.get("action") or budget_decision.get("verdict")
    else:
        encoding = coding_artifact
        decision = cardinality_decision(encoding, policy)
        flip = build_flip_attribution(
            combo,
            encoding,
            decision,
            cardinality_artifact,
            policy,
            point_plans_by_width,
            rank,
        )
        budget_decision = {
            **{k: v for k, v in decision.items() if k != "final_codes"},
            "point": point_record,
            "point_status": "admitted",
            "combined_verdict": decision["verdict"],
        }
        view["flip_attribution"] = flip
        view["cardinality_ruling"] = budget_decision
        view["budget_decision"] = budget_decision
        view["status"] = decision["action"]

        adjudication_artifact = {
            "cache_key": keys["adjudication"],
            "rank": rank,
            "canonical_key": combo["canonical_key"],
            "combo_fingerprint": combo["combo_fingerprint"],
            "content_fingerprint": combo["content_fingerprint"],
            "column_fingerprints": combo["column_fingerprints"],
            "decision": budget_decision,
            "flip_attribution": flip,
        }
        if store:
            store.artifacts.put("adjudication", keys["adjudication"], adjudication_artifact)
        adj_reasons = invalidation_reasons(
            combo, previous_index, previous_manifest, policy, stage="adjudication"
        )
        ledger_entries.append(
            _ledger_entry(
                "adjudication",
                combo,
                keys["adjudication"],
                "recomputed",
                adj_reasons,
                fingerprint("adjudication", adjudication_artifact),
            )
        )

    return {
        "view": view,
        "ledger_entries": ledger_entries,
        "coding_artifact": coding_artifact,
        "cardinality_artifact": cardinality_artifact,
        "adjudication_artifact": adjudication_artifact,
        "combo": combo,
    }


def _public_enum_item(combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_key": combo["canonical_key"],
        "columns": combo["columns"],
        "order": combo["order"],
        "content_fingerprint": combo["content_fingerprint"],
        "combo_fingerprint": combo["combo_fingerprint"],
        "fingerprint": combo["fingerprint"],
        "column_fingerprints": combo["column_fingerprints"],
        "reusable_pairs": combo["reusable_pairs"],
        "reusable_pair_fingerprints": combo["reusable_pair_fingerprints"],
    }


def _public_table(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": table["seed"],
        "row_count": table["row_count"],
        "order": table["order"],
        "table_fingerprint": table["table_fingerprint"],
        "columns": {
            name: {
                key: value
                for key, value in column.items()
                if key not in ("rng",)
            }
            for name, column in table["columns"].items()
        },
    }


def _removed_combo_entries(
    previous_index: dict[str, Any],
    current_combinations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_keys = {combo["canonical_key"] for combo in current_combinations}
    entries = []
    for key in sorted(set(previous_index) - current_keys):
        old = previous_index[key]
        combo_like = {
            "canonical_key": key,
            "columns": old.get("columns", key.split("|")),
            "combo_fingerprint": old.get("combo_fingerprint", "removed"),
        }
        entries.append(
            {
                "stage": "enumeration",
                "canonical_key": key,
                "columns": combo_like["columns"],
                "combo_fingerprint": combo_like["combo_fingerprint"],
                "cache_key": None,
                "result": "invalidated_removed",
                "basis": ["combination_no_longer_enumerated"],
                "artifact_digest": None,
                "stable_sort_key": task_sort_key(combo_like["combo_fingerprint"], "enumeration"),
            }
        )
    return entries


def _reuse_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        stage = entry["stage"]
        result = entry["result"]
        summary.setdefault(stage, {})
        summary[stage][result] = summary[stage].get(result, 0) + 1
    totals: dict[str, int] = {}
    for stage_results in summary.values():
        for result, count in stage_results.items():
            totals[result] = totals.get(result, 0) + count
    return {"by_stage": summary, "totals": totals}


def comparison_view(report: dict[str, Any]) -> dict[str, Any]:
    combinations_list = []
    for combo in report["combinations"]:
        encoding = combo.get("encoding")
        ruling = combo.get("budget_decision") or {}
        combinations_list.append(
            {
                "rank": combo["rank"],
                "columns": combo["columns"],
                "content_fingerprint": combo["content_fingerprint"],
                "combo_fingerprint": combo["combo_fingerprint"],
                "cache_keys": combo["cache_keys"],
                "codes": None if not encoding else encoding["codes"],
                "unique_count": None if not encoding else encoding["unique_count"],
                "collision_count": None if not encoding else encoding["collision_count"],
                "collision_rate": None if not encoding else encoding["collision_rate"],
                "point_status": combo["point"]["point_status"],
                "point_rank": combo["point"]["rank"],
                "verdict": ruling.get("combined_verdict") or ruling.get("verdict"),
                "cardinality_action": ruling.get("action"),
                "bucket_count": ruling.get("bucket_count"),
                "final_unique_count": ruling.get("final_unique_count"),
            }
        )
    return {
        "status": report["status"],
        "width_profile": report["width_profile"],
        "policy_fingerprint": report["policy"]["policy_fingerprint"],
        "enumeration_order": report["enumeration_order"],
        "combinations": combinations_list,
        "pending_queue": [
            {
                "rank": item["rank"],
                "columns": item["columns"],
                "combo_fingerprint": item["combo_fingerprint"],
                "code": item["code"],
            }
            for item in report["pending_queue"]
        ],
        "failures": report["failures"],
        "business_code_counts": report["business_code_counts"],
    }
