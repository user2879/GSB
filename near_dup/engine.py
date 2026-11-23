"""Incremental deterministic near-duplicate ledger engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .canonical import sha256_json, stable_key
from .config import FingerprintConfig
from .errors import (
    E_CLUSTER_ID_CONFLICT,
    E_SIGNATURE_BUDGET_PENDING,
    BusinessError,
    Failure,
    sort_failures,
)
from .features import NormalizedText, normalize_and_window
from .hashing import (
    SignatureResult,
    compute_minhash,
    pair_similarity,
    signature_task_key,
    signed_projection_bucket,
    similarity_task_key,
)
from .snapshot import PRIMARY_STAGES, SnapshotStore
from .text_store import TextRecord, records_by_id


@dataclass(frozen=True)
class SimulatedCrash(Exception):
    stage: str

    def __str__(self) -> str:
        return f"simulated crash after stage {self.stage!r}"


def _empty_snapshot() -> dict[str, Any]:
    return {
        "manifest": {
            "schema_version": 1,
            "global_version": 0,
            "config_hash": None,
            "stage_versions": {stage: 0 for stage in PRIMARY_STAGES},
            "stage_hashes": {},
            "pending_signatures": [],
            "completed_texts": [],
            "signature_budget": {
                "batch_limit": 0,
                "batch_consumed": 0,
                "consumed_total": 0,
                "per_text_components": 0,
                "entries": [],
            },
        },
        "stages": {
            "normalization": {"texts": {}},
            "signatures": {"texts": {}},
            "buckets": {"texts": {}, "groups": {}},
            "ledger": {"clusters": [], "boundary_texts": [], "merge_log": []},
        },
    }


def _chunked(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


class NearDuplicateEngine:
    def __init__(self, snapshot_dir: str, *, crash_after_stage: str | None = None):
        if crash_after_stage not in {None, *PRIMARY_STAGES}:
            raise ValueError(f"unknown crash stage: {crash_after_stage}")
        self.store = SnapshotStore(snapshot_dir)
        self.crash_after_stage = crash_after_stage

    def latest_snapshot(self) -> dict[str, Any] | None:
        return self.store.load_latest()

    def recover_manifest(
        self,
        *,
        expected_version: int | None = None,
        expected_config_hash: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.latest_snapshot()
        return self.store.validate_recovery(
            snapshot,
            expected_version=expected_version,
            expected_config_hash=expected_config_hash,
        )

    def sync(
        self,
        records: Iterable[TextRecord],
        config: FingerprintConfig,
        *,
        external_failures: Iterable[Failure | dict[str, Any]] | None = None,
        runtime_injection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = records_by_id(records)
        current_ids = sorted(current)
        previous = self.store.load_latest() or _empty_snapshot()
        prev_manifest = previous["manifest"]
        prev_stage_versions = dict(prev_manifest.get("stage_versions", {}))
        for stage in PRIMARY_STAGES:
            prev_stage_versions.setdefault(stage, 0)
        prev_stage_hashes = dict(prev_manifest.get("stage_hashes", {}))
        prev_norm = previous["stages"].get("normalization", {}).get("texts", {})
        prev_sigs = previous["stages"].get("signatures", {}).get("texts", {})
        prev_bucket_stage = previous["stages"].get("buckets", {})
        prev_buckets = prev_bucket_stage.get("texts", {})
        prev_groups = prev_bucket_stage.get("groups", {})
        prev_config = prev_manifest.get("config") or {}
        prev_ids = set(prev_norm)

        deleted_ids = sorted(prev_ids - set(current_ids))
        added_ids = sorted(set(current_ids) - prev_ids)
        replaced_ids = sorted(
            text_id
            for text_id in set(current_ids) & prev_ids
            if prev_norm.get(text_id, {}).get("content_fingerprint") != current[text_id].content_fingerprint
        )

        normalization_entries: dict[str, dict[str, Any]] = {}
        normalized_objects: dict[str, NormalizedText] = {}
        normalization_reused: list[str] = []
        normalization_invalidated: list[str] = []
        for text_id in current_ids:
            record = current[text_id]
            normalize_key = sha256_json(
                {
                    "content_fingerprint": record.content_fingerprint,
                    "algorithm_fingerprint": config.algorithm_fingerprint,
                    "window_count": config.window_count,
                    "normalization_version": 1,
                }
            )
            cached = prev_norm.get(text_id)
            normalized: NormalizedText | None = None
            if cached and cached.get("cache_key") == normalize_key:
                try:
                    normalized = NormalizedText.from_payload(cached["normalized"])
                except Exception:
                    normalized = None
            if normalized is None:
                normalized = normalize_and_window(record, config)
                normalization_invalidated.append(text_id)
            else:
                normalization_reused.append(text_id)
            normalized_objects[text_id] = normalized
            normalization_entries[text_id] = {
                "cache_key": normalize_key,
                "content_fingerprint": record.content_fingerprint,
                "task_key": stable_key(text_id, "normalization"),
                "normalized": normalized.payload(),
            }

        if self.crash_after_stage == "normalization":
            raise SimulatedCrash("normalization")

        signature_entries: dict[str, dict[str, Any]] = {}
        signature_keys: dict[str, str] = {}
        needs_signature: list[str] = []
        signature_reused: list[str] = []
        for text_id in current_ids:
            signature_key = sha256_json(
                {
                    "normalization_fingerprint": normalized_objects[text_id].fingerprint,
                    "algorithm_fingerprint": config.algorithm_fingerprint,
                    "signature_dimension": config.signature_dimension,
                    "fixed_decimal_places": config.fixed_decimal_places,
                    "quantization_levels": config.quantization_levels,
                    "signature_version": 1,
                }
            )
            signature_keys[text_id] = signature_key
            cached = prev_sigs.get(text_id)
            valid = False
            if cached and cached.get("cache_key") == signature_key:
                try:
                    signature_result = SignatureResult.from_payload(cached)
                    valid = len(signature_result.signature) == config.signature_dimension
                except Exception:
                    valid = False
            if valid:
                signature_entries[text_id] = dict(cached)
                signature_reused.append(text_id)
            else:
                needs_signature.append(text_id)

        needs_set = set(needs_signature)
        previous_pending = [
            text_id for text_id in prev_manifest.get("pending_signatures", []) if text_id in needs_set
        ]
        pending_set = set(previous_pending)
        candidates = previous_pending + [text_id for text_id in needs_signature if text_id not in pending_set]
        per_text_cost = config.signature_dimension
        affordable = min(len(candidates), config.signature_batch_budget // per_text_cost)
        work = candidates[:affordable]
        pending_after_budget = candidates[affordable:]

        batch_consumed = 0
        consumed_total = int(
            prev_manifest.get("signature_budget", {}).get("consumed_total", 0)
        )
        computed_ids: list[str] = []
        cluster_invalidated: set[str] = set()
        cluster_reused: set[str] = set()
        bucket_invalidated: list[str] = []
        bucket_reused: list[str] = []
        last_groups = prev_groups
        last_result: dict[str, Any] | None = None
        last_version = int(prev_manifest.get("global_version", 0))
        chunks = _chunked(work, config.snapshot_commit_interval)
        if not chunks:
            chunks = [[]]

        for chunk_index, chunk in enumerate(chunks):
            if chunk:
                computed = self._compute_signatures(chunk, normalized_objects, signature_keys, config)
                for text_id in chunk:
                    result = computed[text_id]
                    signature_entries[text_id] = {
                        "cache_key": signature_keys[text_id],
                        "signature": list(result.signature),
                        "layout": [{"window": window, "lane": lane} for window, lane in result.layout],
                        "task_key": signature_task_key(text_id),
                    }
                    computed_ids.append(text_id)
                batch_consumed += len(chunk) * per_text_cost

            remaining_chunks = chunks[chunk_index + 1 :]
            pending_now = [text_id for remaining in remaining_chunks for text_id in remaining] + pending_after_budget
            completed_ids = sorted(signature_entries)
            bucket_entries, groups, ledger, audit = self._build_buckets_and_ledger(
                completed_ids,
                normalized_objects,
                signature_entries,
                signature_keys,
                prev_buckets,
                last_groups,
                config,
                runtime_injection,
            )
            last_groups = groups
            cluster_invalidated.update(audit["cluster_invalidated_buckets"])
            cluster_reused.update(audit["cluster_reused_buckets"])
            for text_id in audit["bucket_invalidated"]:
                if text_id not in bucket_invalidated:
                    bucket_invalidated.append(text_id)
            for text_id in audit["bucket_reused"]:
                if text_id not in bucket_reused:
                    bucket_reused.append(text_id)

            pending_failures = [
                Failure(
                    E_SIGNATURE_BUDGET_PENDING,
                    "signature_budget",
                    "signature components remain queued for the next batch",
                    item_id=text_id,
                    detail={"per_text_components": per_text_cost},
                )
                for text_id in pending_now
            ]
            failures = sort_failures(list(external_failures or []) + pending_failures)
            budget_entries = self._budget_entries(
                completed_ids,
                pending_now,
                signature_keys,
                signature_entries,
                computed_ids,
                per_text_cost,
            )
            manifest = {
                "schema_version": 1,
                "global_version": last_version + 1,
                "config_hash": config.config_hash,
                "algorithm_fingerprint": config.algorithm_fingerprint,
                "config": config.to_dict(),
                "completed_texts": completed_ids,
                "pending_signatures": pending_now,
                "signature_budget": {
                    "batch_limit": config.signature_batch_budget,
                    "batch_consumed": batch_consumed,
                    "consumed_total": consumed_total + batch_consumed,
                    "per_text_components": per_text_cost,
                    "entries": budget_entries,
                },
                "counts": {
                    "texts": len(current_ids),
                    "completed": len(completed_ids),
                    "pending": len(pending_now),
                    "clusters": len(ledger["clusters"]),
                },
            }
            stages = {
                "normalization": {"texts": normalization_entries},
                "signatures": {"texts": signature_entries},
                "buckets": {"texts": bucket_entries, "groups": groups},
                "ledger": {**ledger, "failures": failures},
            }
            config_changed_fields = sorted(
                key
                for key in set(prev_config) | set(config.to_dict())
                if prev_config.get(key) != config.to_dict().get(key)
            )
            audit_payload = {
                "config_changed_fields": config_changed_fields,
                "added_texts": added_ids,
                "replaced_texts": replaced_ids,
                "deleted_texts": deleted_ids,
                "normalization_reused": sorted(normalization_reused),
                "normalization_invalidated": sorted(normalization_invalidated),
                "signature_invalidated": sorted(needs_signature),
                "signature_reused": sorted(signature_reused),
                "signature_computed": sorted(computed_ids),
                "bucket_invalidated": sorted(bucket_invalidated),
                "bucket_reused": sorted(bucket_reused),
                "cluster_invalidated_buckets": sorted(cluster_invalidated),
                "cluster_reused_buckets": sorted(cluster_reused),
                "similarity_comparisons": sum(len(group.get("merge_log", [])) for group in groups.values()),
            }
            manifest = self._commit_with_stage_versions(
                manifest,
                stages,
                prev_stage_versions,
                prev_stage_hashes,
                audit_payload,
            )
            last_version = int(manifest["global_version"])
            prev_stage_versions = dict(manifest["stage_versions"])
            prev_stage_hashes = dict(manifest["stage_hashes"])
            last_result = {
                "manifest": manifest,
                "clusters": ledger["clusters"],
                "boundary_texts": ledger["boundary_texts"],
                "merge_log": ledger["merge_log"],
                "failures": failures,
                "audit": audit_payload,
                "signatures": {
                    text_id: signature_entries[text_id]["signature"]
                    for text_id in sorted(signature_entries)
                },
                "buckets": {
                    text_id: bucket_entries[text_id]["bucket"]
                    for text_id in sorted(bucket_entries)
                },
                "pending_signatures": pending_now,
                "snapshot_generation": manifest.get("generation"),
            }
            if self.crash_after_stage in {"signatures", "buckets", "ledger"}:
                raise SimulatedCrash(self.crash_after_stage)

        if last_result is None:
            raise AssertionError("engine produced no result")
        return last_result

    def _compute_signatures(
        self,
        text_ids: Sequence[str],
        normalized_objects: dict[str, NormalizedText],
        signature_keys: dict[str, str],
        config: FingerprintConfig,
    ) -> dict[str, SignatureResult]:
        ordered = sorted(text_ids, key=lambda text_id: (signature_task_key(text_id), text_id))
        with ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="near-dup-sign") as executor:
            futures = {
                text_id: executor.submit(compute_minhash, normalized_objects[text_id], config)
                for text_id in ordered
            }
            return {text_id: futures[text_id].result() for text_id in ordered}

    def _build_buckets_and_ledger(
        self,
        completed_ids: Sequence[str],
        normalized_objects: dict[str, NormalizedText],
        signature_entries: dict[str, dict[str, Any]],
        signature_keys: dict[str, str],
        prev_buckets: dict[str, Any],
        prev_groups: dict[str, Any],
        config: FingerprintConfig,
        runtime_injection: dict[str, Any] | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, list[str]]]:
        bucket_entries: dict[str, dict[str, Any]] = {}
        bucket_invalidated: list[str] = []
        bucket_reused: list[str] = []
        members_by_bucket: dict[str, list[str]] = {}
        for text_id in completed_ids:
            signature = tuple(int(value) for value in signature_entries[text_id]["signature"])
            cached = prev_buckets.get(text_id)
            cached_bucket = cached.get("bucket") if cached else None
            if cached_bucket is None:
                bucket = signed_projection_bucket(signature, config)
            bucket_key = sha256_json(
                {
                    "signature_key": signature_keys[text_id],
                    "bucket": cached_bucket if cached_bucket is not None else bucket,
                    "bucket_bits": config.bucket_bits,
                    "bucket_fingerprint": config.bucket_fingerprint,
                    "bucket_version": 1,
                }
            )
            if cached and cached.get("cache_key") == bucket_key:
                bucket = cached["bucket"]
                bucket_reused.append(text_id)
                entry = dict(cached)
            else:
                bucket = signed_projection_bucket(signature, config)
                bucket_key = sha256_json(
                    {
                        "signature_key": signature_keys[text_id],
                        "bucket": bucket,
                        "bucket_bits": config.bucket_bits,
                        "bucket_fingerprint": config.bucket_fingerprint,
                        "bucket_version": 1,
                    }
                )
                bucket_invalidated.append(text_id)
                entry = {
                    "cache_key": bucket_key,
                    "bucket": bucket,
                    "task_key": stable_key(text_id, "bucket"),
                }
            bucket_entries[text_id] = entry
            members_by_bucket.setdefault(entry["bucket"], []).append(text_id)

        cluster_params_key = sha256_json(
            {
                "algorithm_fingerprint": config.algorithm_fingerprint,
                "threshold_units": config.threshold_units,
                "bucket_bits": config.bucket_bits,
                "fixed_decimal_places": config.fixed_decimal_places,
                "quantization_levels": config.quantization_levels,
                "clustering_version": 1,
            }
        )
        current_text_key = lambda text_id: sha256_json(
            {
                "content_fingerprint": normalized_objects[text_id].fingerprint,
                "signature_key": signature_keys[text_id],
                "bucket_key": bucket_entries[text_id]["cache_key"],
                "text_cache_version": 1,
            }
        )

        groups: dict[str, Any] = {}
        cluster_invalidated: list[str] = []
        cluster_reused: list[str] = []
        with ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="near-dup-cmp") as executor:
            for bucket in sorted(members_by_bucket):
                members = sorted(members_by_bucket[bucket])
                member_rows = [
                    {
                        "text_id": text_id,
                        "signature_key": signature_keys[text_id],
                        "bucket_key": bucket_entries[text_id]["cache_key"],
                        "text_cache_key": current_text_key(text_id),
                    }
                    for text_id in members
                ]
                group_fingerprint = sha256_json(
                    {"cluster_params_key": cluster_params_key, "members": member_rows}
                )
                cached_group = prev_groups.get(bucket)
                if cached_group and cached_group.get("fingerprint") == group_fingerprint:
                    group = cached_group
                    cluster_reused.append(bucket)
                else:
                    group = self._cluster_bucket(
                        bucket,
                        members,
                        signature_entries,
                        current_text_key,
                        config,
                        executor,
                    )
                    group["fingerprint"] = group_fingerprint
                    group["cluster_params_key"] = cluster_params_key
                    cluster_invalidated.append(bucket)
                groups[bucket] = group

        clusters = [cluster for bucket in sorted(groups) for cluster in groups[bucket]["clusters"]]
        clusters.sort(key=lambda cluster: cluster["cluster_id"])
        ledger = {
            "clusters": clusters,
            "boundary_texts": sorted(
                [item for bucket in sorted(groups) for item in groups[bucket].get("boundary_texts", [])],
                key=lambda item: (tuple(item["text_ids"]), item["similarity_units"]),
            ),
            "merge_log": sorted(
                [item for bucket in sorted(groups) for item in groups[bucket].get("merge_log", [])],
                key=lambda item: (item["bucket"], item["text_ids"][0], item["text_ids"][1]),
            ),
        }
        if runtime_injection and runtime_injection.get("kind") == "cluster_id_conflict":
            ledger = self._inject_cluster_id_conflict(ledger, runtime_injection)
        self._validate_ledger(ledger, completed_ids, runtime_injection)
        audit = {
            "bucket_invalidated": bucket_invalidated,
            "bucket_reused": bucket_reused,
            "cluster_invalidated_buckets": cluster_invalidated,
            "cluster_reused_buckets": cluster_reused,
        }
        return bucket_entries, groups, ledger, audit

    def _cluster_bucket(
        self,
        bucket: str,
        members: Sequence[str],
        signature_entries: dict[str, dict[str, Any]],
        text_cache_key,
        config: FingerprintConfig,
        executor: ThreadPoolExecutor,
    ) -> dict[str, Any]:
        parent: dict[str, str] = {}
        cluster_members: dict[str, list[str]] = {}
        representatives: list[str] = []
        boundary_items: list[dict[str, Any]] = []
        merge_log: list[dict[str, Any]] = []

        def find(text_id: str) -> str:
            root = text_id
            while parent[root] != root:
                root = parent[root]
            while parent[text_id] != text_id:
                next_id = parent[text_id]
                parent[text_id] = root
                text_id = next_id
            return root

        for candidate in sorted(members):
            parent[candidate] = candidate
            cluster_members[candidate] = [candidate]
            pair_rows: list[tuple[str, dict[str, Any], str]] = []
            comparison_reps = sorted(representatives)
            futures = {
                representative: executor.submit(
                    pair_similarity,
                    tuple(int(value) for value in signature_entries[candidate]["signature"]),
                    tuple(int(value) for value in signature_entries[representative]["signature"]),
                    config,
                )
                for representative in comparison_reps
            }
            for representative in comparison_reps:
                similarity = futures[representative].result()
                pair_rows.append((representative, similarity, similarity_task_key(candidate, representative)))
                if similarity["boundary"]:
                    boundary_items.append(
                        {
                            "text_ids": sorted([candidate, representative]),
                            "similarity_units": similarity["similarity_units"],
                            "similarity_fixed": similarity["similarity_fixed"],
                            "decision": "merge" if similarity["qualifies"] else "separate",
                            "ordered_by": "text_id_lexicographic",
                            "comparison_task_key": similarity_task_key(candidate, representative),
                        }
                    )
            best: tuple[str, dict[str, Any], str] | None = None
            for row in pair_rows:
                if not row[1]["qualifies"]:
                    continue
                if best is None or row[1]["similarity_units"] > best[1]["similarity_units"]:
                    best = row
            if best is not None:
                representative, similarity, comparison_task_key = best
                candidate_root = find(candidate)
                representative_root = find(representative)
                if candidate_root != representative_root:
                    merged_root = min(candidate_root, representative_root)
                    removed_root = max(candidate_root, representative_root)
                    before_candidate = sorted(cluster_members[candidate_root])
                    before_representative = sorted(cluster_members[representative_root])
                    parent[removed_root] = merged_root
                    cluster_members[merged_root] = sorted(
                        cluster_members[candidate_root] + cluster_members[representative_root]
                    )
                    if removed_root != merged_root:
                        cluster_members.pop(removed_root, None)
                    representatives = [
                        root for root in representatives if find(root) == merged_root
                    ] + [merged_root]
                    representatives = sorted(set(representatives))
                    merge_log.append(
                        {
                            "bucket": bucket,
                            "text_ids": sorted([candidate, representative]),
                            "clusters_before": [candidate_root, representative_root],
                            "cluster_after": merged_root,
                            "members_before": [before_candidate, before_representative],
                            "similarity_units": similarity["similarity_units"],
                            "similarity_fixed": similarity["similarity_fixed"],
                            "boundary": similarity["boundary"],
                            "comparison_task_key": comparison_task_key,
                            "reason": "lsh_bucket_representative_threshold",
                        }
                    )
                    continue
            representatives = sorted(set(representatives + [find(candidate)]))

        grouped: dict[str, list[str]] = {}
        for member_id in sorted(members):
            grouped.setdefault(find(member_id), []).append(member_id)
        clusters: list[dict[str, Any]] = []
        for cluster_id in sorted(grouped):
            member_ids = sorted(grouped[cluster_id])
            cache_keys = [text_cache_key(text_id) for text_id in member_ids]
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "bucket": bucket,
                    "members": member_ids,
                    "member_text_cache_keys": cache_keys,
                    "representative_id": member_ids[0],
                    "singleton": len(member_ids) == 1,
                    "cluster_fingerprint": sha256_json(cache_keys),
                }
            )
        return {
            "bucket": bucket,
            "members": sorted(members),
            "clusters": clusters,
            "boundary_texts": sorted(boundary_items, key=lambda item: tuple(item["text_ids"])),
            "merge_log": sorted(merge_log, key=lambda item: tuple(item["text_ids"])),
        }

    def _inject_cluster_id_conflict(self, ledger: dict[str, Any], injection: dict[str, Any]) -> dict[str, Any]:
        targets = set(injection.get("text_ids", []))
        forced = str(injection.get("forced_cluster_id", "INJECTED-CONFLICT"))
        clusters = [dict(cluster) for cluster in ledger["clusters"]]
        selected: list[int] = []
        for index, cluster in enumerate(clusters):
            if targets & set(cluster["members"]):
                selected.append(index)
        if len(selected) < 2:
            for index in range(len(clusters)):
                if index not in selected:
                    selected.append(index)
                if len(selected) == 2:
                    break
        if len(selected) < 2:
            raise BusinessError(
                E_CLUSTER_ID_CONFLICT,
                "cluster conflict injection requires at least two clusters",
                detail={"injection_id": injection.get("injection_id"), "text_ids": sorted(targets)},
            )
        for index in selected[:2]:
            clusters[index]["cluster_id"] = forced
            clusters[index]["injected_conflict"] = True
        return {
            "clusters": clusters,
            "boundary_texts": list(ledger["boundary_texts"]),
            "merge_log": list(ledger["merge_log"]),
        }

    def _validate_ledger(
        self,
        ledger: dict[str, Any],
        completed_ids: Sequence[str],
        runtime_injection: dict[str, Any] | None,
    ) -> None:
        cluster_ids = [cluster["cluster_id"] for cluster in ledger["clusters"]]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise BusinessError(
                E_CLUSTER_ID_CONFLICT,
                "duplicate cluster identifier detected",
                detail={
                    "cluster_ids": cluster_ids,
                    "injection_id": (runtime_injection or {}).get("injection_id"),
                },
            )
        members = [member for cluster in ledger["clusters"] for member in cluster["members"]]
        if len(members) != len(set(members)):
            raise BusinessError(E_CLUSTER_ID_CONFLICT, "a text appears in multiple clusters")
        expected = set(completed_ids)
        if set(members) != expected:
            raise BusinessError(
                E_CLUSTER_ID_CONFLICT,
                "ledger coverage does not match completed texts",
                detail={"missing": sorted(expected - set(members)), "extra": sorted(set(members) - expected)},
            )
        for cluster in ledger["clusters"]:
            if cluster["members"] != sorted(cluster["members"]):
                raise BusinessError(E_CLUSTER_ID_CONFLICT, "cluster member order is not stable")
            if cluster["members"] and cluster["cluster_id"] not in cluster["members"] and not cluster.get("injected_conflict"):
                raise BusinessError(E_CLUSTER_ID_CONFLICT, "cluster id must be its lexicographically smallest member")

    def _budget_entries(
        self,
        completed_ids: Sequence[str],
        pending_ids: Sequence[str],
        signature_keys: dict[str, str],
        signature_entries: dict[str, dict[str, Any]],
        computed_ids: Sequence[str],
        per_text_cost: int,
    ) -> list[dict[str, Any]]:
        computed = set(computed_ids)
        entries: list[dict[str, Any]] = []
        for text_id in pending_ids:
            entries.append(
                {
                    "text_id": text_id,
                    "signature_key": signature_keys[text_id],
                    "task_key": signature_task_key(text_id),
                    "components": per_text_cost,
                    "status": "pending",
                }
            )
        for text_id in completed_ids:
            entries.append(
                {
                    "text_id": text_id,
                    "signature_key": signature_entries[text_id]["cache_key"],
                    "task_key": signature_task_key(text_id),
                    "components": per_text_cost,
                    "status": "computed" if text_id in computed else "reused",
                }
            )
        return entries

    def _commit_with_stage_versions(
        self,
        manifest: dict[str, Any],
        stages: dict[str, Any],
        previous_versions: dict[str, int],
        previous_hashes: dict[str, str],
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        versions = dict(previous_versions)
        stage_hashes = {stage: sha256_json(stages.get(stage, {})) for stage in PRIMARY_STAGES}
        for stage in PRIMARY_STAGES:
            versions[stage] = int(versions.get(stage, 0))
            if previous_hashes.get(stage) != stage_hashes[stage]:
                versions[stage] += 1
        manifest = dict(manifest)
        manifest["stage_versions"] = versions
        manifest["audit"] = audit
        committed = self.store.commit(manifest, stages)
        return committed

    @staticmethod
    def state_projection(result_or_snapshot: dict[str, Any]) -> dict[str, Any]:
        if "stages" in result_or_snapshot:
            stages = result_or_snapshot["stages"]
            ledger = stages["ledger"]
            signatures = {
                text_id: entry["signature"] for text_id, entry in stages["signatures"]["texts"].items()
            }
            buckets = {
                text_id: entry["bucket"] for text_id, entry in stages["buckets"]["texts"].items()
            }
            manifest = result_or_snapshot["manifest"]
            failures = result_or_snapshot.get("failures", ledger.get("failures", []))
        else:
            ledger = {
                "clusters": result_or_snapshot.get("clusters", []),
                "boundary_texts": result_or_snapshot.get("boundary_texts", []),
                "merge_log": result_or_snapshot.get("merge_log", []),
            }
            signatures = result_or_snapshot.get("signatures", {})
            buckets = result_or_snapshot.get("buckets", {})
            manifest = result_or_snapshot.get("manifest", {})
            failures = result_or_snapshot.get("failures", [])
        return {
            "signatures": {text_id: signatures[text_id] for text_id in sorted(signatures)},
            "buckets": {text_id: buckets[text_id] for text_id in sorted(buckets)},
            "clusters": ledger["clusters"],
            "boundary_texts": ledger["boundary_texts"],
            "merge_log": ledger["merge_log"],
            "failure_list": failures,
            "pending_signatures": manifest.get("pending_signatures", []),
        }
