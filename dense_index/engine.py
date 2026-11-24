from __future__ import annotations

import json
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .canonical import cache_key, full_fingerprint
from .concurrency import ConcurrencyMonitor, run_ordered
from .config import ConfigBundle
from .errors import E_POINT_BUDGET_PENDING, DiagnosticError
from .features import build_count_vector, feature_cache_key, project_pool, projection_cache_key
from .quant import bucket_key_from_vector, diagnose_pair, quantize_vector, routing_tie_flip
from .receipts import build_unified_ledger
from .snapshot import SnapshotStore


def _empty_cache() -> Dict[str, Any]:
    return {"feature": {}, "projection": {}, "fixed": {}, "records": {}}


class IndexEngine:
    def __init__(
        self,
        config: ConfigBundle,
        receipts: Optional[List[Dict[str, Any]]] = None,
        queries: Optional[List[Dict[str, Any]]] = None,
        snapshot_store: Optional[SnapshotStore] = None,
        cache: Optional[Dict[str, Any]] = None,
        pending_ids: Optional[List[str]] = None,
        recovery: Optional[Dict[str, Any]] = None,
        receipt_preference: str = "v3",
    ) -> None:
        self.config = config
        self.receipts = receipts or []
        self.queries = queries or []
        self.store = snapshot_store
        self.cache = cache or _empty_cache()
        self.cache.setdefault("feature", {})
        self.cache.setdefault("projection", {})
        self.cache.setdefault("fixed", {})
        self.cache.setdefault("records", {})
        self.pending_ids = list(pending_ids or [])
        self.recovery = recovery or {"recovered": False}
        self.receipt_preference = receipt_preference
        self.monitor = ConcurrencyMonitor()
        self._cache_lock = threading.RLock()
        self.last_manifest: Optional[Dict[str, Any]] = None

    @classmethod
    def recover(
        cls,
        config: ConfigBundle,
        store: SnapshotStore,
        receipts: Optional[List[Dict[str, Any]]] = None,
        queries: Optional[List[Dict[str, Any]]] = None,
        receipt_preference: str = "v3",
    ) -> "IndexEngine":
        snapshot = store.load_current()
        if not snapshot:
            return cls(config, receipts=receipts, queries=queries, snapshot_store=store, recovery={"recovered": False, "reason": "no_committed_snapshot"}, receipt_preference=receipt_preference)
        manifest = snapshot["manifest"]
        artifacts = snapshot.get("artifacts", {})
        cache = artifacts.get("cache") or _empty_cache()
        recovery = {
            "recovered": True,
            "global_version": manifest.get("global_version"),
            "snapshot_dir": snapshot.get("snapshot_dir"),
            "manifest_config_hash": manifest.get("config_hash"),
            "current_config_hash": config.config_hash,
            "config_hash_match": manifest.get("config_hash") == config.config_hash,
            "completed_blocks": manifest.get("completed_blocks", []),
            "pending_blocks": manifest.get("pending_blocks", []),
        }
        return cls(
            config,
            receipts=receipts,
            queries=queries,
            snapshot_store=store,
            cache=cache,
            pending_ids=manifest.get("pending_blocks", []),
            recovery=recovery,
            receipt_preference=receipt_preference,
        )

    def _projection_task(self, block: Dict[str, Any]) -> Dict[str, Any]:
        block_id = block["block_id"]
        content_fingerprint = block["content_fingerprint"]
        content_digest = full_fingerprint(
            {key: block.get(key) for key in ("block_id", "document_id", "block_seq", "paragraph_id", "chunk_ord", "terms", "padded", "padding_count", "overridden")}
        )
        content_key = cache_key("content-v1", content_fingerprint, self.config.chunk_fingerprint)
        f_key = feature_cache_key(content_fingerprint, self.config)

        with self._cache_lock:
            counts = self.cache["feature"].get(f_key)
        feature_reused = counts is not None
        feature_examples: List[Dict[str, Any]] = []
        if counts is None:
            counts, feature_examples = build_count_vector(block["terms"], self.config)
            with self._cache_lock:
                self.cache["feature"].setdefault(f_key, counts)

        p_key = projection_cache_key(f_key, self.config)
        with self._cache_lock:
            pool = self.cache["projection"].get(p_key)
        projection_reused = pool is not None
        if pool is None:
            pool = project_pool(counts, self.config)
            injection = getattr(self, "injection", None) or {}
            if injection.get("kind") == "nonfinite_vector" and injection.get("target_block_id") == block_id:
                pool[0] = float("nan")
            finite = np.asarray(pool, dtype=np.float64)
            if not np.all(np.isfinite(finite)):
                from .errors import E_VECTOR_NONFINITE

                raise DiagnosticError(E_VECTOR_NONFINITE, f"non-finite projection for {block_id}", injection_id=injection.get("id"))
            with self._cache_lock:
                self.cache["projection"].setdefault(p_key, pool)

        dim = int(self.config.data["projection_dim"])
        raw_vector = list(pool[:dim])
        fixed_key = cache_key("fixed-v1", p_key, dim, self.config.quant_fingerprint)
        with self._cache_lock:
            quantized = self.cache["fixed"].get(fixed_key)
        fixed_reused = quantized is not None
        if quantized is None:
            quantized = quantize_vector(raw_vector, self.config)
            with self._cache_lock:
                self.cache["fixed"].setdefault(fixed_key, quantized)

        bucket = bucket_key_from_vector(raw_vector, quantized["fixed_vector"], self.config)
        alternative_rule = "negative" if self.config.data["sign_tie_rule"] == "positive" else "positive"
        alternative_bucket = bucket_key_from_vector(
            raw_vector,
            quantized["fixed_vector"],
            self.config,
            alternative_tie_rule=alternative_rule,
        )
        vector_digest = full_fingerprint(
            {
                "content_digest": content_digest,
                "projection_dim": dim,
                "raw_vector": raw_vector,
                "fixed_vector": quantized["fixed_vector"],
                "quant_fingerprint": self.config.quant_fingerprint,
            }
        )
        bucket_cache_key = cache_key("bucket-v1", fixed_key, self.config.bucket_fingerprint, bucket["bucket_key"])
        return {
            **block,
            "content_digest": content_digest,
            "content_cache_key": content_key,
            "feature_cache_key": f_key,
            "feature_reused": feature_reused,
            "feature_examples": feature_examples,
            "projection_cache_key": p_key,
            "projection_reused": projection_reused,
            "projection_dim": dim,
            "raw_vector": raw_vector,
            "fixed_cache_key": fixed_key,
            "fixed_reused": fixed_reused,
            **quantized,
            "bucket_cache_key": bucket_cache_key,
            "bucket_key": bucket["bucket_key"],
            "bucket_id": bucket["bucket_id"],
            "sign_tie_breaks": bucket["tie_breaks"],
            "alternative_bucket_key": alternative_bucket["bucket_key"],
            "alternative_bucket_id": alternative_bucket["bucket_id"],
            "vector_digest": vector_digest,
        }

    def _scheduled_candidates(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id = {block["block_id"]: block for block in blocks}
        if self.pending_ids:
            pending = [by_id[block_id] for block_id in self.pending_ids if block_id in by_id]
            if len(pending) == len([block_id for block_id in self.pending_ids if block_id in by_id]):
                return sorted(pending, key=lambda item: (item["document_id"], item["block_seq"], item["block_id"]))
        return sorted(blocks, key=lambda item: (item["document_id"], item["block_seq"], item["block_id"]))

    def run_batch(
        self,
        blocks: List[Dict[str, Any]],
        budget: Optional[int] = None,
        injection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.injection = injection or {}
        point_budget = int(self.config.data["projection_point_budget"] if budget is None else budget)
        candidates = self._scheduled_candidates(blocks)
        scheduled = candidates[:point_budget]
        pending = candidates[point_budget:]
        scheduled_ids = [block["block_id"] for block in scheduled]
        pending_ids = [block["block_id"] for block in pending]

        old_records = dict(self.cache.get("records", {}))
        results, errors = run_ordered(
            scheduled,
            self._projection_task,
            max_workers=int(self.config.data["max_concurrency"]),
            monitor=self.monitor,
        )
        first_error = next((error for error in errors if error is not None), None)
        if first_error is not None:
            if isinstance(first_error, DiagnosticError):
                raise first_error
            raise DiagnosticError("E_PROJECTION_FAILED", str(first_error)) from first_error

        new_records: Dict[str, Dict[str, Any]] = {}
        for result in results:
            if result is not None:
                record = self._finalize_block_duplicates_placeholder(result)
                new_records[record["block_id"]] = record
                self.cache["records"][record["block_id"]] = record
        self.pending_ids = pending_ids

        partial = bool(pending_ids)
        if partial:
            manifest = None
            if self.store:
                manifest = self.store.commit(
                    {
                        "config_hash": self.config.config_hash,
                        "param_fingerprint": self.config.param_fingerprint,
                        "quant_level": self.config.data["quant_level"],
                        "completed_blocks": sorted(self.cache["records"].keys()),
                        "pending_blocks": pending_ids,
                        "cache": self.cache,
                    }
                )
            return {
                "status": E_POINT_BUDGET_PENDING,
                "scheduled_ids": scheduled_ids,
                "pending_ids": pending_ids,
                "point_budget": point_budget,
                "observed_max_concurrency": self.monitor.observed_max,
                "manifest": manifest,
                "recovery": self.recovery,
            }

        current_ids = [block["block_id"] for block in sorted(blocks, key=lambda item: (item["document_id"], item["block_seq"], item["block_id"]))]
        records = [self.cache["records"][block_id] for block_id in current_ids]
        records = self._assign_duplicates(records)
        for record in records:
            self.cache["records"][record["block_id"]] = record

        sorted_records = sorted(records, key=lambda item: (item["document_id"], item["block_seq"], item["block_id"]))
        bucket_stage = Path(tempfile.mkdtemp(prefix="dense-bucket-write-"))
        try:
            bucket_fragments, bucket_errors = run_ordered(
                sorted_records,
                lambda record: self._bucket_write_task(record, bucket_stage),
                max_workers=int(self.config.data["max_concurrency"]),
                monitor=self.monitor,
            )
            bucket_error = next((error for error in bucket_errors if error is not None), None)
            if bucket_error is not None:
                if isinstance(bucket_error, DiagnosticError):
                    raise bucket_error
                raise DiagnosticError("E_BUCKET_WRITE_FAILED", str(bucket_error)) from bucket_error
            buckets, bucket_artifacts = self._build_buckets(sorted_records, bucket_fragments)
        finally:
            shutil.rmtree(bucket_stage, ignore_errors=True)
        invalidation = self._build_invalidation(records, old_records, buckets)
        for cached_block_id in list(self.cache["records"].keys()):
            if cached_block_id not in current_ids:
                self.cache["records"].pop(cached_block_id, None)
        digests = {
            "blocks": {record["block_id"]: record["content_digest"] for record in records},
            "vector_digests": {record["block_id"]: record["vector_digest"] for record in records},
            **{bucket_id: artifact["bucket_digest"] for bucket_id, artifact in bucket_artifacts.items()},
            "bucket_cache_keys": {bucket_id: artifact["membership_cache_key"] for bucket_id, artifact in bucket_artifacts.items()},
        }
        ledger = build_unified_ledger(
            self.receipts,
            current_digests=digests,
            quant_level=str(self.config.data["quant_level"]),
            param_fingerprint=self.config.param_fingerprint,
        )
        recall = self.run_recall(records, buckets)
        view = self.canonical_view(records, buckets, bucket_artifacts, ledger, recall)
        manifest = None
        if self.store:
            manifest = self.store.commit(
                {
                    "config_hash": self.config.config_hash,
                    "param_fingerprint": self.config.param_fingerprint,
                    "quant_level": self.config.data["quant_level"],
                    "completed_blocks": current_ids,
                    "pending_blocks": [],
                    "blocks": records,
                    "buckets": {"buckets": buckets, "artifacts": bucket_artifacts},
                    "ledger": ledger,
                    "recall": recall,
                    "cache": self.cache,
                    "view": view,
                }
            )
        self.last_manifest = manifest
        return {
            "status": "OK",
            "view": view,
            "records": records,
            "buckets": buckets,
            "bucket_artifacts": bucket_artifacts,
            "ledger": ledger,
            "recall": recall,
            "invalidation": invalidation,
            "scheduled_ids": scheduled_ids,
            "pending_ids": [],
            "point_budget": point_budget,
            "observed_max_concurrency": self.monitor.observed_max,
            "bucket_write_fragments": len(bucket_fragments),
            "manifest": manifest,
            "recovery": self.recovery,
        }

    def _finalize_block_duplicates_placeholder(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record.setdefault("duplicate_of", None)
        record.setdefault("duplicate_refs", [])
        return record

    def _assign_duplicates(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(record["content_fingerprint"], []).append(record)
        for group in groups.values():
            group.sort(key=lambda item: item["block_id"])
            canonical_id = group[0]["block_id"]
            duplicate_ids = [item["block_id"] for item in group[1:]]
            for index, record in enumerate(group):
                record["duplicate_of"] = None if index == 0 else canonical_id
                record["duplicate_refs"] = [] if index else duplicate_ids
        return records

    def _bucket_write_task(self, record: Dict[str, Any], stage_dir: Path) -> Dict[str, Any]:
        fragment = {
            "schema": "bucket-write-fragment/v1",
            "block_id": record["block_id"],
            "document_id": record["document_id"],
            "block_seq": record["block_seq"],
            "stage_name": "bucket_write",
            "stable_task_key": record["task_keys"]["bucket_write"],
            "bucket_key": record["bucket_key"],
            "bucket_id": record["bucket_id"],
            "bucket_cache_key": record["bucket_cache_key"],
            "fixed_cache_key": record["fixed_cache_key"],
        }
        fragment_path = stage_dir / f"{record['block_id']}-{uuid.uuid4().hex[:8]}.json"
        fragment_path.write_text(json.dumps(fragment, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        fragment["fragment_written"] = True
        return fragment

    def _build_buckets(self, records: List[Dict[str, Any]], fragments: Optional[List[Dict[str, Any]]] = None):
        buckets: Dict[str, List[str]] = {}
        for record in records:
            buckets.setdefault(record["bucket_id"], []).append(record["block_id"])
        artifacts: Dict[str, Any] = {}
        fragment_by_block = {item["block_id"]: item for item in (fragments or []) if item is not None}
        for bucket_id in sorted(buckets):
            members = buckets[bucket_id]
            member_keys = [next(record for record in records if record["block_id"] == block_id)["bucket_cache_key"] for block_id in members]
            membership_key = cache_key("membership-v1", bucket_id, self.config.bucket_fingerprint, member_keys)
            bucket_digest = full_fingerprint({"bucket_id": bucket_id, "members": members, "member_keys": member_keys})
            artifacts[bucket_id] = {
                "bucket_id": bucket_id,
                "bucket_key": next(record for record in records if record["block_id"] == members[0])["bucket_key"],
                "members": members,
                "membership_cache_key": membership_key,
                "bucket_digest": bucket_digest,
                "member_cache_keys": member_keys,
                "bucket_write_task_keys": [fragment_by_block[block_id]["stable_task_key"] for block_id in members if block_id in fragment_by_block],
            }
        return dict(sorted(buckets.items())), artifacts

    def _build_invalidation(self, records: List[Dict[str, Any]], old_records: Dict[str, Dict[str, Any]], buckets: Dict[str, List[str]]):
        current_ids = {record["block_id"] for record in records}
        new_blocks: List[str] = []
        content_changed: List[str] = []
        bucket_changed: List[Dict[str, Any]] = []
        reused_blocks: List[Dict[str, Any]] = []
        reprojected_blocks: List[str] = []
        requantized_blocks: List[str] = []
        decisions: List[Dict[str, Any]] = []

        for record in records:
            block_id = record["block_id"]
            old = old_records.get(block_id)
            if record["projection_reused"]:
                reused_blocks.append(
                    {
                        "block_id": block_id,
                        "reason": "projection_cache_hit_stable_coordinate_prefix",
                        "projection_cache_key": record["projection_cache_key"],
                        "bucket_id": record["bucket_id"],
                    }
                )
            else:
                reprojected_blocks.append(block_id)
            if not record["fixed_reused"]:
                requantized_blocks.append(block_id)

            if old is None:
                new_blocks.append(block_id)
                bucket_changed.append({"block_id": block_id, "old_bucket_id": None, "new_bucket_id": record["bucket_id"], "reason": "new_block"})
                reason = "new_block"
            else:
                old_bucket = old.get("bucket_id")
                if old.get("content_fingerprint") != record["content_fingerprint"]:
                    content_changed.append(block_id)
                if old_bucket != record["bucket_id"]:
                    bucket_changed.append({"block_id": block_id, "old_bucket_id": old_bucket, "new_bucket_id": record["bucket_id"], "reason": "bucket_key_changed"})
                reason = "bucket_key_stable" if old_bucket == record["bucket_id"] else "bucket_key_changed"
            decisions.append(
                {
                    "block_id": block_id,
                    "decision": reason if old is not None else "new_block",
                    "content_cache_key": record["content_cache_key"],
                    "feature_cache_key": record["feature_cache_key"],
                    "projection_cache_key": record["projection_cache_key"],
                    "fixed_cache_key": record["fixed_cache_key"],
                    "bucket_cache_key": record["bucket_cache_key"],
                    "quant_level": self.config.data["quant_level"],
                    "upstream_content_fingerprint": record["content_fingerprint"],
                    "param_fingerprint": self.config.param_fingerprint,
                }
            )

        removed_blocks = sorted(block_id for block_id in old_records if block_id not in current_ids)
        invalidated_buckets = set()
        placement_invalidated_buckets = set()
        for item in bucket_changed:
            if item["old_bucket_id"]:
                invalidated_buckets.add(item["old_bucket_id"])
                placement_invalidated_buckets.add(item["old_bucket_id"])
            invalidated_buckets.add(item["new_bucket_id"])
            placement_invalidated_buckets.add(item["new_bucket_id"])
        for block_id in removed_blocks:
            old_bucket = old_records[block_id].get("bucket_id")
            if old_bucket:
                invalidated_buckets.add(old_bucket)
                placement_invalidated_buckets.add(old_bucket)
        for block_id in content_changed:
            new_bucket = next(record["bucket_id"] for record in records if record["block_id"] == block_id)
            invalidated_buckets.add(new_bucket)

        reused_bucket_memberships = []
        sibling_reuse = []
        changed_or_removed = set(content_changed) | set(removed_blocks) | {item["block_id"] for item in bucket_changed}
        for bucket_id, member_ids in buckets.items():
            affected = [block_id for block_id in member_ids if block_id in changed_or_removed]
            if not affected:
                reused_bucket_memberships.append({"bucket_id": bucket_id, "members": member_ids, "reason": "no_member_content_or_placement_change"})
            for block_id in affected:
                unaffected = [member for member in member_ids if member not in changed_or_removed]
                sibling_reuse.append({"bucket_id": bucket_id, "affected_block_id": block_id, "reused_sibling_ids": unaffected})

        canonical_order = [record["block_id"] for record in sorted(records, key=lambda item: (item["document_id"], item["block_seq"]))]
        order_index = {block_id: index for index, block_id in enumerate(canonical_order)}

        def is_canonical_subsequence(values: Sequence[str]) -> bool:
            indexes = [order_index[value] for value in values if value in order_index]
            return indexes == sorted(indexes)

        return {
            "new_blocks": sorted(new_blocks, key=order_index.get),
            "content_changed_blocks": sorted(content_changed, key=order_index.get),
            "removed_blocks": removed_blocks,
            "bucket_key_changed_blocks": bucket_changed,
            "reprojected_blocks": sorted(reprojected_blocks, key=order_index.get),
            "requantized_blocks": sorted(requantized_blocks, key=order_index.get),
            "reused_blocks": sorted(reused_blocks, key=lambda item: order_index.get(item["block_id"], 10**9)),
            "invalidated_buckets": sorted(invalidated_buckets),
            "placement_invalidated_buckets": sorted(placement_invalidated_buckets),
            "reused_bucket_memberships": reused_bucket_memberships,
            "sibling_reuse": sibling_reuse,
            "decisions": decisions,
            "canonical_order": canonical_order,
            "recomputed_order_is_canonical_subsequence": is_canonical_subsequence(reprojected_blocks),
            "reused_order_is_canonical_subsequence": is_canonical_subsequence([item["block_id"] for item in reused_blocks]),
        }

    def run_recall(self, records: List[Dict[str, Any]], buckets: Dict[str, List[str]]) -> Dict[str, Any]:
        by_id = {record["block_id"]: record for record in records}
        results = []
        diagnostics = []
        boundary_blocks = []

        for query in sorted(self.queries, key=lambda item: item.get("query_id", "")):
            query_id = query.get("query_id", "query")
            counts, _ = build_count_vector(query.get("terms", []), self.config)
            pool = project_pool(counts, self.config)
            raw_query = pool[: int(self.config.data["projection_dim"])]
            quantized_query = quantize_vector(raw_query, self.config)
            query_bucket = bucket_key_from_vector(raw_query, quantized_query["fixed_vector"], self.config)
            alternative_rule = "negative" if self.config.data["sign_tie_rule"] == "positive" else "positive"
            query_alt = bucket_key_from_vector(raw_query, quantized_query["fixed_vector"], self.config, alternative_tie_rule=alternative_rule)

            member_ids = buckets.get(query_bucket["bucket_id"], [])
            query_results = []
            query_boundaries = []
            for block_id in member_ids:
                record = by_id[block_id]
                diagnosis = diagnose_pair(
                    raw_query,
                    quantized_query["fixed_vector"],
                    record["raw_vector"],
                    record["fixed_vector"],
                    block_id,
                    self.config,
                    threshold=float(query.get("threshold", self.config.data["similarity_threshold"])),
                    query_id=query_id,
                )
                if diagnosis["boundary"]:
                    query_boundaries.append(diagnosis)
                    boundary_blocks.append(diagnosis)
                if diagnosis["decision"]:
                    query_results.append(
                        {
                            "query_id": query_id,
                            "block_id": block_id,
                            "score": diagnosis["score"],
                            "reference_score": diagnosis["reference_score"],
                            "boundary": diagnosis["boundary"],
                            "exact_threshold": diagnosis["exact_threshold"],
                            "vector_digest": record["vector_digest"],
                        }
                    )
                if diagnosis["code"]:
                    diagnostics.append(diagnosis)

            query_results.sort(key=lambda item: (-item["score"], item["block_id"]))
            for rank, item in enumerate(query_results, 1):
                item["rank"] = rank
            query_boundaries.sort(key=lambda item: item["block_id"])
            for rank, item in enumerate(query_boundaries, 1):
                item["lexicographic_rank"] = rank

            alt_buckets: Dict[str, List[str]] = {}
            for record in records:
                alt_buckets.setdefault(record["alternative_bucket_id"], []).append(record["block_id"])
            alt_member_ids = alt_buckets.get(query_alt["bucket_id"], [])
            alt_recalled = set()
            threshold = float(query.get("threshold", self.config.data["similarity_threshold"]))
            for block_id in alt_member_ids:
                record = by_id[block_id]
                alt_score = float(np.dot(quantized_query["fixed_vector"], record["fixed_vector"]) / (np.linalg.norm(quantized_query["fixed_vector"]) * np.linalg.norm(record["fixed_vector"]) or 1.0))
                if alt_score >= threshold:
                    alt_recalled.add(block_id)
            default_recalled = {item["block_id"] for item in query_results}
            routing = routing_tie_flip(query_bucket, {"bucket_id": next((r["bucket_id"] for r in records if r["block_id"] in member_ids), query_bucket["bucket_id"])}, query_alt, {"bucket_id": next((r["alternative_bucket_id"] for r in records if r["block_id"] in alt_member_ids), query_alt["bucket_id"])})
            if default_recalled != alt_recalled:
                tie_diagnostic = {
                    "query_id": query_id,
                    "code": "E_BOUNDARY_FLIP",
                    "cause": "tie_rule",
                    "default_recalled": sorted(default_recalled),
                    "alternative_recalled": sorted(alt_recalled),
                    "current_query_bucket": query_bucket["bucket_id"],
                    "alternative_query_bucket": query_alt["bucket_id"],
                    "default_bucket_members": member_ids,
                    "alternative_bucket_members": alt_member_ids,
                    "routing": routing,
                }
                diagnostics.append(tie_diagnostic)

            results.append(
                {
                    "query_id": query_id,
                    "query_bucket_id": query_bucket["bucket_id"],
                    "query_alternative_bucket_id": query_alt["bucket_id"],
                    "recalled": query_results,
                    "boundary_blocks": query_boundaries,
                    "tie_rule": "equal_or_half_step_scores_are_ordered_by_block_id_lexicographic_asc",
                    "lexicographic_tie_winner": min((item["block_id"] for item in query_boundaries), default=None),
                    "default_recalled_ids": sorted(default_recalled),
                    "tie_rule_alternative_recalled_ids": sorted(alt_recalled),
                }
            )

        diagnostics.sort(key=lambda item: (item.get("query_id", ""), item.get("block_id", ""), item.get("cause", "")))
        return {"queries": results, "diagnostics": diagnostics}

    def canonical_view(self, records, buckets, bucket_artifacts, ledger, recall) -> Dict[str, Any]:
        block_fields = [
            "block_id",
            "document_id",
            "block_seq",
            "paragraph_id",
            "paragraph_sequence_no",
            "chunk_ord",
            "start",
            "end",
            "raw_term_count",
            "terms",
            "padded",
            "padding_count",
            "overridden",
            "content_fingerprint",
            "content_digest",
            "feature_cache_key",
            "projection_cache_key",
            "projection_dim",
            "raw_vector",
            "fixed_vector",
            "scale",
            "quant_step",
            "effective_fractional_bits",
            "residuals",
            "clipped_coordinates",
            "fixed_cache_key",
            "bucket_key",
            "bucket_id",
            "sign_tie_breaks",
            "alternative_bucket_id",
            "bucket_cache_key",
            "vector_digest",
            "duplicate_of",
            "duplicate_refs",
        ]
        canonical_blocks = [{key: record.get(key) for key in block_fields} for record in sorted(records, key=lambda item: (item["document_id"], item["block_seq"]))]
        return {
            "schema": "dense-index-canonical-view/v1",
            "config_hash": self.config.config_hash,
            "param_fingerprint": self.config.param_fingerprint,
            "quant_level": self.config.data["quant_level"],
            "receipt_preference": self.receipt_preference,
            "blocks": canonical_blocks,
            "buckets": buckets,
            "bucket_artifacts": {key: {k: value for k, value in artifact.items() if k != "member_cache_keys"} for key, artifact in sorted(bucket_artifacts.items())},
            "ledger": ledger,
            "recall": recall,
            "duplicate_groups": [
                {"content_fingerprint": record["content_fingerprint"], "canonical_block_id": record["block_id"], "duplicate_refs": record["duplicate_refs"]}
                for record in canonical_blocks
                if record.get("duplicate_refs")
            ],
        }
