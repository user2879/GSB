from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import (
    CacheStore,
    column_cc_key,
    domain_key,
    keypoint_key,
    reading_key,
    window_key,
)
from .clock import Clock, MonotonicClock
from .codes import ALL_CODES, E_LEASE_LIMIT
from .hashing import fingerprint
from .leases import LeaseExecutor
from .loader import load_workspace
from .pipeline import (
    assign_reading_and_conflicts,
    compute_window,
    merge_prepared_column,
    prepare_column,
    regress_domain,
)
from .scheduler import allocate_comparisons, plan_point_budget, run_ordered
from .snapshot import read_current_manifest, write_atomic_snapshot
from .versions import diff_annotation_versions
from .vision import window_rects


class WorkbenchEngine:
    def __init__(
        self,
        root: str | Path,
        clock: Clock | None = None,
        cache: CacheStore | None = None,
        lease_executor: LeaseExecutor | None = None,
        write_snapshots: bool = True,
    ):
        self.root = Path(root)
        self.clock = clock or MonotonicClock()
        self.cache = cache or CacheStore()
        self.leases = lease_executor or LeaseExecutor(self.clock)
        self.write_snapshots = write_snapshots
        self.global_version = 0
        self.run_seq = 0
        self.pending_ids: list[str] = []
        self.previous_annotation: dict[str, Any] | None = None
        self.previous_image_versions: dict[str, str] = {}
        self.previous_lease_generations: dict[str, int] = {}
        self.previous_config_hash: str | None = None
        self.last_result: dict[str, Any] | None = None
        self.injection_ids: dict[str, str] = {}

    def _empty_counts(self) -> dict[str, int]:
        return {code: 0 for code in ALL_CODES}

    def _base_result(self, global_version: int, config: Any = None) -> dict[str, Any]:
        return {
            "global_version": global_version,
            "run_seq": self.run_seq,
            "config_hash": getattr(config, "config_hash", None),
            "kernel_hash": getattr(config, "kernel_hash", None),
            "quant": config.quant_fingerprint() if config is not None else None,
            "annotation_version": None,
            "annotation_digest": None,
            "samples": [],
            "windows": [],
            "components": [],
            "keypoints": [],
            "reading": [],
            "conflicts": [],
            "failures": [],
            "business_code_counts": self._empty_counts(),
            "budget": {},
            "pending_queue": [],
            "lease_snapshot": [],
            "lease_events": [],
            "version_events": [],
            "image_version_events": [],
            "config_events": [],
            "invalidations": [],
            "reuse": [],
            "recompute": [],
            "cache_stats": {},
        }

    def run(self, global_version: int | None = None) -> dict[str, Any]:
        self.run_seq += 1
        if global_version is not None:
            self.global_version = global_version
        result_version = self.global_version
        loaded = load_workspace(self.root, injection_ids=self.injection_ids)
        config = loaded.config
        result = self._base_result(result_version, config)
        result["failures"] = loaded.failures
        counts = result["business_code_counts"]
        for failure in loaded.failures:
            counts[failure["code"]] = counts.get(failure["code"], 0) + 1

        if loaded.failures and any("config" in item["target"] for item in loaded.failures):
            result["budget"] = {"status": "config_invalid"}
            result["cache_stats"] = self.cache.stats()
            self.last_result = result
            if self.write_snapshots:
                write_atomic_snapshot(self.root, result, result_version, "invalid-config")
            return result

        self.leases.max_active_leases = config.max_active_leases
        lease_event_start = len(self.leases.events)
        self.leases.reap_expired()

        current_image_versions = {
            sample["image_id"]: sample["image_fingerprint"] for sample in loaded.samples
        }
        image_events = self._diff_image_versions(current_image_versions)
        result["image_version_events"] = image_events
        version_events = diff_annotation_versions(
            self.previous_annotation, loaded.annotation_bundle or None
        )
        result["version_events"] = version_events
        config_events = self._diff_config(config.config_hash, config.to_dict())
        result["config_events"] = config_events

        units: list[dict[str, Any]] = []
        for sample in loaded.samples:
            for column in sample["truth"]["columns"]:
                rank = int(column["column_rank"])
                points = [
                    point
                    for point in sample["truth"]["keypoints"]
                    if int(point["column_rank"]) == rank
                ]
                units.append(
                    {
                        "unit_id": f"{sample['image_id']}:c{rank}",
                        "image_id": sample["image_id"],
                        "column_rank": rank,
                        "x_min": int(column["x_min"]),
                        "x_max": int(column["x_max"]),
                        "planned_points": len(points),
                        "stable_order": len(units),
                        "sample": sample,
                        "column": column,
                    }
                )
        units.sort(key=lambda u: (u["image_id"], u["column_rank"]))
        for index, unit in enumerate(units):
            unit["stable_order"] = index
        current_unit_ids = {u["unit_id"] for u in units}
        self.pending_ids = [uid for uid in self.pending_ids if uid in current_unit_ids]

        budget_plan = plan_point_budget(
            units,
            self.pending_ids,
            config.global_point_budget,
            config.column_point_quota,
        )
        units_by_id = {u["unit_id"]: u for u in units}
        for event in budget_plan.events:
            counts[event["code"]] = counts.get(event["code"], 0) + 1

        lease_generations: dict[str, int] = {}
        lease_blocked: list[str] = []
        for unit_id in budget_plan.approved_ids:
            unit = units_by_id[unit_id]
            target = unit_id
            existing = self.leases.active_lease_for_target(target)
            lease_response: dict[str, Any] | None = None
            if existing:
                lease_response = self.leases.renew(
                    existing["lease_id"], existing["generation"], config.lease_ttl_ms
                )
                if not lease_response.get("ok") and lease_response.get("code") == "E_LEASE_EXPIRED":
                    lease_response = self.leases.acquire(
                        target,
                        unit["x_min"],
                        unit["x_max"],
                        unit["planned_points"],
                        config.lease_ttl_ms,
                    )
            else:
                lease_response = self.leases.acquire(
                    target,
                    unit["x_min"],
                    unit["x_max"],
                    unit["planned_points"],
                    config.lease_ttl_ms,
                )
            if not lease_response.get("ok"):
                code = lease_response.get("code", E_LEASE_LIMIT)
                counts[code] = counts.get(code, 0) + 1
                lease_blocked.append(unit_id)
            else:
                lease_generations[unit_id] = int(lease_response["lease"]["generation"])

        processable_ids = [uid for uid in budget_plan.approved_ids if uid in lease_generations]
        images_with_leases = sorted({units_by_id[uid]["image_id"] for uid in processable_ids})
        samples_by_id = {sample["image_id"]: sample for sample in loaded.samples}

        window_contexts = [
            (samples_by_id[image_id], [uid for uid in processable_ids if units_by_id[uid]["image_id"] == image_id])
            for image_id in images_with_leases
        ]
        window_outputs = run_ordered(
            window_contexts,
            lambda context: self._windows_worker(context, config),
            config.max_workers,
        )
        windows_by_image: dict[str, list[dict[str, Any]]] = {}
        reuse_entries: list[dict[str, Any]] = []
        recompute_entries: list[dict[str, Any]] = []
        for image_id, windows, reuse, recompute in window_outputs:
            windows_by_image[image_id] = windows
            result["windows"].extend(windows)
            reuse_entries.extend(reuse)
            recompute_entries.extend(recompute)

        miss_units: list[dict[str, Any]] = []
        column_results: dict[str, dict[str, Any]] = {}
        column_cache_keys: dict[str, str] = {}
        for unit_id in processable_ids:
            unit = units_by_id[unit_id]
            image_id = unit["image_id"]
            windows = windows_by_image.get(image_id, [])
            dependent_keys = sorted(
                w["cache_key"]
                for w in windows
                if w["x"] < unit["x_max"]
                and w["x"] + w["w"] > unit["x_min"]
                and w["y"] < config.height
                and w["y"] + w["h"] > 0
            )
            cc_key = column_cc_key(
                unit["sample"]["image_fingerprint"],
                unit["column_rank"],
                dependent_keys,
                config.kernel_hash,
                config.quant_fingerprint(),
                lease_generations[unit_id],
            )
            column_cache_keys[unit_id] = cc_key
            cached = self.cache.get("connected_component_column", cc_key)
            if cached is not None:
                column_results[unit_id] = cached
                reuse_entries.append(
                    self._entry("connected_component", unit_id, cc_key, "column_cache_key_hit")
                )
            else:
                prepared = prepare_column(unit["sample"], unit["column"], windows, config)
                prepared["unit_id"] = unit_id
                miss_units.append(prepared)
                recompute_entries.append(
                    self._entry("connected_component_prepare", unit_id, cc_key, "column_cache_miss")
                )

        comparison_approved, comparison_events, comparisons_used = allocate_comparisons(
            miss_units, config.max_component_comparisons
        )
        for event in comparison_events:
            counts[event["code"]] = counts.get(event["code"], 0) + 1
        comparison_approved_set = set(comparison_approved)
        merge_inputs = [p for p in miss_units if p["unit_id"] in comparison_approved_set]
        merge_outputs = run_ordered(merge_inputs, lambda p: (p["unit_id"], merge_prepared_column(p, config)), config.max_workers)
        comparison_deferred = [p["unit_id"] for p in miss_units if p["unit_id"] not in comparison_approved_set]

        for unit_id, column_result in merge_outputs:
            cc_key = column_cache_keys[unit_id]
            unit = units_by_id[unit_id]
            for domain in column_result["domains"]:
                domain["column_rank"] = unit["column_rank"]
                mask_fp = fingerprint({"coords": domain["coords"]})
                d_key = domain_key(
                    unit["sample"]["image_fingerprint"],
                    mask_fp,
                    config.kernel_hash,
                    config.quant_fingerprint(),
                    lease_generations[unit_id],
                )
                domain["domain_cache_key"] = d_key
            for public_domain in column_result["public_domains"]:
                matching = next(
                    (d for d in column_result["domains"] if d["domain_id"] == public_domain["domain_id"]),
                    None,
                )
                if matching is not None:
                    public_domain["cache_key"] = matching["domain_cache_key"]
                    public_domain["column_cache_key"] = cc_key
            self.cache.put("connected_component_column", cc_key, column_result)
            column_results[unit_id] = column_result
            recompute_entries.append(
                self._entry("connected_component", unit_id, cc_key, "pair_merge_and_split_recomputed")
            )

        keypoints_by_image: dict[str, list[dict[str, Any]]] = {
            sample["image_id"]: [] for sample in loaded.samples
        }
        for unit_id in processable_ids:
            if unit_id in comparison_deferred:
                continue
            column_result = column_results.get(unit_id)
            if column_result is None:
                continue
            unit = units_by_id[unit_id]
            image_id = unit["image_id"]
            annotation_doc = (loaded.annotation_bundle or {}).get("documents", {}).get(image_id, {})
            annotation_digest = annotation_doc.get("content_digest") or fingerprint(
                annotation_doc.get("keypoints", [])
            )
            for domain in column_result["domains"]:
                kp_key = keypoint_key(
                    unit["sample"]["image_fingerprint"],
                    domain["domain_cache_key"],
                    config.kernel_hash,
                    config.quant_fingerprint(),
                    lease_generations[unit_id],
                    annotation_digest,
                )
                cached_kp = self.cache.get("keypoint", kp_key)
                if cached_kp is not None:
                    point = dict(cached_kp)
                    reuse_entries.append(self._entry("keypoint", point["keypoint_id"], kp_key, "keypoint_cache_hit"))
                else:
                    point = regress_domain(unit["sample"], domain, config)
                    point["cache_key"] = kp_key
                    point["annotation_digest"] = annotation_digest
                    self.cache.put("keypoint", kp_key, point)
                    recompute_entries.append(
                        self._entry("keypoint", point["keypoint_id"], kp_key, "domain_or_annotation_version_changed")
                    )
                point["column_rank"] = unit["column_rank"]
                point["image_id"] = image_id
                keypoints_by_image[image_id].append(point)

        for sample in loaded.samples:
            image_id = sample["image_id"]
            points = keypoints_by_image.get(image_id, [])
            annotation_doc = (loaded.annotation_bundle or {}).get("documents", {}).get(image_id, {})
            annotation_digest = annotation_doc.get("content_digest") or fingerprint(
                annotation_doc.get("keypoints", [])
            )
            lease_map = {
                uid.split(":", 1)[1]: lease_generations[uid]
                for uid in lease_generations
                if units_by_id[uid]["image_id"] == image_id
            }
            preordered = sorted(points, key=lambda p: (int(p["column_rank"]), int(p["y"]), int(p["x"])))
            r_key = reading_key(
                sample["image_fingerprint"],
                [p["cache_key"] for p in preordered],
                sample["truth"]["content_digest"],
                annotation_digest,
                config.quant_fingerprint(),
                lease_map,
            )
            cached_reading = self.cache.get("reading_order", r_key)
            if cached_reading is not None:
                reading_result = cached_reading
                order_by_id = {
                    item["keypoint_id"]: item["reading_order"]
                    for item in reading_result["reading_sequence"]
                }
                for point in points:
                    if point["keypoint_id"] in order_by_id:
                        point["reading_order"] = order_by_id[point["keypoint_id"]]
                reuse_entries.append(self._entry("reading_order", image_id, r_key, "reading_cache_hit"))
            else:
                reading_result = assign_reading_and_conflicts(image_id, points, sample["truth"])
                reading_result["cache_key"] = r_key
                self.cache.put("reading_order", r_key, reading_result)
                recompute_entries.append(
                    self._entry("reading_order", image_id, r_key, "keypoint_or_truth_dependency_changed")
                )

            sample_public = {
                "image_id": image_id,
                "image_fingerprint": sample["image_fingerprint"],
                "png_fingerprint": sample["png_fingerprint"],
                "cache_keys": {"reading_order": r_key},
                "keypoint_count": len(reading_result["keypoints"]),
                "conflict_count": len(reading_result["conflicts"]),
            }
            result["samples"].append(sample_public)
            result["keypoints"].extend(reading_result["keypoints"])
            result["reading"].extend(reading_result["reading_sequence"])
            result["conflicts"].extend(reading_result["conflicts"])

        for unit_id in processable_ids:
            column_result = column_results.get(unit_id)
            if column_result:
                result["components"].extend(column_result["public_domains"])

        self.pending_ids = self._unique(
            budget_plan.pending_ids + comparison_deferred + lease_blocked
        )
        pending_queue = []
        for unit_id in self.pending_ids:
            unit = units_by_id.get(unit_id)
            pending_queue.append(
                {
                    "unit_id": unit_id,
                    "planned_points": unit["planned_points"] if unit else None,
                    "reason": "E_POINT_BUDGET_PENDING"
                    if unit_id in budget_plan.pending_ids
                    else ("E_COMPARISON_LIMIT" if unit_id in comparison_deferred else E_LEASE_LIMIT),
                }
            )

        lease_delta = self.leases.events[lease_event_start:]
        for event in lease_delta:
            code = event.get("code")
            if code in counts:
                counts[code] += 1
        result["lease_events"] = lease_delta
        result["lease_snapshot"] = self.leases.active_leases()
        result["pending_queue"] = pending_queue
        result["budget"] = {
            **budget_plan.to_dict(),
            "comparison_events": comparison_events,
            "comparisons_used_this_run": comparisons_used,
            "global_point_budget": config.global_point_budget,
            "column_point_quota": config.column_point_quota,
            "max_component_comparisons": config.max_component_comparisons,
            "lease_blocked_ids": lease_blocked,
            "comparison_deferred_ids": comparison_deferred,
            "output_keypoint_count": len(result["keypoints"]),
            "remaining_points": config.global_point_budget - budget_plan.consumed_points,
            "reserved_lease_points": self.leases.reserved_points,
        }
        result["annotation_version"] = (loaded.annotation_bundle or {}).get("version_id")
        result["annotation_digest"] = (loaded.annotation_bundle or {}).get("content_digest")
        result["cache_stats"] = self.cache.stats()
        result["reuse"] = sorted(reuse_entries, key=lambda e: (e["stage_rank"], e["target"], e["cache_key"]))
        result["recompute"] = sorted(
            recompute_entries, key=lambda e: (e["stage_rank"], e["target"], e["cache_key"])
        )
        result["invalidations"] = self._build_invalidations(
            image_events, version_events, config_events, lease_generations
        )

        self.previous_annotation = loaded.annotation_bundle
        self.previous_image_versions = current_image_versions
        self.previous_lease_generations = dict(lease_generations)
        self.previous_config_hash = config.config_hash
        self.last_result = result
        previous_manifest = read_current_manifest(self.root) if self.write_snapshots else None
        previous_global_version = (
            previous_manifest.get("global_version") if isinstance(previous_manifest, dict) else None
        )
        previous_config_hash = (
            previous_manifest.get("config_hash") if isinstance(previous_manifest, dict) else None
        )
        result["previous_global_version"] = previous_global_version
        result["snapshot_transition"] = {
            "previous_global_version": previous_global_version,
            "previous_config_hash": previous_config_hash,
            "global_version": result_version,
            "config_hash": config.config_hash,
            "version_changed": previous_global_version != result_version
            or previous_config_hash != config.config_hash,
        }
        if self.write_snapshots:
            write_atomic_snapshot(self.root, result, result_version, config.config_hash)
        return result

    def _windows_worker(
        self, context: tuple[dict[str, Any], list[str]], config: Any
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        sample, unit_ids = context
        rects = window_rects(config.width, config.height, config.window_size, config.stride)
        lease_generations = {}
        for unit_id in unit_ids:
            target = unit_id
            lease = self.leases.active_lease_for_target(target)
            if lease:
                rank = int(target.rsplit("c", 1)[1])
                lease_generations[rank] = int(lease["generation"])

        reuse: list[dict[str, Any]] = []
        recompute: list[dict[str, Any]] = []
        image_fp = sample["image_fingerprint"]
        windows: list[dict[str, Any]] = []
        for rect in rects:
            intersect_gens = {
                f"c{rank}": gen
                for rank, gen in lease_generations.items()
                if rect["x"] < sample["truth"]["columns"][rank]["x_max"]
                and rect["x"] + rect["w"] > sample["truth"]["columns"][rank]["x_min"]
            }
            key = window_key(
                image_fp,
                rect,
                config.kernel_hash,
                config.quant_fingerprint(),
                intersect_gens,
            )
            cached = self.cache.get("window", key)
            if cached is not None:
                window = dict(cached)
                window["cache_key"] = key
                windows.append(window)
                reuse.append(self._entry("window", window["window_id"], key, "window_rectangle_and_fingerprint_hit"))
            else:
                window = compute_window(sample["image_id"], sample["pixels"], rect, config)
                window["cache_key"] = key
                self.cache.put("window", key, window)
                windows.append(window)
                recompute.append(self._entry("window", window["window_id"], key, "new_or_changed_window_rectangle"))
        return sample["image_id"], windows, reuse, recompute

    @staticmethod
    def _entry(stage: str, target: str, cache_key: str, reason: str) -> dict[str, Any]:
        stage_ranks = {
            "window": 1,
            "connected_component_prepare": 2,
            "connected_component": 3,
            "keypoint": 4,
            "reading_order": 5,
        }
        return {
            "stage": stage,
            "stage_rank": stage_ranks.get(stage, 9),
            "target": target,
            "cache_key": cache_key,
            "reason": reason,
        }

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        result = []
        for value in values:
            if value not in result:
                result.append(value)
        return result

    def _diff_image_versions(self, current: dict[str, str]) -> list[dict[str, Any]]:
        if not self.previous_image_versions:
            return [
                {
                    "kind": "IMAGE_BASELINE",
                    "image_id": image_id,
                    "previous_fingerprint": None,
                    "current_fingerprint": digest,
                    "decision": "compute_all_stages",
                }
                for image_id, digest in sorted(current.items())
            ]
        events = []
        for image_id in sorted(set(current) | set(self.previous_image_versions)):
            old = self.previous_image_versions.get(image_id)
            new = current.get(image_id)
            if old is None:
                kind, decision = "IMAGE_ADDED", "compute_all_stages"
            elif new is None:
                kind, decision = "IMAGE_DELETED", "drop_all_dependencies"
            elif old != new:
                kind, decision = "IMAGE_REPLACED", "invalidate_all_image_stages"
            else:
                kind, decision = "IMAGE_UNCHANGED", "content_fingerprint_hit"
            events.append(
                {
                    "kind": kind,
                    "image_id": image_id,
                    "previous_fingerprint": old,
                    "current_fingerprint": new,
                    "decision": decision,
                }
            )
        return events

    def _diff_config(self, config_hash: str, config_dict: dict[str, Any]) -> list[dict[str, Any]]:
        if self.previous_config_hash is None:
            return [{"kind": "CONFIG_BASELINE", "config_hash": config_hash, "decision": "compute_all_stages"}]
        if self.previous_config_hash == config_hash:
            return [{"kind": "CONFIG_UNCHANGED", "config_hash": config_hash, "decision": "reuse_by_cache_key"}]
        return [
            {
                "kind": "CONFIG_REPLACED",
                "previous_config_hash": self.previous_config_hash,
                "current_config_hash": config_hash,
                "decision": "cache_keys_include_kernel_quant_and_lease_generation",
                "config": config_dict,
            }
        ]

    def _build_invalidations(
        self,
        image_events: list[dict[str, Any]],
        version_events: list[dict[str, Any]],
        config_events: list[dict[str, Any]],
        lease_generations: dict[str, int],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for event in image_events:
            if event["kind"] in ("IMAGE_REPLACED", "IMAGE_ADDED"):
                for stage in ("window", "connected_component", "keypoint", "reading_order"):
                    records.append(
                        {
                            "stage": stage,
                            "target": event["image_id"],
                            "basis": event["kind"],
                            "fingerprint": event["current_fingerprint"],
                        }
                    )
            elif event["kind"] == "IMAGE_DELETED":
                records.append(
                    {"stage": "all", "target": event["image_id"], "basis": "IMAGE_DELETED"}
                )
        for event in version_events:
            if event.get("kind") == "ANNOTATION_DOC_CHANGED":
                for stage in ("keypoint", "reading_order"):
                    records.append(
                        {
                            "stage": stage,
                            "target": event["image_id"],
                            "basis": "ANNOTATION_DOC_CHANGED",
                            "previous_digest": event["previous_digest"],
                            "current_digest": event["current_digest"],
                        }
                    )
        for event in config_events:
            if event["kind"] == "CONFIG_REPLACED":
                for stage in ("window", "connected_component", "keypoint", "reading_order"):
                    records.append(
                        {"stage": stage, "target": "*", "basis": "CONFIG_REPLACED", "config_hash": event["current_config_hash"]}
                    )
        for target, generation in sorted(lease_generations.items()):
            previous = self.previous_lease_generations.get(target)
            if previous is not None and previous != generation:
                for stage in ("window", "connected_component", "keypoint", "reading_order"):
                    records.append(
                        {
                            "stage": stage,
                            "target": target,
                            "basis": "LEASE_GENERATION_CHANGED",
                            "previous_generation": previous,
                            "current_generation": generation,
                        }
                    )
        records.sort(key=lambda r: (r["stage"], str(r["target"]), r["basis"]))
        return records
