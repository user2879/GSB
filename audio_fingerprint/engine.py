from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

from .canonical import canonical_json, fingerprint
from .config import full_config_fingerprint, parse_config
from .fingerprints import build_pairs_for_clip
from .manifest import load_json_file, validate_manifest
from .normalize import normalize_clip
from .peaks import extract_peaks, frame_threshold, peak_cache_key
from .retrieval import run_query
from .scheduler import allocate_frame_budget, ordered_map, plan_sort_key
from .spectrogram import compute_frame, frame_cache_key, frame_count
from .store import PersistentStore
from .buckets import build_buckets


SCHEDULER_CONFIG_KEYS = {"max_concurrency", "batch_sample_budget", "clip_sample_quota"}


class FingerprintEngine:
    def __init__(
        self,
        manifest: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        data_dir: str | Path = "data",
        state_dir: str | Path = "state",
        *,
        config_issues: list[dict[str, Any]] | None = None,
        serial: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.state_dir = Path(state_dir)
        self.store = PersistentStore(self.state_dir)
        self.serial = serial
        self.config_issues = config_issues or []
        self.stop_requested = False
        self.signal_received: int | None = None
        self.snapshot = self.store.load_snapshot() or self._fresh_snapshot()
        self.manifest = manifest if manifest is not None else self.snapshot.get("manifest") or {"clips": []}
        self.config = config if config is not None else self.snapshot.get("config") or {}
        self._context: dict[str, Any] | None = None

    def _fresh_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "global_version": 0,
            "config_fingerprint": None,
            "manifest_fingerprint": None,
            "active_run": None,
            "issues": [],
            "todos": [],
            "normalizations": [],
            "clips": [],
            "frames": {},
            "peak_sets": {},
            "pairs": {},
            "buckets": {},
            "query_results": {},
            "receipts": [],
            "last_batch": None,
            "last_receipt": None,
        }

    @classmethod
    def from_workspace(cls, workspace_dir: str | Path, state_dir: str | Path | None = None) -> "FingerprintEngine":
        root = Path(workspace_dir)
        manifest, manifest_load_issues = load_json_file(root / "manifest.json")
        raw_config, config_load_issues = load_json_file(root / "config.json")
        config = raw_config if isinstance(raw_config, dict) else {}
        engine = cls(
            manifest if isinstance(manifest, dict) else {"clips": []},
            config,
            root / "data",
            state_dir or root / "state",
            config_issues=list(config_load_issues) + list(manifest_load_issues),
        )
        return engine

    def request_stop(self, signum: int | None = None) -> None:
        self.stop_requested = True
        self.signal_received = signum

    def install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.request_stop(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, handler)
            except ValueError:
                pass

    def prepare(self, force: bool = True) -> dict[str, Any]:
        if self._context is not None and not force:
            return self._context
        config, parsed_issues = parse_config(self.config)
        issues = list(self.config_issues) + list(parsed_issues)
        valid_clips, manifest_issues = validate_manifest(self.manifest, self.data_dir)
        issues.extend(manifest_issues)

        normalizations: list[dict[str, Any]] = []
        samples_by_clip: dict[str, Any] = {}
        clips_by_id: dict[str, dict[str, Any]] = {}
        plans: list[dict[str, Any]] = []

        for clip in valid_clips:
            clips_by_id[clip["clip_id"]] = clip
            normalization, samples, read_issues = normalize_clip(clip, self.data_dir, config["internal_sample_rate"])
            if read_issues:
                issues.extend(read_issues)
                continue
            normalizations.append(normalization)
            samples_by_clip[clip["clip_id"]] = samples
            count = frame_count(samples.shape[0], config)
            for frame_index in range(count):
                start = frame_index * int(config["hop_size"])
                key = frame_cache_key(clip, normalization, config, frame_index, start)
                plans.append(
                    {
                        "product_type": "spec_frame",
                        "product_id": f"frame:{clip['clip_id']}:{frame_index:06d}",
                        "cache_key": key,
                        "track_id": clip["track_id"],
                        "version": int(clip["version"]),
                        "clip_id": clip["clip_id"],
                        "role": clip.get("role", "library"),
                        "frame_index": frame_index,
                        "start_sample": start,
                        "points_cost": int(config["frame_size"]),
                    }
                )

        plans.sort(key=plan_sort_key)
        validation_todos = [
            {
                "todo_id": f"todo_val_{index:04d}_" + fingerprint(issue)[:12],
                "kind": "validation_failure",
                "business_code": issue["code"],
                "status": "open",
                "priority": 100,
                "target": issue["target"],
                "message": issue["message"],
            }
            for index, issue in enumerate(issues)
        ]
        clip_index = [
            {
                "clip_id": clip["clip_id"],
                "track_id": clip["track_id"],
                "version": int(clip["version"]),
                "role": clip.get("role", "library"),
                "seed": clip.get("seed"),
                "path": clip.get("path"),
                "source_sample_rate": clip["source_sample_rate"],
                "bit_depth": clip["bit_depth"],
                "declared_sample_rates": clip.get("declared_sample_rates", []),
                "declared_bit_depths": clip.get("declared_bit_depths", []),
                "content_fingerprint": clip.get("content_fingerprint"),
                "format_fingerprint": clip["format_fingerprint"],
            }
            for clip in valid_clips
        ]
        manifest_fp = fingerprint(self.manifest, "manifest_")
        config_fp = full_config_fingerprint(config)
        self.config = config
        self._context = {
            "config": config,
            "issues": issues,
            "validation_todos": validation_todos,
            "valid_clips": valid_clips,
            "clip_index": clip_index,
            "clips_by_id": clips_by_id,
            "normalizations": normalizations,
            "samples_by_clip": samples_by_clip,
            "plans": plans,
            "manifest_fingerprint": manifest_fp,
            "config_fingerprint": config_fp,
        }
        return self._context

    def start_run(self, command: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
        ctx = self.prepare()
        old_snapshot = self.snapshot
        hit_keys: list[str] = []
        missing_keys: list[str] = []
        for plan in ctx["plans"]:
            if self.store.has(plan["cache_key"]):
                hit_keys.append(plan["cache_key"])
            else:
                missing_keys.append(plan["cache_key"])

        frame_id_to_key = {plan["product_id"]: plan["cache_key"] for plan in ctx["plans"]}
        invalidations = self._product_key_changes(
            old_snapshot.get("frames", {}),
            frame_id_to_key,
            "spec_frame",
        )
        run_id = run_id or (command.get("command_id") if command else "rebuild")
        active_run = {
            "run_id": run_id,
            "command": command,
            "complete": False,
            "batch_count": 0,
            "frame_plan_count": len(ctx["plans"]),
            "hit_frame_keys": sorted(hit_keys),
            "missing_frame_keys": sorted(missing_keys),
            "built_frame_keys": [],
            "frame_id_to_key": frame_id_to_key,
            "invalidation_entries": invalidations,
        }
        self.snapshot.update(
            {
                "manifest": self.manifest,
                "config": ctx["config"],
                "config_fingerprint": ctx["config_fingerprint"],
                "manifest_fingerprint": ctx["manifest_fingerprint"],
                "active_run": active_run,
                "issues": ctx["issues"],
                "todos": ctx["validation_todos"],
                "clips": ctx["clip_index"],
                "normalizations": ctx["normalizations"],
            }
        )
        self._commit()
        return {
            "run_id": run_id,
            "frame_count": len(ctx["plans"]),
            "cache_hit_frames": len(hit_keys),
            "missing_frames": len(missing_keys),
            "invalidation_entries": invalidations,
            "global_version": self.snapshot["global_version"],
        }

    def _product_key_changes(
        self,
        old_products: dict[str, Any],
        new_id_to_key: dict[str, str],
        product_type: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for product_id in sorted(old_products):
            old = old_products[product_id]
            old_key = old.get("cache_key") if isinstance(old, dict) else None
            new_key = new_id_to_key.get(product_id)
            if new_key is None:
                entries.append(
                    {
                        "event": "removed",
                        "product_type": product_type,
                        "product_id": product_id,
                        "old_cache_key": old_key,
                        "new_cache_key": None,
                        "reason": "product_no_longer_referenced",
                    }
                )
            elif old_key != new_key:
                entries.append(
                    {
                        "event": "invalidated",
                        "product_type": product_type,
                        "product_id": product_id,
                        "old_cache_key": old_key,
                        "new_cache_key": new_key,
                        "reason": "upstream_fingerprint_changed",
                    }
                )
        for product_id, new_key in sorted(new_id_to_key.items()):
            if product_id not in old_products:
                entries.append(
                    {
                        "event": "created",
                        "product_type": product_type,
                        "product_id": product_id,
                        "old_cache_key": None,
                        "new_cache_key": new_key,
                        "reason": "new_product",
                    }
                )
        return entries

    def run_next_batch(self) -> dict[str, Any]:
        if self.stop_requested:
            return {"status": "graceful_stop_before_batch", "completed": False}
        ctx = self.prepare()
        if self.snapshot.get("active_run") is None:
            self.start_run()
        active_run = self.snapshot["active_run"]
        config = ctx["config"]
        pending_plans = [plan for plan in ctx["plans"] if not self.store.has(plan["cache_key"])]
        allocation = allocate_frame_budget(
            pending_plans,
            int(config["batch_sample_budget"]),
            int(config["clip_sample_quota"]),
        )

        def worker(plan: dict[str, Any]) -> dict[str, Any]:
            try:
                clip = ctx["clips_by_id"][plan["clip_id"]]
                samples = ctx["samples_by_clip"][plan["clip_id"]]
                normalization = next(
                    item for item in ctx["normalizations"] if item["clip_id"] == plan["clip_id"]
                )
                return compute_frame(clip, normalization, samples, plan["frame_index"], config)
            except Exception as exc:  # deterministic failure unit, do not kill whole batch
                return {"__error__": True, "product_id": plan["product_id"], "cache_key": plan["cache_key"], "error": repr(exc)}

        results = ordered_map(
            allocation["selected"],
            worker,
            int(config["max_concurrency"]),
            serial=self.serial,
        )
        batch_issues: list[dict[str, Any]] = []
        built_keys = set(active_run.get("built_frame_keys", []))
        for result in results:
            if result.get("__error__"):
                batch_issues.append(
                    {
                        "code": "B4003",
                        "target": result["product_id"],
                        "message": "frame computation failed",
                        "detail": {"error": result["error"]},
                    }
                )
                continue
            self.store.put(result)
            built_keys.add(result["cache_key"])

        pending_after = [plan for plan in ctx["plans"] if not self.store.has(plan["cache_key"])]
        scheduler_todos = list(ctx["validation_todos"])
        scheduler_todos.extend(allocation["todos"])
        active_run["batch_count"] = int(active_run.get("batch_count", 0)) + 1
        active_run["built_frame_keys"] = sorted(built_keys)
        active_run["missing_frame_keys"] = sorted(plan["cache_key"] for plan in pending_after)
        self.snapshot["issues"] = list(ctx["issues"]) + batch_issues
        self.snapshot["todos"] = scheduler_todos
        self.snapshot["last_batch"] = {
            "batch_number": active_run["batch_count"],
            "serial": self.serial,
            "max_concurrency": int(config["max_concurrency"]),
            "selected_count": len(allocation["selected"]),
            "selected_product_ids": [plan["product_id"] for plan in allocation["selected"]],
            "deferred_count": len(allocation["deferred"]),
            "events": allocation["events"],
            "used_budget_points": allocation["used_budget_points"],
            "remaining_budget_points": allocation["remaining_budget_points"],
            "used_by_clip": allocation["used_by_clip"],
            "pending_frame_count": len(pending_after),
        }
        self.snapshot["active_run"] = active_run
        self.snapshot["frames"] = {
            plan["product_id"]: self.store.get(plan["cache_key"])
            for plan in ctx["plans"]
            if self.store.has(plan["cache_key"])
        }
        self._commit()

        if self.stop_requested:
            return {
                "status": "graceful_stop_after_committed_batch",
                "completed": False,
                "batch": self.snapshot["last_batch"],
                "signal": self.signal_received,
            }
        if pending_after:
            return {
                "status": "batch_committed",
                "completed": False,
                "batch": self.snapshot["last_batch"],
            }

        receipt = self._complete_run(ctx, active_run)
        return {"status": "run_completed", "completed": True, "receipt": receipt}

    def run_until_complete(self, max_batches: int = 10000) -> dict[str, Any]:
        if self.snapshot.get("active_run") is None:
            self.start_run()
        last: dict[str, Any] = {}
        for _ in range(max_batches):
            if self.stop_requested:
                self._commit()
                return {"status": "graceful_shutdown", "last": last, "global_version": self.snapshot["global_version"]}
            last = self.run_next_batch()
            if last.get("completed"):
                return last
            if last.get("status", "").startswith("graceful_stop"):
                return {"status": "graceful_shutdown", "last": last, "global_version": self.snapshot["global_version"]}
        raise RuntimeError("max_batches exceeded before frame budget completed")

    def _complete_run(self, ctx: dict[str, Any], active_run: dict[str, Any]) -> dict[str, Any]:
        config = ctx["config"]
        plans = ctx["plans"]
        frames_by_id = {plan["product_id"]: self.store.get(plan["cache_key"]) for plan in plans}
        frames_by_clip: dict[str, list[dict[str, Any]]] = {}
        for frame in frames_by_id.values():
            frames_by_clip.setdefault(frame["clip_id"], []).append(frame)
        for frames in frames_by_clip.values():
            frames.sort(key=lambda item: item["frame_index"])

        ledger: list[dict[str, Any]] = []
        for plan in plans:
            record = frames_by_id[plan["product_id"]]
            if plan["cache_key"] in set(active_run.get("built_frame_keys", [])):
                status = "rebuilt"
                reason = "cache_miss_invalid_unit"
            elif plan["cache_key"] in set(active_run.get("hit_frame_keys", [])):
                status = "reuse_hit"
                reason = "content_addressed_key_match"
            else:
                status = "reuse_hit"
                reason = "committed_prior_product"
            ledger.append(self._ledger_entry("spec_frame", plan["product_id"], plan["cache_key"], status, reason, record.get("upstream_refs", [])))

        new_peak_id_to_key: dict[str, str] = {}
        peak_products: dict[str, dict[str, Any]] = {}
        pair_products: dict[str, dict[str, Any]] = {}
        pairs_by_clip: dict[str, list[dict[str, Any]]] = {}

        for clip in ctx["valid_clips"]:
            clip_id = clip["clip_id"]
            frames = frames_by_clip.get(clip_id, [])
            peak_sets: list[dict[str, Any]] = []
            radius = int(config["peak_neighborhood_radius"])
            for frame in frames:
                index = int(frame["frame_index"])
                neighbors = [
                    item
                    for item in frames
                    if abs(int(item["frame_index"]) - index) <= radius
                ]
                threshold = frame_threshold(config, clip_id, index)
                key = peak_cache_key(clip, index, threshold, neighbors, config)
                new_peak_id_to_key[f"peaks:{clip_id}:{index:06d}"] = key
                if self.store.has(key):
                    product = self.store.get(key)
                    status = "reuse_hit"
                    reason = "peak_rule_and_frame_refs_match"
                else:
                    product = extract_peaks(clip, index, neighbors, config)
                    self.store.put(product)
                    status = "rebuilt"
                    reason = "peak_rule_or_upstream_frame_changed"
                peak_sets.append(product)
                peak_products[product["product_id"]] = product
                ledger.append(self._ledger_entry("peak_set", product["product_id"], key, status, reason, product.get("upstream_refs", [])))

            pairs = build_pairs_for_clip(clip, peak_sets, config)
            pairs_by_clip[clip_id] = pairs
            for pair in pairs:
                if self.store.has(pair["cache_key"]):
                    product = self.store.get(pair["cache_key"])
                    status = "reuse_hit"
                    reason = "peak_set_refs_and_pair_rule_match"
                else:
                    product = pair
                    self.store.put(product)
                    status = "rebuilt"
                    reason = "upstream_peak_or_quant_rule_changed"
                pair_products[product["product_id"]] = product
                ledger.append(self._ledger_entry("fingerprint_pair", product["product_id"], product["cache_key"], status, reason, product.get("upstream_refs", [])))

        library_pairs = [
            pair
            for clip in ctx["valid_clips"]
            if clip.get("role", "library") != "query"
            for pair in pairs_by_clip.get(clip["clip_id"], [])
        ]
        computed_buckets = build_buckets(library_pairs, config)
        bucket_products: dict[str, dict[str, Any]] = {}
        new_bucket_id_to_key: dict[str, str] = {}
        for bucket in computed_buckets:
            new_bucket_id_to_key[bucket["product_id"]] = bucket["cache_key"]
            if self.store.has(bucket["cache_key"]):
                product = self.store.get(bucket["cache_key"])
                status = "reuse_hit"
                reason = "all_referenced_pair_keys_and_capacity_match"
            else:
                product = bucket
                self.store.put(product)
                status = "rebuilt"
                reason = "referenced_pair_or_capacity_changed"
            bucket_products[bucket["fingerprint_key"]] = product
            ledger.append(self._ledger_entry("inverted_bucket", product["product_id"], product["cache_key"], status, reason, product.get("upstream_refs", [])))

        query_results: dict[str, dict[str, Any]] = {}
        new_query_id_to_key: dict[str, str] = {}
        for clip in sorted((c for c in ctx["valid_clips"] if c.get("role") == "query"), key=lambda c: c["clip_id"]):
            result = run_query(clip, pairs_by_clip.get(clip["clip_id"], []), computed_buckets, config)
            new_query_id_to_key[result["product_id"]] = result["cache_key"]
            if self.store.has(result["cache_key"]):
                product = self.store.get(result["cache_key"])
                status = "reuse_hit"
                reason = "query_pairs_and_touched_buckets_match"
            else:
                product = result
                self.store.put(product)
                status = "rebuilt"
                reason = "query_or_bucket_vote_input_changed"
            query_results[clip["clip_id"]] = product
            ledger.append(self._ledger_entry("query_result", product["product_id"], product["cache_key"], status, reason, product.get("upstream_refs", [])))

        invalidation_entries = list(active_run.get("invalidation_entries", []))
        invalidation_entries.extend(self._product_key_changes(self.snapshot.get("peak_sets", {}), new_peak_id_to_key, "peak_set"))
        invalidation_entries.extend(self._product_key_changes(self.snapshot.get("pairs", {}), {pid: p["cache_key"] for pid, p in pair_products.items()}, "fingerprint_pair"))
        invalidation_entries.extend(self._product_key_changes(self.snapshot.get("buckets", {}), new_bucket_id_to_key, "inverted_bucket"))
        invalidation_entries.extend(self._product_key_changes(self.snapshot.get("query_results", {}), new_query_id_to_key, "query_result"))

        ledger.sort(key=lambda entry: (entry["product_type_order"], entry["product_id"], entry["cache_key"], entry["status"]))
        for index, entry in enumerate(ledger):
            entry["sequence"] = index
        receipt = {
            "run_id": active_run["run_id"],
            "command": active_run.get("command"),
            "batch_count": active_run.get("batch_count", 0),
            "ledger_entry_count": len(ledger),
            "reuse_hit_count": sum(entry["status"] == "reuse_hit" for entry in ledger),
            "rebuilt_count": sum(entry["status"] == "rebuilt" for entry in ledger),
            "invalidation_count": len(invalidation_entries),
            "ledger": ledger,
            "invalidation_entries": invalidation_entries,
        }
        active_run["complete"] = True
        active_run["completed"] = True
        self.snapshot.update(
            {
                "active_run": active_run,
                "todos": ctx["validation_todos"],
                "frames": {pid: frames_by_id[pid] for pid in sorted(frames_by_id)},
                "peak_sets": peak_products,
                "pairs": pair_products,
                "buckets": bucket_products,
                "query_results": query_results,
                "last_receipt": receipt,
            }
        )
        self.snapshot.setdefault("receipts", []).append(
            {key: value for key, value in receipt.items() if key != "ledger"}
        )
        self._commit()
        return receipt

    def _ledger_entry(
        self,
        product_type: str,
        product_id: str,
        cache_key: str,
        status: str,
        reason: str,
        upstream_refs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        order = {
            "spec_frame": 1,
            "peak_set": 2,
            "fingerprint_pair": 3,
            "inverted_bucket": 4,
            "query_result": 5,
        }[product_type]
        return {
            "product_type": product_type,
            "product_type_order": order,
            "product_id": product_id,
            "cache_key": cache_key,
            "status": status,
            "reason": reason,
            "upstream_ref_count": len(upstream_refs),
            "upstream_refs": upstream_refs,
        }

    def _commit(self) -> None:
        self.snapshot["global_version"] = int(self.snapshot.get("global_version", 0)) + 1
        if self._context is not None:
            self.snapshot["config_fingerprint"] = self._context["config_fingerprint"]
            self.snapshot["manifest_fingerprint"] = self._context["manifest_fingerprint"]
        else:
            self.snapshot["config_fingerprint"] = full_config_fingerprint(self.config)
        self.store.commit_snapshot(self.snapshot)

    def export_canonical(self) -> dict[str, Any]:
        ctx = self.prepare()
        snapshot = self.snapshot
        config = {key: value for key, value in ctx["config"].items() if key not in SCHEDULER_CONFIG_KEYS}
        frames = sorted(snapshot.get("frames", {}).values(), key=lambda item: item["product_id"])
        peak_sets = sorted(snapshot.get("peak_sets", {}).values(), key=lambda item: item["product_id"])
        pairs = sorted(snapshot.get("pairs", {}).values(), key=lambda item: item["product_id"])
        buckets = sorted(snapshot.get("buckets", {}).values(), key=lambda item: item["fingerprint_key"])
        queries = sorted(snapshot.get("query_results", {}).values(), key=lambda item: item["query_clip_id"])
        return {
            "schema_version": snapshot.get("schema_version", 1),
            "retrieval_config": config,
            "issues": ctx["issues"],
            "failure_codes": [issue["code"] for issue in ctx["issues"]],
            "clips": ctx["clip_index"],
            "normalizations": sorted(ctx["normalizations"], key=lambda item: item["clip_id"]),
            "frames": frames,
            "peak_sets": peak_sets,
            "pairs": pairs,
            "buckets": buckets,
            "query_results": queries,
            "cache_key_index": {
                "frames": {item["product_id"]: item["cache_key"] for item in frames},
                "peak_sets": {item["product_id"]: item["cache_key"] for item in peak_sets},
                "pairs": {item["product_id"]: item["cache_key"] for item in pairs},
                "buckets": {item["product_id"]: item["cache_key"] for item in buckets},
                "query_results": {item["product_id"]: item["cache_key"] for item in queries},
            },
        }

    def canonical_fingerprint(self) -> str:
        return fingerprint(self.export_canonical(), "canonical_")


def deep_diff(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append(f"{path}.{key}: missing in left")
            elif key not in right:
                differences.append(f"{path}.{key}: missing in right")
            else:
                differences.extend(deep_diff(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: list length {len(left)} != {len(right)}"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(deep_diff(left_item, right_item, f"{path}[{index}]"))
        return differences
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def canonical_dump(value: dict[str, Any]) -> str:
    return canonical_json(value)
