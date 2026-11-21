from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, validate_config
from .errors import SnapshotStaleError, WorkbenchError
from .loader import initial_sources_hash, load_shards, materialize_input, read_json
from .merge import merge_candidates
from .model import LoadResult, MaterializedInput
from .rerank import rerank
from .scoring import (
    contribution_breakdown,
    fuse_candidates,
    normalize_channels,
)
from .util import (
    atomic_publish_directory,
    atomic_write_json,
    canonical_json,
    stable_hash,
)


SCHEMA_VERSION = 1
STAGE_NAMES = ("alignment", "merge", "normalize", "fusion", "rerank")


@dataclass
class RunConfig:
    samples_dir: Path
    output_dir: Path
    step: int
    mode: str = "incremental"
    serial: bool = False
    budget_override: int | None = None
    config_override: Path | None = None
    simulate_stale: bool = False


class Workbench:
    def __init__(self, config: RunConfig):
        self.config = config
        self.samples_dir = config.samples_dir.resolve()
        self.output_dir = config.output_dir.resolve()
        self.live_dir = self.output_dir / "live"
        self.tmp_root = self.output_dir / ".tmp"

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        previous = self._load_previous_snapshot() if self.config.mode == "incremental" else None
        if self.config.simulate_stale:
            error = SnapshotStaleError(
                "simulated stale snapshot",
                details={"reason": "INJ snapshot-stale injection"},
            )
            self._write_error(error)
            raise error

        materialized = materialize_input(
            self.samples_dir,
            self.config.step,
            config_override=self.config.config_override,
        )
        cfg = validate_config(materialized.config_raw)
        if previous is not None:
            reason = self._stale_reason(previous, materialized)
            if reason:
                error = SnapshotStaleError(
                    reason,
                    details={
                        "previous_steps": previous["manifest"].get("applied_steps"),
                        "previous_config_hash": previous["manifest"].get("config_hash"),
                        "current_config_hash": cfg.config_hash,
                    },
                )
                self._write_error(error)
                raise error

        load_result = load_shards(materialized, cfg, self.samples_dir)
        failures = self._sorted_failures(load_result)
        workers = max(1, int(cfg.raw["thread_pool_workers"]))
        if self.config.serial:
            workers = 1
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            previous_caches = previous["cache"] if previous else {}
            previous_pending = (
                previous["state"].get("pending", [])
                if previous
                else []
            )
            merge_records, merge_cache, merge_meta = merge_candidates(
                load_result.shards,
                cfg,
                previous_caches.get("merge", {}),
            )
            norm_channels, norm_cache, norm_meta = normalize_channels(
                merge_records,
                cfg,
                previous_caches.get("normalize", {}),
                executor=executor,
            )
            previous_rounds = (
                previous["state"].get("budget_rounds", [])
                if previous
                else []
            )
            fused_records, fusion_cache, fusion_meta = fuse_candidates(
                norm_channels,
                cfg,
                previous_caches.get("fusion", {}),
                pending_items=previous_pending,
                executor=executor,
                budget_override=self.config.budget_override,
                round_number=len(previous_rounds) + 1,
            )
            rerank_results, rerank_cache, rerank_meta = rerank(
                fused_records,
                load_result.attributes,
                load_result.attributes_fingerprint,
                cfg,
                previous_caches.get("rerank", {}),
                pending_items=fusion_meta["pending"],
                executor=executor,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        current_caches = {
            "merge": merge_cache,
            "normalize": norm_cache,
            "fusion": fusion_cache,
            "rerank": rerank_cache,
        }
        cache_report = self._cache_report(previous_caches, current_caches, {
            "merge": merge_meta,
            "normalize": norm_meta,
            "fusion": fusion_meta,
            "rerank": rerank_meta,
        })
        shard_changes, affected_shards = self._shard_changes(previous, load_result)
        budget_round = {
            **fusion_meta["budget"],
            "global_version": (previous["manifest"]["global_version"] + 1) if previous else 1,
            "input_pending_count": len(previous_pending),
            "deduction_sequence_hash": stable_hash(fusion_meta["budget"]["deductions"]),
        }
        budget_rounds = (
            previous["state"].get("budget_rounds", []) if previous else []
        ) + [budget_round]

        injection_hits, injection_history = self._evaluate_injections(
            failures,
            fusion_meta,
            rerank_meta,
            cfg,
            previous,
        )
        boundary_samples = self._boundary_samples(fusion_meta, rerank_meta)
        report = self._build_report(
            materialized,
            cfg,
            load_result,
            failures,
            merge_records,
            norm_channels,
            fused_records,
            rerank_results,
            fusion_meta,
            cache_report,
            shard_changes,
            affected_shards,
            budget_rounds,
            injection_hits,
            boundary_samples,
            previous,
        )
        canonical = self._canonical(report)
        canonical_hash = stable_hash(canonical)
        report["canonical_hash"] = canonical_hash
        report["comparison_contract"] = (
            "canonical.json contains every business field. Runtime cache telemetry, "
            "cumulative budget history and stage generations are intentionally outside "
            "the incremental-vs-full rebuild equivalence contract."
        )

        stage_versions = self._next_stage_versions(previous, report)
        report["snapshots"] = {
            "global_version": report["run"]["global_version"],
            "config_hash": cfg.config_hash,
            "stage_versions": stage_versions,
            "commit_interval_shards": int(cfg.raw["snapshot_commit_interval_shards"]),
            "changed_shards_this_run": len(
                [entry for entry in shard_changes if entry["change"] != "unchanged"]
            ),
        }
        state = {
            "pending": fusion_meta["pending"],
            "budget_rounds": budget_rounds,
            "injection_history": injection_history,
            "last_canonical_hash": canonical_hash,
            "last_mode": self.config.mode,
        }
        manifest = self._manifest(
            previous,
            materialized,
            cfg,
            load_result,
            stage_versions,
            budget_rounds,
            self.config.mode,
        )
        self._publish_snapshot(
            manifest,
            state,
            current_caches,
            report,
            canonical,
        )
        atomic_write_json(self.output_dir / "report.json", report)
        atomic_write_json(self.output_dir / "canonical.json", canonical)
        error_path = self.output_dir / "error_report.json"
        if error_path.exists():
            error_path.unlink()
        return report

    def _load_previous_snapshot(self) -> dict[str, Any] | None:
        manifest_path = self.live_dir / "manifest.json"
        cache_path = self.live_dir / "cache.json"
        state_path = self.live_dir / "state.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = read_json(manifest_path)
            cache = read_json(cache_path) if cache_path.exists() else {}
            state = read_json(state_path) if state_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            error = SnapshotStaleError(f"cannot read prior snapshot: {exc}")
            self._write_error(error)
            raise error from exc
        return {"manifest": manifest, "cache": cache, "state": state}

    def _stale_reason(
        self,
        previous: dict[str, Any],
        current: MaterializedInput,
    ) -> str | None:
        manifest = previous["manifest"]
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return "snapshot schema version differs from current workbench"
        if not isinstance(manifest.get("global_version"), int):
            return "snapshot global_version is missing or invalid"
        if manifest.get("mode") == "full":
            return "a full-rebuild snapshot cannot be resumed incrementally"
        if manifest.get("base_source_hash") != initial_sources_hash(self.samples_dir):
            return "base sample fingerprint differs from snapshot"
        previous_steps = manifest.get("applied_steps")
        if not isinstance(previous_steps, list):
            return "snapshot applied_steps is missing or invalid"
        if current.applied_steps[: len(previous_steps)] != previous_steps:
            return "requested plan prefix diverges from snapshot"
        last_step = previous_steps[-1] if previous_steps else 0
        prefix = materialize_input(
            self.samples_dir,
            last_step,
            config_override=self.config.config_override,
        )
        prefix_cfg = validate_config(prefix.config_raw)
        if manifest.get("plan_hash") != prefix.effective_plan_hash:
            return "plan prefix fingerprint differs from snapshot"
        if manifest.get("config_hash") != prefix_cfg.config_hash:
            return "config hash differs from snapshot"
        stage_versions = manifest.get("stage_versions")
        if not isinstance(stage_versions, dict):
            return "snapshot stage_versions is missing"
        for stage in STAGE_NAMES:
            entry = stage_versions.get(stage)
            if not isinstance(entry, dict) or not isinstance(entry.get("version"), int):
                return f"snapshot stage version is missing for {stage}"
        return None

    def _sorted_failures(self, load_result: LoadResult) -> list[dict[str, Any]]:
        combined = load_result.failures + load_result.attributes_failures
        combined.sort(
            key=lambda failure: (
                failure.source,
                failure.shard_id or "",
                failure.line_no if failure.line_no is not None else -1,
                failure.reason,
                failure.code,
            )
        )
        return [failure.as_dict() for failure in combined]

    def _cache_report(
        self,
        previous: dict[str, dict[str, Any]],
        current: dict[str, dict[str, Any]],
        telemetry: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        total_invalidated = 0
        total_reused = 0
        total_computed = 0
        for stage in ("merge", "normalize", "fusion", "rerank"):
            old_keys = set(previous.get(stage, {}))
            new_keys = set(current.get(stage, {}))
            invalidated = sorted(old_keys - new_keys)
            reused = sorted(old_keys & new_keys)
            added = sorted(new_keys - old_keys)
            total_invalidated += len(invalidated)
            total_reused += len(reused)
            total_computed += int(telemetry[stage].get("computed", 0))
            stages[stage] = {
                "active_keys": len(new_keys),
                "invalidated_count": len(invalidated),
                "reused_count": len(reused),
                "added_count": len(added),
                "computed_count": int(telemetry[stage].get("computed", 0)),
                "telemetry_reused_count": int(telemetry[stage].get("reused", 0)),
                "invalidated_keys": invalidated,
                "reused_keys": reused,
                "added_keys": added,
            }
        return {
            "totals": {
                "invalidated": total_invalidated,
                "reused": total_reused,
                "computed": total_computed,
            },
            "stages": stages,
        }

    def _shard_changes(
        self,
        previous: dict[str, Any] | None,
        load_result: LoadResult,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        previous_shards = (
            {
                entry["shard_id"]: entry["fingerprint"]
                for entry in previous["manifest"].get("completed_shards", [])
            }
            if previous
            else {}
        )
        current_shards = {
            entry["shard_id"]: entry["fingerprint"]
            for entry in load_result.complete_shard_set
        }
        changes = []
        for shard_id in sorted(set(previous_shards) | set(current_shards)):
            if shard_id not in previous_shards:
                change = "added"
            elif shard_id not in current_shards:
                change = "removed"
            elif previous_shards[shard_id] != current_shards[shard_id]:
                change = "changed"
            else:
                change = "unchanged"
            changes.append({"shard_id": shard_id, "change": change})
        affected = [
            entry["shard_id"] for entry in changes if entry["change"] != "unchanged"
        ]
        return changes, affected

    def _boundary_samples(
        self,
        fusion_meta: dict[str, Any],
        rerank_meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        samples = list(fusion_meta.get("boundary_samples", [])) + list(
            rerank_meta.get("boundary_records", [])
        )
        samples.sort(key=lambda item: canonical_json(item))
        return samples

    def _evaluate_injections(
        self,
        failures: list[dict[str, Any]],
        fusion_meta: dict[str, Any],
        rerank_meta: dict[str, Any],
        cfg: RuntimeConfig,
        previous: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        path = self.samples_dir / "injections.json"
        if not path.exists():
            return [], (previous["state"].get("injection_history", []) if previous else [])
        try:
            raw = read_json(path)
        except (OSError, json.JSONDecodeError):
            return [], (previous["state"].get("injection_history", []) if previous else [])
        items = raw.get("items", []) if isinstance(raw, dict) else []
        current_hits = []
        for item in items:
            if not isinstance(item, dict) or item.get("enabled", True) is False:
                continue
            injection_id = item.get("id")
            injection_type = item.get("type", "")
            evidence: list[dict[str, Any]] = []
            if injection_type == "invalid_shard":
                expected = item.get("expected_code", "E_SHARD_INVALID")
                for failure in failures:
                    if failure["code"] == expected and (
                        failure.get("injection_id") == injection_id
                        or not failure.get("injection_id")
                    ):
                        evidence.append(
                            {
                                "source": failure["source"],
                                "line_no": failure.get("line_no"),
                                "code": failure["code"],
                            }
                        )
            elif injection_type in {"fusion_clip_boundary", "fusion_tie", "budget_pending"}:
                if injection_type == "fusion_clip_boundary":
                    evidence = [
                        sample
                        for sample in fusion_meta.get("boundary_samples", [])
                        if sample.get("criterion") == "fusion_score_exact_clip_boundary"
                    ]
                elif injection_type == "fusion_tie":
                    evidence = [
                        sample
                        for sample in fusion_meta.get("boundary_samples", [])
                        if sample.get("criterion") == "fusion_score_tie_item_id_order"
                    ]
                else:
                    evidence = list(fusion_meta.get("pending", []))
            elif injection_type in {
                "similarity_threshold_equal",
                "rerank_tie",
                "discount",
            }:
                if injection_type == "similarity_threshold_equal":
                    evidence = [
                        sample
                        for sample in rerank_meta.get("boundary_records", [])
                        if sample.get("criterion")
                        == "similarity_equal_threshold_no_discount"
                    ]
                elif injection_type == "rerank_tie":
                    evidence = [
                        sample
                        for sample in rerank_meta.get("boundary_records", [])
                        if sample.get("criterion") == "rerank_score_tie_item_id_order"
                    ]
                else:
                    evidence = list(rerank_meta.get("discount_records", []))
            elif injection_type == "zero_denominator":
                evidence = []
            elif injection_type == "snapshot_stale":
                evidence = []
            if evidence:
                current_hits.append(
                    {
                        "id": injection_id,
                        "type": injection_type,
                        "hit": True,
                        "evidence_count": len(evidence),
                        "evidence": evidence[:10],
                    }
                )
        history = previous["state"].get("injection_history", []) if previous else []
        by_id = {entry["id"]: dict(entry) for entry in history if isinstance(entry, dict)}
        for hit in current_hits:
            existing = by_id.get(hit["id"])
            if existing is None:
                existing = {
                    "id": hit["id"],
                    "type": hit["type"],
                    "first_evidence_count": hit["evidence_count"],
                    "evidence": hit["evidence"],
                }
                by_id[hit["id"]] = existing
        history_out = [by_id[key] for key in sorted(by_id)]
        merged_hits = []
        for hit in current_hits:
            merged_hits.append(hit)
        current_ids = {hit["id"] for hit in current_hits}
        for entry in history_out:
            if entry["id"] not in current_ids:
                merged_hits.append(
                    {
                        "id": entry["id"],
                        "type": entry["type"],
                        "hit": True,
                        "historical": True,
                        "evidence_count": entry.get("first_evidence_count", 0),
                        "evidence": entry.get("evidence", [])[:10],
                    }
                )
        merged_hits.sort(key=lambda entry: entry["id"])
        return merged_hits, history_out

    def _build_report(
        self,
        materialized: MaterializedInput,
        cfg: RuntimeConfig,
        load_result: LoadResult,
        failures: list[dict[str, Any]],
        merge_records: list[dict[str, Any]],
        norm_channels: list[dict[str, Any]],
        fused_records: list[dict[str, Any]],
        rerank_results: list[dict[str, Any]],
        fusion_meta: dict[str, Any],
        cache_report: dict[str, Any],
        shard_changes: list[dict[str, Any]],
        affected_shards: list[str],
        budget_rounds: list[dict[str, Any]],
        injection_hits: list[dict[str, Any]],
        boundary_samples: list[dict[str, Any]],
        previous: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merge_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in merge_records:
            merge_by_request[record["request_id"]].append(record)
        fusion_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in fused_records:
            fusion_by_request[record["request_id"]].append(record)
        rerank_by_request = {result["request_id"]: result for result in rerank_results}
        pending_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in fusion_meta["pending"]:
            pending_by_request[entry["request_id"]].append(entry)
        contributions = contribution_breakdown(merge_records)
        requests = sorted(
            set(merge_by_request)
            | set(fusion_by_request)
            | set(pending_by_request)
            | set(rerank_by_request)
        )
        request_reports = []
        for request_id in requests:
            rmerge = sorted(
                merge_by_request.get(request_id, []),
                key=lambda record: (
                    record["order"]["channel"],
                    record["order"]["coverage_start_ms"],
                    record["order"]["shard_seq"],
                    record["item_id"],
                ),
            )
            rfused = sorted(
                fusion_by_request.get(request_id, []),
                key=lambda record: (-record["fusion_fixed"], record["item_id"]),
            )
            pending = sorted(
                pending_by_request.get(request_id, []),
                key=lambda item: item["item_id"],
            )
            unique_items = sorted({record["item_id"] for record in rmerge})
            request_reports.append(
                {
                    "request_id": request_id,
                    "candidate_count": len(unique_items),
                    "unique_item_ids": unique_items,
                    "merge_record_count": len(rmerge),
                    "fused_candidate_count": len(rfused),
                    "pending_candidate_count": len(pending),
                    "channel_contribution_breakdown": contributions.get(request_id, []),
                    "merge_records": rmerge,
                    "fused_candidates": rfused,
                    "rerank": rerank_by_request.get(request_id),
                    "pending": pending,
                }
            )
        total_unique = sum(item["candidate_count"] for item in request_reports)
        report = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "mode": self.config.mode,
                "global_version": (previous["manifest"]["global_version"] + 1) if previous else 1,
                "step": self.config.step,
                "applied_steps": materialized.applied_steps,
                "config_hash": cfg.config_hash,
                "plan_hash": materialized.effective_plan_hash,
                "base_source_hash": initial_sources_hash(self.samples_dir),
                "serial_execution": self.config.serial,
                "thread_pool_workers": int(cfg.raw["thread_pool_workers"]),
            },
            "summary": {
                "request_count": len(request_reports),
                "unique_candidate_count": total_unique,
                "merge_record_count": len(merge_records),
                "fused_candidate_count": len(fused_records),
                "pending_count": len(fusion_meta["pending"]),
                "failure_count": len(failures),
                "shard_count": len(load_result.shards),
                "invalidated_cache_entries": cache_report["totals"]["invalidated"],
                "reused_cache_entries": cache_report["totals"]["reused"],
                "computed_cache_entries": cache_report["totals"]["computed"],
                "budget_consumed_this_round": fusion_meta["budget"]["consumed"],
                "budget_remaining_this_round": fusion_meta["budget"]["remaining"],
            },
            "requests": request_reports,
            "complete_shards": load_result.complete_shard_set,
            "timeline": self._timeline(load_result, cfg),
            "shard_changes": shard_changes,
            "affected_shards": sorted(affected_shards),
            "failure_list": failures,
            "failures": failures,
            "merge_records": merge_records,
            "normalization": norm_channels,
            "fusion_records": fused_records,
            "pending": fusion_meta["pending"],
            "discount_records": [
                record
                for result in rerank_results
                for record in result["discount_records"]
            ],
            "boundary_samples": boundary_samples,
            "cache": cache_report,
            "budget": {
                "budget_per_run": int(
                    cfg.raw["recompute_budget_per_run"]
                    if self.config.budget_override is None
                    else self.config.budget_override
                ),
                "rounds": budget_rounds,
                "current_round": fusion_meta["budget"],
                "pending_count": len(fusion_meta["pending"]),
                "exhausted": (
                    fusion_meta["budget"]["remaining"] == 0
                    and bool(fusion_meta["pending"])
                ),
                "audit_order": "pending_first_then_request_id_item_id",
            },
            "injection_hits": injection_hits,
            "snapshot_stale": None,
        }
        report["discount_records"].sort(
            key=lambda record: (
                record["request_id"],
                record["window_index"],
                record["candidate_item_id"],
                record["reference_item_id"],
                record["before_fixed"],
            )
        )
        return report

    def _timeline(self, load_result: LoadResult, cfg: RuntimeConfig) -> dict[str, Any]:
        by_channel: dict[str, list[Any]] = defaultdict(list)
        for shard in load_result.shards:
            by_channel[shard.channel].append(shard)
        channels = {}
        for channel, shards in sorted(by_channel.items()):
            ordered = sorted(
                shards,
                key=lambda shard: (
                    shard.coverage_start_ms,
                    shard.coverage_end_ms,
                    shard.shard_seq,
                ),
            )
            channels[channel] = {
                "start_ms": ordered[0].coverage_start_ms,
                "end_ms": ordered[-1].coverage_end_ms,
                "shards": [
                    {
                        "shard_id": shard.shard_id,
                        "start_ms": shard.coverage_start_ms,
                        "end_ms": shard.coverage_end_ms,
                    }
                    for shard in ordered
                ],
            }
        return {
            "alignment_ms": cfg.alignment_ms,
            "channels": channels,
        }

    def _canonical(self, report: dict[str, Any]) -> dict[str, Any]:
        rerank_canonical = []
        for result in report["requests"]:
            rerank = result.get("rerank")
            if rerank is not None:
                rerank_canonical.append(
                    {
                        key: value
                        for key, value in rerank.items()
                        if key not in {"pending"}
                    }
                )
            elif result.get("pending"):
                rerank_canonical.append(
                    {
                        "request_id": result["request_id"],
                        "before": [],
                        "after": [],
                        "windows": [],
                        "discount_records": [],
                        "boundary_records": [],
                        "order_changed": False,
                        "before_items": [],
                        "after_items": [],
                        "complete": False,
                    }
                )
        rerank_canonical.sort(key=lambda item: item["request_id"])
        return {
            "schema_version": SCHEMA_VERSION,
            "config_hash": report["run"]["config_hash"],
            "complete_shards": report["complete_shards"],
            "timeline": report["timeline"],
            "merge_records": [
                record
                for request in report["requests"]
                for record in request["merge_records"]
            ],
            "normalization": report["normalization"],
            "fusion_records": report["fusion_records"],
            "rerank": rerank_canonical,
            "pending": report["pending"],
            "failure_list": report["failure_list"],
            "boundary_samples": report["boundary_samples"],
            "discount_records": report["discount_records"],
            "channel_contribution_breakdown": {
                request["request_id"]: request["channel_contribution_breakdown"]
                for request in report["requests"]
            },
            "current_injection_hits": [
                hit for hit in report["injection_hits"] if not hit.get("historical")
            ],
        }

    def _next_stage_versions(
        self,
        previous: dict[str, Any] | None,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        old_versions = (
            previous["manifest"].get("stage_versions", {}) if previous else {}
        )
        stage_payloads = {
            "alignment": {
                "complete_shards": report["complete_shards"],
                "failure_list": report["failure_list"],
                "config_hash": report["run"]["config_hash"],
            },
            "merge": report["requests"] and [
                request["merge_records"] for request in report["requests"]
            ],
            "normalize": report["normalization"],
            "fusion": {
                "records": report["fusion_records"],
                "pending": report["pending"],
            },
            "rerank": [
                request.get("rerank") for request in report["requests"]
            ],
        }
        versions: dict[str, Any] = {}
        for stage in STAGE_NAMES:
            content_hash = stable_hash(stage_payloads[stage])
            old = old_versions.get(stage, {})
            if old.get("content_hash") == content_hash:
                version = int(old.get("version", 1))
                changed = False
            else:
                version = int(old.get("version", 0)) + 1
                changed = True
            versions[stage] = {
                "version": version,
                "content_hash": content_hash,
                "changed_this_run": changed,
            }
        return versions

    def _manifest(
        self,
        previous: dict[str, Any] | None,
        materialized: MaterializedInput,
        cfg: RuntimeConfig,
        load_result: LoadResult,
        stage_versions: dict[str, Any],
        budget_rounds: list[dict[str, Any]],
        mode: str,
    ) -> dict[str, Any]:
        global_version = (
            previous["manifest"]["global_version"] + 1 if previous else 1
        )
        previous_consumed = (
            int(previous["manifest"].get("consumed_recompute_budget_total", 0))
            if previous
            else 0
        )
        consumed = (
            int(budget_rounds[-1]["consumed"]) if budget_rounds else 0
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "global_version": global_version,
            "mode": mode,
            "base_source_hash": initial_sources_hash(self.samples_dir),
            "plan_hash": materialized.effective_plan_hash,
            "config_hash": cfg.config_hash,
            "config": cfg.raw,
            "applied_steps": materialized.applied_steps,
            "completed_shards": load_result.complete_shard_set,
            "stage_versions": stage_versions,
            "consumed_recompute_budget_total": previous_consumed + consumed,
            "snapshot_commit_interval_shards": int(
                cfg.raw["snapshot_commit_interval_shards"]
            ),
        }

    def _publish_snapshot(
        self,
        manifest: dict[str, Any],
        state: dict[str, Any],
        caches: dict[str, dict[str, Any]],
        report: dict[str, Any],
        canonical: dict[str, Any],
    ) -> None:
        for child in self.tmp_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        staging = Path(
            tempfile.mkdtemp(prefix="snapshot-", dir=self.tmp_root)
        )
        stages = {
            "alignment.json": {
                "complete_shards": report["complete_shards"],
                "failure_list": report["failure_list"],
            },
            "merge.json": {
                "records": [
                    record
                    for request in report["requests"]
                    for record in request["merge_records"]
                ]
            },
            "fusion.json": {
                "normalization": report["normalization"],
                "records": report["fusion_records"],
                "pending": report["pending"],
            },
            "rerank.json": {
                "requests": [
                    request.get("rerank") for request in report["requests"]
                ]
            },
        }
        for filename, payload in stages.items():
            atomic_write_json(staging / filename, payload)
        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_json(staging / "state.json", state)
        atomic_write_json(staging / "cache.json", caches)
        atomic_write_json(staging / "canonical.json", canonical)
        atomic_write_json(staging / "report.json", report)
        atomic_publish_directory(staging, self.live_dir)

    def _write_error(self, error: WorkbenchError) -> None:
        atomic_write_json(
            self.output_dir / "error_report.json",
            {
                "ok": False,
                **error.as_dict(),
                "schema_version": SCHEMA_VERSION,
            },
        )
