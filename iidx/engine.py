from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .cache import CacheStore
from .config import AppConfig
from .indexer import (
    build_blocks,
    build_invalidation_ledger,
    canonical_digest,
    compress_blocks,
    run_queries,
)
from .loader import LoadResult, load_inputs
from .scheduler import run_scheduler, serialize_schedule_state
from .snapshot import CheckpointLoad, SnapshotStore, utc_now
from .util import canonical_json, sort_failures


@dataclass
class RunResult:
    report: dict[str, Any]
    exit_code: int
    interrupted: bool
    report_path: Path
    snapshot_path: Path | None = None


class Engine:
    def __init__(
        self,
        *,
        data_dir: Path,
        state_dir: Path,
        config: AppConfig,
        config_failures: list[dict[str, Any]] | None = None,
        stop_event: threading.Event | None = None,
        stop_requested: Callable[[], bool] | None = None,
        expected_version: int | None = None,
        run_id: str | None = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.config = config
        self.config_failures = list(config_failures or [])
        self.stop_event = stop_event or threading.Event()
        self.stop_requested = stop_requested
        self.expected_version = expected_version
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:16]}"
        self.snapshot_store = SnapshotStore(self.state_dir)
        self.cache = CacheStore(self.snapshot_store.cache_dir)

    def _stop_check(self) -> bool:
        if self.stop_event.is_set():
            return True
        return bool(self.stop_requested and self.stop_requested())

    def _config_changed_fields(self, previous_report: dict[str, Any] | None) -> list[str]:
        if not previous_report:
            return []
        previous = previous_report.get("config")
        if not isinstance(previous, dict):
            return []
        current = self.config.to_dict()
        return [key for key in sorted(set(previous) | set(current)) if previous.get(key) != current.get(key)]

    def _affected_shards(
        self,
        blocks: list[dict[str, Any]],
        reuse: dict[str, Any],
        previous_report: dict[str, Any] | None,
        changed_fields: list[str],
    ) -> list[str]:
        affected = {
            item.get("shard")
            for item in reuse.get("recomputed_blocks", []) + reuse.get("invalidated_blocks", [])
            if item.get("shard")
        }
        if any(
            field
            in {
                "chunk_size",
                "batch_max_docs",
                "global_token_budget",
                "per_shard_token_quota",
                "compression_tier",
                "compression_switch_points",
                "low_frequency_threshold",
            }
            for field in changed_fields
        ):
            affected.update(block["shard"] for block in blocks)
            if previous_report:
                affected.update(
                    item.get("shard")
                    for item in previous_report.get("blocks", [])
                    if item.get("shard")
                )
        return sorted(item for item in affected if item)

    def _checkpoint_stage_loading(self, load_result: LoadResult) -> dict[str, Any]:
        return {
            "source_fingerprint": load_result.source_fingerprint,
            "config_hash": self.config.config_hash,
            "input_fingerprints": load_result.input_fingerprints,
            "document_rows": load_result.stats["document_rows"],
            "query_rows": load_result.stats["query_rows"],
            "accepted_documents": load_result.stats["accepted_documents"],
            "accepted_queries": load_result.stats["accepted_queries"],
            "accepted_document_ids": sorted(load_result.documents),
            "accepted_query_ids": sorted(load_result.queries),
            "failures": load_result.failures,
        }

    def _stage_loading(
        self,
        staging: Path,
        load_result: LoadResult,
    ) -> dict[str, Any]:
        summary = self._checkpoint_stage_loading(load_result)
        files = {
            "loading-summary.json": json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            "input-fingerprints.json": json.dumps(
                load_result.input_fingerprints, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
        }
        return self.snapshot_store.write_stage(
            staging,
            "stage-00-loading",
            files,
            {
                "source_fingerprint": load_result.source_fingerprint,
                "config_hash": self.config.config_hash,
                "accepted_documents": load_result.accepted_document_count,
                "accepted_queries": load_result.accepted_query_count,
            },
        )

    def _stage_build(
        self,
        staging: Path,
        blocks: list[dict[str, Any]],
        schedule_completed_shards: list[str],
        schedule_partial_shards: list[str],
        schedule_pending_shards: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        lines = [
            canonical_json(
                {
                    "block_no": block["block_no"],
                    "shard": block["shard"],
                    "local_block_no": block["local_block_no"],
                    "terms": block["terms"],
                    "doc_count": block["doc_count"],
                    "position_count": block["position_count"],
                    "fingerprint": block["fingerprint"],
                    "content_cache_key": block["content_cache_key"],
                }
            )
            for block in blocks
        ]
        index = {
            "blocks": [
                {
                    "block_no": block["block_no"],
                    "shard": block["shard"],
                    "local_block_no": block["local_block_no"],
                    "fingerprint": block["fingerprint"],
                    "doc_count": block["doc_count"],
                    "position_count": block["position_count"],
            "content_cache_key": block["content_cache_key"],
                }
                for block in blocks
            ]
        }
        files = {
            "blocks.jsonl": "\n".join(lines) + ("\n" if lines else ""),
            "block-index.json": json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        }
        manifest = self.snapshot_store.write_stage(
            staging,
            "stage-01-build",
            files,
            {
                "block_count": len(blocks),
                "completed_shards": sorted(schedule_completed_shards),
                "partial_shards": sorted(schedule_partial_shards),
                "pending_shards": sorted(schedule_pending_shards),
                "index_config_hash": self.config.index_config_hash,
            },
        )
        return manifest, index

    def _stage_query(
        self,
        staging: Path,
        queries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        canonical_queries = [
            {key: value for key, value in query.items() if key != "reused"}
            for query in queries
        ]
        payload = {
            "queries": canonical_queries,
            "query_config_hash": self.config.query_config_hash,
        }
        files = {
            "query-results.json": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        }
        return self.snapshot_store.write_stage(
            staging,
            "stage-03-query",
            files,
            {
                "query_count": len(queries),
                "query_config_hash": self.config.query_config_hash,
            },
        )

    def run(self) -> RunResult:
        started = time.monotonic()
        started_at = utc_now()
        staging: Path | None = None
        global_version = self.snapshot_store.next_version()
        inspection = self.snapshot_store.inspect_current()
        previous_report = inspection.report
        changed_fields = self._config_changed_fields(previous_report)
        snapshot_events: list[dict[str, Any]] = []
        snapshot_stale = bool(inspection.stale_codes)

        if inspection.stale_codes:
            snapshot_events.append(
                {
                    "code": "E_SNAPSHOT_STALE",
                    "severity": "warning",
                    "message": "current snapshot pointer or manifest is inconsistent; stages will be rebuilt",
                    "details": inspection.details,
                }
            )
        if previous_report and previous_report.get("run", {}).get("config_hash") != self.config.config_hash:
            snapshot_stale = True
            snapshot_events.append(
                {
                    "code": "E_SNAPSHOT_STALE",
                    "severity": "warning",
                    "message": "configuration hash differs from the current snapshot; affected stages are rebuilt",
                    "previous_config_hash": previous_report.get("run", {}).get("config_hash"),
                    "current_config_hash": self.config.config_hash,
                }
            )
        if self.expected_version is not None and (
            inspection.info is None
            or int(inspection.info.get("global_version", -1)) != self.expected_version
        ):
            snapshot_stale = True
            snapshot_events.append(
                {
                    "code": "E_SNAPSHOT_STALE",
                    "severity": "warning",
                    "message": "expected snapshot version does not match current version",
                    "expected_version": self.expected_version,
                    "actual_version": inspection.info.get("global_version") if inspection.info else None,
                }
            )

        load_result = load_inputs(self.data_dir, self.config, self.cache)
        checkpoint = self.snapshot_store.load_checkpoint(
            self.config.config_hash, load_result.source_fingerprint, global_version
        )
        if checkpoint.available and snapshot_stale:
            checkpoint = CheckpointLoad(False, None, checkpoint.manifest, "stale_snapshot_forced_rebuild")
        checkpoint_event: dict[str, Any] = {
            "available": checkpoint.available,
            "reason": checkpoint.reason,
        }
        if checkpoint.reason == "config_hash_mismatch":
            snapshot_events.append(
                {
                    "code": "E_SNAPSHOT_STALE",
                    "severity": "warning",
                    "message": "checkpoint configuration hash is stale; run restarts from the loading stage",
                    "checkpoint_reason": checkpoint.reason,
                }
            )
        elif checkpoint.reason in {
            "source_fingerprint_mismatch",
            "checkpoint_incomplete",
            "checkpoint_version_mismatch",
        }:
            snapshot_events.append(
                {
                    "code": "E_SNAPSHOT_STALE",
                    "severity": "warning",
                    "message": "checkpoint signature or contents are stale; affected stages are rebuilt",
                    "checkpoint_reason": checkpoint.reason,
                }
            )
        resume_state = checkpoint.state if checkpoint.available else None
        if resume_state is not None:
            checkpoint_event["resumed"] = True

        try:
            staging = self.snapshot_store.allocate_staging(self.run_id)
            loading_manifest = self._stage_loading(staging, load_result)
            checkpoint_commits = 0

            def checkpoint_callback(state: dict[str, Any], reason: str) -> None:
                nonlocal checkpoint_commits
                checkpoint_commits += 1
                self.snapshot_store.commit_checkpoint(
                    run_id=self.run_id,
                    global_version=global_version,
                    config_hash=self.config.config_hash,
                    source_fingerprint=load_result.source_fingerprint,
                    state=serialize_schedule_state(state),
                    reason=reason,
                    stage_loading=self._checkpoint_stage_loading(load_result),
                )

            schedule = run_scheduler(
                load_result.documents_by_shard,
                self.config,
                resume_state=resume_state,
                stop_check=self._stop_check,
                checkpoint_callback=checkpoint_callback,
            )
            if schedule.interrupted:
                elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
                partial_report = self._build_report(
                    status="interrupted",
                    started_at=started_at,
                    elapsed_ms=elapsed_ms,
                    global_version=global_version,
                    load_result=load_result,
                    schedule=schedule,
                    blocks=[],
                    queries=[],
                    decisions=[],
                    reuse={},
                    compression_failures=[],
                    build_metrics={},
                    compression_metrics={},
                    query_metrics={},
                    snapshot_events=snapshot_events,
                    checkpoint_event=checkpoint_event,
                    changed_fields=changed_fields,
                    stage_versions={},
                    affected_shards=[],
                )
                report_path = self.snapshot_store.write_latest_report(partial_report)
                shutil.rmtree(staging, ignore_errors=True)
                staging = None
                return RunResult(partial_report, 0, True, report_path, None)

            accepted = set(schedule.accepted_doc_ids)
            blocks, build_metrics = build_blocks(
                load_result.documents_by_shard, accepted, self.config, self.cache
            )
            build_manifest, _ = self._stage_build(
                staging,
                blocks,
                schedule.completed_shards,
                schedule.partial_shards,
                schedule.pending_shards,
            )
            block_records, compression_metrics, retrieval_blocks, compression_failures = compress_blocks(
                blocks, self.config, self.cache
            )
            decisions = [
                {
                    "block_no": item["block_no"],
                    "shard": item["shard"],
                    "selected_tier": item["compression"]["selected_tier"],
                    "reason": item["compression"]["selection_reason"],
                    "matched_switch_point": item["compression"]["matched_switch_point"],
                    "tier1_bytes": item["compression"]["tier1"]["encoded_bytes"],
                    "tier2_bytes": item["compression"]["tier2"]["encoded_bytes"],
                    "byte_length_delta": item["compression"]["tier2"]["encoded_bytes"]
                    - item["compression"]["tier1"]["encoded_bytes"],
                    "filtered_count": item["compression"]["filtered_count"],
                }
                for item in block_records
            ]
            # Re-run compression stage with the real binaries included in the initial stage commit.
            stage_files: dict[str, bytes | str] = {}
            binary_index: list[dict[str, Any]] = []
            encoded_by_block = self._collect_encoded_artifacts(block_records)
            for block in block_records:
                block_no = block["block_no"]
                for tier in (1, 2):
                    artifact = encoded_by_block[(block_no, tier)]
                    stage_files[f"blocks/b{block_no:08d}.tier{tier}.bin"] = artifact
                binary_index.append(
                    {
                        "block_no": block_no,
                        "tier1_bytes": block["compression"]["tier1"]["encoded_bytes"],
                        "tier2_bytes": block["compression"]["tier2"]["encoded_bytes"],
                        "selected_tier": block["compression"]["selected_tier"],
                        "selected_tier_bytes": block["compression"]["selected_encoded_bytes"],
                        "decoded_checksum": block["compression"]["decoded_checksum"],
                        "filtered_count": block["compression"]["filtered_count"],
                    }
                )
            stage_files["compression-index.json"] = (
                json.dumps(
                    {
                        "blocks": binary_index,
                        "decisions": decisions,
                        "compression_config_hash": self.config.compression_config_hash,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            shutil.rmtree(staging / "stage-02-compress", ignore_errors=True)
            compression_manifest = self.snapshot_store.write_stage(
                staging,
                "stage-02-compress",
                stage_files,
                {
                    "block_count": len(block_records),
                    "compression_config_hash": self.config.compression_config_hash,
                    "tier1_bytes": sum(item["tier1_bytes"] for item in binary_index),
                    "tier2_bytes": sum(item["tier2_bytes"] for item in binary_index),
                    "binaries_complete": True,
                },
            )

            query_results, query_metrics = run_queries(
                load_result.queries,
                retrieval_blocks,
                load_result.documents,
                self.config,
                self.cache,
            )
            query_manifest = self._stage_query(staging, query_results)
            reuse = build_invalidation_ledger(previous_report, block_records, query_results)
            affected_shards = self._affected_shards(
                block_records, reuse, previous_report, changed_fields
            )
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
            all_failures = sort_failures(
                self.config_failures
                + load_result.failures
                + schedule.failures
                + compression_failures
            )
            report = self._build_report(
                status="complete",
                started_at=started_at,
                elapsed_ms=elapsed_ms,
                global_version=global_version,
                load_result=load_result,
                schedule=schedule,
                blocks=block_records,
                queries=query_results,
                decisions=decisions,
                reuse=reuse,
                compression_failures=compression_failures,
                build_metrics=build_metrics,
                compression_metrics=compression_metrics,
                query_metrics=query_metrics,
                snapshot_events=snapshot_events,
                checkpoint_event=checkpoint_event,
                changed_fields=changed_fields,
                stage_versions={
                    "loading": loading_manifest["artifact_version"],
                    "build": build_manifest["artifact_version"],
                    "compress": compression_manifest["artifact_version"],
                    "query": query_manifest["artifact_version"],
                },
                affected_shards=affected_shards,
                failures_override=all_failures,
            )
            previous_stage_versions = (
                previous_report.get("stages", {}) if isinstance(previous_report, dict) else {}
            )
            report["diagnostics"]["stage_reuse"] = {
                stage: bool(
                    previous_stage_versions
                    and previous_stage_versions.get(stage) == version
                )
                for stage, version in report["stages"].items()
            }
            report["canonical_digest"] = canonical_digest(report)
            report["run"]["finished_at"] = utc_now()
            stage_manifests = {
                "loading": loading_manifest,
                "build": build_manifest,
                "compress": compression_manifest,
                "query": query_manifest,
            }
            snapshot_path = self.snapshot_store.commit_snapshot(
                staging,
                global_version=global_version,
                config_hash=self.config.config_hash,
                source_fingerprint=load_result.source_fingerprint,
                run_id=self.run_id,
                completed_shards=schedule.completed_shards,
                consumed_token_budget=self.config.global_token_budget - schedule.global_remaining,
                stage_manifests=stage_manifests,
                report=report,
                extra_manifest={"affected_shards": affected_shards},
            )
            staging = None
            report_path = self.snapshot_store.write_latest_report(report)
            self.snapshot_store.clear_checkpoint(
                run_id=self.run_id,
                config_hash=self.config.config_hash,
                source_fingerprint=load_result.source_fingerprint,
            )
            return RunResult(report, 0, False, report_path, snapshot_path)
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _collect_encoded_artifacts(self, block_records: list[dict[str, Any]]) -> dict[tuple[int, int], bytes]:
        output: dict[tuple[int, int], Any] = {}
        for block in block_records:
            block_no = block["block_no"]
            for tier in (1, 2):
                key = block["compression"][f"tier{tier}"]["encoding_cache_key"]
                cached = self.cache.lookup_encoding(key)
                if cached is None:
                    raise RuntimeError(f"compressed artifact disappeared for block {block_no} tier {tier}")
                metadata, data = cached
                output[(block_no, tier)] = data
        return output

    def _build_report(
        self,
        *,
        status: str,
        started_at: str,
        elapsed_ms: float,
        global_version: int,
        load_result: LoadResult,
        schedule: Any,
        blocks: list[dict[str, Any]],
        queries: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        reuse: dict[str, Any],
        compression_failures: list[dict[str, Any]],
        build_metrics: dict[str, Any],
        compression_metrics: dict[str, Any],
        query_metrics: dict[str, Any],
        snapshot_events: list[dict[str, Any]],
        checkpoint_event: dict[str, Any],
        changed_fields: list[str],
        stage_versions: dict[str, str],
        affected_shards: list[str],
        failures_override: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        failures = failures_override or sort_failures(
            self.config_failures + load_result.failures + schedule.failures + compression_failures
        )
        selected_tiers: dict[str, int] = {}
        for block in blocks:
            key = str(block["compression"]["selected_tier"])
            selected_tiers[key] = selected_tiers.get(key, 0) + 1
        consumed = self.config.global_token_budget - schedule.global_remaining
        report = {
            "schema_version": 1,
            "app": {"name": "iidx", "version": __version__},
            "run": {
                "run_id": self.run_id,
                "status": status,
                "global_version": global_version,
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_ms": elapsed_ms,
                "config_hash": self.config.config_hash,
                "source_fingerprint": load_result.source_fingerprint,
                "state_dir": str(self.state_dir),
                "data_dir": str(self.data_dir),
            },
            "config": self.config.to_dict(),
            "config_fingerprints": {
                "config": self.config.config_hash,
                "index": self.config.index_config_hash,
                "compression": self.config.compression_config_hash,
                "query": self.config.query_config_hash,
            },
            "summary": {
                "accepted_documents": load_result.accepted_document_count,
                "accepted_queries": load_result.accepted_query_count,
                "completed_shards": schedule.completed_shards,
                "partial_shards": schedule.partial_shards,
                "pending_shards": schedule.pending_shards,
                "quota_exhausted_shards": schedule.quota_exhausted_shards,
                "block_count": len(blocks),
                "query_count": len(queries),
                "compression_tier": self.config.compression_tier,
                "selected_tier_counts": selected_tiers,
                "affected_shards": affected_shards,
                "consumed_token_budget": consumed,
                "remaining_token_budget": schedule.global_remaining,
                "total_postings": sum(
                    entry["doc_freq"] for block in blocks for entry in block["terms"]
                ),
                "total_positions": sum(block["position_count"] for block in blocks),
                "pending_queue": self._pending_queue(load_result, schedule, schedule.pending_shards, "E_BUDGET_PENDING"),
                "quota_exhausted_queue": self._pending_queue(
                    load_result,
                    schedule,
                    schedule.quota_exhausted_shards,
                    "E_SHARD_QUOTA_EXHAUSTED",
                ),
            },
            "input": {
                "paths": {
                    "documents": str(load_result.paths.documents),
                    "queries": str(load_result.paths.queries),
                    "vocabulary": str(load_result.paths.vocabulary),
                    "query_versions": str(load_result.paths.query_versions),
                },
                "fingerprints": load_result.input_fingerprints,
                "vocabulary_size": len(load_result.vocabulary),
                "declared_query_versions": list(load_result.declared_query_versions),
                "load_stats": load_result.stats,
            },
            "blocks": blocks,
            "queries": queries,
            "compression_decisions": decisions,
            "budget_ledger": schedule.ledger,
            "pending_queue": self._pending_queue(
                load_result, schedule, schedule.pending_shards, "E_BUDGET_PENDING"
            ),
            "reuse": reuse,
            "failures": failures,
            "stages": dict(sorted(stage_versions.items())),
            "diagnostics": {
                "snapshot_events": snapshot_events,
                "checkpoint": checkpoint_event,
                "changed_config_fields": changed_fields,
                "concurrency": {
                    "build": build_metrics,
                    "compression": compression_metrics,
                    "query": query_metrics,
                },
                "cache": self.cache.snapshot_metrics(),
                "validation": {
                    "document_rows": load_result.stats["document_rows"],
                    "query_rows": load_result.stats["query_rows"],
                    "accepted_documents": load_result.accepted_document_count,
                    "accepted_queries": load_result.accepted_query_count,
                    "failure_count": len(failures),
                },
                "reproducibility": {
                    "normalization": "NFKC + Unicode casefold + whitespace collapse",
                    "position_base": 0,
                    "block_order": "shard lexicographic, then local block ordinal",
                    "query_order": "query ID lexicographic",
                    "failure_order": "stage, stable key, code, message",
                },
            },
            "canonical_digest": "",
        }
        report["canonical_digest"] = canonical_digest(report)
        return report

    def _pending_queue(
        self,
        load_result: LoadResult,
        schedule: Any,
        shards: list[str],
        reason: str,
    ) -> list[dict[str, Any]]:
        accepted = set(schedule.accepted_doc_ids)
        queue: list[dict[str, Any]] = []
        for shard in sorted(set(shards)):
            docs = [
                doc
                for doc in load_result.documents_by_shard.get(shard, ())
                if doc.doc_id not in accepted
            ]
            queue.append(
                {
                    "shard": shard,
                    "reason": reason,
                    "pending_document_ids": [doc.doc_id for doc in docs],
                    "pending_documents": len(docs),
                    "pending_tokens": sum(doc.token_count for doc in docs),
                }
            )
        return queue
