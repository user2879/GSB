"""End-to-end staged sampling pipeline."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import (
    CacheStore,
    bucket_content_fingerprint,
    bucket_history_cache_key,
    bucket_invalidation_reason,
    evidence_cache_key,
    final_sample_cache_key,
)
from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    fingerprint,
    sha256_bytes,
    sha256_text,
)
from .diagnostics import compute_diagnostics
from .evidence import build_evidence_slots, serialize_evidence
from .models import AcceptedRecord, Failure, ValidationResult
from .reservoir import (
    bucket_fingerprint,
    bucket_params,
    bucket_random_source_digest,
    build_bucket_history,
    record_fingerprint,
)
from .snapshot import SnapshotManager, StageContext
from .truncation import apply_global_budget
from .validation import (
    bucket_capacity,
    bucket_priority,
    resolve_sampling,
    validate_input,
)


@dataclass(frozen=True)
class RuntimeConfig:
    state_root: Path
    crash_after: str | None = None


@dataclass
class RunResult:
    manifest: dict[str, Any]
    logical_result: dict[str, Any]
    ledger: dict[str, Any]
    version_dir: Path


class Pipeline:
    def __init__(
        self,
        config: dict[str, Any],
        batches: list[dict[str, Any]],
        injections: list[dict[str, Any]] | None,
        runtime: RuntimeConfig,
    ):
        self.config = config
        self.batches = batches
        self.injections = injections or []
        self.runtime = runtime
        self.snapshots = SnapshotManager(runtime.state_root)
        self.cache = CacheStore(runtime.state_root / "cache")
        self._previous_index = self.cache.load_index()
        self.config_hash = fingerprint(self.config)
        self.input_hash = fingerprint(
            {
                "config": self.config,
                "batches": self.batches,
                "injections": self.injections,
            }
        )
        self.random_source_digest = fingerprint(
            {
                "seed": self.config.get("seed"),
                "profile": self.config.get("active_profile"),
                "reservoir": "algorithm-r-int256-slot-cycle-v1",
                "quantization_enum": self.config.get("allowed_quantization"),
            }
        )

    def run(self) -> RunResult:
        context = self.snapshots.begin(
            config_hash=self.config_hash,
            input_hash=self.input_hash,
            random_source_digest=self.random_source_digest,
        )

        validation_artifact = self._load_validation(context)
        if validation_artifact is None:
            validation = validate_input(
                self.config, self.batches, self.injections
            )
            validation_artifact = _validation_artifact(validation)
            self._write_stage_json(context, "validation.json", validation_artifact)
        else:
            validation = _validation_from_artifact(validation_artifact)
        validation_hash = fingerprint(validation_artifact)
        context.manifest["stage_versions"]["validation"] = validation_hash
        context.manifest["completed_batch_cursor"] = (
            validation.completed_batch_cursor
        )
        self.snapshots.save(context)
        self._crash_if("validate")

        sampling = resolve_sampling(self.config)
        global_budget = sampling["global_budget"]
        if not isinstance(global_budget, int) or global_budget <= 0:
            global_budget = 1
        profile = sampling["profile"]
        quantization = sampling["quantization"]
        if not isinstance(quantization, dict):
            quantization = {"decimals": 3, "tier": "fine"}

        records_by_bucket = _group_records(validation.accepted_records)
        bucket_entries, tasks = self._run_buckets(
            context, records_by_bucket, sampling, global_budget
        )
        buckets = {
            entry["bucket"]["bucket_id"]: entry["bucket"]
            for entry in bucket_entries
        }
        buckets_artifact = {
            "schema": "lds-bucket-histories-v1",
            "buckets": buckets,
        }
        buckets_hash = fingerprint(buckets_artifact)
        if not (context.stage_dir / "buckets.json").exists():
            self._write_stage_json(context, "buckets.json", buckets_artifact)
        context.manifest["stage_versions"]["buckets"] = buckets_hash
        self.snapshots.save(context)

        truncation, truncation_meta = self._run_truncation(
            context,
            buckets,
            bucket_entries,
            global_budget,
            quantization,
        )
        truncation_hash = fingerprint(truncation)
        context.manifest["stage_versions"]["truncation"] = truncation_hash
        context.manifest["consumed_budget"] = truncation["consumed_budget"]
        self.snapshots.save(context)
        self._crash_if("truncate")

        evidence_slots = build_evidence_slots(buckets)
        serialization_meta = self._run_serialization(
            context,
            evidence_slots,
            truncation_meta["cache_key"],
            truncation,
            buckets,
            quantization,
        )
        context.manifest["stage_versions"]["serialization"] = serialization_meta[
            "stage_hash"
        ]
        self._crash_if("serialize")

        final_samples = truncation["final_samples"]
        diagnostics = serialization_meta.pop("diagnostics")
        ledger = self._build_ledger(
            bucket_entries=bucket_entries,
            truncation_meta=truncation_meta,
            serialization_meta=serialization_meta,
            final_samples=final_samples,
            tasks=tasks,
        )
        self._write_stage_json(context, "ledger.json", ledger)
        self._write_stage_json(context, "diagnostics.json", diagnostics)

        logical_result = self._build_logical_result(
            validation=validation,
            buckets=buckets,
            truncation=truncation,
            diagnostics=diagnostics,
            sampling=sampling,
            global_budget=global_budget,
            quantization=quantization,
        )
        self._write_stage_json(context, "logical_result.json", logical_result)
        self._write_stage_bytes(
            context,
            "logical_result.canonical.json",
            canonical_json(logical_result).encode("utf-8") + b"\n",
        )

        context.manifest["pending_buckets"] = []
        context.manifest["consumed_budget"] = truncation["consumed_budget"]
        manifest = self.snapshots.commit(context)
        version_dir = self.runtime.state_root / manifest["version_dir"]
        atomic_write_json(self.runtime.state_root / "latest_result.json", logical_result)
        self._update_cache_index(bucket_entries, truncation_meta, serialization_meta)
        return RunResult(
            manifest=manifest,
            logical_result=logical_result,
            ledger=ledger,
            version_dir=version_dir,
        )

    def _load_validation(self, context: StageContext) -> dict[str, Any] | None:
        value = self.snapshots.read_stage_json(context, "validation.json")
        return value if isinstance(value, dict) else None

    def _run_buckets(
        self,
        context: StageContext,
        records_by_bucket: dict[str, list[AcceptedRecord]],
        sampling: dict[str, Any],
        global_budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tasks: list[dict[str, Any]] = []
        bucket_entries: list[dict[str, Any]] = []
        completed_ids = set(context.manifest.get("completed_buckets", []))

        for bucket_id, records in sorted(records_by_bucket.items()):
            capacity = bucket_capacity(self.config, bucket_id)
            priority = bucket_priority(self.config, bucket_id)
            params = bucket_params(
                self.config,
                bucket_id,
                capacity,
                priority,
                sampling["profile"],
                global_budget,
                sampling["quantization"],
            )
            params_fp = fingerprint(params)
            random_digest = bucket_random_source_digest(params)
            record_fps = {
                record.log_id: record_fingerprint(record) for record in records
            }
            content_fp = bucket_content_fingerprint(record_fps)
            cache_key = bucket_history_cache_key(
                bucket_id=bucket_id,
                content_fingerprint=content_fp,
                params_fingerprint=params_fp,
                random_source_digest=random_digest,
                profile=sampling["profile"],
                budget_tier=global_budget,
            )
            checkpoint_path = context.stage_dir / "buckets" / f"{cache_key}.json"
            checkpoint = self._load_json_file(checkpoint_path)
            if (
                bucket_id in completed_ids
                and isinstance(checkpoint, dict)
                and checkpoint.get("cache_key") == cache_key
            ):
                entry = checkpoint["entry"]
                entry["status"] = "stage_resume_hit"
                entry["reason"] = "resumed_from_staged_bucket_checkpoint"
                for record in entry.get("record_reasons", []):
                    record["status"] = "reuse_hit"
                    record["reason"] = "resumed_from_staged_bucket_checkpoint"
                bucket_entries.append(entry)
                continue

            sort_key = (
                bucket_id,
                records[0].order_key[0] if records else 0,
                bucket_fingerprint(params),
            )
            task = {
                "bucket_id": bucket_id,
                "cache_key": cache_key,
                "params_fingerprint": params_fp,
                "content_fingerprint": content_fp,
                "random_source_digest": random_digest,
                "profile": sampling["profile"],
                "budget_tier": global_budget,
                "record_fingerprints": record_fps,
                "records": records,
                "capacity": capacity,
                "priority": priority,
                "params": params,
                "checkpoint_path": checkpoint_path,
                "sort_key": list(sort_key),
            }
            tasks.append(task)

        context.manifest["pending_buckets"] = [task["bucket_id"] for task in tasks]
        context.manifest["completed_buckets"] = sorted(completed_ids)
        self.snapshots.save(context)
        if not tasks:
            return bucket_entries, []

        max_workers = int(self.config.get("max_workers", 4))
        workers = max(1, min(max_workers, len(tasks)))
        completed_entries: list[dict[str, Any]] = []
        completed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._compute_bucket_task, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                entry = future.result()
                completed_entries.append(entry)
                completed_count += 1
                self._write_stage_json(
                    context,
                    str(Path("buckets") / f"{task['cache_key']}.json"),
                    {
                        "schema": "lds-bucket-checkpoint-v1",
                        "cache_key": task["cache_key"],
                        "entry": entry,
                    },
                )
                context.manifest["completed_buckets"] = sorted(
                    {
                        *context.manifest.get("completed_buckets", []),
                        task["bucket_id"],
                    }
                )
                context.manifest["pending_buckets"] = sorted(
                    item
                    for item in context.manifest.get("pending_buckets", [])
                    if item != task["bucket_id"]
                )
                self.snapshots.save(context)
                if self.runtime.crash_after == f"bucket:{completed_count}":
                    os._exit(137)

        all_entries = [*bucket_entries, *completed_entries]
        all_entries.sort(key=lambda entry: entry["bucket"]["bucket_id"])
        return all_entries, tasks

    def _compute_bucket_task(self, task: dict[str, Any]) -> dict[str, Any]:
        cached = self.cache.load_bucket(task["cache_key"])
        if cached is not None:
            return {
                "bucket": cached,
                "cache_key": task["cache_key"],
                "params_fingerprint": task["params_fingerprint"],
                "content_fingerprint": task["content_fingerprint"],
                "random_source_digest": task["random_source_digest"],
                "profile": task["profile"],
                "budget_tier": task["budget_tier"],
                "status": "cache_hit",
                "reason": "bucket_content_and_params_match",
                "record_reasons": [
                    {
                        "log_id": log_id,
                        "status": "reuse_hit",
                        "reason": "bucket_content_and_params_match",
                    }
                    for log_id in sorted(task["record_fingerprints"])
                ],
                "sort_key": task["sort_key"],
            }
        previous = self._previous_index.get("buckets", {}).get(task["bucket_id"])
        reason, record_reasons = bucket_invalidation_reason(
            previous,
            current_params_fingerprint=task["params_fingerprint"],
            current_record_fingerprints=task["record_fingerprints"],
            current_profile=task["profile"],
            current_budget=task["budget_tier"],
            cache_entry_found=False,
        )
        bucket = build_bucket_history(
            bucket_id=task["bucket_id"],
            records=task["records"],
            capacity=task["capacity"],
            priority=task["priority"],
            config=self.config,
            profile=task["profile"],
            global_budget=task["budget_tier"],
            quantization=self.config["profiles"][task["profile"]]["quantization"],
        )
        self.cache.store_bucket(task["cache_key"], bucket)
        return {
            "bucket": bucket,
            "cache_key": task["cache_key"],
            "params_fingerprint": task["params_fingerprint"],
            "content_fingerprint": task["content_fingerprint"],
            "random_source_digest": task["random_source_digest"],
            "profile": task["profile"],
            "budget_tier": task["budget_tier"],
            "status": "cache_miss",
            "reason": reason,
            "record_reasons": record_reasons,
            "sort_key": task["sort_key"],
        }

    def _run_truncation(
        self,
        context: StageContext,
        buckets: dict[str, dict[str, Any]],
        bucket_entries: list[dict[str, Any]],
        global_budget: int,
        quantization: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        staged = self.snapshots.read_stage_json(context, "truncation.json")
        if isinstance(staged, dict) and isinstance(staged.get("truncation"), dict):
            truncation = staged["truncation"]
            meta = staged.get("meta", {})
            return truncation, meta
        bucket_keys = {
            entry["bucket"]["bucket_id"]: entry["cache_key"]
            for entry in bucket_entries
        }
        priorities = {
            entry["bucket"]["bucket_id"]: int(entry["bucket"]["priority"])
            for entry in bucket_entries
        }
        final_key = final_sample_cache_key(
            bucket_keys=bucket_keys,
            global_budget=global_budget,
            priorities=priorities,
            quantization=quantization,
        )
        cached = self.cache.load_final(final_key)
        if cached is not None:
            truncation = cached
            meta = {
                "cache_key": final_key,
                "status": "cache_hit",
                "reason": "final_sample_key_match",
            }
        else:
            truncation = apply_global_budget(buckets, global_budget)
            self.cache.store_final(final_key, truncation)
            bucket_reasons = {
                entry["reason"] for entry in bucket_entries if entry["status"] != "cache_hit"
            }
            meta = {
                "cache_key": final_key,
                "status": "cache_miss",
                "reason": (
                    "bucket_downstream_changed"
                    if bucket_reasons
                    else "budget_priority_or_quantization_changed"
                ),
            }
        self._write_stage_json(
            context,
            "truncation.json",
            {"truncation": truncation, "meta": meta},
        )
        return truncation, meta

    def _run_serialization(
        self,
        context: StageContext,
        evidence_slots: list[Any],
        final_key: str,
        truncation: dict[str, Any],
        buckets: dict[str, dict[str, Any]],
        quantization: dict[str, Any],
    ) -> dict[str, Any]:
        serializations: dict[str, dict[str, Any]] = {}
        filenames = {
            "p1": "evidence.p1.txt",
            "p2": "evidence.p2.json",
            "p3": "evidence.p3.bin",
        }
        for version in ("p1", "p2", "p3"):
            key = evidence_cache_key(final_key, version)
            cached = self.cache.load_evidence(version, key)
            if cached is not None:
                payload, sidecar = cached
                status = "cache_hit"
                reason = "evidence_key_match"
            else:
                payload, sidecar = serialize_evidence(evidence_slots, version)
                self.cache.store_evidence(version, key, payload, sidecar)
                status = "cache_miss"
                reason = "final_sample_or_version_changed"
            self._write_stage_bytes(context, filenames[version], payload)
            sidecar_name = f"{filenames[version]}.metadata.json"
            self._write_stage_json(context, sidecar_name, sidecar)
            serializations[version] = {
                "cache_key": key,
                "status": status,
                "reason": reason,
                "payload_sha256": sha256_bytes(payload),
                "byte_length": len(payload),
                "artifact": filenames[version],
                "sidecar": sidecar_name,
                "sidecar_sha256": fingerprint(sidecar),
            }
        diagnostics = compute_diagnostics(buckets, truncation, quantization)
        serialization_hash = fingerprint(
            {
                "schema": "lds-serialization-stage-v1",
                "artifacts": [
                    {
                        "version": version,
                        "payload_sha256": item["payload_sha256"],
                        "sidecar_sha256": item["sidecar_sha256"],
                    }
                    for version, item in sorted(serializations.items())
                ],
                "diagnostics_sha256": fingerprint(diagnostics),
            }
        )
        return {
            "serializations": serializations,
            "stage_hash": serialization_hash,
            "diagnostics": diagnostics,
        }

    def _build_ledger(
        self,
        *,
        bucket_entries: list[dict[str, Any]],
        truncation_meta: dict[str, Any],
        serialization_meta: dict[str, Any],
        final_samples: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        final_source = {
            sample["log_id"]: {
                "bucket_id": sample["bucket_id"],
                "slot_no": sample["slot_no"],
                "global_rank": sample["rank"],
            }
            for sample in final_samples
        }
        bucket_ledger: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for entry in sorted(
            bucket_entries, key=lambda item: item["bucket"]["bucket_id"]
        ):
            bucket = entry["bucket"]
            record_reasons = {
                item["log_id"]: item for item in entry.get("record_reasons", [])
            }
            bucket_ledger.append(
                {
                    "bucket_id": bucket["bucket_id"],
                    "cache_key": entry["cache_key"],
                    "content_fingerprint": entry["content_fingerprint"],
                    "params_fingerprint": entry["params_fingerprint"],
                    "random_source_digest": entry["random_source_digest"],
                    "profile": entry["profile"],
                    "budget_tier": entry["budget_tier"],
                    "status": entry["status"],
                    "reason": entry["reason"],
                    "record_count": bucket["record_count"],
                    "capacity": bucket["capacity"],
                    "priority": bucket["priority"],
                }
            )
            for event in bucket["history"]:
                log_id = event["log_id"]
                detail = record_reasons.get(
                    log_id,
                    {
                        "log_id": log_id,
                        "status": "reuse_hit",
                        "reason": "bucket_reused",
                    },
                )
                source = final_source.get(log_id)
                records.append(
                    {
                        "log_id": log_id,
                        "bucket_id": bucket["bucket_id"],
                        "bucket_cache_status": entry["status"],
                        "status": detail["status"],
                        "reason": detail["reason"],
                        "record_fingerprint": event["record_fingerprint"],
                        "bucket_arrival_ordinal": event["arrival_ordinal"],
                        "bucket_decision": event["decision"],
                        "source_slot": source["slot_no"] if source else None,
                        "global_rank": source["global_rank"] if source else None,
                        "in_final_sample": source is not None,
                    }
                )
            for detail in entry.get("record_reasons", []):
                if detail["status"] == "removed" and detail["log_id"] not in {
                    event["log_id"] for event in bucket["history"]
                }:
                    records.append(
                        {
                            "log_id": detail["log_id"],
                            "bucket_id": bucket["bucket_id"],
                            "bucket_cache_status": entry["status"],
                            "status": "removed",
                            "reason": detail["reason"],
                            "record_fingerprint": None,
                            "bucket_arrival_ordinal": None,
                            "bucket_decision": None,
                            "source_slot": None,
                            "global_rank": None,
                            "in_final_sample": False,
                        }
                    )

        current_bucket_ids = {
            entry["bucket"]["bucket_id"] for entry in bucket_entries
        }
        for bucket_id, previous in sorted(
            self._previous_index.get("buckets", {}).items()
        ):
            if bucket_id in current_bucket_ids:
                continue
            previous_records = previous.get("record_fingerprints", {})
            if not isinstance(previous_records, dict):
                previous_records = {}
            bucket_ledger.append(
                {
                    "bucket_id": bucket_id,
                    "cache_key": previous.get("key"),
                    "content_fingerprint": previous.get("content_fingerprint"),
                    "params_fingerprint": previous.get("params_fingerprint"),
                    "random_source_digest": previous.get("random_source_digest"),
                    "profile": previous.get("profile"),
                    "budget_tier": previous.get("budget"),
                    "status": "cache_miss",
                    "reason": "bucket_removed_no_current_records",
                    "record_count": 0,
                    "capacity": None,
                    "priority": None,
                }
            )
            for log_id in sorted(previous_records):
                records.append(
                    {
                        "log_id": log_id,
                        "bucket_id": bucket_id,
                        "bucket_cache_status": "cache_miss",
                        "status": "removed",
                        "reason": "record_deleted_or_bucket_removed",
                        "record_fingerprint": previous_records[log_id],
                        "bucket_arrival_ordinal": None,
                        "bucket_decision": None,
                        "source_slot": None,
                        "global_rank": None,
                        "in_final_sample": False,
                    }
                )
        bucket_ledger.sort(key=lambda item: item["bucket_id"])

        return {
            "schema": "lds-reuse-ledger-v1",
            "buckets": bucket_ledger,
            "records": sorted(
                records,
                key=lambda item: (
                    item["bucket_id"],
                    item["bucket_arrival_ordinal"] or 0,
                    item["log_id"],
                ),
            ),
            "final_sample": {
                "cache_key": truncation_meta["cache_key"],
                "status": truncation_meta["status"],
                "reason": truncation_meta["reason"],
            },
            "serializations": serialization_meta["serializations"],
            "execution_tasks": [
                {
                    "bucket_id": task["bucket_id"],
                    "sort_key": task["sort_key"],
                    "cache_key": task["cache_key"],
                }
                for task in sorted(tasks, key=lambda item: tuple(item["sort_key"]))
            ],
            "final_sources": sorted(
                [
                    {"log_id": log_id, **source}
                    for log_id, source in final_source.items()
                ],
                key=lambda item: item["global_rank"],
            ),
        }

    def _build_logical_result(
        self,
        *,
        validation: ValidationResult,
        buckets: dict[str, dict[str, Any]],
        truncation: dict[str, Any],
        diagnostics: dict[str, Any],
        sampling: dict[str, Any],
        global_budget: int,
        quantization: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "seed": self.config.get("seed"),
            "random_source_digest": self.random_source_digest,
            "profile": sampling["profile"],
            "exchange_version": self.config.get("evidence_format", "p3"),
            "global_budget": global_budget,
            "quantization": {
                "decimals": int(quantization.get("decimals", 3)),
                "tier": str(quantization.get("tier", "fine")),
            },
            "validation": {
                "failures": [failure.to_dict() for failure in validation.failures],
                "accepted_records": len(validation.accepted_records),
                "accepted_batches": validation.accepted_batches,
                "rejected_batches": validation.rejected_batches,
                "completed_batch_cursor": validation.completed_batch_cursor,
                "injections_applied": validation.injections_applied,
            },
            "buckets": buckets,
            "truncation": truncation,
            "final_samples": truncation["final_samples"],
            "diagnostics": diagnostics,
        }

    def _update_cache_index(
        self,
        bucket_entries: list[dict[str, Any]],
        truncation_meta: dict[str, Any],
        serialization_meta: dict[str, Any],
    ) -> None:
        index = self.cache.load_index()
        buckets = index.setdefault("buckets", {})
        for entry in bucket_entries:
            bucket_id = entry["bucket"]["bucket_id"]
            buckets[bucket_id] = {
                "key": entry["cache_key"],
                "content_fingerprint": entry["content_fingerprint"],
                "params_fingerprint": entry["params_fingerprint"],
                "random_source_digest": entry["random_source_digest"],
                "profile": entry["profile"],
                "budget": entry["budget_tier"],
                "record_fingerprints": entry["bucket"]["record_fingerprints"],
            }
        for bucket_id in list(buckets):
            if bucket_id not in {
                entry["bucket"]["bucket_id"] for entry in bucket_entries
            }:
                del buckets[bucket_id]
        index["final"] = truncation_meta
        index["serializations"] = {
            version: {
                "key": item["cache_key"],
                "payload_sha256": item["payload_sha256"],
            }
            for version, item in serialization_meta["serializations"].items()
        }
        self.cache.store_index(index)

    def _crash_if(self, point: str) -> None:
        if self.runtime.crash_after == point:
            os._exit(137)

    def _write_stage_json(
        self, context: StageContext, relative: str, value: Any
    ) -> None:
        atomic_write_json(context.stage_dir / relative, value)

    def _write_stage_bytes(
        self, context: StageContext, relative: str, payload: bytes
    ) -> None:
        atomic_write_bytes(context.stage_dir / relative, payload)

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            from .canonical import read_json

            value = read_json(path)
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None


def _group_records(
    records: list[AcceptedRecord],
) -> dict[str, list[AcceptedRecord]]:
    grouped: dict[str, list[AcceptedRecord]] = {}
    for record in records:
        grouped.setdefault(record.bucket_id, []).append(record)
    for bucket_records in grouped.values():
        bucket_records.sort(key=lambda record: record.order_key)
    return grouped


def _validation_artifact(validation: ValidationResult) -> dict[str, Any]:
    return {
        "schema": "lds-validation-stage-v1",
        "accepted_records": [
            {
                "log_id": record.log_id,
                "timestamp": record.timestamp,
                "device_id": record.device_id,
                "severity": record.severity,
                "template_id": record.template_id,
                "metric": record.metric,
                "arrival_seq": record.arrival_seq,
                "record_index": record.record_index,
                "injection_ids": list(record.injection_ids),
            }
            for record in validation.accepted_records
        ],
        "failures": [failure.to_dict() for failure in validation.failures],
        "accepted_batches": validation.accepted_batches,
        "rejected_batches": validation.rejected_batches,
        "completed_batch_cursor": validation.completed_batch_cursor,
        "injections_applied": validation.injections_applied,
        "config_issues": [failure.to_dict() for failure in validation.config_issues],
    }


def _validation_from_artifact(artifact: dict[str, Any]) -> ValidationResult:
    accepted = [
        AcceptedRecord(
            log_id=item["log_id"],
            timestamp=item["timestamp"],
            device_id=item["device_id"],
            severity=item["severity"],
            template_id=item["template_id"],
            metric=item["metric"],
            arrival_seq=item["arrival_seq"],
            record_index=item["record_index"],
            injection_ids=tuple(item.get("injection_ids", [])),
        )
        for item in artifact.get("accepted_records", [])
    ]
    failures = [Failure(**item) for item in artifact.get("failures", [])]
    config_issues = [Failure(**item) for item in artifact.get("config_issues", [])]
    return ValidationResult(
        accepted_records=accepted,
        failures=failures,
        accepted_batches=list(artifact.get("accepted_batches", [])),
        rejected_batches=list(artifact.get("rejected_batches", [])),
        completed_batch_cursor=int(artifact.get("completed_batch_cursor", 0)),
        injections_applied=list(artifact.get("injections_applied", [])),
        config_issues=config_issues,
    )


def compare_logical_results(
    left: dict[str, Any], right: dict[str, Any], *, path: str = "$"
) -> list[str]:
    """Return exact structural differences, excluding volatile cache metadata."""
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        diffs: list[str] = []
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys | right_keys):
            if key not in left:
                diffs.append(f"{path}.{key}: missing on left")
            elif key not in right:
                diffs.append(f"{path}.{key}: missing on right")
            else:
                diffs.extend(
                    compare_logical_results(left[key], right[key], path=f"{path}.{key}")
                )
        return diffs
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: list length {len(left)} != {len(right)}"]
        diffs: list[str] = []
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            diffs.extend(
                compare_logical_results(
                    left_value, right_value, path=f"{path}[{index}]"
                )
            )
        return diffs
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []
