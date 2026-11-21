"""Deterministic concurrent arbitration engine."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arbitration import apply_relations, decide_ticket
from .cache import (
    CacheStore,
    build_reference_index,
    convert_cache_key,
    decision_cache_key,
    relations_cache_key,
    score_cache_key,
)
from .config import config_fingerprint
from .constants import (
    ERROR_STAGE_EXCEPTION,
    ERROR_VERSION_LOSSY,
    STAGE_ORDER,
    STAGE_RANK,
)
from .diagnostics import (
    build_diagnostic_report,
    empty_stage_stats,
    merge_stage_stats,
    write_diagnostic_report,
)
from .loader import load_evidence_text
from .model import ArbitrationConfig, Evidence, Rules
from .numeric import fingerprint, sha256_bytes, write_json
from .rules import rules_fingerprint
from .scoring import score_ticket
from .snapshot import SnapshotStore
from .versions import (
    decision_to_ir,
    read_v1,
    read_v2,
    read_v3,
    roundtrip_consistency,
    v1_degradation_detail,
    write_v1,
    write_v2,
    write_v3,
)


class InjectedStageError(RuntimeError):
    def __init__(self, exception_type: str, message: str):
        super().__init__(message)
        self.exception_type = exception_type


class StopRequested(Exception):
    pass


@dataclass
class RunResult:
    tickets: list[dict]
    failures: list[dict]
    diagnostics: dict
    global_version: int
    interrupted: bool
    manifest: dict | None
    output_info: dict
    reference_index: dict
    cache_payload: dict

    def summary(self) -> dict:
        upgraded = sum(1 for ticket in self.tickets if int(ticket["upgrade_code"]) == 1)
        return {
            "completed_ticket_count": len(self.tickets),
            "current_upgrade_count": upgraded,
            "failure_count": len(self.failures),
            "boundary_ticket_ids": [ticket["ticket_id"] for ticket in self.tickets if ticket.get("boundary")],
            "snapshot_version": self.global_version,
            "interrupted": self.interrupted,
        }

    def comparison_payload(self) -> dict:
        return {
            "tickets": self.tickets,
            "failures": self.failures,
            "artifact_sha256s": self.output_info.get("artifact_sha256s", {}),
            "conversion_report_core": self.output_info.get("conversion_report_core", {}),
        }


def _normalize_collection(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    normalized = empty_stage_stats()
    merge_stage_stats(normalized, stats)
    return normalized


def _exact_ir_equal(left: list[dict], right: list[dict]) -> bool:
    left_map = {item["ticket_id"]: item for item in left}
    right_map = {item["ticket_id"]: item for item in right}
    if sorted(left_map) != sorted(right_map):
        return False
    for ticket_id in left_map:
        left_item = left_map[ticket_id]
        right_item = right_map[ticket_id]
        for key in (
            "upgrade_code",
            "priority_level",
            "priority_name",
            "score_fixed",
            "precision",
            "reason_codes",
            "missing",
            "boundary",
        ):
            if left_item.get(key) != right_item.get(key):
                return False
    return True


class ArbitrationEngine:
    def __init__(
        self,
        *,
        rules: Rules,
        config: ArbitrationConfig,
        evidence_lines: list[str],
        workspace_root: str | Path = ".",
        cache_store: CacheStore | None = None,
        snapshot_store: SnapshotStore | None = None,
        global_version: int = 0,
        exception_injections: list[dict] | None = None,
        output_dir: str | Path | None = "outputs",
        write_outputs: bool = True,
        pre_run_delay_ms: int = 0,
    ):
        self.rules = rules
        self.config = config
        self.evidence_lines = [str(line) for line in evidence_lines]
        self.workspace_root = Path(workspace_root)
        self.cache = cache_store or CacheStore()
        self.snapshot_store = snapshot_store
        self.global_version = int(global_version)
        self.publish_sequence = 0
        self.exception_injections = sorted(
            [item for item in (exception_injections or []) if isinstance(item, dict)],
            key=lambda item: (
                str(item.get("ticket_id", "*")),
                STAGE_RANK.get(str(item.get("stage", "")), 99),
                str(item.get("exception_type", "")),
                str(item.get("message", "")),
            ),
        )
        self.output_dir = (self.workspace_root / Path(output_dir)) if output_dir is not None else None
        self.write_outputs_enabled = bool(write_outputs and output_dir is not None)
        self.pre_run_delay_ms = max(0, int(pre_run_delay_ms))
        self._run_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._active_workers = 0
        self._observed_max_workers = 0
        self.reference_index: dict[str, Any] = {"tickets": [], "evidence_types": {}, "roles": {}, "relations": {}}
        self.last_result: RunResult | None = None

    @property
    def config_hash(self) -> str:
        return config_fingerprint(self.config)

    @property
    def rules_hash(self) -> str:
        return rules_fingerprint(self.rules)

    def request_stop(self) -> None:
        self._stop_event.set()

    def clear_stop(self) -> None:
        self._stop_event.clear()

    def _injection(self, ticket_id: str, stage: str) -> dict | None:
        for injection in self.exception_injections:
            if not injection.get("active", True):
                continue
            target_ticket = str(injection.get("ticket_id", "*"))
            target_stage = str(injection.get("stage", "*"))
            if target_ticket not in (ticket_id, "*"):
                continue
            if target_stage not in (stage, "*"):
                continue
            return injection
        return None

    def _run_injection(self, ticket_id: str, stage: str) -> None:
        injection = self._injection(ticket_id, stage)
        if injection is None:
            return
        delay_ms = int(injection.get("delay_ms", 0) or 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        action = str(injection.get("action", "raise")).lower()
        if action == "delay":
            return
        raise InjectedStageError(
            str(injection.get("exception_type", "InjectedStageError")),
            str(injection.get("message", f"injected failure at {stage}")),
        )

    def _enter_worker(self) -> None:
        with self._state_lock:
            self._active_workers += 1
            self._observed_max_workers = max(self._observed_max_workers, self._active_workers)

    def _leave_worker(self) -> None:
        with self._state_lock:
            self._active_workers -= 1

    def _stage_value(
        self,
        *,
        stage: str,
        ticket_id: str,
        key: str,
        old_cache: CacheStore,
        new_entries: dict[str, dict],
        stats: dict[str, dict[str, int]],
        compute,
    ) -> dict:
        cached = old_cache.lookup(stage, ticket_id, key)
        previous = old_cache.entry(stage, ticket_id)
        if cached is not None:
            stats[stage]["reused"] += 1
            return cached
        stats[stage]["computed"] += 1
        if previous is not None:
            stats[stage]["invalidated"] += 1
        else:
            stats[stage]["cache_miss"] += 1
        self._run_injection(ticket_id, stage)
        value = compute()
        new_entries[stage] = {"key": key, "value": value}
        return value

    def _process_ticket(
        self,
        ticket_id: str,
        evidences: tuple[Evidence, ...],
        old_cache: CacheStore,
        use_cache: bool,
    ) -> dict:
        self._enter_worker()
        stats = empty_stage_stats()
        new_entries: dict[str, dict] = {}
        task_sort_keys = {
            stage_name: fingerprint({"ticket_id": ticket_id, "stage": stage_name})
            for stage_name in STAGE_ORDER
        }
        stage = "score"
        try:
            if self._stop_event.is_set():
                return {"ticket_id": ticket_id, "skipped": True, "task_sort_keys": task_sort_keys}
            lookup_cache = old_cache if use_cache else CacheStore()
            score_key = score_cache_key(ticket_id, evidences, self.rules, self.config)
            score_result = self._stage_value(
                stage="score",
                ticket_id=ticket_id,
                key=score_key,
                old_cache=lookup_cache,
                new_entries=new_entries,
                stats=stats,
                compute=lambda: score_ticket(ticket_id, evidences, self.rules, self.config),
            )
            stage = "relations"
            relations_key = relations_cache_key(
                ticket_id, evidences, score_result, score_key, self.rules, self.config
            )
            relation_result = self._stage_value(
                stage="relations",
                ticket_id=ticket_id,
                key=relations_key,
                old_cache=lookup_cache,
                new_entries=new_entries,
                stats=stats,
                compute=lambda: apply_relations(
                    ticket_id, evidences, score_result, self.rules, self.config
                ),
            )
            stage = "decision"
            decision_key = decision_cache_key(
                ticket_id, relation_result, relations_key, self.rules, self.config
            )
            decision = self._stage_value(
                stage="decision",
                ticket_id=ticket_id,
                key=decision_key,
                old_cache=lookup_cache,
                new_entries=new_entries,
                stats=stats,
                compute=lambda: decide_ticket(ticket_id, relation_result, self.rules, self.config),
            )
            stage = "convert"
            conversion_key = convert_cache_key(ticket_id, decision, decision_key, self.config)

            def compute_conversion() -> dict:
                ir = decision_to_ir(decision, self.config)
                return {
                    "target_version": self.config.target_version,
                    "ir": ir,
                    "v1_degradation_from_ir": v1_degradation_detail(
                        ir, self.config, source_version="ir"
                    ),
                    "v1_degradation_from_v3": v1_degradation_detail(
                        ir, self.config, source_version="v3"
                    ),
                }

            conversion = self._stage_value(
                stage="convert",
                ticket_id=ticket_id,
                key=conversion_key,
                old_cache=lookup_cache,
                new_entries=new_entries,
                stats=stats,
                compute=compute_conversion,
            )
            return {
                "ticket_id": ticket_id,
                "skipped": False,
                "decision": decision,
                "conversion": conversion,
                "cache_entries": new_entries,
                "stage_stats": stats,
                "failure": None,
                "task_sort_keys": task_sort_keys,
            }
        except StopRequested:
            return {"ticket_id": ticket_id, "skipped": True, "task_sort_keys": task_sort_keys}
        except InjectedStageError as exc:
            failure = {
                "ticket_id": ticket_id,
                "line_no": None,
                "evidence_seq": None,
                "stage": stage,
                "code": ERROR_STAGE_EXCEPTION,
                "exception_type": exc.exception_type,
                "message": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - one ticket must never poison the batch
            failure = {
                "ticket_id": ticket_id,
                "line_no": None,
                "evidence_seq": None,
                "stage": stage,
                "code": ERROR_STAGE_EXCEPTION,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            self._leave_worker()
        return {
            "ticket_id": ticket_id,
            "skipped": False,
            "decision": None,
            "conversion": None,
            "cache_entries": new_entries,
            "stage_stats": stats,
            "failure": failure,
            "task_sort_keys": task_sort_keys,
        }

    def _failure_sort_key(self, failure: dict) -> tuple:
        return (
            str(failure.get("ticket_id", "")),
            int(failure.get("line_no") or 0),
            STAGE_RANK.get(str(failure.get("stage", "")), 99),
            str(failure.get("stage", "")),
            str(failure.get("code", "")),
            str(failure.get("message", "")),
        )

    def _stage_versions(self, cache: CacheStore) -> dict[str, str]:
        return {stage: cache.stage_version(stage) for stage in STAGE_ORDER}

    def _summary(self, tickets: list[dict], failures: list[dict], partial: bool) -> dict:
        return {
            "completed_ticket_count": len(tickets),
            "current_upgrade_count": sum(1 for ticket in tickets if int(ticket["upgrade_code"]) == 1),
            "failure_count": len(failures),
            "completed_ticket_ids": sorted(ticket["ticket_id"] for ticket in tickets),
            "partial": bool(partial),
        }

    def _degradations_from_tickets(self, tickets: list[dict]) -> dict:
        details = {"v1": [], "v2": [], "v3": [], "v3_to_v1": []}
        for ticket in tickets:
            ir = decision_to_ir(ticket, self.config)
            detail = v1_degradation_detail(ir, self.config, source_version="ir")
            if detail is not None:
                details["v1"].append(detail)
            detail_v3 = v1_degradation_detail(ir, self.config, source_version="v3")
            if detail_v3 is not None:
                details["v3_to_v1"].append(detail_v3)
        return details

    def _commit_snapshot(
        self,
        *,
        cache: CacheStore,
        tickets: list[dict],
        failures: list[dict],
        partial: bool,
        extra_manifest: dict | None = None,
    ) -> dict | None:
        if self.snapshot_store is None:
            return None
        self.global_version += 1
        self.publish_sequence += 1
        degraded = self._degradations_from_tickets(tickets)
        conversions = {
            "schema": "snapshot-conversions/v1",
            "target_version": self.config.target_version,
            "degradation_details": degraded,
        }
        manifest = self.snapshot_store.commit(
            global_version=self.global_version,
            config_hash=self.config_hash,
            rules_fingerprint=self.rules_hash,
            cache_payload=cache.as_dict(),
            decisions=tickets,
            failures=failures,
            summary=self._summary(tickets, failures, partial),
            conversions=conversions,
            completed_ticket_ids=[ticket["ticket_id"] for ticket in tickets],
            stage_versions=self._stage_versions(cache),
            partial=partial,
            publish_sequence=self.publish_sequence,
            extra_manifest=extra_manifest,
        )
        return manifest

    def _write_exchange_outputs(self, tickets: list[dict]) -> dict:
        if not self.write_outputs_enabled:
            return {
                "artifact_sha256s": {},
                "conversion_report_core": {},
                "conversion_report": {},
            }
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(self.config.temp_dir)
        if not temp_dir.is_absolute():
            temp_dir = self.workspace_root / temp_dir
        ir_records = [decision_to_ir(ticket, self.config) for ticket in tickets]
        ir_records.sort(key=lambda item: item["ticket_id"])
        v1_path = self.output_dir / "arbitration.v1.txt"
        v2_path = self.output_dir / "arbitration.v2.json"
        v3_path = self.output_dir / "arbitration.v3.bin"
        v1_info = write_v1(ir_records, v1_path, self.config, temp_dir=temp_dir)
        v2_info = write_v2(ir_records, v2_path, self.config, temp_dir=temp_dir)
        v3_info = write_v3(ir_records, v3_path, self.config, temp_dir=temp_dir)
        v1_read = read_v1(v1_path, self.config, sidecar_path=v1_info["sidecar_path"])
        v2_read = read_v2(v2_path, self.config)
        v3_read = read_v3(v3_path, self.config)
        roundtrip = {
            "v1": {
                **roundtrip_consistency(ir_records, v1_read),
                "exact_ir_equal": _exact_ir_equal(ir_records, v1_read),
            },
            "v2": {
                **roundtrip_consistency(ir_records, v2_read),
                "exact_ir_equal": _exact_ir_equal(ir_records, v2_read),
            },
            "v3": {
                **roundtrip_consistency(ir_records, v3_read),
                "exact_ir_equal": _exact_ir_equal(ir_records, v3_read),
            },
        }
        v3_to_v1 = []
        for record in ir_records:
            detail = v1_degradation_detail(record, self.config, source_version="v3")
            if detail is not None:
                v3_to_v1.append(detail)
        report_core = {
            "schema": "version-conversion-report/v1",
            "record_count": len(ir_records),
            "targets": {
                "v1": {
                    "record_count": v1_info["record_count"],
                    "degradation_details": v1_info["degradations"],
                    "sidecar": True,
                },
                "v2": {"record_count": v2_info["record_count"], "degradation_details": []},
                "v3": {
                    "record_count": v3_info["record_count"],
                    "degradation_details": [],
                    "feature_names": v3_info["feature_names"],
                },
            },
            "v3_to_v1": v3_to_v1,
            "degradation_details": {
                "v1": v1_info["degradations"],
                "v2": [],
                "v3": [],
                "v3_to_v1": v3_to_v1,
            },
            "codes": [ERROR_VERSION_LOSSY] if v1_info["degradations"] or v3_to_v1 else [],
            "triple_roundtrip": roundtrip,
        }
        report_path = self.output_dir / "version_conversion_report.json"
        write_json(report_path, report_core, temp_dir=temp_dir, pretty=True)
        artifact_hashes = {
            "v1": sha256_bytes(v1_path.read_bytes()),
            "v1_sidecar": sha256_bytes(Path(v1_info["sidecar_path"]).read_bytes()),
            "v2": sha256_bytes(v2_path.read_bytes()),
            "v3": sha256_bytes(v3_path.read_bytes()),
        }
        return {
            "paths": {
                "v1": str(v1_path),
                "v1_sidecar": v1_info["sidecar_path"],
                "v2": str(v2_path),
                "v3": str(v3_path),
                "conversion_report": str(report_path),
            },
            "artifact_sha256s": artifact_hashes,
            "conversion_report": report_core,
            "conversion_report_core": report_core,
        }

    def run(self, *, use_cache: bool = True, mode: str = "incremental") -> RunResult:
        with self._run_lock:
            if self.pre_run_delay_ms:
                time.sleep(self.pre_run_delay_ms / 1000.0)
            self._observed_max_workers = 0
            loaded = load_evidence_text("\n".join(self.evidence_lines), self.rules)
            self.reference_index = build_reference_index(loaded.evidence_by_ticket, self.rules)
            ticket_ids = sorted(loaded.evidence_by_ticket)
            old_cache = self.cache if use_cache else CacheStore()
            working_cache = CacheStore.from_dict(old_cache.as_dict()) if use_cache else CacheStore()
            aggregate_stats = empty_stage_stats()
            stage_failures: list[dict] = []
            decisions: list[dict] = []
            completed_ids: list[str] = []
            completion_order: list[str] = []
            task_sort_keys: list[str] = []
            skipped = 0
            processed = 0
            final_manifest: dict | None = None

            effective_workers = max(1, min(self.config.worker_threads, max(1, len(ticket_ids))))
            if ticket_ids:
                with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="arb-ticket") as executor:
                    future_map = {
                        executor.submit(
                            self._process_ticket,
                            ticket_id,
                            loaded.evidence_by_ticket[ticket_id],
                            old_cache,
                            use_cache,
                        ): ticket_id
                        for ticket_id in ticket_ids
                    }
                    for future in as_completed(future_map):
                        ticket_id = future_map[future]
                        try:
                            outcome = future.result()
                        except Exception as exc:  # pragma: no cover - executor safety net
                            outcome = {
                                "ticket_id": ticket_id,
                                "skipped": False,
                                "decision": None,
                                "cache_entries": {},
                                "stage_stats": empty_stage_stats(),
                                "failure": {
                                    "ticket_id": ticket_id,
                                    "line_no": None,
                                    "evidence_seq": None,
                                    "stage": "pipeline",
                                    "code": ERROR_STAGE_EXCEPTION,
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                },
                            }
                        if outcome.get("skipped"):
                            skipped += 1
                            continue
                        processed += 1
                        completion_order.append(ticket_id)
                        task_sort_keys.extend(outcome.get("task_sort_keys", {}).values())
                        merge_stage_stats(aggregate_stats, outcome.get("stage_stats", {}))
                        for stage, entry in outcome.get("cache_entries", {}).items():
                            working_cache.stages.setdefault(stage, {})[ticket_id] = entry
                        if outcome.get("failure") is not None:
                            stage_failures.append(outcome["failure"])
                        elif outcome.get("decision") is not None:
                            decisions.append(outcome["decision"])
                            completed_ids.append(ticket_id)
                        if (
                            self.snapshot_store is not None
                            and processed % max(1, self.config.snapshot_commit_interval) == 0
                        ):
                            final_manifest = self._commit_snapshot(
                                cache=working_cache,
                                tickets=sorted(decisions, key=lambda item: item["ticket_id"]),
                                failures=sorted(loaded.failures + stage_failures, key=self._failure_sort_key),
                                partial=True,
                                extra_manifest={"checkpoint_completed": processed},
                            )

            all_failures = sorted(loaded.failures + stage_failures, key=self._failure_sort_key)
            decisions.sort(key=lambda item: item["ticket_id"])
            completed_set = {ticket["ticket_id"] for ticket in decisions}
            working_cache.prune(completed_set)
            interrupted = self._stop_event.is_set() or skipped > 0
            self.cache = working_cache
            output_info = self._write_exchange_outputs(decisions)
            final_manifest = self._commit_snapshot(
                cache=working_cache,
                tickets=decisions,
                failures=all_failures,
                partial=interrupted,
                extra_manifest={"checkpoint_completed": processed, "interrupted": interrupted},
            )
            recovery = self.snapshot_store.recovery_info(final_manifest) if self.snapshot_store else {"available": False}
            final_order = [ticket["ticket_id"] for ticket in decisions]
            concurrency = {
                "configured_workers": self.config.worker_threads,
                "effective_workers": effective_workers,
                "observed_max_workers": self._observed_max_workers,
                "worker_limit_respected": self._observed_max_workers <= self.config.worker_threads,
                "completion_order_matches_final_order": sorted(completion_order) == final_order,
                "final_order_is_ticket_id_sorted": final_order == sorted(final_order),
                "stable_task_keys_unique": len(task_sort_keys) == len(set(task_sort_keys)),
                "stable_task_key_count": len(task_sort_keys),
                "failed_ticket_count": len({failure["ticket_id"] for failure in stage_failures}),
                "comparison_basis": "stable per-ticket/per-stage keys and final ticket-id sort",
            }
            concurrency["conclusion"] = (
                "PASS"
                if concurrency["worker_limit_respected"]
                and concurrency["completion_order_matches_final_order"]
                and concurrency["final_order_is_ticket_id_sorted"]
                and concurrency["stable_task_keys_unique"]
                else "FAIL"
            )
            degradation = output_info["conversion_report"].get("degradation_details") or {
                "v1": [],
                "v2": [],
                "v3": [],
                "v3_to_v1": [],
            }
            # Keep a stable aggregate even when output writing is disabled.
            if not degradation.get("v1") and not degradation.get("v3_to_v1"):
                computed_degradation = self._degradations_from_tickets(decisions)
                degradation = computed_degradation
            report = build_diagnostic_report(
                mode=mode,
                config_hash=self.config_hash,
                rules_fingerprint=self.rules_hash,
                tickets=decisions,
                failures=all_failures,
                stage_stats=_normalize_collection(aggregate_stats),
                concurrency=concurrency,
                snapshot_version=self.global_version,
                recovery=recovery,
                degradation_details=degradation,
                conversion_roundtrip=output_info["conversion_report"].get("triple_roundtrip", {}),
            )
            if self.write_outputs_enabled and self.output_dir is not None:
                temp_dir = Path(self.config.temp_dir)
                if not temp_dir.is_absolute():
                    temp_dir = self.workspace_root / temp_dir
                write_diagnostic_report(self.output_dir / "diagnostic_report.json", report, temp_dir=temp_dir)
            result = RunResult(
                tickets=decisions,
                failures=all_failures,
                diagnostics=report,
                global_version=self.global_version,
                interrupted=interrupted,
                manifest=final_manifest,
                output_info=output_info,
                reference_index=deepcopy(self.reference_index),
                cache_payload=working_cache.as_dict(),
            )
            self.last_result = result
            return result

    def compare_with_full(self) -> dict:
        if self.last_result is None:
            raise RuntimeError("incremental run must complete before full comparison")
        reference = ArbitrationEngine(
            rules=self.rules,
            config=self.config,
            evidence_lines=self.evidence_lines,
            workspace_root=self.workspace_root,
            cache_store=CacheStore(),
            snapshot_store=None,
            exception_injections=self.exception_injections,
            output_dir=self.workspace_root / "tmp" / "full-comparison",
            write_outputs=True,
        )
        full = reference.run(use_cache=False, mode="full")
        incremental_payload = self.last_result.comparison_payload()
        full_payload = full.comparison_payload()
        equal = incremental_payload == full_payload
        return {
            "equal": equal,
            "incremental_ticket_count": len(self.last_result.tickets),
            "full_ticket_count": len(full.tickets),
            "incremental_failure_count": len(self.last_result.failures),
            "full_failure_count": len(full.failures),
            "artifact_hashes_equal": (
                incremental_payload.get("artifact_sha256s") == full_payload.get("artifact_sha256s")
            ),
            "diff_basis": "tickets+contributions+decisions+failures+exchange artifact hashes",
        }
