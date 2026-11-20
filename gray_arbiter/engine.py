"""Request execution, mirror isolation, version replacement, and reporting."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import LEDGER_FORMAT_VERSION
from .config import match_rule, validate_config
from .errors import BusinessError, StructuredLogger
from .hashing import gray_destination, mirror_sampled
from .ledger import Ledger, canonical_json
from .model import Config, RequestSpec
from .pool import AcquiredConnection, PoolManager, mono_ms
from .stats import build_error_report


def _logical_ms() -> int:
    return int(time.time() * 1000)


class _TurnGate:
    """Serialize main-pool acquisition attempts in arrival ordinal order."""

    def __init__(self, first_ordinal: int) -> None:
        self.condition = threading.Condition()
        self.next_ordinal = first_ordinal

    def wait_turn(self, ordinal: int) -> None:
        with self.condition:
            while ordinal != self.next_ordinal:
                self.condition.wait(timeout=0.1)

    def advance(self) -> None:
        with self.condition:
            self.next_ordinal += 1
            self.condition.notify_all()

    def pass_without_acquisition(self, ordinal: int) -> None:
        self.wait_turn(ordinal)
        self.advance()


class ArbiterEngine:
    """Coordinate routing, pools, mirroring, statistics, versions, and ledger."""

    def __init__(
        self,
        config: Config,
        *,
        ledger: Ledger | None = None,
        logger: StructuredLogger | None = None,
        snapshot_dir: str | Path | None = None,
        endpoint_overrides: Mapping[str, tuple[str, int]] | None = None,
        worker_count: int = 16,
        start_reaper: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger or StructuredLogger()
        self.ledger = ledger
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.worker_count = max(1, worker_count)
        self.endpoint_overrides = dict(endpoint_overrides or {})
        self.manager = PoolManager(
            config,
            self.logger,
            endpoint_overrides=self.endpoint_overrides,
            start_reaper=start_reaper,
        )
        self.version_history = [config.version]
        self.stale_versions: set[int] = set()
        self.current_report: dict[str, Any] | None = None
        self.replacement_reports: list[dict[str, Any]] = []
        self.last_request_specs: list[RequestSpec] = []
        if self.ledger is not None:
            self.ledger.ensure_epoch(config.version, self._config_digest(config))
            for epoch in self.ledger.ledger_snapshot()["epochs"]:
                if epoch.get("status") == "stale":
                    self.stale_versions.add(int(epoch["version"]))

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------
    def run(
        self,
        requests: Iterable[RequestSpec],
        *,
        batch_size: int = 100,
        resume: bool = False,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        normalized = self._normalize_requests(requests)
        self.last_request_specs = normalized
        recovery: dict[str, Any] | None = None
        resume_ordinal = 1
        if self.ledger is not None:
            recovery = self.ledger.recover()
            if not self.ledger.is_empty():
                replay = self.ledger.replay_self_check()
                recovery["replay"] = replay
                committed = self.ledger.list_batches()
                active_versions = {
                    batch["config_version"] for batch in committed if not batch["stale"]
                }
                if active_versions and active_versions != {self.config.version}:
                    raise BusinessError(
                        "E_REPLAY",
                        "resume requires the ledger's active configuration version",
                        field="config_version",
                        request_id=None,
                        pool_id=None,
                        phase="resume",
                        details={
                            "ledger_active_versions": sorted(active_versions),
                            "ledger_all_versions": sorted(
                                {batch["config_version"] for batch in committed}
                            ),
                            "current_version": self.config.version,
                        },
                    )
                if not resume:
                    raise BusinessError(
                        "E_REPLAY",
                        "ledger already contains committed batches; use resume or a new ledger",
                        field="ledger",
                        request_id=None,
                        pool_id=None,
                        phase="resume",
                        details={"committed_batches": len(recovery["committed_batches"])},
                    )
                resume_ordinal = int(recovery["resume_cursor"])
        if resume_ordinal > len(normalized):
            return {
                "status": "ok",
                "config_version": self.config.version,
                "recovery": recovery,
                "batches": [],
                "summary": {"processed_requests": 0, "resumed_at_cursor": resume_ordinal},
            }
        pending = normalized[resume_ordinal - 1 :]
        batches: list[dict[str, Any]] = []
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset : offset + batch_size]
            batches.append(self._run_batch(chunk))
        result = {
            "status": "ok",
            "config_version": self.config.version,
            "recovery": recovery,
            "batches": batches,
            "summary": {
                "processed_requests": sum(len(batch["requests"]) for batch in batches),
                "resumed_at_cursor": resume_ordinal,
                "committed_batches": len(batches),
                "start_cursor": batches[0]["start_cursor"] if batches else None,
                "end_cursor": batches[-1]["end_cursor"] if batches else None,
            },
            "current_view": self.current_view(),
            "error_report": batches[-1]["error_report"] if batches else None,
        }
        self._write_current_snapshot(result)
        return result

    def _normalize_requests(self, requests: Iterable[RequestSpec]) -> list[RequestSpec]:
        ordered = sorted(
            requests,
            key=lambda item: (item.timestamp_ms, item.line_no, item.request_id),
        )
        normalized = []
        seen: set[str] = set()
        for index, request in enumerate(ordered, start=1):
            if request.request_id in seen:
                raise BusinessError(
                    "E_POOL",
                    "duplicate request_id",
                    field="requests.request_id",
                    request_id=request.request_id,
                    pool_id=None,
                    phase="validation",
                )
            seen.add(request.request_id)
            normalized.append(replace(request, ordinal=index))
        return normalized

    def _run_batch(self, requests: list[RequestSpec]) -> dict[str, Any]:
        start_cursor = requests[0].ordinal
        end_cursor = requests[-1].ordinal
        batch_id = f"v{self.config.version}-b{start_cursor:06d}-{end_cursor:06d}"
        initial_resources = self._resource_snapshot()
        if self.ledger is not None:
            self.ledger.begin_batch(
                batch_id,
                config_version=self.config.version,
                start_cursor=start_cursor,
                end_cursor=end_cursor,
                resource_snapshot=initial_resources,
            )
        self.manager.begin_batch(batch_id)
        if self.ledger is not None:
            self.manager.resource_change_callback = lambda snapshot: self.ledger.update_inflight_resources(
                batch_id, snapshot
            )
        tracker_stop = threading.Event()
        tracker_thread: threading.Thread | None = None
        if self.ledger is not None:
            tracker_thread = threading.Thread(
                target=self._track_inflight,
                args=(batch_id, tracker_stop),
                name=f"inflight-{batch_id}",
                daemon=True,
            )
            tracker_thread.start()

        worker_count = min(64, max(1, len(requests)), self.worker_count)
        turn_gate = _TurnGate(start_cursor)
        request_errors: dict[int, list[BusinessError]] = {}
        outcomes: list[dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self._execute_request, request, batch_id, turn_gate): request
                    for request in requests
                }
                for future, request in futures.items():
                    outcome, errors = future.result()
                    outcomes.append(outcome)
                    request_errors[request.ordinal] = errors
        finally:
            self.manager.resource_change_callback = None
            tracker_stop.set()
            if tracker_thread is not None:
                tracker_thread.join(timeout=1.0)

        outcomes.sort(key=lambda item: item["ordinal"])
        # Logging is deliberately deferred until this point.  This makes log
        # ordering independent of thread scheduling.
        for outcome in outcomes:
            for error in request_errors.get(outcome["ordinal"], []):
                self.logger.emit_error(error)

        stats, events = self.manager.finalize_batch()
        pool_stats = [stat.to_dict() for stat in stats]
        error_report = build_error_report(self.config, outcomes, pool_stats)
        borrows = [borrow.to_dict() for borrow in self.manager.batch_borrows()]
        mirror_records = [
            {
                "ordinal": outcome["ordinal"],
                "request_id": outcome["request_id"],
                "rule_id": outcome["rule_id"],
                "sample_rate_bp": outcome["mirror_sample_rate_bp"],
                "sample_bucket": outcome["mirror_sample_bucket"],
                "pool_id": outcome["mirror_pool_id"],
                "status": outcome["mirror_status"],
                "http_status": outcome["mirror_http_status"],
                "error_code": outcome["mirror_error_code"],
                "response_summary": outcome["mirror_response_summary"],
                "elapsed_ms": outcome["mirror_elapsed_ms"],
                "connection_id": outcome["mirror_connection_id"],
                "reuse_sequence": outcome["mirror_reuse_sequence"],
                "queue_wait_ms": outcome["mirror_queue_wait_ms"],
                "watermarks": outcome["mirror_watermarks"],
            }
            for outcome in outcomes
            if outcome["mirrored"]
        ]
        invariants = self.manager.invariant_violations()
        payload: dict[str, Any] = {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "batch_id": batch_id,
            "config_version": self.config.version,
            "config_digest": self._config_digest(self.config),
            "start_cursor": start_cursor,
            "end_cursor": end_cursor,
            "requests": outcomes,
            "mirror_records": mirror_records,
            "pool_stats": pool_stats,
            "borrows": borrows,
            "connection_events": [event.to_dict() for event in events],
            "error_report": error_report,
            "invariants": invariants,
        }
        digest = ""
        if self.ledger is not None:
            digest = self.ledger.commit_batch(payload)
        snapshot = {
            **payload,
            "content_digest": digest,
            "committed": self.ledger is not None,
            "generated_at_ms": _logical_ms(),
            "config": self._config_snapshot(self.config),
            "pool_watermark_series": self.manager.batch_watermark_series(),
            "runtime": self.manager.snapshot(),
            "ledger": self.ledger.ledger_snapshot() if self.ledger is not None else None,
        }
        self._write_batch_snapshot(snapshot)
        self.current_report = snapshot
        return snapshot

    def _track_inflight(self, batch_id: str, stop: threading.Event) -> None:
        while not stop.wait(0.025):
            if self.ledger is None:
                return
            try:
                self.ledger.update_inflight_resources(batch_id, self._resource_snapshot())
            except Exception:
                # The committed batch remains authoritative.  A failure to
                # refresh the diagnostic snapshot must not abort request work.
                return

    def _resource_snapshot(self) -> dict[str, Any]:
        snapshot = self.manager.snapshot()
        return {
            "connections": [
                conn
                for conn in snapshot.get("connections", [])
                if conn.get("state") != "closed"
            ],
            "borrows": [
                borrow
                for borrow in snapshot.get("borrows", [])
                if borrow.get("state") == "active"
            ],
            "captured_mono_ms": mono_ms(),
        }

    # ------------------------------------------------------------------
    # One request
    # ------------------------------------------------------------------
    def _execute_request(
        self, request: RequestSpec, batch_id: str, turn_gate: _TurnGate
    ) -> tuple[dict[str, Any], list[BusinessError]]:
        errors: list[BusinessError] = []
        outcome: dict[str, Any] = {
            "ordinal": request.ordinal,
            "line_no": request.line_no,
            "request_id": request.request_id,
            "path": request.path,
            "client_key": request.client_key,
            "timestamp_ms": request.timestamp_ms,
            "rule_id": None,
            "route_bucket": None,
            "gray_weight_bp": None,
            "main_destination": None,
            "main_pool_id": None,
            "main_status": "error",
            "main_http_status": None,
            "main_error_code": None,
            "main_response_summary": None,
            "main_elapsed_ms": None,
            "connection_id": None,
            "reuse_sequence": None,
            "borrow_id": None,
            "lender_pool": None,
            "borrow_deadline_mono_ms": None,
            "connection_state_at_use": None,
            "queue_started_mono_ms": None,
            "queue_ended_mono_ms": None,
            "queue_wait_ms": 0,
            "mirrored": False,
            "mirror_sample_bucket": None,
            "mirror_sample_rate_bp": None,
            "mirror_pool_id": None,
            "mirror_status": "not_sampled",
            "mirror_http_status": None,
            "mirror_error_code": None,
            "mirror_response_summary": None,
            "mirror_elapsed_ms": None,
            "mirror_connection_id": None,
            "mirror_reuse_sequence": None,
            "mirror_queue_wait_ms": 0,
            "watermarks": {},
            "mirror_watermarks": {},
        }
        rule = match_rule(request.path, self.config.rules)
        if rule is None:
            turn_gate.pass_without_acquisition(request.ordinal)
            error = BusinessError(
                "E_ROUTE",
                "no routing rule matches the request path",
                field="path",
                request_id=request.request_id,
                pool_id=None,
                phase="routing",
            )
            errors.append(error)
            outcome["main_error_code"] = error.code
            outcome["watermarks"] = self.manager.request_watermarks()
            outcome["mirrored"] = False
            return outcome, errors

        destination, route_bucket = gray_destination(
            request.client_key, rule.rule_id, rule.gray_weight_bp
        )
        main_pool = rule.gray_pool if destination == "gray" else rule.stable_pool
        outcome.update(
            {
                "rule_id": rule.rule_id,
                "route_bucket": route_bucket,
                "gray_weight_bp": rule.gray_weight_bp,
                "main_destination": destination,
                "main_pool_id": main_pool,
            }
        )

        acquired: AcquiredConnection | None = None
        main_error: BusinessError | None = None
        try:
            turn_gate.wait_turn(request.ordinal)
            try:
                acquired = self.manager.acquire(
                    main_pool,
                    request.request_id,
                    batch_id,
                    allow_borrow=True,
                    block=True,
                    timeout_ms=self.config.queue_timeout_ms,
                    http_timeout_ms=self.config.main_timeout_ms,
                )
            finally:
                turn_gate.advance()
            outcome.update(
                {
                    "connection_id": acquired.connection.connection_id,
                    "reuse_sequence": acquired.reuse_sequence,
                    "borrow_id": acquired.borrow_id,
                    "lender_pool": acquired.lender_pool,
                    "borrow_deadline_mono_ms": acquired.borrow_deadline_mono_ms,
                    "connection_state_at_use": acquired.connection.state,
                    "queue_started_mono_ms": acquired.queue_started_mono_ms,
                    "queue_ended_mono_ms": acquired.queue_ended_mono_ms,
                    "queue_wait_ms": acquired.wait_ms,
                }
            )
            main_timeout_ms = self.config.main_timeout_ms
            if acquired.borrow_deadline_mono_ms is not None:
                remaining = acquired.borrow_deadline_mono_ms - mono_ms()
                main_timeout_ms = max(1, min(main_timeout_ms, remaining))
            result = self.manager.execute_http(
                acquired,
                request.path,
                request.request_id,
                timeout_ms=main_timeout_ms,
                phase="main",
            )
            outcome.update(
                {
                    "main_status": result["status"],
                    "main_http_status": result["http_status"],
                    "main_response_summary": result["response_summary"],
                    "main_elapsed_ms": result["elapsed_ms"],
                }
            )
        except BusinessError as error:
            error.details.setdefault("failure_stage", error.phase)
            error.phase = "main"
            main_error = error
            errors.append(error)
            outcome["main_status"] = "error"
            outcome["main_error_code"] = error.code
            if error.details:
                outcome["queue_started_mono_ms"] = error.details.get(
                    "queue_started_mono_ms", outcome["queue_started_mono_ms"]
                )
                outcome["queue_ended_mono_ms"] = error.details.get(
                    "queue_ended_mono_ms", outcome["queue_ended_mono_ms"]
                )
                outcome["queue_wait_ms"] = error.details.get(
                    "wait_ms", outcome["queue_wait_ms"]
                )
                outcome["main_elapsed_ms"] = error.details.get("elapsed_ms")
                if error.details.get("borrow_id"):
                    outcome["borrow_id"] = error.details["borrow_id"]
                    outcome["lender_pool"] = error.details.get("lender_pool")
        except Exception as exc:  # pragma: no cover - defensive conversion
            error = BusinessError(
                "E_TARGET",
                "unexpected main request failure",
                field="request",
                request_id=request.request_id,
                pool_id=main_pool,
                phase="main",
                details={"exception": type(exc).__name__, "trace": traceback.format_exc(limit=3)},
            )
            main_error = error
            errors.append(error)
            outcome["main_status"] = "error"
            outcome["main_error_code"] = error.code
        finally:
            if acquired is not None:
                self.manager.release(acquired, request.request_id, error=main_error)
            outcome["watermarks"] = self.manager.request_watermarks()

        # Mirror sampling is decided independently of the main outcome.  A
        # mirror failure is recorded in mirror_* fields only.
        sampled, mirror_bucket = mirror_sampled(
            request.request_id, rule.rule_id, rule.mirror_sample_bp
        )
        outcome.update(
            {
                "mirrored": sampled,
                "mirror_sample_bucket": mirror_bucket,
                "mirror_sample_rate_bp": rule.mirror_sample_bp,
                "mirror_pool_id": rule.mirror_pool,
            }
        )
        if sampled:
            mirror_acquired: AcquiredConnection | None = None
            mirror_error: BusinessError | None = None
            try:
                mirror_acquired = self.manager.acquire(
                    rule.mirror_pool,
                    request.request_id,
                    batch_id,
                    allow_borrow=False,
                    block=False,
                    timeout_ms=self.config.mirror_timeout_ms,
                    http_timeout_ms=self.config.mirror_timeout_ms,
                )
                outcome.update(
                    {
                        "mirror_connection_id": mirror_acquired.connection.connection_id,
                        "mirror_reuse_sequence": mirror_acquired.reuse_sequence,
                        "mirror_queue_wait_ms": mirror_acquired.wait_ms,
                    }
                )
                result = self.manager.execute_http(
                    mirror_acquired,
                    request.path,
                    request.request_id,
                    timeout_ms=self.config.mirror_timeout_ms,
                    phase="mirror",
                )
                outcome.update(
                    {
                        "mirror_status": result["status"],
                        "mirror_http_status": result["http_status"],
                        "mirror_response_summary": result["response_summary"],
                        "mirror_elapsed_ms": result["elapsed_ms"],
                    }
                )
            except BusinessError as error:
                error.details.setdefault("failure_stage", error.phase)
                error.phase = "mirror"
                mirror_error = error
                errors.append(error)
                outcome["mirror_status"] = (
                    "timeout"
                    if error.code == "E_TIMEOUT"
                    else "pool_exhausted"
                    if error.code == "E_MIRROR"
                    else "error"
                )
                outcome["mirror_error_code"] = error.code
                outcome["mirror_elapsed_ms"] = error.details.get("elapsed_ms")
                outcome["mirror_queue_wait_ms"] = error.details.get(
                    "wait_ms", outcome["mirror_queue_wait_ms"]
                )
            except Exception as exc:  # pragma: no cover - defensive conversion
                mirror_error = BusinessError(
                    "E_MIRROR",
                    "unexpected mirror request failure",
                    field="mirror",
                    request_id=request.request_id,
                    pool_id=rule.mirror_pool,
                    phase="mirror",
                    details={"exception": type(exc).__name__},
                )
                errors.append(mirror_error)
                outcome["mirror_status"] = "error"
                outcome["mirror_error_code"] = mirror_error.code
            finally:
                if mirror_acquired is not None:
                    self.manager.release(
                        mirror_acquired, request.request_id, error=mirror_error
                    )
                outcome["mirror_watermarks"] = self.manager.request_watermarks()
        return outcome, errors

    # ------------------------------------------------------------------
    # Whole-config replacement
    # ------------------------------------------------------------------
    def replace_config(
        self,
        proposed: Config,
        *,
        borrow_policy: str = "force_return",
    ) -> dict[str, Any]:
        if borrow_policy not in {"force_return", "recompute", "invalidate"}:
            raise ValueError("unsupported borrow policy")
        validated = validate_config(proposed, detect_borrow_cycles=True, cycle_code="E_POOL")
        if len(self.config.pools) and set(validated.pools) != set(self.config.pools):
            # Full replacement is allowed to add/remove pools, but references
            # have already been checked above.
            pass
        old_config = self.config
        new_version = max(old_config.version + 1, validated.version)
        new_config = replace(validated, version=new_version)
        old_digest = self._config_digest(old_config)
        new_digest = self._config_digest(new_config)
        if old_digest == new_digest and new_version == old_config.version:
            raise BusinessError(
                "E_CONFIG",
                "replacement config is unchanged",
                field="config",
                request_id=None,
                pool_id=None,
                phase="replacement",
            )
        dispositions = self.manager.adjudicate_borrows(borrow_policy, new_config)
        self.manager.close_all("version_replaced")
        self.manager.shutdown()
        old_manager = self.manager
        self.manager = PoolManager(
            new_config,
            self.logger,
            endpoint_overrides=self.endpoint_overrides,
            start_reaper=True,
        )
        self.config = new_config
        self.stale_versions.add(old_config.version)
        self.version_history.append(new_config.version)
        if self.ledger is not None:
            self.ledger.mark_epoch_stale(old_config.version, new_config.version, new_digest)
        changed_rules = self._changed_rules(old_config, new_config)
        changed_pools = self._changed_pools(old_config, new_config)
        report = {
            "old_version": old_config.version,
            "new_version": new_config.version,
            "version_monotonic": new_config.version > old_config.version,
            "old_config_digest": old_digest,
            "new_config_digest": new_digest,
            "stale_versions": sorted(self.stale_versions),
            "current_versions": [new_config.version],
            "borrow_policy": borrow_policy,
            "borrow_dispositions": dispositions,
            "changed_rules": changed_rules,
            "changed_pools": changed_pools,
            "replacement_semantics": "whole_snapshot_replace_not_incremental_merge",
            "old_manager_summary": old_manager.snapshot(),
            "current_view": {
                "config": self._config_snapshot(new_config),
                "pools": self.manager.snapshot(),
                "ledger": self.ledger.ledger_snapshot() if self.ledger is not None else None,
            },
        }
        self.replacement_reports.append(report)
        self.current_report = None
        self._write_replacement_snapshot(report)
        return report

    def _changed_rules(self, old: Config, new: Config) -> list[dict[str, Any]]:
        changes = []
        for rule_id in sorted(set(old.rules) | set(new.rules)):
            old_rule = old.rules.get(rule_id)
            new_rule = new.rules.get(rule_id)
            old_dict = asdict(old_rule) if old_rule else None
            new_dict = asdict(new_rule) if new_rule else None
            if old_dict == new_dict:
                continue
            changes.append(
                {
                    "rule_id": rule_id,
                    "old": old_dict,
                    "new": new_dict,
                    "old_distribution": self._distribution_for_rule(
                        self.last_request_specs, old, old_rule
                    ),
                    "new_distribution": self._distribution_for_rule(
                        self.last_request_specs, new, new_rule
                    ),
                    "distribution_basis": "last_normalized_request_batch",
                }
            )
        return changes

    def _distribution_for_rule(
        self,
        requests: Iterable[RequestSpec],
        config: Config,
        rule: Any,
    ) -> dict[str, int]:
        result = {"stable": 0, "gray": 0, "mirror_sampled": 0, "unmatched": 0}
        if rule is None:
            result["unmatched"] = len(self.last_request_specs)
            return result
        for request in requests:
            matched = match_rule(request.path, config.rules)
            if matched is None or matched.rule_id != rule.rule_id:
                result["unmatched"] += 1
                continue
            destination, _ = gray_destination(
                request.client_key, rule.rule_id, rule.gray_weight_bp
            )
            result[destination] += 1
            sampled, _ = mirror_sampled(
                request.request_id, rule.rule_id, rule.mirror_sample_bp
            )
            if sampled:
                result["mirror_sampled"] += 1
        return result

    def _changed_pools(self, old: Config, new: Config) -> list[dict[str, Any]]:
        changes = []
        keys = (
            "max_connections",
            "max_requests_per_connection",
            "idle_timeout_ms",
            "lendable_to",
            "host",
            "port",
        )
        for pool_id in sorted(set(old.pools) | set(new.pools)):
            old_pool = old.pools.get(pool_id)
            new_pool = new.pools.get(pool_id)
            old_values = asdict(old_pool) if old_pool else None
            new_values = asdict(new_pool) if new_pool else None
            if old_values == new_values:
                continue
            changes.append(
                {
                    "pool_id": pool_id,
                    "old": old_values,
                    "new": new_values,
                    "old_capacity": old_pool.max_connections if old_pool else None,
                    "new_capacity": new_pool.max_connections if new_pool else None,
                    "changed_fields": [
                        key
                        for key in keys
                        if (old_values or {}).get(key) != (new_values or {}).get(key)
                    ],
                }
            )
        return changes

    # ------------------------------------------------------------------
    # Reports and snapshots
    # ------------------------------------------------------------------
    def current_view(self) -> dict[str, Any]:
        current = self.current_report
        if current is not None and current.get("config_version") != self.config.version:
            current = None
        return {
            "config_version": self.config.version,
            "version_history": list(self.version_history),
            "stale_versions": sorted(self.stale_versions),
            "config": self._config_snapshot(self.config),
            "pools": self.manager.snapshot(),
            "report_scope": "current_version_only",
            "current_batch": current,
            "ledger": self.ledger.ledger_snapshot() if self.ledger is not None else None,
            "replacement_reports": list(self.replacement_reports),
            "structured_logs": self.logger.sorted_events(),
        }

    def _config_snapshot(self, config: Config) -> dict[str, Any]:
        return {
            "version": config.version,
            "digest": self._config_digest(config),
            "borrow_ttl_ms": config.borrow_ttl_ms,
            "queue_timeout_ms": config.queue_timeout_ms,
            "main_timeout_ms": config.main_timeout_ms,
            "mirror_timeout_ms": config.mirror_timeout_ms,
            "pools": [
                {
                    "pool_id": pool.pool_id,
                    "host": pool.host,
                    "port": pool.port,
                    "max_connections": pool.max_connections,
                    "max_requests_per_connection": pool.max_requests_per_connection,
                    "idle_timeout_ms": pool.idle_timeout_ms,
                    "lendable_to": list(pool.lendable_to),
                }
                for pool in sorted(config.pools.values(), key=lambda item: item.pool_id)
            ],
            "rules": [
                asdict(rule) for rule in sorted(config.rules.values(), key=lambda item: item.rule_id)
            ],
        }

    @staticmethod
    def _config_digest(config: Config) -> str:
        return hashlib.sha256(canonical_json(config.digest_material()).encode("utf-8")).hexdigest()

    def _write_batch_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self.snapshot_dir is None:
            return
        name = (
            f"batch-v{snapshot['config_version']}-"
            f"{snapshot['start_cursor']:06d}-{snapshot['end_cursor']:06d}.json"
        )
        self._atomic_write(self.snapshot_dir / name, snapshot)

    def _write_replacement_snapshot(self, report: dict[str, Any]) -> None:
        if self.snapshot_dir is None:
            return
        name = f"replacement-v{report['old_version']}-to-v{report['new_version']}.json"
        self._atomic_write(self.snapshot_dir / name, report)

    def _write_current_snapshot(self, result: dict[str, Any]) -> None:
        if self.snapshot_dir is None:
            return
        self._atomic_write(self.snapshot_dir / "latest.json", result)

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)

    def close(self) -> None:
        self.manager.shutdown()
        if self.ledger is not None:
            self.ledger.close()
