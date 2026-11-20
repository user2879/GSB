"""Scaling engine: fast cache, authoritative ledger replay, version lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import DivergenceError, ReplayError, VersionConflictError
from .estimator import MinuteDecision, decide_minute
from .ledger import Ledger
from .logging_util import StructuredLogger
from .spec import ClusterSpec, parse_spec


class ForcedKill(RuntimeError):
    """Raised by the deterministic kill hook before the minute is committed."""


@dataclass(frozen=True)
class VersionReplacement:
    old_version: int
    new_version: int
    created: bool
    affected_services: tuple[str, ...]
    affected_minute_range: tuple[int, int]
    parameter_diffs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RecoveryReport:
    run_id: str
    state: str
    committed_minutes: tuple[int, ...]
    unknown_minutes: tuple[int, ...]
    last_committed_minute: int | None
    note: str


@dataclass(frozen=True)
class ReplayReport:
    run_id: str
    checked_minutes: int
    checked_service_rows: int
    ok: bool


class ScalingEngine:
    """Owns the ledger, the active spec and the in-memory fast cache."""

    def __init__(self, db_path: str, log_path: str,
                 run_id: str = "run-current") -> None:
        self.ledger = Ledger(db_path)
        self.logger = StructuredLogger(log_path)
        self.run_id = run_id
        self.spec_version = 0
        self.spec: ClusterSpec | None = None
        self._cache: dict[int, MinuteDecision] = {}
        self._cache_basis_version = 0

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "ScalingEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def initialize(self, spec: ClusterSpec) -> tuple[int, bool]:
        spec_version, created = self.ledger.register_spec(spec, 0)
        self.spec_version = spec_version
        self.spec = spec
        run = self.ledger.get_run(self.run_id)
        if run is None:
            self.ledger.create_run(self.run_id, spec_version, spec.horizon)
        self._cache = {}
        self._cache_basis_version = spec_version
        self.logger.log(
            "engine_initialize",
            run_id=self.run_id,
            spec_version=spec_version,
            created=created,
            digest=spec.digest,
        )
        return spec_version, created

    @property
    def next_minute(self) -> int:
        highest = self.ledger.highest_committed_minute(self.run_id)
        return 0 if highest is None else highest + 1

    # --------------------------------------------------------------- advance
    def advance_minute(self, minute: int | None = None,
                       simulate_kill_at: int | None = None) -> MinuteDecision:
        if self.spec is None:
            raise RuntimeError("engine is not initialized with a cluster spec")
        if minute is None:
            minute = self.next_minute
        if not (0 <= minute < self.spec.horizon):
            raise ValueError(f"minute {minute} outside horizon {self.spec.horizon}")
        existing = self.ledger.get_minute(self.run_id, minute)
        if existing is not None and existing.status == "COMMITTED":
            decision = self._recompute_minute(minute)
            self._cache[minute] = decision
            return decision
        if simulate_kill_at == minute:
            self.logger.log(
                "forced_kill_injected",
                run_id=self.run_id,
                minute=minute,
                spec_version=self.spec_version,
            )
            raise ForcedKill(f"simulated forced kill before committing minute {minute}")

        decision = decide_minute(self.spec, minute, self.spec_version)
        self.ledger.commit_minute(self.run_id, decision)
        self._cache[minute] = decision
        self._cache_basis_version = self.spec_version
        self.logger.log(
            "minute_committed",
            run_id=self.run_id,
            minute=minute,
            spec_version=self.spec_version,
            budget_used=decision.budget_used,
            total_rejected=decision.total_rejected,
        )
        return decision

    def advance_all(self, simulate_kill_at: int | None = None) -> list[MinuteDecision]:
        results = []
        while self.next_minute < (self.spec.horizon if self.spec else 0):
            results.append(self.advance_minute(simulate_kill_at=simulate_kill_at))
        return results

    # ------------------------------------------------------------ recovery
    def recover(self) -> RecoveryReport:
        run = self.ledger.get_run(self.run_id)
        if run is None:
            raise RuntimeError(f"run {self.run_id!r} does not exist in the ledger")
        committed = self.ledger.committed_minutes(self.run_id)
        last = committed[-1] if committed else None
        horizon = int(run["horizon"])
        expected_next = 0 if last is None else last + 1
        unknown: list[int] = []
        if run["state"] == "RUNNING" and expected_next < horizon:
            unknown = [expected_next]
            self.ledger.mark_run_unknown(self.run_id, expected_next)
            self.logger.log(
                "recovery_unknown_minute",
                run_id=self.run_id,
                unknown_minute=expected_next,
            )
        spec_version = int(run["spec_version"])
        spec_payload = self.ledger.get_spec(spec_version)
        if spec_payload is None:
            raise VersionConflictError(
                f"run references missing spec version {spec_version}",
                fields=[f"/runs/{self.run_id}/spec_version"],
            )
        self.spec_version = spec_version
        self.spec = parse_spec(spec_payload)
        self._cache = {
            record.minute: self._decision_from_record(record)
            for record in self.ledger.iter_committed(self.run_id)
        }
        self._cache_basis_version = spec_version
        report = RecoveryReport(
            run_id=self.run_id,
            state=self.ledger.get_run(self.run_id)["state"],
            committed_minutes=tuple(committed),
            unknown_minutes=tuple(unknown),
            last_committed_minute=last,
            note=(
                "no forced-termination exit report can be produced; "
                "the unknown minute must be re-run"
                if unknown
                else "all reachable minutes were durably committed"
            ),
        )
        self.logger.log("recovery", **{
            "run_id": report.run_id,
            "state": report.state,
            "committed_minutes": list(report.committed_minutes),
            "unknown_minutes": list(report.unknown_minutes),
        })
        return report

    def resume_after_recovery(self) -> list[MinuteDecision]:
        if self.spec is None:
            raise RuntimeError("call recover() before resume_after_recovery()")
        results = []
        while self.next_minute < self.spec.horizon:
            results.append(self.advance_minute())
        return results

    # -------------------------------------------------------------- replay
    def verify_replay(self) -> ReplayReport:
        run = self.ledger.get_run(self.run_id)
        if run is None:
            raise RuntimeError(f"run {self.run_id!r} does not exist")
        checked_minutes = 0
        checked_rows = 0
        for record in self.ledger.iter_committed(self.run_id):
            recomputed = self._recompute_minute(record.minute)
            self._compare_decision(record, recomputed)
            checked_minutes += 1
            checked_rows += len(recomputed.allocations)
        report = ReplayReport(
            run_id=self.run_id,
            checked_minutes=checked_minutes,
            checked_service_rows=checked_rows,
            ok=True,
        )
        self.logger.log("replay_verified", **{
            "run_id": self.run_id,
            "checked_minutes": checked_minutes,
            "checked_service_rows": checked_rows,
        })
        return report

    def _compare_decision(self, record: Any, recomputed: MinuteDecision) -> None:
        stored = record.decision
        top_fields = ("budget_used", "budget_remaining", "total_rejected",
                      "spec_version", "spec_digest")
        for field_name in top_fields:
            if stored.get(field_name) != getattr(recomputed, field_name):
                raise ReplayError(
                    "ledger replay mismatch on minute-level field",
                    fields=[f"/runs/{self.run_id}/minutes/{record.minute}/{field_name}"],
                    context={
                        "minute": record.minute,
                        "service": None,
                        "field": field_name,
                        "ledger_value": stored.get(field_name),
                        "replayed_value": getattr(recomputed, field_name),
                        "basis": f"spec_version={record.spec_version}",
                    },
                )
        stored_rows = {item["service"]: item for item in stored.get("allocations", [])}
        for alloc in recomputed.allocations:
            stored_row = stored_rows.get(alloc.service)
            if stored_row is None:
                raise ReplayError(
                    "ledger replay missing service row",
                    fields=[f"/runs/{self.run_id}/minutes/{record.minute}/services/{alloc.service}"],
                    context={"minute": record.minute, "service": alloc.service,
                             "field": "service"},
                )
            for field_name in ("allocated_replicas", "rejected_requests"):
                if stored_row.get(field_name) != getattr(alloc, field_name):
                    raise ReplayError(
                        "ledger replay mismatch on service field",
                        fields=[f"/runs/{self.run_id}/minutes/{record.minute}/services/{alloc.service}/{field_name}"],
                        context={
                            "minute": record.minute,
                            "service": alloc.service,
                            "field": field_name,
                            "ledger_value": stored_row.get(field_name),
                            "replayed_value": getattr(alloc, field_name),
                            "basis": f"spec_version={record.spec_version}",
                        },
                    )
            if list(stored_row.get("trim_reasons", [])) != list(alloc.trim_reasons):
                raise ReplayError(
                    "ledger replay mismatch on trim reasons",
                    fields=[f"/runs/{self.run_id}/minutes/{record.minute}/services/{alloc.service}/trim_reasons"],
                    context={
                        "minute": record.minute,
                        "service": alloc.service,
                        "field": "trim_reasons",
                        "ledger_value": stored_row.get("trim_reasons"),
                        "replayed_value": list(alloc.trim_reasons),
                        "basis": f"spec_version={record.spec_version}",
                    },
                )
            stored_est = stored_row.get("estimate", {})
            for field_name in ("raw_demand", "demand", "per_replica_capacity",
                               "effective_capacity", "requests"):
                if stored_est.get(field_name) != getattr(alloc.estimate, field_name):
                    raise ReplayError(
                        "ledger replay mismatch on estimate field",
                        fields=[f"/runs/{self.run_id}/minutes/{record.minute}/services/{alloc.service}/estimate/{field_name}"],
                        context={
                            "minute": record.minute,
                            "service": alloc.service,
                            "field": f"estimate.{field_name}",
                            "ledger_value": stored_est.get(field_name),
                            "replayed_value": getattr(alloc.estimate, field_name),
                            "basis": f"spec_version={record.spec_version}",
                        },
                    )

    # --------------------------------------------------- fast / authoritative
    def fast_query(self, minute: int) -> MinuteDecision:
        """One in-memory read; never touches SQLite during the decision read."""
        if minute in self._cache:
            return self._cache[minute]
        record = self.ledger.get_minute(self.run_id, minute)
        if record is None or record.status != "COMMITTED":
            raise KeyError(f"minute {minute} is not committed in run {self.run_id!r}")
        decision = self._decision_from_record(record)
        self._cache[minute] = decision
        return decision

    def authoritative_query(self, minute: int) -> MinuteDecision:
        record = self.ledger.get_minute(self.run_id, minute)
        if record is None or record.status != "COMMITTED":
            raise KeyError(f"minute {minute} is not committed in run {self.run_id!r}")
        return self._recompute_minute(minute)

    def verify_fast_vs_authoritative(self) -> None:
        for minute in self.ledger.committed_minutes(self.run_id):
            fast = self.fast_query(minute)
            authoritative = self._recompute_minute(minute)
            self._assert_same_decision(
                fast,
                authoritative,
                error_cls=DivergenceError,
                cache_basis=self._cache_basis_version,
            )

    def rebuild_authoritative_view(self) -> list[MinuteDecision]:
        return [self._recompute_minute(m) for m in self.ledger.committed_minutes(self.run_id)]

    def debug_corrupt_cache(self, minute: int, service: str,
                            field_name: str = "allocated_replicas") -> None:
        """Test hook: mutate only the cache to force an E_DIVERGE report."""
        decision = self.fast_query(minute)
        allocations = []
        for alloc in decision.allocations:
            if alloc.service == service:
                object.__setattr__(alloc, field_name,
                                   getattr(alloc, field_name) + 1)
            allocations.append(alloc)
        object.__setattr__(decision, "allocations", tuple(allocations))
        self._cache[minute] = decision

    def _assert_same_decision(self, fast: MinuteDecision,
                              authoritative: MinuteDecision,
                              error_cls: type[Exception],
                              cache_basis: int) -> None:
        top_fields = ("budget_used", "budget_remaining", "total_rejected")
        for field_name in top_fields:
            if getattr(fast, field_name) != getattr(authoritative, field_name):
                raise error_cls(
                    "fast-path result diverges from authoritative replay",
                    fields=[f"/runs/{self.run_id}/minutes/{fast.minute}/{field_name}"],
                    context={
                        "minute": fast.minute,
                        "service": None,
                        "field": field_name,
                        "fast_value": getattr(fast, field_name),
                        "authoritative_value": getattr(authoritative, field_name),
                        "cache_basis_spec_version": cache_basis,
                    },
                )
        auth_rows = {item.service: item for item in authoritative.allocations}
        for fast_alloc in fast.allocations:
            auth_alloc = auth_rows[fast_alloc.service]
            for field_name in ("allocated_replicas", "rejected_requests"):
                if getattr(fast_alloc, field_name) != getattr(auth_alloc, field_name):
                    raise error_cls(
                        "fast-path result diverges from authoritative replay",
                        fields=[f"/runs/{self.run_id}/minutes/{fast.minute}/services/{fast_alloc.service}/{field_name}"],
                        context={
                            "minute": fast.minute,
                            "service": fast_alloc.service,
                            "field": field_name,
                            "fast_value": getattr(fast_alloc, field_name),
                            "authoritative_value": getattr(auth_alloc, field_name),
                            "cache_basis_spec_version": cache_basis,
                        },
                    )
            if list(fast_alloc.trim_reasons) != list(auth_alloc.trim_reasons):
                raise error_cls(
                    "fast-path result diverges from authoritative replay",
                    fields=[f"/runs/{self.run_id}/minutes/{fast.minute}/services/{fast_alloc.service}/trim_reasons"],
                    context={
                        "minute": fast.minute,
                        "service": fast_alloc.service,
                        "field": "trim_reasons",
                        "fast_value": list(fast_alloc.trim_reasons),
                        "authoritative_value": list(auth_alloc.trim_reasons),
                        "cache_basis_spec_version": cache_basis,
                    },
                )

    def _recompute_minute(self, minute: int) -> MinuteDecision:
        record = self.ledger.get_minute(self.run_id, minute)
        if record is None:
            raise KeyError(minute)
        spec_payload = self.ledger.get_spec(record.spec_version)
        if spec_payload is None:
            raise VersionConflictError(
                f"ledger references unknown spec version {record.spec_version}",
                fields=[f"/spec_versions/{record.spec_version}"],
            )
        versioned_spec = parse_spec(spec_payload)
        return decide_minute(versioned_spec, minute, record.spec_version)

    def _decision_from_record(self, record: Any) -> MinuteDecision:
        from .estimator import Allocation, Estimate
        allocations = []
        for row in record.decision.get("allocations", []):
            est_raw = row["estimate"]
            estimate = Estimate(
                service=est_raw["service"],
                minute=est_raw["minute"],
                requests=est_raw["requests"],
                per_replica_capacity=est_raw["per_replica_capacity"],
                effective_capacity=est_raw["effective_capacity"],
                raw_demand=est_raw["raw_demand"],
                min_demand=est_raw["min_demand"],
                quota_replica_cap=est_raw["quota_replica_cap"],
                quota_demand=est_raw["quota_demand"],
                demand=est_raw["demand"],
                clamped_min=est_raw["clamped_min"],
                clamped_max=est_raw["clamped_max"],
                quota_limited=est_raw["quota_limited"],
                rules=tuple(est_raw["rules"]),
            )
            allocations.append(Allocation(
                service=row["service"],
                minute=row["minute"],
                allocated_replicas=row["allocated_replicas"],
                servable_requests=row["servable_requests"],
                rejected_requests=row["rejected_requests"],
                trim_reasons=tuple(row["trim_reasons"]),
                estimate=estimate,
            ))
        return MinuteDecision(
            minute=record.decision["minute"],
            spec_digest=record.decision["spec_digest"],
            spec_version=record.decision["spec_version"],
            budget=record.decision["budget"],
            budget_used=record.decision["budget_used"],
            budget_remaining=record.decision["budget_remaining"],
            total_rejected=record.decision["total_rejected"],
            allocations=tuple(allocations),
        )

    # -------------------------------------------------- version replacement
    def replace_spec(self, new_spec: ClusterSpec,
                     new_run_id: str | None = None) -> VersionReplacement:
        if self.spec is None:
            raise RuntimeError("engine has no active spec to replace")
        old_version = self.spec_version
        old_spec = self.spec
        new_version, created = self.ledger.register_spec(new_spec, self.next_minute)
        if not created:
            new_spec_version = new_version
        else:
            self.ledger.mark_spec_read_only(old_version)
            new_spec_version = new_version

        self.run_id = new_run_id or f"run-v{new_spec_version}"
        if self.ledger.get_run(self.run_id) is None:
            self.ledger.create_run(self.run_id, new_spec_version, new_spec.horizon)
        self.spec_version = new_spec_version
        self.spec = new_spec
        # Whole-view replacement: the cache is discarded, never minute-merged.
        self._cache = {}
        self._cache_basis_version = new_spec_version

        affected = sorted({svc.name for svc in (old_spec.services + new_spec.services)
                           if _service_changed(old_spec, new_spec, svc.name)})
        upper = max(old_spec.horizon, new_spec.horizon) - 1
        diffs = tuple(_diff_specs(old_spec, new_spec))
        report = VersionReplacement(
            old_version=old_version,
            new_version=new_spec_version,
            created=created,
            affected_services=tuple(affected),
            affected_minute_range=(0, upper),
            parameter_diffs=diffs,
        )
        self.logger.log("spec_replaced", **{
            "old_version": report.old_version,
            "new_version": report.new_version,
            "created": report.created,
            "affected_services": list(report.affected_services),
            "affected_minute_range": list(report.affected_minute_range),
            "parameter_diffs": list(report.parameter_diffs),
        })
        return report

    def get_version_view(self, spec_version: int) -> dict[str, Any]:
        status = self.ledger.get_spec_status(spec_version)
        if status is None:
            raise VersionConflictError(
                f"unknown spec version {spec_version}",
                fields=[f"/spec_versions/{spec_version}"],
            )
        runs = [row for row in self.ledger.list_runs()
                if int(row["spec_version"]) == spec_version]
        return {
            "spec_version": spec_version,
            "status": status,
            "read_only": status in ("STALE", "READ_ONLY"),
            "runs": runs,
            "spec": self.ledger.get_spec(spec_version),
        }

    def list_versions(self) -> list[dict[str, Any]]:
        return self.ledger.list_spec_versions()

    # -------------------------------------------------------------- snapshots
    def save_snapshot(self, snapshot_id: str, target_minute: int,
                      note: str = "") -> None:
        self.ledger.save_snapshot(
            snapshot_id, self.run_id, self.spec_version, target_minute, note
        )
        self.logger.log("snapshot_saved", snapshot_id=snapshot_id,
                        run_id=self.run_id, spec_version=self.spec_version,
                        target_minute=target_minute)

    def restore_snapshot(self, snapshot_id: str) -> RecoveryReport:
        self.ledger.assert_snapshot_version(snapshot_id, self.spec_version)
        snapshot = self.ledger.get_snapshot(snapshot_id)
        self.run_id = snapshot["run_id"]
        report = self.recover()
        self.logger.log("snapshot_restored", snapshot_id=snapshot_id,
                        run_id=self.run_id, target_minute=snapshot["target_minute"])
        return report

    # ----------------------------------------------------------- aggregates
    def numpy_load_summary(self) -> dict[str, Any]:
        if self.spec is None:
            raise RuntimeError("engine is not initialized")
        matrix = np.array(
            [list(svc.requests) for svc in self.spec.services],
            dtype=np.int64,
        )
        return {
            "dtype": str(matrix.dtype),
            "service_order": [svc.name for svc in self.spec.services],
            "per_service_total": matrix.sum(axis=1, dtype=np.int64).tolist(),
            "per_minute_total": matrix.sum(axis=0, dtype=np.int64).tolist(),
            "grand_total": int(matrix.sum(dtype=np.int64)),
            "peak_per_service": matrix.max(axis=1).tolist(),
        }

    @property
    def cache_basis_version(self) -> int:
        return self._cache_basis_version


def _service_changed(old: ClusterSpec, new: ClusterSpec, name: str) -> bool:
    old_map = {svc.name: svc for svc in old.services}
    new_map = {svc.name: svc for svc in new.services}
    if name not in old_map or name not in new_map:
        return True
    return old_map[name] != new_map[name]


def _diff_specs(old: ClusterSpec, new: ClusterSpec) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if old.global_replica_budget != new.global_replica_budget:
        diffs.append({
            "path": "/global_replica_budget",
            "old": old.global_replica_budget,
            "new": new.global_replica_budget,
        })
    names = sorted({svc.name for svc in old.services} |
                   {svc.name for svc in new.services})
    old_map = {svc.name: svc for svc in old.services}
    new_map = {svc.name: svc for svc in new.services}
    attrs = ("requests", "concurrency_limit", "request_ms",
             "scale_threshold_pct", "min_replicas", "max_replicas",
             "concurrency_quota")
    for name in names:
        old_svc = old_map.get(name)
        new_svc = new_map.get(name)
        if old_svc is None or new_svc is None:
            diffs.append({"path": f"/services/{name}",
                          "old": None if old_svc is None else "present",
                          "new": None if new_svc is None else "present"})
            continue
        for attr in attrs:
            old_value = getattr(old_svc, attr)
            new_value = getattr(new_svc, attr)
            if old_value != new_value:
                diffs.append({
                    "path": f"/services/{name}/{attr}",
                    "old": list(old_value) if attr == "requests" else old_value,
                    "new": list(new_value) if attr == "requests" else new_value,
                })
    return diffs
