from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .arbitration import Candidate, KeyResult, ViewSummary, arbitrate_key, summarize_view
from .canonical import canonical_json
from .client import Receipt, ServiceManager
from .errors import SpecError
from .fixedpoint import quantize_decimal
from .ledger import Ledger
from .scheduler import BoundedScheduler, DispatchResult, SliceTask
from .spec import (
    DeploymentSpec,
    SliceSpec,
    ViewSpec,
    parse_slice,
)


@dataclass
class ViewRun:
    summary: ViewSummary
    dispatches: tuple[dict[str, Any], ...]
    service_errors: dict[str, dict[str, Any]]
    propagation: dict[str, Any] | None = None
    ledger_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "dispatches": list(self.dispatches),
            "service_errors": self.service_errors,
            "propagation": self.propagation,
            "ledger_digest": self.ledger_digest,
        }


@dataclass
class InvalidationReport:
    sequence: int
    view_id: str
    previous_version: int
    new_version: int
    affected_keys: tuple[str, ...]
    reasons: tuple[dict[str, Any], ...]
    run: ViewRun | None = None
    removed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "view_id": self.view_id,
            "previous_version": self.previous_version,
            "new_version": self.new_version,
            "affected_keys": list(self.affected_keys),
            "reasons": list(self.reasons),
            "removed": self.removed,
            "run": self.run.to_dict() if self.run is not None else None,
        }


@dataclass
class EngineRunReport:
    views: tuple[ViewRun, ...]
    invalidations: tuple[InvalidationReport, ...]
    start_errors: dict[str, dict[str, Any]]
    scheduler: dict[str, Any]
    ledger: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "views": [item.to_dict() for item in self.views],
            "invalidations": [item.to_dict() for item in self.invalidations],
            "start_errors": self.start_errors,
            "scheduler": self.scheduler,
            "ledger": self.ledger,
        }


class ArbiterEngine:
    def __init__(self, spec: DeploymentSpec, ledger: Ledger) -> None:
        self.spec = spec
        self.ledger = ledger
        self.services = ServiceManager(dict(spec.environments), spec.session.seed)
        self.scheduler = BoundedScheduler(spec.session)
        self.current_slices: dict[str, SliceSpec] = spec.current_slices()
        self.views: dict[str, ViewSpec] = dict(spec.views)
        self.versions: dict[str, int] = {
            view_id: (ledger.view_state(view_id).version if ledger.view_state(view_id) else 0)
            for view_id in self.views
        }
        self.latest_runs: dict[str, ViewRun] = {}

    def start_services(self) -> dict[str, dict[str, Any]]:
        return self.services.start_all(self.spec.session.handshake_timeout_ms)

    def stop_services(self) -> None:
        self.services.stop_all()

    def close(self) -> None:
        self.stop_services()
        self.ledger.close()

    def _execute_task(self, task: SliceTask) -> Receipt:
        return self.services.get(task.env_id).send_slice(task.slice_spec, self.spec.session)

    def _environment_snapshot(self, view: ViewSpec) -> dict[str, Any]:
        return {
            env_id: {
                "version": self.current_slices[env_id].version,
                "values": dict(sorted(self.current_slices[env_id].values.items())),
            }
            for env_id in sorted(view.environments)
        }

    def _snapshot_slice_changes(
        self,
        old_snapshot: dict[str, Any],
        current_snapshot: dict[str, Any],
        view: ViewSpec,
    ) -> tuple[list[dict[str, Any]], set[str], bool]:
        reasons: list[dict[str, Any]] = []
        changed_keys: set[str] = set()
        incomplete = False
        for env_id in sorted(view.environments):
            old_env = old_snapshot.get(env_id)
            current_env = current_snapshot[env_id]
            if not isinstance(old_env, dict) or not isinstance(old_env.get("values"), dict):
                incomplete = True
                continue
            old_values = old_env["values"]
            new_values = current_env["values"]
            old_version = old_env.get("version")
            new_version = current_env["version"]
            for key in sorted(view.keys):
                old_value = old_values.get(key)
                new_value = new_values.get(key)
                if old_value == new_value:
                    continue
                if old_value is None:
                    kind = "SLICE_KEY_ADDED"
                elif new_value is None:
                    kind = "SLICE_KEY_REMOVED"
                else:
                    kind = "SLICE_VALUE_CHANGED"
                reasons.append(
                    {
                        "kind": kind,
                        "environment": env_id,
                        "key": key,
                        "from_value": old_value,
                        "to_value": new_value,
                        "from_version": old_version,
                        "to_version": new_version,
                    }
                )
                changed_keys.add(key)
        return reasons, changed_keys, incomplete

    @staticmethod
    def _view_change_affected_keys(
        old: ViewSpec | None,
        new: ViewSpec,
        reasons: Iterable[dict[str, Any]],
    ) -> set[str]:
        old_keys = set(old.keys) if old is not None else set()
        new_keys = set(new.keys)
        if old is None or set(old.environments) != set(new.environments):
            return old_keys | new_keys
        return {str(reason["key"]) for reason in reasons if reason.get("key") is not None}

    def _compute_view(
        self,
        view: ViewSpec,
        version: int,
    ) -> tuple[ViewSummary, tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
        tasks = [
            SliceTask(
                task_id=f"{view.id}/{env_id}",
                view_id=view.id,
                env_id=env_id,
                slice_spec=self.current_slices[env_id],
            )
            for env_id in sorted(view.environments)
        ]
        dispatches: list[DispatchResult[Receipt]] = self.scheduler.run_batch(
            tasks, self._execute_task
        )
        receipts: dict[str, Receipt] = {}
        service_errors: dict[str, dict[str, Any]] = {}
        dispatch_reports: list[dict[str, Any]] = []
        for dispatch in dispatches:
            env_id = dispatch.task.env_id
            dispatch_reports.append(
                {
                    "task_id": dispatch.task.task_id,
                    "view_id": dispatch.task.view_id,
                    "environment": env_id,
                    "slice_version": dispatch.task.slice_spec.version,
                    "ok": dispatch.error is None,
                    "error": dispatch.error,
                    "metrics": dispatch.metrics.to_dict(),
                    "receipt": dispatch.value.to_dict() if dispatch.value is not None else None,
                }
            )
            if dispatch.error is not None:
                service_errors[env_id] = dispatch.error
            elif dispatch.value is not None:
                receipts[env_id] = dispatch.value

        participant_errors = [
            service_errors[env_id]
            for env_id in sorted(service_errors)
        ]
        key_results: list[KeyResult] = []
        for key in sorted(view.keys):
            rule = view.keys[key]
            candidates: list[Candidate] = []
            for env_id in sorted(view.environments):
                receipt = receipts.get(env_id)
                environment = self.spec.environments[env_id]
                if receipt is None or key not in receipt.aggregate_summary:
                    continue
                value = self.current_slices[env_id].values[key]
                quantized = quantize_decimal(
                    value,
                    self.spec.session.scale_exponent,
                    f"slice.{env_id}.{key}",
                )
                candidates.append(
                    Candidate(
                        env_id=env_id,
                        slice_version=receipt.slice_version,
                        aggregate_units=receipt.aggregate_summary[key],
                        weight_units=environment.weight_units,
                        value_quantization_error_units=quantized.error_units,
                        weight_quantization_error_units=environment.weight_error_units,
                        receipt_digest=receipt.digest,
                        sample_digest=receipt.sample_digest,
                    )
                )
            key_results.append(
                arbitrate_key(
                    view=view,
                    key=key,
                    rule=rule,
                    candidates=candidates,
                    scale_exponent=self.spec.session.scale_exponent,
                    source_errors=participant_errors,
                )
            )
        summary = summarize_view(view.id, version, key_results)
        self.latest_runs[view.id] = ViewRun(
            summary=summary,
            dispatches=tuple(dispatch_reports),
            service_errors=dict(sorted(service_errors.items())),
        )
        return summary, tuple(dispatch_reports), dict(sorted(service_errors.items()))

    def _commit(
        self,
        view: ViewSpec,
        summary: ViewSummary,
        dispatches: tuple[dict[str, Any], ...],
        service_errors: dict[str, dict[str, Any]],
        propagation: dict[str, Any] | None,
    ) -> ViewRun:
        digest = self.ledger.commit_view(
            summary,
            config_digest=view.config_digest(),
            view_config_json=canonical_json(view.config_payload()),
            environment_snapshot_json=canonical_json(self._environment_snapshot(view)),
            propagation=propagation,
            active=True,
        )
        self.versions[view.id] = summary.version
        run = ViewRun(
            summary=summary,
            dispatches=dispatches,
            service_errors=service_errors,
            propagation=propagation,
            ledger_digest=digest,
        )
        self.latest_runs[view.id] = run
        return run

    def run_view(
        self,
        view_id: str,
        *,
        force_bump: bool | None = None,
        extra_reasons: Iterable[dict[str, Any]] = (),
    ) -> ViewRun:
        if view_id not in self.views:
            raise SpecError(f"$.views.{view_id}", "view is not active in this engine")
        view = self.views[view_id]
        state = self.ledger.view_state(view_id)
        current_rows = self.ledger.current_results(view_id)
        reasons: list[dict[str, Any]] = list(extra_reasons)
        bump = bool(force_bump)
        affected_keys: set[str] = set()
        if state is None:
            version = 1
        else:
            if not state.active:
                bump = True
                reasons.append({"kind": "VIEW_REACTIVATED"})
                affected_keys |= set(view.keys)
            if state.config_digest != view.config_digest():
                bump = True
                old_view = ViewSpec.from_config_payload(
                    view_id, _json_object_or_empty(state.view_config_json)
                )
                if old_view is not None:
                    config_reasons = self._view_change_reasons(old_view, view)
                    reasons.extend(config_reasons)
                    affected_keys |= self._view_change_affected_keys(
                        old_view, view, config_reasons
                    )
                else:
                    reasons.append({"kind": "VIEW_CONFIG_DRIFT"})
                    affected_keys |= set(view.keys)
            if not current_rows:
                bump = True
                reasons.append({"kind": "NO_CURRENT_RESULTS"})
                affected_keys |= set(view.keys)
            old_snapshot = _json_object_or_empty(state.environment_snapshot_json)
            current_snapshot = self._environment_snapshot(view)
            if old_snapshot:
                slice_reasons, slice_keys, incomplete = self._snapshot_slice_changes(
                    old_snapshot, current_snapshot, view
                )
                if slice_reasons:
                    bump = True
                    reasons.extend(slice_reasons)
                    affected_keys |= slice_keys
            else:
                incomplete = True
            if incomplete:
                stored_versions: dict[str, int] = {}
                if current_rows:
                    try:
                        loaded = json.loads(current_rows[0]["slice_versions_json"])
                        if isinstance(loaded, dict):
                            stored_versions = {
                                str(key): int(value) for key, value in loaded.items()
                            }
                    except (TypeError, ValueError, json.JSONDecodeError):
                        stored_versions = {}
                for env_id in sorted(view.environments):
                    expected_version = self.current_slices[env_id].version
                    if stored_versions.get(env_id) != expected_version:
                        bump = True
                        reasons.append(
                            {
                                "kind": "ENVIRONMENT_SLICE_DRIFT",
                                "environment": env_id,
                                "from_version": stored_versions.get(env_id),
                                "to_version": expected_version,
                            }
                        )
                        affected_keys |= set(view.keys)
            version = state.version + 1 if bump else state.version
        if bump and not reasons:
            reasons.append({"kind": "EXPLICIT_VIEW_BUMP"})
            affected_keys |= set(view.keys)
        propagation: dict[str, Any] | None = None
        if bump and state is not None:
            reasons = _stable_reasons(reasons)
            propagation = {
                "previous_version": state.version,
                "new_version": version,
                "affected_keys": sorted(affected_keys),
                "reasons": reasons,
            }
        summary, dispatches, service_errors = self._compute_view(view, version)
        return self._commit(
            view,
            summary,
            dispatches,
            service_errors,
            propagation,
        )

    def run_all(self) -> EngineRunReport:
        runs: list[ViewRun] = []
        invalidations: list[InvalidationReport] = []
        ledger_states = self.ledger.view_states()
        for view_id in sorted(ledger_states):
            state = ledger_states[view_id]
            if not state.active or view_id in self.views:
                continue
            removed_rows = self.ledger.current_results(view_id)
            affected_keys = tuple(sorted(str(row["key"]) for row in removed_rows))
            reasons = _stable_reasons([{"kind": "VIEW_REMOVED_FROM_SPEC"}])
            new_version = state.version + 1
            self.ledger.mark_view_removed(
                view_id=view_id,
                previous_version=state.version,
                removed_version=new_version,
                config_digest=state.config_digest,
                affected_keys=affected_keys,
                reasons=reasons,
            )
            self.versions[view_id] = new_version
            invalidations.append(
                InvalidationReport(
                    sequence=len(invalidations) + 1,
                    view_id=view_id,
                    previous_version=state.version,
                    new_version=new_version,
                    affected_keys=affected_keys,
                    reasons=tuple(reasons),
                    run=None,
                    removed=True,
                )
            )
        for view_id in sorted(self.views):
            run = self.run_view(view_id)
            runs.append(run)
            if run.propagation is not None:
                invalidations.append(
                    InvalidationReport(
                        sequence=len(invalidations) + 1,
                        view_id=view_id,
                        previous_version=int(run.propagation["previous_version"]),
                        new_version=int(run.propagation["new_version"]),
                        affected_keys=tuple(run.propagation["affected_keys"]),
                        reasons=tuple(run.propagation["reasons"]),
                        run=run,
                    )
                )
        return EngineRunReport(
            views=tuple(runs),
            invalidations=tuple(invalidations),
            start_errors=dict(self.services.start_errors),
            scheduler={
                **self.scheduler.gate.snapshot(),
                "executor_workers": self.scheduler.last_executor_workers,
            },
            ledger=self.ledger.snapshot_report(),
        )

    @staticmethod
    def _slice_value_changes(old: SliceSpec, new: SliceSpec) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for key in sorted(set(old.values) | set(new.values)):
            old_value = old.values.get(key)
            new_value = new.values.get(key)
            if old_value == new_value:
                continue
            if old_value is None:
                kind = "SLICE_KEY_ADDED"
            elif new_value is None:
                kind = "SLICE_KEY_REMOVED"
            else:
                kind = "SLICE_VALUE_CHANGED"
            changes.append(
                {
                    "kind": kind,
                    "key": key,
                    "from_value": old_value,
                    "to_value": new_value,
                }
            )
        return changes

    def change_environment_slice(
        self,
        env_id: str,
        *,
        version: int,
        values: dict[str, Any],
    ) -> list[InvalidationReport]:
        if env_id not in self.current_slices:
            raise SpecError(f"$.environments.{env_id}", "unknown environment")
        old = self.current_slices[env_id]
        if isinstance(version, bool) or not isinstance(version, int):
            raise SpecError(f"$.environments.{env_id}.slices.version", "must be an integer")
        if version <= old.version:
            raise SpecError(
                f"$.environments.{env_id}.slices.version",
                f"must be greater than current version {old.version}",
            )
        new = parse_slice(version, values, f"$.environments.{env_id}.slices")
        changes = self._slice_value_changes(old, new)
        self.current_slices[env_id] = new
        changed_keys = {item["key"] for item in changes}
        referencing: list[tuple[str, list[str]]] = []
        for view_id in sorted(self.views):
            view = self.views[view_id]
            if env_id not in view.environments:
                continue
            keys = sorted(changed_keys.intersection(view.keys))
            referencing.append((view_id, keys))
        reports: list[InvalidationReport] = []
        for view_id, keys in referencing:
            view = self.views[view_id]
            state = self.ledger.view_state(view_id)
            if state is None:
                raise RuntimeError(f"view {view_id!r} has no ledger state")
            if not keys:
                # A version-only or out-of-view change refreshes provenance in the
                # same view version without invalidating any business conclusion.
                summary, dispatches, service_errors = self._compute_view(
                    view, state.version
                )
                self._commit(
                    view,
                    summary,
                    dispatches,
                    service_errors,
                    None,
                )
                continue
            previous_version = state.version
            new_version = previous_version + 1
            reasons = [
                {
                    **change,
                    "environment": env_id,
                    "from_version": old.version,
                    "to_version": new.version,
                }
                for change in changes
                if change["key"] in keys
            ]
            reasons = _stable_reasons(reasons)
            propagation = {
                "previous_version": previous_version,
                "new_version": new_version,
                "affected_keys": keys,
                "reasons": reasons,
            }
            summary, dispatches, service_errors = self._compute_view(view, new_version)
            run = self._commit(
                view,
                summary,
                dispatches,
                service_errors,
                propagation,
            )
            reports.append(
                InvalidationReport(
                    sequence=len(reports) + 1,
                    view_id=view_id,
                    previous_version=previous_version,
                    new_version=new_version,
                    affected_keys=tuple(keys),
                    reasons=tuple(reasons),
                    run=run,
                )
            )
        return reports

    @staticmethod
    def _view_change_reasons(old: ViewSpec | None, new: ViewSpec) -> list[dict[str, Any]]:
        if old is None:
            return [{"kind": "VIEW_ADDED"}]
        reasons: list[dict[str, Any]] = []
        old_envs = set(old.environments)
        new_envs = set(new.environments)
        for env_id in sorted(new_envs - old_envs):
            reasons.append({"kind": "PARTICIPANT_ADDED", "environment": env_id})
        for env_id in sorted(old_envs - new_envs):
            reasons.append({"kind": "PARTICIPANT_REMOVED", "environment": env_id})
        for key in sorted(set(old.keys) | set(new.keys)):
            old_rule = old.keys.get(key)
            new_rule = new.keys.get(key)
            if old_rule is None:
                reasons.append({"kind": "VIEW_KEY_ADDED", "key": key})
                continue
            if new_rule is None:
                reasons.append({"kind": "VIEW_KEY_REMOVED", "key": key})
                continue
            if old_rule.mode != new_rule.mode:
                reasons.append(
                    {
                        "kind": "ARBITRATION_MODE_CHANGED",
                        "key": key,
                        "from": old_rule.mode,
                        "to": new_rule.mode,
                    }
                )
            if old_rule.tolerance != new_rule.tolerance:
                reasons.append(
                    {
                        "kind": "TOLERANCE_CHANGED",
                        "key": key,
                        "from": old_rule.tolerance,
                        "to": new_rule.tolerance,
                    }
                )
        return _stable_reasons(reasons)

    def upsert_view(self, view: ViewSpec) -> InvalidationReport | None:
        for env_id in view.environments:
            if env_id not in self.current_slices:
                raise SpecError(f"$.views.{view.id}.environments", f"unknown environment {env_id!r}")
        old = self.views.get(view.id)
        state = self.ledger.view_state(view.id)
        if old is not None and old.config_digest() == view.config_digest() and state and state.active:
            return None
        reasons = self._view_change_reasons(old, view)
        if old is None and state is not None:
            reasons = _stable_reasons([{"kind": "VIEW_REACTIVATED"}, *reasons])
        if not reasons:
            reasons = [{"kind": "VIEW_CHANGED"}]
        previous_version = state.version if state is not None else 0
        new_version = previous_version + 1 if state is not None else 1
        old_keys = set(old.keys) if old is not None else set()
        new_keys = set(view.keys)
        participants_changed = old is None or set(old.environments) != set(view.environments)
        if participants_changed:
            affected_keys = sorted(old_keys | new_keys)
        else:
            affected_keys = sorted(
                {
                    str(reason.get("key"))
                    for reason in reasons
                    if reason.get("key") is not None
                }
            )
        propagation = {
            "previous_version": previous_version,
            "new_version": new_version,
            "affected_keys": affected_keys,
            "reasons": reasons,
        }
        self.views[view.id] = view
        summary, dispatches, service_errors = self._compute_view(view, new_version)
        run = self._commit(view, summary, dispatches, service_errors, propagation)
        return InvalidationReport(
            sequence=1,
            view_id=view.id,
            previous_version=previous_version,
            new_version=new_version,
            affected_keys=tuple(affected_keys),
            reasons=tuple(reasons),
            run=run,
        )

    def remove_view(self, view_id: str) -> InvalidationReport | None:
        view = self.views.get(view_id)
        state = self.ledger.view_state(view_id)
        if view is None or state is None or not state.active:
            return None
        new_version = state.version + 1
        affected_keys = sorted(view.keys)
        reasons = [{"kind": "VIEW_REMOVED"}]
        self.ledger.mark_view_removed(
            view_id=view_id,
            previous_version=state.version,
            removed_version=new_version,
            config_digest=view.config_digest(),
            affected_keys=affected_keys,
            reasons=reasons,
        )
        del self.views[view_id]
        self.versions[view_id] = new_version
        self.latest_runs.pop(view_id, None)
        return InvalidationReport(
            sequence=1,
            view_id=view_id,
            previous_version=state.version,
            new_version=new_version,
            affected_keys=tuple(affected_keys),
            reasons=tuple(reasons),
            run=None,
            removed=True,
        )

    def current_display(self) -> dict[str, Any]:
        views: dict[str, Any] = {}
        for view_id in sorted(self.views):
            rows = self.ledger.current_results(view_id)
            views[view_id] = [
                {
                    "key": row["key"],
                    "view_version": int(row["view_version"]),
                    "status": row["status"],
                    "summary": json.loads(row["result_json"]).get("summary"),
                    "authoritative_digest": row["summary_digest"],
                    "adopted": json.loads(row["adopted_json"]),
                    "rejected": json.loads(row["rejected_json"]),
                    "error_bound": json.loads(row["bound_json"]),
                    "attribution": json.loads(row["attribution_json"]),
                }
                for row in rows
            ]
        return {"views": views, "ledger_stale_excluded": True}

    def replay_current(self) -> dict[str, Any]:
        expected: dict[tuple[str, str], KeyResult] = {}
        versions: dict[str, int] = {}
        for view_id in sorted(self.views):
            state = self.ledger.view_state(view_id)
            version = state.version if state is not None and state.active else 1
            versions[view_id] = version
            summary, _, _ = self._compute_view(self.views[view_id], version)
            for result in summary.key_results:
                expected[(view_id, result.key)] = result
        return self.ledger.replay(expected, versions)


def _stable_reasons(reasons: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for reason in reasons:
        key = canonical_json(reason)
        unique[key] = reason
    return sorted(
        unique.values(),
        key=lambda reason: (
            str(reason.get("kind", "")),
            str(reason.get("environment", "")),
            str(reason.get("key", "")),
            canonical_json(reason),
        ),
    )


def _json_object_or_empty(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_engine(spec_path: str | Path, ledger_path: str | Path) -> tuple[ArbiterEngine, Any]:
    from .spec import load_spec

    spec = load_spec(spec_path)
    ledger = Ledger(ledger_path)
    return ArbiterEngine(spec, ledger), spec
