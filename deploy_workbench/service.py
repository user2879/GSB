from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json
from .engine import resolve_all, resolve_environment
from .errors import E_REPLAY, WorkbenchError
from .fixed import FixedQuantizer
from .ledger import Ledger
from .model import VersionState, WorkbenchSpec
from .snapshot import (
    build_snapshot,
    compare_replay,
    read_snapshot,
    snapshot_versions,
    write_snapshot,
)
from .spec import load_spec


class WorkbenchService:
    """Coordinates declaration versions, append-only ledger, and snapshots."""

    def __init__(
        self,
        spec_path: str | Path,
        state_root: str | Path,
        *,
        scale_power_override: int | str | None = None,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.state_root = Path(state_root)
        self.scale_power_override = scale_power_override
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.state_root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.spec = load_spec(self.spec_path, scale_power_override=scale_power_override)
        self.ledger = Ledger(self.state_root / "ledger.sqlite3")
        self.recovery_report = self.ledger.recover_incomplete()
        self.versions = self.ledger.plan_versions(self.spec.fingerprints())
        self.pending_version_change = self.ledger.apply_pending_versions(self.versions)

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "WorkbenchService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def latest_snapshot_path(self) -> Path:
        return self.snapshot_dir / "latest.json"

    def _idempotent_run(self) -> dict[str, Any] | None:
        run = self.ledger.current_run()
        if run is None or run["status"] != "complete":
            return None
        if tuple(
            int(run[field])
            for field in ("environment_version", "priority_version", "scale_version")
        ) != self.versions.version_tuple():
            return None
        if run["spec_digest"] != self.versions.spec_digest:
            return None
        payload = self.ledger.run_payload(run)
        return {
            "status": "unchanged",
            "run_id": int(run["run_id"]),
            "versions": self.versions.payload(),
            "completed_environments": sorted(self.spec.environments),
            "unknown_environments": [],
            "stats": payload["stats"],
            "snapshot_path": str(self.latest_snapshot_path),
        }

    def _old_environment_payloads(self) -> dict[str, dict[str, Any]]:
        run = self.ledger.latest_completed_run()
        if run is None:
            return {}
        return {
            record["env_id"]: record["payload"]
            for record in self.ledger.fetch_run_environments(int(run["run_id"]))
        }

    def _affected_stats(
        self,
        old: Mapping[str, Mapping[str, Any]],
        new: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        affected_environments: list[str] = []
        affected_keys: list[dict[str, str]] = []
        old_env_ids = set(old)
        new_env_ids = set(new)
        for env_id in sorted(old_env_ids - new_env_ids):
            affected_environments.append(env_id)
            for key in sorted(old[env_id].get("keys", {})):
                affected_keys.append({"environment": env_id, "key": key})
        for env_id in sorted(new_env_ids):
            new_env = new[env_id]
            old_env = old.get(env_id)
            if old_env is None:
                affected_environments.append(env_id)
                for key in sorted(new_env.get("keys", {})):
                    affected_keys.append({"environment": env_id, "key": key})
                continue
            env_changed = canonical_json(old_env) != canonical_json(new_env)
            if env_changed:
                affected_environments.append(env_id)
            old_keys = old_env.get("keys", {})
            new_keys = new_env.get("keys", {})
            for key in sorted(set(old_keys) | set(new_keys)):
                if canonical_json(old_keys.get(key)) != canonical_json(new_keys.get(key)):
                    affected_keys.append({"environment": env_id, "key": key})
            old_tasks = old_env.get("tasks", {})
            new_tasks = new_env.get("tasks", {})
            for task_id in sorted(set(old_tasks) | set(new_tasks)):
                if canonical_json(old_tasks.get(task_id)) != canonical_json(new_tasks.get(task_id)):
                    referenced = new_tasks.get(task_id, {}).get("referenced_keys", [])
                    if not referenced:
                        referenced = old_tasks.get(task_id, {}).get("referenced_keys", [])
                    for key in referenced:
                        pair = {"environment": env_id, "key": str(key)}
                        if pair not in affected_keys:
                            affected_keys.append(pair)
        affected_environments = sorted(set(affected_environments))
        affected_keys = sorted(affected_keys, key=lambda item: (item["environment"], item["key"]))
        return {
            "affected_environment_count": len(affected_environments),
            "affected_key_count": len(affected_keys),
            "affected_environments": affected_environments,
            "affected_keys": affected_keys,
        }

    def run_all(self, *, force: bool = False) -> dict[str, Any]:
        if not force:
            existing = self._idempotent_run()
            if existing is not None:
                return existing

        latest_plan = self.ledger.plan_versions(self.spec.fingerprints())
        if latest_plan.payload() != self.versions.payload():
            self.versions = latest_plan
            self.pending_version_change = self.ledger.apply_pending_versions(self.versions)

        old_payloads = self._old_environment_payloads()
        expected_envs = [env.id for env in self.spec.environment_order()]
        run_id = self.ledger.begin_run(
            self.versions,
            expected_envs,
            {"state": "computing", "affected_environment_count": None, "affected_key_count": None},
        )
        quantizer = FixedQuantizer(self.spec.scale_power, self.spec.overflow_policy)
        completed: dict[str, dict[str, Any]] = {}
        try:
            for sequence, env in enumerate(self.spec.environment_order(), start=1):
                self.ledger.mark_computing(run_id, env.id)
                result = resolve_environment(self.spec, env, quantizer)
                self.ledger.commit_environment(
                    run_id,
                    env.id,
                    result,
                    self.versions,
                    sequence=sequence,
                )
                completed[env.id] = result
            final = self.ledger.finalize_run(run_id)
            stats = self._affected_stats(old_payloads, completed)
            stats["state"] = final["status"]
            self.ledger.update_run_stats(run_id, stats)
        except BaseException:
            try:
                self.ledger.finalize_run(run_id)
            except Exception:
                pass
            raise

        if final["status"] != "complete":
            return {
                "status": final["status"],
                "run_id": run_id,
                "versions": self.versions.payload(),
                "completed_environments": final["completed_environments"],
                "unknown_environments": sorted(set(expected_envs) - set(completed)),
                "stats": stats,
                "snapshot_path": None,
            }

        snapshot = build_snapshot(
            self.spec,
            self.versions,
            run_id=run_id,
            environments=completed,
            completed_environments=completed,
            stats=stats,
        )
        history_path = self.snapshot_dir / f"snapshot-run-{run_id}.json"
        write_snapshot(history_path, snapshot)
        write_snapshot(self.latest_snapshot_path, snapshot)
        return {
            "status": "complete",
            "run_id": run_id,
            "versions": self.versions.payload(),
            "completed_environments": final["completed_environments"],
            "unknown_environments": [],
            "stats": stats,
            "snapshot_path": str(self.latest_snapshot_path),
            "snapshot_history_path": str(history_path),
            "ledger_content_digest": final["content_digest"],
        }

    def get_current_view(self) -> dict[str, Any]:
        view = self.ledger.current_view()
        current_run = view["run"]
        stale = current_run is None
        stale_reason = "no_current_conclusion" if stale else None
        if current_run is not None:
            run_versions = (
                current_run["environment_version"],
                current_run["priority_version"],
                current_run["scale_version"],
            )
            if run_versions != self.versions.version_tuple():
                stale = True
                stale_reason = "version_changed"
                view = {
                    "run": current_run,
                    "completed_environments": [],
                    "unknown_environments": [],
                    "environments": [],
                }
            elif current_run["spec_digest"] != self.versions.spec_digest:
                stale = True
                stale_reason = "declaration_changed"
                view = {
                    "run": current_run,
                    "completed_environments": [],
                    "unknown_environments": [],
                    "environments": [],
                }
        view["stale"] = stale
        view["stale_reason"] = stale_reason
        view["current_versions"] = self.versions.payload()
        view["pending_version_change"] = self.pending_version_change
        view["recovery"] = self.recovery_report
        if not stale and current_run is not None:
            view["complete"] = current_run["status"] == "complete"
        else:
            view["complete"] = False
        return view

    def get_historical_view(self, run_id: int) -> dict[str, Any]:
        run = self.ledger.get_run(run_id)
        if run is None:
            raise WorkbenchError(E_REPLAY, "requested ledger run does not exist", path="$", details={"run_id": run_id})
        return {
            "run": self.ledger.run_payload(run),
            "environments": self.ledger.fetch_run_environments(run_id),
            "read_only": True,
        }

    def list_versions(self) -> list[dict[str, Any]]:
        versions = self.ledger.list_runs()
        current_tuple = self.versions.version_tuple()
        for item in versions:
            item_tuple = (item["environment_version"], item["priority_version"], item["scale_version"])
            item["stale"] = (not item["is_current"]) or item_tuple != current_tuple or item["spec_digest"] != self.versions.spec_digest
            item["read_only"] = item["stale"]
        return versions

    def recover(self) -> dict[str, Any]:
        report = self.ledger.recover_incomplete()
        self.recovery_report = report
        return report

    def replay_snapshot(self, path: str | Path | None = None, *, rebuild_on_mismatch: bool = True) -> dict[str, Any]:
        snapshot_path = Path(path) if path is not None else self.latest_snapshot_path
        snapshot = read_snapshot(snapshot_path)
        recorded_versions = snapshot_versions(snapshot)
        mismatch_reasons: list[str] = []
        if recorded_versions != self.versions.version_tuple():
            mismatch_reasons.append("version_tuple")
        if snapshot.get("spec_digest") != self.spec.content_digest():
            mismatch_reasons.append("spec_digest")
        if int(snapshot.get("scale_power", -1)) != self.spec.scale_power:
            mismatch_reasons.append("scale_power")
        if mismatch_reasons:
            replacement = self.run_all(force=True) if rebuild_on_mismatch else None
            return {
                "status": "replaced" if rebuild_on_mismatch else "version_mismatch",
                "reason": "snapshot version differs from current declaration",
                "mismatch_fields": mismatch_reasons,
                "snapshot_versions": {
                    "environment_version": recorded_versions[0],
                    "priority_version": recorded_versions[1],
                    "scale_version": recorded_versions[2],
                },
                "current_versions": self.versions.version_tuple(),
                "replacement": replacement,
            }

        quantizer = FixedQuantizer(self.spec.scale_power, self.spec.overflow_policy)
        replay = resolve_all(self.spec, quantizer)
        result = compare_replay(self.spec, snapshot, replay)
        result["snapshot_path"] = str(snapshot_path)
        return result

