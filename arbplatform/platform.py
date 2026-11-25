"""Top-level orchestration for the multi-environment arbitration platform."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .client import ServiceSupervisor
from .determinism import digest_of, now_ms
from .engine import ArbitrationEngine, KeyVerdict, ViewResult
from .errors import StateError
from .ledger import Ledger, ReplayOutcome
from .planner import ScaleParameters, choose_path
from .propagation import Change, propagate
from .scheduler import QuotaScheduler
from .spec import Manifest, ViewSpec


@dataclass
class ArbitrationReport:
    view_id: str
    view_version: int
    path_decision: dict[str, Any]
    result: ViewResult
    propagation: dict[str, Any] | None
    service_replies: dict[str, dict[str, Any]] = field(default_factory=dict)
    scheduling: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "view_version": self.view_version,
            "path_decision": self.path_decision,
            "result": self.result.to_record(),
            "propagation": self.propagation,
            "service_replies": dict(sorted(self.service_replies.items())),
            "scheduling": self.scheduling,
        }


class ArbitrationPlatform:
    def __init__(self, manifest: Manifest, db_path: str | Path,
                 supervisor: ServiceSupervisor | None = None) -> None:
        self.manifest = manifest
        self.engine = ArbitrationEngine(manifest)
        self.supervisor = supervisor or ServiceSupervisor()
        self.ledger = Ledger(db_path, manifest)
        self.scheduler = QuotaScheduler(manifest)
        # obs_id -> observation; order independent, all output sorts explicitly.
        self.store: dict[str, dict[str, Any]] = {}
        self.view_versions: dict[str, int] = {}
        self._last_results: dict[str, ViewResult] = {}

    #-- lifecycle --------------------------------------------------------
    def open_snapshot(self) -> ReplayOutcome:
        outcome = self.ledger.open_or_replace()
        for view_id, version in self.ledger.latest_committed().items():
            self.view_versions[view_id] = version["view_version"]
        return outcome

    def start_services(self) -> dict[str, Any]:
        info = {}
        for env in self.manifest.environments:
            handle = self.supervisor.start(env.env_id, env.host, env.port)
            info[env.env_id] = {"host": handle.host, "port": handle.port}
        return info

    def shutdown(self) -> None:
        self.supervisor.stop_all()
        self.ledger.close()

    #-- ingestion --------------------------------------------------------
    def ingest(self, observations: Iterable[dict[str, Any]],
               batch_size: int = 4) -> dict[str, Any]:
        """Send observations to environment services in deterministic batches."""
        by_env: dict[str, list[dict[str, Any]]] = {}
        for obs in observations:
            self.store[obs["obs_id"]] = obs
            by_env.setdefault(obs["env_id"], []).append(obs)
        replies: dict[str, list[dict[str, Any]]] = {}
        for env_id in sorted(by_env):
            items = sorted(by_env[env_id],
                           key=lambda o: (o["key"], o["seq"], o["instance_id"]))
            env_replies: list[dict[str, Any]] = []
            for start in range(0, len(items), max(1, batch_size)):
                batch = items[start:start + max(1, batch_size)]
                reply = self.supervisor.request(
                    env_id, {"type": "batch", "observations": batch}
                )
                env_replies.append(reply)
            replies[env_id] = env_replies
        return {"envs": dict(sorted(replies.items())),
                "stored_total": len(self.store)}

    def replace_observation(self, obs_id: str, new_value: Any) -> Change:
        if obs_id not in self.store:
            raise StateError(f"unknown observation {obs_id}", field=f"obs.{obs_id}")
        old = self.store[obs_id]
        updated = dict(old)
        updated["value"] = new_value
        self.store[obs_id] = updated
        return Change(change_type="observation_replaced",
                      env_id=old["env_id"], key=old["key"],
                      detail={"obs_id": obs_id})

    def delete_observation(self, obs_id: str) -> Change:
        if obs_id not in self.store:
            raise StateError(f"unknown observation {obs_id}", field=f"obs.{obs_id}")
        old = self.store.pop(obs_id)
        return Change(change_type="observation_deleted",
                      env_id=old["env_id"], key=old["key"],
                      detail={"obs_id": obs_id})

    #-- arbitration ------------------------------------------------------
    def run_view(self, view_id: str,
                 changes: list[Change] | None = None,
                 affected_keys: tuple[str, ...] | None = None,
                 use_scheduler: bool = True,
                 force_path: str | None = None) -> ArbitrationReport:
        view = self.manifest.view(view_id)
        observations = list(self.store.values())
        plan = None
        only_keys: tuple[str, ...] | None = None
        previous: dict[str, KeyVerdict] | None = None

        if changes:
            plan = propagate(self.manifest, self.view_versions, changes)
            if view_id in plan.view_versions:
                self.view_versions[view_id] = plan.view_versions[view_id]
            step = next((s for s in plan.steps if s.view_id == view_id), None)
            if step is not None:
                only_keys = step.affected_keys
        version = self.view_versions.get(view_id, 0)
        if version == 0:
            version = 1
            self.view_versions[view_id] = 1
        if only_keys is not None and view_id in self._last_results:
            previous = dict(self._last_results[view_id].verdicts)
        if affected_keys is not None:
            only_keys = tuple(sorted(affected_keys))

        changed_count = self._count_changed(changes)
        params = ScaleParameters(
            env_count=len(view.env_refs),
            key_count=len(view.policies),
            observation_count=sum(
                1 for o in observations if o["env_id"] in set(view.env_refs)
            ),
            changed_observation_count=changed_count,
            affected_key_count=len(only_keys) if only_keys else 0,
        )
        decision = choose_path(params)
        is_first_run = view_id not in self._last_results
        chosen_path = force_path or decision.path
        if is_first_run and force_path is None:
            chosen_path = "full"
            decision.path = "full"
            decision.reason = (
                "first arbitration for the view requires a full baseline; "
                + decision.reason
            )
        if force_path is not None:
            decision.path = chosen_path
            decision.reason = (
                f"path overridden to {force_path} for scheduling verification; "
                + decision.reason
            )

        self.ledger.begin_view(view_id, version, now_ms())
        if use_scheduler and chosen_path == "incremental":
            result, scheduling = self._run_with_scheduler(
                view, version, observations, only_keys, previous)
        else:
            result = self.engine.arbitrate_view(
                view, version, observations, only_keys, previous)
            scheduling = None
        self.ledger.commit_view(result, now_ms())
        self._last_results[view_id] = result

        replies = self._collect_service_replies(view)
        return ArbitrationReport(
            view_id=view_id, view_version=version,
            path_decision=decision.as_dict(), result=result,
            propagation=plan.as_dict() if plan else None,
            service_replies=replies, scheduling=scheduling,
        )

    def _run_with_scheduler(self, view: ViewSpec, version: int,
                            observations: list[dict[str, Any]],
                            only_keys: tuple[str, ...] | None,
                            previous: dict[str, KeyVerdict] | None
                            ) -> tuple[ViewResult, dict[str, Any]]:
        keys = sorted(view.policies) if only_keys is None else sorted(only_keys)
        tasks: list[tuple[str, str, str, Any]] = []
        for env_id in view.env_refs:
            for key in keys:
                task_id = f"{view.view_id}:{env_id}:{key}"
                tasks.append((
                    task_id, env_id, key,
                    lambda eid=env_id, k=key:
                    self.engine.build_chain_for_env(observations, eid, k),
                ))
        records = self.scheduler.run(tasks)
        # Chains built under quotas; deterministic merge happens on the parent.
        chains_by_key: dict[str, list[Any]] = {key: [] for key in keys}
        for record in records:
            if record.error:
                raise StateError(record.error, field=f"task.{record.task_id}")
            chains_by_key[record.key].append(record.result)
        verdicts = dict(previous) if previous else {}
        for key in keys:
            chains = sorted(chains_by_key[key], key=lambda c: c.env_id)
            verdicts[key] = self.engine.arbitrate_key(view, key, chains)
        ordered = [verdicts[k].to_record() for k in sorted(verdicts)]
        result = ViewResult(
            view_id=view.view_id, view_version=version,
            scale_power=self.manifest.scale_power, verdicts=verdicts,
            content_digest=digest_of(
                {"view_id": view.view_id, "view_version": version,
                 "scale_power": self.manifest.scale_power,
                 "verdicts": ordered}),
        )
        return result, self.scheduler.stats.as_dict()

    def _collect_service_replies(self, view: ViewSpec) -> dict[str, Any]:
        keys = sorted(view.policies)
        out: dict[str, Any] = {}
        for env_id in view.env_refs:
            out[env_id] = self.supervisor.request(
                env_id, {"type": "arbitrate", "keys": keys}
            )
        return out

    @staticmethod
    def _count_changed(changes: list[Change] | None) -> int:
        if not changes:
            return 0
        return sum(
            1 for c in changes
            if c.change_type in ("observation_replaced", "observation_deleted")
        )

    #-- recovery ---------------------------------------------------------
    def force_kill_service(self, env_id: str) -> int:
        return self.supervisor.force_kill(env_id)

    def restart_service(self, env_id: str) -> dict[str, Any]:
        env = self.manifest.env(env_id)
        handle = self.supervisor.start(env.env_id, env.host, env.port)
        # Recovery only uses fully committed ledger rows; repush current store.
        items = sorted(
            (o for o in self.store.values() if o["env_id"] == env_id),
            key=lambda o: (o["key"], o["seq"], o["instance_id"]),
        )
        replies = []
        for start in range(0, len(items), 4):
            replies.append(self.supervisor.request(
                env_id, {"type": "batch",
                         "observations": items[start:start + 4]}))
        return {"env_id": env_id, "port": handle.port,
                "repushed": len(items), "batch_replies": replies}

    def recover_after_kill(self) -> dict[str, Any]:
        unknown = self.ledger.mark_unknown_after_kill()
        return {"unknown_views": unknown,
                "action": "rerun required for uncommitted views"}

    def replay_ledger(self) -> ReplayOutcome:
        recomputed: dict[str, ViewResult] = {}
        for view in self.manifest.views:
            version = self.view_versions.get(view.view_id, 1)
            recomputed[view.view_id] = self.engine.arbitrate_view(
                view, version, list(self.store.values()))
        return self.ledger.replay(recomputed)

    def last_result(self, view_id: str) -> ViewResult:
        if view_id not in self._last_results:
            raise StateError(f"no result yet for view {view_id}",
                             field=f"views.{view_id}")
        return self._last_results[view_id]
