"""Invalidation propagation along env -> key -> view reference edges."""
from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Declarations


@dataclass(frozen=True)
class ChangeSet:
    added_envs: tuple[str, ...] = ()
    removed_envs: tuple[str, ...] = ()
    replaced_obs_envs: tuple[str, ...] = ()
    changed_policies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # changed_policies: view_id -> tuple of keys whose policy changed


@dataclass(frozen=True)
class PropagationStep:
    step_no: int
    source: str
    edge: str
    target: str
    basis: str


@dataclass(frozen=True)
class ViewInvalidation:
    view_id: str
    new_version: int
    dirty_keys: tuple[str, ...]
    steps: tuple[PropagationStep, ...]


class PropagationEngine:
    def __init__(self, decls: Declarations):
        self.decls = decls
        # view versions are monotonically increasing integers per view
        self._versions: dict[str, int] = {view_id: 0 for view_id in decls.views}
        self._env_keys: dict[str, set[str]] = {
            env_id: set(decls.keys) for env_id in decls.environments
        }

    def current_versions(self) -> dict[str, int]:
        return dict(self._versions)

    def propagate(self, change: ChangeSet) -> list[ViewInvalidation]:
        affected_envs = sorted(
            set(change.added_envs)
            | set(change.removed_envs)
            | set(change.replaced_obs_envs)
        )
        results: list[ViewInvalidation] = []
        step_counter = 0
        for view_id in sorted(self.decls.views):
            view = self.decls.views[view_id]
            steps: list[PropagationStep] = []
            dirty: set[str] = set()
            view_env_set = set(view.envs)

            for env_id in affected_envs:
                # Only views currently referencing the env are affected. A removed
                # env invalidates its former participating views; views that never
                # referenced it stay untouched.
                if env_id not in view_env_set:
                    continue
                step_counter += 1
                if env_id in change.added_envs:
                    basis = "environment added to declaration"
                elif env_id in change.removed_envs:
                    basis = "environment removed from declaration"
                else:
                    basis = "observations replaced for environment"
                steps.append(
                    PropagationStep(step_counter, env_id, "env->view", view_id, basis)
                )
                if env_id in view_env_set:
                    for key in sorted(self._env_keys.get(env_id, set(self.decls.keys))):
                        step_counter += 1
                        steps.append(
                            PropagationStep(
                                step_counter, env_id, "env->key", key,
                                "environment participates in view; key chain touched",
                            )
                        )
                        dirty.add(key)

            for changed_view_id, keys in change.changed_policies.items():
                if changed_view_id != view_id:
                    continue
                for key in sorted(keys):
                    step_counter += 1
                    steps.append(
                        PropagationStep(
                            step_counter, view_id, "policy->key", key,
                            "arbitration policy changed for key",
                        )
                    )
                    dirty.add(key)

            if dirty:
                self._versions[view_id] += 1
                results.append(
                    ViewInvalidation(
                        view_id=view_id,
                        new_version=self._versions[view_id],
                        dirty_keys=tuple(sorted(dirty)),
                        steps=tuple(steps),
                    )
                )
        return results

    def mark_view_dirty_all(self, view_id: str) -> ViewInvalidation:
        self._versions[view_id] += 1
        return ViewInvalidation(
            view_id=view_id,
            new_version=self._versions[view_id],
            dirty_keys=tuple(self.decls.keys),
            steps=(
                PropagationStep(0, view_id, "view->all_keys", "*",
                                "full view invalidation requested"),
            ),
        )
