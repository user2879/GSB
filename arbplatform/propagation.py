"""Invalidation propagation and affected-subset recomputation planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spec import Manifest

CHANGE_OBS_REPLACE = "observation_replaced"
CHANGE_OBS_DELETE = "observation_deleted"
CHANGE_VIEW_ENVS = "view_envs_changed"
CHANGE_VIEW_POLICY = "view_policy_changed"
CHANGE_VIEW_ADDED = "view_added"
CHANGE_VIEW_REMOVED = "view_removed"
CHANGE_SCALE = "scale_power_changed"


@dataclass
class Change:
    change_type: str
    env_id: str | None = None
    key: str | None = None
    view_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationStep:
    order: int
    view_id: str
    basis: str
    affected_keys: tuple[str, ...]
    from_version: int
    to_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "view_id": self.view_id,
            "basis": self.basis,
            "affected_keys": list(self.affected_keys),
            "from_version": self.from_version,
            "to_version": self.to_version,
        }


@dataclass
class PropagationPlan:
    steps: list[PropagationStep]
    affected_views: tuple[str, ...]
    view_versions: dict[str, int]
    removed_views: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "affected_views": list(self.affected_views),
            "view_versions": dict(sorted(self.view_versions.items())),
            "removed_views": list(self.removed_views),
        }


def propagate(manifest: Manifest, current_versions: dict[str, int],
              changes: list[Change]) -> PropagationPlan:
    """Derive affected views/keys and monotonically bump view versions.

    Output ordering is stable: views are processed by view id and keys are
    sorted. Every step records the exact propagation basis so an auditor can
    reproduce why each key was invalidated.
    """
    versions = dict(current_versions)
    removed: set[str] = set()
    # view -> (set of affected keys, set of basis strings)
    affected: dict[str, tuple[set[str], list[str]]] = {}

    def touch(view_id: str, keys: set[str], basis: str) -> None:
        keyset, bases = affected.setdefault(view_id, (set(), []))
        keyset.update(keys)
        bases.append(basis)

    all_view_ids = sorted(v.view_id for v in manifest.views)
    view_by_id = {v.view_id: v for v in manifest.views}

    for change in sorted(changes, key=lambda c: (
        c.change_type, c.view_id or "", c.env_id or "", c.key or ""
    )):
        if change.change_type == CHANGE_VIEW_REMOVED:
            removed.add(change.view_id)  # type: ignore[arg-type]
            continue
        if change.change_type == CHANGE_VIEW_ADDED:
            vid = change.view_id  # type: ignore[assignment]
            keys = set(view_by_id[vid].policies)
            touch(vid, keys, "view added: all declared keys invalidated")
            continue
        if change.change_type == CHANGE_SCALE:
            for vid in all_view_ids:
                touch(vid, set(view_by_id[vid].policies),
                      "session scale power changed: all keys invalidated")
            continue
        if change.change_type == CHANGE_VIEW_ENVS:
            vid = change.view_id  # type: ignore[assignment]
            touch(vid, set(view_by_id[vid].policies),
                  f"participating env set changed for view {vid}")
            continue
        if change.change_type == CHANGE_VIEW_POLICY:
            vid = change.view_id  # type: ignore[assignment]
            key = change.key  # type: ignore[assignment]
            touch(vid, {key}, f"arbitration policy changed for key {key}")
            continue
        if change.change_type in (CHANGE_OBS_REPLACE, CHANGE_OBS_DELETE):
            env_id = change.env_id
            key = change.key
            for view in manifest.views:
                if env_id in view.env_refs and key in view.policies:
                    touch(view.view_id, {key},  # type: ignore[arg-type]
                          f"env {env_id} observation {change.change_type} for key {key}")
            continue

    steps: list[PropagationStep] = []
    order = 0
    for vid in sorted(affected):
        if vid in removed:
            continue
        order += 1
        from_version = versions.get(vid, 0)
        to_version = from_version + 1
        keyset, bases = affected[vid]
        steps.append(
            PropagationStep(
                order=order, view_id=vid,
                basis=" | ".join(bases),
                affected_keys=tuple(sorted(keyset)),
                from_version=from_version, to_version=to_version,
            )
        )
        versions[vid] = to_version

    affected_views = tuple(sorted(v for v in affected if v not in removed))
    return PropagationPlan(
        steps=steps,
        affected_views=affected_views,
        view_versions=versions,
        removed_views=tuple(sorted(removed)),
    )
