from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes
from .clock import monotonic_ms
from .diagnostics import build_replay_view
from .errors import DivergeError
from .ledger import first_difference
from .models import Spec
from .topology import stable_topological_order


@dataclass(frozen=True, slots=True)
class InvalidationPlan:
    change_type: str
    direct_shards: tuple[str, ...]
    invalid_shards: tuple[str, ...]
    reasons: dict[str, str]
    propagation_steps: tuple[dict[str, str], ...]
    invalid_scopes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "direct_shards": list(self.direct_shards),
            "invalid_shards": list(self.invalid_shards),
            "reasons": dict(self.reasons),
            "propagation_steps": [dict(step) for step in self.propagation_steps],
            "invalid_scopes": list(self.invalid_scopes),
        }


def plan_change(spec: Spec, change: dict[str, Any]) -> InvalidationPlan:
    change_type = str(change.get("type", ""))
    topo = stable_topological_order(spec.shards)
    shard_map = {shard.shard_id: shard for shard in spec.shards}
    dependents: dict[str, list[str]] = defaultdict(list)
    for shard in spec.shards:
        for dependency in shard.dependencies:
            dependents[dependency].append(shard.shard_id)

    direct: set[str] = set()
    direct_reason = ""
    scopes = ("summary", "lease", "water", "decision")

    if change_type == "capacity_version":
        node_id = str(change["node_id"])
        direct = {shard.shard_id for shard in spec.shards if shard.node_id == node_id}
        direct_reason = f"node_reference:{node_id}:capacity_version"
        scopes = ("summary", "lease", "water", "decision")
    elif change_type == "resource_units":
        shard_id = str(change["shard_id"])
        if shard_id not in shard_map:
            raise KeyError(shard_id)
        direct = {shard_id}
        direct_reason = f"shard_field:{shard_id}:resource_units"
        scopes = ("resource", "water", "budget_decision", "schedule")
    elif change_type == "scale_factor":
        direct = set(shard_map)
        direct_reason = "global_field:scale_power"
        scopes = ("water", "error_attribution", "decision")
    elif change_type == "revision_chain":
        node_id = change.get("node_id")
        if node_id is None:
            direct = set(shard_map)
            direct_reason = "global_field:revision_chain"
        else:
            direct = {
                shard.shard_id for shard in spec.shards if shard.node_id == str(node_id)
            }
            direct_reason = f"node_reference:{node_id}:revision_chain"
        scopes = ("water", "revision_gate", "decision")
    else:
        raise ValueError(f"unsupported change type {change_type!r}")

    invalid = set(direct)
    reasons = {shard_id: direct_reason for shard_id in direct}
    steps: list[dict[str, str]] = []
    queue = deque(sorted(direct))
    while queue:
        current = queue.popleft()
        for dependent in sorted(dependents.get(current, ())):
            if dependent not in invalid:
                invalid.add(dependent)
                reason = f"dependency:{current}->{dependent}"
                reasons[dependent] = reason
                steps.append(
                    {
                        "from_shard_id": current,
                        "shard_id": dependent,
                        "kind": "propagated",
                        "basis": reason,
                    }
                )
                queue.append(dependent)

    ordered_invalid = tuple(shard_id for shard_id in topo if shard_id in invalid)
    ordered_steps = tuple(
        sorted(
            steps,
            key=lambda step: (
                topo.index(step["shard_id"]),
                step["from_shard_id"],
            ),
        )
    )
    return InvalidationPlan(
        change_type=change_type,
        direct_shards=tuple(shard_id for shard_id in topo if shard_id in direct),
        invalid_shards=ordered_invalid,
        reasons=reasons,
        propagation_steps=ordered_steps,
        invalid_scopes=scopes,
    )


class ConclusionStore:
    """Keeps current bytes and stale generations excluded from current views."""

    def __init__(self) -> None:
        self.current: dict[str, dict[str, Any]] = {}
        self.stale: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def put_current(self, shard_id: str, record: dict[str, Any]) -> bytes:
        self.current[shard_id] = {
            "record": record,
            "status": "current",
            "updated_at_ms": monotonic_ms(),
        }
        return self.current_bytes(shard_id)

    def current_record(self, shard_id: str) -> dict[str, Any]:
        return self.current[shard_id]["record"]

    def current_bytes(self, shard_id: str) -> bytes:
        return canonical_bytes(self.current[shard_id]["record"])

    def mark_stale(self, plan: InvalidationPlan, *, new_generation: int | str | None = None) -> list[str]:
        marked: list[str] = []
        for shard_id in plan.invalid_shards:
            entry = self.current.pop(shard_id, None)
            if entry is None:
                continue
            entry["status"] = "stale"
            entry["stale_at_ms"] = monotonic_ms()
            entry["invalidation"] = plan.to_dict()
            entry["new_generation"] = new_generation
            self.stale[shard_id].append(entry)
            marked.append(shard_id)
        return marked

    def current_view(self) -> dict[str, dict[str, Any]]:
        return {shard_id: entry["record"] for shard_id, entry in sorted(self.current.items())}


def full_rebuild_views(spec: Spec) -> dict[str, dict[str, Any]]:
    return {shard_id: build_replay_view(spec, shard_id) for shard_id in stable_topological_order(spec.shards)}


def assert_matches_full_rebuild(
    spec: Spec,
    current_views: dict[str, dict[str, Any]],
    *,
    order: list[str] | None = None,
) -> None:
    full = full_rebuild_views(spec)
    check_order = order or stable_topological_order(spec.shards)
    for shard_id in check_order:
        if shard_id not in current_views:
            raise DivergeError("incremental view missing shard", object=shard_id, field="$")
        difference = first_difference(full[shard_id], current_views[shard_id], "$")
        if difference is not None:
            field, expected, actual = difference
            raise DivergeError(
                "incremental recomputation diverged from blank full rebuild",
                object=shard_id,
                field=field,
                details={"expected": expected, "actual": actual},
            )
