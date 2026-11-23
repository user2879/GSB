"""Revision-evolution invalidation propagation and selective recompilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import E_DIVERGE, E_MANIFEST, PipelineError
from .manifest import (
    Manifest,
    Revision,
    revision_dependency_closure,
)
from .planner import Plan, compile_plan
from .util import canonical_json


@dataclass(frozen=True)
class ValueChange:
    key: str
    from_version: int
    to_version: int
    old_value: Any
    new_value: Any


@dataclass(frozen=True)
class Invalidation:
    stable_id: str
    revision_version: int
    first_revision: int
    revision_from: int | None
    kind: str
    basis: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "shard_id": self.stable_id,
            "revision_version": self.revision_version,
            "first_revision": self.first_revision,
            "revision_from": self.revision_from,
            "kind": self.kind,
            "basis": list(self.basis),
        }


@dataclass
class EvolutionDelta:
    old_plan: Plan
    new_plan: Plan
    preserved_revisions: dict[str, int]
    recompute_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]
    invalidations: tuple[Invalidation, ...]
    propagation: tuple[dict[str, Any], ...]
    value_changes: tuple[ValueChange, ...]
    diff: dict[str, Any]

    @property
    def recompute_set(self) -> set[str]:
        return set(self.recompute_ids)

    def canonical(self) -> dict[str, Any]:
        return {
            "recompute_ids": list(self.recompute_ids),
            "removed_ids": list(self.removed_ids),
            "invalidations": [item.canonical() for item in self.invalidations],
            "propagation": [dict(item) for item in self.propagation],
            "value_changes": [
                {
                    "key": change.key,
                    "from_version": change.from_version,
                    "to_version": change.to_version,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                }
                for change in self.value_changes
            ],
            "preserved_revisions": {
                key: self.preserved_revisions[key]
                for key in sorted(self.preserved_revisions)
            },
            "diff": self.diff,
        }


def _dependency_path(
    revision: Revision,
    source_key: str,
    target_key: str,
    *,
    seen: set[str] | None = None,
) -> tuple[str, ...]:
    """Return target -> ... -> source, where source influences target."""

    if source_key == target_key:
        return (target_key,)
    stack = set() if seen is None else set(seen)
    if target_key in stack:
        return ()
    stack.add(target_key)
    rule = revision.effective_derived.get(target_key)
    if rule is None:
        return ()
    for input_key in rule.inputs:
        if source_key in revision_dependency_closure(revision, input_key):
            tail = _dependency_path(revision, source_key, input_key, seen=stack)
            if tail:
                return (target_key,) + tail
    return ()


def _field_difference(left: Any, right: Any, prefix: str = "") -> tuple[str, Any, Any] | None:
    if type(left) is not type(right):
        return (prefix or "<value>", left, right)
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in left:
                return (field, "<missing>", right[key])
            if key not in right:
                return (field, left[key], "<missing>")
            difference = _field_difference(left[key], right[key], field)
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        for index in range(min(len(left), len(right))):
            difference = _field_difference(
                left[index], right[index], f"{prefix}[{index}]"
            )
            if difference is not None:
                return difference
        if len(left) != len(right):
            return (f"{prefix}[{min(len(left), len(right))}]", left, right)
        return None
    if left != right:
        return (prefix or "<value>", left, right)
    return None


def diff_shard_views(
    old_view: Mapping[str, Mapping[str, Any]],
    new_view: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Diff two complete logical-shard views in stable id order."""

    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for stable_id in sorted(set(old_view) | set(new_view)):
        if stable_id not in old_view:
            added.append({"shard_id": stable_id})
            continue
        if stable_id not in new_view:
            removed.append({"shard_id": stable_id})
            continue
        difference = _field_difference(
            dict(old_view[stable_id]), dict(new_view[stable_id])
        )
        if difference is None:
            unchanged.append(stable_id)
        else:
            field, before, after = difference
            changed.append(
                {
                    "shard_id": stable_id,
                    "field": field,
                    "before": before,
                    "after": after,
                }
            )
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


def _validate_evolution_chain(old: Manifest, new: Manifest) -> None:
    old_latest = old.latest_revision.version
    new_latest = new.latest_revision.version
    if new_latest <= old_latest:
        raise PipelineError(
            E_MANIFEST,
            "evolved manifest must have a strictly greater latest revision",
            field="revisions",
            details={"old_latest": old_latest, "new_latest": new_latest},
        )
    new_versions = new.revision_by_version
    old_versions = old.revision_by_version
    for version in sorted(set(new_versions) & set(old_versions)):
        old_input = old_versions[version].canonical_input()
        new_input = new_versions[version].canonical_input()
        if canonical_json(old_input) != canonical_json(new_input):
            raise PipelineError(
                E_MANIFEST,
                "existing revision history was mutated during evolution",
                field=f"revisions[{version}]",
                details={"version": version},
            )
    if old_latest not in new_versions:
        raise PipelineError(
            E_MANIFEST,
            "evolved manifest omitted the previous latest revision",
            field="revisions",
            details={"missing_version": old_latest},
        )


def _value_changes(old: Manifest, new: Manifest) -> tuple[ValueChange, ...]:
    old_latest = old.latest_revision
    changes: list[ValueChange] = []
    seen: set[tuple[str, int, int, str]] = set()

    def add_change(change: ValueChange) -> None:
        identity = (
            change.key,
            change.from_version,
            change.to_version,
            canonical_json([change.old_value, change.new_value]),
        )
        if identity not in seen:
            seen.add(identity)
            changes.append(change)

    for version in sorted(new.revision_by_version):
        if version <= old_latest.version:
            continue
        revision = new.revision_by_version[version]
        if revision.parent is None:
            continue
        parent = new.revision_by_version[revision.parent]
        for key in sorted(
            set(parent.effective_values) | set(revision.effective_values)
        ):
            before = parent.effective_values.get(key, "<missing>")
            after = revision.effective_values.get(key, "<missing>")
            if before == after:
                continue
            add_change(
                ValueChange(
                    key=key,
                    from_version=parent.version,
                    to_version=revision.version,
                    old_value=before,
                    new_value=after,
                )
            )
    # Add a latest-to-latest fallback only when a non-linear branch left a final
    # difference unexplained.  Linear transitions are already represented
    # individually above.
    last_new_value: dict[str, Any] = {}
    for change in changes:
        last_new_value[change.key] = change.new_value
    for key in sorted(
        set(old_latest.effective_values) | set(new.latest_revision.effective_values)
    ):
        before = old_latest.effective_values.get(key, "<missing>")
        after = new.latest_revision.effective_values.get(key, "<missing>")
        if before != after and last_new_value.get(key, object()) != after:
            add_change(
                ValueChange(
                    key=key,
                    from_version=old_latest.version,
                    to_version=new.latest_revision.version,
                    old_value=before,
                    new_value=after,
                )
            )
    changes.sort(key=lambda item: (item.to_version, item.key, item.from_version))
    return tuple(changes)


def build_evolution(
    old: Manifest,
    new: Manifest,
    *,
    old_plan: Plan | None = None,
) -> EvolutionDelta:
    """Compute the complete invalid subset and its stable propagation trail."""

    _validate_evolution_chain(old, new)
    previous_plan = old_plan or compile_plan(old)
    old_shards = previous_plan.by_id
    old_tasks = old.task_by_id
    new_tasks = new.task_by_id
    old_clusters = old.cluster_by_id
    new_clusters = new.cluster_by_id
    old_latest = old.latest_revision
    new_latest = new.latest_revision
    changes = _value_changes(old, new)

    reasons: dict[str, list[str]] = {}
    first_revision: dict[str, int] = {}
    revision_from: dict[str, int | None] = {}
    kinds: dict[str, str] = {}

    def add(
        stable_id: str,
        reason: str,
        *,
        revision: int,
        kind: str,
        from_revision: int | None,
    ) -> None:
        reasons.setdefault(stable_id, [])
        if reason not in reasons[stable_id]:
            reasons[stable_id].append(reason)
        first_revision[stable_id] = min(
            first_revision.get(stable_id, revision), revision
        )
        if stable_id not in revision_from or (
            from_revision is not None
            and (revision_from[stable_id] is None or from_revision < revision_from[stable_id])
        ):
            revision_from[stable_id] = from_revision
        priority = {"removed": 0, "added": 1, "rebuild": 2}
        if stable_id not in kinds or priority[kind] > priority.get(kinds[stable_id], -1):
            kinds[stable_id] = kind

    changed_task_ids: set[str] = set()
    for task_id in sorted(set(old_tasks) | set(new_tasks)):
        old_task = old_tasks.get(task_id)
        new_task = new_tasks.get(task_id)
        if old_task is None:
            changed_task_ids.add(task_id)
            for shard in new_plan_shards_proxy(new, task_id):
                add(
                    shard,
                    f"task:{task_id}:added@{new_latest.version}",
                    revision=new_latest.version,
                    kind="added",
                    from_revision=None,
                )
            continue
        if new_task is None:
            for shard in old_plan_shards_proxy(previous_plan, task_id):
                add(
                    shard,
                    f"task:{task_id}:removed@{new_latest.version}",
                    revision=old_latest.version,
                    kind="removed",
                    from_revision=old_latest.version,
                )
            continue
        if old_task.canonical() != new_task.canonical():
            changed_task_ids.add(task_id)
            old_targets = set(old_task.target_clusters)
            new_targets = set(new_task.target_clusters)
            for shard in new_plan_shards_proxy(new, task_id):
                add(
                    shard,
                    f"task:{task_id}:definition_changed@{new_latest.version}",
                    revision=new_latest.version,
                    kind="rebuild",
                    from_revision=old_latest.version,
                )
            for shard in old_plan_shards_proxy(previous_plan, task_id):
                old_shard = old_shards[shard]
                if old_shard.cluster_id not in new_targets:
                    add(
                        shard,
                        f"task:{task_id}:target_removed:{old_shard.cluster_id}",
                        revision=old_latest.version,
                        kind="removed",
                        from_revision=old_latest.version,
                    )
                elif old_task.shards > new_task.shards and old_shard.shard_index >= new_task.shards:
                    add(
                        shard,
                        f"task:{task_id}:shard_count_reduced:{old_task.shards}->{new_task.shards}",
                        revision=old_latest.version,
                        kind="removed",
                        from_revision=old_latest.version,
                    )
            if old_targets != new_targets:
                for shard in new_plan_shards_proxy(new, task_id):
                    add(
                        shard,
                        f"task:{task_id}:targets_changed:{sorted(old_targets)}->{sorted(new_targets)}",
                        revision=new_latest.version,
                        kind="rebuild",
                        from_revision=old_latest.version,
                    )

    changed_cluster_ids: set[str] = set()
    for cluster_id in sorted(set(old_clusters) | set(new_clusters)):
        before = old_clusters.get(cluster_id)
        after = new_clusters.get(cluster_id)
        if before is None or after is None or before.canonical() != after.canonical():
            changed_cluster_ids.add(cluster_id)
    for task in new.tasks:
        for cluster_id in task.target_clusters:
            if cluster_id not in changed_cluster_ids:
                continue
            for shard in new_plan_shards_proxy(new, task.id, cluster_id=cluster_id):
                add(
                    shard,
                    f"cluster:{cluster_id}:definition_changed@{new_latest.version}",
                    revision=new_latest.version,
                    kind="rebuild",
                    from_revision=old_latest.version,
                )
    for cluster_id in changed_cluster_ids:
        if cluster_id not in old_clusters:
            continue
        for shard in previous_plan.shards:
            if shard.cluster_id == cluster_id and cluster_id not in new_clusters:
                add(
                    shard.stable_id,
                    f"cluster:{cluster_id}:removed@{new_latest.version}",
                    revision=old_latest.version,
                    kind="removed",
                    from_revision=old_latest.version,
                )

    # Config dependencies: direct changes flow through derived definitions to
    # task key references, then across every task target cluster.
    for change in changes:
        for task_id in sorted(set(old_tasks) | set(new_tasks)):
            old_task = old_tasks.get(task_id)
            new_task = new_tasks.get(task_id)
            references: set[str] = set()
            if new_task is not None:
                references.update(new_task.config_keys)
            if old_task is not None:
                references.update(old_task.config_keys)
            revision = new_latest if new_task is not None else old_latest
            affected = False
            task_reasons: list[str] = []
            for reference in sorted(references):
                if reference not in revision.effective_values:
                    continue
                closure = revision_dependency_closure(revision, reference)
                if change.key not in closure:
                    continue
                affected = True
                path = _dependency_path(revision, change.key, reference)
                rendered_path = " -> ".join(path) if path else change.key
                task_reasons.append(
                    "config/"
                    f"v{change.from_version}->v{change.to_version}/"
                    f"{change.key}:{change.old_value}->{change.new_value};"
                    f"dependency:{rendered_path};task:{task_id};key:{reference}"
                )
            if not affected:
                continue
            for shard in new_plan_shards_proxy(new, task_id):
                for reason in task_reasons:
                    add(
                        shard,
                        reason,
                        revision=change.to_version,
                        kind="rebuild",
                        from_revision=change.from_version,
                    )
            new_ids = set(new_plan_shards_proxy(new, task_id))
            for shard in old_plan_shards_proxy(previous_plan, task_id):
                if shard not in new_ids and shard not in reasons:
                    add(
                        shard,
                        "config-dependent shard removed by task/target evolution",
                        revision=old_latest.version,
                        kind="removed",
                        from_revision=change.from_version,
                    )

    # A changed derivation definition can alter semantics even when all current
    # integer values happen to be equal.
    for task_id in sorted(set(old_tasks) | set(new_tasks)):
        old_task = old_tasks.get(task_id)
        new_task = new_tasks.get(task_id)
        references = set()
        if old_task is not None:
            references.update(old_task.config_keys)
        if new_task is not None:
            references.update(new_task.config_keys)
        relation_change_notes: list[str] = []
        for reference in sorted(references):
            old_dependencies = set(
                revision_dependency_closure(old_latest, reference)
            )
            new_dependencies = set(
                revision_dependency_closure(new_latest, reference)
            )
            for key in sorted(old_dependencies | new_dependencies):
                old_rule = old_latest.effective_derived.get(key)
                new_rule = new_latest.effective_derived.get(key)
                old_canonical = (
                    canonical_json(old_rule.canonical()) if old_rule else None
                )
                new_canonical = (
                    canonical_json(new_rule.canonical()) if new_rule else None
                )
                if old_canonical != new_canonical:
                    relation_change_notes.append(
                        f"derivation:{key}:definition_changed;"
                        f"task:{task_id};key:{reference}"
                    )
        if relation_change_notes:
            for shard in new_plan_shards_proxy(new, task_id):
                for note in relation_change_notes:
                    add(
                        shard,
                        note,
                        revision=new_latest.version,
                        kind="rebuild",
                        from_revision=old_latest.version,
                    )

    # Every new logical shard not already affected by another rule but absent
    # from the old plan is still a newly materialised shard.
    for shard in new_plan_shards_proxy(new):
        if shard not in old_shards and shard not in reasons:
            add(
                shard,
                "new logical shard",
                revision=new_latest.version,
                kind="added",
                from_revision=None,
            )

    # Compile the new plan, preserving the source revision for unaffected
    # shards.  This is the only path that touches recompilation.
    preserved: dict[str, int] = {}
    new_ids_all = set(new_plan_shards_proxy(new))
    for stable_id, old_shard in old_shards.items():
        if stable_id in reasons:
            continue
        if stable_id in new_ids_all:
            preserved[stable_id] = old_shard.revision_version
    new_plan = compile_plan(new, preserved)

    for stable_id in sorted(preserved):
        old_shard = old_shards[stable_id]
        new_shard = new_plan.by_id[stable_id]
        if old_shard.compile_digest != new_shard.compile_digest:
            raise PipelineError(
                E_DIVERGE,
                "an unaffected shard changed compile digest during evolution",
                shard_id=stable_id,
                field="compile_digest",
                details={
                    "expected": old_shard.compile_digest,
                    "actual": new_shard.compile_digest,
                },
            )

    recompute_ids = tuple(
        item[1]
        for item in sorted(
            (
                new_plan.by_id[stable_id].revision_version,
                stable_id,
            )
            for stable_id in new_plan.by_id
            if stable_id in reasons and kinds.get(stable_id) != "removed"
        )
    )
    removed_ids = tuple(
        item[1]
        for item in sorted(
            (
                old_shards[stable_id].revision_version,
                stable_id,
            )
            for stable_id in reasons
            if stable_id not in new_plan.by_id
        )
    )
    invalidations: list[Invalidation] = []
    for stable_id in sorted(reasons, key=lambda item: (first_revision[item], item)):
        target_revision = (
            new_plan.by_id[stable_id].revision_version
            if stable_id in new_plan.by_id
            else old_shards[stable_id].revision_version
        )
        invalidations.append(
            Invalidation(
                stable_id=stable_id,
                revision_version=target_revision,
                first_revision=first_revision[stable_id],
                revision_from=revision_from.get(stable_id),
                kind=kinds[stable_id],
                basis=tuple(reasons[stable_id]),
            )
        )

    propagation = tuple(
        {
            "revision_version": item.first_revision,
            "shard_id": item.stable_id,
            "kind": item.kind,
            "basis": list(item.basis),
        }
        for item in invalidations
    )

    old_view = {
        stable_id: {
            "revision_version": shard.revision_version,
            "compile_digest": shard.compile_digest,
        }
        for stable_id, shard in old_shards.items()
    }
    new_view = {
        stable_id: {
            "revision_version": shard.revision_version,
            "compile_digest": shard.compile_digest,
        }
        for stable_id, shard in new_plan.by_id.items()
    }
    return EvolutionDelta(
        old_plan=previous_plan,
        new_plan=new_plan,
        preserved_revisions=preserved,
        recompute_ids=recompute_ids,
        removed_ids=removed_ids,
        invalidations=tuple(invalidations),
        propagation=propagation,
        value_changes=changes,
        diff=diff_shard_views(old_view, new_view),
    )


def new_plan_shards_proxy(
    manifest: Manifest,
    task_id: str | None = None,
    cluster_id: str | None = None,
) -> list[str]:
    """Expand only logical shard ids without compiling values.

    This helper keeps invalidation analysis independent from the final selective
    plan while still using the package's one stable-id implementation.
    """

    from .planner import stable_shard_id

    result: list[str] = []
    for task in manifest.tasks:
        if task_id is not None and task.id != task_id:
            continue
        for target in task.target_clusters:
            if cluster_id is not None and target != cluster_id:
                continue
            for index in range(task.shards):
                result.append(stable_shard_id(target, task.id, index))
    return sorted(result)


def old_plan_shards_proxy(plan: Plan, task_id: str) -> list[str]:
    return sorted(
        shard.stable_id for shard in plan.shards if shard.task_id == task_id
    )
