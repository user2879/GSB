"""Incremental recompilation and derivation/reference invalidation."""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import E_REF, ErrorDetail, PlatformError
from ..util.canonical import digest
from .compiler import CompiledVersion, compile_version
from .grid import Grid
from .manifest import Item, ManifestVersion


@dataclass(frozen=True)
class InvalidationStep:
    key: str
    trigger: str          # direct-change | derivation-downstream | node-reference
    source: str           # configuration item key that triggered this step


def _value_fingerprint(item: Item) -> str:
    return digest(
        None if item.value is None else f"{item.value}"
    )


def _targets_fingerprint(item: Item) -> str:
    return digest(sorted(item.targets))


def _derivation_fingerprint(item: Item) -> str:
    return digest(None if item.derivation is None else item.derivation.to_dict())


def reject_deleted_references(old: ManifestVersion, new: ManifestVersion) -> None:
    """Deleting an item that is still referenced must fail with E_REF."""
    deleted = set(old.items) - set(new.items)
    if not deleted:
        return
    details: list[ErrorDetail] = []
    for gone in sorted(deleted):
        referrers = [
            key
            for key, item in new.items.items()
            if item.derivation is not None and gone in item.derivation.coefficients
        ]
        if referrers:
            details.append(
                ErrorDetail(
                    reason=(
                        f"cannot delete '{gone}': still referenced by "
                        + ", ".join(sorted(referrers))
                    ),
                    key=gone,
                    source="manifest",
                )
            )
    PlatformError.raise_if(E_REF, "deletion blocked by live references", details)


def _derivation_children(version: ManifestVersion) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {key: [] for key in version.order}
    for key in version.order:
        item = version.items[key]
        if item.derivation is not None:
            for upstream in item.derivation.coefficients:
                children.setdefault(upstream, []).append(key)
    for key in children:
        children[key].sort()
    return children


def _node_reference_children(version: ManifestVersion) -> dict[str, list[str]]:
    """Items co-located on a target node form node-reference propagation edges."""
    by_node: dict[str, list[str]] = {}
    for key in version.order:
        for node_id in version.items[key].targets:
            by_node.setdefault(node_id, []).append(key)
    children: dict[str, list[str]] = {key: [] for key in version.order}
    for members in by_node.values():
        members_sorted = sorted(members)
        for key in members_sorted:
            for other in members_sorted:
                if other != key and other not in children[key]:
                    children[key].append(other)
    for key in children:
        children[key].sort()
    return children


def _change_kinds(
    old: ManifestVersion, new: ManifestVersion
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (all_direct, value_or_coef_changed, targets_changed, new_keys)."""
    direct: set[str] = set()
    value_or_coef: set[str] = set()
    targets_changed: set[str] = set()
    new_keys: set[str] = set()

    for key, item in new.items.items():
        if key not in old.items:
            direct.add(key)
            new_keys.add(key)
            continue
        previous = old.items[key]
        changed = False
        if _value_fingerprint(previous) != _value_fingerprint(item):
            value_or_coef.add(key)
            changed = True
        if _derivation_fingerprint(previous) != _derivation_fingerprint(item):
            value_or_coef.add(key)
            changed = True
        if _targets_fingerprint(previous) != _targets_fingerprint(item):
            targets_changed.add(key)
            changed = True
        if changed:
            direct.add(key)
    return direct, value_or_coef, targets_changed, new_keys


def derive_invalidation(
    old: ManifestVersion | None,
    new: ManifestVersion,
) -> tuple[set[str], list[InvalidationStep]]:
    """Return the invalidated key set with stable, source-tagged steps."""
    steps: list[InvalidationStep] = []
    if old is None:
        changed = set(new.items)
        for key in new.order:
            steps.append(InvalidationStep(key, "direct-change", "initial-compile"))
        return changed, steps

    reject_deleted_references(old, new)
    direct, value_or_coef, targets_changed, new_keys = _change_kinds(old, new)

    for key in sorted(direct):
        source = "new-key" if key in new_keys else (
            "targets-changed" if key in targets_changed else "value-or-coefficient"
        )
        steps.append(InvalidationStep(key, "direct-change", source))

    invalidated: set[str] = set(direct)
    derivation_children = _derivation_children(new)
    node_children = _node_reference_children(new)

    # BFS.  A node-reference edge only propagates from an item whose target set
    # actually changed (a distribution topology change).  Plain value/coefficient
    # changes propagate solely through the derivation graph.
    queue = sorted(direct)
    while queue:
        source = queue.pop(0)
        for child in derivation_children.get(source, []):
            if child not in invalidated:
                invalidated.add(child)
                steps.append(InvalidationStep(child, "derivation-downstream", source))
                queue.append(child)
        if source in targets_changed or source in new_keys:
            for peer in node_children.get(source, []):
                if peer not in invalidated:
                    invalidated.add(peer)
                    steps.append(InvalidationStep(peer, "node-reference", source))
                    queue.append(peer)

    return invalidated, steps


def incremental_compile(
    old_manifest: ManifestVersion | None,
    new_manifest: ManifestVersion,
    grid: Grid,
    previous: CompiledVersion | None,
) -> tuple[CompiledVersion, set[str], list[InvalidationStep], dict[str, bool]]:
    """Recompile only invalidated items; keep untouched artifacts byte-identical."""
    invalidated, steps = derive_invalidation(old_manifest, new_manifest)
    compiled = compile_version(new_manifest, grid)

    byte_check: dict[str, bool] = {}
    if previous is not None and old_manifest is not None:
        for key in new_manifest.order:
            if key in previous.items and key not in invalidated:
                byte_check[key] = (
                    previous.items[key].artifact_bytes
                    == compiled.items[key].artifact_bytes
                )
    return compiled, invalidated, steps, byte_check
