"""Stable invalidation propagation along the shard dependency graph.

Directly invalidated shards are those whose declared ``config_keys`` changed in
the revision, whose node capacity version changed, or whose shard parameters
changed. Their dependents are then invalidated transitively. The output is in
the stable topological order of the dependency graph and records the exact
reason and basis for each propagation step.
"""


def diff_integer_mappings(old_values, new_values):
    changed = {}
    for key in sorted(set(old_values) | set(new_values)):
        if old_values.get(key) != new_values.get(key):
            changed[key] = {"old": old_values.get(key), "new": new_values.get(key)}
    return changed


def propagate_invalidation(spec, old_revision, new_revision,
                           old_capacity_versions=None,
                           new_capacity_versions=None,
                           changed_shard_params=None):
    """Return ``(invalidated_in_topo_order, steps_by_shard)``."""
    old_values = spec.revision_values(old_revision) if old_revision else {}
    new_values = spec.revision_values(new_revision)
    changed_keys = diff_integer_mappings(old_values, new_values)
    old_capacity_versions = old_capacity_versions or {}
    new_capacity_versions = new_capacity_versions or {}
    changed_shard_params = changed_shard_params or set()

    reasons = {}

    def add(shard_id, reason, basis):
        # Keep the first (most direct) reason while still recording every
        # propagation basis for auditability.
        reasons.setdefault(shard_id, {"reasons": []})
        reasons[shard_id]["reasons"].append({"reason": reason, "basis": basis})

    for shard_id in spec.shard_order:
        shard = spec.shards[shard_id]
        node_id = shard.node_id
        node_changed = (
            node_id in old_capacity_versions
            and old_capacity_versions.get(node_id)
            != new_capacity_versions.get(node_id)
        )
        key_hits = sorted(key for key in shard.config_keys if key in changed_keys)
        if key_hits:
            add(
                shard_id,
                "config_key_change",
                {
                    "revision": new_revision,
                    "changed_keys": {key: changed_keys[key] for key in key_hits},
                },
            )
        if node_changed:
            add(
                shard_id,
                "capacity_version_change",
                {
                    "node_id": node_id,
                    "old": old_capacity_versions.get(node_id),
                    "new": new_capacity_versions.get(node_id),
                },
            )
        if shard_id in changed_shard_params:
            add(
                shard_id,
                "shard_parameter_change",
                {"shard_id": shard_id, "revision": new_revision},
            )

    # Transitive propagation, processed in topological order so every basis is
    # emitted before its dependents.
    for parent_id in spec.topo_order:
        if parent_id not in reasons:
            continue
        for child_id in spec.dependents.get(parent_id, []):
            add(
                child_id,
                "dependency_invalidation",
                {
                    "parent_shard_id": parent_id,
                    "path_basis": "depends_on",
                    "revision": new_revision,
                },
            )

    ordered = [sid for sid in spec.topo_order if sid in reasons]
    return ordered, reasons
