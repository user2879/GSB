"""Loading, normalisation and strong validation of the local manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from .errors import E_MANIFEST, PipelineError
from .util import digest_json, is_int, stable_int


_MISSING = object()


class _Field(NamedTuple):
    value: Any
    source_name: str


def _read_field(
    mapping: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    default: Any = _MISSING,
) -> _Field:
    for name in names:
        if name in mapping:
            return _Field(mapping[name], name)
    if default is not _MISSING:
        return _Field(default, names[0])
    return _Field(_MISSING, names[0])


def _manifest_error(field: str, message: str, **details: Any) -> PipelineError:
    return PipelineError(E_MANIFEST, message, field=field, details=details)


def _require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not is_int(value):
        raise _manifest_error(field, "must be an integer (booleans are not integers)")
    number = int(value)
    if minimum is not None and number < minimum:
        raise _manifest_error(field, f"must be >= {minimum}", actual=number)
    if maximum is not None and number > maximum:
        raise _manifest_error(field, f"must be <= {maximum}", actual=number)
    return number


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _manifest_error(field, "must be a non-empty string")
    if "\x00" in value:
        raise _manifest_error(field, "must not contain NUL characters")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _manifest_error(field, "must be an array")
    return value


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _manifest_error(field, "must be an object")
    return value


def _unique_strings(value: Any, field: str, *, non_empty: bool = True) -> tuple[str, ...]:
    values = _require_list(value, field)
    if non_empty and not values:
        raise _manifest_error(field, "must contain at least one entry")
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        item = _require_string(raw, f"{field}[{index}]")
        if item in seen:
            raise _manifest_error(f"{field}[{index}]", "duplicate identifier", value=item)
        seen.add(item)
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class Cluster:
    id: str
    host: str
    port: int
    nodes: tuple[str, ...]
    concurrency: int
    seed: int

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "nodes": list(self.nodes),
            "concurrency": self.concurrency,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Task:
    id: str
    config_keys: tuple[str, ...]
    target_clusters: tuple[str, ...]
    shards: int
    resource_units: int
    concurrency_limit: int

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config_keys": list(self.config_keys),
            "target_clusters": list(self.target_clusters),
            "shards": self.shards,
            "resource_units": self.resource_units,
            "concurrency_limit": self.concurrency_limit,
        }


@dataclass(frozen=True)
class DerivedRule:
    op: str
    inputs: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {"op": self.op, "inputs": list(self.inputs)}


_DERIVED_OPS = {"sum", "min", "max", "product", "sub"}


@dataclass(frozen=True)
class Revision:
    version: int
    parent: int | None
    values: dict[str, int]
    derived: dict[str, DerivedRule]
    base_values: dict[str, int]
    effective_derived: dict[str, DerivedRule]
    effective_values: dict[str, int]

    def canonical_input(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "values": dict(self.values),
            "derived": {key: self.derived[key].canonical() for key in sorted(self.derived)},
        }

    def canonical_effective(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent": self.parent,
            "values": {key: self.effective_values[key] for key in sorted(self.effective_values)},
            "derived": {
                key: self.effective_derived[key].canonical()
                for key in sorted(self.effective_derived)
            },
        }


@dataclass(frozen=True)
class Manifest:
    global_resource_budget: int
    clusters: tuple[Cluster, ...]
    tasks: tuple[Task, ...]
    revisions: tuple[Revision, ...]

    @property
    def latest_revision(self) -> Revision:
        return self.revisions[-1]

    @property
    def cluster_by_id(self) -> dict[str, Cluster]:
        return {cluster.id: cluster for cluster in self.clusters}

    @property
    def task_by_id(self) -> dict[str, Task]:
        return {task.id: task for task in self.tasks}

    @property
    def revision_by_version(self) -> dict[int, Revision]:
        return {revision.version: revision for revision in self.revisions}

    @property
    def cluster_version(self) -> str:
        return digest_json([cluster.canonical() for cluster in self.clusters])

    @property
    def revision_chain_version(self) -> str:
        return digest_json([revision.canonical_input() for revision in self.revisions])

    @property
    def budget_version(self) -> str:
        return digest_json(
            {
                "global_resource_budget": self.global_resource_budget,
                "task_limits": {
                    task.id: task.concurrency_limit for task in self.tasks
                },
                "cluster_quotas": {
                    cluster.id: cluster.concurrency for cluster in self.clusters
                },
            }
        )

    @property
    def manifest_digest(self) -> str:
        return digest_json(self.canonical())

    def canonical(self) -> dict[str, Any]:
        return {
            "global_resource_budget": self.global_resource_budget,
            "clusters": [cluster.canonical() for cluster in self.clusters],
            "tasks": [task.canonical() for task in self.tasks],
            "revisions": [revision.canonical_input() for revision in self.revisions],
        }


def load_manifest(source: str | Path | Mapping[str, Any]) -> Manifest:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _manifest_error("$", f"cannot read manifest: {exc}") from exc
        try:
            raw = json.loads(text, object_pairs_hook=_reject_duplicate_object_keys)
        except PipelineError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise _manifest_error("$", f"invalid JSON: {exc}") from exc
    else:
        raw = dict(source)
    return manifest_from_dict(raw)


def _reject_duplicate_object_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _manifest_error("$", "duplicate JSON object key", key=key)
        result[key] = value
    return result


def manifest_from_dict(raw: Any) -> Manifest:
    root = _require_mapping(raw, "$")

    budget_field = _read_field(
        root,
        ("global_resource_budget", "resource_budget", "global_budget", "budget"),
    )
    if budget_field.value is _MISSING:
        raise _manifest_error(
            "global_resource_budget",
            "missing required field (accepted names: global_resource_budget, resource_budget, global_budget, budget)",
        )
    budget_value = budget_field.value
    budget_field_name = (
        "global_resource_budget."
        if budget_field.source_name in {"global_resource_budget", "resource_budget"}
        else "budget"
    )
    if isinstance(budget_value, dict):
        nested_budget = _read_field(budget_value, ("resource_units", "units"))
        if nested_budget.value is _MISSING:
            raise _manifest_error(
                "budget.resource_units", "missing required integer field"
            )
        budget_value = nested_budget.value
        budget_field_name = "budget.resource_units"
    global_budget = _require_int(budget_value, budget_field_name, minimum=0)

    raw_clusters = _require_list(root.get("clusters", _MISSING), "clusters")
    if not raw_clusters:
        raise _manifest_error("clusters", "must contain at least one cluster")
    clusters: list[Cluster] = []
    cluster_ids: set[str] = set()
    ports: dict[int, int] = {}
    for index, raw_cluster in enumerate(raw_clusters):
        prefix = f"clusters[{index}]"
        cluster_map = _require_mapping(raw_cluster, prefix)
        id_field = _read_field(cluster_map, ("id", "cluster_id"))
        cluster_id = _require_string(id_field.value, f"{prefix}.{id_field.source_name}")
        if cluster_id in cluster_ids:
            raise _manifest_error(f"{prefix}.{id_field.source_name}", "duplicate cluster id", value=cluster_id)
        cluster_ids.add(cluster_id)

        host_field = _read_field(
            cluster_map, ("host", "loopback_host"), default="127.0.0.1"
        )
        if host_field.value != "127.0.0.1":
            raise _manifest_error(
                f"{prefix}.{host_field.source_name}",
                'must be the literal loopback address "127.0.0.1"',
            )

        port_field = _read_field(cluster_map, ("port", "loopback_port"))
        if port_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.port", "missing required loopback port")
        port = _require_int(port_field.value, f"{prefix}.{port_field.source_name}", minimum=1, maximum=65535)
        if port in ports:
            raise _manifest_error(
                f"{prefix}.{port_field.source_name}",
                "loopback port must be globally unique",
                value=port,
                other_cluster_index=ports[port],
            )
        ports[port] = index

        nodes_field = _read_field(cluster_map, ("nodes", "node_ids"))
        if nodes_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.nodes", "missing required node list")
        nodes = tuple(sorted(_unique_strings(nodes_field.value, f"{prefix}.nodes")))

        concurrency_field = _read_field(
            cluster_map, ("concurrency", "concurrency_quota", "max_concurrency")
        )
        if concurrency_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.concurrency", "missing required concurrency quota")
        concurrency = _require_int(
            concurrency_field.value,
            f"{prefix}.{concurrency_field.source_name}",
            minimum=1,
        )

        seed_field = _read_field(cluster_map, ("seed",), default=stable_int(cluster_id))
        seed = _require_int(seed_field.value, f"{prefix}.{seed_field.source_name}")
        clusters.append(
            Cluster(
                id=cluster_id,
                host="127.0.0.1",
                port=port,
                nodes=nodes,
                concurrency=concurrency,
                seed=seed,
            )
        )

    raw_tasks = _require_list(root.get("tasks", _MISSING), "tasks")
    tasks: list[Task] = []
    task_ids: set[str] = set()
    for index, raw_task in enumerate(raw_tasks):
        prefix = f"tasks[{index}]"
        task_map = _require_mapping(raw_task, prefix)
        id_field = _read_field(task_map, ("id", "task_id"))
        task_id = _require_string(id_field.value, f"{prefix}.{id_field.source_name}")
        if task_id in task_ids:
            raise _manifest_error(f"{prefix}.{id_field.source_name}", "duplicate task id", value=task_id)
        task_ids.add(task_id)

        keys_field = _read_field(
            task_map, ("config_keys", "config_key_set", "keys")
        )
        if keys_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.config_keys", "missing required config key set")
        config_keys = _unique_strings(keys_field.value, f"{prefix}.{keys_field.source_name}")
        config_keys = tuple(sorted(config_keys))

        targets_field = _read_field(
            task_map,
            ("target_clusters", "targets", "target_cluster_ids", "clusters"),
        )
        if targets_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.target_clusters", "missing required target cluster list")
        targets = _unique_strings(targets_field.value, f"{prefix}.{targets_field.source_name}")
        for target_index, target in enumerate(targets):
            if target not in cluster_ids:
                raise _manifest_error(
                    f"{prefix}.{targets_field.source_name}[{target_index}]",
                    "references an unknown cluster",
                    value=target,
                )
        targets = tuple(sorted(targets))

        shards_field = _read_field(task_map, ("shards", "shard_count"))
        if shards_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.shards", "missing required shard count")
        shards = _require_int(shards_field.value, f"{prefix}.{shards_field.source_name}", minimum=0)

        resources_field = _read_field(
            task_map,
            (
                "resource_units",
                "resources_per_shard",
                "resource_units_per_shard",
                "resources",
            ),
        )
        if resources_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.resource_units", "missing required per-shard resource units")
        resource_units = _require_int(
            resources_field.value,
            f"{prefix}.{resources_field.source_name}",
            minimum=0,
        )

        limit_field = _read_field(
            task_map,
            (
                "concurrency_limit",
                "task_concurrency",
                "max_concurrency",
                "concurrency",
            ),
            default=max(1, shards),
        )
        concurrency_limit = _require_int(
            limit_field.value,
            f"{prefix}.{limit_field.source_name}",
            minimum=1,
        )
        tasks.append(
            Task(
                id=task_id,
                config_keys=config_keys,
                target_clusters=targets,
                shards=shards,
                resource_units=resource_units,
                concurrency_limit=concurrency_limit,
            )
        )

        revisions_source = _read_field(
            root, ("revisions", "revision_chain", "config_revisions")
        )
        raw_revisions = _require_list(revisions_source.value, "revisions")
    if not raw_revisions:
        raise _manifest_error("revisions", "must contain at least one revision")
    revisions: list[Revision] = []
    known_versions: dict[int, Revision] = {}
    previous_version: int | None = None
    for index, raw_revision in enumerate(raw_revisions):
        prefix = f"revisions[{index}]"
        revision_map = _require_mapping(raw_revision, prefix)
        version_field = _read_field(revision_map, ("version",))
        if version_field.value is _MISSING:
            raise _manifest_error(f"{prefix}.version", "missing required revision version")
        version = _require_int(version_field.value, f"{prefix}.version", minimum=0)
        if version in known_versions:
            raise _manifest_error(f"{prefix}.version", "duplicate revision version", value=version)
        if previous_version is not None and version <= previous_version:
            raise _manifest_error(
                f"{prefix}.version",
                "revision versions must be strictly monotonically increasing in array order",
                previous=previous_version,
                actual=version,
            )

        parent_field = _read_field(revision_map, ("parent", "parent_version"))
        parent_value = parent_field.value
        if index == 0:
            if parent_value not in (_MISSING, None):
                raise _manifest_error(
                    f"{prefix}.{parent_field.source_name}",
                    "first revision must have a null parent",
                )
            parent = None
        else:
            if parent_value is _MISSING or parent_value is None:
                raise _manifest_error(
                    f"{prefix}.{parent_field.source_name}",
                    "non-first revision must reference an existing parent revision",
                )
            parent = _require_int(parent_value, f"{prefix}.{parent_field.source_name}", minimum=0)
            if parent >= version:
                raise _manifest_error(
                    f"{prefix}.{parent_field.source_name}",
                    "parent version must be lower than the child version",
                    actual=parent,
                    version=version,
                )
            if parent not in known_versions:
                raise _manifest_error(
                    f"{prefix}.{parent_field.source_name}",
                    "parent revision does not exist earlier in the chain",
                    actual=parent,
                )

        values_field = _read_field(
            revision_map, ("values", "config_values", "config"), default={}
        )
        values_map = _require_mapping(values_field.value, f"{prefix}.{values_field.source_name}")
        values: dict[str, int] = {}
        for key in values_map:
            value_field = f"{prefix}.{values_field.source_name}.{key}"
            config_key = _require_string(key, value_field)
            values[config_key] = _require_int(values_map[key], value_field)

        derived_field = _read_field(revision_map, ("derived", "derived_values"), default={})
        derived_map = _require_mapping(derived_field.value, f"{prefix}.{derived_field.source_name}")
        derived: dict[str, DerivedRule] = {}
        for key in derived_map:
            rule_field = f"{prefix}.{derived_field.source_name}.{key}"
            config_key = _require_string(key, rule_field)
            if config_key in values:
                raise _manifest_error(
                    rule_field,
                    "a key cannot be both a direct value and a derived value in one revision",
                )
            rule_obj = _require_mapping(derived_map[key], rule_field)
            op_field = _read_field(rule_obj, ("op", "operation", "function"))
            if op_field.value is _MISSING:
                raise _manifest_error(f"{rule_field}.op", "missing derivation operation")
            op = op_field.value
            if isinstance(op, str):
                op = {
                    "add": "sum",
                    "multiply": "product",
                    "subtract": "sub",
                }.get(op, op)
            if not isinstance(op, str) or op not in _DERIVED_OPS:
                raise _manifest_error(
                    f"{rule_field}.{op_field.source_name}",
                    f"unsupported operation; expected one of {sorted(_DERIVED_OPS)}",
                )
            inputs_field = _read_field(
                rule_obj, ("inputs", "from", "keys", "depends_on")
            )
            if inputs_field.value is _MISSING:
                raise _manifest_error(f"{rule_field}.inputs", "missing derivation inputs")
            inputs = _unique_strings(inputs_field.value, f"{rule_field}.{inputs_field.source_name}")
            if op in {"sum", "min", "max", "product"}:
                inputs = tuple(sorted(inputs))
            if len(inputs) < 1:
                raise _manifest_error(f"{rule_field}.inputs", "must contain at least one input")
            if op == "sub" and len(inputs) < 2:
                raise _manifest_error(
                    f"{rule_field}.inputs", "sub requires at least two inputs"
                )
            derived[config_key] = DerivedRule(op=op, inputs=inputs)

        parent_revision = known_versions.get(parent) if parent is not None else None
        inherited_base = dict(parent_revision.base_values) if parent_revision else {}
        inherited_derived = dict(parent_revision.effective_derived) if parent_revision else {}

        for key in values:
            if key in inherited_derived and key not in derived:
                raise _manifest_error(
                    f"{prefix}.{values_field.source_name}.{key}",
                    "cannot replace an inherited derived key with a direct value without redefining it",
                )
        base_values = dict(inherited_base)
        base_values.update(values)
        effective_derived = dict(inherited_derived)
        effective_derived.update(derived)
        for key in effective_derived:
            if key in base_values:
                raise _manifest_error(
                    f"{prefix}.{derived_field.source_name}.{key}",
                    "derived key conflicts with an effective direct value",
                )

        resolving: set[str] = set()
        resolved: dict[str, int] = {}

        def resolve_value(key: str, field: str) -> int:
            if key in resolved:
                return resolved[key]
            if key in base_values:
                resolved[key] = base_values[key]
                return resolved[key]
            if key not in effective_derived:
                raise _manifest_error(field, "references an unknown config key", config_key=key)
            if key in resolving:
                raise _manifest_error(field, "derived key cycle detected", config_key=key)
            rule = effective_derived[key]
            resolving.add(key)
            inputs = [resolve_value(item, field) for item in rule.inputs]
            resolving.remove(key)
            if rule.op == "sum":
                result = sum(inputs)
            elif rule.op == "min":
                result = min(inputs)
            elif rule.op == "max":
                result = max(inputs)
            elif rule.op == "product":
                result = 1
                for item in inputs:
                    result *= item
            elif rule.op == "sub":
                result = inputs[0]
                for item in inputs[1:]:
                    result -= item
            else:  # pragma: no cover - guarded by validation
                raise _manifest_error(field, "unsupported derivation operation", op=rule.op)
            if not is_int(result):
                raise _manifest_error(field, "derived result is not an integer")
            resolved[key] = int(result)
            return resolved[key]

        effective_values: dict[str, int] = {}
        for base_key in sorted(base_values):
            effective_values[base_key] = resolve_value(
                base_key, f"{prefix}.{values_field.source_name}.{base_key}"
            )
        for derived_key in sorted(effective_derived):
            effective_values[derived_key] = resolve_value(
                derived_key,
                f"{prefix}.{derived_field.source_name}.{derived_key}",
            )

        revision = Revision(
            version=version,
            parent=parent,
            values=dict(sorted(values.items())),
            derived={key: derived[key] for key in sorted(derived)},
            base_values=dict(sorted(base_values.items())),
            effective_derived={key: effective_derived[key] for key in sorted(effective_derived)},
            effective_values=dict(sorted(effective_values.items())),
        )
        revisions.append(revision)
        known_versions[version] = revision
        previous_version = version

    latest_values = revisions[-1].effective_values
    for task_index, task in enumerate(tasks):
        for key_index, key in enumerate(task.config_keys):
            if key not in latest_values:
                raise _manifest_error(
                    f"tasks[{task_index}].config_keys[{key_index}]",
                    "config key is unavailable in the latest effective revision",
                    config_key=key,
                    revision=revisions[-1].version,
                )

        if task.resource_units > global_budget:
            raise _manifest_error(
                f"tasks[{task_index}].resource_units",
                "per-shard resource units exceed the global resource budget",
                resource_units=task.resource_units,
                global_resource_budget=global_budget,
            )

    return Manifest(
        global_resource_budget=global_budget,
        clusters=tuple(sorted(clusters, key=lambda item: item.id)),
        tasks=tuple(sorted(tasks, key=lambda item: item.id)),
        revisions=tuple(revisions),
    )


def revision_dependency_closure(
    revision: Revision, key: str, *, _stack: set[str] | None = None
) -> tuple[str, ...]:
    """Return the stable set of config keys that can influence ``key``.

    The result includes the requested key.  It is used for invalidation
    propagation across derived relations.
    """

    stack = set() if _stack is None else set(_stack)
    if key in stack:
        return ()
    stack.add(key)
    rule = revision.effective_derived.get(key)
    if rule is None:
        return (key,)
    dependencies: set[str] = {key}
    for item in rule.inputs:
        dependencies.update(revision_dependency_closure(revision, item, _stack=stack))
    return tuple(sorted(dependencies))
