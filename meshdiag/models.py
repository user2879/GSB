from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_json, sha256_json, sha256_text
from .errors import diag, fail_cycle, fail_ref, fail_rule


@dataclass(frozen=True, order=True)
class Instance:
    instance_id: str
    host: str
    port: int

    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.instance_id, "host": self.host, "port": self.port}


@dataclass(frozen=True, order=True)
class Service:
    service_id: str
    max_concurrency: int
    instances: tuple[Instance, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.service_id,
            "max_concurrency": self.max_concurrency,
            "instances": [instance.as_dict() for instance in self.instances],
        }


@dataclass(frozen=True, order=True)
class Dependency:
    kind: str
    target_id: str

    def label(self) -> str:
        return f"{self.kind}:{self.target_id}"


@dataclass(frozen=True)
class NormalizedRule:
    rule_id: str
    prefix: str
    target_service: str
    format_version: int
    revision_version: int
    instance_weights: tuple[tuple[str, int], ...]
    timeout_ms: int
    retries: int
    circuit_threshold: int | None
    cooldown_ms: int
    depends_on: tuple[Dependency, ...]

    def weights(self) -> dict[str, int]:
        return dict(self.instance_weights)

    def weight_for(self, instance_id: str) -> int:
        return dict(self.instance_weights).get(instance_id, 0)

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule_id,
            "prefix": self.prefix,
            "target_service": self.target_service,
            "instance_weights": dict(self.instance_weights),
            "timeout_ms": self.timeout_ms,
            "retries": self.retries,
            "circuit_threshold": self.circuit_threshold,
            "cooldown_ms": self.cooldown_ms,
            "depends_on": sorted(dependency.label() for dependency in self.depends_on),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.semantic_dict()
        payload["format_version"] = self.format_version
        payload["revision_version"] = self.revision_version
        return payload


@dataclass(frozen=True)
class DependencyGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": list(self.nodes),
            "edges": [
                {"from": source, "to": target, "kind": kind}
                for source, target, kind in self.edges
            ],
        }


@dataclass(frozen=True)
class NormalizedRevision:
    workspace_id: str
    revision_version: int
    rules: tuple[NormalizedRule, ...]
    digest: str
    canonical_text: str
    source_document: dict[str, Any]
    dependency_graph: DependencyGraph

    def rule_map(self) -> dict[str, NormalizedRule]:
        return {rule.rule_id: rule for rule in self.rules}

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "revision_version": self.revision_version,
            "digest": self.digest,
            "canonical_text": self.canonical_text,
            "dependency_graph": self.dependency_graph.as_dict(),
            "rules": [rule.as_dict() for rule in self.rules],
        }


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    method: str
    path: str
    index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "method": self.method,
            "path": self.path,
            "index": self.index,
        }


@dataclass(frozen=True)
class AttemptTrace:
    request_id: str
    attempt_no: int
    rule_id: str
    service_id: str
    instance_id: str
    outcome: str
    reason: str
    started_mono_ms: int
    ended_mono_ms: int
    status_code: int | None = None
    detail: str = ""

    @property
    def elapsed_ms(self) -> int:
        return max(0, self.ended_mono_ms - self.started_mono_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempt_no": self.attempt_no,
            "rule_id": self.rule_id,
            "service_id": self.service_id,
            "instance_id": self.instance_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "started_mono_ms": self.started_mono_ms,
            "ended_mono_ms": self.ended_mono_ms,
            "elapsed_ms": self.elapsed_ms,
            "status_code": self.status_code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    request_index: int
    method: str
    path: str
    rule_id: str | None
    service_id: str | None
    outcome: str
    reason: str
    attempt_count: int
    instance_sequence: tuple[str, ...]
    budget_consumed_ms: int
    wall_elapsed_ms: int
    traces: tuple[AttemptTrace, ...]
    unknown: bool = False
    rerun_required: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_index": self.request_index,
            "method": self.method,
            "path": self.path,
            "rule_id": self.rule_id,
            "service_id": self.service_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "attempt_count": self.attempt_count,
            "instance_sequence": list(self.instance_sequence),
            "budget_consumed_ms": self.budget_consumed_ms,
            "wall_elapsed_ms": self.wall_elapsed_ms,
            "traces": [trace.as_dict() for trace in self.traces],
            "unknown": self.unknown,
            "rerun_required": self.rerun_required,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RevisionChange:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]
    affected_services: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": list(self.changed),
            "affected_services": list(self.affected_services),
        }


@dataclass
class RoundSummary:
    round_id: str
    revision_version: int
    revision_digest: str
    dependency_graph_version: str
    instance_fingerprint: str
    results: tuple[RequestResult, ...]
    total_requests: int
    success: int
    failure: int
    timeout: int
    circuit_open: int
    unknown: int
    max_global_active: int
    max_service_active: dict[str, int] = field(default_factory=dict)
    change: RevisionChange | None = None
    snapshot_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "revision_version": self.revision_version,
            "revision_digest": self.revision_digest,
            "dependency_graph_version": self.dependency_graph_version,
            "instance_fingerprint": self.instance_fingerprint,
            "total_requests": self.total_requests,
            "success": self.success,
            "failure": self.failure,
            "timeout": self.timeout,
            "circuit_open": self.circuit_open,
            "unknown": self.unknown,
            "max_global_active": self.max_global_active,
            "max_service_active": dict(sorted(self.max_service_active.items())),
            "change": self.change.as_dict() if self.change else None,
            "snapshot_path": self.snapshot_path,
            "requests": [result.as_dict() for result in self.results],
        }


def parse_service_rows(rows: Any) -> tuple[Service, ...]:
    errors = []
    if not isinstance(rows, list):
        fail_rule([diag("E_RULE", "services", "services 必须是数组")])
    services = []
    seen_services: set[str] = set()
    for service_index, row in enumerate(rows):
        service_path = f"services[{service_index}]"
        if not isinstance(row, dict):
            errors.append(diag("E_RULE", service_path, "服务必须是对象"))
            continue
        service_id = row.get("id")
        if not isinstance(service_id, str) or not service_id:
            errors.append(diag("E_RULE", f"{service_path}.id", "服务 id 缺失或不是非空字符串"))
            continue
        if service_id in seen_services:
            errors.append(diag("E_RULE", f"{service_path}.id", "服务 id 重复", service_id))
            continue
        seen_services.add(service_id)
        limit = row.get("max_concurrency")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            errors.append(
                diag(
                    "E_RULE",
                    f"{service_path}.max_concurrency",
                    "max_concurrency 必须是正整数",
                    limit,
                )
            )
            limit = 1
        instance_rows = row.get("instances")
        if not isinstance(instance_rows, list) or not instance_rows:
            errors.append(
                diag("E_RULE", f"{service_path}.instances", "instances 必须是非空数组")
            )
            continue
        instances = []
        seen_instances: set[str] = set()
        for instance_index, instance_row in enumerate(instance_rows):
            instance_path = f"{service_path}.instances[{instance_index}]"
            if not isinstance(instance_row, dict):
                errors.append(diag("E_RULE", instance_path, "实例必须是对象"))
                continue
            instance_id = instance_row.get("id")
            if not isinstance(instance_id, str) or not instance_id:
                errors.append(
                    diag("E_RULE", f"{instance_path}.id", "实例 id 缺失或不是非空字符串")
                )
                continue
            if instance_id in seen_instances:
                errors.append(
                    diag("E_RULE", f"{instance_path}.id", "实例 id 在服务内重复", instance_id)
                )
                continue
            seen_instances.add(instance_id)
            host = instance_row.get("host")
            if host != "127.0.0.1":
                errors.append(
                    diag(
                        "E_RULE",
                        f"{instance_path}.host",
                        "仅允许本地回环地址 127.0.0.1",
                        host,
                    )
                )
                host = "127.0.0.1"
            port = instance_row.get("port")
            if (
                not isinstance(port, int)
                or isinstance(port, bool)
                or not (1024 <= port <= 65535)
            ):
                errors.append(
                    diag(
                        "E_RULE",
                        f"{instance_path}.port",
                        "port 必须是 1024..65535 的整数",
                        port,
                    )
                )
                port = 1
            instances.append(Instance(instance_id=instance_id, host=host, port=port))
        if instances:
            services.append(
                Service(
                    service_id=service_id,
                    max_concurrency=limit,
                    instances=tuple(sorted(instances)),
                )
            )
    if errors:
        fail_rule(errors, "服务与实例声明校验失败")
    return tuple(sorted(services))


def _normalize_depends_on(
    value: Any,
    path: str,
    errors: list,
) -> tuple[Dependency, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        errors.append(diag("E_RULE", path, "depends_on 必须是数组", value))
        return ()
    dependencies = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str) or ":" not in item:
            errors.append(
                diag(
                    "E_RULE",
                    item_path,
                    "依赖必须使用 rule:<id> 或 service:<id> 形式",
                    item,
                )
            )
            continue
        kind, target_id = item.split(":", 1)
        if kind not in {"rule", "service"} or not target_id:
            errors.append(
                diag(
                    "E_RULE",
                    item_path,
                    "依赖类型只能是 rule 或 service 且 id 非空",
                    item,
                )
            )
            continue
        dependencies.add(Dependency(kind=kind, target_id=target_id))
    return tuple(sorted(dependencies))


def normalize_revision(
    document: dict[str, Any],
    services: tuple[Service, ...],
    *,
    workspace_id: str,
    defaults: dict[str, Any] | None = None,
) -> NormalizedRevision:
    defaults = defaults or {}
    errors = []
    reference_errors = []
    if not isinstance(document, dict):
        fail_rule([diag("E_RULE", "$", "修订文档必须是 JSON 对象")])
    revision_version = document.get("revision")
    if (
        not isinstance(revision_version, int)
        or isinstance(revision_version, bool)
        or revision_version < 1
    ):
        errors.append(
            diag(
                "E_RULE",
                "revision",
                "revision 必须是大于等于 1 的整数",
                revision_version,
            )
        )
        revision_version = 1
    rows = document.get("rules")
    if not isinstance(rows, list) or not rows:
        errors.append(diag("E_RULE", "rules", "rules 必须是非空数组"))
        rows = []
    service_map = {service.service_id: service for service in services}
    default_timeout = defaults.get("timeout_ms", 500)
    default_retries = defaults.get("retries", 0)
    default_cooldown = defaults.get("cooldown_ms", 1000)
    rules = []
    seen_rule_ids: set[str] = set()
    for index, row in enumerate(rows):
        path = f"rules[{index}]"
        if not isinstance(row, dict):
            errors.append(diag("E_RULE", path, "规则必须是对象"))
            continue
        format_version = row.get("format_version")
        if format_version not in {1, 2, 3} or isinstance(format_version, bool):
            errors.append(
                diag(
                    "E_RULE",
                    f"{path}.format_version",
                    "未知规则格式版本，仅支持 1、2、3",
                    format_version,
                )
            )
            continue
        rule_id = row.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append(diag("E_RULE", f"{path}.id", "规则 id 缺失或为空"))
            continue
        if rule_id in seen_rule_ids:
            errors.append(diag("E_RULE", f"{path}.id", "规则 id 重复", rule_id))
            continue
        seen_rule_ids.add(rule_id)
        allowed_fields = {
            1: {"id", "format_version", "prefix", "target_service"},
            2: {
                "id",
                "format_version",
                "prefix",
                "target_service",
                "instance_weights",
                "timeout_ms",
            },
            3: {
                "id",
                "format_version",
                "prefix",
                "target_service",
                "instance_weights",
                "timeout_ms",
                "retries",
                "circuit_threshold",
                "cooldown_ms",
                "depends_on",
            },
        }[format_version]
        for field_name in sorted(set(row) - allowed_fields):
            errors.append(
                diag(
                    "E_RULE",
                    f"{path}.{field_name}",
                    f"v{format_version} 不支持该字段",
                    row[field_name],
                )
            )
        prefix = row.get("prefix")
        if not isinstance(prefix, str) or not prefix.startswith("/"):
            errors.append(
                diag("E_RULE", f"{path}.prefix", "prefix 必须是 / 开头的字符串", prefix)
            )
            prefix = "/"
        target_service = row.get("target_service")
        if not isinstance(target_service, str) or not target_service:
            errors.append(
                diag(
                    "E_RULE",
                    f"{path}.target_service",
                    "target_service 缺失或为空",
                    target_service,
                )
            )
            target_service = ""
        if target_service and target_service not in service_map:
            reference_errors.append(
                diag("E_REF", f"{path}.target_service", "目标服务不存在", target_service)
            )
        target_instances = (
            service_map[target_service].instances if target_service in service_map else ()
        )
        known_instance_ids = {instance.instance_id for instance in target_instances}
        timeout_ms = default_timeout
        retries = default_retries
        circuit_threshold: int | None = None
        cooldown_ms = default_cooldown
        depends_on: tuple[Dependency, ...] = ()
        raw_weights = row.get("instance_weights")
        if format_version == 1:
            instance_weights = tuple(
                (instance.instance_id, 1) for instance in sorted(target_instances)
            )
        else:
            if not isinstance(raw_weights, dict):
                errors.append(
                    diag(
                        "E_RULE",
                        f"{path}.instance_weights",
                        "v2/v3 必须提供实例权重对象",
                        raw_weights,
                    )
                )
                raw_weights = {}
            weight_entries = []
            for instance_id in (
                sorted(raw_weights, key=str) if target_service in service_map else []
            ):
                weight_path = f"{path}.instance_weights.{instance_id}"
                weight = raw_weights[instance_id]
                if not isinstance(instance_id, str):
                    errors.append(
                        diag("E_RULE", weight_path, "实例 id 必须是字符串", instance_id)
                    )
                    continue
                if instance_id not in known_instance_ids:
                    errors.append(
                        diag(
                            "E_RULE",
                            weight_path,
                            "权重引用了目标服务不存在的实例",
                            instance_id,
                        )
                    )
                    continue
                if not isinstance(weight, int) or isinstance(weight, bool):
                    errors.append(diag("E_RULE", weight_path, "权重必须是整数", weight))
                    continue
                if weight < 0:
                    errors.append(diag("E_RULE", weight_path, "权重不得为负数", weight))
                    continue
                weight_entries.append((instance_id, weight))
            weight_map = dict(weight_entries)
            for instance in target_instances:
                weight_map.setdefault(instance.instance_id, 0)
            instance_weights = tuple(sorted(weight_map.items()))
            if target_service in service_map and sum(weight_map.values()) == 0:
                errors.append(
                    diag(
                        "E_RULE",
                        f"{path}.instance_weights",
                        "同一目标服务的权重和不得为零",
                        0,
                    )
                )
            raw_timeout = row.get("timeout_ms")
            if (
                not isinstance(raw_timeout, int)
                or isinstance(raw_timeout, bool)
                or raw_timeout < 1
            ):
                errors.append(
                    diag(
                        "E_RULE",
                        f"{path}.timeout_ms",
                        "timeout_ms 必须是正整数毫秒预算",
                        raw_timeout,
                    )
                )
                timeout_ms = default_timeout
            else:
                timeout_ms = raw_timeout
        if format_version == 3:
            raw_retries = row.get("retries")
            if (
                not isinstance(raw_retries, int)
                or isinstance(raw_retries, bool)
                or raw_retries < 0
            ):
                errors.append(
                    diag(
                        "E_RULE",
                        f"{path}.retries",
                        "retries 必须是大于等于 0 的整数",
                        raw_retries,
                    )
                )
            else:
                retries = raw_retries
            raw_threshold = row.get("circuit_threshold")
            if (
                not isinstance(raw_threshold, int)
                or isinstance(raw_threshold, bool)
                or raw_threshold < 1
            ):
                errors.append(
                    diag(
                        "E_RULE",
                        f"{path}.circuit_threshold",
                        "circuit_threshold 必须是正整数",
                        raw_threshold,
                    )
                )
            else:
                circuit_threshold = raw_threshold
            raw_cooldown = row.get("cooldown_ms")
            if (
                not isinstance(raw_cooldown, int)
                or isinstance(raw_cooldown, bool)
                or raw_cooldown < 0
            ):
                errors.append(
                    diag(
                        "E_RULE",
                        f"{path}.cooldown_ms",
                        "cooldown_ms 必须是大于等于 0 的整数",
                        raw_cooldown,
                    )
                )
            else:
                cooldown_ms = raw_cooldown
            if "depends_on" not in row:
                errors.append(
                    diag("E_RULE", f"{path}.depends_on", "v3 必须提供 depends_on 数组")
                )
            depends_on = _normalize_depends_on(
                row.get("depends_on", []), f"{path}.depends_on", errors
            )
        rules.append(
            NormalizedRule(
                rule_id=rule_id,
                prefix=prefix,
                target_service=target_service,
                format_version=format_version,
                revision_version=revision_version,
                instance_weights=tuple(sorted(instance_weights)),
                timeout_ms=timeout_ms,
                retries=retries,
                circuit_threshold=circuit_threshold,
                cooldown_ms=cooldown_ms,
                depends_on=depends_on,
            )
        )
    if errors:
        fail_rule(errors)
    if reference_errors:
        fail_ref(reference_errors)
    if not rules:
        fail_rule([diag("E_RULE", "rules", "没有可执行规则")])
    sorted_rules = tuple(sorted(rules, key=lambda rule: rule.rule_id))
    graph = build_dependency_graph(sorted_rules, services)
    payload = {
        "workspace_id": workspace_id,
        "rules": [rule.semantic_dict() for rule in sorted_rules],
    }
    canonical_text = canonical_json(payload)
    digest = sha256_text(canonical_text)
    return NormalizedRevision(
        workspace_id=workspace_id,
        revision_version=revision_version,
        rules=sorted_rules,
        digest=digest,
        canonical_text=canonical_text,
        source_document=document,
        dependency_graph=graph,
    )


def build_dependency_graph(
    rules: Iterable[NormalizedRule], services: Iterable[Service]
) -> DependencyGraph:
    service_ids = {service.service_id for service in services}
    rule_map = {rule.rule_id: rule for rule in rules}
    errors = []
    edges = set()
    for rule in sorted(rule_map.values(), key=lambda item: item.rule_id):
        if rule.target_service not in service_ids:
            errors.append(
                diag(
                    "E_REF",
                    f"rules[{rule.rule_id}].target_service",
                    "目标服务不存在",
                    rule.target_service,
                )
            )
        else:
            edges.add((rule.rule_id, rule.target_service, "routes_to"))
        for dependency in rule.depends_on:
            if dependency.kind == "rule":
                if dependency.target_id not in rule_map:
                    errors.append(
                        diag(
                            "E_REF",
                            f"rules[{rule.rule_id}].depends_on",
                            "依赖规则不存在",
                            dependency.target_id,
                        )
                    )
                    continue
                edges.add((rule.rule_id, dependency.target_id, "depends_on_rule"))
            elif dependency.kind == "service":
                if dependency.target_id not in service_ids:
                    errors.append(
                        diag(
                            "E_REF",
                            f"rules[{rule.rule_id}].depends_on",
                            "依赖服务不存在",
                            dependency.target_id,
                        )
                    )
                    continue
                edges.add((rule.rule_id, dependency.target_id, "depends_on_service"))
    if errors:
        fail_ref(errors)
    cycle = find_rule_cycle(rule_map)
    if cycle:
        fail_cycle(cycle)
    nodes = sorted(
        [f"rule:{rule_id}" for rule_id in rule_map]
        + [f"service:{service_id}" for service_id in service_ids]
    )
    edge_rows = tuple(
        sorted(
            (
                (f"rule:{source}", _typed_target(target, kind), kind)
                for source, target, kind in edges
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
    )
    version = sha256_json(
        {
            "nodes": nodes,
            "edges": [
                {"from": source, "to": target, "kind": kind}
                for source, target, kind in edge_rows
            ],
        }
    )
    return DependencyGraph(nodes=tuple(nodes), edges=edge_rows, version=version)


def _typed_target(target: str, kind: str) -> str:
    if kind in {"routes_to", "depends_on_service"}:
        return f"service:{target}"
    return f"rule:{target}"


def find_rule_cycle(rule_map: dict[str, NormalizedRule]) -> tuple[str, ...]:
    adjacency = {
        rule_id: sorted(
            dependency.target_id
            for dependency in rule.depends_on
            if dependency.kind == "rule"
        )
        for rule_id, rule in rule_map.items()
    }
    state: dict[str, int] = {}
    stack: list[str] = []
    positions: dict[str, int] = {}

    def visit(node: str) -> tuple[str, ...] | None:
        state[node] = 1
        positions[node] = len(stack)
        stack.append(node)
        for neighbour in adjacency.get(node, ()):
            if neighbour not in rule_map:
                continue
            if state.get(neighbour, 0) == 0:
                found = visit(neighbour)
                if found:
                    return found
            elif state.get(neighbour) == 1:
                start = positions[neighbour]
                cycle = stack[start:]
                minimum = min(cycle)
                rotate = cycle.index(minimum)
                rotated = cycle[rotate:] + cycle[:rotate]
                return tuple(rotated + [rotated[0]])
        stack.pop()
        positions.pop(node, None)
        state[node] = 2
        return None

    for rule_id in sorted(rule_map):
        if state.get(rule_id, 0) == 0:
            found = visit(rule_id)
            if found:
                return found
    return ()


def diff_revisions(
    previous: NormalizedRevision | None,
    current: NormalizedRevision,
) -> RevisionChange:
    if previous is None:
        added = tuple(rule.rule_id for rule in current.rules)
        affected = sorted({rule.target_service for rule in current.rules})
        return RevisionChange(
            added=added, removed=(), changed=(), affected_services=tuple(affected)
        )
    before = previous.rule_map()
    after = current.rule_map()
    added = tuple(sorted(set(after) - set(before)))
    removed = tuple(sorted(set(before) - set(after)))
    changed = tuple(
        sorted(
            rule_id
            for rule_id in set(before) & set(after)
            if before[rule_id].semantic_dict() != after[rule_id].semantic_dict()
        )
    )
    affected = set()
    for rule_id in set(added) | set(changed):
        affected.add(after[rule_id].target_service)
    for rule_id in removed:
        affected.add(before[rule_id].target_service)
    return RevisionChange(
        added=added,
        removed=removed,
        changed=changed,
        affected_services=tuple(sorted(affected)),
    )


def validate_deleted_rules(
    previous_rules: Iterable[NormalizedRule],
    new_rules: Iterable[NormalizedRule],
) -> None:
    before = {rule.rule_id: rule for rule in previous_rules}
    after = {rule.rule_id: rule for rule in new_rules}
    removed = sorted(set(before) - set(after))
    errors = []
    for removed_rule_id in removed:
        referrers = sorted(
            rule.rule_id
            for rule in after.values()
            if any(
                dependency.kind == "rule"
                and dependency.target_id == removed_rule_id
                for dependency in rule.depends_on
            )
        )
        if referrers:
            errors.append(
                diag(
                    "E_REF",
                    f"rules[{removed_rule_id}]",
                    "规则仍被其他规则引用，不能删除",
                    removed_rule_id,
                    referrers=referrers,
                )
            )
    if errors:
        fail_ref(errors, "删除规则仍被引用")


def validate_deleted_rule_document(
    previous_rules: Iterable[NormalizedRule],
    document: dict[str, Any],
) -> None:
    """Reject referenced deletions before full normalization reports E_REF.

    A new revision that deletes a rule but leaves a dangling v3 rule reference
    must still name the referrers. This pre-check operates on the raw document
    so that the deletion diagnostic wins over a generic dangling reference.
    """

    before = {rule.rule_id for rule in previous_rules}
    rows = document.get("rules", []) if isinstance(document, dict) else []
    after = {
        row.get("id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    removed = sorted(before - after)
    if not removed:
        return
    errors = []
    for removed_rule_id in removed:
        referrers = []
        for row in rows:
            if not isinstance(row, dict) or row.get("format_version") != 3:
                continue
            dependencies = row.get("depends_on", [])
            if not isinstance(dependencies, list):
                continue
            if f"rule:{removed_rule_id}" in dependencies:
                referrers.append(str(row.get("id", "<unknown>")))
        if referrers:
            errors.append(
                diag(
                    "E_REF",
                    f"rules[{removed_rule_id}]",
                    "规则仍被其他规则引用，不能删除",
                    removed_rule_id,
                    referrers=sorted(referrers),
                )
            )
    if errors:
        fail_ref(errors, "删除规则仍被引用")


def instance_fingerprint(services: Iterable[Service]) -> str:
    return sha256_json(
        [
            {
                "id": service.service_id,
                "max_concurrency": service.max_concurrency,
                "instances": [instance.as_dict() for instance in service.instances],
            }
            for service in sorted(services)
        ]
    )
