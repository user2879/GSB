from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import PipelineError
from .models import (
    KNOWN_LEVELS,
    NormalizedAlert,
    Provenance,
    RootRule,
    RootRules,
    ServiceGraph,
)
from .util import canonical_json, digest_file, digest_json, load_json_file


SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
MAX_SAFE_TIMESTAMP_MS = (2**53) - 1
SERVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
FINGERPRINT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:=/-]{0,127}$")
STRATEGY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")

ALIASES = {
    "version": ("version", "format_version", "schema_version"),
    "timestamp_ms": ("timestamp_ms", "timestamp"),
    "service_id": ("service_id", "service"),
    "level": ("level", "alert_level"),
    "fingerprint": ("fingerprint", "fingerprint_key"),
    "silence_window_ms": ("silence_window_ms", "silence_ms", "window_ms"),
    "root_candidates": ("root_candidates", "root_candidate_refs"),
    "strategy_id": ("strategy_id", "denoise_strategy", "silence_strategy"),
    "tags": ("tags", "labels"),
}


@dataclass(frozen=True)
class LoadedDataset:
    graph: ServiceGraph
    rules: RootRules
    alerts: tuple[NormalizedAlert, ...]
    alerts_digest: str
    services_digest: str
    rules_digest: str

    @property
    def alert_count(self) -> int:
        return len(self.alerts)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _spec_error(message: str, field: str, line: int = 1, **details: Any) -> PipelineError:
    return PipelineError("E_SPEC", message, line=line, field=field, details=details)


def _field_error(message: str, field: str, line: int, **details: Any) -> PipelineError:
    return PipelineError("E_FIELD", message, line=line, field=field, details=details)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = load_json_file(path)
    except FileNotFoundError as exc:
        raise PipelineError("E_IO", f"{label} file does not exist: {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        line = getattr(exc, "lineno", 1)
        raise _spec_error(
            f"{label} JSON is invalid: {exc}", "$", line=line, file=str(path)
        ) from exc
    if not isinstance(value, dict):
        raise _spec_error(f"{label} root must be a JSON object", "$", line=1)
    return value


def _parse_service_names(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise _spec_error("services must be a non-empty array", "services")
    services: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            service_id = item
        elif isinstance(item, dict):
            service_id = item.get("service_id")
        else:
            service_id = None
        if not isinstance(service_id, str) or not SERVICE_ID_PATTERN.fullmatch(service_id):
            raise _spec_error(
                "service_id must match the service identifier format",
                f"services[{index}].service_id",
                service_id=service_id,
            )
        if service_id in seen:
            raise _spec_error(
                "service identifiers must be unique",
                f"services[{index}].service_id",
                service_id=service_id,
            )
        seen.add(service_id)
        services.append(service_id)
    return tuple(services)


def _parse_dependencies(raw: Any, services: set[str]) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, list):
        raise _spec_error("dependencies must be an array", "dependencies")
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            source = item.get("from", item.get("dependent"))
            target = item.get("to", item.get("dependency"))
        elif isinstance(item, list) and len(item) == 2:
            source, target = item
        else:
            source = target = None
        if not isinstance(source, str) or not isinstance(target, str):
            raise _spec_error(
                "dependency requires string endpoints 'from' and 'to'",
                f"dependencies[{index}]",
            )
        if source not in services:
            raise _spec_error(
                "dependency source service does not exist",
                f"dependencies[{index}].from",
                service_id=source,
            )
        if target not in services:
            raise _spec_error(
                "dependency target service does not exist",
                f"dependencies[{index}].to",
                service_id=target,
            )
        edge = (source, target)
        if edge in seen:
            raise _spec_error(
                "dependency edges must be unique",
                f"dependencies[{index}]",
                edge=list(edge),
            )
        seen.add(edge)
        result.append(edge)
    return tuple(result)


def _topological_order(
    services: tuple[str, ...], dependencies: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], dict[str, int], dict[str, int]]:
    adjacency: dict[str, list[str]] = {service: [] for service in services}
    indegree: dict[str, int] = {service: 0 for service in services}
    for source, target in dependencies:
        adjacency[source].append(target)
        indegree[target] += 1
    for service in adjacency:
        adjacency[service].sort()
    ready = [service for service in services if indegree[service] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    level: dict[str, int] = {service: 0 for service in services}
    while ready:
        service = heapq.heappop(ready)
        order.append(service)
        for target in adjacency[service]:
            level[target] = max(level[target], level[service] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != len(services):
        cycle_services = sorted(service for service, degree in indegree.items() if degree > 0)
        raise _spec_error(
            "service dependency graph must be acyclic",
            "dependencies",
            cycle_services=cycle_services,
        )
    rank = {service: index for index, service in enumerate(order)}
    return tuple(order), rank, level


def load_service_graph(path: str | Path) -> ServiceGraph:
    source_path = Path(path)
    raw = _load_json_object(source_path, "service graph")
    version = raw.get("version")
    if not _is_int(version) or version <= 0:
        raise _spec_error("graph version must be a positive integer", "version")
    services = _parse_service_names(raw.get("services"))
    dependencies = _parse_dependencies(raw.get("dependencies"), set(services))
    topology, topology_rank, topology_level = _topological_order(services, dependencies)
    return ServiceGraph(
        version=version,
        services=services,
        dependencies=dependencies,
        source_digest=digest_file(source_path),
        topology=topology,
        topology_rank=topology_rank,
        topology_level=topology_level,
    )


def _parse_directions(raw: Any, field: str) -> tuple[str, ...]:
    if raw is None:
        raw = ["upstream", "downstream"]
    if not isinstance(raw, list) or not raw:
        raise _spec_error("directions must be a non-empty array", field)
    directions: list[str] = []
    for index, direction in enumerate(raw):
        if direction not in {"upstream", "downstream"}:
            raise _spec_error(
                "direction must be 'upstream' or 'downstream'",
                f"{field}[{index}]",
            )
        if direction not in directions:
            directions.append(direction)
    return tuple(directions)


def _parse_defaults(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _spec_error("rules defaults must be an object", "defaults")
    directions = _parse_directions(raw.get("directions"), "defaults.directions")
    max_depth = raw.get("max_depth", 2)
    include_self = raw.get("include_self", True)
    if not _is_int(max_depth) or max_depth < 0:
        raise _spec_error("defaults.max_depth must be a non-negative integer", "defaults.max_depth")
    if not isinstance(include_self, bool):
        raise _spec_error("defaults.include_self must be a boolean", "defaults.include_self")
    return {
        "directions": list(directions),
        "max_depth": max_depth,
        "include_self": include_self,
    }


def load_root_rules(path: str | Path, graph: ServiceGraph) -> RootRules:
    source_path = Path(path)
    raw = _load_json_object(source_path, "root rules")
    version = raw.get("version")
    if not _is_int(version) or version <= 0:
        raise _spec_error("rules version must be a positive integer", "version")
    defaults = _parse_defaults(raw.get("defaults"))
    rules_raw = raw.get("rules", [])
    if not isinstance(rules_raw, list):
        raise _spec_error("rules must be an array", "rules")
    known_services = set(graph.services)
    seen_rule_ids: set[str] = set()
    assigned_services: set[str] = set()
    rules: list[RootRule] = []
    for index, item in enumerate(rules_raw):
        field = f"rules[{index}]"
        if not isinstance(item, dict):
            raise _spec_error("root rule must be an object", field)
        rule_id = item.get("rule_id")
        if not isinstance(rule_id, str) or not STRATEGY_PATTERN.fullmatch(rule_id):
            raise _spec_error("rule_id has an invalid format", f"{field}.rule_id")
        if rule_id in seen_rule_ids:
            raise _spec_error("rule_id must be unique", f"{field}.rule_id", rule_id=rule_id)
        seen_rule_ids.add(rule_id)
        services_raw = item.get("services")
        if not isinstance(services_raw, list) or not services_raw:
            raise _spec_error("rule services must be a non-empty array", f"{field}.services")
        rule_services: list[str] = []
        for service_index, service in enumerate(services_raw):
            if not isinstance(service, str):
                raise _spec_error(
                    "rule service reference must be a string",
                    f"{field}.services[{service_index}]",
                )
            if service not in known_services:
                raise _spec_error(
                    "rule references a service absent from the graph",
                    f"{field}.services[{service_index}]",
                    service_id=service,
                )
            if service in assigned_services:
                raise _spec_error(
                    "a service may be covered by only one explicit root rule",
                    f"{field}.services[{service_index}]",
                    service_id=service,
                )
            if service in rule_services:
                raise _spec_error(
                    "rule service references must be unique",
                    f"{field}.services[{service_index}]",
                    service_id=service,
                )
            rule_services.append(service)
            assigned_services.add(service)
        directions = _parse_directions(item.get("directions"), f"{field}.directions")
        max_depth = item.get("max_depth", defaults["max_depth"])
        include_self = item.get("include_self", defaults["include_self"])
        if not _is_int(max_depth) or max_depth < 0:
            raise _spec_error(
                "rule max_depth must be a non-negative integer", f"{field}.max_depth"
            )
        if not isinstance(include_self, bool):
            raise _spec_error("rule include_self must be a boolean", f"{field}.include_self")
        rules.append(
            RootRule(
                rule_id=rule_id,
                services=tuple(rule_services),
                directions=directions,
                max_depth=max_depth,
                include_self=include_self,
            )
        )
    return RootRules(
        version=version,
        rules=tuple(rules),
        source_digest=digest_file(source_path),
        defaults=defaults,
    )


def _find_alias(record: dict[str, Any], field: str, line: int) -> tuple[Any, str | None]:
    present = [(alias, record[alias]) for alias in ALIASES[field] if alias in record]
    if not present:
        return None, None
    first_name, first_value = present[0]
    for alias, value in present[1:]:
        if canonical_json(value) != canonical_json(first_value):
            raise _field_error(
                f"conflicting aliases supplied for {field}: {first_name}, {alias}",
                field,
                line,
            )
    return first_value, first_name


def _required_alias(
    record: dict[str, Any], field: str, line: int, version: int
) -> tuple[Any, str]:
    value, alias = _find_alias(record, field, line)
    if alias is None:
        raise _field_error(
            f"required field {field} is missing for v{version}", field, line
        )
    return value, alias


def _validate_tags(value: Any, line: int) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _field_error("tags must be an object of string labels", "tags", line)
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise _field_error("tag keys must be non-empty strings", "tags", line)
        if not isinstance(item, (str, int, float, bool)) or item is None:
            raise _field_error("tag values must be scalar JSON values", f"tags.{key}", line)
        result[key] = str(item)
    return dict(sorted(result.items()))


def decode_alert(
    record: Any,
    line_number: int,
    graph: ServiceGraph,
    config: PipelineConfig,
    previous_timestamp_ms: int | None,
) -> NormalizedAlert:
    if not isinstance(record, dict):
        raise _field_error("alert record must be a JSON object", "$", line_number)
    version_value, version_alias = _find_alias(record, "version", line_number)
    if version_alias is None:
        raise _field_error("format version is missing", "version", line_number)
    if not _is_int(version_value):
        raise _field_error("format version must be an integer", version_alias, line_number)
    version = version_value
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise _field_error(
            f"unsupported alert format version: {version}",
            version_alias,
            line_number,
            supported_versions=list(SUPPORTED_FORMAT_VERSIONS),
        )

    timestamp, _ = _required_alias(record, "timestamp_ms", line_number, version)
    service_id, _ = _required_alias(record, "service_id", line_number, version)
    level, _ = _required_alias(record, "level", line_number, version)
    if not _is_int(timestamp):
        raise _field_error("timestamp_ms must be an integer", "timestamp_ms", line_number)
    if not isinstance(service_id, str) or service_id not in set(graph.services):
        raise _spec_error(
            "alert references an unknown service", "service_id", line_number, service_id=service_id
        )
    if level not in KNOWN_LEVELS:
        raise _spec_error(
            "alert level is not in the predefined set",
            "level",
            line_number,
            level=level,
            allowed_levels=list(KNOWN_LEVELS),
        )
    if previous_timestamp_ms is not None and timestamp < previous_timestamp_ms:
        raise _spec_error(
            "alert timestamps must be monotonically non-decreasing",
            "timestamp_ms",
            line_number,
            previous_timestamp_ms=previous_timestamp_ms,
            timestamp_ms=timestamp,
        )
    if timestamp < 0 or timestamp > MAX_SAFE_TIMESTAMP_MS:
        raise PipelineError(
            "E_WINDOW",
            "timestamp is outside the safe millisecond range",
            line=line_number,
            field="timestamp_ms",
            details={
                "timestamp_ms": timestamp,
                "min": 0,
                "max": MAX_SAFE_TIMESTAMP_MS,
            },
        )

    provenance: dict[str, Provenance] = {
        "timestamp_ms": Provenance("record", "record.timestamp_ms"),
        "service_id": Provenance("record", "record.service_id"),
        "level": Provenance("record", "record.level"),
    }

    fingerprint_value, fingerprint_alias = _find_alias(record, "fingerprint", line_number)
    if fingerprint_alias is None:
        if version >= 2:
            raise _field_error(
                f"required field fingerprint is missing for v{version}",
                "fingerprint",
                line_number,
            )
        fingerprint = f"derived:{service_id}:{level}"
        provenance["fingerprint"] = Provenance(
            "default", "v1.fingerprint=derived:<service_id>:<level>"
        )
    else:
        fingerprint = fingerprint_value
        provenance["fingerprint"] = Provenance("record", f"record.{fingerprint_alias}")
    if not isinstance(fingerprint, str) or not FINGERPRINT_PATTERN.fullmatch(fingerprint):
        raise _spec_error(
            "fingerprint key has an invalid format",
            "fingerprint",
            line_number,
            fingerprint=fingerprint,
        )

    window_value, window_alias = _find_alias(record, "silence_window_ms", line_number)
    if window_alias is None:
        if version >= 2:
            raise _field_error(
                f"required field silence_window_ms is missing for v{version}",
                "silence_window_ms",
                line_number,
            )
        silence_window_ms = config.default_silence_window_ms
        provenance["silence_window_ms"] = Provenance(
            "default", "policy.default_silence_window_ms"
        )
    else:
        silence_window_ms = window_value
        provenance["silence_window_ms"] = Provenance(
            "record", f"record.{window_alias}"
        )
    if not _is_int(silence_window_ms):
        raise _field_error(
            "silence_window_ms must be an integer", "silence_window_ms", line_number
        )
    if silence_window_ms < 0:
        raise _field_error(
            "silence_window_ms must not be negative",
            "silence_window_ms",
            line_number,
            silence_window_ms=silence_window_ms,
        )
    if silence_window_ms > MAX_SAFE_TIMESTAMP_MS:
        raise _field_error(
            "silence_window_ms exceeds the safe integer range",
            "silence_window_ms",
            line_number,
        )
    window_end = timestamp + silence_window_ms
    if window_end > MAX_SAFE_TIMESTAMP_MS:
        raise PipelineError(
            "E_WINDOW",
            "silence window conflicts with timestamp safe range",
            line=line_number,
            field="silence_window_ms",
            details={
                "timestamp_ms": timestamp,
                "silence_window_ms": silence_window_ms,
                "window_end_ms": window_end,
                "max": MAX_SAFE_TIMESTAMP_MS,
            },
        )

    root_value, root_alias = _find_alias(record, "root_candidates", line_number)
    if root_alias is None:
        if version >= 3:
            raise _field_error(
                "required field root_candidates is missing for v3",
                "root_candidates",
                line_number,
            )
        root_candidates: tuple[str, ...] = tuple()
        provenance["root_candidates"] = Provenance(
            "default", f"v{version}.root_candidates=[]"
        )
    else:
        if not isinstance(root_value, list):
            raise _field_error("root_candidates must be an array", "root_candidates", line_number)
        refs: list[str] = []
        for index, ref in enumerate(root_value):
            if not isinstance(ref, str):
                raise _field_error(
                    "root candidate reference must be a string",
                    f"root_candidates[{index}]",
                    line_number,
                )
            if ref not in set(graph.services):
                raise _field_error(
                    "root candidate reference is dangling",
                    f"root_candidates[{index}]",
                    line_number,
                    service_id=ref,
                )
            if ref in refs:
                raise _field_error(
                    "root candidate references must be unique",
                    f"root_candidates[{index}]",
                    line_number,
                    service_id=ref,
                )
            refs.append(ref)
        root_candidates = tuple(refs)
        provenance["root_candidates"] = Provenance("record", f"record.{root_alias}")

    strategy_value, strategy_alias = _find_alias(record, "strategy_id", line_number)
    if strategy_alias is None:
        if version >= 3:
            raise _field_error(
                "required field strategy_id is missing for v3", "strategy_id", line_number
            )
        strategy_id = "legacy-default"
        provenance["strategy_id"] = Provenance(
            "default", f"v{version}.strategy_id=legacy-default"
        )
    else:
        if not isinstance(strategy_value, str) or not STRATEGY_PATTERN.fullmatch(
            strategy_value
        ):
            raise _field_error("strategy_id has an invalid format", "strategy_id", line_number)
        strategy_id = strategy_value
        provenance["strategy_id"] = Provenance("record", f"record.{strategy_alias}")

    tags_value, tags_alias = _find_alias(record, "tags", line_number)
    tags = _validate_tags(tags_value, line_number)
    if tags_alias is None:
        provenance["tags"] = Provenance("default", f"v{version}.tags={{}}")
    else:
        provenance["tags"] = Provenance("record", f"record.{tags_alias}")

    return NormalizedAlert(
        line_number=line_number,
        timestamp_ms=timestamp,
        service_id=service_id,
        level=level,
        fingerprint=fingerprint,
        tags=tags,
        silence_window_ms=silence_window_ms,
        root_candidate_refs=root_candidates,
        strategy_id=strategy_id,
        format_version=version,
        declared_version=config.declaration_version,
        provenance=provenance,
        raw_digest=digest_json(record),
    )


def encode_alert(alert: NormalizedAlert, version: int | None = None) -> dict[str, Any]:
    target = alert.format_version if version is None else version
    if target not in SUPPORTED_FORMAT_VERSIONS:
        raise PipelineError(
            "E_FIELD",
            f"unsupported alert format version: {target}",
            field="version",
        )
    base: dict[str, Any] = {
        "version": target,
        "timestamp_ms": alert.timestamp_ms,
        "service_id": alert.service_id,
        "level": alert.level,
    }
    if alert.tags:
        base["tags"] = dict(sorted(alert.tags.items()))
    if target >= 2:
        base["fingerprint"] = alert.fingerprint
        base["silence_window_ms"] = alert.silence_window_ms
    if target >= 3:
        base["root_candidates"] = list(alert.root_candidate_refs)
        base["strategy_id"] = alert.strategy_id
    return base


def load_alert_stream(
    path: str | Path, graph: ServiceGraph, config: PipelineConfig
) -> tuple[tuple[NormalizedAlert, ...], str]:
    source_path = Path(path)
    try:
        handle = source_path.open("r", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise PipelineError("E_IO", f"alert stream does not exist: {source_path}") from exc
    alerts: list[NormalizedAlert] = []
    previous_timestamp: int | None = None
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _spec_error(
                    f"alert NDJSON record is invalid JSON: {exc.msg}",
                    "$",
                    line=line_number,
                    column=exc.colno,
                ) from exc
            alert = decode_alert(record, line_number, graph, config, previous_timestamp)
            previous_timestamp = alert.timestamp_ms
            alerts.append(alert)
    if not alerts:
        raise _spec_error(
            "alert stream must contain at least one record",
            "alerts",
            line=1,
        )
    return tuple(alerts), digest_file(source_path)


def load_dataset(config: PipelineConfig) -> LoadedDataset:
    graph = load_service_graph(config.services_path)
    rules = load_root_rules(config.rules_path, graph)
    alerts, alerts_digest = load_alert_stream(config.alerts_path, graph, config)
    return LoadedDataset(
        graph=graph,
        rules=rules,
        alerts=alerts,
        alerts_digest=alerts_digest,
        services_digest=digest_file(config.services_path),
        rules_digest=digest_file(config.rules_path),
    )
