from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import DriftError, violation
from .models import (
    AGGREGATION_KINDS,
    BOUNDARY_RULES,
    METRICS,
    AgentSpec,
    AggregationSpec,
    Criteria,
    HostSpec,
    Snapshot,
    SnapshotChain,
    is_int,
)


def _read_json(path: str | Path, code: str) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise DriftError(code, f"cannot read {source}", violations=[violation("$", str(exc))]) from exc
    except json.JSONDecodeError as exc:
        item = violation("$", "invalid JSON", actual=str(exc))
        item["line"] = exc.lineno
        item["column"] = exc.colno
        raise DriftError(code, f"cannot parse {source}", violations=[item]) from exc


def _raw_snapshots(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        value = data.get("snapshots")
        return value if isinstance(value, list) else []
    return []


def _metric_int(
    raw: Mapping[str, Any],
    path: str,
    violations: list[dict[str, Any]],
    *,
    maximum: int | None = None,
) -> int | None:
    if not is_int(raw):
        violations.append(violation(path, "must be an integer", actual=raw))
        return None
    value = int(raw)
    if value < 0:
        violations.append(violation(path, "must be non-negative", actual=value))
    if maximum is not None and value > maximum:
        violations.append(violation(path, f"must be <= {maximum}", actual=value))
    return value


def _parse_tolerances(
    raw: Any,
    path: str,
    violations: list[dict[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    if is_int(raw):
        for metric in METRICS:
            value = _metric_int(raw, path, violations, maximum=10000)
            if value is not None:
                result[metric] = value
        return result
    if not isinstance(raw, dict):
        violations.append(violation(path, "tolerance must be an integer or metric map", actual=raw))
        return result
    unknown = sorted(set(raw) - set(METRICS))
    for metric in unknown:
        violations.append(violation(f"{path}.{metric}", "unknown metric"))
    for metric in METRICS:
        if metric not in raw:
            violations.append(violation(f"{path}.{metric}", "missing tolerance"))
            continue
        value = _metric_int(raw[metric], f"{path}.{metric}", violations, maximum=10000)
        if value is not None:
            result[metric] = value
    return result


def _parse_expected(
    raw: Any,
    path: str,
    violations: list[dict[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(raw, dict):
        violations.append(violation(path, "expected limits must be a metric map", actual=raw))
        return result
    unknown = sorted(set(raw) - set(METRICS))
    for metric in unknown:
        violations.append(violation(f"{path}.{metric}", "unknown metric"))
    for metric in METRICS:
        if metric not in raw:
            violations.append(violation(f"{path}.{metric}", "missing expected limit"))
            continue
        value = _metric_int(raw[metric], f"{path}.{metric}", violations)
        if value is not None:
            if value <= 0:
                violations.append(violation(f"{path}.{metric}", "expected limit must be positive", actual=value))
            else:
                result[metric] = value
    return result


def parse_snapshot_chain(data: Any) -> SnapshotChain:
    violations: list[dict[str, Any]] = []
    if isinstance(data, dict) and "schema_version" in data and data["schema_version"] != 1:
        violations.append(
            violation("$.schema_version", "unsupported schema version", actual=data["schema_version"], expected="1")
        )
    raw_items = _raw_snapshots(data)
    if not raw_items:
        raise DriftError.snapshot([violation("$.snapshots", "must be a non-empty array")])

    pending: dict[int, Snapshot] = {}
    duplicate_seqs: set[int] = set()
    seen_seqs: set[int] = set()
    root_tolerance: Any = data.get("tolerance_bps") if isinstance(data, dict) else None

    for index, raw in enumerate(raw_items):
        base = f"$.snapshots[{index}]"
        if not isinstance(raw, dict):
            violations.append(violation(base, "snapshot must be an object", actual=raw))
            continue
        seq_raw = raw.get("seq", raw.get("snapshot_seq"))
        if not is_int(seq_raw):
            violations.append(violation(f"{base}.seq", "must be an integer", actual=seq_raw))
            continue
        seq = int(seq_raw)
        if seq < 0:
            violations.append(violation(f"{base}.seq", "must be non-negative", actual=seq))
        if seq in seen_seqs:
            duplicate_seqs.add(seq)
            violations.append(violation(f"{base}.seq", "snapshot sequence must be unique", actual=seq))
        seen_seqs.add(seq)

        parent_raw = raw.get("parent_seq", raw.get("parent"))
        if parent_raw is not None and not is_int(parent_raw):
            violations.append(violation(f"{base}.parent_seq", "must be an integer or null", actual=parent_raw))
            parent_seq: int | None = None
        else:
            parent_seq = int(parent_raw) if parent_raw is not None else None

        raw_hosts = raw.get("hosts")
        host_entries: list[tuple[str, Any, str]] = []
        if isinstance(raw_hosts, dict):
            for host_id in sorted(raw_hosts):
                host_entries.append((str(host_id), raw_hosts[host_id], f"{base}.hosts.{host_id}"))
        elif isinstance(raw_hosts, list):
            used_hosts: set[str] = set()
            for host_index, entry in enumerate(raw_hosts):
                host_path = f"{base}.hosts[{host_index}]"
                if not isinstance(entry, dict):
                    violations.append(violation(host_path, "host entry must be an object", actual=entry))
                    continue
                host_id = entry.get("host_id", entry.get("host"))
                if not isinstance(host_id, str) or not host_id:
                    violations.append(violation(f"{host_path}.host_id", "must be a non-empty string", actual=host_id))
                    continue
                if host_id in used_hosts:
                    violations.append(violation(f"{host_path}.host_id", "duplicate host", actual=host_id))
                    continue
                used_hosts.add(host_id)
                host_entries.append((host_id, entry, host_path))
        else:
            violations.append(violation(f"{base}.hosts", "must be a host map or array", actual=raw_hosts))

        hosts: dict[str, HostSpec] = {}
        if not host_entries:
            violations.append(violation(f"{base}.hosts", "at least one host is required"))
        for host_id, entry, host_path in host_entries:
            if not isinstance(entry, dict):
                violations.append(violation(host_path, "host declaration must be an object", actual=entry))
                continue
            expected_raw = entry.get("expected", entry.get("limits"))
            if expected_raw is None and any(metric in entry for metric in METRICS):
                expected_raw = {metric: entry[metric] for metric in METRICS if metric in entry}
            tolerance_raw = entry.get("tolerance_bps", entry.get("tolerance", root_tolerance))
            if tolerance_raw is None:
                violations.append(violation(f"{host_path}.tolerance_bps", "missing tolerance"))
                continue
            expected = _parse_expected(expected_raw, f"{host_path}.expected", violations)
            tolerance = _parse_tolerances(tolerance_raw, f"{host_path}.tolerance_bps", violations)
            if len(expected) == len(METRICS) and len(tolerance) == len(METRICS):
                hosts[host_id] = HostSpec(host_id=host_id, expected=expected, tolerance_bps=tolerance)
        if seq not in duplicate_seqs and len(hosts) == len(host_entries):
            pending[seq] = Snapshot(seq=seq, parent_seq=parent_seq, hosts=hosts)

    for seq, snapshot in pending.items():
        if snapshot.parent_seq is not None and snapshot.parent_seq not in pending:
            if snapshot.parent_seq in duplicate_seqs or snapshot.parent_seq not in seen_seqs:
                violations.append(
                    violation(
                        f"$.snapshots[{seq}].parent_seq",
                        "parent snapshot does not exist",
                        actual=snapshot.parent_seq,
                    )
                )

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(seq: int, stack: list[int]) -> None:
        if seq in visited:
            return
        if seq in visiting:
            cycle = stack[stack.index(seq) :] + [seq]
            violations.append(
                violation("$.snapshots", "parent cycle detected", actual=" -> ".join(str(item) for item in cycle))
            )
            return
        visiting.add(seq)
        parent = pending[seq].parent_seq
        if parent is not None and parent in pending:
            visit(parent, stack + [seq])
        visiting.remove(seq)
        visited.add(seq)

    for seq in sorted(pending):
        visit(seq, [])

    if violations:
        raise DriftError.snapshot(_stable_violations(violations))
    return SnapshotChain(snapshots={seq: pending[seq] for seq in sorted(pending)})


def _stable_violations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        unique[key] = item
    return [unique[key] for key in sorted(unique)]


def default_criteria() -> Criteria:
    return Criteria(
        global_concurrency=4,
        per_agent_concurrency=1,
        min_sources=2,
        scale_exp=0,
        socket_timeout_ms=1000,
        handshake_timeout_ms=5000,
        max_retries=1,
        retry_backoff_ms=25,
        boundary_rule="tolerance_floor",
    )


def parse_criteria(raw: Any) -> Criteria:
    violations: list[dict[str, Any]] = []
    if raw is None:
        return default_criteria()
    if not isinstance(raw, dict):
        raise DriftError.agent([violation("$.criteria", "criteria must be an object", actual=raw)])
    if "schema_version" in raw and raw["schema_version"] != 1:
        violations.append(
            violation("$.criteria.schema_version", "unsupported schema version", actual=raw["schema_version"])
        )
    base = default_criteria().to_dict()
    values: dict[str, Any] = {}
    for key, default in base.items():
        value = raw.get(key, default)
        if key == "boundary_rule":
            if value not in BOUNDARY_RULES:
                violations.append(
                    violation(f"$.criteria.{key}", "unsupported boundary rule", actual=value, expected=str(BOUNDARY_RULES))
                )
            else:
                values[key] = value
            continue
        if not is_int(value):
            violations.append(violation(f"$.criteria.{key}", "must be an integer", actual=value))
            continue
        values[key] = int(value)

    minimums = {
        "global_concurrency": 1,
        "per_agent_concurrency": 1,
        "min_sources": 1,
        "scale_exp": 0,
        "socket_timeout_ms": 1,
        "handshake_timeout_ms": 1,
        "max_retries": 0,
        "retry_backoff_ms": 0,
    }
    maximums = {
        "scale_exp": 12,
        "socket_timeout_ms": 600000,
        "handshake_timeout_ms": 600000,
        "max_retries": 10,
        "retry_backoff_ms": 60000,
    }
    for key, minimum in minimums.items():
        if key in values and values[key] < minimum:
            violations.append(violation(f"$.criteria.{key}", f"must be >= {minimum}", actual=values[key]))
    for key, maximum in maximums.items():
        if key in values and values[key] > maximum:
            violations.append(violation(f"$.criteria.{key}", f"must be <= {maximum}", actual=values[key]))
    if violations:
        raise DriftError.agent(_stable_violations(violations))
    return Criteria(**values)


def _parse_aggregation(
    raw: dict[str, Any],
    base: str,
    violations: list[dict[str, Any]],
) -> AggregationSpec | None:
    agg_raw = raw.get("aggregation")
    if isinstance(agg_raw, str):
        agg: dict[str, Any] = {"version": agg_raw}
    elif isinstance(agg_raw, dict):
        agg = dict(agg_raw)
    else:
        agg = {}
        if "aggregation_version" in raw:
            agg["version"] = raw["aggregation_version"]
        if "aggregation_kind" in raw:
            agg["kind"] = raw["aggregation_kind"]
        if "window" in raw:
            agg["window"] = raw["window"]
    agg.setdefault("version", raw.get("aggregation_version"))
    agg.setdefault("kind", raw.get("aggregation_kind", "median"))
    agg.setdefault("window", raw.get("window", 5))

    kind = agg.get("kind")
    version = agg.get("version")
    window = agg.get("window")
    ok = True
    if kind not in AGGREGATION_KINDS:
        violations.append(
            violation(f"{base}.aggregation.kind", "unsupported aggregation kind", actual=kind, expected=str(AGGREGATION_KINDS))
        )
        ok = False
    if not (isinstance(version, str) or is_int(version)) or str(version) == "":
        violations.append(violation(f"{base}.aggregation.version", "must be a non-empty string or integer", actual=version))
        ok = False
    if not is_int(window) or not (1 <= int(window) <= 101):
        violations.append(violation(f"{base}.aggregation.window", "must be an integer in [1, 101]", actual=window))
        ok = False
    elif kind == "trimmed_mean" and int(window) < 3:
        violations.append(violation(f"{base}.aggregation.window", "trimmed_mean requires window >= 3", actual=window))
        ok = False
    if not ok:
        return None
    return AggregationSpec(kind=str(kind), window=int(window), version=str(version))


def _parse_profiles(
    raw: Any,
    agent_id: str,
    hosts: set[str],
    metrics: set[str],
    base: str,
    violations: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    profiles: dict[str, dict[str, dict[str, int]]] = {}
    if raw is None:
        return profiles
    if not isinstance(raw, dict):
        violations.append(violation(f"{base}.profiles", "profiles must be an object", actual=raw))
        return profiles
    for host_id, metric_map in raw.items():
        if host_id not in hosts:
            violations.append(violation(f"{base}.profiles.{host_id}", "profile host is outside agent coverage", actual=host_id))
            continue
        if not isinstance(metric_map, dict):
            violations.append(violation(f"{base}.profiles.{host_id}", "profile must be a metric map", actual=metric_map))
            continue
        profiles[host_id] = {}
        for metric, profile in metric_map.items():
            profile_path = f"{base}.profiles.{host_id}.{metric}"
            if metric not in METRICS or metric not in metrics:
                violations.append(violation(profile_path, "profile metric is not covered by the agent", actual=metric))
                continue
            if not isinstance(profile, dict):
                violations.append(violation(profile_path, "profile must be an object", actual=profile))
                continue
            profile_base = profile.get("base")
            spread = profile.get("spread")
            if not is_int(profile_base) or int(profile_base) <= 0:
                violations.append(violation(f"{profile_path}.base", "must be a positive integer", actual=profile_base))
                continue
            if not is_int(spread) or int(spread) < 0:
                violations.append(violation(f"{profile_path}.spread", "must be a non-negative integer", actual=spread))
                continue
            profiles[host_id][metric] = {"base": int(profile_base), "spread": int(spread)}
    return profiles


def parse_agent_manifest(
    data: Any,
    chain: SnapshotChain,
    criteria: Criteria | None = None,
) -> tuple[AgentSpec, ...]:
    criteria = criteria or default_criteria()
    violations: list[dict[str, Any]] = []
    raw_items = data if isinstance(data, list) else data.get("agents") if isinstance(data, dict) else None
    if not isinstance(raw_items, list) or not raw_items:
        raise DriftError.agent([violation("$.agents", "must be a non-empty array")])
    all_hosts = set(chain.host_ids())
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    agents: dict[str, AgentSpec] = {}

    for index, raw in enumerate(raw_items):
        base = f"$.agents[{index}]"
        if not isinstance(raw, dict):
            violations.append(violation(base, "agent must be an object", actual=raw))
            continue
        agent_id = raw.get("agent_id")
        good = True
        if not isinstance(agent_id, str) or not agent_id:
            violations.append(violation(f"{base}.agent_id", "must be a non-empty string", actual=agent_id))
            good = False
        elif agent_id in seen_ids:
            violations.append(violation(f"{base}.agent_id", "agent identifier must be unique", actual=agent_id))
            good = False
        else:
            seen_ids.add(agent_id)

        host = raw.get("host", raw.get("bind_host", "127.0.0.1"))
        if host != "127.0.0.1":
            violations.append(violation(f"{base}.host", "must be the literal loopback address 127.0.0.1", actual=host))
            good = False
        port = raw.get("port")
        if not is_int(port) or not (1024 <= int(port) <= 65535):
            violations.append(violation(f"{base}.port", "must be an integer high port in [1024, 65535]", actual=port))
            good = False
        elif int(port) in seen_ports:
            violations.append(violation(f"{base}.port", "port must be globally unique", actual=port))
            good = False
        else:
            seen_ports.add(int(port))
        seed = raw.get("seed")
        if not is_int(seed):
            violations.append(violation(f"{base}.seed", "must be an integer", actual=seed))
            good = False

        aggregation = _parse_aggregation(raw, base, violations)
        if aggregation is None:
            good = False

        hosts_raw = raw.get("hosts")
        if hosts_raw is None:
            hosts = tuple(sorted(all_hosts))
        elif isinstance(hosts_raw, list) and hosts_raw:
            host_list: list[str] = []
            for host_index, host_item in enumerate(hosts_raw):
                if not isinstance(host_item, str) or not host_item:
                    violations.append(
                        violation(f"{base}.hosts[{host_index}]", "must be a non-empty host identifier", actual=host_item)
                    )
                    good = False
                elif host_item not in all_hosts:
                    violations.append(
                        violation(f"{base}.hosts[{host_index}]", "host is absent from the snapshot chain", actual=host_item)
                    )
                    good = False
                elif host_item in host_list:
                    violations.append(
                        violation(f"{base}.hosts[{host_index}]", "duplicate host coverage", actual=host_item)
                    )
                    good = False
                else:
                    host_list.append(host_item)
            hosts = tuple(sorted(host_list))
        else:
            violations.append(violation(f"{base}.hosts", "must be a non-empty array when present", actual=hosts_raw))
            hosts = ()
            good = False

        metrics_raw = raw.get("metrics", list(METRICS))
        if not isinstance(metrics_raw, list) or not metrics_raw:
            violations.append(violation(f"{base}.metrics", "must be a non-empty array", actual=metrics_raw))
            metrics: tuple[str, ...] = ()
            good = False
        else:
            metric_list: list[str] = []
            for metric_index, metric in enumerate(metrics_raw):
                if metric not in METRICS:
                    violations.append(
                        violation(f"{base}.metrics[{metric_index}]", "unsupported metric", actual=metric)
                    )
                    good = False
                elif metric in metric_list:
                    violations.append(
                        violation(f"{base}.metrics[{metric_index}]", "duplicate metric coverage", actual=metric)
                    )
                    good = False
                else:
                    metric_list.append(metric)
            metrics = tuple(sorted(metric_list, key=METRICS.index))

        quota = raw.get("per_agent_concurrency", criteria.per_agent_concurrency)
        if not is_int(quota) or int(quota) < 1:
            violations.append(violation(f"{base}.per_agent_concurrency", "must be a positive integer", actual=quota))
            good = False

        profiles = _parse_profiles(raw.get("profiles"), str(agent_id), set(hosts), set(metrics), base, violations)
        if good and aggregation is not None and agents.get(str(agent_id)) is None:
            agents[str(agent_id)] = AgentSpec(
                agent_id=str(agent_id),
                host=str(host),
                port=int(port),
                seed=int(seed),
                aggregation=aggregation,
                hosts=hosts,
                metrics=metrics,
                per_agent_concurrency=int(quota),
                profiles=profiles,
            )

    if violations:
        raise DriftError.agent(_stable_violations(violations))
    return tuple(agents[agent_id] for agent_id in sorted(agents))


def load_snapshot_chain(path: str | Path) -> SnapshotChain:
    return parse_snapshot_chain(_read_json(path, "E_SNAPSHOT"))


def load_criteria(path: str | Path) -> Criteria:
    return parse_criteria(_read_json(path, "E_AGENT"))


def load_agent_manifest(path: str | Path, chain: SnapshotChain, criteria: Criteria | None = None) -> tuple[AgentSpec, ...]:
    data = _read_json(path, "E_AGENT")
    if criteria is None and isinstance(data, dict) and "criteria" in data:
        criteria = parse_criteria(data["criteria"])
    return parse_agent_manifest(data, chain, criteria)


def load_inputs(
    snapshot_path: str | Path,
    agent_path: str | Path,
    criteria_path: str | Path | None = None,
) -> tuple[SnapshotChain, tuple[AgentSpec, ...], Criteria]:
    chain = load_snapshot_chain(snapshot_path)
    agent_data = _read_json(agent_path, "E_AGENT")
    criteria = load_criteria(criteria_path) if criteria_path else parse_criteria(
        agent_data.get("criteria") if isinstance(agent_data, dict) else None
    )
    agents = parse_agent_manifest(agent_data, chain, criteria)
    return chain, agents, criteria
