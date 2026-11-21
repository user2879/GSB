"""Strict, all-or-nothing policy and cascade validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest
from .errors import PolicyError, Violation
from .models import CascadeEdge, PolicySet, ShaperPolicy


DEFAULT_MAX_RATE_KBPS = 10_000_000


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(
            [Violation("<file>", "json", f"cannot read policy: {exc}")]
        ) from exc
    if not isinstance(value, dict):
        raise PolicyError([Violation("<file>", "json", "top level must be an object")])
    return value


def _field_name(raw: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name in raw:
            return name
    return names[0]


def _find_cycles(edges: tuple[CascadeEdge, ...], node_ids: set[str]) -> list[tuple[str, ...]]:
    children: dict[str, list[str]] = {node: [] for node in node_ids}
    for edge in edges:
        if edge.upstream in node_ids and edge.downstream in node_ids:
            children[edge.upstream].append(edge.downstream)
    for values in children.values():
        values.sort()

    state: dict[str, int] = {node: 0 for node in node_ids}
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for child in children[node]:
            if state[child] == 0:
                visit(child)
            elif state[child] == 1:
                index = stack.index(child)
                cycle = stack[index:] + [child]
                rotations = [tuple(cycle[i:-1] + cycle[:i] + [cycle[i]]) for i in range(len(cycle) - 1)]
                cycles.add(min(rotations))
        stack.pop()
        state[node] = 2

    for node in sorted(node_ids):
        if state[node] == 0:
            visit(node)
    return sorted(cycles)


def load_policy(
    source: str | Path | Mapping[str, Any],
    *,
    max_rate_kbps: int | None = None,
) -> PolicySet:
    """Load and validate policy declarations without partial acceptance.

    The source is either a JSON file path or an already-decoded mapping. Every
    detected problem is returned in a single E_POLICY error, sorted by shaper ID
    and field so callers receive stable diagnostics.
    """

    raw = _read_json(source)
    violations: list[Violation] = []

    policy_version = raw.get("policy_version")
    if not isinstance(policy_version, str) or not policy_version.strip():
        violations.append(Violation("<policy>", "policy_version", "must be a non-empty string"))
        policy_version = str(policy_version)

    declared_limit = raw.get(
        "max_committed_rate_kbps",
        raw.get("max_rate_kbps", raw.get("rate_limit_kbps", DEFAULT_MAX_RATE_KBPS)),
    )
    if max_rate_kbps is not None:
        declared_limit = max_rate_kbps
    if not _is_int(declared_limit) or declared_limit <= 0:
        violations.append(
            Violation("<policy>", "max_committed_rate_kbps", "must be a positive integer")
        )
        declared_limit = DEFAULT_MAX_RATE_KBPS

    raw_shapers = raw.get("shapers")
    if not isinstance(raw_shapers, list):
        violations.append(Violation("<policy>", "shapers", "must be a list"))
        raw_shapers = []
    elif not raw_shapers:
        violations.append(Violation("<policy>", "shapers", "must contain at least one shaper"))

    shapers: list[ShaperPolicy] = []
    seen_ids: dict[str, int] = {}
    duplicate_ids: set[str] = set()

    for index, item in enumerate(raw_shapers):
        location = f"shapers[{index}]"
        if not isinstance(item, Mapping):
            violations.append(Violation(location, "shaper", "must be an object"))
            continue
        id_field = _field_name(item, "shaper_id", "id")
        shaper_id = item.get(id_field)
        if not isinstance(shaper_id, str) or not shaper_id.strip():
            violations.append(Violation(location, id_field, "must be a non-empty string"))
            shaper_id = location
        elif shaper_id != shaper_id.strip():
            violations.append(
                Violation(shaper_id, id_field, "must not have leading or trailing whitespace")
            )
            shaper_id = shaper_id.strip()
        else:
            shaper_id = shaper_id.strip()
            seen_ids[shaper_id] = seen_ids.get(shaper_id, 0) + 1
            if seen_ids[shaper_id] > 1:
                duplicate_ids.add(shaper_id)

        rate_field = _field_name(item, "committed_rate_kbps", "rate_kbps")
        rate = item.get(rate_field)
        if not _is_int(rate) or rate <= 0:
            violations.append(Violation(shaper_id, rate_field, "must be a positive integer"))
        elif rate > declared_limit:
            violations.append(
                Violation(
                    shaper_id,
                    rate_field,
                    f"must not exceed declared max_committed_rate_kbps={declared_limit}",
                )
            )

        burst_field = _field_name(item, "burst_bytes")
        burst = item.get(burst_field)
        if not _is_int(burst) or burst < 0:
            violations.append(Violation(shaper_id, burst_field, "must be a non-negative integer"))

        queue_field = _field_name(item, "queue_limit_packets", "queue_limit")
        queue_limit = item.get(queue_field)
        if not _is_int(queue_limit) or queue_limit < 0:
            violations.append(
                Violation(shaper_id, queue_field, "must be a non-negative integer")
            )

        prefix_field = _field_name(
            item,
            "match_dst_port_prefix",
            "dst_port_prefix",
            "match_port_prefix",
        )
        prefix = item.get(prefix_field)
        if not _is_int(prefix) or not 1 <= prefix <= 65535:
            violations.append(
                Violation(shaper_id, prefix_field, "must be an integer in [1, 65535]")
            )

        if (
            _is_int(rate)
            and rate > 0
            and _is_int(burst)
            and burst >= 0
            and _is_int(queue_limit)
            and queue_limit >= 0
            and _is_int(prefix)
            and 1 <= prefix <= 65535
        ):
            shapers.append(
                ShaperPolicy(
                    shaper_id=shaper_id,
                    committed_rate_kbps=rate,
                    burst_bytes=burst,
                    queue_limit_packets=queue_limit,
                    match_dst_port_prefix=prefix,
                )
            )

    for shaper_id in sorted(duplicate_ids):
        violations.append(Violation(shaper_id, "shaper_id", "must be unique"))

    raw_edges = raw.get("edges", raw.get("cascade", []))
    if not isinstance(raw_edges, list):
        violations.append(Violation("<policy>", "edges", "must be a list"))
        raw_edges = []

    edges: list[CascadeEdge] = []
    edge_pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_edges):
        location = f"edges[{index}]"
        if not isinstance(item, Mapping):
            violations.append(Violation(location, "edge", "must be an object"))
            continue
        upstream = item.get("upstream", item.get("from"))
        downstream = item.get("downstream", item.get("to"))
        valid = True
        if not isinstance(upstream, str) or not upstream:
            violations.append(Violation(location, "upstream", "must be a non-empty string"))
            valid = False
        if not isinstance(downstream, str) or not downstream:
            violations.append(Violation(location, "downstream", "must be a non-empty string"))
            valid = False
        if not valid:
            continue
        if upstream == downstream:
            violations.append(
                Violation(upstream, "cascade", "self-loop is not allowed")
            )
            continue
        if upstream not in seen_ids:
            violations.append(
                Violation(upstream, "cascade", "upstream shaper does not exist")
            )
        if downstream not in seen_ids:
            violations.append(
                Violation(downstream, "cascade", "downstream shaper does not exist")
            )
        pair = (upstream, downstream)
        if pair in edge_pairs:
            violations.append(Violation(downstream, "cascade", "duplicate edge"))
            continue
        edge_pairs.add(pair)
        edges.append(CascadeEdge(upstream=upstream, downstream=downstream))

    upstream_count: dict[str, list[str]] = {}
    for edge in sorted(edges, key=lambda item: (item.downstream, item.upstream)):
        upstream_count.setdefault(edge.downstream, []).append(edge.upstream)
    for downstream, upstreams in sorted(upstream_count.items()):
        if len(upstreams) > 1:
            violations.append(
                Violation(
                    downstream,
                    "cascade",
                    "at most one upstream allowed; found " + ','.join(sorted(upstreams)),
                )
            )

    for cycle in _find_cycles(tuple(edges), set(seen_ids)):
        violations.append(
            Violation(
                ','.join(cycle),
                "cascade",
                "cycle detected: " + '->'.join(cycle),
            )
        )

    if violations:
        raise PolicyError(violations)

    return PolicySet(
        policy_version=policy_version,
        max_committed_rate_kbps=declared_limit,
        shapers=tuple(sorted(shapers, key=lambda item: item.shaper_id)),
        edges=tuple(sorted(edges, key=lambda item: (item.upstream, item.downstream))),
        source=raw,
    )


def policy_as_dict(policy: PolicySet) -> dict[str, Any]:
    return {
        "policy_version": policy.policy_version,
        "max_committed_rate_kbps": policy.max_committed_rate_kbps,
        "shapers": [shaper.as_dict() for shaper in policy.shapers],
        "edges": [edge.as_dict() for edge in policy.edges],
    }


def shaper_digest(policy: PolicySet) -> str:
    return digest([shaper.as_dict() for shaper in policy.shapers])


def cascade_digest(policy: PolicySet) -> str:
    return digest([edge.as_dict() for edge in policy.edges])


def policy_fingerprint(policy: PolicySet) -> str:
    return digest(policy_as_dict(policy))
