"""State normalization, dependency validation and deterministic propagation order."""

from __future__ import annotations

import heapq
from copy import deepcopy
from typing import Any, Iterable

from .errors import dependency_error, protocol_error
from .util import is_int, normalize_ref


SCHEMA_VERSION = 1
STATUSES = {"active", "expired", "invalidated", "unknown"}


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise protocol_error(f"{where} must be an object", where=where)
    return value


def _normalize_token(raw: Any, where: str = "token") -> dict[str, Any]:
    obj = _require_object(raw, where)
    token_id = obj.get("id")
    if not isinstance(token_id, str) or not token_id:
        raise protocol_error("token id must be a non-empty string", where=where)
    healthy = obj.get("healthy", True)
    version = obj.get("version", 0)
    if not isinstance(healthy, bool):
        raise protocol_error("token healthy must be boolean", token=token_id)
    if not is_int(version) or version < 0:
        raise protocol_error("token version must be a non-negative integer", token=token_id)
    return {"id": token_id, "healthy": healthy, "version": version}


def normalize_tokens(raw_tokens: Any) -> dict[str, dict[str, Any]]:
    if raw_tokens is None:
        return {}
    if isinstance(raw_tokens, list):
        tokens: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(raw_tokens):
            token = _normalize_token(raw, f"tokens[{index}]")
            if token["id"] in tokens:
                raise dependency_error(
                    "duplicate token identifier", token=token["id"], chain=[token["id"]]
                )
            tokens[token["id"]] = token
        return tokens
    if isinstance(raw_tokens, dict):
        tokens = {}
        for token_id, raw in raw_tokens.items():
            obj = dict(raw) if isinstance(raw, dict) else {}
            obj.setdefault("id", token_id)
            token = _normalize_token(obj, f"tokens.{token_id}")
            if token["id"] != token_id:
                raise protocol_error("token map key/id mismatch", token=token_id)
            tokens[token_id] = token
        return tokens
    raise protocol_error("tokens must be a list or object")


def normalize_member(
    raw: Any,
    *,
    where: str,
    member_ids: set[str] | None = None,
    token_ids: set[str] | None = None,
    strict_refs: bool = False,
) -> dict[str, Any]:
    obj = _require_object(raw, where)
    member_id = obj.get("id")
    if not isinstance(member_id, str) or not member_id:
        raise protocol_error("member id must be a non-empty string", where=where)
    term = obj.get("term", 1)
    lease_ms = obj.get("lease_ms", 2500)
    lease_expiry_ms = obj.get("lease_expiry_ms", 0)
    lease_version = obj.get("lease_version", 0)
    lease_valid = obj.get("lease_valid", False)
    status = obj.get("status", "active")
    dependencies = obj.get("dependencies", [])
    invalidation_version = obj.get("invalidation_version", 0)
    candidate_eligible = obj.get("candidate_eligible", False)

    if not is_int(term) or term < 0:
        raise protocol_error("member term must be a non-negative integer", member=member_id)
    if not is_int(lease_ms) or lease_ms <= 0:
        raise protocol_error("member lease_ms must be a positive integer", member=member_id)
    if not is_int(lease_expiry_ms) or lease_expiry_ms < 0:
        raise protocol_error("member lease_expiry_ms must be a non-negative integer", member=member_id)
    if not is_int(lease_version) or lease_version < 0:
        raise protocol_error("member lease_version must be a non-negative integer", member=member_id)
    if not is_int(invalidation_version) or invalidation_version < 0:
        raise protocol_error(
            "member invalidation_version must be a non-negative integer", member=member_id
        )
    if not isinstance(lease_valid, bool) or not isinstance(candidate_eligible, bool):
        raise protocol_error("lease_valid and candidate_eligible must be boolean", member=member_id)
    if status not in STATUSES:
        raise protocol_error(f"unknown member status {status!r}", member=member_id)
    if not isinstance(dependencies, list):
        raise protocol_error("member dependencies must be a list", member=member_id)
    if any(not isinstance(item, str) for item in dependencies):
        raise protocol_error("dependency references must be strings", member=member_id)

    normalized_dependencies = list(dependencies)
    if strict_refs:
        ids = member_ids if member_ids is not None else set()
        tokens = token_ids if token_ids is not None else set()
        seen: set[str] = set()
        normalized_dependencies = []
        for dependency in dependencies:
            try:
                ref = normalize_ref(dependency, ids, tokens)
            except ValueError as exc:
                missing = dependency
                if dependency.startswith(("member:", "token:")):
                    missing = dependency.partition(":")[2]
                chain = [f"member:{member_id}", dependency]
                if dependency.startswith("member:") and missing == member_id:
                    chain = [f"member:{member_id}", f"member:{member_id}"]
                raise dependency_error(
                    f"missing dependency {missing!r} referenced by {member_id}",
                    member=member_id,
                    dependency=dependency,
                    chain=chain,
                ) from exc
            if ref in seen:
                raise dependency_error(
                    "duplicate dependency reference",
                    member=member_id,
                    dependency=ref,
                    chain=[f"member:{member_id}", ref],
                )
            seen.add(ref)
            normalized_dependencies.append(ref)

    return {
        "id": member_id,
        "term": term,
        "lease_ms": lease_ms,
        "lease_expiry_ms": lease_expiry_ms,
        "lease_version": lease_version,
        "lease_valid": lease_valid,
        "status": status,
        "dependencies": normalized_dependencies,
        "candidate_eligible": candidate_eligible,
        "invalidation_version": invalidation_version,
    }


def normalize_members(raw_members: Any, token_ids: set[str]) -> dict[str, dict[str, Any]]:
    if isinstance(raw_members, list):
        items: list[tuple[Any, str]] = [
            (raw, f"members[{index}]") for index, raw in enumerate(raw_members)
        ]
    elif isinstance(raw_members, dict):
        items = []
        for member_id, raw in raw_members.items():
            if not isinstance(raw, dict):
                raise protocol_error("member entry must be an object", member=member_id)
            obj = dict(raw)
            if "id" in obj and obj["id"] != member_id:
                raise dependency_error(
                    "duplicate or conflicting member identifier",
                    member=member_id,
                    chain=[f"member:{member_id}"],
                )
            obj.setdefault("id", member_id)
            items.append((obj, f"members.{member_id}"))
    else:
        raise protocol_error("members must be a list or object")

    members: dict[str, dict[str, Any]] = {}
    raw_by_id: dict[str, Any] = {}
    for raw, where in items:
        member_id = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(member_id, str) and member_id in raw_by_id:
            raise dependency_error(
                "duplicate member identifier",
                member=member_id,
                chain=[f"member:{member_id}"],
            )
        if isinstance(member_id, str):
            raw_by_id[member_id] = raw
    member_ids = set(raw_by_id)
    for raw, where in items:
        member = normalize_member(
            raw,
            where=where,
            member_ids=member_ids,
            token_ids=token_ids,
            strict_refs=True,
        )
        members[member["id"]] = member
    return members


def graph_from_members(members: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {member_id: list(members[member_id]["dependencies"]) for member_id in sorted(members)}


def _graph_nodes(members: dict[str, dict[str, Any]], tokens: dict[str, dict[str, Any]]) -> set[str]:
    return {f"member:{member_id}" for member_id in members} | {
        f"token:{token_id}" for token_id in tokens
    }


def _dependencies_map(
    members: dict[str, dict[str, Any]], tokens: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    graph = {f"token:{token_id}": [] for token_id in sorted(tokens)}
    for member_id in sorted(members):
        graph[f"member:{member_id}"] = list(members[member_id]["dependencies"])
    return graph


def detect_dependency_cycle(
    members: dict[str, dict[str, Any]], tokens: dict[str, dict[str, Any]]
) -> list[str] | None:
    graph = _dependencies_map(members, tokens)
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            if node in stack:
                index = stack.index(node)
                return stack[index:] + [node]
            return [node, node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for dependency in graph.get(node, []):
            cycle = visit(dependency)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(graph):
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def validate_graph(
    members: dict[str, dict[str, Any]], tokens: dict[str, dict[str, Any]]
) -> None:
    """Validate references and cycles, rejecting the whole state with E_DEPEND."""

    member_ids = set(members)
    token_ids = set(tokens)
    for member_id in sorted(members):
        dependencies = members[member_id].get("dependencies", [])
        seen: set[str] = set()
        for dependency in dependencies:
            if not isinstance(dependency, str):
                raise dependency_error(
                    "dependency reference must be a string",
                    member=member_id,
                    chain=[f"member:{member_id}"],
                )
            try:
                ref = normalize_ref(dependency, member_ids, token_ids)
            except ValueError:
                missing = dependency.partition(":")[2]
                raise dependency_error(
                    f"missing dependency {missing!r} referenced by {member_id}",
                    member=member_id,
                    dependency=dependency,
                    chain=[f"member:{member_id}", dependency],
                ) from None
            if ref in seen:
                raise dependency_error(
                    "duplicate dependency reference",
                    member=member_id,
                    dependency=ref,
                    chain=[f"member:{member_id}", ref],
                )
            seen.add(ref)
            members[member_id]["dependencies"] = list(dependencies)
    cycle = detect_dependency_cycle(members, tokens)
    if cycle:
        edge = cycle[0]
        member = edge.partition(":")[2] if edge.startswith("member:") else None
        raise dependency_error(
            "dependency graph contains a cycle",
            member=member,
            chain=cycle,
        )


def normalize_state(raw: Any, *, strict: bool = True) -> dict[str, Any]:
    """Normalize a persisted state/definition document. Invalid graphs raise E_DEPEND."""

    state = _require_object(raw, "state")
    tokens = normalize_tokens(state.get("tokens", {}))
    members = normalize_members(state.get("members", []), set(tokens))
    if strict:
        validate_graph(members, tokens)

    term = state.get("term", 1)
    if not is_int(term) or term < 1:
        raise protocol_error("state term must be a positive integer")
    state_version = state.get("state_version", 0)
    if not is_int(state_version) or state_version < 0:
        raise protocol_error("state_version must be a non-negative integer")
    votes = state.get("votes", {})
    if not isinstance(votes, dict):
        raise protocol_error("votes must be an object")
    traces = state.get("propagation_traces", [])
    if not isinstance(traces, list):
        raise protocol_error("propagation_traces must be a list")
    next_trace_id = state.get("next_trace_id", len(traces) + 1)
    if not is_int(next_trace_id) or next_trace_id < 1:
        raise protocol_error("next_trace_id must be a positive integer")
    for member in members.values():
        member["term"] = term if state.get("force_member_term", False) else member["term"]

    election = state.get("election")
    if not isinstance(election, dict):
        election = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "state_version": state_version,
        "term": term,
        "members": members,
        "tokens": tokens,
        "votes": deepcopy(votes),
        "election": deepcopy(election),
        "propagation_traces": deepcopy(traces),
        "next_trace_id": next_trace_id,
        "election_requires_rerun": bool(state.get("election_requires_rerun", False)),
    }


def state_from_definition(definition: Any) -> dict[str, Any]:
    raw = deepcopy(definition)
    if not isinstance(raw, dict):
        raise protocol_error("member definition must be an object")
    term = raw.get("term", 1)
    raw.setdefault("state_version", 0)
    raw.setdefault("election", {})
    raw.setdefault("votes", {})
    raw.setdefault("propagation_traces", [])
    raw.setdefault("next_trace_id", 1)
    raw.setdefault("election_requires_rerun", False)
    state = normalize_state(raw)
    state["term"] = term
    for member in state["members"].values():
        member["term"] = term
    state["election"] = {
        "status": "unknown",
        "leader": None,
        "term": term,
        "eligible": [],
        "vote_counts": {},
        "view_version": 0,
        "requires_rerun": True,
    }
    state["election_requires_rerun"] = True
    return state


def stable_downstream(
    members: dict[str, dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    trigger_ref: str,
    *,
    include_trigger: bool = False,
) -> list[str]:
    """Return member IDs in stable topological order downstream of a reference."""

    ref = trigger_ref
    if not ref.startswith(("member:", "token:")):
        ref = f"member:{ref}"
    graph = _dependencies_map(members, tokens)
    reverse: dict[str, set[str]] = {node: set() for node in graph}
    for node, dependencies in graph.items():
        for dependency in dependencies:
            reverse.setdefault(dependency, set()).add(node)

    affected: set[str] = set()
    queue = [ref]
    while queue:
        node = queue.pop(0)
        for dependent in sorted(reverse.get(node, ())):
            if dependent not in affected:
                affected.add(dependent)
                queue.append(dependent)
    if include_trigger:
        affected.add(ref)
    if not affected:
        return []

    indegree = {node: 0 for node in affected}
    for node in affected:
        for dependency in graph.get(node, []):
            if dependency in affected:
                indegree[node] += 1
    ready = [node for node in affected if indegree[node] == 0]
    heapq.heapify(ready)
    ordered_nodes: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        ordered_nodes.append(node)
        for dependent in sorted(reverse.get(node, ())):
            if dependent in affected:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(ready, dependent)
    if len(ordered_nodes) != len(affected):
        raise dependency_error("dependency graph contains a cycle", chain=sorted(affected))
    return [node.partition(":")[2] for node in ordered_nodes if node.startswith("member:")]


def state_members_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return state["members"]


def iter_member_ids(members: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(member["id"] for member in members)
