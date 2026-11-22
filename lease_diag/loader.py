"""Strong validation loader for pool declarations, requests and rollbacks.

Any invalid input causes a single :class:`LeaseError` (``E_POOL`` for
declaration/stream problems, ``E_ROLLBACK`` for rollback graph problems) that
rejects the whole load with precise locations; nothing is returned partially.
"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import E_POOL, E_ROLLBACK, LeaseError
from .model import Pool, Request, RollbackEdge


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_ndjson(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LeaseError(
                    E_POOL, "malformed NDJSON line",
                    [f"{path}:{lineno}"], {"detail": str(exc)}) from exc
    return rows


def _load_pools(doc: dict, source: str) -> dict[str, Pool]:
    problems: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("pools"), list):
        raise LeaseError(E_POOL, "pool declaration must contain a 'pools' list",
                         [source])
    pools: dict[str, Pool] = {}
    for idx, row in enumerate(doc["pools"]):
        loc = f"{source}#pools[{idx}]"
        if not isinstance(row, dict):
            problems.append(f"{loc}: entry must be an object")
            continue
        pid = row.get("pool_id")
        if not isinstance(pid, str) or not pid:
            problems.append(f"{loc}: pool_id must be a non-empty string")
            continue
        if pid in pools:
            problems.append(f"{loc}: duplicate pool_id {pid!r}")
            continue
        capacity = row.get("capacity")
        lease_ms = row.get("lease_ms")
        renew_ms = row.get("renew_ms")
        priority = row.get("priority")
        quota = row.get("quota", capacity)
        if not _is_int(capacity) or capacity <= 0:
            problems.append(f"{loc}: capacity must be a positive integer")
        if not _is_int(lease_ms) or lease_ms <= 0:
            problems.append(f"{loc}: lease_ms must be a positive integer")
        if not _is_int(renew_ms) or renew_ms < 0:
            problems.append(f"{loc}: renew_ms must be a non-negative integer")
        elif _is_int(lease_ms) and lease_ms > 0 and renew_ms >= lease_ms:
            problems.append(
                f"{loc}: renew_ms ({renew_ms}) must be < lease_ms ({lease_ms})")
        if not _is_int(priority):
            problems.append(f"{loc}: priority must be an integer")
        if not _is_int(quota) or quota <= 0:
            problems.append(f"{loc}: quota must be a positive integer")
        if all(_is_int(v) for v in (capacity, lease_ms, renew_ms,
                                    priority, quota)):
            pools[pid] = Pool(pid, capacity, lease_ms, renew_ms, priority,
                              min(quota, capacity))
    if problems:
        raise LeaseError(E_POOL, "pool declaration rejected", problems)
    if not pools:
        raise LeaseError(E_POOL, "at least one pool is required", [source])
    return pools


def _load_requests(rows: list[dict], pools: dict[str, Pool],
                   source: str) -> list[Request]:
    problems: list[str] = []
    requests: list[Request] = []
    seen_ids: set[str] = set()
    last_arrival = -1
    key_last_rev: dict[str, int] = {}
    required = ("request_id", "key", "pool_id", "revision", "priority",
                "arrive_ms")
    for idx, row in enumerate(rows):
        loc = f"{source}#line[{idx + 1}]"
        if not isinstance(row, dict):
            problems.append(f"{loc}: request must be an object")
            continue
        missing = [name for name in required if name not in row]
        if missing:
            problems.append(f"{loc}: missing fields {sorted(missing)}")
            continue
        rid, key, pid = row["request_id"], row["key"], row["pool_id"]
        rev, prio, arrive = (row["revision"], row["priority"],
                             row["arrive_ms"])
        if not isinstance(rid, str) or not rid:
            problems.append(f"{loc}: request_id must be a non-empty string")
        elif rid in seen_ids:
            problems.append(f"{loc}: duplicate request_id {rid!r}")
        else:
            seen_ids.add(rid)
        if not isinstance(key, str) or not key:
            problems.append(f"{loc}: key must be a non-empty string")
        if pid not in pools:
            problems.append(f"{loc}: unknown target pool_id {pid!r}")
        if not _is_int(rev) or rev < 0:
            problems.append(f"{loc}: revision must be a non-negative integer")
        if not _is_int(prio):
            problems.append(f"{loc}: priority must be an integer")
        if not _is_int(arrive) or arrive < 0:
            problems.append(f"{loc}: arrive_ms must be a non-negative integer")
        elif arrive < last_arrival:
            problems.append(
                f"{loc}: arrive_ms {arrive} decreases (previous {last_arrival})")
        else:
            last_arrival = arrive
        if isinstance(key, str) and key and _is_int(rev) and rev >= 0:
            prev = key_last_rev.get(key)
            if prev is not None and rev < prev:
                problems.append(
                    f"{loc}: revision {rev} for key {key!r} decreases "
                    f"(previous {prev})")
            else:
                key_last_rev[key] = max(rev if prev is None else prev, rev)
        requests.append(Request(rid, key, pid, rev, prio, arrive))
    if problems:
        raise LeaseError(E_POOL, "request stream rejected", problems)
    if not requests:
        raise LeaseError(E_POOL, "request stream is empty", [source])
    return requests


def _load_rollback(doc: dict, requests: list[Request],
                   source: str) -> list[RollbackEdge]:
    problems: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("rollback"), list):
        raise LeaseError(E_ROLLBACK, "declaration must contain a 'rollback' list",
                         [source])
    rows = doc["rollback"]
    revisions_by_key: dict[str, set[int]] = {}
    for req in requests:
        revisions_by_key.setdefault(req.key, set()).add(req.revision)
    edges: list[RollbackEdge] = []
    seen_edges: set[tuple[str, int]] = set()
    for idx, row in enumerate(rows):
        loc = f"{source}#rollback[{idx}]"
        if not isinstance(row, dict):
            problems.append(f"{loc}: entry must be an object")
            continue
        key, from_rev, target_rev = (row.get("key"), row.get("from_rev"),
                                     row.get("target_rev"))
        if not isinstance(key, str) or not key:
            problems.append(f"{loc}: key must be a non-empty string")
            continue
        known = revisions_by_key.get(key, set())
        if not key in revisions_by_key:
            problems.append(f"{loc}: key {key!r} never appears in requests")
        if not _is_int(from_rev) or from_rev < 0:
            problems.append(f"{loc}: from_rev must be a non-negative integer")
        elif from_rev not in known:
            problems.append(
                f"{loc}: from_rev {from_rev} for key {key!r} has no request")
        if not _is_int(target_rev) or target_rev < 0:
            problems.append(f"{loc}: target_rev must be a non-negative integer")
        elif target_rev not in known:
            problems.append(
                f"{loc}: target_rev {target_rev} for key {key!r} "
                "has no request")
        edge_key = (key, from_rev)
        if edge_key in seen_edges:
            problems.append(f"{loc}: duplicate rollback edge {edge_key}")
        else:
            seen_edges.add(edge_key)
        if _is_int(from_rev) and _is_int(target_rev) and from_rev == target_rev:
            problems.append(f"{loc}: rollback edge must change revision")
        edges.append(RollbackEdge(key, from_rev, target_rev))
    # Acyclic check per key (only meaningful when ints parsed).
    graph: dict[str, dict[int, int]] = {}
    for edge in edges:
        if _is_int(edge.from_rev) and _is_int(edge.target_rev):
            graph.setdefault(edge.key, {})[edge.from_rev] = edge.target_rev
    for key, adj in graph.items():
        for start in adj:
            seen: set[int] = set()
            node = start
            while node in adj:
                if node in seen:
                    cycle = _describe_cycle(adj, start)
                    problems.append(
                        f"{source}#rollback: cyclic rollback chain for key "
                        f"{key!r}: {' -> '.join(map(str, cycle))}")
                    break
                seen.add(node)
                node = adj[node]
    if problems:
        raise LeaseError(E_ROLLBACK, "rollback declaration rejected", problems)
    return edges


def _describe_cycle(adj: dict[int, int], start: int) -> list[int]:
    path = [start]
    node = adj[start]
    while node != start:
        path.append(node)
        node = adj[node]
    path.append(start)
    return path


def load_workspace(pools_path: str | Path,
                   requests_path: str | Path) -> dict:
    """Load and strongly validate the whole workspace atomically."""
    pools_path, requests_path = Path(pools_path), Path(requests_path)
    doc = _read_json(pools_path)
    pools = _load_pools(doc, str(pools_path))
    rows = _read_ndjson(requests_path)
    requests = _load_requests(rows, pools, str(requests_path))
    edges = _load_rollback(doc, requests, str(pools_path))
    targets = {edge.key: {"from_rev": edge.from_rev,
                          "target_rev": edge.target_rev} for edge in edges}
    return {"pools": pools, "requests": requests,
            "rollback_edges": edges, "rollback_targets": targets,
            "pool_set_version": doc.get("pool_set_version", "pool-set/v1"),
            "rollback_version": doc.get("rollback_version", "rollback/v1")}
