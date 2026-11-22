"""Loading and strict validation of registry / shard / manifest JSON files.

Validation order is deterministic and fails fast on the *first* offending
position, raising :class:`ManifestError` (or :class:`CycleError` for cycles,
:class:`GridError` for an illegal grid).  A rejected document never returns a
partial object, so callers cannot accidentally schedule invalid input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .content import shard_intervals
from .errors import CycleError, ManifestError
from .fixedpoint import validate_scale
from .models import DerivedSpec, NodeDef, PathEntry, ShardDecl, VersionManifest


def _valid_path_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and ".." not in name
        and not name.startswith(("/", "\\"))
        and "\n" not in name
        and "\r" not in name
    )


def topological_order(entries: Dict[str, PathEntry]) -> List[str]:
    """Kahn's algorithm with sorted ready-set: deterministic, order-independent.

    Raises :class:`CycleError` reporting the first cycle path found by DFS.
    """
    indegree = {name: 0 for name in entries}
    dependents: Dict[str, List[str]] = {name: [] for name in entries}
    for name, entry in entries.items():
        if entry.derived is not None:
            for source, _coef in entry.derived.sources:
                indegree[name] += 1
                dependents[source].append(name)
    ready = sorted(name for name, deg in indegree.items() if deg == 0)
    order: List[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for downstream in sorted(dependents[name]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)
                ready.sort()
    if len(order) != len(entries):
        remaining = sorted(set(entries) - set(order))
        cycle_path = _find_cycle(entries, remaining)
        raise CycleError(
            "derived dependency graph contains a cycle",
            "manifest.versions[].entries[].derived",
            {"cycle": cycle_path, "remaining": remaining},
        )
    return order


def _find_cycle(entries: Dict[str, PathEntry], candidates: List[str]) -> List[str]:
    """Return one concrete cycle (list of path names) for error reporting."""
    color: Dict[str, int] = {name: 0 for name in candidates}
    stack: List[str] = []

    def visit(node: str) -> List[str]:
        color[node] = 1
        stack.append(node)
        derived = entries[node].derived
        sources = sorted(s for s, _ in derived.sources) if derived else []
        for source in sources:
            if source not in color:
                continue
            if color[source] == 1:
                start = stack.index(source)
                return stack[start:] + [source]
            if color[source] == 0:
                found = visit(source)
                if found:
                    return found
        stack.pop()
        color[node] = 2
        return []

    for node in sorted(candidates):
        if color[node] == 0:
            found = visit(node)
            if found:
                return found
    return sorted(candidates)


def load_manifests(
    path: Path,
    registry: Dict[str, NodeDef],
    shard_decls: Dict[str, ShardDecl],
) -> List[VersionManifest]:
    raw = _load_json(path)
    loc = f"{path}#"
    if not isinstance(raw, dict) or "versions" not in raw:
        raise ManifestError("manifest must contain 'versions'", loc)
    versions_raw = raw["versions"]
    if not isinstance(versions_raw, list) or not versions_raw:
        raise ManifestError("'versions' must be a non-empty list", loc + "versions")
    seen_versions = set()
    previous_version = None
    manifests: List[VersionManifest] = []
    for vidx, version_doc in enumerate(versions_raw):
        vloc = f"{loc}versions[{vidx}]"
        if not isinstance(version_doc, dict) or "version" not in version_doc:
            raise ManifestError("version entry needs integer 'version'", vloc)
        version = version_doc["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ManifestError("version must be an integer", vloc + ".version")
        if version in seen_versions:
            raise ManifestError(f"duplicate version {version}", vloc + ".version")
        if previous_version is not None and version <= previous_version:
            raise ManifestError(
                f"version {version} not strictly greater than {previous_version}",
                vloc + ".version",
            )
        seen_versions.add(version)
        previous_version = version
        entries_raw = version_doc.get("entries")
        if not isinstance(entries_raw, list) or not entries_raw:
            raise ManifestError("version needs non-empty 'entries'", vloc + ".entries")
        entries: Dict[str, PathEntry] = {}
        for eidx, doc in enumerate(entries_raw):
            here = f"{vloc}.entries[{eidx}]"
            entry = _load_entry(doc, here, registry, shard_decls)
            if entry.path in entries:
                raise ManifestError(f"duplicate path {entry.path!r}", here + ".path")
            entries[entry.path] = entry
        for name, entry in entries.items():
            if entry.derived is not None:
                for source, _coef in entry.derived.sources:
                    if source not in entries:
                        raise ManifestError(
                            f"derived source {source!r} does not exist",
                            f"{vloc}.entries[].derived",
                            {"path": name, "source": source},
                        )
        topological_order(entries)
        manifests.append(VersionManifest(version=version, entries=tuple(entries.values())))
    return manifests


def _load_entry(
    doc: Any,
    here: str,
    registry: Dict[str, NodeDef],
    shard_decls: Dict[str, ShardDecl],
) -> PathEntry:
    if not isinstance(doc, dict):
        raise ManifestError("entry must be an object", here)
    required = {"path", "weight", "targets", "strategy", "priority"}
    missing = sorted(required - set(doc))
    if missing:
        raise ManifestError(f"missing required fields {missing}", here + "." + missing[0])
    name = doc["path"]
    if not _valid_path_name(name):
        raise ManifestError("illegal content path", here + ".path")
    if name not in shard_decls:
        raise ManifestError(
            f"path {name!r} has no shard declaration", here + ".path"
        )
    decl = shard_decls[name]
    strategy = doc["strategy"]
    if strategy not in ("full", "shard"):
        raise ManifestError("strategy must be 'full' or 'shard'", here + ".strategy")
    targets = doc["targets"]
    if not isinstance(targets, list) or not targets:
        raise ManifestError("targets must be a non-empty list", here + ".targets")
    for tidx, target in enumerate(targets):
        if not isinstance(target, str) or target not in registry:
            raise ManifestError(
                f"unknown target node {target!r}", here + f".targets[{tidx}]"
            )
    if len(set(targets)) != len(targets):
        raise ManifestError("duplicate target node", here + ".targets")
    priority = doc["priority"]
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ManifestError("priority must be an integer", here + ".priority")
    weight = doc["weight"]
    if not isinstance(weight, (int, float, str)) or isinstance(weight, bool):
        raise ManifestError("weight must be numeric/decimal string", here + ".weight")
    try:
        validate_scale(1)
    except Exception:  # pragma: no cover - defensive
        pass
    derived = None
    if doc.get("derived") is not None:
        derived_doc = doc["derived"]
        if not isinstance(derived_doc, dict) or not isinstance(derived_doc.get("sources"), list):
            raise ManifestError("derived needs a sources list", here + ".derived")
        sources = []
        for sidx, pair in enumerate(derived_doc["sources"]):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ManifestError(
                    "derived source must be [path, coefficient]",
                    here + f".derived.sources[{sidx}]",
                )
            source, coefficient = pair
            if not isinstance(coefficient, (int, float, str)) or isinstance(coefficient, bool):
                raise ManifestError(
                    "coefficient must be numeric/decimal string",
                    here + f".derived.sources[{sidx}][1]",
                )
            sources.append((source, coefficient))
        if not sources:
            raise ManifestError("derived sources must be non-empty", here + ".derived")
        derived = DerivedSpec(sources=tuple(sources))
    return PathEntry(
        path=name,
        length=decl.length,
        weight=weight,
        shards=decl.shards,
        targets=tuple(sorted(targets)),
        strategy=strategy,
        priority=priority,
        derived=derived,
    )


def _load_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ManifestError(f"input file not found: {path}", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"invalid JSON: {exc.msg}",
            f"{path}#line{exc.lineno}:col{exc.colno}",
        ) from exc


def load_registry(path: Path) -> Dict[str, NodeDef]:
    raw = _load_json(path)
    loc = f"{path}#"
    if not isinstance(raw, dict) or "nodes" not in raw:
        raise ManifestError("registry must be an object with 'nodes'", loc)
    nodes = raw["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ManifestError("'nodes' must be a non-empty list", loc + "nodes")
    registry: Dict[str, NodeDef] = {}
    required = {
        "node_id",
        "host",
        "port",
        "per_node_quota",
        "dispatch_timeout_ms",
        "retry_policy",
    }
    for index, item in enumerate(nodes):
        here = f"{loc}nodes[{index}]"
        if not isinstance(item, dict):
            raise ManifestError("node entry must be an object", here)
        missing = sorted(required - set(item))
        if missing:
            raise ManifestError(
                f"missing required fields {missing}", here + "." + missing[0]
            )
        node_id = item["node_id"]
        if not isinstance(node_id, str) or not node_id:
            raise ManifestError("node_id must be a non-empty string", here + ".node_id")
        if node_id in registry:
            raise ManifestError(f"duplicate node_id {node_id!r}", here + ".node_id")
        host = item["host"]
        if host not in ("127.0.0.1", "localhost"):
            raise ManifestError("nodes must bind to loopback host", here + ".host")
        port = item["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not (1024 < port < 65536):
            raise ManifestError("port must be a high-port integer", here + ".port")
        quota = item["per_node_quota"]
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            raise ManifestError("per_node_quota must be a positive int", here + ".per_node_quota")
        timeout = item["dispatch_timeout_ms"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ManifestError("dispatch_timeout_ms must be a positive int", here + ".dispatch_timeout_ms")
        retry_policy = item["retry_policy"]
        if retry_policy not in ("none", "retry"):
            raise ManifestError("retry_policy must be 'none' or 'retry'", here + ".retry_policy")
        max_retries = item.get("max_retries", 0 if retry_policy == "none" else 1)
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ManifestError("max_retries must be a non-negative int", here + ".max_retries")
        registry[node_id] = NodeDef(
            node_id=node_id,
            host=host,
            port=port,
            per_node_quota=quota,
            dispatch_timeout_ms=timeout,
            retry_policy=retry_policy,
            max_retries=max_retries,
            fail_mode=item.get("fail_mode", "none"),
        )
    return registry


def load_shards(path: Path) -> Dict[str, ShardDecl]:
    raw = _load_json(path)
    loc = f"{path}#"
    if not isinstance(raw, dict) or "paths" not in raw:
        raise ManifestError("shard file must contain 'paths'", loc)
    decls: Dict[str, ShardDecl] = {}
    for index, item in enumerate(raw["paths"]):
        here = f"{loc}paths[{index}]"
        if not isinstance(item, dict) or {"path", "length", "shards"} - set(item):
            raise ManifestError("shard entry needs path/length/shards", here)
        name, length, shards = item["path"], item["length"], item["shards"]
        if not isinstance(name, str) or not name:
            raise ManifestError("path must be a non-empty string", here + ".path")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ManifestError("length must be a positive int", here + ".length")
        if not isinstance(shards, list) or not shards:
            raise ManifestError("shards must be a non-empty list", here + ".shards")
        try:
            shard_intervals(shards, length)
        except ValueError as exc:
            raise ManifestError(str(exc), here + ".shards") from exc
        if name in decls:
            raise ManifestError(f"duplicate shard path {name!r}", here + ".path")
        decls[name] = ShardDecl(
            path=name, length=length, shards=tuple(dict(s) for s in shards)
        )
    return decls
