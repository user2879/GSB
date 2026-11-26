"""Loading and whole-batch validation of the local configuration.

Four JSON documents describe a run:

* ``nodes.json``   - node declarations and global limits
* ``quotas.json``  - monotonic quota version chain with explicit parents
* ``shards.json``  - shard manifest with upstream dependency graph
* ``commands.json``- mutating command stream exercised between batches

Any violation rejects the *entire* batch with ``E_CONFIG_INVALID`` records;
nothing is partially applied.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .errors import E_CONFIG_INVALID, ConfigError, Failure

ALLOWED_FIXED_POINTS = (0, 2, 4, 6)
ALLOWED_QUANT_LEVELS = (1, 2, 5, 10)
ALLOWED_COMMANDS = frozenset(
    {
        "add_node",
        "remove_node",
        "replace_shard",
        "set_quota_version",
        "set_fixed_point",
        "format_roundtrip",
    }
)
PORT_LOW, PORT_HIGH = 20000, 20099


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    port: int
    quota_version: str
    fixed_point: int
    quant_level: int
    lease_limit: int
    capacity_slots: int
    soft_watermark_permille: int
    hard_watermark_permille: int


@dataclass(frozen=True)
class QuotaVersion:
    version: str
    parent: str | None
    seq: int


@dataclass(frozen=True)
class ShardConfig:
    shard_id: str
    index: int
    size_units: int
    seed: int
    sample_size: int
    upstream: tuple[str, ...]


@dataclass(frozen=True)
class Command:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class BatchConfig:
    global_concurrency: int
    batch_point_budget: int
    nodes: tuple[NodeConfig, ...]
    quotas: tuple[QuotaVersion, ...]
    shards: tuple[ShardConfig, ...]
    commands: tuple[Command, ...]
    source_hash: str

    def node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def shard(self, shard_id: str) -> ShardConfig:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)

    def quota_seq(self, version: str) -> int:
        for quota in self.quotas:
            if quota.version == version:
                return quota.seq
        raise KeyError(version)


def _is_pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fail(failures: list[Failure], message: str, **context: Any) -> None:
    failures.append(Failure(E_CONFIG_INVALID, message, context))


def _load_json(path: str, failures: list[Failure]) -> Any:
    if not os.path.isfile(path):
        _fail(failures, "configuration file missing", path=path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(failures, "configuration file unreadable", path=path, error=str(exc))
        return None


def _validate_global(raw: dict[str, Any], failures: list[Failure]) -> tuple[int, int]:
    global_concurrency = raw.get("global_concurrency")
    budget = raw.get("batch_point_budget")
    if not _is_pos_int(global_concurrency):
        _fail(failures, "global_concurrency must be a positive integer")
        global_concurrency = 1
    if not _is_pos_int(budget):
        _fail(failures, "batch_point_budget must be a positive integer")
        budget = 1
    return int(global_concurrency), int(budget)


def _validate_nodes(
    raw_nodes: Any,
    global_concurrency: int,
    quota_versions: set[str],
    failures: list[Failure],
) -> tuple[NodeConfig, ...]:
    nodes: list[NodeConfig] = []
    if not isinstance(raw_nodes, list) or not raw_nodes:
        _fail(failures, "nodes must be a non-empty list")
        return tuple()
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    required = (
        "node_id",
        "port",
        "quota_version",
        "fixed_point",
        "quant_level",
        "lease_limit",
        "capacity_slots",
        "soft_watermark_permille",
        "hard_watermark_permille",
    )
    for pos, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            _fail(failures, "node declaration must be an object", position=pos)
            continue
        missing = [key for key in required if key not in item]
        if missing:
            _fail(failures, "node declaration missing fields", missing=missing, position=pos)
            continue
        node_id = item["node_id"]
        if not isinstance(node_id, str) or not node_id:
            _fail(failures, "node_id must be a non-empty string", position=pos)
            continue
        if node_id in seen_ids:
            _fail(failures, "duplicate node_id", node_id=node_id)
            continue
        port = item["port"]
        if not isinstance(port, int) or isinstance(port, bool) or not (PORT_LOW <= port <= PORT_HIGH):
            _fail(failures, "port outside 20000-20099 loopback range", node_id=node_id, port=port)
            continue
        if port in seen_ports:
            _fail(failures, "duplicate loopback port", node_id=node_id, port=port)
            continue
        if item["quota_version"] not in quota_versions:
            _fail(failures, "node references unknown quota version",
                  node_id=node_id, quota_version=item["quota_version"])
            continue
        if item["fixed_point"] not in ALLOWED_FIXED_POINTS:
            _fail(failures, "fixed_point not in allowed enum",
                  node_id=node_id, fixed_point=item["fixed_point"],
                  allowed=list(ALLOWED_FIXED_POINTS))
            continue
        if item["quant_level"] not in ALLOWED_QUANT_LEVELS:
            _fail(failures, "quant_level not in allowed enum",
                  node_id=node_id, quant_level=item["quant_level"],
                  allowed=list(ALLOWED_QUANT_LEVELS))
            continue
        lease_limit = item["lease_limit"]
        if not _is_pos_int(lease_limit) or lease_limit > global_concurrency:
            _fail(failures, "lease_limit must be positive and <= global_concurrency",
                  node_id=node_id, lease_limit=lease_limit, global_concurrency=global_concurrency)
            continue
        capacity = item["capacity_slots"]
        if not _is_pos_int(capacity):
            _fail(failures, "capacity_slots must be a positive integer", node_id=node_id)
            continue
        soft = item["soft_watermark_permille"]
        hard = item["hard_watermark_permille"]
        if not (isinstance(soft, int) and isinstance(hard, int) and 0 < soft <= hard <= 1000):
            _fail(failures, "watermarks must satisfy 0 < soft <= hard <= 1000", node_id=node_id)
            continue
        seen_ids.add(node_id)
        seen_ports.add(port)
        nodes.append(
            NodeConfig(
                node_id=node_id,
                port=port,
                quota_version=item["quota_version"],
                fixed_point=int(item["fixed_point"]),
                quant_level=int(item["quant_level"]),
                lease_limit=int(lease_limit),
                capacity_slots=int(capacity),
                soft_watermark_permille=int(soft),
                hard_watermark_permille=int(hard),
            )
        )
    return tuple(nodes)


def _validate_quotas(raw_quotas: Any, failures: list[Failure]) -> tuple[QuotaVersion, ...]:
    if not isinstance(raw_quotas, list) or not raw_quotas:
        _fail(failures, "quotas must be a non-empty list")
        return tuple()
    versions: list[QuotaVersion] = []
    seen: set[str] = set()
    last_seq = 0
    for pos, item in enumerate(raw_quotas):
        if not isinstance(item, dict) or "version" not in item or "seq" not in item:
            _fail(failures, "quota entry requires version and seq", position=pos)
            continue
        name = item["version"]
        parent = item.get("parent")
        seq = item["seq"]
        if not isinstance(name, str) or not name:
            _fail(failures, "quota version name must be a non-empty string", position=pos)
            continue
        if name in seen:
            _fail(failures, "duplicate quota version", version=name)
            continue
        if not _is_pos_int(seq) or seq <= last_seq:
            _fail(failures, "quota seq must be strictly increasing positive integers",
                  version=name, seq=seq, previous=last_seq)
            continue
        if pos == 0:
            if parent is not None:
                _fail(failures, "root quota version must declare parent null", version=name)
                continue
        else:
            if not isinstance(parent, str) or parent not in seen:
                _fail(failures, "quota version must reference an existing parent",
                      version=name, parent=parent)
                continue
        seen.add(name)
        last_seq = int(seq)
        versions.append(QuotaVersion(version=name, parent=parent, seq=int(seq)))
    # Explicit acyclicity defence: every parent must appear earlier (checked
    # above); additionally walk each chain to confirm it terminates at root.
    if versions:
        by_name = {q.version: q for q in versions}
        for quota in versions:
            cursor = quota
            hops = 0
            while cursor.parent is not None:
                hops += 1
                if hops > len(versions):
                    _fail(failures, "quota parent chain contains a cycle", version=quota.version)
                    break
                cursor = by_name[cursor.parent]  # type: ignore[assignment]
    return tuple(versions)


def _validate_shards(raw_shards: Any, failures: list[Failure]) -> tuple[ShardConfig, ...]:
    if not isinstance(raw_shards, list) or not raw_shards:
        _fail(failures, "shards must be a non-empty list")
        return tuple()
    shards: list[ShardConfig] = []
    seen: set[str] = set()
    seen_index: set[int] = set()
    for pos, item in enumerate(raw_shards):
        if not isinstance(item, dict):
            _fail(failures, "shard entry must be an object", position=pos)
            continue
        shard_id = item.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            _fail(failures, "shard_id must be a non-empty string", position=pos)
            continue
        if shard_id in seen:
            _fail(failures, "duplicate shard_id", shard_id=shard_id)
            continue
        index = item.get("index")
        if not _is_pos_int(index) or index in seen_index:
            _fail(failures, "shard index must be a unique positive integer", shard_id=shard_id)
            continue
        size_units = item.get("size_units")
        if not _is_pos_int(size_units):
            _fail(failures, "size_units must be a positive integer", shard_id=shard_id)
            continue
        seed = item.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            _fail(failures, "seed must be a non-negative integer", shard_id=shard_id)
            continue
        sample_size = item.get("sample_size")
        if not _is_pos_int(sample_size):
            _fail(failures, "sample_size must be a positive integer", shard_id=shard_id)
            continue
        upstream = item.get("upstream", [])
        if not isinstance(upstream, list) or not all(isinstance(u, str) for u in upstream):
            _fail(failures, "upstream must be a list of shard ids", shard_id=shard_id)
            continue
        seen.add(shard_id)
        seen_index.add(int(index))
        shards.append(
            ShardConfig(
                shard_id=shard_id,
                index=int(index),
                size_units=int(size_units),
                seed=int(seed),
                sample_size=int(sample_size),
                upstream=tuple(upstream),
            )
        )
    # Second pass: upstream references and DAG check.
    ids = {shard.shard_id for shard in shards}
    by_id = {shard.shard_id: shard for shard in shards}
    for shard in shards:
        for upstream_id in shard.upstream:
            if upstream_id not in ids:
                _fail(failures, "upstream shard does not exist",
                      shard_id=shard.shard_id, upstream=upstream_id)
    color: dict[str, int] = {}

    def visit(node_id: str, stack: set[str]) -> None:
        state = color.get(node_id, 0)
        if state == 1 or node_id in stack:
            _fail(failures, "upstream dependency cycle detected", shard_id=node_id)
            return
        if state == 2:
            return
        color[node_id] = 1
        for child in by_id[node_id].upstream:
            if child in ids:
                visit(child, stack | {node_id})
        color[node_id] = 2

    for shard in shards:
        visit(shard.shard_id, set())
    return tuple(shards)


def _validate_commands(raw_commands: Any, failures: list[Failure]) -> tuple[Command, ...]:
    if raw_commands is None:
        raw_commands = []
    if not isinstance(raw_commands, list):
        _fail(failures, "commands must be a list")
        return tuple()
    commands: list[Command] = []
    for pos, item in enumerate(raw_commands):
        if not isinstance(item, dict) or "name" not in item:
            _fail(failures, "command requires a name", position=pos)
            continue
        name = item["name"]
        if name not in ALLOWED_COMMANDS:
            _fail(failures, "unknown command name", name=name, position=pos)
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            _fail(failures, "command args must be an object", name=name, position=pos)
            continue
        commands.append(Command(name=name, args=dict(args)))
    return tuple(commands)


def load_config_dir(config_dir: str) -> BatchConfig:
    """Load and validate every configuration document as one batch.

    Raises :class:`ConfigError` carrying the full failure list on any
    violation; partial state never escapes this function.
    """
    failures: list[Failure] = []
    nodes_raw = _load_json(os.path.join(config_dir, "nodes.json"), failures)
    quotas_raw = _load_json(os.path.join(config_dir, "quotas.json"), failures)
    shards_raw = _load_json(os.path.join(config_dir, "shards.json"), failures)
    commands_raw = _load_json(os.path.join(config_dir, "commands.json"), failures)
    if failures:
        raise ConfigError(failures)

    assert isinstance(nodes_raw, dict)
    global_concurrency, budget = _validate_global(nodes_raw, failures)
    quotas = _validate_quotas(quotas_raw, failures)
    quota_names = {quota.version for quota in quotas}
    nodes = _validate_nodes(
        nodes_raw.get("nodes"), global_concurrency, quota_names, failures
    )
    shards = _validate_shards(shards_raw, failures)
    commands = _validate_commands(commands_raw, failures)
    if failures:
        raise ConfigError(failures)

    source_hash = fingerprint_config(
        global_concurrency, budget, nodes, quotas, shards
    )
    return BatchConfig(
        global_concurrency=global_concurrency,
        batch_point_budget=budget,
        nodes=nodes,
        quotas=quotas,
        shards=shards,
        commands=commands,
        source_hash=source_hash,
    )


def fingerprint_config(
    global_concurrency: int,
    budget: int,
    nodes: tuple[NodeConfig, ...],
    quotas: tuple[QuotaVersion, ...],
    shards: tuple[ShardConfig, ...],
) -> str:
    """Stable hash of the validated configuration surface."""
    import hashlib

    payload = {
        "global_concurrency": global_concurrency,
        "batch_point_budget": budget,
        "nodes": [node.__dict__ for node in sorted(nodes, key=lambda n: n.node_id)],
        "quotas": [quota.__dict__ for quota in quotas],
        "shards": [
            {
                "shard_id": shard.shard_id,
                "index": shard.index,
                "size_units": shard.size_units,
                "seed": shard.seed,
                "sample_size": shard.sample_size,
                "upstream": list(shard.upstream),
            }
            for shard in sorted(shards, key=lambda s: s.index)
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
