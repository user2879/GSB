from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_digest
from .errors import SpecError
from .graph import stable_topological_order


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def pointer(*parts: str | int) -> str:
    if not parts:
        return "$"
    out = "$"
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += "." + part
    return out


def spec_error(field: str, location: str, message: str) -> SpecError:
    return SpecError(message, field=field, location=location)


@dataclass(frozen=True)
class Spec:
    path: Path | None
    raw: dict[str, Any]
    topo_order: tuple[str, ...]

    @property
    def digest(self) -> str:
        return content_digest(self.raw)

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return list(self.raw["nodes"])

    @property
    def shards(self) -> list[dict[str, Any]]:
        return list(self.raw["shards"])

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self.raw.get("events", []))

    @property
    def node_by_id(self) -> dict[str, dict[str, Any]]:
        return {node["node_id"]: node for node in self.nodes}

    @property
    def shard_by_id(self) -> dict[str, dict[str, Any]]:
        return {shard["shard_id"]: shard for shard in self.shards}

    @property
    def dependencies(self) -> dict[str, tuple[str, ...]]:
        return {
            shard["shard_id"]: tuple(shard.get("upstreams", ()))
            for shard in self.shards
        }

    @property
    def scale_exponent(self) -> int:
        return int(self.raw.get("session", {}).get("scale_exponent", 0))

    @property
    def lease_ttl_ms(self) -> int:
        return int(self.raw.get("session", {}).get("lease_ttl_ms", 5_000))

    @property
    def socket_timeout_ms(self) -> int:
        return int(self.raw.get("session", {}).get("socket_timeout_ms", 2_000))

    @property
    def handshake_timeout_ms(self) -> int:
        return int(self.raw.get("session", {}).get("handshake_timeout_ms", 1_000))


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(
            f"cannot read spec: {exc}",
            field="$",
            location=str(path),
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        location = f"{path}:line {exc.lineno}, column {exc.colno}"
        raise SpecError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            field="$",
            location=location,
        ) from exc
    return validate_spec(raw, source=str(path))


def validate_spec(raw: Any, source: str = "<dict>") -> Spec:
    if not isinstance(raw, dict):
        raise spec_error("$", source, "spec root must be a JSON object")

    version = raw.get("spec_format_version", 1)
    if not is_int(version) or version != 1:
        raise spec_error(
            "spec_format_version",
            pointer("spec_format_version"),
            "spec_format_version must be integer 1",
        )

    session = raw.get("session", {})
    if session is not None and not isinstance(session, dict):
        raise spec_error("session", pointer("session"), "session must be an object")
    session = session or {}
    scale = session.get("scale_exponent", 0)
    if not is_int(scale) or not 0 <= scale <= 3:
        raise spec_error(
            "session.scale_exponent",
            pointer("session", "scale_exponent"),
            "session.scale_exponent must be an integer in [0, 3]",
        )
    for key in ("lease_ttl_ms", "socket_timeout_ms", "handshake_timeout_ms"):
        value = session.get(key, 1_000)
        if not is_int(value) or value <= 0:
            raise spec_error(
                f"session.{key}",
                pointer("session", key),
                f"session.{key} must be a positive integer",
            )

    budget = raw.get("global_budget_units")
    if not is_int(budget) or budget < 0:
        raise spec_error(
            "global_budget_units",
            pointer("global_budget_units"),
            "global_budget_units must be a nonnegative integer",
        )

    chain = raw.get("capacity_chain")
    if not isinstance(chain, list) or not chain:
        raise spec_error(
            "capacity_chain",
            pointer("capacity_chain"),
            "capacity_chain must be a nonempty list",
        )
    versions: set[int] = set()
    previous: int | None = None
    for index, item in enumerate(chain):
        loc = pointer("capacity_chain", index)
        if not isinstance(item, dict):
            raise spec_error("capacity_chain[]", loc, "capacity version must be an object")
        ver = item.get("version")
        if not is_int(ver) or ver <= 0:
            raise spec_error("capacity_chain.version", loc + ".version", "version must be a positive integer")
        if previous is not None and ver <= previous:
            raise spec_error(
                "capacity_chain.version",
                loc + ".version",
                f"version {ver} is not strictly greater than {previous}",
            )
        parent = item.get("parent")
        if parent is not None and (not is_int(parent) or parent not in versions or parent >= ver):
            raise spec_error(
                "capacity_chain.parent",
                loc + ".parent",
                f"parent {parent!r} does not reference an earlier existing version",
            )
        versions.add(ver)
        previous = ver

    current_version = raw.get("capacity_version", previous)
    if not is_int(current_version) or current_version not in versions:
        raise spec_error(
            "capacity_version",
            pointer("capacity_version"),
            "capacity_version must reference a declared chain version",
        )

    nodes = raw.get("nodes")
    if not isinstance(nodes, list):
        raise spec_error("nodes", pointer("nodes"), "nodes must be a list")
    node_ids: set[str] = set()
    ports: set[int] = set()
    for index, node in enumerate(nodes):
        loc = pointer("nodes", index)
        if not isinstance(node, dict):
            raise spec_error("nodes[]", loc, "node must be an object")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise spec_error("nodes.node_id", loc + ".node_id", "node_id must be a nonempty string")
        if node_id in node_ids:
            raise spec_error("nodes.node_id", loc + ".node_id", f"duplicate node_id {node_id!r}")
        node_ids.add(node_id)

        host = node.get("host")
        if host != "127.0.0.1":
            raise spec_error("nodes.host", loc + ".host", "host must be the literal 127.0.0.1")
        port = node.get("port")
        if not is_int(port) or not 1 <= port <= 65535:
            raise spec_error("nodes.port", loc + ".port", "port must be an integer in [1, 65535]")
        if port in ports:
            raise spec_error("nodes.port", loc + ".port", f"duplicate loopback port {port}")
        ports.add(port)

        for field in ("concurrency_quota", "lease_limit"):
            value = node.get(field)
            if not is_int(value) or value < 0:
                raise spec_error(f"nodes.{field}", loc + "." + field, f"{field} must be a nonnegative integer")
        seed = node.get("seed")
        if not is_int(seed):
            raise spec_error("nodes.seed", loc + ".seed", "seed must be an integer")
        node_version = node.get("capacity_version")
        if not is_int(node_version) or node_version not in versions:
            raise spec_error(
                "nodes.capacity_version",
                loc + ".capacity_version",
                "node capacity_version must reference a declared chain version",
            )
        fp_exp = node.get("fixed_point_exponent", 0)
        if not is_int(fp_exp) or not 0 <= fp_exp <= 3:
            raise spec_error(
                "nodes.fixed_point_exponent",
                loc + ".fixed_point_exponent",
                "fixed_point_exponent must be an integer in [0, 3]",
            )

    shards = raw.get("shards")
    if not isinstance(shards, list):
        raise spec_error("shards", pointer("shards"), "shards must be a list")
    shard_ids: set[str] = set()
    for index, shard in enumerate(shards):
        loc = pointer("shards", index)
        if not isinstance(shard, dict):
            raise spec_error("shards[]", loc, "shard must be an object")
        shard_id = shard.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            raise spec_error("shards.shard_id", loc + ".shard_id", "shard_id must be a nonempty string")
        if shard_id in shard_ids:
            raise spec_error("shards.shard_id", loc + ".shard_id", f"duplicate shard_id {shard_id!r}")
        shard_ids.add(shard_id)
        if not is_int(shard.get("sample_rows")) or shard["sample_rows"] <= 0:
            raise spec_error("shards.sample_rows", loc + ".sample_rows", "sample_rows must be a positive integer")
        if not is_int(shard.get("resource_units")) or shard["resource_units"] <= 0:
            raise spec_error("shards.resource_units", loc + ".resource_units", "resource_units must be a positive integer")
        if not is_int(shard.get("priority", 0)):
            raise spec_error("shards.priority", loc + ".priority", "priority must be an integer")
        if "frozen" in shard and not isinstance(shard["frozen"], bool):
            raise spec_error("shards.frozen", loc + ".frozen", "frozen must be a boolean")
        if "node_id" in shard and shard["node_id"] is not None:
            if not isinstance(shard["node_id"], str) or shard["node_id"] not in node_ids:
                raise spec_error("shards.node_id", loc + ".node_id", "node_id must reference a declared node")
        upstreams = shard.get("upstreams", [])
        if not isinstance(upstreams, list) or not all(isinstance(x, str) for x in upstreams):
            raise spec_error("shards.upstreams", loc + ".upstreams", "upstreams must be a list of shard id strings")

    for index, shard in enumerate(shards):
        loc = pointer("shards", index, "upstreams")
        for up_index, upstream in enumerate(shard.get("upstreams", [])):
            if upstream not in shard_ids:
                raise spec_error(
                    "shards.upstreams",
                    loc + f"[{up_index}]",
                    f"upstream shard {upstream!r} does not exist",
                )

    dependency_map = {
        shard["shard_id"]: list(shard.get("upstreams", ())) for shard in shards
    }
    try:
        topo_order = stable_topological_order([s["shard_id"] for s in shards], dependency_map)
    except ValueError as exc:
        cyclic = str(exc).split(":", 1)[-1].strip()
        first = cyclic.split(",")[0] if cyclic else "$"
        idx = next((i for i, s in enumerate(shards) if s["shard_id"] == first), 0)
        raise spec_error(
            "shards.upstreams",
            pointer("shards", idx, "upstreams"),
            "dependency graph contains a cycle",
        ) from exc

    events = raw.get("events", [])
    if not isinstance(events, list):
        raise spec_error("events", pointer("events"), "events must be a list")
    last_seq: int | None = None
    event_seqs: set[int] = set()
    for index, event in enumerate(events):
        loc = pointer("events", index)
        if not isinstance(event, dict):
            raise spec_error("events[]", loc, "event must be an object")
        seq = event.get("seq")
        if not is_int(seq) or seq < 0:
            raise spec_error("events.seq", loc + ".seq", "event seq must be a nonnegative integer")
        if seq in event_seqs or (last_seq is not None and seq <= last_seq):
            raise spec_error("events.seq", loc + ".seq", f"event seq {seq} is not strictly increasing")
        event_seqs.add(seq)
        last_seq = seq
        if not isinstance(event.get("type", "event"), str) or not event.get("type", "event"):
            raise spec_error("events.type", loc + ".type", "event type must be a nonempty string")

    return Spec(path=None if source == "<dict>" else Path(source), raw=raw, topo_order=tuple(topo_order))


def dump_example_spec(path: str | Path) -> Path:
    # Imported lazily so this module has no filesystem template dependency.
    from .example import build_example_spec

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_example_spec(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
