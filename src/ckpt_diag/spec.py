"""Requirement 1: load and strongly validate the declarative JSON spec."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SpecError

HOST = "127.0.0.1"
EVENT_KINDS = {
    "submit", "rollback", "dispatch", "quota_change",
    "lease_expire", "param_release",
}
NON_NEG_INT_FIELDS = ("lease_cap", "concurrency_quota", "seed")


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    dim_in: int
    dim_out: int
    scale: int
    parent: str | None  # None == input root; otherwise another layer id


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    host: str
    port: int
    lease_cap: int
    concurrency_quota: int
    seed: int
    param_version: int
    input_dim: int
    sample_count: int
    layers: tuple[LayerSpec, ...]


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    shard_id: str
    generation: int
    param_version: int
    loss_baseline: str  # hex digest frozen at declaration time


@dataclass(frozen=True)
class EventSpec:
    seq: int
    kind: str
    shard_id: str
    batch_id: str | None = None
    target_checkpoint_id: str | None = None
    param_version: int | None = None
    lease_cap: int | None = None
    concurrency_quota: int | None = None
    cost_units: int = 1
    sample_ids: tuple[str, ...] = ()
    scope_layers: tuple[str, ...] = ()


@dataclass
class Spec:
    global_budget: int
    shards: tuple[ShardSpec, ...]
    checkpoints: tuple[CheckpointSpec, ...]
    events: tuple[EventSpec, ...]
    raw: dict[str, Any] = field(default_factory=dict)

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise SpecError(f"unknown shard_id {shard_id!r}", "shards", "shard_id")

    def layer(self, shard_id: str, layer_id: str) -> LayerSpec:
        for layer in self.shard(shard_id).layers:
            if layer.layer_id == layer_id:
                return layer
        raise SpecError(f"unknown layer {layer_id!r}", f"shard:{shard_id}", "layer_id")


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_ident(value: Any) -> bool:
    return isinstance(value, str) and value != ""


class _Reporter:
    """Raises at the first problem so the first error location is preserved."""

    def fail(self, location: str, field_name: str, message: str) -> None:
        raise SpecError(message, location, field_name)


def _validate_layers(shard_raw: dict, loc: str, rep: _Reporter) -> tuple[LayerSpec, ...]:
    layers_raw = shard_raw.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        rep.fail(loc, "layers", "layers must be a non-empty list")
    seen: set[str] = set()
    layers: list[LayerSpec] = []
    for index, layer_raw in enumerate(layers_raw):
        lloc = f"{loc}.layers[{index}]"
        if not isinstance(layer_raw, dict):
            rep.fail(lloc, "<item>", "layer must be an object")
        layer_id = layer_raw.get("layer_id")
        if not _is_ident(layer_id):
            rep.fail(lloc, "layer_id", "layer_id must be a non-empty string")
        if layer_id in seen:
            rep.fail(lloc, "layer_id", f"duplicate layer_id {layer_id!r}")
        seen.add(layer_id)
        for dim_field in ("dim_in", "dim_out", "scale"):
            if not _is_nonneg_int(layer_raw.get(dim_field)):
                rep.fail(lloc, dim_field, f"{dim_field} must be a non-negative integer")
        if layer_raw["dim_in"] == 0 or layer_raw["dim_out"] == 0:
            rep.fail(lloc, "dim_in", "layer dimensions must be positive")
        parent = layer_raw.get("parent")
        if parent is not None and not _is_ident(parent):
            rep.fail(lloc, "parent", "parent must be null or a non-empty string")
        layers.append(
            LayerSpec(layer_id, layer_raw["dim_in"], layer_raw["dim_out"],
                      layer_raw["scale"], parent)
        )
    ids = {layer.layer_id for layer in layers}
    # parent references must exist
    for index, layer in enumerate(layers):
        if layer.parent is not None and layer.parent not in ids:
            rep.fail(f"{loc}.layers[{index}]", "parent",
                     f"parent {layer.parent!r} does not reference an existing layer")
    # acyclic dependency graph (edge parent -> layer)
    graph = {layer.layer_id: (layer.parent,) if layer.parent else () for layer in layers}
    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        if state.get(node) == 1:
            rep.fail(f"{loc}.layers", "layers",
                     f"layer dependency cycle detected: {' -> '.join(stack + [node])}")
        if state.get(node) == 2:
            return
        state[node] = 1
        for nxt in graph[node]:
            visit(nxt, stack + [node])
        state[node] = 2

    for layer_id in graph:
        visit(layer_id, [])
    return tuple(layers)


def _validate_shards(raw: dict, rep: _Reporter) -> tuple[ShardSpec, ...]:
    shards_raw = raw.get("shards")
    if not isinstance(shards_raw, list) or not shards_raw:
        rep.fail("shards", "<root>", "shards must be a non-empty list")
    shard_ids: set[str] = set()
    ports: set[int] = set()
    shards: list[ShardSpec] = []
    for index, shard_raw in enumerate(shards_raw):
        loc = f"shards[{index}]"
        if not isinstance(shard_raw, dict):
            rep.fail(loc, "<item>", "shard must be an object")
        shard_id = shard_raw.get("shard_id")
        if not _is_ident(shard_id):
            rep.fail(loc, "shard_id", "shard_id must be a non-empty string")
        if shard_id in shard_ids:
            rep.fail(loc, "shard_id", f"duplicate shard_id {shard_id!r}")
        shard_ids.add(shard_id)
        if shard_raw.get("host") != HOST:
            rep.fail(loc, "host", f"host must be the literal {HOST}")
        port = shard_raw.get("port")
        if not _is_nonneg_int(port) or not (1 <= port <= 65535):
            rep.fail(loc, "port", "port must be an integer in [1, 65535]")
        if port in ports:
            rep.fail(loc, "port", f"duplicate loopback port {port}")
        ports.add(port)
        for field_name in NON_NEG_INT_FIELDS:
            if not _is_nonneg_int(shard_raw.get(field_name)):
                rep.fail(loc, field_name, f"{field_name} must be a non-negative integer")
        if not _is_nonneg_int(shard_raw.get("param_version")):
            rep.fail(loc, "param_version", "param_version must be a non-negative integer")
        if not _is_nonneg_int(shard_raw.get("input_dim")) or shard_raw["input_dim"] == 0:
            rep.fail(loc, "input_dim", "input_dim must be a positive integer")
        if not _is_nonneg_int(shard_raw.get("sample_count")) or shard_raw["sample_count"] == 0:
            rep.fail(loc, "sample_count", "sample_count must be a positive integer")
        layers = _validate_layers(shard_raw, loc, rep)
        shards.append(
            ShardSpec(shard_id, HOST, port, shard_raw["lease_cap"],
                      shard_raw["concurrency_quota"], shard_raw["seed"],
                      shard_raw["param_version"], shard_raw["input_dim"],
                      shard_raw["sample_count"], layers)
        )
    return tuple(shards)


def _validate_checkpoints(raw: dict, shard_ids: set[str],
                          rep: _Reporter) -> tuple[CheckpointSpec, ...]:
    cps_raw = raw.get("checkpoints")
    if not isinstance(cps_raw, list):
        rep.fail("checkpoints", "<root>", "checkpoints must be a list")
    ids: set[str] = set()
    last_gen: dict[str, int] = {}
    checkpoints: list[CheckpointSpec] = []
    for index, cp_raw in enumerate(cps_raw):
        loc = f"checkpoints[{index}]"
        if not isinstance(cp_raw, dict):
            rep.fail(loc, "<item>", "checkpoint must be an object")
        cp_id = cp_raw.get("checkpoint_id")
        if not _is_ident(cp_id):
            rep.fail(loc, "checkpoint_id", "checkpoint_id must be a non-empty string")
        if cp_id in ids:
            rep.fail(loc, "checkpoint_id", f"duplicate checkpoint_id {cp_id!r}")
        ids.add(cp_id)
        shard_id = cp_raw.get("shard_id")
        if shard_id not in shard_ids:
            rep.fail(loc, "shard_id", f"checkpoint references unknown shard {shard_id!r}")
        gen = cp_raw.get("generation")
        if not _is_nonneg_int(gen):
            rep.fail(loc, "generation", "generation must be a non-negative integer")
        if shard_id in last_gen and gen <= last_gen[shard_id]:
            rep.fail(loc, "generation",
                     "generation must be strictly increasing within a shard chain")
        last_gen[shard_id] = gen
        if not _is_nonneg_int(cp_raw.get("param_version")):
            rep.fail(loc, "param_version", "param_version must be a non-negative integer")
        baseline = cp_raw.get("loss_baseline")
        if not _is_ident(baseline):
            rep.fail(loc, "loss_baseline", "loss_baseline must be a non-empty hex string")
        checkpoints.append(
            CheckpointSpec(cp_id, shard_id, gen, cp_raw["param_version"], baseline)
        )
    return tuple(checkpoints)


def _validate_events(raw: dict, shard_ids: set[str], cp_ids: set[str],
                     layer_ids: dict[str, set[str]], rep: _Reporter) -> tuple[EventSpec, ...]:
    events_raw = raw.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        rep.fail("events", "<root>", "events must be a non-empty list")
    prev_seq: int | None = None
    events: list[EventSpec] = []
    for index, ev_raw in enumerate(events_raw):
        loc = f"events[{index}]"
        if not isinstance(ev_raw, dict):
            rep.fail(loc, "<item>", "event must be an object")
        seq = ev_raw.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            rep.fail(loc, "seq", "seq must be an integer")
        if prev_seq is not None and seq <= prev_seq:
            rep.fail(loc, "seq", "event seq must be strictly increasing")
        prev_seq = seq
        kind = ev_raw.get("kind")
        if kind not in EVENT_KINDS:
            rep.fail(loc, "kind", f"unknown event kind {kind!r}; allowed {sorted(EVENT_KINDS)}")
        shard_id = ev_raw.get("shard_id")
        if shard_id not in shard_ids:
            rep.fail(loc, "shard_id", f"event references unknown shard {shard_id!r}")
        batch_id = ev_raw.get("batch_id")
        if kind in {"dispatch", "submit"} and not _is_ident(batch_id):
            rep.fail(loc, "batch_id", f"{kind} event requires a non-empty batch_id")
        if batch_id is not None and not _is_ident(batch_id):
            rep.fail(loc, "batch_id", "batch_id must be a non-empty string or null")
        target = ev_raw.get("target_checkpoint_id")
        if kind == "rollback":
            if target not in cp_ids:
                rep.fail(loc, "target_checkpoint_id",
                         "rollback target must be a checkpoint already on the chain")
        elif target is not None and not _is_ident(target):
            rep.fail(loc, "target_checkpoint_id", "target must be a non-empty string or null")
        param_version = ev_raw.get("param_version")
        if kind in {"dispatch", "param_release"} and not _is_nonneg_int(param_version):
            rep.fail(loc, "param_version", f"{kind} event requires a non-negative param_version")
        lease_cap = ev_raw.get("lease_cap")
        if kind == "quota_change" and not _is_nonneg_int(lease_cap):
            rep.fail(loc, "lease_cap", "quota_change requires a non-negative lease_cap")
        quota = ev_raw.get("concurrency_quota")
        if kind == "quota_change" and not _is_nonneg_int(quota):
            rep.fail(loc, "concurrency_quota",
                     "quota_change requires a non-negative concurrency_quota")
        cost = ev_raw.get("cost_units", 1)
        if not _is_nonneg_int(cost) or cost == 0:
            rep.fail(loc, "cost_units", "cost_units must be a positive integer")
        sample_ids = ev_raw.get("sample_ids", [])
        if not isinstance(sample_ids, list) or any(not _is_ident(s) for s in sample_ids):
            rep.fail(loc, "sample_ids", "sample_ids must be a list of non-empty strings")
        if len(set(sample_ids)) != len(sample_ids):
            rep.fail(loc, "sample_ids", "sample_ids must be unique within an event")
        scope = ev_raw.get("scope_layers", [])
        if not isinstance(scope, list) or any(not _is_ident(s) for s in scope):
            rep.fail(loc, "scope_layers", "scope_layers must be a list of non-empty strings")
        unknown_scope = set(scope) - layer_ids.get(shard_id, set())
        if unknown_scope:
            rep.fail(loc, "scope_layers",
                     f"unknown layers for shard {shard_id!r}: {sorted(unknown_scope)}")
        events.append(
            EventSpec(seq, kind, shard_id, batch_id, target, param_version,
                      lease_cap, quota, cost, tuple(sample_ids), tuple(scope))
        )
    return tuple(events)


def load_spec(path: str | Path) -> Spec:
    """Load and fully validate a specification JSON file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"spec file not found: {path}", str(path), "<file>")
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc.msg}", str(path), f"line {exc.lineno}")
    if not isinstance(raw, dict):
        raise SpecError("top-level spec must be an object", "<root>", "<root>")
    rep = _Reporter()
    if not _is_nonneg_int(raw.get("global_budget")) or raw["global_budget"] == 0:
        rep.fail("global_budget", "global_budget", "global_budget must be a positive integer")
    shards = _validate_shards(raw, rep)
    shard_ids = {shard.shard_id for shard in shards}
    layer_ids = {shard.shard_id: {layer.layer_id for layer in shard.layers}
                 for shard in shards}
    checkpoints = _validate_checkpoints(raw, shard_ids, rep)
    cp_ids = {cp.checkpoint_id for cp in checkpoints}
    events = _validate_events(raw, shard_ids, cp_ids, layer_ids, rep)
    return Spec(raw["global_budget"], shards, checkpoints, events, raw)
