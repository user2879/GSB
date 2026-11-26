from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_value
from .errors import CkptError, spec_error


EVENT_TYPES = frozenset({"batch", "renew", "checkpoint", "rollback", "advance"})


def is_int(value: Any) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class JsonDocument:
    path: Path
    text: str
    data: Any
    lines: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "JsonDocument":
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise spec_error(f"cannot read {source}: {exc}", item=str(source), line=1) from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise spec_error(
                f"invalid JSON: {exc.msg}", item=str(source), line=exc.lineno
            ) from exc
        return cls(source, text, data, tuple(text.splitlines()))

    def line_for(self, key: str, value: Any, occurrence: int = 0) -> int:
        key_token = f'"{key}"'
        if isinstance(value, str):
            value_token = json.dumps(value, ensure_ascii=False)
        else:
            value_token = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        hits = [
            number
            for number, line in enumerate(self.lines, start=1)
            if key_token in line and value_token in line
        ]
        if not hits:
            hits = [
                number
                for number, line in enumerate(self.lines, start=1)
                if value_token in line
            ]
        if occurrence < len(hits):
            return hits[occurrence]
        return hits[-1] if hits else 1

    def line_for_seq(self, seq: int, occurrence: int = 0) -> int:
        pattern = re.compile(rf'"seq"\s*:\s*{seq}(?:\s*[,}}])')
        hits = [
            number
            for number, line in enumerate(self.lines, start=1)
            if pattern.search(line)
        ]
        if occurrence < len(hits):
            return hits[occurrence]
        return hits[-1] if hits else 1


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    dimension: int
    scale_factor: int
    derived_from: tuple[str, ...]
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "dimension": self.dimension,
            "scale_factor": self.scale_factor,
            "derived_from": list(self.derived_from),
        }


@dataclass(frozen=True)
class ParameterVersion:
    parameter_version: int
    generation: int
    layers: tuple[LayerSpec, ...]
    line: int

    def by_id(self) -> dict[str, LayerSpec]:
        return {layer.layer_id: layer for layer in self.layers}

    def topological_layers(self) -> tuple[LayerSpec, ...]:
        layers = self.by_id()
        pending = {
            layer_id: set(layer.derived_from) for layer_id, layer in layers.items()
        }
        ordered: list[LayerSpec] = []
        ready = sorted(layer_id for layer_id, deps in pending.items() if not deps)
        while ready:
            layer_id = ready.pop(0)
            ordered.append(layers[layer_id])
            for candidate in sorted(layers):
                if layer_id in pending[candidate]:
                    pending[candidate].remove(layer_id)
                    if not pending[candidate] and candidate not in ready:
                        ready.append(candidate)
            ready.sort()
        return tuple(ordered)

    def descendants(self) -> dict[str, tuple[str, ...]]:
        reverse: dict[str, set[str]] = {layer.layer_id: set() for layer in self.layers}
        for layer in self.layers:
            for parent in layer.derived_from:
                reverse[parent].add(layer.layer_id)
        result: dict[str, tuple[str, ...]] = {}
        for layer_id in sorted(reverse):
            seen: set[str] = set()
            stack = list(sorted(reverse[layer_id]))
            while stack:
                current = stack.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(sorted(reverse[current]))
            result[layer_id] = tuple(sorted(seen))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_version": self.parameter_version,
            "generation": self.generation,
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    host: str
    port: int
    lease_limit: int
    concurrency: int
    seed: int
    parameter_version: int
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "host": self.host,
            "port": self.port,
            "lease_limit": self.lease_limit,
            "concurrency": self.concurrency,
            "seed": self.seed,
            "parameter_version": self.parameter_version,
        }


@dataclass(frozen=True)
class ShardConfig:
    global_lease_limit: int
    shards: tuple[ShardSpec, ...]

    def by_id(self) -> dict[str, ShardSpec]:
        return {shard.shard_id: shard for shard in self.shards}

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_lease_limit": self.global_lease_limit,
            "shards": [shard.to_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class LayerConfig:
    versions: tuple[ParameterVersion, ...]

    def by_version(self) -> dict[int, ParameterVersion]:
        return {version.parameter_version: version for version in self.versions}

    def to_dict(self) -> dict[str, Any]:
        return {"parameter_versions": [version.to_dict() for version in self.versions]}


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    parameter_version: int
    generation: int
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "parameter_version": self.parameter_version,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoints: tuple[CheckpointSpec, ...]

    def by_id(self) -> dict[str, CheckpointSpec]:
        return {checkpoint.checkpoint_id: checkpoint for checkpoint in self.checkpoints}

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints]}


@dataclass(frozen=True)
class EventSpec:
    seq: int
    type: str
    data: dict[str, Any]
    line: int

    def require_str(self, field: str) -> str:
        value = self.data.get(field)
        if not isinstance(value, str) or not value:
            raise spec_error(
                f"event {self.seq} requires non-empty string {field}",
                item=f"event:{self.seq}",
                line=self.line,
            )
        return value

    def require_int(self, field: str, *, positive: bool = False) -> int:
        value = self.data.get(field)
        if not is_int(value) or (positive and value <= 0):
            qualifier = "positive " if positive else ""
            raise spec_error(
                f"event {self.seq} requires {qualifier}integer {field}",
                item=f"event:{self.seq}",
                line=self.line,
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "type": self.type, **self.data}


@dataclass(frozen=True)
class EventConfig:
    events: tuple[EventSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self.events]}


@dataclass(frozen=True)
class SpecBundle:
    shards: ShardConfig
    layers: LayerConfig
    checkpoints: CheckpointConfig
    events: EventConfig
    paths: dict[str, str]
    shard_set_version: str
    parameter_dialect: str
    content_digest: str


def _object(value: Any, item: str, line: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise spec_error(f"{item} must be a JSON object", item=item, line=line)
    return value


def _array(value: Any, item: str, line: int, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = "non-empty " if nonempty else ""
        raise spec_error(f"{item} must be a {suffix}JSON array", item=item, line=line)
    return value


def load_shards(path: str | Path) -> ShardConfig:
    document = JsonDocument.load(path)
    root = _object(document.data, "shards-root", 1)
    global_limit = root.get("global_lease_limit")
    if not is_int(global_limit) or global_limit <= 0:
        raise spec_error(
            "global_lease_limit must be a positive integer",
            item="global_lease_limit",
            line=document.line_for("global_lease_limit", global_limit),
        )
    entries = _array(root.get("shards"), "shards", 1)
    shards: list[ShardSpec] = []
    ids: dict[str, int] = {}
    ports: dict[int, int] = {}
    for index, raw in enumerate(entries):
        line = document.line_for("shard_id", raw.get("shard_id")) if isinstance(raw, dict) else 1
        entry = _object(raw, f"shard[{index}]", line)
        shard_id = entry.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            raise spec_error(
                f"shard[{index}] requires non-empty string shard_id",
                item=f"shard[{index}]",
                line=line,
            )
        line = document.line_for("shard_id", shard_id)
        if shard_id in ids:
            raise spec_error(
                f"duplicate shard_id {shard_id!r}",
                item=shard_id,
                line=document.line_for("shard_id", shard_id, 1),
            )
        ids[shard_id] = line
        host = entry.get("host", "127.0.0.1")
        if host != "127.0.0.1":
            raise spec_error(
                f"shard {shard_id!r} host must be the literal 127.0.0.1",
                item=shard_id,
                line=document.line_for("host", host),
            )
        port = entry.get("port")
        if not is_int(port) or not (1 <= port <= 65535):
            raise spec_error(
                f"shard {shard_id!r} port must be an integer in 1..65535",
                item=shard_id,
                line=document.line_for("port", port),
            )
        if port in ports:
            raise spec_error(
                f"duplicate loopback port {port}",
                item=shard_id,
                line=document.line_for("port", port, 1),
            )
        ports[port] = line
        numbers: dict[str, int] = {}
        for field in ("lease_limit", "concurrency"):
            value = entry.get(field)
            if not is_int(value) or value <= 0:
                raise spec_error(
                    f"shard {shard_id!r} {field} must be a positive integer",
                    item=shard_id,
                    line=document.line_for(field, value),
                )
            numbers[field] = value
        seed = entry.get("seed")
        if not is_int(seed):
            raise spec_error(
                f"shard {shard_id!r} seed must be an integer",
                item=shard_id,
                line=document.line_for("seed", seed),
            )
        parameter_version = entry.get("parameter_version")
        if not is_int(parameter_version) or parameter_version <= 0:
            raise spec_error(
                f"shard {shard_id!r} parameter_version must be a positive integer",
                item=shard_id,
                line=document.line_for("parameter_version", parameter_version),
            )
        shards.append(
            ShardSpec(
                shard_id=shard_id,
                host=host,
                port=port,
                lease_limit=numbers["lease_limit"],
                concurrency=numbers["concurrency"],
                seed=seed,
                parameter_version=parameter_version,
                line=line,
            )
        )
    return ShardConfig(global_lease_limit=global_limit, shards=tuple(shards))


def load_layers(path: str | Path) -> LayerConfig:
    document = JsonDocument.load(path)
    root = _object(document.data, "layers-root", 1)
    entries = _array(root.get("parameter_versions"), "parameter_versions", 1)
    versions: list[ParameterVersion] = []
    version_lines: dict[int, int] = {}
    prior_generation = 0
    for index, raw in enumerate(entries):
        line = (
            document.line_for("parameter_version", raw.get("parameter_version"))
            if isinstance(raw, dict)
            else 1
        )
        entry = _object(raw, f"parameter_version[{index}]", line)
        version_number = entry.get("parameter_version")
        if not is_int(version_number) or version_number <= 0:
            raise spec_error(
                f"parameter_version[{index}] must be a positive integer",
                item=f"parameter_version[{index}]",
                line=line,
            )
        line = document.line_for("parameter_version", version_number)
        if version_number in version_lines:
            raise spec_error(
                f"duplicate parameter_version {version_number}",
                item=str(version_number),
                line=document.line_for("parameter_version", version_number, 1),
            )
        version_lines[version_number] = line
        generation = entry.get("generation")
        if not is_int(generation) or generation <= 0:
            raise spec_error(
                f"parameter_version {version_number} generation must be a positive integer",
                item=str(version_number),
                line=document.line_for("generation", generation),
            )
        if generation <= prior_generation:
            raise spec_error(
                "parameter version generations must be strictly increasing in manifest order",
                item=str(version_number),
                line=document.line_for("generation", generation),
            )
        prior_generation = generation
        raw_layers = _array(entry.get("layers"), f"version {version_number} layers", line)
        layers: list[LayerSpec] = []
        layer_ids: dict[str, int] = {}
        for layer_index, raw_layer in enumerate(raw_layers):
            layer_line = (
                document.line_for("layer_id", raw_layer.get("layer_id"))
                if isinstance(raw_layer, dict)
                else line
            )
            layer_entry = _object(
                raw_layer, f"version {version_number} layer[{layer_index}]", layer_line
            )
            layer_id = layer_entry.get("layer_id")
            if not isinstance(layer_id, str) or not layer_id:
                raise spec_error(
                    f"version {version_number} layer[{layer_index}] requires non-empty layer_id",
                    item=f"version:{version_number}",
                    line=layer_line,
                )
            layer_line = document.line_for("layer_id", layer_id)
            if layer_id in layer_ids:
                raise spec_error(
                    f"duplicate layer_id {layer_id!r} in version {version_number}",
                    item=layer_id,
                    line=document.line_for("layer_id", layer_id, 1),
                )
            layer_ids[layer_id] = layer_line
            dimension = layer_entry.get("dimension")
            if not is_int(dimension) or dimension <= 0:
                raise spec_error(
                    f"layer {layer_id!r} dimension must be a positive integer",
                    item=layer_id,
                    line=document.line_for("dimension", dimension),
                )
            scale_factor = layer_entry.get("scale_factor")
            if not is_int(scale_factor) or scale_factor <= 0:
                raise spec_error(
                    f"layer {layer_id!r} scale_factor must be a positive integer",
                    item=layer_id,
                    line=document.line_for("scale_factor", scale_factor),
                )
            if "derived_from" in layer_entry:
                raw_parents = layer_entry.get("derived_from")
            else:
                raw_parents = layer_entry.get("depends_on")
            if raw_parents is None:
                parents: tuple[str, ...] = ()
            elif isinstance(raw_parents, str):
                parents = (raw_parents,)
            elif isinstance(raw_parents, list):
                if not raw_parents or any(
                    not isinstance(parent, str) or not parent for parent in raw_parents
                ):
                    raise spec_error(
                        f"layer {layer_id!r} derived_from must contain non-empty strings",
                        item=layer_id,
                        line=layer_line,
                    )
                if len(set(raw_parents)) != len(raw_parents):
                    raise spec_error(
                        f"layer {layer_id!r} derived_from contains duplicates",
                        item=layer_id,
                        line=layer_line,
                    )
                parents = tuple(raw_parents)
            else:
                raise spec_error(
                    f"layer {layer_id!r} derived_from must be null, string or string array",
                    item=layer_id,
                    line=layer_line,
                )
            if layer_id in parents:
                raise spec_error(
                    f"layer {layer_id!r} cannot derive from itself",
                    item=layer_id,
                    line=layer_line,
                )
            layers.append(
                LayerSpec(
                    layer_id=layer_id,
                    dimension=dimension,
                    scale_factor=scale_factor,
                    derived_from=parents,
                    line=layer_line,
                )
            )
        by_id = {layer.layer_id: layer for layer in layers}
        for layer in layers:
            for parent in layer.derived_from:
                if parent not in by_id:
                    raise spec_error(
                        f"layer {layer.layer_id!r} derives from missing layer {parent!r}",
                        item=layer.layer_id,
                        line=layer.line,
                    )
        version = ParameterVersion(
            parameter_version=version_number,
            generation=generation,
            layers=tuple(layers),
            line=line,
        )
        ordered = version.topological_layers()
        if len(ordered) != len(version.layers):
            ordered_ids = {layer.layer_id for layer in ordered}
            cyclic = sorted(
                layer.layer_id for layer in version.layers if layer.layer_id not in ordered_ids
            )
            layer = by_id[cyclic[0]]
            raise spec_error(
                f"layer dependency cycle involving {layer.layer_id!r}",
                item=layer.layer_id,
                line=layer.line,
            )
        versions.append(version)
    return LayerConfig(versions=tuple(versions))


def load_checkpoints(path: str | Path) -> CheckpointConfig:
    document = JsonDocument.load(path)
    root = _object(document.data, "checkpoints-root", 1)
    entries = _array(root.get("checkpoints"), "checkpoints", 1, nonempty=False)
    checkpoints: list[CheckpointSpec] = []
    ids: dict[str, int] = {}
    prior_generation = 0
    for index, raw in enumerate(entries):
        line = (
            document.line_for("checkpoint_id", raw.get("checkpoint_id"))
            if isinstance(raw, dict)
            else 1
        )
        entry = _object(raw, f"checkpoint[{index}]", line)
        checkpoint_id = entry.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise spec_error(
                f"checkpoint[{index}] requires non-empty checkpoint_id",
                item=f"checkpoint[{index}]",
                line=line,
            )
        line = document.line_for("checkpoint_id", checkpoint_id)
        if checkpoint_id in ids:
            raise spec_error(
                f"duplicate checkpoint_id {checkpoint_id!r}",
                item=checkpoint_id,
                line=document.line_for("checkpoint_id", checkpoint_id, 1),
            )
        ids[checkpoint_id] = line
        parameter_version = entry.get("parameter_version")
        if not is_int(parameter_version) or parameter_version <= 0:
            raise spec_error(
                f"checkpoint {checkpoint_id!r} parameter_version must be a positive integer",
                item=checkpoint_id,
                line=document.line_for("parameter_version", parameter_version),
            )
        generation = entry.get("generation")
        if not is_int(generation) or generation <= 0:
            raise spec_error(
                f"checkpoint {checkpoint_id!r} generation must be a positive integer",
                item=checkpoint_id,
                line=document.line_for("generation", generation),
            )
        if generation <= prior_generation:
            raise spec_error(
                "checkpoint generations must be strictly increasing in manifest order",
                item=checkpoint_id,
                line=document.line_for("generation", generation),
            )
        prior_generation = generation
        checkpoints.append(
            CheckpointSpec(checkpoint_id, parameter_version, generation, line)
        )
    return CheckpointConfig(checkpoints=tuple(checkpoints))


def load_events(path: str | Path) -> EventConfig:
    document = JsonDocument.load(path)
    if isinstance(document.data, list):
        raw_events = document.data
    elif isinstance(document.data, dict):
        raw_events = _array(document.data.get("events"), "events", 1, nonempty=False)
    else:
        raise spec_error("events root must be an array or object", item="events", line=1)
    events: list[EventSpec] = []
    prior_seq = -1
    batch_keys: set[tuple[str, str]] = set()
    committed: set[str] = set()
    prior_commit_generation: dict[str, int] = {}
    last_commit_generation = 0
    event_seq_occurrences: dict[int, int] = {}
    for raw in raw_events:
        raw_seq = raw.get("seq") if isinstance(raw, dict) else None
        seq_occurrence = event_seq_occurrences.get(raw_seq, 0) if is_int(raw_seq) else 0
        if is_int(raw_seq):
            event_seq_occurrences[raw_seq] = seq_occurrence + 1
        line = (
            document.line_for_seq(raw_seq, seq_occurrence)
            if is_int(raw_seq)
            else document.line_for("type", raw.get("type")) if isinstance(raw, dict) else 1
        )
        entry = _object(raw, "event", line)
        seq = entry.get("seq")
        if not is_int(seq) or seq < 0:
            raise spec_error(
                "event seq must be a non-negative integer",
                item=f"event[{len(events)}]",
                line=line,
            )
        line = document.line_for_seq(seq, event_seq_occurrences[seq] - 1)
        if seq < prior_seq:
            raise spec_error(
                f"event seq must be nondecreasing: {seq} follows {prior_seq}",
                item=f"event:{seq}",
                line=line,
            )
        prior_seq = seq
        event_type = entry.get("type")
        if event_type not in EVENT_TYPES:
            raise spec_error(
                f"unsupported event type {event_type!r}",
                item=f"event:{seq}",
                line=line,
            )
        payload = {key: value for key, value in entry.items() if key not in {"seq", "type"}}
        event = EventSpec(seq=seq, type=event_type, data=payload, line=line)
        if event_type == "batch":
            shard_id = event.require_str("shard_id")
            batch_id = event.require_str("batch_id")
            key = (shard_id, batch_id)
            if key in batch_keys:
                raise spec_error(
                    f"duplicate batch_id {batch_id!r} for shard {shard_id!r}",
                    item=f"{shard_id}/{batch_id}",
                    line=line,
                )
            batch_keys.add(key)
            sample_ids = event.data.get("sample_ids")
            if (
                not isinstance(sample_ids, list)
                or not sample_ids
                or any(not is_int(sample_id) or sample_id < 0 for sample_id in sample_ids)
                or len(set(sample_ids)) != len(sample_ids)
            ):
                raise spec_error(
                    f"batch {shard_id}/{batch_id} sample_ids must be a non-empty unique integer array",
                    item=f"{shard_id}/{batch_id}",
                    line=line,
                )
            if "parameter_version" in payload:
                version = event.data["parameter_version"]
                if not is_int(version) or version <= 0:
                    raise spec_error(
                        f"batch {shard_id}/{batch_id} parameter_version must be positive",
                        item=f"{shard_id}/{batch_id}",
                        line=line,
                    )
        elif event_type == "renew":
            shard_id = event.require_str("shard_id")
            batch_id = event.require_str("batch_id")
            event.require_int("lease_version", positive=True)
            if (shard_id, batch_id) not in batch_keys:
                raise spec_error(
                    f"renew references unknown prior batch {shard_id}/{batch_id}",
                    item=f"{shard_id}/{batch_id}",
                    line=line,
                )
        elif event_type == "checkpoint":
            checkpoint_id = event.require_str("checkpoint_id")
            if checkpoint_id in committed:
                raise spec_error(
                    f"checkpoint {checkpoint_id!r} is committed more than once",
                    item=checkpoint_id,
                    line=line,
                )
            committed.add(checkpoint_id)
            raw_generation = event.data.get("generation")
            if raw_generation is not None:
                if not is_int(raw_generation) or raw_generation <= last_commit_generation:
                    raise spec_error(
                        "checkpoint commit generations must be strictly increasing",
                        item=checkpoint_id,
                        line=line,
                    )
                last_commit_generation = raw_generation
        elif event_type == "rollback":
            checkpoint_id = event.require_str("checkpoint_id")
            if checkpoint_id not in committed:
                raise spec_error(
                    f"rollback targets uncommitted checkpoint {checkpoint_id!r}",
                    item=checkpoint_id,
                    line=line,
                )
        elif event_type == "advance":
            version = event.require_int("parameter_version", positive=True)
            shard_ids = event.data.get("shard_ids")
            if shard_ids is not None and (
                not isinstance(shard_ids, list)
                or not shard_ids
                or any(not isinstance(shard_id, str) or not shard_id for shard_id in shard_ids)
                or len(set(shard_ids)) != len(shard_ids)
            ):
                raise spec_error(
                    "advance.shard_ids must be a non-empty unique string array",
                    item=f"event:{seq}",
                    line=line,
                )
        events.append(event)
    return EventConfig(events=tuple(events))


def load_bundle(
    shards_path: str | Path,
    layers_path: str | Path,
    checkpoints_path: str | Path,
    events_path: str | Path,
) -> SpecBundle:
    shards = load_shards(shards_path)
    layers = load_layers(layers_path)
    checkpoints = load_checkpoints(checkpoints_path)
    events = load_events(events_path)
    versions = layers.by_version()
    checkpoint_by_id = checkpoints.by_id()
    shard_by_id = shards.by_id()
    for shard in shards.shards:
        if shard.parameter_version not in versions:
            raise spec_error(
                f"shard {shard.shard_id!r} references missing parameter_version {shard.parameter_version}",
                item=shard.shard_id,
                line=shard.line,
            )
    for checkpoint in checkpoints.checkpoints:
        if checkpoint.parameter_version not in versions:
            raise spec_error(
                f"checkpoint {checkpoint.checkpoint_id!r} references missing parameter_version {checkpoint.parameter_version}",
                item=checkpoint.checkpoint_id,
                line=checkpoint.line,
            )
    for event in events.events:
        if event.type == "batch" and "parameter_version" in event.data:
            version = event.data["parameter_version"]
            if version not in versions:
                raise spec_error(
                    f"batch references missing parameter_version {version}",
                    item=f"event:{event.seq}",
                    line=event.line,
                )
        elif event.type == "advance":
            version = event.data["parameter_version"]
            if version not in versions:
                raise spec_error(
                    f"advance references missing parameter_version {version}",
                    item=f"event:{event.seq}",
                    line=event.line,
                )
            for shard_id in event.data.get("shard_ids", shard_by_id.keys()):
                if shard_id not in shard_by_id:
                    raise spec_error(
                        f"advance references unknown shard {shard_id!r}",
                        item=f"event:{event.seq}",
                        line=event.line,
                    )
        elif event.type in {"checkpoint", "rollback"}:
            checkpoint_id = event.data["checkpoint_id"]
            if checkpoint_id not in checkpoint_by_id:
                raise spec_error(
                    f"event references unknown checkpoint {checkpoint_id!r}",
                    item=f"event:{event.seq}",
                    line=event.line,
                )
    shard_set_version = sha256_value(shards.to_dict())
    parameter_dialect = sha256_value(
        {
            "dialect": "numpy-int64-fixed-point-v1",
            "versions": [version.to_dict() for version in layers.versions],
        }
    )
    content_digest = sha256_value(
        {
            "shards": shards.to_dict(),
            "layers": layers.to_dict(),
            "checkpoints": checkpoints.to_dict(),
            "events": events.to_dict(),
        }
    )
    return SpecBundle(
        shards=shards,
        layers=layers,
        checkpoints=checkpoints,
        events=events,
        paths={
            "shards": str(Path(shards_path)),
            "layers": str(Path(layers_path)),
            "checkpoints": str(Path(checkpoints_path)),
            "events": str(Path(events_path)),
        },
        shard_set_version=shard_set_version,
        parameter_dialect=parameter_dialect,
        content_digest=content_digest,
    )


def specs_as_canonical_json(bundle: SpecBundle) -> str:
    return canonical_json(
        {
            "shards": bundle.shards.to_dict(),
            "layers": bundle.layers.to_dict(),
            "checkpoints": bundle.checkpoints.to_dict(),
            "events": bundle.events.to_dict(),
        }
    )


def iter_event_lines(events: Iterable[EventSpec]) -> list[tuple[int, int]]:
    return [(event.seq, event.line) for event in events]
