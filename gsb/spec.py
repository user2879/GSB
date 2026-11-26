"""Line-aware JSON loading and fail-fast specification validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import DiagnosticError, spec_error
from .model import BatchSpec, CheckpointSpec, EventSpec, LayerSpec, SessionSpec, ShardSpec

FORMAT_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceLocation:
    source: str
    line: int


@dataclass
class LoadedJson:
    data: Any
    source: str
    lines: dict[tuple[Any, ...], int]


class SourceIndex:
    def __init__(self) -> None:
        self._locations: dict[str, SourceLocation] = {}
        self.default_source = "<cli>"

    def add(self, field: str, source: str, line: int) -> None:
        self._locations[field] = SourceLocation(source, line)

    def location(self, field: str | None) -> SourceLocation | None:
        if field is None:
            return None
        if field in self._locations:
            return self._locations[field]
        current = field
        while True:
            if current.endswith("]") and "[" in current:
                current = current[: current.rfind("[")]
            elif "." in current:
                current = current.rsplit(".", 1)[0]
            else:
                break
            if current in self._locations:
                return self._locations[current]
        return None


def _path_string(parts: Sequence[Any]) -> str:
    text = ""
    for part in parts:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("" if not text else ".") + str(part)
    return text


@dataclass
class _Token:
    kind: str
    value: Any
    line: int


def _lex_json(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    line = 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\n":
            line += 1
            index += 1
            continue
        if char.isspace():
            index += 1
            continue
        if char in "{}[]:,":
            tokens.append(_Token(char, char, line))
            index += 1
            continue
        if char == '"':
            start_line = line
            start = index
            index += 1
            while index < length:
                current = text[index]
                if current == "\\":
                    index += 2
                    continue
                if current == '"':
                    index += 1
                    break
                if current == "\n":
                    line += 1
                index += 1
            raw = text[start:index]
            tokens.append(_Token("string", json.loads(raw), start_line))
            continue
        start = index
        while index < length and not text[index].isspace() and text[index] not in "{}[]:,":
            index += 1
        raw = text[start:index]
        if raw == "true":
            tokens.append(_Token("literal", True, line))
        elif raw == "false":
            tokens.append(_Token("literal", False, line))
        elif raw == "null":
            tokens.append(_Token("literal", None, line))
        else:
            tokens.append(_Token("number", raw, line))
    return tokens


def _build_line_map(text: str) -> dict[tuple[Any, ...], int]:
    tokens = _lex_json(text)
    lines: dict[tuple[Any, ...], int] = {(): 1}
    index = 0

    def parse_value(path: tuple[Any, ...]) -> None:
        nonlocal index
        token = tokens[index]
        lines[path] = token.line
        if token.kind == "{":
            index += 1
            while tokens[index].kind != "}":
                key_token = tokens[index]
                if key_token.kind != "string":
                    raise ValueError("invalid token stream")
                key = key_token.value
                child = path + (key,)
                lines[child] = key_token.line
                index += 1
                if tokens[index].kind != ":":
                    raise ValueError("missing colon")
                index += 1
                parse_value(child)
                if tokens[index].kind == ",":
                    index += 1
            index += 1
            return
        if token.kind == "[":
            index += 1
            item = 0
            while tokens[index].kind != "]":
                parse_value(path + (item,))
                item += 1
                if tokens[index].kind == ",":
                    index += 1
            index += 1
            return
        index += 1

    parse_value(())
    return lines


def load_json(path: str | Path) -> LoadedJson:
    source = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError("E_SPEC", f"cannot read JSON file: {exc}", source=source) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DiagnosticError(
            "E_SPEC",
            f"invalid JSON: {exc.msg}",
            source=source,
            line=exc.lineno,
        ) from exc
    try:
        lines = _build_line_map(text)
    except (ValueError, IndexError, json.JSONDecodeError):
        lines = {(): 1}
    return LoadedJson(data=data, source=source, lines=lines)


def _extract_group(doc: LoadedJson, key: str) -> tuple[Any, str | None]:
    if isinstance(doc.data, list):
        return doc.data, None
    if not isinstance(doc.data, dict):
        return doc.data, key
    return doc.data.get(key, []), key


def _copy_group_locations(
    index: SourceIndex,
    doc: LoadedJson,
    source_prefix: tuple[Any, ...],
    target_group: str,
) -> None:
    for path, line in doc.lines.items():
        if path[: len(source_prefix)] != source_prefix:
            continue
        suffix = path[len(source_prefix) :]
        target = _path_string((target_group,) + suffix)
        index.add(target, doc.source, line)


def _copy_top_level_location(
    index: SourceIndex, doc: LoadedJson, source_path: tuple[Any, ...], target: str
) -> None:
    if source_path in doc.lines:
        index.add(target, doc.source, doc.lines[source_path])


def load_session(
    *,
    config_path: str | Path | None = None,
    shards_path: str | Path | None = None,
    checkpoints_path: str | Path | None = None,
    batches_path: str | Path | None = None,
    events_path: str | Path | None = None,
    global_budget: int | None = None,
) -> SessionSpec:
    """Load a combined config or separate declarations, retaining source lines."""

    if config_path is not None:
        doc = load_json(config_path)
        if not isinstance(doc.data, dict):
            raise spec_error("top-level JSON value must be an object", source=doc.source, line=1)
        index = SourceIndex()
        index.default_source = doc.source
        for path, line in doc.lines.items():
            index.add(_path_string(path), doc.source, line)
        return _validate_session(doc.data, index, doc.source)

    if shards_path is None:
        raise spec_error("a config file or shards declaration is required", field="shards")
    data: dict[str, Any] = {"format_version": FORMAT_VERSION}
    index = SourceIndex()
    index.default_source = str(shards_path)
    if global_budget is None:
        raise spec_error("--global-budget is required with separate declarations", field="global_budget")
    data["global_budget"] = global_budget
    index.add("global_budget", "<cli>", 1)
    index.add("format_version", "<default>", 1)

    documents: dict[str, LoadedJson | None] = {
        "shards": load_json(shards_path),
        "checkpoints": load_json(checkpoints_path) if checkpoints_path else None,
        "batches": load_json(batches_path) if batches_path else None,
        "events": load_json(events_path) if events_path else None,
    }
    for group, document in documents.items():
        if document is None:
            data[group] = []
            continue
        items, object_key = _extract_group(document, group)
        if object_key is None:
            source_prefix = ()
            _copy_top_level_location(index, document, (), group)
        else:
            source_prefix = (group,)
            _copy_top_level_location(index, document, (group,), group)
        data[group] = items
        if isinstance(items, list):
            for item_index in range(len(items)):
                item_prefix = source_prefix + (item_index,)
                if source_prefix:
                    _copy_group_locations(index, document, item_prefix, group)
                else:
                    # Root-array documents use item paths such as (0,), (0,'port').
                    for path, line in document.lines.items():
                        if path[:1] == (item_index,):
                            suffix = path[1:]
                            index.add(
                                _path_string((group, item_index) + suffix),
                                document.source,
                                line,
                            )
    return _validate_session(data, index, "<combined>")


def _error(index: SourceIndex, field: str, message: str) -> None:
    location = index.location(field)
    raise spec_error(
        message,
        field=field,
        source=location.source if location else index.default_source,
        line=location.line if location else None,
    )


def _expect_object(value: Any, index: SourceIndex, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _error(index, field, "must be an object")
    return value


def _expect_array(value: Any, index: SourceIndex, field: str) -> list[Any]:
    if not isinstance(value, list):
        _error(index, field, "must be an array")
    return value


def _check_keys(
    obj: dict[str, Any],
    *,
    index: SourceIndex,
    field: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    for key in obj:
        if key not in allowed:
            _error(index, f"{field}.{key}" if field else key, f"unknown field {key!r}")
    for key in required:
        if key not in obj:
            _error(index, f"{field}.{key}" if field else key, f"missing required field {key!r}")


def _string(
    obj: dict[str, Any], key: str, *, index: SourceIndex, field: str, nonempty: bool = True
) -> str:
    value = obj[key]
    if not isinstance(value, str) or (nonempty and not value):
        _error(index, f"{field}.{key}", "must be a non-empty string")
    return value


def _integer(
    obj: dict[str, Any],
    key: str,
    *,
    index: SourceIndex,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _error(index, f"{field}.{key}", "must be an integer")
    if minimum is not None and value < minimum:
        _error(index, f"{field}.{key}", f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _error(index, f"{field}.{key}", f"must be <= {maximum}")
    return value


def _validate_session(data: dict[str, Any], index: SourceIndex, source: str) -> SessionSpec:
    _check_keys(
        data,
        index=index,
        field="",
        required=("format_version", "global_budget", "shards", "checkpoints", "batches", "events"),
    )
    format_version = _integer(
        data, "format_version", index=index, field="", minimum=1, maximum=1
    )
    global_budget = _integer(
        data, "global_budget", index=index, field="", minimum=0
    )
    shard_values = _expect_array(data["shards"], index, "shards")
    if not shard_values:
        _error(index, "shards", "must declare at least one shard")
    cp_values = _expect_array(data["checkpoints"], index, "checkpoints")
    batch_values = _expect_array(data["batches"], index, "batches")
    event_values = _expect_array(data["events"], index, "events")

    shards: list[ShardSpec] = []
    shard_ids: set[str] = set()
    ports: set[int] = set()
    for pos, raw in enumerate(shard_values):
        field = f"shards[{pos}]"
        obj = _expect_object(raw, index, field)
        _check_keys(
            obj,
            index=index,
            field=field,
            required=(
                "shard_id",
                "host",
                "port",
                "lease_limit",
                "concurrency_limit",
                "seed",
                "parameter_version",
                "sample_dim",
                "lease_ttl_ms",
                "layers",
            ),
        )
        shard_id = _string(obj, "shard_id", index=index, field=field)
        if shard_id in shard_ids:
            _error(index, f"{field}.shard_id", f"duplicate shard_id {shard_id!r}")
        shard_ids.add(shard_id)
        host = _string(obj, "host", index=index, field=field)
        if host != "127.0.0.1":
            _error(index, f"{field}.host", "host must be the literal 127.0.0.1")
        port = _integer(obj, "port", index=index, field=field, minimum=1024, maximum=65535)
        if port in ports:
            _error(index, f"{field}.port", f"duplicate loopback port {port}")
        ports.add(port)
        lease_limit = _integer(obj, "lease_limit", index=index, field=field, minimum=0)
        concurrency_limit = _integer(
            obj, "concurrency_limit", index=index, field=field, minimum=0
        )
        seed = _integer(obj, "seed", index=index, field=field)
        parameter_version = _integer(
            obj, "parameter_version", index=index, field=field, minimum=1
        )
        sample_dim = _integer(
            obj, "sample_dim", index=index, field=field, minimum=1, maximum=4096
        )
        lease_ttl_ms = _integer(
            obj, "lease_ttl_ms", index=index, field=field, minimum=1
        )
        layer_values = _expect_array(obj["layers"], index, f"{field}.layers")
        if not layer_values:
            _error(index, f"{field}.layers", "must declare at least one layer")
        layers: list[LayerSpec] = []
        layer_ids: set[str] = set()
        for layer_pos, raw_layer in enumerate(layer_values):
            layer_field = f"{field}.layers[{layer_pos}]"
            layer_obj = _expect_object(raw_layer, index, layer_field)
            _check_keys(
                layer_obj,
                index=index,
                field=layer_field,
                required=("layer_id", "input_dim", "output_dim", "scale"),
                optional=("depends_on",),
            )
            layer_id = _string(layer_obj, "layer_id", index=index, field=layer_field)
            if layer_id in layer_ids:
                _error(index, f"{layer_field}.layer_id", f"duplicate layer_id {layer_id!r}")
            layer_ids.add(layer_id)
            input_dim = _integer(
                layer_obj, "input_dim", index=index, field=layer_field, minimum=1, maximum=4096
            )
            output_dim = _integer(
                layer_obj, "output_dim", index=index, field=layer_field, minimum=1, maximum=4096
            )
            scale = _integer(
                layer_obj, "scale", index=index, field=layer_field, minimum=1
            )
            raw_deps = layer_obj.get("depends_on", [])
            dep_values = _expect_array(raw_deps, index, f"{layer_field}.depends_on")
            deps: list[str] = []
            for dep_pos, dep in enumerate(dep_values):
                if not isinstance(dep, str) or not dep:
                    _error(
                        index,
                        f"{layer_field}.depends_on[{dep_pos}]",
                        "must be a non-empty layer id",
                    )
                if dep in deps:
                    _error(
                        index,
                        f"{layer_field}.depends_on[{dep_pos}]",
                        f"duplicate dependency {dep!r}",
                    )
                deps.append(dep)
            layers.append(LayerSpec(layer_id, input_dim, output_dim, scale, tuple(deps)))

        by_id = {layer.layer_id: layer for layer in layers}
        for layer_pos, layer in enumerate(layers):
            for dep_pos, dep in enumerate(layer.depends_on):
                if dep not in by_id:
                    _error(
                        index,
                        f"{field}.layers[{layer_pos}].depends_on[{dep_pos}]",
                        f"unknown layer reference {dep!r}",
                    )
                if dep == layer.layer_id:
                    _error(
                        index,
                        f"{field}.layers[{layer_pos}].depends_on[{dep_pos}]",
                        "a layer cannot depend on itself",
                    )

        # Stable topological sort is also the cycle check.
        state: dict[str, int] = {layer.layer_id: 0 for layer in layers}

        def visit(layer_id: str) -> None:
            if state[layer_id] == 1:
                layer_pos = next(i for i, layer in enumerate(layers) if layer.layer_id == layer_id)
                _error(
                    index,
                    f"{field}.layers[{layer_pos}].depends_on",
                    "layer dependency graph contains a cycle",
                )
            if state[layer_id] == 2:
                return
            state[layer_id] = 1
            for dep in by_id[layer_id].depends_on:
                visit(dep)
            state[layer_id] = 2

        for layer in layers:
            visit(layer.layer_id)

        for layer_pos, layer in enumerate(layers):
            expected_input = (
                sample_dim
                if not layer.depends_on
                else sum(by_id[dep].output_dim for dep in layer.depends_on)
            )
            if layer.input_dim != expected_input:
                _error(
                    index,
                    f"{field}.layers[{layer_pos}].input_dim",
                    f"must equal {expected_input} for its declared inputs",
                )
        shards.append(
            ShardSpec(
                shard_id=shard_id,
                host=host,
                port=port,
                lease_limit=lease_limit,
                concurrency_limit=concurrency_limit,
                seed=seed,
                parameter_version=parameter_version,
                sample_dim=sample_dim,
                lease_ttl_ms=lease_ttl_ms,
                layers=tuple(layers),
            )
        )

    checkpoints: list[CheckpointSpec] = []
    checkpoint_ids: set[str] = set()
    shard_by_id = {shard.shard_id: shard for shard in shards}
    for pos, raw in enumerate(cp_values):
        field = f"checkpoints[{pos}]"
        obj = _expect_object(raw, index, field)
        _check_keys(
            obj,
            index=index,
            field=field,
            required=("checkpoint_id", "shard_id", "parameter_version", "loss_baseline_digest"),
        )
        checkpoint_id = _string(obj, "checkpoint_id", index=index, field=field)
        if checkpoint_id in checkpoint_ids:
            _error(index, f"{field}.checkpoint_id", f"duplicate checkpoint_id {checkpoint_id!r}")
        checkpoint_ids.add(checkpoint_id)
        shard_id = _string(obj, "shard_id", index=index, field=field)
        if shard_id not in shard_by_id:
            _error(index, f"{field}.shard_id", f"unknown shard reference {shard_id!r}")
        parameter_version = _integer(
            obj, "parameter_version", index=index, field=field, minimum=1
        )
        baseline = _string(obj, "loss_baseline_digest", index=index, field=field)
        if not _HEX64.match(baseline):
            _error(
                index,
                f"{field}.loss_baseline_digest",
                "must be a 64-character lowercase SHA-256 digest",
            )
        checkpoints.append(
            CheckpointSpec(checkpoint_id, shard_id, parameter_version, baseline)
        )

    batches: list[BatchSpec] = []
    batch_ids: set[str] = set()
    batch_orders: set[int] = set()
    checkpoint_by_id = {cp.checkpoint_id: cp for cp in checkpoints}
    for pos, raw in enumerate(batch_values):
        field = f"batches[{pos}]"
        obj = _expect_object(raw, index, field)
        _check_keys(
            obj,
            index=index,
            field=field,
            required=("batch_id", "shard_id", "sample_ids", "resource_units", "order"),
            optional=("waterline", "checkpoint_id", "depends_on_batches"),
        )
        batch_id = _string(obj, "batch_id", index=index, field=field)
        if batch_id in batch_ids:
            _error(index, f"{field}.batch_id", f"duplicate batch_id {batch_id!r}")
        batch_ids.add(batch_id)
        shard_id = _string(obj, "shard_id", index=index, field=field)
        if shard_id not in shard_by_id:
            _error(index, f"{field}.shard_id", f"unknown shard reference {shard_id!r}")
        sample_values = _expect_array(obj["sample_ids"], index, f"{field}.sample_ids")
        if not sample_values:
            _error(index, f"{field}.sample_ids", "must contain at least one sample id")
        sample_ids: list[int] = []
        for sample_pos, sample in enumerate(sample_values):
            if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
                _error(
                    index,
                    f"{field}.sample_ids[{sample_pos}]",
                    "must be a non-negative integer",
                )
            sample_ids.append(sample)
        if len(set(sample_ids)) != len(sample_ids):
            _error(index, f"{field}.sample_ids", "sample ids must be unique")
        resource_units = _integer(
            obj, "resource_units", index=index, field=field, minimum=1
        )
        order = _integer(obj, "order", index=index, field=field, minimum=0)
        if order in batch_orders:
            _error(index, f"{field}.order", f"duplicate batch order {order}")
        batch_orders.add(order)
        waterline = (
            _integer(obj, "waterline", index=index, field=field)
            if "waterline" in obj
            else 0
        )
        checkpoint_id = obj.get("checkpoint_id")
        if checkpoint_id is not None:
            if not isinstance(checkpoint_id, str) or not checkpoint_id:
                _error(index, f"{field}.checkpoint_id", "must be a non-empty checkpoint id")
            checkpoint = checkpoint_by_id.get(checkpoint_id)
            if checkpoint is None:
                _error(
                    index,
                    f"{field}.checkpoint_id",
                    f"unknown checkpoint reference {checkpoint_id!r}",
                )
            if checkpoint.shard_id != shard_id:
                _error(
                    index,
                    f"{field}.checkpoint_id",
                    "checkpoint belongs to a different shard",
                )
        raw_dep_batches = obj.get("depends_on_batches", [])
        dep_values = _expect_array(
            raw_dep_batches, index, f"{field}.depends_on_batches"
        )
        dep_batches: list[str] = []
        for dep_pos, dep in enumerate(dep_values):
            if not isinstance(dep, str) or not dep:
                _error(
                    index,
                    f"{field}.depends_on_batches[{dep_pos}]",
                    "must be a non-empty batch id",
                )
            if dep in dep_batches:
                _error(
                    index,
                    f"{field}.depends_on_batches[{dep_pos}]",
                    f"duplicate batch dependency {dep!r}",
                )
            dep_batches.append(dep)
        batches.append(
            BatchSpec(
                batch_id,
                shard_id,
                tuple(sample_ids),
                resource_units,
                order,
                waterline,
                checkpoint_id,
                tuple(dep_batches),
            )
        )

    for pos, batch in enumerate(batches):
        for dep_pos, dep in enumerate(batch.depends_on_batches):
            if dep not in batch_ids:
                _error(
                    index,
                    f"batches[{pos}].depends_on_batches[{dep_pos}]",
                    f"unknown batch reference {dep!r}",
                )
            if dep == batch.batch_id:
                _error(
                    index,
                    f"batches[{pos}].depends_on_batches[{dep_pos}]",
                    "a batch cannot depend on itself",
                )

    events: list[EventSpec] = []
    event_seqs: set[int] = set()
    previous_seq: int | None = None
    for pos, raw in enumerate(event_values):
        field = f"events[{pos}]"
        obj = _expect_object(raw, index, field)
        event_type = obj.get("type")
        if not isinstance(event_type, str) or not event_type:
            _error(index, f"{field}.type", "must be a non-empty event type")
        allowed_by_type = {
            "commit": {"seq", "type", "shard_id", "checkpoint_id"},
            "rollback": {"seq", "type", "shard_id", "checkpoint_id"},
            "renew": {"seq", "type", "shard_id", "lease_id", "lease_version", "ttl_ms"},
            "cancel": {"seq", "type", "shard_id", "lease_id"},
        }
        if event_type not in allowed_by_type:
            _error(index, f"{field}.type", f"unsupported event type {event_type!r}")
        required_by_type = {
            "commit": ("seq", "type", "shard_id", "checkpoint_id"),
            "rollback": ("seq", "type", "shard_id", "checkpoint_id"),
            "renew": ("seq", "type", "shard_id", "lease_id", "lease_version"),
            "cancel": ("seq", "type", "shard_id", "lease_id"),
        }
        _check_keys(
            obj,
            index=index,
            field=field,
            required=required_by_type[event_type],
            optional=allowed_by_type[event_type] - set(required_by_type[event_type]),
        )
        seq = _integer(obj, "seq", index=index, field=field, minimum=0)
        if seq in event_seqs:
            _error(index, f"{field}.seq", f"duplicate event seq {seq}")
        event_seqs.add(seq)
        if previous_seq is not None and seq <= previous_seq:
            _error(index, f"{field}.seq", "events must be ordered by strictly increasing seq")
        previous_seq = seq
        shard_id = _string(obj, "shard_id", index=index, field=field)
        if shard_id not in shard_by_id:
            _error(index, f"{field}.shard_id", f"unknown shard reference {shard_id!r}")
        checkpoint_id: str | None = None
        lease_id: str | None = None
        lease_version: int | None = None
        ttl_ms: int | None = None
        if event_type in ("commit", "rollback"):
            checkpoint_id = _string(obj, "checkpoint_id", index=index, field=field)
            checkpoint = checkpoint_by_id.get(checkpoint_id)
            if checkpoint is None:
                _error(
                    index,
                    f"{field}.checkpoint_id",
                    f"unknown checkpoint reference {checkpoint_id!r}",
                )
            if checkpoint.shard_id != shard_id:
                _error(
                    index,
                    f"{field}.checkpoint_id",
                    "checkpoint belongs to a different shard",
                )
        else:
            lease_id = _string(obj, "lease_id", index=index, field=field)
        if event_type == "renew":
            lease_version = _integer(
                obj, "lease_version", index=index, field=field, minimum=1
            )
            if "ttl_ms" in obj:
                ttl_ms = _integer(obj, "ttl_ms", index=index, field=field, minimum=1)
        events.append(
            EventSpec(
                seq=seq,
                type=event_type,
                shard_id=shard_id,
                checkpoint_id=checkpoint_id,
                lease_id=lease_id,
                lease_version=lease_version,
                ttl_ms=ttl_ms,
            )
        )
    return SessionSpec(
        format_version=format_version,
        global_budget=global_budget,
        shards=tuple(shards),
        checkpoints=tuple(checkpoints),
        batches=tuple(batches),
        events=tuple(events),
        source=source,
    )
