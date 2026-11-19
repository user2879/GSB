from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import PipelineError, e_spec
from .fixed import FixedGrid, validate_grid_power

SUPPORTED_CACHE_KEY_FIELDS = frozenset(
    {"path", "query", "variant", "tenant", "accept_encoding", "region"}
)
SUPPORTED_EVICTIONS = frozenset({"LRU", "FIFO"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


@dataclass(frozen=True)
class Dependency:
    path: str
    coefficient: int

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "coefficient": self.coefficient}


@dataclass(frozen=True)
class ContentEntry:
    path: str
    etag: str
    body: bytes
    ttl_seconds: int
    dependencies: tuple[Dependency, ...] = ()

    @property
    def derived(self) -> bool:
        return bool(self.dependencies)

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "etag": self.etag,
            "body_hex": self.body.hex(),
            "ttl_seconds": self.ttl_seconds,
            "derivation": (
                {"inputs": [dep.as_dict() for dep in self.dependencies]}
                if self.dependencies
                else None
            ),
        }


@dataclass(frozen=True)
class NodePolicy:
    node_id: str
    port: int
    cache_key_fields: tuple[str, ...]
    capacity: int
    eviction: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.node_id,
            "port": self.port,
            "cache_key_fields": list(self.cache_key_fields),
            "capacity": self.capacity,
            "eviction": self.eviction,
        }


@dataclass(frozen=True)
class RequestSpec:
    seq: int
    request_id: str
    node_id: str
    path: str
    query: str
    at_ms: int
    batch: int
    key_values: dict[str, str]

    def cache_key_values(self, fields: Iterable[str]) -> list[list[str]]:
        return [[field, self.key_values[field]] for field in fields]

    def as_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "request_id": self.request_id,
            "node_id": self.node_id,
            "path": self.path,
            "query": self.query,
            "at_ms": self.at_ms,
            "batch": self.batch,
            "key": dict(sorted(self.key_values.items())),
        }


@dataclass(frozen=True)
class SourceUpdateSpec:
    at_ms: int
    path: str
    etag: str | None = None
    body: bytes | None = None
    ttl_seconds: int | None = None
    dependencies: tuple[Dependency, ...] | None = None

    @property
    def derivation_explicit(self) -> bool:
        return self.dependencies is not None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"at_ms": self.at_ms, "path": self.path}
        if self.etag is not None:
            payload["etag"] = self.etag
        if self.body is not None:
            payload["body_hex"] = self.body.hex()
        if self.ttl_seconds is not None:
            payload["ttl_seconds"] = self.ttl_seconds
        if self.dependencies is not None:
            payload["derivation"] = {
                "inputs": [dep.as_dict() for dep in self.dependencies]
            }
        return payload


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    grid_power: int
    batch_size: int
    source_port: int
    host: str

    @property
    def grid(self) -> FixedGrid:
        return FixedGrid(self.grid_power)


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    session: SessionSpec
    content: dict[str, ContentEntry]
    nodes: dict[str, NodePolicy]
    requests: tuple[RequestSpec, ...]
    source_updates: tuple[SourceUpdateSpec, ...]

    @property
    def grid(self) -> FixedGrid:
        return FixedGrid(self.session.grid_power)

    def batch_ids(self) -> list[int]:
        return sorted({request.batch for request in self.requests})

    def requests_for_batch(self, batch_id: int) -> list[RequestSpec]:
        return [request for request in self.requests if request.batch == batch_id]

    def node_sort_key(self, node_id: str) -> str:
        return node_id

    def to_jsonable(self) -> dict[str, Any]:
        return self.raw


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _need_object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise e_spec("expected a JSON object", path, "$")
    return value


def _need_list(value: object, path: str, field: str = "$") -> list[Any]:
    if not isinstance(value, list):
        raise e_spec("expected a JSON array", path, field)
    return value


def _need_str(value: object, path: str, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise e_spec("expected a non-empty string", path, field)
    return value


def _need_positive_int(value: object, path: str, field: str) -> int:
    if not _is_int(value) or value <= 0:
        raise e_spec("expected a positive integer", path, field)
    return int(value)


def _need_nonnegative_int(value: object, path: str, field: str) -> int:
    if not _is_int(value) or value < 0:
        raise e_spec("expected a non-negative integer", path, field)
    return int(value)


def _need_port(value: object, path: str, field: str = "port") -> int:
    port = _need_positive_int(value, path, field)
    if port < 1024 or port > 65535:
        raise e_spec("loopback port must be in [1024, 65535]", path, field, received=port)
    return port


def _parse_body(obj: dict[str, Any], path: str) -> bytes:
    if "body_hex" in obj:
        text = _need_str(obj.get("body_hex"), path, "body_hex", nonempty=False)
        try:
            return bytes.fromhex(text)
        except ValueError as exc:
            raise e_spec("body_hex must contain an even number of hex digits", path, "body_hex") from exc
    if "body" not in obj:
        raise e_spec("missing byte content; provide body or body_hex", path, "body")
    body = _need_str(obj.get("body"), path, "body", nonempty=False)
    return body.encode("utf-8")


def _parse_dependencies(
    value: object,
    path: str,
    field: str,
    content_paths: set[str],
) -> tuple[Dependency, ...]:
    if value is None:
        return ()
    obj = _need_object(value, f"{path}.{field}")
    if "inputs" not in obj:
        raise e_spec("derivation object requires inputs", f"{path}.{field}", "inputs")
    inputs = _need_list(obj["inputs"], f"{path}.{field}.inputs")
    if not inputs:
        raise e_spec("derivation inputs must not be empty", f"{path}.{field}.inputs", "$")
    result: list[Dependency] = []
    seen: set[str] = set()
    for index, raw_dep in enumerate(inputs):
        dep_path = f"{path}.{field}.inputs[{index}]"
        dep = _need_object(raw_dep, dep_path)
        target = _need_str(dep.get("path"), dep_path, "path")
        if target not in content_paths:
            raise e_spec(
                "derivation references an unknown content path",
                dep_path,
                "path",
                referenced_path=target,
            )
        if target in seen:
            raise e_spec(
                "derivation input paths must be unique within a content entry",
                dep_path,
                "path",
                referenced_path=target,
            )
        coefficient = dep.get("coefficient")
        if not _is_int(coefficient) or coefficient <= 0:
            raise e_spec(
                "derivation coefficient must be a positive integer",
                dep_path,
                "coefficient",
                received=coefficient,
            )
        seen.add(target)
        result.append(Dependency(target, int(coefficient)))
    return tuple(result)


def _cycle_for(content: dict[str, ContentEntry]) -> list[str] | None:
    state: dict[str, int] = {path: 0 for path in content}
    stack: list[str] = []

    def visit(path: str) -> list[str] | None:
        state[path] = 1
        stack.append(path)
        for dep in content[path].dependencies:
            if state[dep.path] == 1:
                start = stack.index(dep.path)
                return stack[start:] + [dep.path]
            if state[dep.path] == 0:
                found = visit(dep.path)
                if found:
                    return found
        stack.pop()
        state[path] = 2
        return None

    for path in sorted(content):
        if state[path] == 0:
            cycle = visit(path)
            if cycle:
                return cycle
    return None


def load_spec(path: str | Path, *, grid_power: int | None = None) -> Spec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(
            "E_SPEC", "cannot read spec file", path=str(source), field="file", details={"reason": str(exc)}
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise e_spec(
            "invalid JSON",
            str(source),
            f"line:{exc.lineno}:column:{exc.colno}",
            reason=exc.msg,
        ) from exc
    return parse_spec(raw, grid_power=grid_power, source_name=str(source))


def parse_spec(raw: object, *, grid_power: int | None = None, source_name: str = "<memory>") -> Spec:
    root = _need_object(raw, source_name)
    schema_version = root.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version != 1:
        raise e_spec("unsupported schema_version", source_name, "schema_version", received=schema_version)

    session_raw = _need_object(root.get("session"), "session")
    session_id = _need_str(session_raw.get("session_id"), "session", "session_id")
    raw_grid = session_raw.get("grid_power")
    if grid_power is None:
        grid_power = raw_grid if raw_grid is not None else 3
    validate_grid_power(grid_power)
    if raw_grid is not None:
        validate_grid_power(raw_grid)
    batch_size = _need_positive_int(session_raw.get("batch_size", 3), "session", "batch_size")
    source_port = _need_port(session_raw.get("source_port"), "session", "source_port")
    host = _need_str(session_raw.get("host", "127.0.0.1"), "session", "host")
    if host not in LOOPBACK_HOSTS:
        raise e_spec("only loopback hosts are permitted", "session", "host", received=host)

    content_raw = _need_list(root.get("content"), "content")
    content_paths: set[str] = set()
    content_entries: dict[str, tuple[str, str, bytes, int, tuple[Dependency, ...]]] = {}
    for index, item in enumerate(content_raw):
        item_path = f"content[{index}]"
        obj = _need_object(item, item_path)
        content_path = _need_str(obj.get("path"), item_path, "path")
        if not content_path.startswith("/"):
            raise e_spec("content path must start with '/'", item_path, "path", received=content_path)
        if content_path in content_paths:
            raise e_spec("duplicate content path", item_path, "path", duplicate=content_path)
        content_paths.add(content_path)
        etag = _need_str(obj.get("etag"), item_path, "etag")
        body = _parse_body(obj, item_path)
        ttl = _need_positive_int(obj.get("ttl_seconds"), item_path, "ttl_seconds")
        # Dependencies are populated after every path is known.
        content_entries[content_path] = (content_path, etag, body, ttl, ())

    content: dict[str, ContentEntry] = {}
    for index, item in enumerate(content_raw):
        item_path = f"content[{index}]"
        obj = _need_object(item, item_path)
        content_path = obj["path"]
        base = content_entries[content_path]
        dependencies = _parse_dependencies(
            obj.get("derivation"), item_path, "derivation", content_paths
        )
        content[content_path] = ContentEntry(
            path=base[0],
            etag=base[1],
            body=base[2],
            ttl_seconds=base[3],
            dependencies=dependencies,
        )

    cycle = _cycle_for(content)
    if cycle:
        first = cycle[0]
        raise PipelineError(
            "E_CYCLE",
            "derivation graph contains a cycle",
            path=f"content[path={first}].derivation",
            field="inputs",
            details={"cycle": cycle, "spec_rejected": True},
        )

    nodes_raw = _need_list(root.get("nodes"), "nodes")
    if len(nodes_raw) < 3:
        raise e_spec("at least three edge nodes are required", "nodes", "length", received=len(nodes_raw))
    nodes: dict[str, NodePolicy] = {}
    used_ports: dict[int, str] = {source_port: "session.source_port"}
    for index, item in enumerate(nodes_raw):
        item_path = f"nodes[{index}]"
        obj = _need_object(item, item_path)
        node_id = _need_str(obj.get("id"), item_path, "id")
        if node_id in nodes:
            raise e_spec("duplicate node id", item_path, "id", duplicate=node_id)
        port = _need_port(obj.get("port"), item_path, "port")
        if port in used_ports:
            raise e_spec(
                "loopback ports must be unique",
                item_path,
                "port",
                duplicate_of=used_ports[port],
            )
        used_ports[port] = item_path + ".port"
        key_fields_raw = _need_list(obj.get("cache_key_fields"), item_path, "cache_key_fields")
        if not key_fields_raw:
            raise e_spec("cache_key_fields must not be empty", item_path, "cache_key_fields")
        key_fields: list[str] = []
        for field_index, raw_field in enumerate(key_fields_raw):
            field_path = f"{item_path}.cache_key_fields[{field_index}]"
            field = _need_str(raw_field, field_path, "$")
            if field not in SUPPORTED_CACHE_KEY_FIELDS:
                raise e_spec(
                    "unsupported cache-key field",
                    field_path,
                    "$",
                    received=field,
                    supported=sorted(SUPPORTED_CACHE_KEY_FIELDS),
                )
            if field in key_fields:
                raise e_spec("cache-key fields must be unique", field_path, "$", duplicate=field)
            key_fields.append(field)
        if "path" not in key_fields:
            raise e_spec(
                "cache-key fields must include path to prevent cross-path aliasing",
                item_path,
                "cache_key_fields",
            )
        capacity = _need_positive_int(obj.get("capacity"), item_path, "capacity")
        eviction_raw = _need_str(obj.get("eviction"), item_path, "eviction").upper()
        if eviction_raw not in SUPPORTED_EVICTIONS:
            raise e_spec(
                "unsupported eviction rule",
                item_path,
                "eviction",
                received=eviction_raw,
                supported=sorted(SUPPORTED_EVICTIONS),
            )
        nodes[node_id] = NodePolicy(node_id, port, tuple(key_fields), capacity, eviction_raw)

    requests_raw = _need_list(root.get("requests"), "requests")
    requests: list[RequestSpec] = []
    request_ids: set[str] = set()
    last_at = -1
    last_batch = 0
    for index, item in enumerate(requests_raw):
        item_path = f"requests[{index}]"
        obj = _need_object(item, item_path)
        request_id = _need_str(obj.get("request_id", f"req-{index + 1:04d}"), item_path, "request_id")
        if request_id in request_ids:
            raise e_spec("duplicate request_id", item_path, "request_id", duplicate=request_id)
        request_ids.add(request_id)
        node_id = _need_str(obj.get("node_id"), item_path, "node_id")
        if node_id not in nodes:
            raise e_spec("request references an unknown node", item_path, "node_id", received=node_id)
        content_path = _need_str(obj.get("path"), item_path, "path")
        if content_path not in content:
            raise e_spec("request references an unknown content path", item_path, "path", received=content_path)
        query = obj.get("query", "")
        if not isinstance(query, str):
            raise e_spec("expected a string", item_path, "query")
        at_ms = _need_nonnegative_int(obj.get("at_ms"), item_path, "at_ms")
        if at_ms < last_at:
            raise e_spec(
                "requests must be in non-decreasing at_ms order",
                item_path,
                "at_ms",
                previous_at_ms=last_at,
                received=at_ms,
            )
        last_at = at_ms
        if "batch" in obj:
            batch = _need_positive_int(obj.get("batch"), item_path, "batch")
        else:
            batch = index // batch_size + 1
        if batch < last_batch:
            raise e_spec(
                "batch ids must be non-decreasing in request order",
                item_path,
                "batch",
                previous_batch=last_batch,
                received=batch,
            )
        last_batch = batch
        key_obj = obj.get("key", {})
        if not isinstance(key_obj, dict):
            raise e_spec("request key must be a JSON object", item_path, "key")
        key_values: dict[str, str] = {"path": content_path, "query": query}
        for field in nodes[node_id].cache_key_fields:
            if field in key_values:
                continue
            if field in key_obj:
                value = key_obj[field]
            elif field in obj:
                value = obj[field]
            else:
                raise e_spec(
                    "request is missing a cache-key field required by its node",
                    item_path,
                    field,
                )
            if not isinstance(value, str):
                raise e_spec("cache-key values must be strings", item_path, field)
            key_values[field] = value
        for field, value in key_obj.items():
            if field not in SUPPORTED_CACHE_KEY_FIELDS:
                raise e_spec(
                    "unsupported request key field",
                    f"{item_path}.key",
                    field,
                    supported=sorted(SUPPORTED_CACHE_KEY_FIELDS),
                )
            if not isinstance(value, str):
                raise e_spec("cache-key values must be strings", f"{item_path}.key", field)
            key_values[field] = value
        requests.append(
            RequestSpec(
                seq=index,
                request_id=request_id,
                node_id=node_id,
                path=content_path,
                query=query,
                at_ms=at_ms,
                batch=batch,
                key_values=key_values,
            )
        )

    updates_raw = root.get("source_updates", [])
    updates_list = _need_list(updates_raw, "source_updates")
    updates: list[SourceUpdateSpec] = []
    last_update_at = -1
    for index, item in enumerate(updates_list):
        item_path = f"source_updates[{index}]"
        obj = _need_object(item, item_path)
        at_ms = _need_nonnegative_int(obj.get("at_ms"), item_path, "at_ms")
        if at_ms < last_update_at:
            raise e_spec(
                "source_updates must be in non-decreasing at_ms order",
                item_path,
                "at_ms",
                previous_at_ms=last_update_at,
                received=at_ms,
            )
        last_update_at = at_ms
        content_path = _need_str(obj.get("path"), item_path, "path")
        if content_path not in content:
            raise e_spec("source update references an unknown content path", item_path, "path")
        provided = [key for key in ("etag", "body", "body_hex", "ttl_seconds", "derivation") if key in obj]
        if not provided:
            raise e_spec("source update must change at least one content field", item_path, "$")
        etag = None if "etag" not in obj else _need_str(obj.get("etag"), item_path, "etag")
        body = None
        if "body" in obj or "body_hex" in obj:
            body = _parse_body(obj, item_path)
        ttl = None
        if "ttl_seconds" in obj:
            ttl = _need_positive_int(obj.get("ttl_seconds"), item_path, "ttl_seconds")
        dependencies = None
        if "derivation" in obj:
            dependencies = _parse_dependencies(
                obj.get("derivation"), item_path, "derivation", set(content)
            )
        updates.append(SourceUpdateSpec(at_ms, content_path, etag, body, ttl, dependencies))

    working_content = dict(content)
    for index, update in enumerate(updates):
        previous = working_content[update.path]
        proposed = ContentEntry(
            path=previous.path,
            etag=update.etag if update.etag is not None else previous.etag,
            body=update.body if update.body is not None else previous.body,
            ttl_seconds=update.ttl_seconds if update.ttl_seconds is not None else previous.ttl_seconds,
            dependencies=(
                update.dependencies if update.dependencies is not None else previous.dependencies
            ),
        )
        trial = dict(working_content)
        trial[update.path] = proposed
        cycle = _cycle_for(trial)
        if cycle:
            raise PipelineError(
                "E_CYCLE",
                "source update history would create a derivation cycle",
                path=f"source_updates[{index}].derivation",
                field="inputs",
                details={"cycle": cycle, "spec_rejected": True, "at_ms": update.at_ms},
            )
        working_content = trial

    session = SessionSpec(session_id, int(grid_power), batch_size, source_port, host)
    return Spec(
        raw=root,
        session=session,
        content=content,
        nodes=nodes,
        requests=tuple(requests),
        source_updates=tuple(updates),
    )

