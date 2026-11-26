"""Strict JSON schema loading for local routing workspaces."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import sha256_digest
from .errors import RouteError, spec_error


LOWER_INCLUSIVE = "lower_inclusive_upper_exclusive"
LOWER_EXCLUSIVE = "lower_exclusive_upper_inclusive"
BOUNDARY_RULES = {
    LOWER_INCLUSIVE,
    LOWER_EXCLUSIVE,
    "left_closed_right_open": LOWER_INCLUSIVE,
    "[)": LOWER_INCLUSIVE,
    "lower_inclusive": LOWER_INCLUSIVE,
    "left_open_right_closed": LOWER_EXCLUSIVE,
    "(]": LOWER_EXCLUSIVE,
    "upper_inclusive": LOWER_EXCLUSIVE,
}

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class DatabaseSpec:
    id: str
    host: str
    port: int
    lease_limit: int
    concurrency_quota: int
    seed: int
    route_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "lease_limit": self.lease_limit,
            "concurrency_quota": self.concurrency_quota,
            "seed": self.seed,
            "route_version": self.route_version,
        }


@dataclass(frozen=True)
class TableSpec:
    id: str
    key_column: str
    boundaries: tuple[int, ...]
    route_version: int
    boundary_rule: str
    database_assignment: tuple[str, ...]

    @property
    def interval_count(self) -> int:
        return len(self.boundaries) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key_column": self.key_column,
            "boundaries": list(self.boundaries),
            "route_version": self.route_version,
            "boundary_rule": self.boundary_rule,
            "database_assignment": list(self.database_assignment),
        }


@dataclass(frozen=True)
class Limits:
    global_in_flight_leases: int = 0
    lease_duration_ms: int = 1000
    socket_read_timeout_ms: int = 1000
    handshake_timeout_ms: int = 2000
    queue_poll_ms: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_in_flight_leases": self.global_in_flight_leases,
            "lease_duration_ms": self.lease_duration_ms,
            "socket_read_timeout_ms": self.socket_read_timeout_ms,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "queue_poll_ms": self.queue_poll_ms,
        }


@dataclass(frozen=True)
class EventSpec:
    seq: int
    table: str
    key: int
    op: str

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "table": self.table, "key": self.key, "op": self.op}


@dataclass(frozen=True)
class RoutingSpec:
    format_version: int
    route_version: int
    scale_factor: int
    databases: tuple[DatabaseSpec, ...]
    tables: tuple[TableSpec, ...]
    limits: Limits
    events: tuple[EventSpec, ...]
    digest: str

    def database(self, database_id: str) -> DatabaseSpec:
        for database in self.databases:
            if database.id == database_id:
                return database
        raise spec_error("events", "referenced database does not exist", database_id=database_id)

    def table(self, table_id: str) -> TableSpec:
        for table in self.tables:
            if table.id == table_id:
                return table
        raise spec_error("events", "referenced table does not exist", table_id=table_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "route_version": self.route_version,
            "session": {"scale_factor": self.scale_factor},
            "limits": self.limits.to_dict(),
            "databases": [item.to_dict() for item in self.databases],
            "tables": [item.to_dict() for item in self.tables],
            "events": [item.to_dict() for item in self.events],
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if not _is_int(value):
        raise spec_error(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise spec_error(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise spec_error(path, f"must be <= {maximum}")
    return value


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise spec_error(path, "must be a non-empty string")
    return value


def _pick(mapping: Mapping[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _load_json(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise spec_error("$", "cannot read JSON file", file=str(path), error=type(exc).__name__) from exc
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise spec_error("$", "invalid JSON", error=str(exc).splitlines()[0]) from exc
    if not isinstance(value, dict):
        raise spec_error("$", "root must be a JSON object")
    return value


def _normalize_boundaries(value: Any, path: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise spec_error(path, "must be a non-empty boundary sequence")

    explicit_intervals = isinstance(value[0], (list, tuple, dict))
    if explicit_intervals:
        intervals: list[tuple[int, int]] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if isinstance(item, dict):
                start = _pick(item, ("start", "lower", "from"))
                end = _pick(item, ("end", "upper", "to"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                start, end = item
            else:
                raise spec_error(item_path, "interval must be [start, end] or {start, end}")
            start_i = _require_int(start, f"{item_path}.start", minimum=INT64_MIN, maximum=INT64_MAX)
            end_i = _require_int(end, f"{item_path}.end", minimum=INT64_MIN, maximum=INT64_MAX)
            if start_i >= end_i:
                raise spec_error(f"{item_path}.end", "interval end must be greater than start")
            if intervals and intervals[-1][1] != start_i:
                raise spec_error(
                    f"{item_path}.start",
                    "adjacent intervals must meet end-to-start",
                    previous_end=intervals[-1][1],
                    actual_start=start_i,
                )
            intervals.append((start_i, end_i))
        boundaries = [intervals[0][0]]
        boundaries.extend(end for _, end in intervals)
    else:
        boundaries = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            number = _require_int(item, item_path, minimum=INT64_MIN, maximum=INT64_MAX)
            if boundaries and number <= boundaries[-1]:
                raise spec_error(item_path, "boundaries must be strictly increasing", previous=boundaries[-1])
            boundaries.append(number)
        if len(boundaries) < 2:
            raise spec_error(path, "at least two boundaries are required")
    return boundaries


def _normalize_rule(value: Any, path: str) -> str:
    text = _require_text(value, path)
    canonical = BOUNDARY_RULES.get(text)
    if canonical is None:
        raise spec_error(path, "unsupported boundary ownership rule", allowed=sorted(set(BOUNDARY_RULES.values())))
    return canonical


def _route_interval(table: TableSpec, key: int) -> int | None:
    boundaries = table.boundaries
    if table.boundary_rule == LOWER_INCLUSIVE:
        if key < boundaries[0] or key >= boundaries[-1]:
            return None
        return bisect.bisect_right(boundaries, key) - 1
    if key <= boundaries[0] or key > boundaries[-1]:
        return None
    return bisect.bisect_left(boundaries, key) - 1


def load_spec(
    source: str | Path | Mapping[str, Any],
    *,
    require_events: bool = True,
) -> RoutingSpec:
    """Load and strictly validate a routing declaration.

    Validation walks canonical fields in a fixed order.  The first violation is
    raised as ``E_SPEC`` with a path such as ``databases[1].port``.
    """

    raw = _load_json(source)

    format_version = _require_int(
        _pick(raw, ("format_version", "schema_version"), 1),
        "format_version",
        minimum=1,
    )
    if format_version != 1:
        raise spec_error("format_version", "only format version 1 is supported", actual=format_version)

    session_raw = _pick(raw, ("session", "session_scaling"), {})
    if not isinstance(session_raw, dict):
        raise spec_error("session", "must be an object")
    scale_factor = _require_int(
        _pick(session_raw, ("scale_factor", "fixedpoint_scale"), _pick(raw, ("scale_factor",), 1000)),
        "session.scale_factor",
        minimum=1,
    )

    databases_raw = _pick(raw, ("databases", "physical_databases", "shards"), None)
    if not isinstance(databases_raw, list) or not databases_raw:
        raise spec_error("databases", "must be a non-empty array")

    tables_raw = _pick(raw, ("tables", "logical_tables"), None)
    if not isinstance(tables_raw, list) or not tables_raw:
        raise spec_error("tables", "must be a non-empty array")

    databases: list[DatabaseSpec] = []
    database_ids: set[str] = set()
    ports: set[int] = set()
    declared_versions: list[int] = []

    for index, item in enumerate(databases_raw):
        base = f"databases[{index}]"
        if not isinstance(item, dict):
            raise spec_error(base, "must be an object")
        database_id = _require_text(_pick(item, ("id", "database_id", "db_id")), f"{base}.id")
        if database_id in database_ids:
            raise spec_error(f"{base}.id", "database id must be unique", value=database_id)
        database_ids.add(database_id)
        host = _require_text(_pick(item, ("host", "ip"), "127.0.0.1"), f"{base}.host")
        if host != "127.0.0.1":
            raise spec_error(f"{base}.host", "loopback host must be the literal 127.0.0.1", actual=host)
        port = _require_int(_pick(item, ("port", "loopback_port")), f"{base}.port", minimum=1, maximum=65535)
        if port in ports:
            raise spec_error(f"{base}.port", "loopback port must be globally unique", value=port)
        ports.add(port)
        lease_limit = _require_int(
            _pick(item, ("lease_limit", "lease_cap", "lease_upper_limit")),
            f"{base}.lease_limit",
            minimum=0,
        )
        concurrency_quota = _require_int(
            _pick(item, ("concurrency_quota", "concurrency_limit", "concurrency")),
            f"{base}.concurrency_quota",
            minimum=0,
        )
        seed = _require_int(
            _pick(item, ("seed", "random_seed")),
            f"{base}.seed",
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        version = _require_int(
            _pick(item, ("route_version", "routing_version")),
            f"{base}.route_version",
            minimum=0,
        )
        declared_versions.append(version)
        databases.append(DatabaseSpec(database_id, host, port, lease_limit, concurrency_quota, seed, version))

    global_version_value = _pick(raw, ("route_version", "routing_version"), None)
    if global_version_value is not None:
        declared_versions.append(_require_int(global_version_value, "route_version", minimum=0))
    table_versions: list[int] = []
    tables: list[TableSpec] = []
    table_ids: set[str] = set()

    for index, item in enumerate(tables_raw):
        base = f"tables[{index}]"
        if not isinstance(item, dict):
            raise spec_error(base, "must be an object")
        table_id = _require_text(_pick(item, ("id", "table_id", "logical_table")), f"{base}.id")
        if table_id in table_ids:
            raise spec_error(f"{base}.id", "logical table id must be unique", value=table_id)
        table_ids.add(table_id)
        key_column = _require_text(
            _pick(item, ("key_column", "shard_key_column", "shard_key")),
            f"{base}.key_column",
        )
        boundaries = _normalize_boundaries(
            _pick(item, ("boundaries", "range_boundaries", "boundary_sequence")),
            f"{base}.boundaries",
        )
        rule = _normalize_rule(
            _pick(item, ("boundary_rule", "boundary_ownership"), LOWER_INCLUSIVE),
            f"{base}.boundary_rule",
        )
        version = _require_int(
            _pick(item, ("route_version", "routing_version")),
            f"{base}.route_version",
            minimum=0,
        )
        table_versions.append(version)
        declared_versions.append(version)

        assignment_raw = _pick(item, ("database_assignment", "placements", "interval_databases"), None)
        interval_count = len(boundaries) - 1
        if assignment_raw is None:
            ordered_ids = sorted(database_ids)
            assignment = tuple(ordered_ids[i % len(ordered_ids)] for i in range(interval_count))
        else:
            if not isinstance(assignment_raw, list):
                raise spec_error(f"{base}.database_assignment", "must be an array")
            if len(assignment_raw) != interval_count:
                raise spec_error(
                    f"{base}.database_assignment",
                    "must contain exactly one database id per interval",
                    expected=interval_count,
                    actual=len(assignment_raw),
                )
            assignment_items: list[str] = []
            for assignment_index, value in enumerate(assignment_raw):
                path = f"{base}.database_assignment[{assignment_index}]"
                database_id = _require_text(value, path)
                if database_id not in database_ids:
                    raise spec_error(path, "references an unknown database", value=database_id)
                assignment_items.append(database_id)
            assignment = tuple(assignment_items)
        tables.append(TableSpec(table_id, key_column, tuple(boundaries), version, rule, assignment))

    if not declared_versions:
        raise spec_error("route_version", "route version is required")
    if len(set(declared_versions)) != 1:
        raise spec_error(
            "route_version",
            "database and table route versions must agree",
            values=sorted(set(declared_versions)),
        )
    route_version = declared_versions[0]

    limits_raw = _pick(raw, ("limits", "session_limits"), {})
    if not isinstance(limits_raw, dict):
        raise spec_error("limits", "must be an object")
    default_global = sum(item.lease_limit for item in databases)
    global_limit = _require_int(
        _pick(
            limits_raw,
            ("global_in_flight_leases", "global_lease_limit", "global_in_flight_lease_limit"),
            _pick(raw, ("global_in_flight_leases", "global_lease_limit"), default_global),
        ),
        "limits.global_in_flight_leases",
        minimum=0,
    )
    lease_duration_ms = _require_int(
        _pick(limits_raw, ("lease_duration_ms",), _pick(raw, ("lease_duration_ms",), 1000)),
        "limits.lease_duration_ms",
        minimum=1,
    )
    read_timeout_ms = _require_int(
        _pick(limits_raw, ("socket_read_timeout_ms", "socket_timeout_ms"), 1000),
        "limits.socket_read_timeout_ms",
        minimum=1,
    )
    handshake_timeout_ms = _require_int(
        _pick(limits_raw, ("handshake_timeout_ms",), 2000),
        "limits.handshake_timeout_ms",
        minimum=1,
    )
    queue_poll_ms = _require_int(
        _pick(limits_raw, ("queue_poll_ms",), 10),
        "limits.queue_poll_ms",
        minimum=1,
    )
    limits = Limits(global_limit, lease_duration_ms, read_timeout_ms, handshake_timeout_ms, queue_poll_ms)

    if "events" not in raw:
        if require_events:
            raise spec_error("events", "must be an array")
        events_raw: Any = []
    else:
        events_raw = raw["events"]
    if not isinstance(events_raw, list):
        raise spec_error("events", "must be an array")

    events: list[EventSpec] = []
    previous_seq: int | None = None
    for index, item in enumerate(events_raw):
        base = f"events[{index}]"
        if not isinstance(item, dict):
            raise spec_error(base, "must be an object")
        seq = _require_int(
            _pick(item, ("seq", "arrival_seq", "arrival")),
            f"{base}.seq",
            minimum=0,
        )
        if previous_seq is not None and seq <= previous_seq:
            raise spec_error(f"{base}.seq", "event arrival sequence must be strictly increasing", previous=previous_seq)
        previous_seq = seq
        table_value = _pick(item, ("table", "table_id", "logical_table"), None)
        if table_value is None and len(tables) == 1:
            table_id = tables[0].id
        else:
            table_id = _require_text(table_value, f"{base}.table")
        table_by_id = next((table for table in tables if table.id == table_id), None)
        if table_by_id is None:
            raise spec_error(f"{base}.table", "references an unknown logical table", value=table_id)
        key_column = _pick(item, ("key_column", "shard_key_column"), None)
        if key_column is not None:
            key_column = _require_text(key_column, f"{base}.key_column")
            if key_column != table_by_id.key_column:
                raise spec_error(
                    f"{base}.key_column",
                    "does not match the logical table shard key column",
                    expected=table_by_id.key_column,
                    actual=key_column,
                )
        key = _require_int(
            _pick(item, ("key", "key_value", "value", "shard_key")),
            f"{base}.key",
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        op_value = _require_text(_pick(item, ("op", "operation", "type")), f"{base}.op").lower()
        if op_value == "read":
            op_value = "query"
        if op_value not in {"query", "write"}:
            raise spec_error(f"{base}.op", "must be 'query' or 'write'", actual=op_value)
        if _route_interval(table_by_id, key) is None:
            raise spec_error(
                f"{base}.key",
                "is not routable under the table boundary ownership rule",
                boundary_rule=table_by_id.boundary_rule,
                boundaries=list(table_by_id.boundaries),
            )
        events.append(EventSpec(seq, table_id, key, op_value))

    normalized = {
        "format_version": format_version,
        "route_version": route_version,
        "scale_factor": scale_factor,
        "limits": limits.to_dict(),
        "databases": [item.to_dict() for item in databases],
        "tables": [item.to_dict() for item in tables],
        "events": [item.to_dict() for item in events],
    }
    spec = RoutingSpec(
        format_version=format_version,
        route_version=route_version,
        scale_factor=scale_factor,
        databases=tuple(databases),
        tables=tuple(tables),
        limits=limits,
        events=tuple(events),
        digest=sha256_digest(normalized),
    )
    return spec


def load_previous_spec(source: str | Path | Mapping[str, Any]) -> RoutingSpec:
    return load_spec(source, require_events=False)


def ensure_spec_difference(current: RoutingSpec, previous: RoutingSpec) -> None:
    if current.route_version <= previous.route_version:
        raise spec_error(
            "route_version",
            "current route version must be newer than the previous route version",
            current=current.route_version,
            previous=previous.route_version,
        )


__all__ = [
    "BOUNDARY_RULES",
    "DatabaseSpec",
    "EventSpec",
    "INT64_MAX",
    "INT64_MIN",
    "LOWER_EXCLUSIVE",
    "LOWER_INCLUSIVE",
    "Limits",
    "RoutingSpec",
    "TableSpec",
    "ensure_spec_difference",
    "load_previous_spec",
    "load_spec",
]
