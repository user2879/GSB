from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_digest, sha256_text
from .errors import DiagnosticError, ErrorCode, spec_error


LOOPBACK_HOST = "127.0.0.1"
SUPPORTED_FORMAT_VERSION = 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fail_db(db_id: str | None, field: str, message: str) -> DiagnosticError:
    return spec_error(message, db_id=db_id, field=field)


def _fail_table(
    db_id: str | None,
    key_column: str | None,
    field: str,
    message: str,
) -> DiagnosticError:
    return spec_error(message, db_id=db_id, key_column=key_column, field=field)


@dataclass(frozen=True, slots=True)
class Segment:
    shard_id: str
    start: int
    end: int

    def contains(self, key: int) -> bool:
        return self.start <= key < self.end

    def as_dict(self) -> dict[str, Any]:
        return {"shard_id": self.shard_id, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class DatabaseSpec:
    id: str
    host: str
    port: int
    lease_limit: int
    concurrency: int
    seed: int
    route_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "lease_limit": self.lease_limit,
            "concurrency": self.concurrency,
            "seed": self.seed,
            "route_version": self.route_version,
        }


@dataclass(frozen=True, slots=True)
class TableRule:
    name: str
    key_column: str
    route_version: int
    segments: tuple[Segment, ...]

    @property
    def boundaries(self) -> tuple[int, ...]:
        return (self.segments[0].start, *(segment.end for segment in self.segments))

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(segment.start for segment in self.segments)

    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(segment.shard_id for segment in self.segments)

    @property
    def key_min(self) -> int:
        return self.segments[0].start

    @property
    def key_max(self) -> int:
        return self.segments[-1].end

    def segment_for_key(self, key: int) -> Segment | None:
        if not self.segments or key < self.key_min or key >= self.key_max:
            return None
        index = bisect.bisect_right(self.starts, key) - 1
        segment = self.segments[index]
        return segment if segment.contains(key) else None

    def shard_for_key(self, key: int) -> str:
        segment = self.segment_for_key(key)
        if segment is None:
            raise DiagnosticError(
                ErrorCode.ROUTE.value,
                f"route key {key} is outside table {self.name!r} version {self.route_version}",
                key_column=self.key_column,
                field="route_key",
                details={"table": self.name, "route_version": self.route_version},
            )
        return segment.shard_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key_column": self.key_column,
            "route_version": self.route_version,
            "boundaries": [segment.as_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    kind: str
    database: str
    table: str
    route_key: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "database": self.database,
            "table": self.table,
            "route_key": self.route_key,
        }


@dataclass(frozen=True, slots=True)
class Spec:
    format_version: int
    global_lease_limit: int
    databases: tuple[DatabaseSpec, ...]
    tables: tuple[TableRule, ...]
    events: tuple[Event, ...]
    source_digest: str = ""
    source_path: str | None = None

    def database(self, db_id: str) -> DatabaseSpec:
        for database in self.databases:
            if database.id == db_id:
                return database
        raise _fail_db(db_id, "database", f"unknown database {db_id!r}")

    def rule(self, table: str, route_version: int) -> TableRule:
        for rule in self.tables:
            if rule.name == table and rule.route_version == route_version:
                return rule
        raise _fail_table(
            None,
            None,
            "table",
            f"no routing rule for table {table!r} version {route_version}",
        )

    def maybe_rule(self, table: str, route_version: int) -> TableRule | None:
        for rule in self.tables:
            if rule.name == table and rule.route_version == route_version:
                return rule
        return None

    def versions(self, table: str) -> tuple[int, ...]:
        return tuple(sorted(rule.route_version for rule in self.tables if rule.name == table))

    def latest_rule(self, table: str) -> TableRule:
        versions = self.versions(table)
        if not versions:
            raise _fail_table(None, None, "table", f"unknown table {table!r}")
        return self.rule(table, versions[-1])

    def previous_rule(self, table: str, route_version: int) -> TableRule | None:
        candidates = [version for version in self.versions(table) if version < route_version]
        return self.rule(table, candidates[-1]) if candidates else None

    def event_rule(self, event: Event) -> TableRule:
        database = self.database(event.database)
        rule = self.maybe_rule(event.table, database.route_version)
        if rule is None:
            raise _fail_table(
                database.id,
                None,
                "route_version",
                f"database {database.id!r} is bound to route version {database.route_version}, "
                f"but table {event.table!r} has no rule at that version",
            )
        return rule

    @property
    def databases_by_id(self) -> dict[str, DatabaseSpec]:
        return {database.id: database for database in self.databases}

    @property
    def routes_by_key(self) -> dict[tuple[str, int], TableRule]:
        return {(rule.name, rule.route_version): rule for rule in self.tables}

    def table_versions(self) -> dict[str, tuple[int, ...]]:
        names = sorted({rule.name for rule in self.tables})
        return {name: self.versions(name) for name in names}

    def routing_snapshot(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "global_lease_limit": self.global_lease_limit,
            "databases": [database.as_dict() for database in self.databases],
            "tables": [rule.as_dict() for rule in self.tables],
        }

    def event_stream_snapshot(self) -> dict[str, Any]:
        return {"events": [event.as_dict() for event in self.events]}

    def spec_digest(self) -> str:
        return sha256_digest(self.routing_snapshot())

    def event_digest(self) -> str:
        return sha256_digest(self.event_stream_snapshot())

    def snapshot(self, *, generation: int) -> dict[str, Any]:
        return {
            "snapshot_format_version": 1,
            "generation": generation,
            "spec_digest": self.spec_digest(),
            "event_digest": self.event_digest(),
            "routing": self.routing_snapshot(),
            "events": self.event_stream_snapshot()["events"],
        }


def _parse_boundaries(
    raw: Any,
    *,
    table_name: str,
    key_column: str | None,
    route_version: int,
) -> tuple[Segment, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise _fail_table(
            None,
            key_column,
            "boundaries",
            f"table {table_name!r} version {route_version} boundaries must be a sequence of at least two integers",
        )

    if all(_is_int(item) for item in raw):
        endpoints = tuple(raw)
        if any(right <= left for left, right in zip(endpoints, endpoints[1:])):
            raise _fail_table(
                None,
                key_column,
                "boundaries",
                f"table {table_name!r} version {route_version} boundaries must be strictly increasing",
            )
        return tuple(
            Segment(f"{table_name}:s{index}", start, end)
            for index, (start, end) in enumerate(zip(endpoints, endpoints[1:]))
        )

    segments: list[Segment] = []
    previous_end: int | None = None
    for index, item in enumerate(raw):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
            shard_id = f"{table_name}:s{index}"
        elif isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            shard_id = item.get("shard_id", f"{table_name}:s{index}")
        else:
            raise _fail_table(
                None,
                key_column,
                "boundaries",
                f"boundary[{index}] must be [start, end] or an object with start/end",
            )
        if not _is_int(start) or not _is_int(end):
            raise _fail_table(None, key_column, "boundaries", f"boundary[{index}] start/end must be integers")
        if end <= start:
            raise _fail_table(
                None,
                key_column,
                "boundaries",
                f"boundary[{index}] end must be greater than start",
            )
        if previous_end is not None and start != previous_end:
            relation = "overlaps" if start < previous_end else "has a gap before"
            raise _fail_table(
                None,
                key_column,
                "boundaries",
                f"boundary[{index}] {relation} boundary[{index - 1}]; ranges must be adjacent",
            )
        if previous_end is not None and start <= previous_end:
            raise _fail_table(None, key_column, "boundaries", f"boundary[{index}] is not strictly increasing")
        if not isinstance(shard_id, str) or not shard_id:
            raise _fail_table(
                None,
                key_column,
                "boundaries",
                f"boundary[{index}] shard_id must be a non-empty string",
            )
        segments.append(Segment(shard_id, start, end))
        previous_end = end

    shard_ids = [segment.shard_id for segment in segments]
    if len(shard_ids) != len(set(shard_ids)):
        raise _fail_table(
            None,
            key_column,
            "boundaries",
            f"table {table_name!r} version {route_version} shard ids must be unique",
        )
    return tuple(segments)


def _load_databases(raw: Any) -> tuple[DatabaseSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise _fail_db(None, "databases", "databases must be a non-empty sequence")
    result: list[DatabaseSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, item in enumerate(raw):
        placeholder = f"@database[{index}]"
        if not isinstance(item, dict):
            raise _fail_db(placeholder, "database", f"database[{index}] must be an object")
        db_id = item.get("id")
        if not isinstance(db_id, str) or not db_id:
            raise _fail_db(placeholder, "id", f"database[{index}].id must be a non-empty string")
        if db_id in seen_ids:
            raise _fail_db(db_id, "id", f"database id {db_id!r} is duplicated")
        seen_ids.add(db_id)
        host = item.get("host", LOOPBACK_HOST)
        if host != LOOPBACK_HOST:
            raise _fail_db(db_id, "host", f"database {db_id!r} host must be {LOOPBACK_HOST}")
        port = item.get("port")
        if not _is_int(port) or not 1 <= port <= 65535:
            raise _fail_db(db_id, "port", f"database {db_id!r} port must be an integer in 1..65535")
        if port in seen_ports:
            raise _fail_db(db_id, "port", f"database {db_id!r} port {port} is already used")
        seen_ports.add(port)
        lease_limit = item.get("lease_limit")
        if not _is_int(lease_limit) or lease_limit <= 0:
            raise _fail_db(db_id, "lease_limit", f"database {db_id!r} lease_limit must be a positive integer")
        concurrency = item.get("concurrency")
        if not _is_int(concurrency) or concurrency <= 0:
            raise _fail_db(db_id, "concurrency", f"database {db_id!r} concurrency must be a positive integer")
        seed = item.get("seed")
        if not _is_int(seed) or seed < 0:
            raise _fail_db(db_id, "seed", f"database {db_id!r} seed must be a non-negative integer")
        route_version = item.get("route_version")
        if not _is_int(route_version) or route_version <= 0:
            raise _fail_db(
                db_id,
                "route_version",
                f"database {db_id!r} route_version must be a positive integer",
            )
        result.append(DatabaseSpec(db_id, host, port, lease_limit, concurrency, seed, route_version))
    return tuple(result)


def _load_tables(raw: Any, databases: tuple[DatabaseSpec, ...]) -> tuple[TableRule, ...]:
    if not isinstance(raw, list) or not raw:
        raise _fail_table(None, None, "tables", "tables must be a non-empty sequence")
    result: list[TableRule] = []
    seen: set[tuple[str, int]] = set()
    table_key_columns: dict[str, str] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _fail_table(None, None, "tables", f"table[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise _fail_table(None, None, "name", f"table[{index}].name must be a non-empty string")
        key_column = item.get("key_column")
        if not isinstance(key_column, str) or not key_column:
            raise _fail_table(None, None, "key_column", f"table {name!r} key_column must be a non-empty string")
        if name in table_key_columns and table_key_columns[name] != key_column:
            raise _fail_table(
                None,
                key_column,
                "key_column",
                f"table {name!r} uses inconsistent key columns across route versions",
            )
        table_key_columns[name] = key_column
        route_version = item.get("route_version")
        if not _is_int(route_version) or route_version <= 0:
            raise _fail_table(
                None,
                key_column,
                "route_version",
                f"table {name!r} route_version must be a positive integer",
            )
        if (name, route_version) in seen:
            raise _fail_table(
                None,
                key_column,
                "route_version",
                f"table {name!r} route version {route_version} is duplicated",
            )
        seen.add((name, route_version))
        segments = _parse_boundaries(
            item.get("boundaries"),
            table_name=name,
            key_column=key_column,
            route_version=route_version,
        )
        result.append(TableRule(name, key_column, route_version, segments))

    for database in databases:
        versions = {(rule.name, rule.route_version) for rule in result}
        tables = sorted({rule.name for rule in result})
        missing = [table for table in tables if (table, database.route_version) not in versions]
        if missing:
            raise _fail_table(
                database.id,
                None,
                "route_version",
                f"database {database.id!r} version {database.route_version} has no routing rule for tables: "
                + ", ".join(missing),
            )
    return tuple(result)


def _load_events(raw: Any, databases: tuple[DatabaseSpec, ...], tables: tuple[TableRule, ...]) -> tuple[Event, ...]:
    if not isinstance(raw, list):
        raise _fail_db(None, "events", "events must be a sequence")
    result: list[Event] = []
    previous_seq: int | None = None
    seen_seq: set[int] = set()
    db_by_id = {database.id: database for database in databases}
    rules = {(rule.name, rule.route_version): rule for rule in tables}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise spec_error(f"event[{index}] must be an object", field="events")
        seq = item.get("seq")
        raw_database = item.get("database")
        context_db_id = raw_database if isinstance(raw_database, str) else None
        raw_table = item.get("table")
        context_rule = None
        if isinstance(raw_table, str) and isinstance(raw_database, str) and raw_database in db_by_id:
            context_rule = rules.get((raw_table, db_by_id[raw_database].route_version))
        context_key_column = context_rule.key_column if context_rule is not None else None
        if not _is_int(seq) or seq < 0:
            raise spec_error(
                f"event[{index}].seq must be a non-negative integer",
                db_id=context_db_id,
                key_column=context_key_column,
                field="seq",
            )
        if seq in seen_seq:
            raise spec_error(
                f"event seq {seq} is duplicated",
                db_id=context_db_id,
                key_column=context_key_column,
                field="seq",
                details={"event_seq": seq},
            )
        if previous_seq is not None and seq <= previous_seq:
            raise spec_error(
                f"event seq {seq} is not strictly increasing after {previous_seq}",
                db_id=context_db_id,
                key_column=context_key_column,
                field="seq",
                details={"event_seq": seq, "previous_seq": previous_seq},
            )
        seen_seq.add(seq)
        previous_seq = seq
        kind = item.get("kind")
        if not isinstance(kind, str) or kind.lower() not in {"write", "query"}:
            raise spec_error(
                f"event seq {seq} kind must be 'write' or 'query'",
                field="kind",
                details={"event_seq": seq},
            )
        database = item.get("database")
        if not isinstance(database, str) or database not in db_by_id:
            raise spec_error(
                f"event seq {seq} references unknown database {database!r}",
                db_id=database if isinstance(database, str) else None,
                field="database",
                details={"event_seq": seq},
            )
        table = item.get("table")
        if not isinstance(table, str):
            raise spec_error(
                f"event seq {seq} table must be a non-empty string",
                db_id=database,
                field="table",
                details={"event_seq": seq},
            )
        rule = rules.get((table, db_by_id[database].route_version))
        if rule is None:
            raise spec_error(
                f"event seq {seq} table {table!r} has no rule at database {database!r} "
                f"route version {db_by_id[database].route_version}",
                db_id=database,
                key_column=None,
                field="table",
            )
        route_key = item.get("route_key")
        if not _is_int(route_key):
            raise spec_error(
                f"event seq {seq} route_key must be an integer",
                db_id=database,
                key_column=rule.key_column,
                field="route_key",
                details={"event_seq": seq},
            )
        if rule.segment_for_key(route_key) is None:
            raise spec_error(
                f"event seq {seq} route_key {route_key} is outside the bound rule range for table {table!r}",
                db_id=database,
                key_column=rule.key_column,
                field="route_key",
                details={"event_seq": seq},
            )
        result.append(Event(seq, kind.lower(), database, table, route_key))
    return tuple(result)


def validate_spec(raw: Any, *, source_path: str | None = None, raw_text: str | None = None) -> Spec:
    if not isinstance(raw, dict):
        raise spec_error("spec root must be an object", field="$")
    format_version = raw.get("format_version", SUPPORTED_FORMAT_VERSION)
    if not _is_int(format_version) or format_version != SUPPORTED_FORMAT_VERSION:
        raise spec_error(
            f"format_version must be the integer {SUPPORTED_FORMAT_VERSION}",
            field="format_version",
        )
    databases = _load_databases(raw.get("databases"))
    if raw.get("global_lease_limit") is None:
        global_lease_limit = sum(database.lease_limit for database in databases)
    elif not _is_int(raw["global_lease_limit"]) or raw["global_lease_limit"] <= 0:
        raise spec_error("global_lease_limit must be a positive integer", field="global_lease_limit")
    else:
        global_lease_limit = raw["global_lease_limit"]
    tables = _load_tables(raw.get("tables"), databases)
    events = _load_events(raw.get("events", []), databases, tables)
    source_digest = sha256_text(raw_text) if raw_text is not None else sha256_digest(raw)
    return Spec(
        format_version=format_version,
        global_lease_limit=global_lease_limit,
        databases=databases,
        tables=tables,
        events=events,
        source_digest=source_digest,
        source_path=source_path,
    )


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise spec_error(f"cannot read spec {path}: {error}", field="path") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise spec_error(
            f"invalid JSON in {path}: {error.msg}",
            field="json",
            details={"line": error.lineno, "column": error.colno},
        ) from error
    return validate_spec(raw, source_path=str(path), raw_text=text)


def dump_spec(spec_values: dict[str, Any]) -> str:
    return canonical_json(spec_values) + "\n"
