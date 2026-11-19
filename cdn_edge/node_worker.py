from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .errors import PipelineError, e_spec
from .netproto import JsonLineServer, atomic_write_json, send_json_line

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT PRIMARY KEY,
    key_json TEXT NOT NULL,
    path TEXT NOT NULL,
    query TEXT NOT NULL,
    etag TEXT NOT NULL,
    body_b64 TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    written_at_ms INTEGER NOT NULL,
    written_mono_ms INTEGER NOT NULL,
    entry_version INTEGER NOT NULL,
    last_access_seq INTEGER NOT NULL,
    inserted_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS request_log (
    seq INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL,
    path TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    hit_source TEXT NOT NULL,
    etag TEXT NOT NULL,
    entry_version INTEGER NOT NULL,
    remaining_ttl_ms INTEGER NOT NULL,
    stale INTEGER NOT NULL,
    error_code TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_mono_ms INTEGER NOT NULL
);
"""

COUNTER_NAMES = (
    "requests",
    "cache_hit",
    "expired_revalidated",
    "stale_hit",
    "miss_origin",
)


def _cache_key(fields: list[str], values: dict[str, str]) -> str:
    return json.dumps([[field, values[field]] for field in fields], ensure_ascii=False, separators=(",", ":"))


def _config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class NodeApp:
    def __init__(
        self,
        *,
        node_id: str,
        source_host: str,
        source_port: int,
        cache_key_fields: list[str],
        capacity: int,
        eviction: str,
        state_db: str,
        monotonic_base_ms: int | None = None,
    ) -> None:
        self.node_id = node_id
        self.source_host = source_host
        self.source_port = source_port
        self.cache_key_fields = cache_key_fields
        self.capacity = capacity
        self.eviction = eviction
        self.state_db = state_db
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(state_db, check_same_thread=False, timeout=15.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;"
        )
        self.conn.executescript(SCHEMA)
        self.config = {
            "node_id": node_id,
            "source_host": source_host,
            "source_port": source_port,
            "cache_key_fields": cache_key_fields,
            "capacity": capacity,
            "eviction": eviction,
        }
        self.config_digest = _config_digest(self.config)
        self._init_metadata(monotonic_base_ms)
        self.base_monotonic_ms = int(self._meta("monotonic_base_ms"))

    def _meta(self, key: str) -> str:
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            raise PipelineError(
                "E_STATE", "node ledger metadata is missing", path=self.state_db, field=key
            )
        return str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _init_metadata(self, monotonic_base_ms: int | None) -> None:
        row = self.conn.execute("SELECT value FROM metadata WHERE key='node_id'").fetchone()
        if row is not None:
            existing_digest = self._meta("config_digest")
            if existing_digest != self.config_digest:
                raise PipelineError(
                    "E_STATE",
                    "node ledger configuration does not match the restarted process",
                    path=self.state_db,
                    field="config_digest",
                    details={"locked_config": json.loads(self._meta("config")), "requested_config": self.config},
                )
            return
        base = monotonic_base_ms if monotonic_base_ms is not None else int(time.monotonic_ns() // 1_000_000)
        with self.conn:
            self._set_meta("node_id", self.node_id)
            self._set_meta("config", json.dumps(self.config, sort_keys=True, separators=(",", ":")))
            self._set_meta("config_digest", self.config_digest)
            self._set_meta("monotonic_base_ms", str(base))
            for counter in COUNTER_NAMES:
                self.conn.execute(
                    "INSERT OR IGNORE INTO counters(name,value) VALUES(?,0)", (counter,)
                )
            self.conn.execute(
                "INSERT INTO events(event,detail_json,created_mono_ms) VALUES(?,?,?)",
                ("created", "{}", base),
            )

    def _counter(self, name: str) -> int:
        row = self.conn.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()
        return int(row["value"]) if row is not None else 0

    def _increment_counter(self, name: str, delta: int = 1) -> None:
        self.conn.execute(
            "INSERT INTO counters(name,value) VALUES(?,?) "
            "ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
            (name, delta),
        )

    def _fetch_source(self, path: str, at_ms: int) -> dict[str, Any]:
        try:
            response = send_json_line(
                self.source_host,
                self.source_port,
                {"op": "get", "path": path, "at_ms": at_ms},
                timeout=10.0,
            )
        except PipelineError as exc:
            raise PipelineError(
                "E_ORIGIN",
                "origin fetch failed",
                path=f"origin:{path}",
                field="socket",
                details={"reason": exc.message, **exc.details},
            ) from exc
        if not response.get("ok"):
            raise PipelineError(
                "E_ORIGIN",
                "origin rejected fetch",
                path=f"origin:{path}",
                field="response",
                details={"response": response},
            )
        return response

    def _row_to_entry(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "cache_key": row["cache_key"],
            "key_values": dict(json.loads(row["key_json"])),
            "path": row["path"],
            "query": row["query"],
            "etag": row["etag"],
            "body_b64": row["body_b64"],
            "ttl_seconds": int(row["ttl_seconds"]),
            "written_at_ms": int(row["written_at_ms"]),
            "written_mono_ms": int(row["written_mono_ms"]),
            "entry_version": int(row["entry_version"]),
            "last_access_seq": int(row["last_access_seq"]),
            "inserted_seq": int(row["inserted_seq"]),
        }

    def _lookup(self, cache_key: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM cache_entries WHERE cache_key=?", (cache_key,)
        ).fetchone()

    def _count_entries(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM cache_entries").fetchone()["n"])

    def _evict_if_needed(self) -> list[str]:
        evicted: list[str] = []
        while self._count_entries() > self.capacity:
            if self.eviction == "LRU":
                row = self.conn.execute(
                    "SELECT cache_key FROM cache_entries "
                    "ORDER BY last_access_seq ASC, inserted_seq ASC, cache_key ASC LIMIT 1"
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT cache_key FROM cache_entries "
                    "ORDER BY inserted_seq ASC, cache_key ASC LIMIT 1"
                ).fetchone()
            if row is None:
                break
            key = str(row["cache_key"])
            self.conn.execute("DELETE FROM cache_entries WHERE cache_key=?", (key,))
            evicted.append(key)
        return evicted

    def _write_request_log(
        self,
        *,
        seq: int,
        request_id: str,
        path: str,
        cache_key: str,
        hit_source: str,
        etag: str,
        entry_version: int,
        remaining_ttl_ms: int,
        stale: bool,
        error_code: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO request_log("
            "seq,request_id,path,cache_key,hit_source,etag,entry_version,"
            "remaining_ttl_ms,stale,error_code,payload_json"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                seq,
                request_id,
                path,
                cache_key,
                hit_source,
                etag,
                entry_version,
                remaining_ttl_ms,
                1 if stale else 0,
                error_code,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            ),
        )

    def _response_from_entry(
        self,
        entry: dict[str, Any],
        *,
        hit_source: str,
        at_ms: int,
        stale: bool = False,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        remaining = 0 if stale else max(
            0, entry["ttl_seconds"] * 1000 - max(0, at_ms - entry["written_at_ms"])
        )
        return {
            "ok": True,
            "service": "edge",
            "node_id": self.node_id,
            "cache_key": entry["cache_key"],
            "path": entry["path"],
            "query": entry["query"],
            "hit_source": hit_source,
            "etag": entry["etag"],
            "body_b64": entry["body_b64"],
            "ttl_seconds": entry["ttl_seconds"],
            "remaining_ttl_ms": remaining,
            "entry_version": entry["entry_version"],
            "stale": stale,
            "error_code": error_code,
            "written_at_ms": entry["written_at_ms"],
            "written_mono_ms": entry["written_mono_ms"],
            "monotonic_base_ms": self.base_monotonic_ms,
        }

    def _fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        seq = payload.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise PipelineError("E_PROTOCOL", "fetch requires non-negative seq", path="seq", field="seq")
        request_id = payload.get("request_id")
        path = payload.get("path")
        at_ms = payload.get("at_ms")
        if not isinstance(request_id, str) or not request_id:
            raise PipelineError("E_PROTOCOL", "fetch requires request_id", path="request_id", field="request_id")
        if not isinstance(path, str) or not path:
            raise PipelineError("E_PROTOCOL", "fetch requires path", path="path", field="path")
        if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
            raise PipelineError("E_PROTOCOL", "fetch requires non-negative at_ms", path="at_ms", field="at_ms")
        key_values = payload.get("key")
        if not isinstance(key_values, dict):
            raise PipelineError("E_PROTOCOL", "fetch requires key object", path="key", field="key")
        normalized_key: dict[str, str] = {}
        for field in self.cache_key_fields:
            value = key_values.get(field)
            if not isinstance(value, str):
                raise PipelineError(
                    "E_PROTOCOL",
                    "fetch is missing a node cache-key field",
                    path="key",
                    field=field,
                )
            normalized_key[field] = value
        cache_key = _cache_key(self.cache_key_fields, normalized_key)
        allow_stale = bool(payload.get("allow_stale", False))

        with self.lock:
            self._increment_counter("requests")
            row = self._lookup(cache_key)
            if row is not None:
                entry = self._row_to_entry(row)
                elapsed = max(0, at_ms - entry["written_at_ms"])
                remaining = entry["ttl_seconds"] * 1000 - elapsed
                if remaining > 0:
                    self.conn.execute(
                        "UPDATE cache_entries SET last_access_seq=? WHERE cache_key=?",
                        (seq, cache_key),
                    )
                    self._increment_counter("cache_hit")
                    response = self._response_from_entry(entry, hit_source="cache_hit", at_ms=at_ms)
                    self._write_request_log(
                        seq=seq,
                        request_id=request_id,
                        path=path,
                        cache_key=cache_key,
                        hit_source="cache_hit",
                        etag=response["etag"],
                        entry_version=response["entry_version"],
                        remaining_ttl_ms=response["remaining_ttl_ms"],
                        stale=False,
                        error_code=None,
                        payload=payload,
                    )
                    self._commit()
                    return response
                try:
                    origin = self._fetch_source(path, at_ms)
                except PipelineError as exc:
                    if allow_stale:
                        self._increment_counter("stale_hit")
                        response = self._response_from_entry(
                            entry,
                            hit_source="stale_hit",
                            at_ms=at_ms,
                            stale=True,
                            error_code="E_STALE",
                        )
                        response["origin_error"] = exc.to_dict()
                        self._write_request_log(
                            seq=seq,
                            request_id=request_id,
                            path=path,
                            cache_key=cache_key,
                            hit_source="stale_hit",
                            etag=response["etag"],
                            entry_version=response["entry_version"],
                            remaining_ttl_ms=0,
                            stale=True,
                            error_code="E_STALE",
                            payload=payload,
                        )
                        self._commit()
                        return response
                    raise
                new_version = entry["entry_version"] + 1
                new_written_mono = self.base_monotonic_ms + at_ms
                self.conn.execute(
                    "UPDATE cache_entries SET etag=?, body_b64=?, ttl_seconds=?, written_at_ms=?, "
                    "written_mono_ms=?, entry_version=?, last_access_seq=? WHERE cache_key=?",
                    (
                        origin["etag"],
                        origin["body_b64"],
                        int(origin["ttl_seconds"]),
                        at_ms,
                        new_written_mono,
                        new_version,
                        seq,
                        cache_key,
                    ),
                )
                refreshed = {
                    **entry,
                    "etag": origin["etag"],
                    "body_b64": origin["body_b64"],
                    "ttl_seconds": int(origin["ttl_seconds"]),
                    "written_at_ms": at_ms,
                    "written_mono_ms": new_written_mono,
                    "entry_version": new_version,
                    "last_access_seq": seq,
                }
                self._increment_counter("expired_revalidated")
                response = self._response_from_entry(
                    refreshed, hit_source="expired_revalidated", at_ms=at_ms
                )
                self._write_request_log(
                    seq=seq,
                    request_id=request_id,
                    path=path,
                    cache_key=cache_key,
                    hit_source="expired_revalidated",
                    etag=response["etag"],
                    entry_version=response["entry_version"],
                    remaining_ttl_ms=response["remaining_ttl_ms"],
                    stale=False,
                    error_code=None,
                    payload=payload,
                )
                self._commit()
                return response

            origin = self._fetch_source(path, at_ms)
            body_b64 = str(origin["body_b64"])
            entry = {
                "cache_key": cache_key,
                "key_values": normalized_key,
                "path": path,
                "query": normalized_key.get("query", ""),
                "etag": str(origin["etag"]),
                "body_b64": body_b64,
                "ttl_seconds": int(origin["ttl_seconds"]),
                "written_at_ms": at_ms,
                "written_mono_ms": self.base_monotonic_ms + at_ms,
                "entry_version": 1,
                "last_access_seq": seq,
                "inserted_seq": seq,
            }
            self.conn.execute(
                "INSERT INTO cache_entries("
                "cache_key,key_json,path,query,etag,body_b64,ttl_seconds,"
                "written_at_ms,written_mono_ms,entry_version,last_access_seq,inserted_seq"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cache_key,
                    json.dumps(normalized_key, sort_keys=True, separators=(",", ":")),
                    path,
                    entry["query"],
                    entry["etag"],
                    body_b64,
                    entry["ttl_seconds"],
                    entry["written_at_ms"],
                    entry["written_mono_ms"],
                    entry["entry_version"],
                    entry["last_access_seq"],
                    entry["inserted_seq"],
                ),
            )
            evicted = self._evict_if_needed()
            self._increment_counter("miss_origin")
            response = self._response_from_entry(entry, hit_source="miss_origin", at_ms=at_ms)
            response["evicted"] = evicted
            self._write_request_log(
                seq=seq,
                request_id=request_id,
                path=path,
                cache_key=cache_key,
                hit_source="miss_origin",
                etag=response["etag"],
                entry_version=response["entry_version"],
                remaining_ttl_ms=response["remaining_ttl_ms"],
                stale=False,
                error_code=None,
                payload=payload,
            )
            self._commit()
            return response

    def _commit(self) -> None:
        self.conn.commit()

    def _dump(self) -> dict[str, Any]:
        entries = [
            self._row_to_entry(row)
            for row in self.conn.execute(
                "SELECT * FROM cache_entries ORDER BY cache_key ASC"
            ).fetchall()
        ]
        counters = {
            str(row["name"]): int(row["value"])
            for row in self.conn.execute("SELECT name,value FROM counters ORDER BY name").fetchall()
        }
        requests = [
            {
                "seq": int(row["seq"]),
                "request_id": row["request_id"],
                "path": row["path"],
                "cache_key": row["cache_key"],
                "hit_source": row["hit_source"],
                "etag": row["etag"],
                "entry_version": int(row["entry_version"]),
                "remaining_ttl_ms": int(row["remaining_ttl_ms"]),
                "stale": bool(row["stale"]),
                "error_code": row["error_code"],
            }
            for row in self.conn.execute(
                "SELECT * FROM request_log ORDER BY seq ASC"
            ).fetchall()
        ]
        return {
            "ok": True,
            "service": "edge",
            "node_id": self.node_id,
            "config": self.config,
            "monotonic_base_ms": self.base_monotonic_ms,
            "capacity": self.capacity,
            "eviction": self.eviction,
            "cache_key_fields": self.cache_key_fields,
            "entries": entries,
            "counters": counters,
            "request_log": requests,
        }

    def _restore(self, payload: dict[str, Any]) -> dict[str, Any]:
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise PipelineError("E_PROTOCOL", "restore requires entries array", path="entries", field="entries")
        replace = bool(payload.get("replace", True))
        counters = payload.get("counters", {})
        if not isinstance(counters, dict):
            raise PipelineError("E_PROTOCOL", "restore counters must be an object", path="counters", field="counters")
        base = payload.get("monotonic_base_ms")
        if base is not None and (isinstance(base, bool) or not isinstance(base, int) or base < 0):
            raise PipelineError("E_PROTOCOL", "restore monotonic_base_ms is invalid", path="monotonic_base_ms")
        with self.lock, self.conn:
            if replace:
                self.conn.execute("DELETE FROM cache_entries")
                self.conn.execute("DELETE FROM request_log")
                self.conn.execute("DELETE FROM counters")
            if base is not None:
                self._set_meta("monotonic_base_ms", str(base))
                self.base_monotonic_ms = int(base)
            for counter in COUNTER_NAMES:
                value = int(counters.get(counter, 0))
                self.conn.execute(
                    "INSERT INTO counters(name,value) VALUES(?,?) "
                    "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                    (counter, value),
                )
            for index, raw_entry in enumerate(entries):
                if not isinstance(raw_entry, dict):
                    raise PipelineError(
                        "E_PROTOCOL", "restore entry must be an object", path=f"entries[{index}]", field="$"
                    )
                key_values = raw_entry.get("key_values")
                if not isinstance(key_values, dict):
                    raise PipelineError(
                        "E_PROTOCOL", "restore entry requires key_values", path=f"entries[{index}]", field="key_values"
                    )
                normalized = {field: str(key_values[field]) for field in self.cache_key_fields}
                cache_key = _cache_key(self.cache_key_fields, normalized)
                written_at = int(raw_entry["written_at_ms"])
                written_mono = self.base_monotonic_ms + written_at
                self.conn.execute(
                    "INSERT OR REPLACE INTO cache_entries("
                    "cache_key,key_json,path,query,etag,body_b64,ttl_seconds,"
                    "written_at_ms,written_mono_ms,entry_version,last_access_seq,inserted_seq"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cache_key,
                        json.dumps(normalized, sort_keys=True, separators=(",", ":")),
                        str(raw_entry["path"]),
                        normalized.get("query", ""),
                        str(raw_entry["etag"]),
                        str(raw_entry["body_b64"]),
                        int(raw_entry["ttl_seconds"]),
                        written_at,
                        written_mono,
                        int(raw_entry.get("entry_version", 1)),
                        int(raw_entry.get("last_access_seq", 0)),
                        int(raw_entry.get("inserted_seq", 0)),
                    ),
                )
            self._evict_if_needed()
            self.conn.execute(
                "INSERT INTO events(event,detail_json,created_mono_ms) VALUES(?,?,?)",
                (
                    "restore_replace" if replace else "restore_merge",
                    json.dumps({"entry_count": len(entries), "whole_replacement": replace}, sort_keys=True),
                    self.base_monotonic_ms,
                ),
            )
        return {
            "ok": True,
            "status": "restored",
            "whole_replacement": replace,
            "entry_count": len(entries),
            "counters": counters,
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        op = payload.get("op")
        if op == "ping":
            return {
                "ok": True,
                "service": "edge",
                "node_id": self.node_id,
                "pid": os.getpid(),
                "monotonic_base_ms": self.base_monotonic_ms,
            }
        if op == "shutdown":
            threading.Thread(target=self.shutdown_server, daemon=True).start()
            return {"ok": True, "status": "stopping", "node_id": self.node_id}
        if op == "fetch":
            return self._fetch(payload)
        with self.lock:
            if op == "dump":
                return self._dump()
            if op == "report":
                return self._dump()
            if op == "restore":
                return self._restore(payload)
        raise PipelineError("E_PROTOCOL", "unsupported edge operation", path="op", field="op", details={"op": op})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback CDN edge node simulator")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--source-port", type=int, required=True)
    parser.add_argument("--cache-key-fields-json", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--eviction", choices=("LRU", "FIFO"), required=True)
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--error-file", required=True)
    parser.add_argument("--monotonic-base-ms", type=int)
    args = parser.parse_args(argv)

    def fail(error: PipelineError) -> int:
        try:
            atomic_write_json(args.error_file, error.to_dict())
        except OSError:
            pass
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return error.exit_code

    try:
        if args.port < 1024 or args.port > 65535:
            raise PipelineError("E_PORT", "edge port must be in [1024, 65535]", path="edge", field="port")
        try:
            fields = json.loads(args.cache_key_fields_json)
        except json.JSONDecodeError as exc:
            raise e_spec("cache-key fields JSON is invalid", "edge", "cache_key_fields") from exc
        if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
            raise e_spec("cache-key fields must be a JSON string array", "edge", "cache_key_fields")
        if len(fields) != len(set(fields)):
            raise e_spec("cache-key fields must be unique", "edge", "cache_key_fields")
        Path(args.state_db).parent.mkdir(parents=True, exist_ok=True)
        app = NodeApp(
            node_id=args.node_id,
            source_host=args.source_host,
            source_port=args.source_port,
            cache_key_fields=fields,
            capacity=args.capacity,
            eviction=args.eviction,
            state_db=args.state_db,
            monotonic_base_ms=args.monotonic_base_ms,
        )
        try:
            server = JsonLineServer((args.host, args.port), app)
        except OSError as exc:
            app.conn.close()
            raise PipelineError(
                "E_PORT",
                "edge port is occupied or unavailable",
                path=f"node:{args.node_id}",
                field="port",
                details={"host": args.host, "port": args.port, "reason": str(exc)},
            ) from exc
        app.shutdown_server = server.shutdown
        atomic_write_json(
            args.ready_file,
            {
                "service": "edge",
                "node_id": args.node_id,
                "pid": os.getpid(),
                "host": args.host,
                "port": args.port,
                "state_db": args.state_db,
                "monotonic_base_ms": app.base_monotonic_ms,
            },
        )
        server.serve_forever(poll_interval=0.05)
        server.server_close()
        app.conn.close()
        return 0
    except PipelineError as exc:
        return fail(exc)
    except Exception as exc:
        return fail(
            PipelineError(
                "E_PROTOCOL",
                "edge worker failed",
                path="edge",
                field="worker",
                details={"exception_type": type(exc).__name__, "reason": str(exc)},
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
