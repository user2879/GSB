"""High-level session orchestration over the ledger and engine."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import Event, SessionSpec, canonical_dumps, load_events
from .engine import SessionEngine
from .errors import (LAT_CODE, LATE_CODE, StructuredLogger,
                     ViewMismatchError)
from .latency import compute_stats, measure_ms
from .ledger import Ledger, SCHEMA_VERSION
from .snapshot import content_digest, read_snapshot, write_snapshot


@dataclass
class QueryResult:
    query_seq: int
    partition: str | None
    latency_ms: int
    budget_ms: int
    over_budget: bool
    injected_ms: int
    code: str | None
    view_bytes: str


class Session:
    def __init__(self, spec: SessionSpec, ledger_path: str,
                 logger: StructuredLogger, *, event_file: str | None = None,
                 ledger: Ledger | None = None,
                 param_version: int | None = None) -> None:
        self.spec = spec
        self.logger = logger
        self.event_file = event_file
        self.owns_ledger = ledger is None
        self.ledger = ledger or Ledger(ledger_path)
        self.ledger.put_meta("schema_version", str(SCHEMA_VERSION))
        self.rule_digest = self.ledger.rule_digest(spec)
        self.ledger.put_meta("rule_digest", self.rule_digest)
        if param_version is None:
            existing = self.ledger.latest_param_version()
            if existing is None:
                self.param_version = 1
                self.ledger.insert_param_version(
                    self.param_version, spec, self.rule_digest,
                    int(time.time() * 1000))
            else:
                row = self.ledger.conn.execute(
                    "SELECT rule_digest,out_of_order_ms,"
                    "latency_budget_ms,scale FROM param_versions "
                    "WHERE version=?", (existing,)).fetchone()
                same = (
                    row["rule_digest"] == self.rule_digest
                    and row["out_of_order_ms"] == spec.out_of_order_ms
                    and row["latency_budget_ms"] == spec.latency_budget_ms
                    and row["scale"] == spec.scale)
                if same:
                    # Same parameters: continue the current active version.
                    self.param_version = existing
                else:
                    # Parameters changed: monotonic replacement version.
                    self.param_version = existing + 1
                    self.ledger.insert_param_version(
                        self.param_version, spec, self.rule_digest,
                        int(time.time() * 1000))
                    self.ledger.mark_versions_stale(self.param_version)
        else:
            self.param_version = param_version
        self.engine = SessionEngine(spec, self.param_version)
        self.drained = False
        self.ledger.conn.commit()

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self.owns_ledger:
            self.ledger.close()

    # ---------------------------------------------------------------- ingest
    def ingest_file(self, path: str) -> dict[str, Any]:
        events = load_events(path, self.spec)
        return self.ingest_events(events)

    def ingest_events(self, events: list[Event]) -> dict[str, Any]:
        known = self._committed_sequences()
        fresh: list[Event] = []
        for event in events:
            if event.seq > known.get(event.partition, -1):
                fresh.append(event)
        steps = self.engine.apply_events(fresh)
        self.ledger.commit_ingest(
            self.param_version, self.spec, fresh, self.engine,
            int(time.time() * 1000))
        late_rows = [
            row for row in self.ledger.late_log_rows(self.param_version)
            if row["partition"] in self.engine.parts
        ]
        for row in late_rows:
            self.logger.emit(
                LATE_CODE,
                "late event truncated from finalized aggregation",
                param_version=self.param_version, partition=row["partition"],
                seq=row["seq"], ts_ms=row["ts_ms"],
                watermark_ms=row["watermark_ms"], source_line=row["source_line"])
        return {
            "ingested": len(fresh),
            "watermark_steps": [
                {"partition": s.partition,
                 "old_watermark_ms": s.old_watermark_ms,
                 "new_watermark_ms": s.new_watermark_ms,
                 "trigger_seq": s.trigger_seq, "reason": s.reason}
                for s in steps],
        }

    def _committed_sequences(self) -> dict[str, int]:
        rows = self.ledger.conn.execute(
            "SELECT partition, MAX(seq) AS m FROM raw_events "
            "WHERE param_version=? GROUP BY partition",
            (self.param_version,)).fetchall()
        return {r["partition"]: r["m"] for r in rows}

    def advance_watermark(self, requested_ms: int) -> dict[str, Any]:
        steps = self.engine.request_watermarks(requested_ms)
        self.ledger._flush_steps(self.param_version, self.engine)
        self.ledger._refresh_state(self.param_version, self.spec, self.engine)
        self.ledger.conn.commit()
        return {"watermark_steps": [
            {"partition": s.partition, "old_watermark_ms": s.old_watermark_ms,
             "new_watermark_ms": s.new_watermark_ms, "trigger_seq": s.trigger_seq,
             "reason": s.reason} for s in steps]}

    def drain(self) -> dict[str, Any]:
        steps = self.engine.drain()
        self.ledger._flush_steps(self.param_version, self.engine)
        self.ledger._refresh_state(self.param_version, self.spec, self.engine)
        self.ledger.conn.commit()
        self.drained = True
        return {"watermark_steps": [
            {"partition": s.partition, "old_watermark_ms": s.old_watermark_ms,
             "new_watermark_ms": s.new_watermark_ms, "trigger_seq": s.trigger_seq,
             "reason": s.reason} for s in steps]}

    # ----------------------------------------------------------------- views
    def cache_view_bytes(self) -> bytes:
        views = self.ledger.cache_views(self.param_version, self.spec)
        return canonical_dumps({
            "param_version": self.param_version,
            "rule_digest": self.rule_digest,
            "partitions": views,
        }).encode("utf-8")

    def authoritative_view_bytes(self) -> bytes:
        views = self.ledger.authoritative_replay(self.param_version, self.spec)
        return canonical_dumps({
            "param_version": self.param_version,
            "rule_digest": self.rule_digest,
            "partitions": views,
        }).encode("utf-8")

    def verify_views(self) -> dict[str, Any]:
        """Compare cache vs authoritative replay byte-for-byte (E_VIEW)."""
        cache = self.ledger.cache_views(self.param_version, self.spec)
        replay = self.ledger.authoritative_replay(
            self.param_version, self.spec)
        if len(cache) != len(replay):
            raise ViewMismatchError(
                "partition set differs between cache and replay",
                first_partition=None, field="<partitions>",
                cache_count=len(cache), replay_count=len(replay))
        fields = ("watermark_ms", "max_ts_seen_ms", "finalized_count",
                  "late_count", "sum_raw", "sum_fixed", "sum_saturated",
                  "cursor_seq")
        for cached, rebuilt in zip(cache, replay):
            if cached["partition"] != rebuilt["partition"]:
                raise ViewMismatchError(
                    "partition ordering/identity differs",
                    first_partition=cached["partition"], field="partition",
                    cache=cached["partition"], replay=rebuilt["partition"])
            for field in fields:
                if cached[field] != rebuilt[field]:
                    raise ViewMismatchError(
                        "cache view diverges from authoritative replay",
                        first_partition=cached["partition"], field=field,
                        cache=cached[field], replay=rebuilt[field])
        if self.cache_view_bytes() != self.authoritative_view_bytes():
            raise ViewMismatchError(
                "canonical serialization differs despite equal fields",
                first_partition=cache[0]["partition"], field="<serialization>")
        return {"status": "identical", "partitions": len(cache),
                "view_bytes": len(self.cache_view_bytes())}

    # --------------------------------------------------------------- queries
    def query(self, partition: str | None = None, *,
              inject_delay_ms: int = 0) -> QueryResult:
        if partition is not None and partition not in self.engine.parts:
            raise KeyError(f"unknown partition '{partition}'")
        start_ns, end_fn = measure_ms()
        # The cache read is the fast UI path; verify_views is run separately
        # on demand and never silently substituted for this result.
        view = self.ledger.cache_views(self.param_version, self.spec)
        if partition is not None:
            view = [v for v in view if v["partition"] == partition]
        if inject_delay_ms > 0:
            time.sleep(inject_delay_ms / 1000.0)
        measured = end_fn()
        latency_ms = measured + max(0, inject_delay_ms)
        budget = self.spec.latency_budget_ms
        over = latency_ms > budget
        query_seq = self.ledger.next_query_seq(self.param_version)
        self.ledger.record_query(
            self.param_version, query_seq, partition, latency_ms, budget,
            over, max(0, inject_delay_ms), start_ns)
        code = LAT_CODE if over else None
        if over:
            self.logger.emit(
                LAT_CODE, "UI query exceeded latency budget",
                param_version=self.param_version, query_seq=query_seq,
                partition=partition, latency_ms=latency_ms,
                budget_ms=budget)
        view_bytes = canonical_dumps({
            "param_version": self.param_version, "partitions": view})
        return QueryResult(query_seq=query_seq, partition=partition,
                           latency_ms=latency_ms, budget_ms=budget,
                           over_budget=over,
                           injected_ms=max(0, inject_delay_ms), code=code,
                           view_bytes=view_bytes)

    def latency_report(self) -> dict[str, Any]:
        """Latency truth source: committed query_log rows only."""
        rows = self.ledger.fetch_queries(self.param_version)
        overall = [r["latency_ms"] for r in rows]
        per_partition: dict[str, list[int]] = {}
        for row in rows:
            key = row["partition"] if row["partition"] is not None else "<global>"
            per_partition.setdefault(key, []).append(row["latency_ms"])
        report = {
            "param_version": self.param_version,
            "truth_source": "ledger.query_log",
            "overall": compute_stats(overall, self.spec.scale).as_dict(),
            "per_partition": {
                key: compute_stats(samples, self.spec.scale).as_dict()
                for key, samples in sorted(per_partition.items())},
            "over_budget_count": sum(1 for r in rows if r["over_budget"]),
        }
        return report

    # ------------------------------------------------------- param versions
    def replace_parameters(self, new_spec: SessionSpec) -> dict[str, Any]:
        """Whole-view replacement after a parameter change (no merging)."""
        old_views = {v["partition"]: v for v in
                     self.ledger.cache_views(self.param_version, self.spec)}
        old_spec = self.spec
        self.spec = new_spec
        self.rule_digest = self.ledger.rule_digest(new_spec)
        self.ledger.put_meta("rule_digest", self.rule_digest)
        self.param_version += 1
        self.ledger.insert_param_version(
            self.param_version, new_spec, self.rule_digest,
            int(time.time() * 1000))
        self.ledger.mark_versions_stale(self.param_version)
        self.engine = SessionEngine(new_spec, self.param_version)
        self.drained = False
        self.ledger.conn.commit()
        affected = []
        for key in sorted(self.engine.parts):
            old = old_views.get(key)
            affected.append({
                "partition": key,
                "old_watermark_ms": None if old is None else old["watermark_ms"],
                "new_watermark_ms": 0,
                "old_scale": old_spec.scale,
                "new_scale": new_spec.scale,
                "old_out_of_order_ms": old_spec.out_of_order_ms,
                "new_out_of_order_ms": new_spec.out_of_order_ms,
                "old_latency_budget_ms": old_spec.latency_budget_ms,
                "new_latency_budget_ms": new_spec.latency_budget_ms,
            })
        self.logger.emit(
            "E_VERSION", "parameters replaced; stale views hidden",
            new_param_version=self.param_version,
            old_out_of_order_ms=old_spec.out_of_order_ms,
            new_out_of_order_ms=new_spec.out_of_order_ms,
            old_scale=old_spec.scale, new_scale=new_spec.scale,
            affected_partitions=affected)
        return {"new_param_version": self.param_version,
                "stale_versions": list(range(1, self.param_version)),
                "affected_partitions": affected}

    # ------------------------------------------------------------- snapshot
    def snapshot_payload(self) -> dict[str, Any]:
        views = self.ledger.cache_views(self.param_version, self.spec)
        queries = [dict(r) for r in
                   self.ledger.fetch_queries(self.param_version)]
        return {
            "snapshot_id": f"snap-v{self.param_version}-"
                           f"{int(time.time() * 1000)}",
            "param_version": self.param_version,
            "rule_digest": self.rule_digest,
            "schema_version": SCHEMA_VERSION,
            "spec": self.spec.digest_input(),
            "event_file": self.event_file,
            "committed_cursor": {
                v["partition"]: v["cursor_seq"] for v in views},
            "partition_views": views,
            "query_ledger": queries,
        }

    def save_snapshot(self, path: str) -> dict[str, Any]:
        payload = self.snapshot_payload()
        write_snapshot(path, payload)
        return {"path": path,
                "content_digest": content_digest(payload),
                "param_version": self.param_version}


def restore_session(snapshot_path: str, ledger_path: str,
                    logger: StructuredLogger, *,
                    tampered: bool = False) -> Session:
    """Open a snapshot; mismatch triggers full replacement (E_SNAPSHOT)."""
    payload = read_snapshot(snapshot_path, tampered=tampered)
    spec = load_spec_from_digest_input(payload["spec"])
    ledger = Ledger(ledger_path)
    snap_version = payload["param_version"]
    snap_digest = payload["rule_digest"]
    current_version = ledger.latest_param_version()
    current_digest = ledger.get_meta("rule_digest")
    replacement_required = (
        current_version is None or current_version != snap_version
        or current_digest != snap_digest)
    if replacement_required:
        logger.emit(
            "E_SNAPSHOT",
            "snapshot differs from current session; full state replacement",
            snapshot_param_version=snap_version,
            current_param_version=current_version,
            snapshot_rule_digest=snap_digest,
            current_rule_digest=current_digest)
        # Full replacement: rebuild ledger content solely from the snapshot.
        ledger.conn.execute("DELETE FROM partition_state")
        ledger.conn.execute("DELETE FROM raw_events")
        ledger.conn.execute("DELETE FROM watermark_log")
        ledger.conn.execute("DELETE FROM late_log")
        ledger.conn.execute("DELETE FROM query_log")
        ledger.put_meta("rule_digest", snap_digest)
        if ledger.latest_param_version() != snap_version:
            ledger.insert_param_version(
                snap_version, spec, snap_digest, int(time.time() * 1000))
        ledger.mark_versions_stale(snap_version)
        for view in payload["partition_views"]:
            ledger.conn.execute(
                "INSERT INTO partition_state(param_version,partition,"
                "watermark_ms,max_ts_seen_ms,finalized_count,late_count,"
                "raw_sum,sum_fixed,sum_saturated,cursor_seq) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (snap_version, view["partition"], view["watermark_ms"],
                 view["max_ts_seen_ms"], view["finalized_count"],
                 view["late_count"], view["sum_raw"], view["sum_fixed"],
                 int(view["sum_saturated"]), view["cursor_seq"]))
        for query in payload["query_ledger"]:
            ledger.conn.execute(
                "INSERT INTO query_log(param_version,query_seq,partition,"
                "latency_ms,budget_ms,over_budget,injected_ms,clock_start_ns)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (snap_version, query["query_seq"], query["partition"],
                 query["latency_ms"], query["budget_ms"],
                 query["over_budget"], query["injected_ms"],
                 query["clock_start_ns"]))
        ledger.conn.commit()
    session = Session(spec, ledger_path, logger, event_file=payload.get("event_file"),
                      ledger=ledger, param_version=snap_version)
    # Re-align the in-memory engine with every committed record so resume
    # continues from the snapshot cursor instead of replaying from zero.
    session.engine, session._hydrated_cursors = ledger.hydrate_engine(
        snap_version, spec)
    session.snapshot_info = payload
    session.replacement_required = replacement_required
    return session


def load_spec_from_digest_input(material: dict[str, Any]) -> SessionSpec:
    from .config import PartitionRule
    return SessionSpec(
        out_of_order_ms=material["out_of_order_ms"],
        latency_budget_ms=material["latency_budget_ms"],
        scale=material["scale"],
        rules=tuple(PartitionRule(key=r["key"],
                                  aggregations=tuple(r["aggregations"]))
                    for r in material["rules"]),
        raw=dict(material))
