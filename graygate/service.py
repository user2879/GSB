from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from .backend import ForwardEngine, LocalBackendClient
from .errors import RequestError
from .gateway import GatewayEngine
from .ledger import LedgerStore
from .metrics import (
    aggregate_hits,
    summarize_latency_full,
    summarize_latency_streaming,
)
from .snapshot import (
    ConsistencyResult,
    Snapshot,
    authoritative_check,
    diff_snapshots,
    materialize_snapshot,
    replay_saved_snapshot,
)
from .spec import GatewaySpec, load_spec
from .util import canonical_json, content_digest, monotonic_ms


class WorkbenchService:
    def __init__(self, spec: GatewaySpec | str | Path, db_path: str | Path):
        self.spec = (
            load_spec(spec) if isinstance(spec, (str, Path)) else spec
        )
        self.db_path = Path(db_path)
        self.ledger = LedgerStore(self.db_path)
        self.forward_engine = ForwardEngine(self.spec)
        self.local_backend = LocalBackendClient(self.forward_engine)
        self.gateway_engine = GatewayEngine(
            self.spec,
            self.ledger,
            self.forward_engine,
            backend=self.local_backend,
        )
        self._lock = threading.RLock()
        self._rebuild_events: dict[str, threading.Event] = {}
        self._rebuild_errors: dict[str, str] = {}
        self._active_rebuild_id: str | None = None
        self._last_changes: list[dict[str, Any]] = []
        latest = self.ledger.latest_snapshot()
        self._current_snapshot = Snapshot.from_row(latest) if latest else None
        self.last_recovery_report: dict[str, Any] | None = None

    @property
    def current_snapshot(self) -> Snapshot | None:
        with self._lock:
            return self._current_snapshot

    @property
    def last_changes(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._last_changes)

    def close(self) -> None:
        with self._lock:
            active = self._active_rebuild_id
        if active is not None:
            event = self._rebuild_events.get(active)
            if event is not None:
                event.wait(timeout=10.0)
        self.ledger.close()

    def ingest(
        self, client_key: str, feature: str, sample: Sequence[float]
    ) -> dict[str, Any]:
        return self.gateway_engine.handle(
            {
                "client_key": client_key,
                "feature": feature,
                "sample": list(sample),
            }
        )

    def start_rebuild(self, *, pause_ms: int = 0) -> str:
        with self._lock:
            if self._active_rebuild_id is not None:
                raise RequestError(
                    "E_REBUILD",
                    "$.rebuild",
                    f"rebuild {self._active_rebuild_id!r} is already running",
                )
            source = self.ledger.current_version()
            started_ms = monotonic_ms()
            rebuild_id = "rb-" + uuid.uuid4().hex
            event = threading.Event()
            self._rebuild_events[rebuild_id] = event
            self._rebuild_errors.pop(rebuild_id, None)
            self._active_rebuild_id = rebuild_id
            self.ledger.create_rebuild(
                rebuild_id=rebuild_id,
                source_ledger_version=source,
                started_ms=started_ms,
                budget_ms=self.spec.rebuild_latency_budget_ms,
            )
            thread = threading.Thread(
                target=self._run_rebuild,
                args=(rebuild_id, source, started_ms, max(0, pause_ms)),
                name=f"snapshot-rebuild-{rebuild_id}",
                daemon=True,
            )
            thread.start()
            return rebuild_id

    def _run_rebuild(
        self,
        rebuild_id: str,
        source_ledger_version: int,
        started_ms: int,
        pause_ms: int,
    ) -> None:
        event = self._rebuild_events[rebuild_id]
        try:
            if pause_ms:
                time.sleep(pause_ms / 1000.0)
            hits = self.ledger.hits_upto(source_ledger_version)
            stats, aggregation_path = aggregate_hits(hits, self.spec)
            observed_version, excluded_hits = self.ledger.snapshot_bound(
                source_ledger_version
            )
            finished_ms = monotonic_ms()
            elapsed_ms = max(0, finished_ms - started_ms)
            snapshot = materialize_snapshot(
                snapshot_id="snap-" + uuid.uuid4().hex,
                rebuild_id=rebuild_id,
                spec=self.spec,
                source_ledger_version=source_ledger_version,
                observed_ledger_version=observed_version,
                excluded_hits=excluded_hits,
                stats=stats,
                aggregation_path=aggregation_path,
                rebuild_elapsed_ms=elapsed_ms,
                budget_ms=self.spec.rebuild_latency_budget_ms,
            )
            self.ledger.complete_rebuild(
                rebuild_id=rebuild_id,
                snapshot=snapshot.to_ledger_payload(),
                observed_ledger_version=observed_version,
                excluded_hits=excluded_hits,
                elapsed_ms=elapsed_ms,
                latency_exceeded=snapshot.latency_exceeded,
            )
            self.ledger.record_latency(
                kind="rebuild",
                ref_id=rebuild_id,
                start_ms=started_ms,
                end_ms=finished_ms,
                budget_ms=self.spec.rebuild_latency_budget_ms,
                impact_scope=(
                    f"snapshot_rebuild:{rebuild_id}; snapshot publication delayed; "
                    "request admission and routing unaffected"
                ),
            )
            with self._lock:
                previous = self._current_snapshot
                self._last_changes = diff_snapshots(previous, snapshot)
                self._current_snapshot = snapshot
        except BaseException as exc:
            finished_ms = monotonic_ms()
            error_code = getattr(exc, "code", "E_REBUILD")
            try:
                self.ledger.fail_rebuild(
                    rebuild_id=rebuild_id,
                    error_code=error_code,
                    detail={"message": str(exc), "type": type(exc).__name__},
                )
                self.ledger.record_latency(
                    kind="rebuild",
                    ref_id=rebuild_id,
                    start_ms=started_ms,
                    end_ms=finished_ms,
                    budget_ms=self.spec.rebuild_latency_budget_ms,
                    impact_scope=(
                        f"snapshot_rebuild:{rebuild_id}; rebuild failed; "
                        "request admission and routing unaffected"
                    ),
                    error_code=error_code,
                )
            finally:
                self._rebuild_errors[rebuild_id] = str(exc)
        finally:
            with self._lock:
                if self._active_rebuild_id == rebuild_id:
                    self._active_rebuild_id = None
            event.set()

    def wait_rebuild(self, rebuild_id: str, timeout: float | None = None) -> dict[str, Any]:
        event = self._rebuild_events.get(rebuild_id)
        if event is None:
            raise RequestError(
                "E_REBUILD", "$.rebuild_id", f"unknown rebuild {rebuild_id!r}"
            )
        if not event.wait(timeout=timeout):
            raise RequestError(
                "E_REBUILD",
                "$.rebuild_id",
                f"rebuild {rebuild_id!r} did not finish in time",
            )
        error = self._rebuild_errors.get(rebuild_id)
        if error is not None:
            raise RequestError(
                "E_REBUILD", "$.rebuild_id", error, {"rebuild_id": rebuild_id}
            )
        info = self.ledger.get_rebuild(rebuild_id)
        if info is None:
            raise RequestError(
                "E_REBUILD", "$.rebuild_id", f"rebuild {rebuild_id!r} vanished"
            )
        return info

    def rebuild_now(self, *, pause_ms: int = 0) -> Snapshot:
        rebuild_id = self.start_rebuild(pause_ms=pause_ms)
        self.wait_rebuild(rebuild_id, timeout=60.0)
        snapshot = self.current_snapshot
        assert snapshot is not None
        return snapshot

    def _snapshot_for_id(self, snapshot_id: str | None) -> Snapshot:
        if snapshot_id is None:
            current = self.current_snapshot
            if current is None:
                raise RequestError(
                    "E_SNAPSHOT", "$.snapshot", "no completed snapshot is available"
                )
            return current
        row = self.ledger.get_snapshot(snapshot_id)
        if row is None:
            raise RequestError(
                "E_SNAPSHOT", "$.snapshot_id", f"unknown snapshot {snapshot_id!r}"
            )
        return Snapshot.from_row(row)

    def check_consistency(self, snapshot_id: str | None = None) -> ConsistencyResult:
        snapshot = self._snapshot_for_id(snapshot_id)
        result = authoritative_check(
            snapshot, self.spec, self.ledger, self.forward_engine
        )
        self.ledger.save_consistency(
            snapshot_id=snapshot.snapshot_id,
            check_kind="AUTHORITATIVE",
            ok=result.ok,
            code=result.code,
            rule_id=result.rule_id,
            stage_id=result.stage_id,
            field_name=result.field_name,
            detail=result.canonical(),
        )
        return result

    def recover_and_replay(self) -> dict[str, Any]:
        chain_ok, chain_version, chain_field = self.ledger.verify_ledger_chain()
        snapshots = list(reversed(self.ledger.list_snapshots(limit=10000)))
        replay_results: list[dict[str, Any]] = []
        for row in snapshots:
            snapshot = Snapshot.from_row(row)
            if not chain_ok:
                result = ConsistencyResult(
                    ok=False,
                    code="E_REPLAY",
                    message=(
                        "ledger content digest chain failed before snapshot replay"
                    ),
                    rule_id="-",
                    stage_id="-",
                    field_name=chain_field,
                    snapshot_value=chain_version,
                    recomputed_value="verified-chain",
                    hit_count=snapshot.source_ledger_version,
                )
            else:
                result = replay_saved_snapshot(snapshot, self.ledger)
            self.ledger.save_consistency(
                snapshot_id=snapshot.snapshot_id,
                check_kind="REPLAY",
                ok=result.ok,
                code=result.code,
                rule_id=result.rule_id,
                stage_id=result.stage_id,
                field_name=result.field_name,
                detail=result.canonical(),
            )
            replay_results.append(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "source_ledger_version": snapshot.source_ledger_version,
                    **result.canonical(),
                }
            )
        report = {
            "ledger_chain": {
                "ok": chain_ok,
                "first_bad_version": chain_version,
                "field": chain_field,
            },
            "unknown_rebuilds": self.ledger.recovered_unknown,
            "replays": replay_results,
            "exit_report_claimed": False,
            "retrigger_required_for_unknown": bool(self.ledger.recovered_unknown),
        }
        self.last_recovery_report = report
        return report

    def _latency_payload(self, kind: str) -> dict[str, Any]:
        values = self.ledger.latency_values(kind)
        full = summarize_latency_full(values)
        streaming = summarize_latency_streaming(values)
        if canonical_json(full.canonical()) != canonical_json(streaming.canonical()):
            raise AssertionError(f"{kind} full/streaming latency summaries differ")
        path = "streaming" if len(values) >= self.spec.stream_threshold else "full"
        summary = streaming if path == "streaming" else full
        return {
            "values_ms": values,
            "aggregation_path": path,
            "summary": summary.canonical(),
            "summary_digest": content_digest(summary.canonical()),
            "full_streaming_equal": True,
            "clock_basis": "local_monotonic_ms",
            "truth_source": "ledger.per_hit_records",
        }

    def dashboard(self, *, record_latency: bool = True) -> dict[str, Any]:
        started_ms = monotonic_ms()
        with self._lock:
            current = self._current_snapshot
            changes = list(self._last_changes)
        snapshot_payload: dict[str, Any] | None = None
        if current is not None:
            consistency = self.ledger.latest_consistency(
                current.snapshot_id, "AUTHORITATIVE"
            )
            replay = self.ledger.latest_consistency(current.snapshot_id, "REPLAY")
            snapshot_payload = {
                "snapshot_id": current.snapshot_id,
                "source_ledger_version": current.source_ledger_version,
                "observed_ledger_version": current.observed_ledger_version,
                "excluded_hits": current.excluded_hits,
                "content_digest": current.content_digest,
                "format_version": current.format_version,
                "aggregation_path": current.aggregation_path,
                "rebuild_elapsed_ms": current.rebuild_elapsed_ms,
                "budget_ms": current.budget_ms,
                "latency_exceeded": current.latency_exceeded,
                "stats": current.stats,
                "last_authoritative_check": consistency,
                "last_replay_check": replay,
            }
        rebuilds = self.ledger.list_rebuilds(limit=100)
        snapshots = [
            {
                "snapshot_id": row["snapshot_id"],
                "source_ledger_version": row["source_ledger_version"],
                "observed_ledger_version": row["observed_ledger_version"],
                "excluded_hits": row["excluded_hits"],
                "content_digest": row["content_digest"],
                "rebuild_elapsed_ms": row["rebuild_elapsed_ms"],
                "budget_ms": row["budget_ms"],
                "latency_exceeded": bool(row["latency_exceeded"]),
            }
            for row in self.ledger.list_snapshots(limit=100)
        ]
        finished_ms = monotonic_ms()
        observation: dict[str, Any] | None = None
        if record_latency:
            observation = self.ledger.record_latency(
                kind="ui_query",
                ref_id=snapshot_payload["snapshot_id"] if snapshot_payload else "none",
                start_ms=started_ms,
                end_ms=finished_ms,
                budget_ms=self.spec.query_latency_budget_ms,
                impact_scope=(
                    "dashboard_query; snapshot view may be stale; "
                    "request admission and routing unaffected"
                ),
            )
        rebuild_latency = self._latency_payload("rebuild")
        query_latency = self._latency_payload("ui_query")
        errors: list[dict[str, Any]] = []
        if observation is not None and observation["exceeded"]:
            errors.append(
                {
                    "code": "E_LATENCY",
                    "field_path": "$.dashboard.query_latency",
                    "message": "dashboard query exceeded the explicit latency budget",
                    "over_by_ms": observation["over_by_ms"],
                    "budget_ms": observation["budget_ms"],
                    "elapsed_ms": observation["elapsed_ms"],
                    "impact_scope": observation["impact_scope"],
                }
            )
        return {
            "ok": not errors,
            "ledger_version": self.ledger.current_version(),
            "ledger_digest": self.ledger.ledger_digest(),
            "snapshot": snapshot_payload,
            "snapshots": snapshots,
            "rebuilds": rebuilds,
            "changes_since_previous": changes,
            "latency": {
                "rebuild": rebuild_latency,
                "ui_query": query_latency,
                "last_query_observation": observation,
            },
            "errors": errors,
            "unknown_rebuilds": self.ledger.recovered_unknown,
            "stream_threshold": self.spec.stream_threshold,
        }

    def write_dashboard_json(self, path: str | Path) -> dict[str, Any]:
        payload = self.dashboard()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return payload

