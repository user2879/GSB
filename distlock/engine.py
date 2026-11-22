"""Middleware engine: declaration -> subprocess arbiters -> scheduler -> ledger.

The engine is the single integration point used by the demo and the diagnostic
UI. It processes the validated event stream in deterministic batches, persists
each batch atomically, writes whole-state snapshots, and can be forced to crash
mid-batch to demonstrate the unknown-state / replay recovery contract.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .clock import Clock, MonotonicClock
from .errors import ExhaustError
from .ledger import Ledger
from .models import (
    SCALE_CALIBRE_VERSION,
    Domain,
    Holder,
    LeaseEvent,
    ScaleFactor,
)
from .orchestrator import ArbiterPool
from .scheduler import BoundedScheduler
from .snapshot import (
    compute_digest,
    replay_ledger_state,
    verify_replay,
    write_snapshot,
)


class EngineCrash(RuntimeError):
    """Raised after the simulated forced-kill hook fires."""


class DistLockEngine:
    def __init__(self, domains: dict[str, Domain], holders: dict[str, Holder],
                 session: dict[str, Any], workdir: str | Path,
                 host: str = "127.0.0.1", base_port: int = 39000,
                 clock: Clock | None = None, batch_size: int = 4,
                 python_exe: str | None = None,
                 crash_after_inflight_seq: int | None = None):
        self.domains = domains
        self.holders = holders
        self.session = session
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.base_port = base_port
        self.clock: Clock = clock or MonotonicClock()
        self.batch_size = batch_size
        self.scale = ScaleFactor(int(session["scale_factor"]))
        self.crash_after_inflight_seq = crash_after_inflight_seq
        self._crashed = False

        self.ledger_path = self.workdir / "ledger.sqlite3"
        self.snapshot_path = self.workdir / "snapshots" / "session.snapshot.json"
        self.runtime_dir = self.workdir / "runtime"
        self.ledger = Ledger(self.ledger_path)
        self.pool = ArbiterPool(
            domains, host, base_port, self.scale.scale,
            int(session["domain_set_version"]),
            runtime_dir=self.runtime_dir, python_exe=python_exe)
        self.scheduler = BoundedScheduler(
            global_limit=int(session["global_concurrency"]),
            per_holder_limit=int(session["per_holder_concurrency"]),
            backoff_ms=list(session["retry_backoff_ms"]),
            clock=self.clock)
        self.results: list[dict[str, Any]] = []

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.pool.start_all()

    def close(self) -> None:
        self.scheduler.shutdown()
        self.pool.shutdown()
        self.ledger.close()

    def __enter__(self) -> "DistLockEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------------- run
    def run(self, events: list[LeaseEvent]) -> dict[str, Any]:
        cursor = self.ledger.progress_cursor()
        pending = [e for e in events if e.seq > cursor]
        summary_batches: list[dict[str, Any]] = []

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            now = self.clock.now_ms()
            # Crash window: inflight rows are durable before the batch tx opens.
            self.ledger.mark_inflight(batch, now)
            if (self.crash_after_inflight_seq is not None
                    and any(e.seq == self.crash_after_inflight_seq for e in batch)
                    and not self._crashed):
                self._crashed = True
                self._force_kill_hook()
                raise EngineCrash(
                    f"simulated forced termination after inflight seq "
                    f"{self.crash_after_inflight_seq}")

            seq_from, seq_to = batch[0].seq, batch[-1].seq
            batch_receipts: list[dict[str, Any]] = []
            batch_transitions: list[dict[str, Any]] = []
            batch_watermarks: list[dict[str, Any]] = []
            seen_transition_keys: set[tuple] = set()

            with self.ledger.batch(seq_from, seq_to, now) as btx:
                for event in batch:
                    holder = self.holders[event.holder_id]
                    client = self.pool.client(event.domain_id)
                    try:
                        scheduled = self.scheduler.submit(
                            event.seq, event.holder_id, event.domain_id,
                            lambda ev=event, h=holder: client.request(
                                ev.seq, ev.holder_id, ev.action, h.priority))
                    except ExhaustError:
                        raise
                    receipt = scheduled.receipt
                    assert receipt is not None
                    batch_receipts.append(receipt)
                    btx.add_receipt(receipt)
                    for tr in (receipt.get("transitions", [])
                               + receipt.get("drain_transitions", [])):
                        key = (tr["seq"], tr["domain_id"], tr["holder_id"],
                               tr["state_to"], tr["reason"], tr["clock_ms"])
                        if key not in seen_transition_keys:
                            seen_transition_keys.add(key)
                            btx.add_transitions([tr])
                            batch_transitions.append(tr)
                    wm = self._watermark_from_receipt(receipt)
                    btx.add_watermark(wm)
                    batch_watermarks.append(wm)
                    self.results.append({
                        "seq": event.seq,
                        "receipt": receipt,
                        "scheduling": {
                            "attempts": [asdict(a) for a in scheduled.attempts],
                            "total_wait_quota_ms": scheduled.total_wait_quota_ms},
                    })
                self.ledger.resolve_inflight([e.seq for e in batch])

            summary_batches.append({
                "seq_from": seq_from, "seq_to": seq_to,
                "digest": self.ledger.latest_digest(),
                "receipt_count": len(batch_receipts)})
            self._write_snapshot()

        return {
            "processed": len(pending),
            "cursor": self.ledger.progress_cursor(),
            "batches": summary_batches,
            "scheduler": self.scheduler.metrics(),
            "generations": self.pool.generations(),
        }

    def _watermark_from_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        dom = self.domains[receipt["domain_id"]]
        return {
            "seq": receipt["seq"],
            "domain_id": receipt["domain_id"],
            "active_leases": receipt["active_leases"],
            "capacity": dom.capacity,
            "free_capacity": receipt["free_capacity"],
            "queue_length": receipt["queue_length"],
            "ratio_fixed": receipt["ratio_fixed"],
            "scale": receipt["scale"],
            "truncation": "floor",
            "error_bound_fixed": receipt["error_bound_fixed"],
        }

    # -------------------------------------------------------------- snapshots
    def _write_snapshot(self) -> None:
        domains_payload: dict[str, Any] = {}
        for did in sorted(self.domains):
            snap = self.pool.client(did)
            # Ask the live arbiter for its view via a lightweight status request.
            status = _request_status(snap)
            domains_payload[did] = status
        payload = {
            "domain_set_version": int(self.session["domain_set_version"]),
            "scale_calibre_version": SCALE_CALIBRE_VERSION,
            "scale_factor": self.scale.scale,
            "ledger_cursor": self.ledger.progress_cursor(),
            "ledger_batch_count": self.ledger.batch_count(),
            "ledger_digest": self.ledger.latest_digest(),
            "domains": domains_payload,
        }
        payload["digest"] = compute_digest(payload)
        write_snapshot(self.snapshot_path, payload)

    # --------------------------------------------------------------- recovery
    def recover(self) -> dict[str, Any]:
        """Rebuild after a forced termination using only committed ledger rows.

        Arbiters were killed with the process group; they are restarted as a
        fresh generation with empty hold tables, then committed events are
        replayed to rehydrate ownership. Uncommitted inflight leases are marked
        UNKNOWN. No exit report for the killed process is fabricated.
        """
        now = self.clock.now_ms()
        unknown = self.ledger.recover_unknown(now)
        cursor = self.ledger.progress_cursor()

        for did in sorted(self.domains):
            self.pool.restart_domain(did)

        # Replay only committed events to rehydrate the fresh, empty arbiters.
        replay_events = self._committed_events_for_replay()
        for event in replay_events:
            holder = self.holders[event.holder_id]
            client = self.pool.client(event.domain_id)
            client.request(event.seq, event.holder_id, event.action,
                           holder.priority)

        report = {
            "cursor": cursor,
            "unknown_events": unknown,
            "replayed_event_count": len(replay_events),
            "generations": self.pool.generations(),
            "note": ("no fabricated exit report for the killed process; "
                     "uncommitted leases marked UNKNOWN and require replay"),
        }
        if self.snapshot_path.exists():
            report["snapshot"] = self.verify_snapshot_replay()
        return report

    def _committed_events_for_replay(self) -> list[LeaseEvent]:
        # Implemented by the runner which owns the event list; engine keeps the
        # last loaded stream on ``self.events`` for this purpose.
        return [e for e in getattr(self, "events", [])
                if e.seq <= self.ledger.progress_cursor()]

    def verify_snapshot_replay(self) -> dict[str, Any]:
        from .snapshot import read_snapshot

        snap = read_snapshot(self.snapshot_path)
        cursor = self.ledger.progress_cursor()
        rows = replay_ledger_state(
            self.ledger.all_receipts(),
            self.ledger.all_transitions(),
            self.ledger.all_watermarks(),
            self.domains,
            {hid: h.priority for hid, h in self.holders.items()},
            up_to_seq=cursor)
        verify_replay(snap, rows, self.domains, up_to_seq=cursor)
        return {"status": "OK", "cursor": cursor,
                "domains_checked": sorted(self.domains)}

    def diagnostic_view(self) -> dict[str, Any]:
        view_domains: dict[str, Any] = {}
        for did in sorted(self.domains):
            status = _request_status(self.pool.client(did))
            view_domains[did] = status
        return {
            "domain_set_version": int(self.session["domain_set_version"]),
            "scale_factor": self.scale.scale,
            "truncation": "floor",
            "error_bound_fixed": self.scale.error_bound_fixed(),
            "ledger_cursor": self.ledger.progress_cursor(),
            "ledger_batch_count": self.ledger.batch_count(),
            "ledger_digest": self.ledger.latest_digest(),
            "generations": self.pool.generations(),
            "holders": {
                hid: {"priority": h.priority,
                      "target_domains": list(h.target_domains)}
                for hid, h in sorted(self.holders.items())},
            "domains": view_domains,
            "scheduler": self.scheduler.metrics(),
        }

    def _force_kill_hook(self) -> None:
        # Real forced termination of every arbiter subprocess; the main process
        # is then signalled via EngineCrash. SQLite WAL discards the open tx.
        for proc in self.pool.processes.values():
            proc.kill()
        self.pool.clients.clear()


def _request_status(client) -> dict[str, Any]:
    """Fetch a stable status snapshot from an arbiter (read-only)."""
    return client.status()
