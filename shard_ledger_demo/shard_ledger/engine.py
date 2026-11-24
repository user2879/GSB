"""Deterministic shard scheduling engine with three-way quota enforcement."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .clock import now_ms
from .errors import LeaseError, TimeoutError as LedgerTimeoutError
from .invalidation import propagate
from .lease import LeaseManager
from .protocol import send_batch
from .spec import Manifest


class Engine:
    def __init__(self, manifest: Manifest, ledger, logger, socket_timeout_ms=2000, max_retries=3):
        self.manifest = manifest
        self.ledger = ledger
        self.logger = logger
        self.socket_timeout_ms = socket_timeout_ms
        self.max_retries = max_retries

        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.leases = LeaseManager(manifest.lease_ttl_ms, on_expire=self._lease_expired)
        self.pool = ThreadPoolExecutor(max_workers=sum(n.concurrency for n in manifest.nodes.values()) + 2)

        self.generation = {sid: 1 for sid in manifest.shards}
        self.config_version = {sid: min(manifest.revisions) for sid in manifest.shards}
        self.state = {sid: "idle" for sid in manifest.shards}
        self.submitted = set()
        self.cancelled = set()
        self.attempts = {}
        self.queue_start = {}
        self.queue_end = {}
        self.inflight = {}
        self.completed = set()
        self.unknown = set()
        self.invalidations = []
        self.quota = {nid: (node.concurrency, node.lease_limit) for nid, node in manifest.nodes.items()}
        self.stop_event = threading.Event()
        self.crash_on_dispatch = None
        self.sweeper = threading.Thread(target=self._sweep_loop, name="lease-sweeper", daemon=True)

    def start(self):
        self.sweeper.start()

    def shutdown(self):
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        self.pool.shutdown(wait=True)

    def is_done(self):
        with self.lock:
            pending = self.submitted - self.cancelled - self.completed - self.unknown
            return not pending and not self.inflight

    def wait_idle(self, timeout=30.0):
        deadline = now_ms() + int(timeout * 1000)
        with self.cond:
            while now_ms() < deadline:
                if self.is_done():
                    return
                self.cond.wait(timeout=0.1)
        raise LedgerTimeoutError("engine did not reach idle state", entity="engine", field="idle")

    def ready_shards(self):
        result = []
        for sid in sorted(self.submitted):
            if self.state[sid] not in ("idle", "queued") or sid in self.cancelled:
                continue
            deps = self.manifest.shards[sid].deps
            if len(deps) == 0 or all(dep in self.completed for dep in deps):
                result.append(sid)
        return result

    def queued_shards(self):
        return sorted(
            sid for sid in self.submitted if self.state[sid] == "queued" and sid not in self.cancelled
        )

    def submit(self, shard_id: str, event_seq: int):
        with self.lock:
            if shard_id in self.submitted or shard_id in self.cancelled:
                return
            self.submitted.add(shard_id)
            self.attempts.setdefault(shard_id, 0)
            self.queue_start[shard_id] = now_ms()
            self.state[shard_id] = "queued"
            self.logger.event("submit", shard_id=shard_id, event_seq=event_seq, state="queued")
            self.cond.notify_all()

    def cancel(self, shard_id: str, event_seq: int):
        with self.lock:
            self.cancelled.add(shard_id)
            self.logger.event("cancel", shard_id=shard_id, event_seq=event_seq)
            if self.state[shard_id] == "queued":
                self.state[shard_id] = "cancelled"
                self.queue_end[shard_id] = now_ms()
                outcome = {
                    "shard_id": shard_id,
                    "generation": self.generation[shard_id],
                    "status": "cancelled",
                    "config_version": self.config_version[shard_id],
                    "summary": None,
                    "units": 0,
                    "event_seq": event_seq,
                }
                self.ledger.write_outcome(outcome)
            self.cond.notify_all()

    def change_quota(self, node_id: str, concurrency: int, lease_limit: int, event_seq: int):
        with self.lock:
            self.quota[node_id] = (concurrency, lease_limit)
            self.logger.event(
                "quota",
                node_id=node_id,
                concurrency=concurrency,
                lease_limit=lease_limit,
                event_seq=event_seq,
            )
            self.cond.notify_all()

    def revise(self, revision_version: int, event_seq: int):
        revision = self.manifest.revisions[revision_version]
        with self.lock:
            self.wait_idle()
            invalidated, basis = propagate(self.manifest, self.completed, revision)
            self.invalidations.append(
                {"event_seq": event_seq, "revision": revision_version, "shards": invalidated, "basis": basis}
            )
            self.ledger.append_propagation(event_seq, revision_version, basis)
            for sid in invalidated:
                self.generation[sid] += 1
                self.config_version[sid] = revision_version
                self.completed.discard(sid)
                self.attempts[sid] = 0
                if sid in self.cancelled:
                    self.state[sid] = "cancelled"
                else:
                    self.state[sid] = "queued"
                    self.queue_start[sid] = now_ms()
            self.logger.event(
                "revise",
                revision=revision_version,
                invalidated=invalidated,
                event_seq=event_seq,
            )
            self.cond.notify_all()
            return invalidated, basis

    def schedule_once(self):
        with self.lock:
            self.leases.sweep_expired()
            ready = self.ready_shards()
            dispatched = []
            for sid in ready:
                spec = self.manifest.shards[sid]
                chosen = []
                blocked_by = []
                for node_id in sorted(spec.targets):
                    concurrency, lease_limit = self.quota[node_id]
                    if not self.leases.can_grant(
                        node_id, spec.units, min(concurrency, lease_limit), self.manifest.global_budget
                    ):
                        blocked_by.append(f"{node_id}:concurrency_lease_or_budget")
                        continue
                    chosen.append(node_id)
                if len(chosen) == len(spec.targets):
                    for node_id in sorted(spec.targets):
                        lease = self.leases.grant(sid, node_id, spec.units)
                        self.ledger.append_lease(lease.to_dict(), -1)
                        token = (sid, node_id, self.generation[sid])
                        self.inflight[token] = {
                            "node_id": node_id,
                            "lease_id": lease.lease_id,
                            "generation": self.generation[sid],
                        }
                        dispatched.append(token)
                    self.state[sid] = "running"
                    self.queue_end[sid] = now_ms()
                    self.attempts[sid] += 1
                    if self.crash_on_dispatch == sid:
                        self.logger.event("forced_termination_mid_dispatch", shard_id=sid)
                        self.logger.close()
                        import os as _os
                        _os._exit(9)
                    self.pool.submit(self._dispatch, sid, list(sorted(spec.targets)), self.generation[sid])
                    self.logger.event(
                        "dispatch",
                        shard_id=sid,
                        generation=self.generation[sid],
                        nodes=list(sorted(spec.targets)),
                        blocked_by=blocked_by,
                    )
            return dispatched

    def run_events(self, events):
        self.start()
        try:
            for event in events:
                self.ledger.append_event(event.__dict__)
                if event.kind == "submit":
                    self.submit(event.shard_id, event.seq)
                elif event.kind == "cancel":
                    self.cancel(event.shard_id, event.seq)
                elif event.kind == "quota":
                    self.change_quota(event.node_id, event.concurrency, event.lease_limit, event.seq)
                elif event.kind == "revise":
                    self.revise(event.revision, event.seq)
                self._pump_until_blocked()
                self.ledger.write_snapshot(event.seq)
            self.wait_idle()
        finally:
            self.shutdown()

    def _pump_until_blocked(self):
        deadline = now_ms() + 30000
        while now_ms() < deadline:
            with self.cond:
                self.leases.sweep_expired()
                ready = self.ready_shards()
                if ready:
                    self.schedule_once()
                inflight = bool(self.inflight)
            if not inflight:
                return
            with self.cond:
                self.cond.wait(timeout=0.02)

    def _dispatch(self, shard_id: str, node_ids: list, generation: int):
        try:
            self._dispatch_inner(shard_id, node_ids, generation)
        except Exception as exc:
            self.logger.event(
                "worker_exception",
                shard_id=shard_id,
                generation=generation,
                error=getattr(exc, "to_dict", lambda: {"code": "E_INTERNAL", "message": repr(exc)})(),
            )
            with self.lock:
                if self.state.get(shard_id) == "running":
                    self._handle_failure(shard_id, generation, exc)
                self.cond.notify_all()

    def _dispatch_inner(self, shard_id: str, node_ids: list, generation: int):
        receipts = []
        lease_ids = {}
        failure = None
        with self.lock:
            for node_id in node_ids:
                token = (shard_id, node_id, generation)
                lease_ids[node_id] = self.inflight[token]["lease_id"]
        for node_id in node_ids:
            node = self.manifest.nodes[node_id]
            try:
                self._renew_if_near_expiry(shard_id, node_id, lease_ids[node_id])
                reply = send_batch(
                    node,
                    shard_id,
                    self.config_version[shard_id],
                    self.socket_timeout_ms,
                )
                receipts.append(reply)
            except LeaseError as exc:
                failure = exc
                break
            except Exception as exc:  # protocol errors carry their own codes
                failure = exc
                break
        with self.lock:
            for node_id in node_ids:
                token = (shard_id, node_id, generation)
                if token in self.inflight:
                    self.leases.release(self.inflight[token]["lease_id"])
                    self.inflight.pop(token, None)
            if failure is not None:
                self._handle_failure(shard_id, generation, failure)
            else:
                self._handle_success(shard_id, generation, receipts)
            self.cond.notify_all()

    def _renew_if_near_expiry(self, shard_id, node_id, lease_id):
        lease = self.leases._leases.get(lease_id)
        if lease is None or not lease.active:
            return
        remaining = lease.expires_ms - now_ms()
        if lease.active and remaining < self.manifest.lease_ttl_ms // 2:
            with self.lock:
                renewed = self.leases.renew(lease_id, lease.version)
                self.ledger.append_lease(renewed.to_dict(), -1)

    def _handle_success(self, shard_id, generation, receipts):
        if self.generation[shard_id] != generation:
            return
        if shard_id in self.cancelled:
            self.state[shard_id] = "cancelled"
            return
        summary = sorted(receipts, key=lambda item: item["node_id"])
        outcome = {
            "shard_id": shard_id,
            "generation": generation,
            "status": "completed",
            "config_version": self.config_version[shard_id],
            "summary": summary,
            "units": self.manifest.shards[shard_id].units,
            "event_seq": -1,
        }
        self.ledger.write_outcome(outcome)
        self.state[shard_id] = "completed"
        self.completed.add(shard_id)
        self.logger.event("complete", shard_id=shard_id, generation=generation)

    def _handle_failure(self, shard_id, generation, error):
        if shard_id in self.cancelled:
            self.state[shard_id] = "cancelled"
            return
        if self.generation[shard_id] != generation:
            return
        code = getattr(error, "code", "E_INTERNAL")
        self.logger.event(
            "dispatch_failed",
            shard_id=shard_id,
            generation=generation,
            error=error.to_dict() if hasattr(error, "to_dict") else {"code": code, "message": str(error)},
        )
        if self.attempts[shard_id] >= self.max_retries:
            self.state[shard_id] = "failed"
            self.unknown.add(shard_id)
            self.ledger.mark_unknown(shard_id, generation, f"retries exhausted: {code}")
            return
        self.state[shard_id] = "queued"
        self.queue_start[shard_id] = now_ms()

    def _lease_expired(self, lease):
        def callback():
            with self.lock:
                token = (lease.shard_id, lease.node_id, self.generation[lease.shard_id])
                self.inflight.pop(token, None)
                self.logger.event(
                    "lease_expired",
                    shard_id=lease.shard_id,
                    node_id=lease.node_id,
                    lease_id=lease.lease_id,
                    units=lease.units,
                )
                gen = self.generation[lease.shard_id]
                remaining = [
                    token
                    for token in self.inflight
                    if token[0] == lease.shard_id and token[2] == gen
                ]
                if not remaining and self.state.get(lease.shard_id) == "running":
                    if lease.shard_id in self.cancelled:
                        self.state[lease.shard_id] = "cancelled"
                    elif self.attempts[lease.shard_id] >= self.max_retries:
                        self.state[lease.shard_id] = "failed"
                        self.unknown.add(lease.shard_id)
                        self.ledger.mark_unknown(lease.shard_id, gen, "lease expired")
                    else:
                        self.state[lease.shard_id] = "queued"
                        self.queue_start[lease.shard_id] = now_ms()
                self.cond.notify_all()
        callback()

    def _sweep_loop(self):
        while not self.stop_event.wait(0.01):
            with self.cond:
                expired = self.leases.sweep_expired()
                if expired:
                    self.cond.notify_all()

    def recover(self):
        latest = self.ledger.latest_outcomes()
        for sid in sorted(self.manifest.shards):
            if sid not in latest:
                continue
            generation, status, version, _summary, _units, _digest = latest[sid]
            self.generation[sid] = generation
            self.config_version[sid] = version
            if status == "completed":
                self.state[sid] = "completed"
                self.completed.add(sid)
                self.submitted.add(sid)
            elif status == "cancelled":
                self.state[sid] = "cancelled"
                self.cancelled.add(sid)
                self.submitted.add(sid)
        for sid, generation, reason in self.ledger.unknown_shards():
            self.unknown.add(sid)
            self.submitted.add(sid)
            self.state[sid] = "unknown"
            self.generation[sid] = generation
        self._mark_inflight_unknown()

    def _mark_inflight_unknown(self):
        processed = self.ledger.processed_events()
        if not processed:
            return
        max_seq = max(processed)
        rows = self.ledger.conn.execute(
            "SELECT payload FROM records WHERE kind='event' ORDER BY seq"
        ).fetchall()
        import json as _json
        submitted, cancelled = set(), set()
        for (payload,) in rows:
            event = _json.loads(payload)
            if event["seq"] > max_seq:
                break
            if event["kind"] == "submit":
                submitted.add(event["shard_id"])
            elif event["kind"] == "cancel":
                cancelled.add(event["shard_id"])
        latest = self.ledger.latest_outcomes()
        for sid in sorted(submitted - cancelled):
            row = latest.get(sid)
            if row is None or row[1] not in ("completed", "cancelled"):
                generation = row[0] if row else self.generation[sid]
                if sid not in self.unknown:
                    self.ledger.mark_unknown(sid, generation, "incomplete at forced termination")
                self.unknown.add(sid)
                self.submitted.add(sid)
                self.state[sid] = "unknown"
                self.generation[sid] = generation

    def replay_verify(self, events):
        from .compute import aggregate
        from .ledger import canonical

        self.recover()
        stored = self.ledger.latest_outcomes()
        for event in events:
            if event.kind == "submit":
                with self.lock:
                    self.submitted.add(event.shard_id)
                    self.attempts.setdefault(event.shard_id, 0)
                    if self.state[event.shard_id] == "idle":
                        self.state[event.shard_id] = "queued"
            elif event.kind == "cancel":
                self.cancelled.add(event.shard_id)
            elif event.kind == "revise":
                revision = self.manifest.revisions[event.revision]
                invalidated, _basis = propagate(self.manifest, self.completed, revision)
                for sid in invalidated:
                    self.generation[sid] += 1
                    self.config_version[sid] = event.revision
                    self.completed.discard(sid)
            elif event.kind == "quota":
                self.quota[event.node_id] = (event.concurrency, event.lease_limit)

        for sid in sorted(stored):
            generation, status, version, stored_summary, units, digest = stored[sid]
            if status != "completed":
                continue
            expected_receipts = []
            for node_id in sorted(self.manifest.shards[sid].targets):
                summary = aggregate(
                    node_id,
                    sid,
                    self.manifest.nodes[node_id].seed,
                    version,
                    self.manifest.revisions,
                )
                expected_receipts.append(
                    {
                        "node_id": node_id,
                        "shard_id": sid,
                        "config_version": version,
                        "summary": summary,
                    }
                )
            stored_receipts = json.loads(stored_summary)
            for receipt in stored_receipts:
                receipt.pop("duration_ms", None)
            expected_canonical = canonical(expected_receipts)
            stored_canonical = canonical(stored_receipts)
            if expected_canonical != stored_canonical:
                self._replay_diff(sid, expected_canonical, stored_canonical)
        return {"verified_shards": sorted(sid for sid in stored if stored[sid][1] == "completed")}

    def _replay_diff(self, shard_id, expected, stored):
        from .errors import ReplayError

        expected_obj = json.loads(expected)
        stored_obj = json.loads(stored)
        for index, (exp, got) in enumerate(zip(expected_obj, stored_obj)):
            for field in sorted(set(exp) | set(got)):
                if exp.get(field) != got.get(field):
                    raise ReplayError(
                        "replay mismatch",
                        entity=shard_id,
                        field=field,
                        node=exp.get("node_id"),
                        expected=exp.get(field),
                        actual=got.get(field),
                    )
        raise ReplayError("replay mismatch with different receipt count", entity=shard_id, field="receipts")
