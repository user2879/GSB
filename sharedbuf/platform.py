"""Platform core: lease granting, deterministic execution verification,
persistence-backed recovery and snapshot replay."""
from __future__ import annotations

import threading
import time

from . import buffer as buf
from . import lease as leasemod
from .arbiter import Adjudicator, Grant
from .errors import (E_RECOVER, E_REPLAY, E_SEQ, E_VER, PlatformError)
from .protocol import validate_summary, wl_version_for_session
from .store import Store


class Platform:
    def __init__(self, cfg, db_path="platform.db", clock=time.monotonic,
                 node_runner=None):
        self.cfg = cfg
        self.clock = clock
        self.store = Store(db_path)
        session_seq = int(self.store.get_meta("session_seq", "0")) + 1
        self.store.set_meta("session_seq", str(session_seq))
        self.session = f"s{session_seq}"
        self.wl = wl_version_for_session(session_seq)
        self.sectors = list(cfg["sectors"])
        self.owners = dict(cfg["owners"])
        self.sector_index = dict(cfg["sector_index"])
        self.seed = cfg["seed"]
        self.store.init_sector_versions(self.sectors, self.owners)
        self.versions = self.store.get_sector_versions()
        self.leases = leasemod.LeaseManager(self.store, clock)
        self.leases.load(self.store.load_leases())
        self.mirror = {s: buf.initial_sector_data(self.seed, self.sector_index[s])
                       for s in self.sectors}
        self._rebuild_mirror()
        self.node_runner = node_runner
        self._seq = int(self.store.get_meta("seq", "0"))
        self._lease_counter = int(self.store.get_meta("lease_counter", "0"))
        self._grant_lock = threading.Lock()
        self.adjudicator = Adjudicator(
            self, global_limit=cfg.get("global_limit", 4),
            node_quota=cfg.get("node_quota", 2),
            timeout_s=cfg.get("adjudication_timeout_s", 2.0),
            max_retries=cfg.get("adjudication_retries", 1), clock=clock)
        self._reaper_stop = threading.Event()
        self._reaper_thread = None

    def close(self):
        self.stop_reaper()
        self.adjudicator.shutdown()
        self.store.close()

    def start_reaper(self, interval_s=0.05):
        def loop():
            while not self._reaper_stop.wait(interval_s):
                self.leases.reap()
        self._reaper_thread = threading.Thread(target=loop, daemon=True,
                                               name="lease-reaper")
        self._reaper_thread.start()

    def stop_reaper(self):
        self._reaper_stop.set()
        if self._reaper_thread:
            self._reaper_thread.join(timeout=2)
            self._reaper_thread = None

    def grant_lease(self, req):
        with self._grant_lock:
            self._lease_counter += 1
            lease_id = f"L{self.session}-{self._lease_counter}"
            seq_lo = self._seq + 1
            seq_hi = self._seq + req.count
            self._seq = seq_hi
            self.store.set_meta("seq", str(self._seq))
            self.store.set_meta("lease_counter", str(self._lease_counter))
            lease = self.leases.grant(lease_id, req.node_id, req.sectors,
                                      seq_lo, seq_hi,
                                      self.cfg.get("lease_ttl_ms", 5000))
            return Grant(lease.lease_id, lease.node_id, lease.sectors,
                         lease.seq_lo, lease.seq_hi,
                         self.cfg.get("lease_ttl_ms", 5000))

    def submit(self, req):
        return self.adjudicator.submit(req)

    def submit_and_wait(self, req, timeout=None):
        return self.submit(req).result(timeout=timeout)

    def _honest_run(self, lease, req):
        datas = [self.mirror[s] for s in lease.sectors]
        value, _ = buf.run_batch(datas, req.op, req.agg, req.delta, req.count)
        return {"lease_id": lease.lease_id, "sectors": list(lease.sectors),
                "seq_lo": lease.seq_lo, "seq_hi": lease.seq_hi,
                "versions": {s: self.versions[s] for s in lease.sectors},
                "value": value}

    def execute(self, grant, req):
        lease = self.leases.get(grant.lease_id)
        self.leases.activate(lease.lease_id)
        try:
            if self.node_runner is not None:
                summary = self.node_runner(self, lease, req)
            else:
                summary = self._honest_run(lease, req)
            validate_summary(summary, self.wl)
            self.leases.check_access(lease.lease_id, summary["sectors"],
                                     summary["seq_hi"])
            if (summary["seq_lo"] != lease.seq_lo
                    or summary["seq_hi"] != lease.seq_hi):
                raise PlatformError(
                    E_SEQ, lease=lease.lease_id,
                    reason="summary seq range differs from lease",
                    expected=[lease.seq_lo, lease.seq_hi],
                    got=[summary["seq_lo"], summary["seq_hi"]])
            pre_versions = {s: self.versions[s] for s in lease.sectors}
            for sid in sorted(lease.sectors):
                got = summary["versions"].get(sid)
                if got != pre_versions[sid]:
                    raise PlatformError(
                        E_VER, node=lease.node_id, lease=lease.lease_id,
                        stage="versions", sector=sid,
                        expected=pre_versions[sid], got=got)
            datas = [self.mirror[s] for s in lease.sectors]
            expected, post = buf.run_batch(datas, req.op, req.agg,
                                           req.delta, req.count)
            if summary["value"] != expected:
                raise PlatformError(
                    E_VER, node=lease.node_id, lease=lease.lease_id,
                    stage="value", sector=sorted(lease.sectors)[0],
                    expected=expected, got=summary["value"])
            if req.op == "write":
                for idx, sid in enumerate(lease.sectors):
                    self.mirror[sid] = post[idx]
                    self.versions[sid] += req.count
                    self.store.set_sector_version(sid, self.versions[sid])
            self.store.insert_batch_result({
                "lease_id": lease.lease_id, "node_id": lease.node_id,
                "op": req.op, "agg": req.agg, "sectors": list(lease.sectors),
                "seq_lo": lease.seq_lo, "seq_hi": lease.seq_hi,
                "delta": req.delta, "count": req.count,
                "versions": pre_versions, "value": summary["value"],
                "expected": expected, "match": True, "mono": self.clock()})
            self.leases.complete(lease.lease_id, progress=lease.seq_hi)
            return {"lease_id": lease.lease_id, "node_id": lease.node_id,
                    "sectors": list(lease.sectors),
                    "seq_lo": lease.seq_lo, "seq_hi": lease.seq_hi,
                    "value": summary["value"], "expected": expected,
                    "versions": pre_versions}
        except PlatformError as exc:
            if exc.code == E_VER:
                self.store.insert_batch_result({
                    "lease_id": lease.lease_id, "node_id": lease.node_id,
                    "op": req.op, "agg": req.agg,
                    "sectors": list(lease.sectors), "seq_lo": lease.seq_lo,
                    "seq_hi": lease.seq_hi, "delta": req.delta,
                    "count": req.count,
                    "versions": {s: self.versions[s] for s in lease.sectors},
                    "value": exc.detail.get("got"), "expected": exc.detail.get("expected"),
                    "match": False, "mono": self.clock()})
            self.leases.fail(lease.lease_id, exc.code)
            raise

    def _rebuild_mirror(self):
        self.mirror = {s: buf.initial_sector_data(self.seed, self.sector_index[s])
                       for s in self.sectors}
        versions = {s: 0 for s in self.sectors}
        for batch in self.store.list_batches():
            if not batch["match"] or batch["op"] != "write":
                continue
            for sid in batch["sectors"]:
                self.mirror[sid] = self.mirror[sid] + batch["delta"]
                versions[sid] += batch["count"]
        return versions

    def recover(self):
        """Rebuild state after a forced kill using only persisted records."""
        report = {"session": self.session, "unknown_leases": [],
                  "errors": []}
        for lease in self.leases.all_leases():
            if lease.state not in leasemod.TERMINAL:
                self.leases.mark_unknown(lease.lease_id, "restart_unsettled")
                report["unknown_leases"].append(lease.lease_id)
        recomputed = self._rebuild_mirror()
        stored = self.store.get_sector_versions()
        for sid in self.sectors:
            if stored.get(sid, 0) != recomputed.get(sid, 0):
                err = {"code": E_RECOVER, "sector": sid,
                       "stored_version": stored.get(sid),
                       "expected_version": recomputed.get(sid),
                       "reason": "sector version does not match persisted "
                                 "lease/batch progress"}
                report["errors"].append(err)
        self.versions = stored
        report["ok"] = not report["errors"]
        return report

    def replay(self):
        """Re-execute the recorded lease/batch sequence deterministically and
        compare every summary; the first divergence is reported with lease
        id, sector and field via E_REPLAY."""
        mirror = {s: buf.initial_sector_data(self.seed, self.sector_index[s])
                  for s in self.sectors}
        versions = {s: 0 for s in self.sectors}
        batches = self.store.list_batches()
        for batch in batches:
            sectors = batch["sectors"]
            pre = {s: versions[s] for s in sectors}
            for sid in sorted(sectors):
                if batch["versions"].get(sid) != pre[sid]:
                    raise PlatformError(
                        E_REPLAY, lease=batch["lease_id"], sector=sid,
                        field="versions",
                        reason="recorded sector version not reproducible",
                        expected=pre[sid], got=batch["versions"].get(sid))
            datas = [mirror[s] for s in sectors]
            expected, post = buf.run_batch(datas, batch["op"], batch["agg"],
                                           batch["delta"], batch["count"])
            if batch["value"] != expected:
                raise PlatformError(
                    E_REPLAY, lease=batch["lease_id"], sector=sorted(sectors)[0],
                    field="value",
                    reason="recorded summary value not reproducible",
                    expected=expected, got=batch["value"])
            if batch["op"] == "write" and batch["match"]:
                for idx, sid in enumerate(sectors):
                    mirror[sid] = post[idx]
                    versions[sid] += batch["count"]
        return {"status": "ok", "replayed": len(batches)}

    def snapshot(self):
        snap_leases = [l.to_dict() for l in self.leases.all_leases()]
        trace = {"adjudications": self.store.list_adjudications(),
                 "transitions": self.store.list_transitions(),
                 "batches": self.store.list_batches()}
        mono = self.clock()
        self.store.save_snapshot(self.wl, mono, snap_leases,
                                 dict(self.versions), trace)
        return {"wl": self.wl, "mono": mono, "leases": len(snap_leases)}

    def restore_and_replay(self):
        snap = self.store.load_snapshot()
        if snap is None:
            return {"status": "no_snapshot"}
        replaced = False
        if snap["wl"] != self.wl:
            self.leases.replace_all(snap["leases"])
            self.store.replace_leases(snap["leases"])
            self.versions = dict(snap["sector_versions"])
            self.store.replace_sector_versions(self.versions, self.owners)
            replaced = True
        replay = self.replay()
        return {"status": "restored", "snapshot_wl": snap["wl"],
                "current_wl": self.wl, "wholesale_replace": replaced,
                "replay": replay}
