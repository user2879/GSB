"""Injection orchestrator: state machine + dual-budget parallel scheduling.

State machine per injection:
    pending -> active -> observed -> reverted      (happy path)
    any     -> failed                               (business error, evidence kept)
    active  -> unknown                              (found uncommitted after crash)

Scheduling honors both budgets: total-injection cap and parallel cap.
Injections conflicting on the same target are serialized through a
per-target lock in stable seq order; queue wait start/end are recorded.
"""

import json
import os
import queue
import threading
import time

from . import observe
from .errors import ChaosError
from .injectors import (ErrorRateOverride, KillProcess, LatencyProxy,
                        SuspendProcess)
from .services import ServiceManager

STATES = ("pending", "active", "observed", "reverted", "failed", "unknown")


def mono_ms():
    return int(time.monotonic() * 1000)


class Orchestrator:
    def __init__(self, spec, ledger, timeline_entries, log,
                 kill_after=None, only_seqs=None):
        self.spec = spec
        self.ledger = ledger
        self.log = log
        self.kill_after = kill_after
        self.mgr = ServiceManager(spec, log,
                            run_dir=str(ledger.path) + ".run")
        committed = {r[0] for r in ledger.conn.execute(
            "SELECT seq FROM injections WHERE committed=1")}
        entries = timeline_entries[: spec.budget.max_total_injections]
        if only_seqs is not None:
            entries = [e for e in timeline_entries if e.seq in set(only_seqs)]
        else:
            entries = [e for e in entries if e.seq not in committed]
        self.entries = sorted(entries, key=lambda e: e.seq)
        self._target_locks = {s.id: threading.Lock() for s in spec.services}
        self._enqueue_mono = {}
        self._committed = 0
        self._cond = threading.Condition()
        self._errors = []
        self._all_done = threading.Event()

    # ---------------------------------------------------------- lifecycle
    def run(self):
        self.mgr.ensure_running()
        work = queue.Queue()
        for e in self.entries:
            self._enqueue_mono[e.seq] = mono_ms()
            work.put(e)
        for _ in range(min(self.spec.budget.max_parallel, len(self.entries))):
            work.put(None)
        workers = [threading.Thread(target=self._worker, args=(work,), daemon=True)
                   for _ in range(min(self.spec.budget.max_parallel,
                                      len(self.entries)))]
        for w in workers:
            w.start()
        if self.kill_after is not None:
            threading.Thread(target=self._watch_and_crash, daemon=True).start()
        for w in workers:
            w.join()
        self._all_done.set()
        self.mgr.stop_all()
        return {"committed": self._committed,
                "errors": [e.to_dict() for e in self._errors]}

    def _worker(self, work):
        while True:
            entry = work.get()
            if entry is None:
                return
            try:
                self.run_injection(entry)
            except ChaosError as exc:
                self._errors.append(exc)
                self.log.error(exc.code, exc.message, seq=entry.seq,
                               evidence=exc.evidence)
            except Exception as exc:  # unexpected: record as failed, keep going
                self.log.error("E_INTERNAL", str(exc), seq=entry.seq)
            finally:
                with self._cond:
                    self._committed += 1
                    self._cond.notify_all()

    # ------------------------------------------------------------ crash
    def _watch_and_crash(self):
        """Simulate forced termination: after kill_after commits, write an
        in-flight intent for the next unstarted injection and die hard."""
        with self._cond:
            while self._committed < self.kill_after:
                if self._all_done.is_set():
                    return
                self._cond.wait(timeout=0.5)
        existing = {r[0] for r in self.ledger.conn.execute(
            "SELECT seq FROM injections")}
        nxt = next((e for e in self.entries if e.seq not in existing), None)
        if nxt is not None:
            now = mono_ms()
            with self.ledger.transaction() as cur:
                try:
                    cur.execute(
                        "INSERT INTO injections(seq,stable_key,itype,target,"
                        " params_json,planned_start_ms,state,committed,"
                        " activated_mono_ms) VALUES(?,?,?,?,?,?, 'active',0,?)",
                        (nxt.seq, nxt.stable_key, nxt.type, nxt.target,
                         json.dumps(nxt.params, sort_keys=True),
                         nxt.planned_start_ms, now))
                    cur.execute(
                        "INSERT INTO transitions(inj_seq,from_state,to_state,"
                        " reason,mono_ms) VALUES(?,?,?,?,?)",
                        (nxt.seq, "pending", "active",
                         "activated just before forced termination", now))
                except Exception:
                    pass  # a worker beat us to it; it is in-flight anyway
        self.log.emit("forced_termination", kill_after=self.kill_after)
        self.ledger.conn.commit()
        os._exit(137)  # hard kill: no cleanup, no exit report

    # ---------------------------------------------------------- injection
    def _make_injector(self, entry):
        svc = self.spec.service_by_id(entry.target)
        if entry.type == "latency":
            return LatencyProxy(svc.port, entry.params["delay_ms"]), None
        if entry.type == "error_rate":
            return ErrorRateOverride(svc.control_port,
                                     entry.params["error_rate_ppm"]), None
        if entry.type == "suspend":
            return SuspendProcess(self.mgr.proc_of(svc.id).pid,
                                  svc.control_port), None
        if entry.type == "kill":
            restart = entry.params.get("restart", True)
            return KillProcess(self.mgr, svc.id, restart), None
        raise ValueError(entry.type)

    def run_injection(self, entry):
        svc = self.spec.service_by_id(entry.target)
        wait_start = self._enqueue_mono.get(entry.seq, mono_ms())
        lock = self._target_locks[svc.id]
        with lock:
            wait_end = mono_ms()
            sampling = self.spec.sampling
            window = min(entry.observe_ms, self.spec.budget.max_observe_ms,
                         entry.duration_ms)
            # pre-injection observation (direct connection)
            pre = observe.observe_window(svc.port, window, sampling.interval_ms)

            injector, _ = self._make_injector(entry)
            activated = mono_ms()
            # intent row: committed=0; if we die now, recovery marks unknown.
            # a redo finds the unknown row and re-activates it in place.
            with self.ledger.transaction() as cur:
                prev = cur.execute(
                    "SELECT state FROM injections WHERE seq=?",
                    (entry.seq,)).fetchone()
                from_state = prev[0] if prev else "pending"
                reason = ("redo after unknown with original seed"
                          if prev else "activated")
                cur.execute(
                    "INSERT INTO injections(seq,stable_key,itype,target,"
                    " params_json,planned_start_ms,state,committed,"
                    " activated_mono_ms,wait_start_mono_ms,wait_end_mono_ms)"
                    " VALUES(?,?,?,?,?,?, 'active',0,?,?,?)"
                    " ON CONFLICT(seq) DO UPDATE SET state='active',"
                    " committed=0, activated_mono_ms=excluded.activated_mono_ms,"
                    " wait_start_mono_ms=excluded.wait_start_mono_ms,"
                    " wait_end_mono_ms=excluded.wait_end_mono_ms",
                    (entry.seq, entry.stable_key, entry.type, entry.target,
                     json.dumps(entry.params, sort_keys=True),
                     entry.planned_start_ms, activated, wait_start, wait_end))
                cur.execute(
                    "INSERT INTO transitions(inj_seq,from_state,to_state,"
                    " reason,mono_ms) VALUES(?,?,?,?,?)",
                    (entry.seq, from_state, "active", reason, activated))
            self.log.emit("injection_active", seq=entry.seq, type=entry.type,
                          target=entry.target)

            proxy_port = None
            try:
                if entry.type == "latency":
                    proxy_port = injector.apply()
                else:
                    injector.apply()
                obs_port = proxy_port or svc.port
                active = observe.observe_window(obs_port, window,
                                                sampling.interval_ms)
                observed_at = mono_ms()
                injector.revert()
                reverted_at = mono_ms()
            except ChaosError as exc:
                self._commit_failure(entry, exc)
                raise
            post = observe.observe_window(svc.port, window,
                                          sampling.interval_ms)
            self._commit_success(entry, pre, active, post,
                                 observed_at, reverted_at)
            self.log.emit("injection_reverted", seq=entry.seq)

    # ------------------------------------------------------------ commits
    def _insert_observations(self, cur, seq, phases):
        sampling = self.spec.sampling
        for phase, samples in phases.items():
            summary = observe.summarize(samples, sampling.scale,
                                        sampling.min_samples)
            for s in samples:
                cur.execute(
                    "INSERT INTO requests(inj_seq,phase,req_idx,mono_ms,ok,"
                    " latency_ms) VALUES(?,?,?,?,?,?)",
                    (seq, phase, s["req_idx"], s["mono_ms"], s["ok"],
                     s["latency_ms"]))
            cur.execute(
                "INSERT INTO observations(inj_seq,phase,samples,success,fail,"
                " rate_fixed,rate_err_bound,p50_fixed,p95_fixed,"
                " quant_err_bound,scale,truncation,decidable)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (seq, phase, summary["samples"], summary["success"],
                 summary["fail"], summary["rate_fixed"],
                 summary["rate_err_bound"], summary["p50_fixed"],
                 summary["p95_fixed"], summary["quant_err_bound"],
                 summary["scale"], summary["truncation"],
                 summary["decidable"]))

    def _commit_success(self, entry, pre, active, post, observed_at, reverted_at):
        with self.ledger.transaction() as cur:
            cur.execute(
                "UPDATE injections SET state='reverted', committed=1,"
                " reverted_mono_ms=? WHERE seq=?",
                (reverted_at, entry.seq))
            cur.execute(
                "INSERT INTO transitions(inj_seq,from_state,to_state,reason,"
                " mono_ms) VALUES(?,?,?,?,?)",
                (entry.seq, "active", "observed",
                 "observation window complete", observed_at))
            cur.execute(
                "INSERT INTO transitions(inj_seq,from_state,to_state,reason,"
                " mono_ms) VALUES(?,?,?,?,?)",
                (entry.seq, "observed", "reverted",
                 "revert verified: effect gone", reverted_at))
            self._insert_observations(cur, entry.seq,
                                      {"pre": pre, "active": active,
                                       "post": post})
            cur.execute(
                "INSERT INTO progress(id,cursor) VALUES(1,?)"
                " ON CONFLICT(id) DO UPDATE SET cursor=excluded.cursor",
                (entry.seq,))

    def _commit_failure(self, entry, exc):
        now = mono_ms()
        with self.ledger.transaction() as cur:
            cur.execute(
                "UPDATE injections SET state='failed', committed=1,"
                " business_code=?, evidence=? WHERE seq=?",
                (exc.code, json.dumps(exc.to_dict(), sort_keys=True),
                 entry.seq))
            cur.execute(
                "INSERT INTO transitions(inj_seq,from_state,to_state,reason,"
                " mono_ms) VALUES(?,?,?,?,?)",
                (entry.seq, "active", "failed", exc.message, now))


def check_observation_isolation(ledger):
    """Diagnostic: no two injections on the same target may have overlapping
    active-observation windows, and every request must belong to exactly the
    injection whose window contains it. Returns a list of violations."""
    violations = []
    windows = {}
    for seq, target in ledger.conn.execute(
            "SELECT seq, target FROM injections WHERE committed=1"):
        row = ledger.conn.execute(
            "SELECT MIN(mono_ms), MAX(mono_ms) FROM requests"
            " WHERE inj_seq=? AND phase='active'", (seq,)).fetchone()
        if row and row[0] is not None:
            windows.setdefault(target, []).append((seq, row[0], row[1]))
    for target, wins in windows.items():
        wins.sort(key=lambda w: w[1])
        for (seq_a, lo_a, hi_a), (seq_b, lo_b, hi_b) in zip(wins, wins[1:]):
            if lo_b <= hi_a:
                violations.append({
                    "target": target, "seq_a": seq_a, "seq_b": seq_b,
                    "overlap_ms": hi_a - lo_b,
                    "detail": "active observation windows overlap"})
    return violations
