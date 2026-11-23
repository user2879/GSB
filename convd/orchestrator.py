"""Convergence pulse orchestrator.

Schedules leased convergence pulses against executor child processes over
loopback UDP, enforces global / per-executor in-flight caps, performs
real socket timeouts with retries booked separately from convergence
conclusions, and commits each completed round to the ledger in a single
transaction.
"""

import os
import queue
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import wire
from .errors import WireError
from .executor import state_digest_for
from .ledger import Ledger

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mono_ms():
    return int(time.monotonic() * 1000)


class ExecutorProcess:
    def __init__(self, spec, scenario_id, base_ms, jitter_ms, drop_rounds):
        self.spec = spec
        args = [sys.executable, "-m", "convd.executor",
                "--scenario-id", scenario_id,
                "--executor-id", spec.executor_id,
                "--port", str(spec.port),
                "--seed", str(spec.seed),
                "--target-digest", spec.target_digest,
                "--base-ms", str(base_ms),
                "--jitter-ms", str(jitter_ms)]
        if drop_rounds:
            args += ["--drop-first-attempt-rounds",
                     ",".join(str(r) for r in drop_rounds)]
        self.proc = subprocess.Popen(
            args, cwd=PKG_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)


class Orchestrator:
    def __init__(self, scenario, executor_set, ledger, base_ms=1,
                 jitter_ms=4, drop_map=None):
        self.scenario = scenario
        self.executor_set = executor_set
        self.ledger = ledger
        self.base_ms = base_ms
        self.jitter_ms = jitter_ms
        self.drop_map = drop_map or {}
        self.processes = []
        self.inboxes = {}
        self.lock = threading.Lock()
        self.current_round = {}
        self.accepted_obs = {}
        self.pending_anomalies = []
        self.sock = None
        self.receiver = None
        self.running = False
        # Orchestrator-side per-executor tracking, restored from the
        # ledger on resume so replay/recovery stays consistent.
        self.tracker = {spec.executor_id: {"version": 0, "digest": None,
                                           "obs_seq": 0}
                        for spec in executor_set.specs}

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.2)
        for spec in self.executor_set.specs:
            self.inboxes[spec.executor_id] = queue.Queue()
            self.current_round[spec.executor_id] = 0
            self.accepted_obs[spec.executor_id] = set()
        self.running = True
        self.receiver = threading.Thread(target=self._recv_loop,
                                         daemon=True)
        self.receiver.start()
        for spec in self.executor_set.specs:
            self.processes.append(ExecutorProcess(
                spec, self.scenario.scenario_id, self.base_ms,
                self.jitter_ms, self.drop_map.get(spec.executor_id, ())))
        self._await_ready()

    def stop(self):
        self.running = False
        if self.receiver is not None:
            self.receiver.join(timeout=2)
        for proc in self.processes:
            proc.stop()
        self.processes = []
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _await_ready(self, timeout_ms=10000):
        deadline = mono_ms() + timeout_ms
        pending = {s.executor_id for s in self.executor_set.specs}
        while pending and mono_ms() < deadline:
            for spec in self.executor_set.specs:
                if spec.executor_id in pending:
                    self.sock.sendto(
                        wire.encode_ping(self.scenario.scenario_id),
                        ("127.0.0.1", spec.port))
            with self.lock:
                ready = getattr(self, "_pongs", set())
                pending -= ready
            if pending:
                time.sleep(0.05)
        if pending:
            raise RuntimeError("executors not ready: %s" % sorted(pending))

    # -- receiver ------------------------------------------------------------
    def _recv_loop(self):
        self._pongs = set()
        while self.running:
            try:
                data, _addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            arrival = mono_ms()
            try:
                kind, msg = wire.decode_datagram(data)
            except WireError:
                with self.lock:
                    self.pending_anomalies.append({
                        "executor_id": "<unknown>", "kind": "wire_error",
                        "payload": {"raw_len": len(data)}})
                continue
            if kind == "pong":
                with self.lock:
                    self._pongs.add(msg["executor_id"])
                continue
            executor_id = msg.get("executor_id")
            if executor_id not in self.inboxes:
                continue
            with self.lock:
                current = self.current_round[executor_id]
                if kind == "receipt" and msg["round"] < current:
                    # Late receipt for a closed round: record only, never
                    # modifies the closed round's conclusion.
                    self.pending_anomalies.append({
                        "executor_id": executor_id, "kind": "late_receipt",
                        "payload": {"round": msg["round"],
                                    "obs_seq": msg["obs_seq"],
                                    "arrival_mono_ms": arrival}})
                    continue
            self.inboxes[executor_id].put((kind, msg, arrival))

    # -- pulse execution -----------------------------------------------------
    def _await_receipt(self, executor_id, round_no, timeout_ms):
        """Wait for a matching receipt; returns (kind, msg, arrival)."""
        deadline = mono_ms() + timeout_ms
        inbox = self.inboxes[executor_id]
        while True:
            remaining = deadline - mono_ms()
            if remaining <= 0:
                return None
            try:
                kind, msg, arrival = inbox.get(
                    timeout=max(remaining, 1) / 1000.0)
            except queue.Empty:
                return None
            if kind == "error":
                return (kind, msg, arrival)
            if msg["round"] < round_no:
                with self.lock:
                    self.pending_anomalies.append({
                        "executor_id": executor_id, "kind": "late_receipt",
                        "payload": {"round": msg["round"],
                                    "obs_seq": msg["obs_seq"],
                                    "arrival_mono_ms": arrival}})
                continue
            if msg["round"] > round_no:
                continue
            return (kind, msg, arrival)

    def _pulse_executor(self, spec, round_no, first_send_holder):
        executor_id = spec.executor_id
        timing = {"queue_wait": None, "send_mono_ms": None,
                  "receipt_arrival_mono_ms": None}
        retries = []
        anomalies = []
        # In-flight caps: global semaphore + per-executor semaphore.
        wait_start = mono_ms()
        self.global_sem.acquire()
        spec.sem.acquire()
        wait_end = mono_ms()
        if wait_end > wait_start:
            timing["queue_wait"] = {"start_mono_ms": wait_start,
                                    "end_mono_ms": wait_end}
        try:
            receipt = None
            conclusion = None
            reason = None
            attempt = 0
            last_error = None
            last_send_mono = None
            while attempt <= self.scenario.max_retries and receipt is None \
                    and conclusion != "stale":
                attempt += 1
                send_mono = mono_ms()
                if attempt == 1:
                    first_send_holder[executor_id] = send_mono
                    timing["send_mono_ms"] = send_mono
                packet = wire.encode_pulse(
                    self.scenario.scenario_id, round_no,
                    self.scenario.lease_ms, spec.target_digest, send_mono)
                with self.lock:
                    self.current_round[executor_id] = round_no
                self.sock.sendto(packet, ("127.0.0.1", spec.port))
                last_send_mono = send_mono
                if attempt > 1:
                    retries.append({"executor_id": executor_id,
                                    "attempt": attempt,
                                    "payload": {"kind": "resend",
                                                "round": round_no,
                                                "send_mono_ms": send_mono}})
                got = self._await_receipt(
                    executor_id, round_no, self.scenario.receipt_timeout_ms)
                if got is None:
                    retries.append({"executor_id": executor_id,
                                    "attempt": attempt,
                                    "payload": {"kind": "timeout",
                                                "round": round_no,
                                                "timeout_ms":
                                                self.scenario
                                                .receipt_timeout_ms}})
                    continue
                kind, msg, arrival = got
                # Duplicate observation numbers: keep the first receipt,
                # register each overridden duplicate, keep waiting.
                while kind == "receipt" \
                        and msg["obs_seq"] in self.accepted_obs[executor_id]:
                    anomalies.append({
                        "executor_id": executor_id,
                        "kind": "duplicate_receipt",
                        "payload": {"round": round_no,
                                    "obs_seq": msg["obs_seq"],
                                    "kept": "first",
                                    "overridden_arrival_mono_ms": arrival}})
                    got = self._await_receipt(
                        executor_id, round_no,
                        self.scenario.receipt_timeout_ms)
                    if got is None:
                        break
                    kind, msg, arrival = got
                if got is None:
                    retries.append({"executor_id": executor_id,
                                    "attempt": attempt,
                                    "payload": {"kind": "timeout",
                                                "round": round_no,
                                                "timeout_ms":
                                                self.scenario
                                                .receipt_timeout_ms}})
                    continue
                if kind == "error":
                    if msg["code"] == "E_STALE":
                        conclusion = "stale"
                        reason = "stale_rejected"
                        last_error = msg
                        break
                    conclusion = "not_converged"
                    reason = "executor_error"
                    last_error = msg
                    break
                try:
                    wire.validate_receipt(msg, self.scenario.scenario_id,
                                          round_no,
                                          {executor_id:
                                           self.tracker[executor_id]
                                           ["obs_seq"]})
                except WireError as exc:
                    conclusion = "not_converged"
                    reason = "wire_error"
                    anomalies.append({
                        "executor_id": executor_id, "kind": "wire_error",
                        "payload": {"offset": exc.offset,
                                    "message": exc.message}})
                    break
                receipt = msg
                timing["receipt_arrival_mono_ms"] = arrival
            first_send = first_send_holder[executor_id]
            if receipt is not None:
                self.accepted_obs[executor_id].add(receipt["obs_seq"])
                in_lease = (receipt["obs_mono_ms"]
                            <= first_send + self.scenario.lease_ms)
                version_after = receipt["obs_seq"]
                digest_after = receipt["state_digest"]
                if in_lease:
                    conclusion = "converged"
                    reason = "ok"
                else:
                    conclusion = "not_converged"
                    reason = "lease_expired"
            elif conclusion is None:
                conclusion = "not_converged"
                reason = "timeout"
                in_lease = False
                version_after = self.tracker[executor_id]["version"]
                digest_after = self.tracker[executor_id]["digest"]
            else:
                in_lease = False
                version_after = self.tracker[executor_id]["version"]
                digest_after = self.tracker[executor_id]["digest"]
            prev = self.tracker[executor_id]
            logical = {
                "round": round_no,
                "executor_id": executor_id,
                "version_before": prev["version"],
                "version_after": version_after,
                "digest_before": prev["digest"],
                "digest_after": digest_after,
                "obs_seq": receipt["obs_seq"] if receipt else None,
                "advance_ms": (receipt["obs_mono_ms"] - last_send_mono
                               if receipt else None),
                "attempts": attempt,
                "in_lease": in_lease,
                "conclusion": "converged" if conclusion == "converged"
                else "not_converged",
                "reason": reason if conclusion != "converged" else "ok",
            }
            if receipt is not None:
                self.tracker[executor_id] = {
                    "version": version_after, "digest": digest_after,
                    "obs_seq": receipt["obs_seq"]}
            anomalies.extend(self._cross_round_anomalies(spec, logical))
            return {"logical": logical, "timing": timing,
                    "retries": retries, "anomalies": anomalies}
        finally:
            spec.sem.release()
            self.global_sem.release()

    def _cross_round_anomalies(self, spec, logical):
        """Compare consecutive rounds: regression / digest jump / obs gap."""
        found = []
        executor_id = spec.executor_id
        prev = self._prev_logical.get(executor_id)
        if prev is None or logical["conclusion"] != "converged":
            return found
        if logical["version_after"] < prev["version_after"]:
            found.append({"executor_id": executor_id,
                          "kind": "version_regression",
                          "payload": {"round": logical["round"],
                                      "from": prev["version_after"],
                                      "to": logical["version_after"]}})
        expected = state_digest_for(self.scenario.scenario_id, executor_id,
                                    logical["version_after"],
                                    spec.target_digest)
        if logical["digest_after"] != expected:
            found.append({"executor_id": executor_id,
                          "kind": "digest_jump",
                          "payload": {"round": logical["round"],
                                      "expected": expected,
                                      "actual": logical["digest_after"]}})
        if (logical["obs_seq"] is not None and prev["obs_seq"] is not None
                and logical["obs_seq"] > prev["obs_seq"] + 1):
            found.append({"executor_id": executor_id,
                          "kind": "obs_seq_gap",
                          "payload": {"round": logical["round"],
                                      "from": prev["obs_seq"],
                                      "to": logical["obs_seq"]}})
        return found

    # -- round driver ----------------------------------------------------------
    def restore_tracker(self):
        """Rebuild per-executor tracking from committed ledger rounds."""
        self._prev_logical = {}
        for round_no, executor_id, entry in self.ledger.round_entries():
            self._prev_logical[executor_id] = entry
            if entry["conclusion"] == "converged":
                self.tracker[executor_id] = {
                    "version": entry["version_after"],
                    "digest": entry["digest_after"],
                    "obs_seq": entry["obs_seq"]}
                self.accepted_obs[executor_id].add(entry["obs_seq"])
            self.current_round[executor_id] = round_no

    def run(self, start_round=1, up_to=None):
        scenario = self.scenario
        max_rounds = up_to or scenario.max_rounds
        self.global_sem = threading.Semaphore(scenario.max_in_flight)
        for spec in self.executor_set.specs:
            spec.sem = threading.Semaphore(
                scenario.max_in_flight_per_executor)
        if not hasattr(self, "_prev_logical"):
            self._prev_logical = {}
        workers = min(scenario.max_in_flight, len(self.executor_set.specs))
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for round_no in range(start_round, max_rounds + 1):
                round_start = mono_ms()
                first_send_holder = {}
                futures = [pool.submit(self._pulse_executor, spec, round_no,
                                       first_send_holder)
                           for spec in self.executor_set.specs]
                results = [f.result() for f in futures]
                results.sort(key=lambda r: r["logical"]["executor_id"])
                entries = [{"executor_id": r["logical"]["executor_id"],
                            "logical": r["logical"], "timing": r["timing"]}
                           for r in results]
                retries = [dict(item, round=round_no)
                           for r in results for item in r["retries"]]
                anomalies = [item for r in results
                             for item in r["anomalies"]]
                with self.lock:
                    anomalies.extend(self.pending_anomalies)
                    self.pending_anomalies = []
                anomalies.sort(key=lambda a: (a["executor_id"], a["kind"]))
                self.ledger.commit_round(round_no, entries, anomalies,
                                         retries)
                for entry in entries:
                    self._prev_logical[entry["executor_id"]] = \
                        entry["logical"]
                # Clock cadence: next pulse round at pulse_interval_ms.
                elapsed = mono_ms() - round_start
                wait = scenario.pulse_interval_ms - elapsed
                if wait > 0 and round_no < max_rounds:
                    time.sleep(wait / 1000.0)
        finally:
            pool.shutdown(wait=True)
        return self.ledger.content_digest()
