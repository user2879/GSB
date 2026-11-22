"""Session runner: opens the store (upgrading stale blocks), compiles plans,
schedules batches under the dual constraints, evaluates them vectorized,
writes the ledger transactionally, merges deterministically and saves the
session snapshot. Supports simulated crashes (crash_after) for recovery and
replay testing.
"""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .errors import ReplayError
from .executor import Vec, aggregate, evaluate_batch, merge_batches, result_digest
from .expr import parse
from .formats import make_bitmap
from .ledger import Ledger
from .plan import compile_plan
from .quant import QuantContext
from .scheduler import Scheduler, batch_mem_units
from .store import BlockStore


class SimulatedCrash(Exception):
    """Raised to simulate a forced kill after N committed batches."""


def vec_to_payload(vec):
    return json.dumps({
        "values": [int(v) for v in vec.values],
        "nulls": make_bitmap([i for i, x in enumerate(vec.nulls) if x],
                             len(vec.values)).hex(),
        "div_zero": make_bitmap([i for i, x in enumerate(vec.div_zero) if x],
                                len(vec.values)).hex(),
    }, sort_keys=True)


def vec_from_payload(payload):
    doc = json.loads(payload)
    nulls = bytes.fromhex(doc["nulls"])
    divz = bytes.fromhex(doc["div_zero"])
    n = len(doc["values"])
    return Vec(doc["values"],
               [bool(nulls[i // 8] >> (i % 8) & 1) for i in range(n)],
               [bool(divz[i // 8] >> (i % 8) & 1) for i in range(n)])


def batch_digest(task_id, batch, vec):
    h = hashlib.sha256()
    h.update(task_id.encode() + b"\x00")
    h.update(batch.row_start.to_bytes(8, "little"))
    h.update(batch.row_end.to_bytes(8, "little"))
    h.update(vec_to_payload(vec).encode())
    return h.hexdigest()


class SessionRunner:
    def __init__(self, data_dir, tasks_path, work_dir,
                 global_budget_units=None, max_workers=None):
        self.data_dir = data_dir
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        with open(tasks_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        self.tasks = spec["tasks"]
        self.global_budget = (global_budget_units
                              or spec["global_memory_budget_units"])
        self.max_workers = max_workers or spec.get("max_workers", 4)
        self.store = BlockStore(data_dir)
        self.ledger = Ledger(os.path.join(work_dir, "ledger.db"))
        self._ledger_lock = threading.Lock()
        self._crash_after = None
        self._crashed = False
        self._committed = 0
        self._crash_lock = threading.Lock()

    def _record_batch(self, *args, **kwargs):
        with self._ledger_lock:
            self.ledger.record_batch(*args, **kwargs)

    def run(self, crash_after=None):
        """Execute the session. crash_after=N simulates a forced kill after
        N newly committed batches (no snapshot is saved)."""
        self._crash_after = crash_after
        self._crashed = False
        self._committed = 0
        upgraded = self.store.upgrade_all()
        snapshot_report = self.ledger.check_snapshot()
        quotas = {t["task_id"]: t.get("quota", 1) for t in self.tasks}
        scheduler = Scheduler(self.global_budget, quotas)
        plans, asts, ctxs, blocks = {}, {}, {}, {}
        for t in self.tasks:
            block = self.store.get_block(t["block_id"])
            blocks[t["task_id"]] = block
            ctxs[t["task_id"]] = QuantContext(
                t.get("scale_exp", block.quant_scale_exp))
            plans[t["task_id"]] = compile_plan(
                t["expression"], block.row_count, t["batch_max_rows"])
            asts[t["task_id"]] = parse(t["expression"])

        batch_results = {t["task_id"]: [] for t in self.tasks}
        futures = []
        crashed = False
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for t in self.tasks:
                tid = t["task_id"]
                done = self.ledger.done_batches(tid)
                payloads = self.ledger.done_batch_payloads(tid)
                for bidx, (row_start, payload) in payloads.items():
                    batch_results[tid].append((row_start,
                                               vec_from_payload(payload)))
                for batch in plans[tid].batches:
                    if batch.index in done:
                        continue  # already committed in the ledger
                    futures.append(pool.submit(
                        self._run_one_batch, scheduler, t, blocks[tid],
                        asts[tid], ctxs[tid], batch, batch_results[tid]))
            try:
                for f in futures:
                    f.result()
            except SimulatedCrash:
                crashed = True
                for f in futures:
                    f.cancel()
        if crashed:
            return {"crashed": True, "upgraded": upgraded,
                    "snapshot": snapshot_report,
                    "committed_batches": self._committed}

        task_reports = {}
        for t in self.tasks:
            tid = t["task_id"]
            vec = merge_batches(batch_results[tid])
            agg = t.get("aggregation", "none")
            agg_value = aggregate(vec, agg)
            digest = result_digest(plans[tid].normalized,
                                   ctxs[tid].scale_exp, vec, agg, agg_value)
            existing = self.ledger.get_result(tid)
            if existing is not None and existing[0] != digest:
                raise ReplayError("resumed run diverges from ledger result",
                                  task_id=tid, recorded=existing[0],
                                  replayed=digest)
            self.ledger.record_result(tid, digest, agg, agg_value,
                                      plans[tid].normalized,
                                      ctxs[tid].scale_exp)
            task_reports[tid] = {
                "final_digest": digest, "aggregation": agg,
                "agg_value": agg_value,
                "div_zero_rows": int(sum(vec.div_zero)),
                "null_rows": int(sum(vec.nulls)),
                "batches": len(plans[tid].batches),
            }
        snapshot_digest = self.ledger.save_snapshot()
        return {
            "crashed": False,
            "upgraded": upgraded,
            "snapshot": snapshot_report,
            "snapshot_digest": snapshot_digest,
            "tasks": task_reports,
            "scheduler": {
                "max_running_units": scheduler.max_running_units,
                "max_running_per_task": scheduler.max_running_per_task,
                "timings": [tm.as_dict() for tm in sorted(
                    scheduler.timings,
                    key=lambda x: (x.task_id, x.batch_index))],
            },
        }

    def _run_one_batch(self, scheduler, task, block, ast, ctx, batch, out):
        tid = task["task_id"]
        mem_units = batch_mem_units(batch.rows, len(block.columns))

        def work():
            return evaluate_batch(block, ast, ctx, batch.row_start,
                                  batch.row_end)

        vec, timing = scheduler.run_batch(tid, batch.index, mem_units, work)
        digest = batch_digest(tid, batch, vec)
        with self._crash_lock:
            if self._crashed:
                raise SimulatedCrash()
            self._record_batch(tid, batch, "done", mem_units, digest,
                               cursor=batch.row_end,
                               payload=vec_to_payload(vec), timing=timing)
            self._committed += 1
            if (self._crash_after is not None
                    and self._committed >= self._crash_after):
                self._crashed = True
                raise SimulatedCrash()
        out.append((batch.row_start, vec))
        return digest

    def close(self):
        self.store.close()
        self.ledger.close()
