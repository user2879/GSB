"""槽位水位与借用约束下的分区聚合推进引擎。

执行单元为 (分区, 子分区) 上的一个批次；默认单线程严格按确定顺序执行，
可通过 workers>1 启用线程池验证“结果与执行交错无关”。
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .aggregation import LocalAggregator, merge_aggregators
from .digest import content_digest, canonical_json
from .hashing import RULE_NAME
from .ledger import Ledger
from .scan import build_scan_plan, mapping_summary
from .slots import SlotManager


class Engine:
    def __init__(self, spec, out_dir, clock, workers=1,
                 inject_cycle=False, inject_deadline=False,
                 inject_digest=False, crash_after_batch=None):
        self.spec = spec
        self.out_dir = out_dir
        self.clock = clock
        self.workers = max(1, workers)
        self.slot_manager = SlotManager(
            spec.topology, clock,
            inject_cycle=inject_cycle, inject_deadline=inject_deadline,
            release_on_key_cap=getattr(spec, "release_on_key_cap", False),
        )
        self.ledger = Ledger(os.path.join(out_dir, "ledger.sqlite"))
        self.inject_digest = inject_digest
        self.crash_after_batch = crash_after_batch
        self.aggregators = {}
        self.table_digest = content_digest(spec.raw.get("tables", []))
        self.thresholds = {
            "hotspot_row_threshold": spec.hotspot.row_threshold,
            "subpartition_count": spec.hotspot.subpartition_count,
            "partition_count": spec.topology.partition_count,
            "hash_rule": RULE_NAME,
        }

    def _aggregate_payload(self, conn_id, key_encoded, values):
        agg = self.aggregators.get(conn_id)
        if agg is None:
            agg = LocalAggregator(conn_id, self.spec.query.aggregates, self.spec.scale)
            self.aggregators[conn_id] = agg
        agg.add_row(key_encoded, values)

    def _build_tasks(self, plan):
        units = defaultdict(list)
        for row in plan.rows:
            sub = row.subpartition if row.subpartition >= 0 else -1
            units[(row.partition, sub)].append(row)
        tasks = []
        batch_size = self.spec.batch_size
        for (partition, sub) in sorted(units):
            rows = sorted(units[(partition, sub)], key=lambda r: r.row_index)
            for start in range(0, len(rows), batch_size):
                tasks.append({
                    "partition": partition,
                    "subpartition": sub,
                    "rows": rows[start:start + batch_size],
                    "last_row_index": rows[min(start + batch_size, len(rows)) - 1].row_index,
                })
        return tasks

    def _run_task(self, task):
        self._current_task = task
        for row in task["rows"]:
            conn = self.slot_manager.acquire(
                task["partition"], row.key_encoded, task["subpartition"],
            )
            self._aggregate_payload(conn.conn_id, row.key_encoded, row.values)
            conn.batch_count += 1
        with self.slot_manager.lock:
            for conn in self.slot_manager.partition_conns[task["partition"]]:
                if conn.state != "released":
                    self.slot_manager.mark_idle(conn.conn_id)
            self.slot_manager.recycle_locked()

    def capture_partial_snapshot(self):
        return [a.to_ledger() for a in sorted(
            self.aggregators.values(), key=lambda a: a.conn_id)]

    def _commit_batch(self, batch_id, task):
        watermark = self.slot_manager.watermark_snapshot()
        partials = [a.to_ledger() for a in sorted(
            self.aggregators.values(), key=lambda a: a.conn_id)]
        borrows = [{
            "borrower": b.borrower, "lender": b.lender,
            "key_cap": b.borrowed_key_cap, "deadline_ms": b.deadline_ms,
        } for b in sorted(self.spec.topology.borrows, key=lambda x: x.borrower)]
        payload = {
            "batch_id": batch_id,
            "partition": task["partition"],
            "subpartition": task["subpartition"],
            "slot_params_version": self.spec.slot_params_version,
            "thresholds": self.thresholds,
            "last_row_index": task["last_row_index"],
            "row_indices": [r.row_index for r in task["rows"]],
        }
        digest = self.ledger.append_batch({
            "batch_id": batch_id,
            "committed_ms": self.clock.now_ms(),
            "payload": payload,
            "connections": watermark["connections"],
            "borrows": borrows,
            "partials": partials,
            "last_row_index": task["last_row_index"],
        })
        if self.inject_digest and batch_id == 0:
            with self.ledger.conn:
                self.ledger.conn.execute(
                    "UPDATE batches SET digest=? WHERE batch_id=0",
                    ("tampered" + digest[:-8],))
        return digest, payload, watermark

    def run(self, resume_from=-1, task_start=None, write=True):
        replaced = self.ledger.initialize(
            self.table_digest, self.spec.slot_params_version, self.thresholds)
        plan = build_scan_plan(self.spec, redistribute=True)
        tasks = self._build_tasks(plan)
        if task_start is None:
            next_id = max(resume_from + 1, self.ledger.last_batch_id() + 1)
            pending = tasks[next_id:]
        else:
            next_id = task_start
            pending = tasks[task_start:]
        if self.workers == 1:
            for offset, task in enumerate(pending):
                self._run_task(task)
                batch_id = next_id + offset
                if write:
                    self._commit_batch(batch_id, task)
                self._maybe_crash(batch_id)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = []
                for offset, task in enumerate(pending):
                    futures.append((next_id + offset, task, pool.submit(self._run_task, task)))
                for batch_id, task, future in futures:
                    future.result()
                    if write:
                        self._commit_batch(batch_id, task)
                    self._maybe_crash(batch_id)
        if write:
            self.ledger.write_snapshot(
                self.table_digest, self.spec.slot_params_version,
                self.thresholds, replaced,
            )
        return plan, replaced

    def _maybe_crash(self, batch_id):
        if self.crash_after_batch is not None and batch_id == self.crash_after_batch:
            self.slot_manager.mark_unknown()
            unknown = [
                c.conn_id for c in self.slot_manager.connections.values() if c.unknown
            ]
            self.ledger.record_crash(batch_id, sorted(unknown))
            self.ledger.conn.commit()
            os._exit(7)

    def finalize(self, plan):
        result = merge_aggregators(
            list(self.aggregators.values()),
            self.spec.query.aggregates, self.spec.scale,
        )
        watermark = self.slot_manager.watermark_snapshot()
        return {
            "mapping_summary": mapping_summary(plan, self.spec),
            "subpartition_assignments": {
                str(p): {
                    key.hex() if isinstance(key, bytes) else _key_hex(key): sub
                    for key, sub in sorted(m.items(), key=lambda kv: _key_hex(kv[0]))
                }
                for p, m in sorted(plan.subpartition_assignments.items())
            },
            "watermark": watermark,
            "result": result,
            "result_digest": content_digest(result),
        }

    def hydrate_aggregators_from_ledger(self, batch_id):
        """从已提交批次的最后快照重建全部历史局部累加器。"""
        stored = self.ledger.load_batch(batch_id)
        if stored is None:
            return
        for partial in stored["partials"]:
            agg = LocalAggregator(partial["conn_id"],
                                  self.spec.query.aggregates, self.spec.scale)
            for group in partial["groups"]:
                key = _hex_to_key(group["key_encoded_hex"])
                acc = agg._acc(key)
                acc.count = group["count"]
                acc.sums = dict(group["sums"])
                acc.avg_inputs = dict(group["avg_inputs"])
            self.aggregators[partial["conn_id"]] = agg

    def seed_slot_sequence(self, batch_id):
        """恢复续跑前，令连接序号从账本已用最大序号之后开始，避免 ID 冲突。"""
        ids = []
        for row in self.ledger.conn.execute(
            "SELECT conn_id FROM connections WHERE batch_id<=?", (batch_id,)):
            ids.append(row[0])
        max_index = 0
        for cid in ids:
            token = cid.split("slot")[-1]
            if token.isdigit():
                max_index = max(max_index, int(token))
        self.slot_manager.id_offset = max_index

    def close(self):
        self.ledger.close()


def _key_hex(key_encoded):
    return b"".join(bytes((t,)) + p for t, p in key_encoded).hex()


def _hex_to_key(hex_string):
    raw = bytes.fromhex(hex_string)
    key = []
    for offset in range(0, len(raw), 9):
        tag = raw[offset]
        payload = bytes(raw[offset + 1:offset + 9])
        key.append((tag, payload))
    return tuple(key)
