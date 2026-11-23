import os
import tempfile
import threading
import unittest

from repairbench.content import BlockRegistry
from repairbench.fixedpoint import ErrorLedger, FixedPoint
from repairbench.scheduler import (
    COMMITTED,
    DEFERRED_BUDGET,
    Scheduler,
)
from repairbench.spec import load_spec

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "groups.json")


def make_group(idx=0):
    return load_spec(SAMPLE)[idx]


class InstrumentedScheduler(Scheduler):
    """记录运行期并发与带宽峰值，用于验证三重约束。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.max_running_total = 0
        self.max_running_per_node = {}
        self.max_bw_per_node = {}
        self._obs_lock = threading.Lock()
        self.thread_ids = set()

    def _execute(self, batch, bandwidth_kbps):
        with self._obs_lock:
            self.thread_ids.add(threading.get_ident())
            self.max_running_total = max(self.max_running_total, self._running_total)
            for nid, cnt in self._running_per_node.items():
                self.max_running_per_node[nid] = max(
                    self.max_running_per_node.get(nid, 0), cnt
                )
            for nid, bw in self._bw_used.items():
                self.max_bw_per_node[nid] = max(self.max_bw_per_node.get(nid, 0), bw)
        super()._execute(batch, bandwidth_kbps)


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.group = make_group(0)
        self.registry = BlockRegistry(os.path.join(self.dir, "blocks"))
        for b in self.group.blocks:
            self.registry.materialize(b)
        self.err = ErrorLedger(FixedPoint(self.group.params.scale_exponent))

    def run_instrumented(self):
        sched = InstrumentedScheduler(self.group, self.registry, self.err)
        summary = sched.run()
        return sched, summary

    def test_triple_constraints_hold(self):
        sched, summary = self.run_instrumented()
        limit = self.group.params.concurrency_limit
        self.assertLessEqual(sched.max_running_total, limit)
        for nid, peak in sched.max_running_per_node.items():
            self.assertLessEqual(peak, limit)
        for node in self.group.nodes:
            peak_bw = sched.max_bw_per_node.get(node.node_id, 0)
            self.assertLessEqual(peak_bw, node.bandwidth_kbps)
        self.assertLessEqual(
            summary["consumed_bytes"], self.group.params.global_budget_bytes
        )

    def test_real_worker_threads(self):
        sched, _ = self.run_instrumented()
        self.assertGreater(len(sched.thread_ids), 1)  # 真实线程池

    def test_budget_exhaustion_defers(self):
        # alpha 总块字节 41624 > 预算 40000 => 必有 DEFERRED_BUDGET
        _, summary = self.run_instrumented()
        self.assertGreater(summary["states"].get(DEFERRED_BUDGET, 0), 0)
        self.assertGreater(summary["states"].get(COMMITTED, 0), 0)

    def test_wait_times_recorded(self):
        _, summary = self.run_instrumented()
        waits = [b["wait_ms"] for b in summary["batches"]]
        self.assertTrue(all(w >= 0 for w in waits))
        self.assertTrue(any(w > 0 for w in waits))  # 并发/配额受限必有排队

    def test_batch_split_covers_blocks(self):
        sched = InstrumentedScheduler(self.group, self.registry, self.err)
        batches = sched.plan()
        for block in self.group.blocks:
            segs = [b for b in batches if b.block_id == block.block_id]
            covered = sum(b.byte_count for b in segs)
            self.assertEqual(covered, block.byte_count)
            self.assertEqual(segs[0].start_byte, 0)
            self.assertEqual(segs[-1].end_byte, block.byte_count)

    def test_error_attribution_accumulates(self):
        self.run_instrumented()
        glob = self.err.global_bound()
        self.assertGreater(glob.quantization_trunc, 0)
        self.assertGreater(glob.batch_split_rounding, 0)  # 10000/4096 等有余数
        self.assertGreater(glob.queue_delay_conv, 0)      # 存在排队等待


if __name__ == "__main__":
    unittest.main()
