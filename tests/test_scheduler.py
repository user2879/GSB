import json
import os
import socket
import unittest

from src.core import load_config
from src.errors import PortError, SocketTimeoutError
from src.scheduler import BatchTask, Scheduler
from tests.util import TmpDir, make_config, make_ports


class TestScheduler(unittest.TestCase):
    def _scheduler(self, tmp, port_base, delay=0, timeout_ms=3000, **cfg_over):
        cfg_path = make_config(tmp, **cfg_over)
        ports_path = make_ports(tmp, base=port_base)
        cfg = load_config(cfg_path)
        cfg["__config_path__"] = cfg_path
        cfg["scheduler"]["timeout_ms"] = timeout_ms
        ports = json.load(open(ports_path))
        return Scheduler(cfg, ports, compute_delay_ms=delay)

    def test_dispatch_over_real_loopback(self):
        with TmpDir() as tmp:
            with self._scheduler(tmp, 24501) as sched:
                tasks = [BatchTask(batch_id="batch-0000", worker_id="w0",
                                   param_version=1,
                                   request_ids=("req-0000", "req-0001"))]
                (res,) = sched.run(tasks)
                self.assertIsNone(res.error)
                self.assertEqual(len(res.receipts), 2)
                self.assertEqual(res.receipts[0].src_format, 3)
                self.assertGreaterEqual(res.wait_end_ms, res.wait_start_ms)

    def test_quota_limits_in_flight(self):
        with TmpDir() as tmp:
            sched = self._scheduler(tmp, 24511, delay=300)
            with sched:
                peaks = []
                orig_run_task = sched._run_task

                def wrapped(task, results, idx):
                    orig_run_task(task, results, idx)

                import threading
                stop = threading.Event()

                def probe():
                    peak = 0
                    while not stop.is_set():
                        snap = sched.stats_snapshot()
                        peak = max(peak, snap["global"]["in_flight"])
                    peaks.append(peak)

                t = threading.Thread(target=probe)
                t.start()
                tasks = [BatchTask(batch_id=f"batch-{i:04d}",
                                   worker_id="w0" if i % 2 == 0 else "w1",
                                   param_version=1,
                                   request_ids=("req-0000",)) for i in range(6)]
                results = sched.run(tasks)
                stop.set()
                t.join()
                self.assertTrue(all(r.error is None for r in results))
                self.assertLessEqual(peaks[0], 2)

    def test_timeout_reported_e_timeout(self):
        with TmpDir() as tmp:
            with self._scheduler(tmp, 24521, delay=1500, timeout_ms=200) as sched:
                tasks = [BatchTask(batch_id="batch-0000", worker_id="w0",
                                   param_version=1, request_ids=("req-0000",))]
                (res,) = sched.run(tasks)
                self.assertIsInstance(res.error, SocketTimeoutError)
                self.assertEqual(res.error.code, "E_TIMEOUT")

    def test_port_occupied_reported_e_port(self):
        with TmpDir() as tmp:
            hold = socket.socket()
            hold.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            hold.bind(("127.0.0.1", 24531))
            hold.listen(1)
            try:
                with self.assertRaises(PortError) as ctx:
                    self._scheduler(tmp, 24531)
                self.assertEqual(ctx.exception.code, "E_PORT")
            finally:
                hold.close()

    def test_wait_window_recorded_monotonic_ms(self):
        with TmpDir() as tmp:
            with self._scheduler(tmp, 24541, delay=200) as sched:
                tasks = [BatchTask(batch_id=f"batch-{i:04d}", worker_id="w0",
                                   param_version=1,
                                   request_ids=("req-0000",)) for i in range(3)]
                results = sched.run(tasks)
                queued = [r for r in results if r.wait_end_ms > r.wait_start_ms]
                self.assertTrue(queued, "串行配额下应存在排队等待")
                for res in results:
                    self.assertGreaterEqual(res.finish_ms, res.wait_end_ms)


if __name__ == "__main__":
    unittest.main()
