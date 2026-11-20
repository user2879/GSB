import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gradcomp.config import Config
from gradcomp.digest import bytewise_equal
from gradcomp.engine import _run_tcp_step, run_step, train
from gradcomp.errors import E_DIVERGE, GradCompError


def _cfg(**overrides):
    base = dict(workers=4, dimension=32, seed=314, samples=32,
                scale_exponent=2, truncation="floor", block_size=8,
                base_port=40100, transport="tcp", compressed=True, steps=1)
    base.update(overrides)
    return Config(**base)


class TestTcpEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gc-tcp-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_four_independent_processes_ring(self):
        cfg = _cfg()
        params = np.zeros(cfg.dimension)
        payloads = _run_tcp_step(cfg, 0, params)
        self.assertEqual(len(payloads), 4)
        first = payloads[0]["global_gradient"]
        for item in payloads[1:]:
            self.assertTrue(bytewise_equal(first, item["global_gradient"]))
        for item in payloads:
            self.assertEqual(item["counters"]["phases"],
                             ["reduce_scatter", "all_gather", "barrier"])

    def test_tcp_matches_loopback_bit_for_bit(self):
        tcp_cfg = _cfg(base_port=40200)
        lb_cfg = Config(**{**tcp_cfg.to_dict(), "transport": "loopback"})
        params = np.zeros(tcp_cfg.dimension)
        tcp_payloads, _ = run_step(tcp_cfg, 0, params)
        lb_payloads, _ = run_step(lb_cfg, 0, params)
        np.testing.assert_array_equal(
            tcp_payloads[0]["global_gradient"],
            lb_payloads[0]["global_gradient"])

    def test_divergence_detected(self):
        cfg = _cfg(base_port=40300)
        params = np.zeros(cfg.dimension)
        payloads = _run_tcp_step(cfg, 0, params)
        # 人为破坏其中一个工作进程的全局梯度，模拟不一致。
        payloads[2]["global_gradient"][0] += 1.0
        with self.assertRaises(GradCompError) as ctx:
            # 直接复用引擎一致性检查使用的逐字节比较与首差定位逻辑。
            from gradcomp.digest import first_diff_index
            base = payloads[0]["global_gradient"]
            for rank, item in enumerate(payloads[1:], start=1):
                if not bytewise_equal(base, item["global_gradient"]):
                    idx = first_diff_index(base, item["global_gradient"])
                    raise GradCompError(E_DIVERGE, "diverge",
                                        details={"first_diff_index": idx})
        self.assertEqual(ctx.exception.code, E_DIVERGE)
        self.assertEqual(ctx.exception.details["first_diff_index"], 0)

    def test_tcp_train_writes_snapshots(self):
        cfg = _cfg(base_port=40400, steps=2)
        report = train(cfg, workdir=self.tmp / "s")
        self.assertTrue(all(s["consistent"] for s in report.steps))
        self.assertEqual(len(report.snapshot_files), 2)


if __name__ == "__main__":
    unittest.main()
