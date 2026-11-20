import threading
import unittest

import numpy as np

from gradcomp import protocol as proto
from gradcomp.config import Config
from gradcomp.errors import E_PROTO, GradCompError
from gradcomp.ring import RingEngine
from gradcomp.transport import LoopbackTransport


def _run_ring(cfg: Config, local: list[list[np.ndarray]],
              reorder_seed=None):
    broker = LoopbackTransport(cfg.workers, reorder_seed=reorder_seed)
    results = [None] * cfg.workers

    def target(rank):
        engine = RingEngine(rank, cfg.workers, cfg.block_size, broker)
        owned = engine.reduce_scatter(local[rank], step=0)
        results[rank] = engine.all_gather(owned, step=0)
        engine.barrier(step=0)

    threads = [threading.Thread(target=target, args=(r,))
               for r in range(cfg.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class TestRing(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(workers=4, dimension=32, block_size=8)
        rng = np.random.default_rng(0)
        self.local = [
            [rng.integers(-5, 5, size=8).astype(np.int16)
             for _ in range(4)]
            for _ in range(4)]

    def test_reduce_scatter_all_gather_sum(self):
        results = _run_ring(self.cfg, self.local)
        expected = sum(
            np.concatenate(self.local[r]) for r in range(self.cfg.workers))
        for got in results:
            np.testing.assert_array_equal(got, expected)

    def test_message_order_independence(self):
        ordered = _run_ring(self.cfg, self.local)
        shuffled = _run_ring(self.cfg, self.local, reorder_seed=12345)
        for a, b in zip(ordered, shuffled):
            np.testing.assert_array_equal(a, b)

    def test_block_size_independent_result(self):
        small = Config(workers=4, dimension=32, block_size=4)
        big = Config(workers=4, dimension=32, block_size=32)
        r_small = _run_ring(small, self.local)
        r_big = _run_ring(big, self.local)
        np.testing.assert_array_equal(r_small[0], r_big[0])

    def test_protocol_bad_magic(self):
        frame = proto.encode(proto.Message(
            proto.MSG_DATA, proto.PHASE_REDUCE_SCATTER, 0, 0, 1, 0, 0, 0,
            np.int16([1])))
        broken = b"XXXX" + frame[4:]
        with self.assertRaises(GradCompError) as ctx:
            proto.decode(broken)
        self.assertEqual(ctx.exception.code, E_PROTO)

    def test_protocol_phase_violation(self):
        # reduce_scatter 阶段收到 all_gather 阶段消息应报 E_PROTO。
        broker = LoopbackTransport(1)
        engine = RingEngine(0, 1, 8, broker)
        broker.send(0, proto.Message(
            proto.MSG_DATA, proto.PHASE_ALL_GATHER, 0, 0, 0, 0, 0, 0,
            np.int16([1])))
        with self.assertRaises(GradCompError) as ctx:
            engine._recv_ordered(
                0, proto.PHASE_REDUCE_SCATTER, 0, 0, 0, set())
        self.assertEqual(ctx.exception.code, E_PROTO)

    def test_duplicate_block_rejected(self):
        broker = LoopbackTransport(1)
        engine = RingEngine(0, 1, 8, broker)
        msg = proto.Message(
            proto.MSG_DATA, proto.PHASE_REDUCE_SCATTER, 0, 0, 0, 0, 0, 0,
            np.int16([1]))
        broker.send(0, msg)
        seen = set()
        first = engine._recv_ordered(
            0, proto.PHASE_REDUCE_SCATTER, 0, 0, 0, seen)
        np.testing.assert_array_equal(first, np.int16([1]))
        broker.send(0, msg)
        with self.assertRaises(GradCompError) as ctx:
            engine._recv_ordered(
                0, proto.PHASE_REDUCE_SCATTER, 0, 0, 0, seen)
        self.assertEqual(ctx.exception.code, E_PROTO)


if __name__ == "__main__":
    unittest.main()
