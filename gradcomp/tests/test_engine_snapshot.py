import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gradcomp.config import Config
from gradcomp.digest import array_digest, bytewise_equal
from gradcomp.engine import train
from gradcomp.errors import (E_DIVERGE, E_PORT, E_PROTO, E_REPLAY,
                             GradCompError)
from gradcomp.gui import build_headless_state
from gradcomp.replay import replay_resume
from gradcomp.snapshot import (list_snapshots, load_latest,
                               restore_state, write_snapshot)
from gradcomp.transport import TcpEndpoint


def _cfg(**overrides):
    base = dict(workers=4, dimension=32, seed=99, samples=32,
                scale_exponent=2, truncation="floor", block_size=8,
                transport="loopback", compressed=True, steps=2)
    base.update(overrides)
    return Config(**base)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gc-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compressed_consistent_and_deterministic(self):
        cfg = _cfg()
        r1 = train(cfg, workdir=self.tmp / "a")
        r2 = train(cfg, workdir=self.tmp / "b")
        self.assertTrue(all(s["consistent"] for s in r1.steps))
        self.assertEqual(r1.final_parameters_digest,
                         r2.final_parameters_digest)
        for s in r1.steps:
            self.assertEqual(len(set(s["worker_global_digests"])), 1)
            self.assertTrue(
                s["error_bounds"]["vector_bound_respected"])

    def test_uncompressed_has_no_quantization_error(self):
        cfg = _cfg(compressed=False, steps=1)
        report = train(cfg, workdir=self.tmp / "u")
        self.assertIn("note", report.steps[0]["error_bounds"])
        trunc = report.steps[0]["attribution"]["truncation_error_2norm"]
        self.assertEqual(trunc, 0.0)

    def test_attribution_independent_of_block_size_and_workers(self):
        c1 = _cfg(block_size=4)
        c2 = _cfg(block_size=16)
        a1 = train(c1, workdir=self.tmp / "bs1").steps[0]["attribution"]
        a2 = train(c2, workdir=self.tmp / "bs2").steps[0]["attribution"]
        self.assertEqual(
            [(t["element"], t["dominant_source"]) for t in a1["top_elements"]],
            [(t["element"], t["dominant_source"]) for t in a2["top_elements"]])
        c3 = _cfg(workers=2, dimension=32)
        a3 = train(c3, workdir=self.tmp / "w2").steps[0]["attribution"]
        self.assertEqual(set(a3["global_share"]),
                         {"compression_truncation", "accumulation_order"})

    def test_snapshot_restore_full_replace_and_replay(self):
        cfg = _cfg(steps=2)
        train(cfg, workdir=self.tmp / "snap")
        snap, pending = load_latest(self.tmp / "snap")
        self.assertEqual(pending, [])
        decision = restore_state(cfg, snap)
        self.assertFalse(decision["incremental_merge"])
        self.assertIn("parameters", decision["replaced_fields"])
        check = replay_resume(cfg, snap, extra_steps=2)
        self.assertTrue(check.matched)
        self.assertIsNone(check.first_mismatch)

    def test_pending_file_marks_unknown_step(self):
        cfg = _cfg(steps=1)
        write_snapshot(self.tmp / "crash", cfg,
                       np.zeros(cfg.dimension), 1, [], "", complete=False)
        listing = list_snapshots(self.tmp / "crash")
        self.assertEqual(len(listing["snapshots"]), 0)
        self.assertTrue(listing["pending_files"][0].endswith(".pending"))
        with self.assertRaises(GradCompError) as ctx:
            load_latest(self.tmp / "crash")
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

    def test_replay_rejects_scale_mismatch(self):
        cfg = _cfg(steps=1)
        train(cfg, workdir=self.tmp / "r")
        snap, _ = load_latest(self.tmp / "r")
        bad = _cfg(scale_exponent=3)
        with self.assertRaises(GradCompError) as ctx:
            replay_resume(bad, snap, extra_steps=1)
        self.assertEqual(ctx.exception.code, E_REPLAY)
        self.assertIn("scale_exponent", ctx.exception.field)

    def test_replay_rejects_shard_mismatch(self):
        cfg = _cfg(steps=1)
        train(cfg, workdir=self.tmp / "r2")
        snap, _ = load_latest(self.tmp / "r2")
        bad = _cfg(workers=2, dimension=32)
        with self.assertRaises(GradCompError) as ctx:
            replay_resume(bad, snap, extra_steps=1)
        self.assertEqual(ctx.exception.code, E_REPLAY)

    def test_headless_gui_state_stable(self):
        cfg = _cfg(steps=1)
        s1 = build_headless_state(cfg, self.tmp / "gui1")
        s2 = build_headless_state(cfg, self.tmp / "gui2")
        self.assertEqual([w["rank"] for w in s1["workers"]],
                         sorted(w["rank"] for w in s1["workers"]))
        self.assertEqual(s1["consistency"], s2["consistency"])
        self.assertEqual(s1["error_bounds"], s2["error_bounds"])


class TestTcpErrors(unittest.TestCase):
    def test_port_in_use_is_E_PORT(self):
        import socket
        cfg = _cfg(transport="tcp", base_port=39900)
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", cfg.base_port))
        holder.listen(1)
        try:
            endpoint = TcpEndpoint(0, cfg)
            with self.assertRaises(GradCompError) as ctx:
                endpoint._bind()
            self.assertEqual(ctx.exception.code, E_PORT)
        finally:
            holder.close()


if __name__ == "__main__":
    unittest.main()
