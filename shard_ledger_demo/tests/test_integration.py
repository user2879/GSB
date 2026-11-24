import json
import os
import socket
import tempfile
import threading
import time
import unittest

from shard_ledger.demo import recover, run, verify
from shard_ledger.engine import Engine
from shard_ledger.fault_inject import inject_snapshot_mismatch
from shard_ledger.ledger import Ledger
from shard_ledger.logger import StructuredLogger
from shard_ledger.protocol import kill_orphans, send_batch, start_all, stop_all
from shard_ledger.spec import NodeSpec, load_events, load_manifest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE = os.path.join(ROOT, "samples", "manifest.json")
EVENTS = os.path.join(ROOT, "samples", "events.ndjson")


def _free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class EndToEndTest(unittest.TestCase):
    def test_full_run_recompute_recovery_and_replay(self):
        workdir = tempfile.mkdtemp(dir=os.path.join(ROOT, "runs"))
        report = run(SAMPLE, EVENTS, workdir)
        outcomes = {(row["shard_id"], row["generation"]): row for row in report["outcomes"]}
        self.assertEqual(outcomes[("S4", 1)]["status"], "completed")
        self.assertNotIn(("S4", 2), outcomes, "S4 must not be recomputed")
        self.assertEqual(outcomes[("S6", 1)]["status"], "cancelled")
        self.assertEqual(outcomes[("S1", 2)]["config_version"], 2)
        verify_report = verify(SAMPLE, EVENTS, workdir)
        self.assertIn("S4", verify_report["verified_shards"])

        crash_dir = tempfile.mkdtemp(dir=os.path.join(ROOT, "runs"))
        import subprocess
        crashed = subprocess.run(
            [
                os.sys.executable, "-m", "shard_ledger.demo", "run",
                "--manifest", SAMPLE, "--events", EVENTS,
                "--workdir", crash_dir, "--crash-dispatch", "S2",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(crashed.returncode, 9)
        time.sleep(0.3)
        kill_orphans(SAMPLE, load_manifest(SAMPLE))
        time.sleep(0.5)
        recovery = recover(SAMPLE, EVENTS, crash_dir)
        self.assertEqual([row[0] for row in recovery["unknown"]], ["S2"])
        self.assertEqual(recovery["rerun_unknown"], ["S2"])
        verify(SAMPLE, EVENTS, crash_dir)

    def test_three_way_constraints_queue_real_work(self):
        manifest = load_manifest(SAMPLE)
        workdir = tempfile.mkdtemp(dir=os.path.join(ROOT, "runs"))
        ledger = Ledger(os.path.join(workdir, "ledger.sqlite"))
        logger = StructuredLogger(os.path.join(workdir, "log.jsonl"))
        engine = Engine(manifest, ledger, logger)
        processes = start_all(SAMPLE, manifest.nodes)
        try:
            engine.start()
            observed_global_units = []
            observed_n2 = []
            for sid in ("S1", "S2"):
                engine.submit(sid, 0)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and not engine.is_done():
                engine._pump_until_blocked()
                with engine.lock:
                    observed_global_units.append(engine.leases.global_units())
                    observed_n2.append(engine.leases.active_count("N2"))
                time.sleep(0.01)
            engine.wait_idle(timeout=8)
            self.assertTrue(engine.is_done())
            self.assertLessEqual(max(observed_global_units), manifest.global_budget)
            self.assertLessEqual(max(observed_n2), manifest.nodes["N2"].lease_limit)
        finally:
            engine.shutdown()
            stop_all(processes)
            ledger.close()
            logger.close()

    def test_protocol_errors_timeout_field_and_port(self):
        port = _free_port()
        node = NodeSpec("NX", "127.0.0.1", port, 1, 1, 7, 1)

        with self.assertRaises(Exception) as ctx:
            send_batch(node, "S1", 1, timeout_ms=300)
        self.assertEqual(ctx.exception.code, "E_PORT")

        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)

        def silent_accept():
            conn, _ = server.accept()
            time.sleep(0.6)
            conn.close()

        thread = threading.Thread(target=silent_accept, daemon=True)
        thread.start()
        with self.assertRaises(Exception) as ctx:
            send_batch(node, "S1", 1, timeout_ms=150)
        self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        thread.join(timeout=2)
        server.close()

        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)

        def bad_reply():
            conn, _ = server.accept()
            conn.recv(4096)
            conn.sendall(b'{"op":"handshake_ack","node_id":"NX"}\n')
            conn.recv(4096)
            conn.sendall(b'{"node_id":"NX"}\n')
            conn.close()

        thread = threading.Thread(target=bad_reply, daemon=True)
        thread.start()
        with self.assertRaises(Exception) as ctx:
            send_batch(node, "S1", 1, timeout_ms=500)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        thread.join(timeout=2)
        server.close()

    def test_replay_mismatch_locates_shard_and_field(self):
        workdir = tempfile.mkdtemp(dir=os.path.join(ROOT, "runs"))
        run(SAMPLE, EVENTS, workdir)
        time.sleep(0.2)
        ledger_path = os.path.join(workdir, "ledger.sqlite")
        import sqlite3
        conn = sqlite3.connect(ledger_path)
        row = conn.execute(
            "SELECT summary FROM outcomes WHERE status='completed' AND shard_id='S4'"
        ).fetchone()
        summary = json.loads(row[0])
        summary[0]["summary"]["total_sum"] += 999
        conn.execute(
            "UPDATE outcomes SET summary=? WHERE status='completed' AND shard_id='S4'",
            (json.dumps(summary, sort_keys=True, separators=(",", ":")),),
        )
        conn.commit()
        conn.close()
        from shard_ledger.errors import ReplayError
        with self.assertRaises(ReplayError) as ctx:
            verify(SAMPLE, EVENTS, workdir)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.entity, "S4")
        self.assertEqual(ctx.exception.field, "summary")

    def test_snapshot_tamper_detected(self):
        workdir = tempfile.mkdtemp(dir=os.path.join(ROOT, "runs"))
        run(SAMPLE, EVENTS, workdir)
        time.sleep(0.2)
        ledger_path = os.path.join(workdir, "ledger.sqlite")
        inject_snapshot_mismatch(ledger_path)
        with self.assertRaises(Exception) as ctx:
            Ledger(ledger_path)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
