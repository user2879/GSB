import tempfile
import unittest
from pathlib import Path

from chaoskit.engine import ChaosEngine
from chaoskit.errors import TargetError
from chaoskit.spec import parse_spec

ROOT = Path(__file__).resolve().parent.parent


def quick_spec():
    return parse_spec({
        "version": 1,
        "experiment_id": "engine-it",
        "seed": 777,
        "budget": {"max_injections": 6, "max_parallel": 1},
        "observation": {"sample_interval_ms": 10, "min_samples": 3},
        "services": [
            {"service_id": "svc_a", "host": "127.0.0.1", "data_port": 18191,
             "proxy_port": 19191, "fixed_cost_ms": 8, "base_error_bps": 0},
            {"service_id": "svc_b", "host": "127.0.0.1", "data_port": 18192,
             "proxy_port": 19192, "fixed_cost_ms": 10, "base_error_bps": 0},
        ],
        "injections": [
            {"injection_id": "d1", "type": "network_delay", "target_id": "svc_a",
             "duration_ms": 80, "params": {"delay_ms": 50}},
            {"injection_id": "e1", "type": "error_rate", "target_id": "svc_a",
             "duration_ms": 80, "params": {"error_bps": 10000}},
            {"injection_id": "s1", "type": "suspend", "target_id": "svc_b",
             "duration_ms": 80, "params": {}},
            {"injection_id": "k1", "type": "terminate", "target_id": "svc_b",
             "duration_ms": 80, "params": {}},
        ],
    })


class EngineIntegrationTest(unittest.TestCase):
    def test_full_run_states_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = quick_spec()
            engine = ChaosEngine(spec, tmp)
            try:
                engine.setup()
                report = engine.run()
            finally:
                engine.teardown()

            self.assertEqual(report["recovered_committed"], 4)
            self.assertEqual(report["remaining_budget"], spec.budget.max_injections - 4)

            conn = engine.ledger.conn if engine.ledger is not None else None
            # ledger was closed in teardown; reopen read-only for assertions
            import sqlite3
            db = sqlite3.connect(str(Path(tmp) / "ledger.sqlite3"))
            db.row_factory = sqlite3.Row
            try:
                states = {
                    row["injection_type"]: row["observed_state"]
                    for row in db.execute("SELECT injection_type, observed_state FROM injections")
                }
                self.assertEqual(states["suspend"], "suspended")
                self.assertEqual(states["terminate"], "terminated")
                self.assertEqual(states["network_delay"], "running")

                # request records are the persisted source of truth
                total_requests = db.execute("SELECT COUNT(*) AS c FROM request_records").fetchone()["c"]
                # 3 phases x 3 samples x 4 injections
                self.assertEqual(total_requests, 3 * 3 * 4)

                # error_rate injection: every active request must fail at 10000bps
                idx = db.execute(
                    "SELECT idx FROM injections WHERE injection_type='error_rate'"
                ).fetchone()["idx"]
                active_fail = db.execute(
                    "SELECT COUNT(*) AS c FROM request_records WHERE injection_idx=? "
                    "AND phase='active' AND success=0", (idx,)
                ).fetchone()["c"]
                self.assertEqual(active_fail, 3)
            finally:
                db.close()

    def test_resume_skips_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = quick_spec()
            first = ChaosEngine(spec, tmp)
            try:
                first.setup()
                first.run()
            finally:
                first.teardown()

            second = ChaosEngine(spec, tmp)
            try:
                second.setup()
                events = []
                second.on_event = events.append
                report = second.run()
            finally:
                second.teardown()
            self.assertEqual(report["recovered_committed"], 4)
            self.assertTrue(any(e["event"] == "skip_committed" for e in events))

    def test_port_conflict_is_target_error(self):
        import socket
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 18191))
        blocker.listen(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                engine = ChaosEngine(quick_spec(), tmp)
                with self.assertRaises(TargetError) as ctx:
                    engine.setup()
                self.assertEqual(ctx.exception.code, "E_TARGET")
                engine.teardown()
        finally:
            blocker.close()


if __name__ == "__main__":
    unittest.main()
