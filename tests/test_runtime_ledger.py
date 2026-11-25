import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from qos_diag.ledger import Ledger
from qos_diag.model import bundle_digest, load_bundle
from qos_diag.runtime import ConcurrencyGovernor, ShaperProcessManager
from qos_diag.samplegen import build_deterministic_bundle_data, write_sample_dir
from qos_diag.service import run_full_into_ledger
from qos_diag.errors import VerifyError
from qos_diag.verify import verify_ledger


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class RuntimeLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = build_deterministic_bundle_data(991)
        for shaper in data["shapers_raw"]["shapers"]:
            shaper["port"] = free_port()
        self.spec_dir = write_sample_dir(self.root / "spec", data)
        self.bundle = load_bundle(self.spec_dir)
        self.bundle["spec_digest"] = bundle_digest(self.bundle)
        self.ledger_path = self.root / "qos.db"
        self.run_dir = self.root / "run"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_governor_enforces_limits(self) -> None:
        governor = ConcurrencyGovernor(global_limit=2, pipeline_limit=1)
        current_global = 0
        current_pipeline: dict[str, int] = {}
        max_global = 0
        max_pipeline = 0
        lock = threading.Lock()

        def worker(pipeline_id: str) -> None:
            nonlocal current_global, max_global, max_pipeline
            with governor.acquire(pipeline_id):
                with lock:
                    current_global += 1
                    current_pipeline[pipeline_id] = current_pipeline.get(pipeline_id, 0) + 1
                    max_global = max(max_global, current_global)
                    max_pipeline = max(max_pipeline, current_pipeline[pipeline_id])
                time.sleep(0.01)
                with lock:
                    current_global -= 1
                    current_pipeline[pipeline_id] -= 1

        threads = [threading.Thread(target=worker, args=(f"p-{i % 3}",)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertLessEqual(max_global, 2)
        self.assertLessEqual(max_pipeline, 1)
        self.assertEqual(governor.max_global_seen, max_global)

    def test_real_subprocess_ledger_and_verify(self) -> None:
        ledger = Ledger.open(self.ledger_path, spec_digest=self.bundle["spec_digest"])
        manager = ShaperProcessManager(self.bundle, self.run_dir)
        try:
            manager.start_all()
            outcome = run_full_into_ledger(self.bundle, ledger, manager, batch_size=12)
            self.assertTrue(outcome["ok"], outcome.get("errors"))
        finally:
            manager.stop_all()
            ledger.close()
        verification = verify_ledger(self.spec_dir, self.ledger_path)
        self.assertTrue(verification["ok"])
        self.assertEqual(verification["verified_events"], 36)

        tampered = Ledger.open(self.ledger_path)
        tampered.conn.execute(
            "UPDATE judgments SET balance_after_decision_q = balance_after_decision_q + 1 "
            "WHERE revision = 1 AND event_seq = 0 AND stage = 'ingress'"
        )
        tampered.update_content_digest()
        tampered.close()
        with self.assertRaises(VerifyError) as context:
            verify_ledger(self.spec_dir, self.ledger_path)
        self.assertEqual(context.exception.code, "E_VERIFY")
        self.assertEqual(context.exception.event_seq, 0)
        self.assertEqual(context.exception.stage, "ingress")
        self.assertEqual(context.exception.field, "balance_after_decision_q")

    def test_digest_mismatch_requests_whole_replacement(self) -> None:
        ledger = Ledger.open(self.ledger_path, spec_digest=self.bundle["spec_digest"])
        ledger.close()
        import sqlite3

        conn = sqlite3.connect(self.ledger_path)
        conn.execute("UPDATE meta SET value='broken-version' WHERE key='format_version'")
        conn.commit()
        conn.close()
        replaced = Ledger.open(self.ledger_path)
        self.assertTrue(replaced.replaced)
        self.assertTrue(replaced.verify_stored_digest())
        replaced.close()


if __name__ == "__main__":
    unittest.main()
