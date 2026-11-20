import os
import tempfile
import unittest

from gpu_scheduler.errors import E_CYCLE, E_LEDGER, E_MEMORY, E_QUOTA
from gpu_scheduler.injector import (
    INJECT_CYCLE, INJECT_MEMORY, INJECT_QUOTA, run_injected_validation,
    tamper_ledger,
)
from gpu_scheduler.ledger import Ledger
from gpu_scheduler.loader import load_declaration
from gpu_scheduler.scheduler import SchedulingSession

SPEC = os.path.join("samples", "specs")
PATHS = (
    os.path.join(SPEC, "session.json"),
    os.path.join(SPEC, "pools.json"),
    os.path.join(SPEC, "tasks.json"),
)


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()

    def test_memory_quota_cycle_injections_map_to_codes(self):
        cases = {
            INJECT_MEMORY: E_MEMORY,
            INJECT_QUOTA: E_QUOTA,
            INJECT_CYCLE: E_CYCLE,
        }
        for kind, expected in cases.items():
            code, detail = run_injected_validation(
                kind, PATHS[0], PATHS[1], PATHS[2], self.out)
            self.assertEqual(code, expected)
            self.assertTrue(detail["problems"])
            # Original samples must remain untouched and valid.
            load_declaration(*PATHS)

    def test_ledger_tampering_is_detected(self):
        work = tempfile.mkdtemp()
        ledger_path = os.path.join(work, "ledger.sqlite")
        session = SchedulingSession(
            load_declaration(*PATHS), ledger_path,
            os.path.join(work, "snapshots"), os.path.join(work, "backup"))
        session.run()
        session.close()
        tamper_ledger(ledger_path)
        with self.assertRaises(Exception) as ctx:
            Ledger(ledger_path).verify()
        self.assertEqual(ctx.exception.code, E_LEDGER)


if __name__ == "__main__":
    unittest.main()
