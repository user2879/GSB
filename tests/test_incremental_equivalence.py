import json
import os
import shutil
import tempfile
import unittest

from receipt_ledger.engine import LedgerEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IncrementalEquivalenceTests(unittest.TestCase):
    def test_stepwise_command_cursor_matches_single_final_run(self):
        with open(os.path.join(ROOT, "samples", "commands.json"), encoding="utf-8") as handle:
            commands = json.load(handle)["commands"]

        step_dir = os.path.join(tempfile.gettempdir(), "receipt-ledger-test", "stepwise")
        shutil.rmtree(step_dir, ignore_errors=True)
        os.makedirs(step_dir, exist_ok=True)

        step_reports = []
        for index in range(1, len(commands) + 1):
            prefix_path = os.path.join(step_dir, f"commands-{index}.json")
            with open(prefix_path, "w", encoding="utf-8") as handle:
                json.dump({"commands": commands[:index]}, handle)
            report = LedgerEngine(
                os.path.join(ROOT, "samples", "workers.json"),
                os.path.join(ROOT, "samples"),
                prefix_path,
                step_dir,
            ).run_authoritative()
            step_reports.append(report)

        full_dir = os.path.join(tempfile.gettempdir(), "receipt-ledger-test", "full-final")
        shutil.rmtree(full_dir, ignore_errors=True)
        os.makedirs(full_dir, exist_ok=True)
        full = LedgerEngine(
            os.path.join(ROOT, "samples", "workers.json"),
            os.path.join(ROOT, "samples"),
            os.path.join(ROOT, "samples", "commands.json"),
            full_dir,
        ).run_authoritative()

        final_step = step_reports[-1]
        for field in (
            "ledger",
            "resources",
            "failures",
            "attribution_summary",
            "totals",
            "conversion_log",
        ):
            self.assertEqual(final_step[field], full[field], field)

        self.assertEqual(
            [item for item in final_step["todo_queue"]],
            [item for item in full["todo_queue"]],
        )
        self.assertEqual(
            final_step["snapshot_manifest"]["command_cursor"],
            full["snapshot_manifest"]["command_cursor"],
        )


if __name__ == "__main__":
    unittest.main()
