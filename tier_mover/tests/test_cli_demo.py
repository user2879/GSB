import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from tier_mover.cli import main as cli_main
from tier_mover.demo import run_demo


class CliDemoTests(unittest.TestCase):
    def test_demo_covers_all_business_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp) / "demo", seed=20261122)
            codes = report["all_business_codes"]
            self.assertEqual(codes["E_CAPACITY"], "E_CAPACITY")
            self.assertEqual(codes["E_INTEGRITY"], "E_INTEGRITY")
            self.assertEqual(codes["E_VERSION"], "E_VERSION")
            hp = report["scenarios"]["happy_path"]
            self.assertTrue(hp["plan_deterministic"])
            self.assertGreater(hp["batch_count"], 1)
            self.assertEqual(hp["replay"]["steps_compared"], hp["steps"])
            rb = report["scenarios"]["crash_rollback"]
            self.assertEqual(rb["recovery"]["unknown_objects"], 1)
            self.assertEqual(rb["recovery"]["rolled_back_objects"], 1)
            self.assertTrue(Path(tmp, "demo", "demo_report.json").is_file())

    def test_cli_init_plan_status_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = str(Path(tmp) / "ws")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli_main(["init", ws, "--seed", "123"])
            self.assertEqual(code, 0)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli_main(["plan", ws]), 0)
                plan_payload = json.loads(out.getvalue())
            self.assertIn("steps", plan_payload)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli_main(["run", ws]), 0)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli_main(["status", ws]), 0)
                view = json.loads(out.getvalue())
            self.assertEqual(view["active_spec_version"], 1)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli_main(["replay", ws]), 0)

    def test_cli_integrity_injection_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = str(Path(tmp) / "ws")
            cli_main(["init", ws, "--seed", "7"])
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = cli_main(["inject", ws, "integrity"])
            self.assertEqual(code, 2)
            payload = json.loads(err.getvalue())
            self.assertEqual(payload["code"], "E_INTEGRITY")
            # 异常后账本仍可打开
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["status", ws]), 0)

    def test_cli_crash_and_recover_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = str(Path(tmp) / "ws")
            cli_main(["init", ws, "--seed", "7"])
            plan_out = io.StringIO()
            with redirect_stdout(plan_out):
                cli_main(["plan", ws])
            first_key = json.loads(plan_out.getvalue())["steps"][0]["key"]
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                code = cli_main(["run", ws, "--crash-after-key", first_key])
            self.assertEqual(code, 137)
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(
                    cli_main(["recover", ws, "--mode", "rollback"]), 0)
                report = json.loads(out.getvalue())
            self.assertEqual(report["unknown_objects"], 1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["replay", ws]), 0)


if __name__ == "__main__":
    unittest.main()
