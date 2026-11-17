from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .support import base_payload, free_ports, parse_json_stdout, run_module, write_spec


class CliInjectionTests(unittest.TestCase):
    def _spec(self, temp: Path) -> Path:
        return write_spec(temp / "spec.json", base_payload(free_ports(2)))

    def test_injected_cycle_returns_e_rev_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            spec = self._spec(temp)
            workdir = temp / "runtime"
            process = run_module(
                [
                    "cluster_diag",
                    "validate",
                    "--spec",
                    str(spec),
                    "--workdir",
                    str(workdir),
                    "--inject-cycle",
                ]
            )
            self.assertEqual(process.returncode, 2)
            payload = parse_json_stdout(process)
            self.assertEqual(payload["error"]["code"], "E_REV")
            log_lines = (workdir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(json.loads(line)["code"] == "E_REV" for line in log_lines))

    def test_injected_port_conflict_returns_e_port_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            spec = self._spec(temp)
            workdir = temp / "runtime"
            process = run_module(
                [
                    "cluster_diag",
                    "run",
                    "--spec",
                    str(spec),
                    "--workdir",
                    str(workdir),
                    "--inject-port-conflict",
                ]
            )
            self.assertEqual(process.returncode, 3)
            payload = parse_json_stdout(process)
            self.assertEqual(payload["error"]["code"], "E_PORT")
            log_lines = (workdir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            codes = [json.loads(line)["code"] for line in log_lines]
            self.assertIn("INJECT_PORT_CONFLICT", codes)
            self.assertIn("E_PORT", codes)

    def test_injected_ledger_corruption_returns_e_ledger_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            spec = self._spec(temp)
            workdir = temp / "runtime"
            first = run_module(
                [
                    "cluster_diag",
                    "run",
                    "--spec",
                    str(spec),
                    "--workdir",
                    str(workdir),
                ]
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            second = run_module(
                [
                    "cluster_diag",
                    "verify",
                    "--spec",
                    str(spec),
                    "--workdir",
                    str(workdir),
                    "--inject-ledger-corruption",
                ]
            )
            self.assertEqual(second.returncode, 4)
            payload = parse_json_stdout(second)
            self.assertEqual(payload["error"]["code"], "E_LEDGER")
            log_lines = (workdir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            codes = [json.loads(line)["code"] for line in log_lines]
            self.assertIn("INJECT_LEDGER_CORRUPTION", codes)
            self.assertIn("E_LEDGER", codes)


if __name__ == "__main__":
    unittest.main()
