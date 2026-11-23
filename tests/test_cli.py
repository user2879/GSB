from __future__ import annotations

import json
import socket
import subprocess
import sys
import unittest
from pathlib import Path

from tests.helpers import Workspace, base_spec, reserve_ports, sample_requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gateway_diag", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Workspace()

    def tearDown(self) -> None:
        self.workspace.close()

    def test_validation_error_has_e_spec_exit_code_and_first_field(self) -> None:
        raw = base_spec([45081, 45082])
        raw["backends"][1]["id"] = raw["backends"][0]["id"]
        path = self.workspace.write_spec(raw)
        completed = run_cli("validate", "--spec", str(path))
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["code"], "E_SPEC")
        self.assertEqual(payload["entry"], "backends[1]")
        self.assertEqual(payload["field"], "id")

    def test_run_preview_and_verify_cli_paths(self) -> None:
        ports = reserve_ports(2)
        raw = base_spec(ports)
        spec_path = self.workspace.write_spec(raw)
        requests_path = self.workspace.write_requests(sample_requests(4))
        run = run_cli(
            "run",
            "--spec",
            str(spec_path),
            "--requests",
            str(requests_path),
            "--db",
            str(self.workspace.db),
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        run_payload = json.loads(run.stdout)
        self.assertTrue(run_payload["ok"])
        self.assertEqual(run_payload["newly_committed"], run_payload["total_batches"])

        preview = run_cli("preview", "--db", str(self.workspace.db), "--request-id", "1")
        self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr)
        preview_payload = json.loads(preview.stdout)
        self.assertEqual(preview_payload["entry"]["request_id"], 1)
        self.assertEqual(len(preview_payload["entry"]["output_digest"]), 64)

        verify = run_cli(
            "verify",
            "--mode",
            "both",
            "--spec",
            str(spec_path),
            "--requests",
            str(requests_path),
            "--db",
            str(self.workspace.db),
        )
        self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
        verify_payload = json.loads(verify.stdout)
        self.assertTrue(verify_payload["authority"]["consistent"])
        self.assertTrue(verify_payload["replay"]["consistent"])

    def test_occupied_port_cli_reports_e_port(self) -> None:
        port = reserve_ports(1)[0]
        occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupant.bind(("127.0.0.1", port))
        occupant.listen(1)
        try:
            raw = base_spec([port, reserve_ports(1)[0]])
            spec_path = self.workspace.write_spec(raw)
            requests_path = self.workspace.write_requests(sample_requests(1))
            completed = run_cli(
                "run",
                "--spec",
                str(spec_path),
                "--requests",
                str(requests_path),
                "--db",
                str(self.workspace.db),
            )
            self.assertEqual(completed.returncode, 3, completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["code"], "E_PORT")
            self.assertEqual(payload["backend_id"], "stable")
            self.assertEqual(payload["port"], port)
        finally:
            occupant.close()


if __name__ == "__main__":
    unittest.main()
