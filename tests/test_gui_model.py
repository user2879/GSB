from __future__ import annotations

import sqlite3
import unittest

from tlschain.gui_model import build_diagnostics_model, model_digest
from tlschain.session import build_full_session
from tlschain.spec import load_spec

from tests.helpers import (
    CERT_PATH,
    HANDSHAKE_PATH,
    free_loopback_port,
    parse_last_json,
    run_cli,
    temporary_directory,
)


class DiagnosticsModelTests(unittest.TestCase):
    def test_model_is_deterministic_and_read_only(self) -> None:
        directory = temporary_directory("tlschain-diag-")
        ledger_path = directory / "ledger.sqlite"
        run = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(ledger_path),
                "--port",
                str(free_loopback_port()),
                "--reset",
            ]
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        with sqlite3.connect(ledger_path) as connection:
            before_digest = connection.execute(
                "SELECT content_digest FROM ledger_info WHERE id = 1"
            ).fetchone()[0]
            before_batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        spec = load_spec(CERT_PATH, HANDSHAKE_PATH)
        session = build_full_session(spec)
        first = build_diagnostics_model(spec, session, ledger_path)
        second = build_diagnostics_model(spec, session, ledger_path)
        self.assertEqual(model_digest(first), model_digest(second))
        self.assertEqual(first, second)
        self.assertEqual(first["render_order"]["verification"], "message_seq_ascending_then_chain_step")
        self.assertEqual(first["ledger"]["progress_cursor"], 8)
        self.assertEqual(first["unknown_message_seqs"], [])
        self.assertEqual(first["error_decomposition"]["modulus_quantization"], 48)
        self.assertEqual(first["error_decomposition"]["time_window"], 48)
        self.assertEqual(first["error_decomposition"]["path_length"], 8)
        self.assertEqual(first["error_decomposition"]["total"], 104)
        with sqlite3.connect(ledger_path) as connection:
            after_digest = connection.execute(
                "SELECT content_digest FROM ledger_info WHERE id = 1"
            ).fetchone()[0]
            after_batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        self.assertEqual(before_digest, after_digest)
        self.assertEqual(before_batches, after_batches)

    def test_cli_diagnostics_refresh_is_byte_identical(self) -> None:
        directory = temporary_directory("tlschain-diag-cli-")
        ledger_path = directory / "ledger.sqlite"
        run = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(ledger_path),
                "--port",
                str(free_loopback_port()),
                "--reset",
            ]
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        command = [
            "diagnostics",
            "--certs",
            str(CERT_PATH),
            "--handshake",
            str(HANDSHAKE_PATH),
            "--ledger",
            str(ledger_path),
        ]
        first = run_cli(command)
        second = run_cli(command)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0)
        self.assertEqual(first.stdout, second.stdout)
        self.assertTrue(parse_last_json(first.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
