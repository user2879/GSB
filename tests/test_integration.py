from __future__ import annotations

import json
import sqlite3
import socket
import unittest
from pathlib import Path

from tlschain.canonical import canonical_text, digest_hex

from tests.helpers import (
    CERT_PATH,
    HANDSHAKE_PATH,
    free_loopback_port,
    parse_last_json,
    run_cli,
    temporary_directory,
)


class LoopbackIntegrationTests(unittest.TestCase):
    def _run_complete(self, directory: Path, port: int) -> dict:
        result = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "ledger.sqlite"),
                "--port",
                str(port),
                "--reset",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return parse_last_json(result.stdout)

    def test_complete_run_and_recovery_replay(self) -> None:
        directory = temporary_directory("tlschain-int-full-")
        report = self._run_complete(directory, free_loopback_port())
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["progress_cursor"], 8)
        self.assertEqual(report["decision_counts"], {"PASS": 5, "FAIL": 1, "INDETERMINATE": 2})
        self.assertEqual([batch["through_seq"] for batch in report["committed_batches"]], [2, 4, 6, 8])
        self.assertTrue((directory / "run-report.json").exists())
        recovered = run_cli(
            [
                "recover",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "ledger.sqlite"),
                "--port",
                str(free_loopback_port()),
            ]
        )
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        recovery_report = parse_last_json(recovered.stdout)
        self.assertEqual(recovery_report["status"], "COMPLETE")
        self.assertEqual(recovery_report["replayed_message_seqs"], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(recovery_report["final_replay"]["verified_seqs"], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_port_timeout_and_missing_reply_errors(self) -> None:
        directory = temporary_directory("tlschain-int-errors-")
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", free_loopback_port()))
        occupied.listen(1)
        occupied_port = occupied.getsockname()[1]
        try:
            result = run_cli(
                [
                    "run",
                    "--certs",
                    str(CERT_PATH),
                    "--handshake",
                    str(HANDSHAKE_PATH),
                    "--ledger",
                    str(directory / "port.sqlite"),
                    "--port",
                    str(occupied_port),
                    "--reset",
                ]
            )
        finally:
            occupied.close()
        self.assertEqual(result.returncode, 2)
        port_error = parse_last_json(result.stdout)
        self.assertEqual(port_error["code"], "E_PORT")
        self.assertEqual(port_error["field"], "socket.port")

        result = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "timeout.sqlite"),
                "--port",
                str(free_loopback_port()),
                "--reset",
                "--drop-client-seq",
                "3",
            ]
        )
        self.assertEqual(result.returncode, 2)
        timeout_error = parse_last_json(result.stdout)
        self.assertEqual(timeout_error["code"], "E_TIMEOUT")
        self.assertEqual(timeout_error["message_seq"], 5)
        self.assertEqual(timeout_error["details"]["timeout_ms"], 750)

        result = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "reply.sqlite"),
                "--port",
                str(free_loopback_port()),
                "--reset",
                "--omit-reply-field",
                "seq",
            ]
        )
        self.assertEqual(result.returncode, 2)
        reply_error = parse_last_json(result.stdout)
        self.assertEqual(reply_error["code"], "E_REPLY")
        self.assertEqual(reply_error["message_seq"], 2)
        self.assertEqual(reply_error["field"], "reply.seq")

    def test_graceful_signal_report_and_recovery(self) -> None:
        directory = temporary_directory("tlschain-int-graceful-")
        result = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "ledger.sqlite"),
                "--port",
                str(free_loopback_port()),
                "--reset",
                "--terminate-after",
                "4",
                "--termination",
                "graceful",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = parse_last_json(result.stdout)
        self.assertEqual(report["status"], "GRACEFUL_EXIT")
        self.assertEqual(report["signal"], "SIGINT")
        self.assertEqual(report["progress_cursor"], 4)
        self.assertEqual(report["remaining_message_seqs"], [5, 6, 7, 8])
        self.assertTrue((directory / "exit-report.json").exists())
        recovered = run_cli(
            [
                "recover",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "ledger.sqlite"),
                "--port",
                str(free_loopback_port()),
            ]
        )
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        recovery_report = parse_last_json(recovered.stdout)
        self.assertEqual(recovery_report["initial_unknown_seqs"], [5, 6, 7, 8])
        self.assertEqual(recovery_report["progress_cursor"], 8)

    def test_hard_termination_has_no_exit_report_and_recovers(self) -> None:
        directory = temporary_directory("tlschain-int-hard-")
        result = run_cli(
            [
                "run",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "ledger.sqlite"),
                "--port",
                str(free_loopback_port()),
                "--reset",
                "--terminate-after",
                "4",
                "--termination",
                "hard",
            ]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((directory / "exit-report.json").exists())
        self.assertFalse((directory / "run-report.json").exists())
        with sqlite3.connect(directory / "ledger.sqlite") as connection:
            cursor = connection.execute("SELECT MAX(progress_cursor) FROM batches").fetchone()[0]
            batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        self.assertEqual(cursor, 4)
        self.assertEqual(batches, 2)
        recovered = run_cli(
            [
                "recover",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(directory / "ledger.sqlite"),
                "--port",
                str(free_loopback_port()),
            ]
        )
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        report = parse_last_json(recovered.stdout)
        self.assertEqual(report["initial_unknown_seqs"], [5, 6, 7, 8])
        self.assertEqual(report["progress_cursor"], 8)

    def test_replay_mismatch_reports_first_sequence_and_field(self) -> None:
        directory = temporary_directory("tlschain-int-replay-")
        self._run_complete(directory, free_loopback_port())
        path = directory / "ledger.sqlite"
        with sqlite3.connect(path) as connection:
            row = connection.execute("SELECT record_json FROM messages WHERE seq = 4").fetchone()
            record = json.loads(row[0])
            record["decision"] = "PASS"
            connection.execute(
                "UPDATE messages SET record_json = ?, record_digest = ? WHERE seq = 4",
                (canonical_text(record), digest_hex(record)),
            )
            connection.commit()
        result = run_cli(
            [
                "recover",
                "--certs",
                str(CERT_PATH),
                "--handshake",
                str(HANDSHAKE_PATH),
                "--ledger",
                str(path),
                "--port",
                str(free_loopback_port()),
            ]
        )
        self.assertEqual(result.returncode, 2)
        error = parse_last_json(result.stdout)
        self.assertEqual(error["code"], "E_REPLAY")
        self.assertEqual(error["message_seq"], 4)
        self.assertEqual(error["cert_id"], "leaf-expired")
        self.assertIn("decision", error["field"])


if __name__ == "__main__":
    unittest.main()
