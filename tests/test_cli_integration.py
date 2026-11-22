from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from .helpers import SAMPLES, parse_stdout_json, read_jsonl, run_cli, run_sample, write_temp_config


class CliIntegrationTests(unittest.TestCase):
    def test_run_verify_and_injected_fault_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            config, result = run_sample(temp_root)
            self.assertEqual(result["status"], "complete")
            out = temp_root / "out"
            verify = run_cli(
                "verify",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--model-digest",
                str(out / "model_digest.txt"),
                "--conclusions",
                str(out / "conclusions.jsonl"),
                "--group-conclusions",
                str(out / "group_conclusions.jsonl"),
                "--conflicts",
                str(out / "conflicts.jsonl"),
                "--stats",
                str(out / "stats.json"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(verify.returncode, 0, verify.stdout)
            self.assertTrue(parse_stdout_json(verify)["byte_identical"])

            divergent = run_cli(
                "verify",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--model-digest",
                str(out / "model_digest.txt"),
                "--conclusions",
                str(out / "conclusions.jsonl"),
                "--group-conclusions",
                str(out / "group_conclusions.jsonl"),
                "--conflicts",
                str(out / "conflicts.jsonl"),
                "--stats",
                str(out / "stats.json"),
                "--inject-divergence",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(divergent.returncode, 6)
            error = parse_stdout_json(divergent)["error"]
            self.assertEqual(error["code"], "E_DIVERGE")
            self.assertEqual(error["details"]["first_mismatch"]["message_id"], "archive:1")
            self.assertEqual(error["details"]["origin"], "semantic_derivation")

            wrong_digest = temp_root / "wrong-model-digest.txt"
            wrong_digest.write_text("0" * 64 + "\n", encoding="utf-8")
            format_divergent = run_cli(
                "verify",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--model-digest",
                str(wrong_digest),
                "--conclusions",
                str(out / "conclusions.jsonl"),
                "--group-conclusions",
                str(out / "group_conclusions.jsonl"),
                "--conflicts",
                str(out / "conflicts.jsonl"),
                "--stats",
                str(out / "stats.json"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(format_divergent.returncode, 6)
            format_divergence = parse_stdout_json(format_divergent)["error"]
            self.assertEqual(format_divergence["code"], "E_DIVERGE")
            self.assertEqual(format_divergence["details"]["origin"], "format_normalization")

            format_injected = run_cli(
                "run",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--inject-format",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(format_injected.returncode, 3)
            format_error = parse_stdout_json(format_injected)["error"]
            self.assertEqual(format_error["code"], "E_FORMAT")
            self.assertEqual(format_error["details"]["line"], 1)
            self.assertEqual(format_error["details"]["field"], "record.retry_count")

            duplicate_injected = run_cli(
                "run",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--inject-dupkey",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(duplicate_injected.returncode, 4)
            self.assertEqual(parse_stdout_json(duplicate_injected)["error"]["code"], "E_DUPKEY")

            log_rows = read_jsonl(temp_root / "logs" / "structured.jsonl")
            log_codes = {row.get("code") for row in log_rows if row.get("event") == "command_error"}
            self.assertTrue({"E_FORMAT", "E_DUPKEY", "E_DIVERGE"}.issubset(log_codes))

    def test_evolution_snapshot_restore_and_whole_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            base_config = write_temp_config(temp_root)
            base_run = run_cli(
                "run",
                "--config",
                str(base_config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(base_run.returncode, 0, base_run.stdout)
            base = parse_stdout_json(base_run)
            self.assertEqual(base["parameter_version"], 1)
            saved = run_cli(
                "snapshot",
                "save",
                "--config",
                str(base_config),
                "--name",
                "baseline",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(saved.returncode, 0, saved.stdout)

            evolved_config = write_temp_config(temp_root, source_name="session_evolved.json", out="out-evolved")
            evolved_run = run_cli(
                "run",
                "--config",
                str(evolved_config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(evolved_run.returncode, 0, evolved_run.stdout)
            evolved = parse_stdout_json(evolved_run)
            self.assertEqual(evolved["parameter_version"], 2)
            self.assertTrue(evolved["parameter_changed"])
            self.assertEqual(evolved["view_diff_counts"]["changed"], 1)
            artifact = json.loads((temp_root / "out-evolved" / "artifact.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["view_diff"][0]["message_id"], "orders:4")
            self.assertIn("basis", artifact["view_diff"][0]["changed_fields"])

            connection = sqlite3.connect(str(temp_root / "state" / "ledger.sqlite3"))
            try:
                stale = connection.execute(
                    "SELECT COUNT(*) FROM batches WHERE parameter_version=1 AND stale=1"
                ).fetchone()[0]
                current = connection.execute(
                    "SELECT COUNT(*) FROM batches WHERE parameter_version=2 AND stale=0"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(stale, 4)
            self.assertEqual(current, 4)

            restored = run_cli(
                "snapshot",
                "restore",
                "--config",
                str(base_config),
                "--name",
                "baseline",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(restored.returncode, 0, restored.stdout)
            restore = parse_stdout_json(restored)
            self.assertEqual(restore["replacement_mode"], "wholesale_copy_no_merge")
            self.assertTrue(restore["replaced_due_to_mismatch"])
            self.assertIn("content_digest", restore["mismatch_basis"])
            self.assertTrue(Path(restore["previous_ledger_backup"]).exists())

            resaved = run_cli(
                "snapshot",
                "save",
                "--config",
                str(base_config),
                "--name",
                "baseline",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(resaved.returncode, 0, resaved.stdout)
            resave = parse_stdout_json(resaved)
            self.assertIsNotNone(resave["previous_snapshot_backup"])
            backup_path = Path(resave["previous_snapshot_backup"])
            self.assertTrue(backup_path.exists())
            if hasattr(Path, "stat"):
                self.assertEqual(backup_path.stat().st_mode & 0o222, 0)

            recover = run_cli(
                "recover",
                "--config",
                str(base_config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(recover.returncode, 0, recover.stdout)
            recovered = parse_stdout_json(recover)
            self.assertEqual(recovered["parameter_version"], 1)
            self.assertTrue(recovered["replay_ok"])

            injected = run_cli(
                "snapshot",
                "restore",
                "--config",
                str(base_config),
                "--name",
                "baseline",
                "--inject-digest",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(injected.returncode, 5)
            self.assertEqual(parse_stdout_json(injected)["error"]["code"], "E_SNAPSHOT")
            self.assertIn(
                "E_SNAPSHOT",
                {row.get("code") for row in read_jsonl(temp_root / "logs" / "structured.jsonl")},
            )

    def test_force_termination_recovery_unknown_and_e_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            config = write_temp_config(temp_root)
            crash_ledger = temp_root / "state" / "crash.sqlite3"
            crashed = run_cli(
                "run",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--ledger",
                str(crash_ledger),
                "--out",
                str(temp_root / "out-crash"),
                "--crash-after-topic",
                "1",
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(crashed.returncode, 91)
            self.assertFalse((temp_root / "out-crash" / "artifact.json").exists())

            recovered = run_cli(
                "recover",
                "--ledger",
                str(crash_ledger),
                "--records",
                str(SAMPLES / "records.jsonl"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(recovered.returncode, 0, recovered.stdout)
            recovered_json = parse_stdout_json(recovered)
            self.assertEqual(recovered_json["status"], "incomplete")
            self.assertEqual(recovered_json["committed_topics"], ["archive"])
            self.assertEqual(recovered_json["rerun_required_topics"], ["audit", "orders", "payments"])
            self.assertFalse(recovered_json["exit_report_available"])
            self.assertTrue(recovered_json["unknown_messages"])

            rerun = run_cli(
                "run",
                "--config",
                str(config),
                "--records",
                str(SAMPLES / "records.jsonl"),
                "--ledger",
                str(crash_ledger),
                "--out",
                str(temp_root / "out-crash"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(rerun.returncode, 0, rerun.stdout)
            complete = parse_stdout_json(
                run_cli(
                    "recover",
                    "--ledger",
                    str(crash_ledger),
                    "--records",
                    str(SAMPLES / "records.jsonl"),
                )
            )
            self.assertEqual(complete["status"], "complete")
            self.assertTrue(complete["exit_report_available"])

            bad_ledger = temp_root / "state" / "bad-replay.sqlite3"
            shutil.copy2(crash_ledger, bad_ledger)
            connection = sqlite3.connect(str(bad_ledger))
            try:
                row = connection.execute(
                    "SELECT conclusion_json FROM conclusions ORDER BY topic, message_seq, message_id LIMIT 1"
                ).fetchone()
                conclusion = json.loads(row[0])
                conclusion["verdict"] = "at_most_once"
                connection.execute(
                    "UPDATE conclusions SET verdict=?, conclusion_json=? WHERE topic=? AND message_seq=? AND message_id=?",
                    (
                        "at_most_once",
                        json.dumps(conclusion, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        conclusion["topic"],
                        conclusion["message_seq"],
                        conclusion["message_id"],
                    ),
                )
                connection.commit()
            finally:
                connection.close()
            replay = run_cli(
                "recover",
                "--ledger",
                str(bad_ledger),
                "--records",
                str(SAMPLES / "records.jsonl"),
                log_file=temp_root / "logs" / "structured.jsonl",
            )
            self.assertEqual(replay.returncode, 7)
            error = parse_stdout_json(replay)["error"]
            self.assertEqual(error["code"], "E_REPLAY")
            self.assertEqual(error["details"]["first_mismatch"]["message_id"], "archive:1")
            self.assertEqual(error["details"]["first_mismatch"]["field"], "verdict")
            self.assertIn("basis", error["details"])


if __name__ == "__main__":
    unittest.main()
