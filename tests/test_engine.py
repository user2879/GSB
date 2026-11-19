from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from backup_engine.canonical import canonical_json_bytes
from backup_engine.engine import BackupEngine
from backup_engine.samples import make_entry, make_snapshot


ROOT = Path(__file__).resolve().parents[1]


class EngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_snapshot(self, name: str, snapshot: dict) -> Path:
        path = self.root / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(snapshot))
        return path

    def sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def codes(report: dict) -> set[str]:
        return {item["code"] for item in report.get("errors", [])}

    @staticmethod
    def entry_paths(report: dict) -> list[str]:
        state = report.get("state") or {}
        return [item["path"] for item in state.get("entries", [])]

    def test_entry_rejections_continue_and_input_is_immutable(self) -> None:
        good = make_entry("good.txt", "good\n")
        bad_algo = make_entry("badalgo.txt", "bad\n")
        bad_algo["digest"] = "md5:" + ("0" * 32)
        bad_length = make_entry("badlen.txt", "length\n")
        bad_length["length"] = -1
        marked = make_entry("marked.txt", "marked\n", tampered=True)
        snapshot, _ = make_snapshot(
            1,
            [
                good,
                make_entry("../escape.txt", "escape\n"),
                make_entry("good.txt", "duplicate\n"),
                bad_algo,
                bad_length,
                marked,
            ],
        )
        path = self.write_snapshot("entry-reject.json", snapshot)
        before = self.sha256(path)
        engine = BackupEngine(self.root / "ws")

        imported = engine.import_snapshots([path])
        self.assertEqual(imported["exit_code"], 1)
        recovered = engine.recover()
        self.assertEqual(recovered["exit_code"], 1)
        self.assertTrue(
            {"E_PATH", "E_DUPLICATE", "E_ALGORITHM", "E_LENGTH", "E_TAMPER"}
            <= self.codes(recovered)
        )
        for finding in recovered["errors"]:
            self.assertEqual(finding.get("version"), 1)
            self.assertIsInstance(finding.get("entry_index"), int)
        status = engine.status()
        self.assertEqual(self.entry_paths(status), ["good.txt"])
        self.assertEqual(self.sha256(path), before)

    def test_bad_snapshot_does_not_abort_other_recovery_requests(self) -> None:
        valid, _ = make_snapshot(1, [make_entry("good.txt", "good\n")])
        valid_path = self.write_snapshot("batch-good.json", valid)
        bad_path = self.root / "inputs" / "batch-bad.json"
        bad_path.write_bytes(b"{not-json")
        bad_hash = self.sha256(bad_path)
        engine = BackupEngine(self.root / "ws")
        report = engine.recover([bad_path, valid_path])
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(self.codes(report), {"E_JSON"})
        self.assertEqual(self.entry_paths(engine.status()), ["good.txt"])
        self.assertEqual(self.sha256(bad_path), bad_hash)

    def test_chain_break_and_fingerprint_mismatch(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        broken, _ = make_snapshot(
            3,
            [make_entry("b.txt", "three\n")],
            previous_fingerprint=fp1,
            previous_version=2,
        )
        broken_path = self.write_snapshot("broken-01.json", v1)
        broken_path3 = self.write_snapshot("broken-03.json", broken)
        broken_engine = BackupEngine(self.root / "broken")
        broken_report = broken_engine.recover([broken_path3, broken_path])
        self.assertIn("E_CHAIN_BREAK", self.codes(broken_report))
        self.assertEqual(
            [item["version"] for item in broken_report["effective_snapshots"]], [1]
        )
        chain_error = next(
            item for item in broken_report["errors"] if item["code"] == "E_CHAIN_BREAK"
        )
        self.assertTrue(chain_error["actual_prefix"])

        mismatch, _ = make_snapshot(
            2,
            [make_entry("b.txt", "two\n")],
            previous_fingerprint="0" * 64,
            previous_version=1,
        )
        mismatch_engine = BackupEngine(self.root / "mismatch")
        report = mismatch_engine.recover(
            [
                self.write_snapshot("mismatch-01.json", v1),
                self.write_snapshot("mismatch-02.json", mismatch),
            ]
        )
        self.assertIn("E_FINGERPRINT", self.codes(report))
        self.assertEqual(
            [item["version"] for item in report["effective_snapshots"]], [1]
        )
        error = next(
            item for item in report["errors"] if item["code"] == "E_FINGERPRINT"
        )
        self.assertTrue(error["actual_prefix"])
        self.assertEqual(len(error["declared_prefix"]), 12)

    def test_cycle_and_self_reference_are_arrival_order_independent(self) -> None:
        def cycle_snapshots() -> tuple[dict, dict]:
            a, fp_a = make_snapshot(1, [make_entry("cycle/a.txt", "a\n")])
            b, fp_b = make_snapshot(
                2,
                [make_entry("cycle/b.txt", "b\n")],
                previous_fingerprint=fp_a,
                previous_version=1,
            )
            a["previous_fingerprint"] = fp_b
            a["previous_version"] = 2
            return a, b

        cycle_a, cycle_b = cycle_snapshots()
        a_path = self.write_snapshot("cycle-a.json", cycle_a)
        b_path = self.write_snapshot("cycle-b.json", cycle_b)
        reports = []
        for index, paths in enumerate(([b_path, a_path], [a_path, b_path])):
            engine = BackupEngine(self.root / f"cycle-{index}")
            reports.append(engine.recover(paths))
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(self.codes(reports[0]), {"E_CYCLE"})
        self.assertEqual(reports[0]["effective_snapshots"], [])

        self_ref, fp = make_snapshot(1, [make_entry("self.txt", "self\n")])
        self_ref["previous_fingerprint"] = fp
        self_ref["previous_version"] = 1
        self_path = self.write_snapshot("self.json", self_ref)
        self_report = BackupEngine(self.root / "self").recover([self_path])
        self.assertEqual(self.codes(self_report), {"E_CYCLE"})
        self.assertEqual(self_report["effective_snapshots"], [])

    def test_fork_preserves_all_branches_and_commits_only_common_prefix(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("base.txt", "base\n")])
        branch_a, fp_a = make_snapshot(
            2,
            [make_entry("a.txt", "a\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        branch_b, fp_b = make_snapshot(
            2,
            [make_entry("b.txt", "b\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        engine = BackupEngine(self.root / "ws")
        report = engine.recover(
            [
                self.write_snapshot("fork-01.json", v1),
                self.write_snapshot("fork-02b.json", branch_b),
                self.write_snapshot("fork-02a.json", branch_a),
            ]
        )
        self.assertIn("E_FORK", self.codes(report))
        self.assertEqual(
            [item["version"] for item in report["effective_snapshots"]], [1]
        )
        self.assertEqual(report["checkpoint"]["watermark"], 1)
        fork = report["forks"][0]
        self.assertEqual(set(fork["branch_identifiers"]), {fp_a, fp_b})
        self.assertEqual(fork["versions"], [2])
        status = engine.status()
        self.assertEqual(self.entry_paths(status), ["base.txt"])

    def test_rollback_uses_persisted_watermark_and_does_not_regress_state(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        v2, fp2 = make_snapshot(
            2,
            [make_entry("a.txt", "two\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        v3, fp3 = make_snapshot(
            3,
            [make_entry("c.txt", "three\n")],
            previous_fingerprint=fp2,
            previous_version=2,
        )
        alt2, _ = make_snapshot(
            2,
            [make_entry("a.txt", "rollback\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        v1_path = self.write_snapshot("rollback-01.json", v1)
        v2_path = self.write_snapshot("rollback-02.json", v2)
        v3_path = self.write_snapshot("rollback-03.json", v3)
        alt_path = self.write_snapshot("rollback-02-alt.json", alt2)
        engine = BackupEngine(self.root / "ws")
        first = engine.recover([v3_path, v1_path, v2_path])
        self.assertEqual(first["exit_code"], 0)
        self.assertEqual(first["checkpoint"]["watermark"], 3)

        replay = engine.recover([v2_path])
        self.assertIn("E_ROLLBACK", self.codes(replay))
        self.assertEqual(replay["rollback_versions"], [2])
        self.assertEqual(replay["checkpoint"]["watermark"], 3)

        rollback = engine.recover([alt_path])
        self.assertIn("E_ROLLBACK", self.codes(rollback))
        self.assertEqual(rollback["rollback_versions"], [2])
        self.assertEqual(rollback["checkpoint"]["watermark"], 3)
        status = engine.status()
        self.assertEqual(status["state"]["watermark"], 3)
        self.assertIn("c.txt", self.entry_paths(status))
        self.assertNotIn("rollback", [
            base64.b64decode(item["content_b64"]).decode() for item in status["state"]["entries"]
        ])

    def test_tampered_entries_are_excluded_but_valid_entries_continue(self) -> None:
        v1, fp1 = make_snapshot(
            1,
            [
                make_entry("a.txt", "a-old\n"),
                make_entry("b.txt", "b-old\n"),
                make_entry("keep.txt", "keep\n"),
            ],
        )
        digest_bad = make_entry("b.txt", "b-new\n")
        digest_bad["digest"] = "sha256:" + ("0" * 64)
        length_bad = make_entry("c.txt", "short\n")
        length_bad["length"] = 1000
        v2, _ = make_snapshot(
            2,
            [
                make_entry("a.txt", "a-new\n", tampered=True),
                digest_bad,
                length_bad,
                make_entry("d.txt", "valid-new\n"),
            ],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        v1_path = self.write_snapshot("tamper-01.json", v1)
        v2_path = self.write_snapshot("tamper-02.json", v2)
        before = (self.sha256(v1_path), self.sha256(v2_path))
        engine = BackupEngine(self.root / "ws")
        report = engine.recover([v2_path, v1_path])
        self.assertEqual(report["exit_code"], 1)
        self.assertTrue({"E_TAMPER", "E_LENGTH"} <= self.codes(report))
        self.assertEqual(
            [item["version"] for item in report["effective_snapshots"]], [1, 2]
        )
        self.assertEqual(
            [item["path"] for item in report["excluded_paths"]],
            ["a.txt", "b.txt", "c.txt"],
        )
        status = engine.status()
        self.assertEqual(self.entry_paths(status), ["d.txt", "keep.txt"])
        self.assertEqual((self.sha256(v1_path), self.sha256(v2_path)), before)
        verified = engine.verify_checkpoint()
        self.assertTrue(verified["checkpoint_valid"])
        self.assertEqual(verified["checkpoint"]["watermark"], 2)

    def test_hmac_sha256_whitelist_and_key_policy(self) -> None:
        key = b"deterministic-test-key"
        entries = [
            make_entry("hmac.txt", "secret\n", algorithm="hmac-sha256", hmac_key=key)
        ]
        snapshot, _ = make_snapshot(
            1, entries, algorithm="hmac-sha256", hmac_key=key
        )
        path = self.write_snapshot("hmac-01.json", snapshot)
        engine = BackupEngine(self.root / "ws", hmac_key=key)
        report = engine.recover([path])
        self.assertEqual(report["exit_code"], 0)
        status = engine.status()
        self.assertTrue(
            status["state"]["entries"][0]["digest"].startswith("hmac-sha256:")
        )
        narrow = BackupEngine(self.root / "ws", hmac_key=key, algorithms=["sha256"])
        narrowed = narrow.recover()
        self.assertIn("E_ALGORITHM", self.codes(narrowed))
        self.assertTrue(narrowed["checkpoint"]["rebuilt"])

    def test_lock_conflict_returns_e_lock(self) -> None:
        snapshot, _ = make_snapshot(1, [make_entry("a.txt", "a\n")])
        path = self.write_snapshot("lock-01.json", snapshot)
        workspace = self.root / "ws"
        workspace.mkdir()
        lock_path = workspace / ".backup-engine.lock"
        lock_path.write_text(
            json.dumps({"pid": os.getpid(), "token": "live-owner"}), encoding="utf-8"
        )
        engine = BackupEngine(workspace)
        report = engine.recover([path])
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(self.codes(report), {"E_LOCK"})
        self.assertFalse((workspace / "checkpoint.json").exists())
        lock_path.unlink()
        malformed = engine.import_snapshots([path])
        self.assertEqual(malformed["exit_code"], 0)
        lock_path.write_text("not json", encoding="utf-8")
        conflict = engine.recover()
        self.assertEqual(self.codes(conflict), {"E_LOCK"})

    def test_forced_termination_recovers_only_last_complete_checkpoint(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        v2, _ = make_snapshot(
            2,
            [make_entry("b.txt", "two\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        v1_path = self.write_snapshot("crash-01.json", v1)
        v2_path = self.write_snapshot("crash-02.json", v2)
        crash_points = [
            "recovery_restore_after_temp",
            "recovery_restore_before_replace",
            "recovery_after_restore_write",
            "recovery_checkpoint_after_temp",
            "recovery_before_checkpoint_replace",
        ]
        for index, crash_point in enumerate(crash_points):
            with self.subTest(crash_point=crash_point):
                workspace = self.root / f"ws-{index}"
                engine = BackupEngine(workspace)
                self.assertEqual(engine.recover([v1_path])["exit_code"], 0)
                before = engine.status()
                self.assertEqual(before["state"]["watermark"], 1)

                env = dict(os.environ)
                env["BACKUP_ENGINE_CRASH_AT"] = crash_point
                crashed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "backup_engine",
                        "recover",
                        "--workspace",
                        str(workspace),
                        str(v2_path),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                self.assertNotEqual(crashed.returncode, 0)

                after_crash = engine.status()
                self.assertTrue(after_crash["ok"])
                self.assertEqual(after_crash["state"]["watermark"], 1)
                self.assertEqual(self.entry_paths(after_crash), ["a.txt"])
                self.assertTrue(engine.verify_checkpoint()["checkpoint_valid"])

                resumed = engine.recover()
                self.assertEqual(resumed["exit_code"], 0)
                self.assertEqual(resumed["checkpoint"]["watermark"], 2)
                self.assertEqual(
                    self.entry_paths(engine.status()), ["a.txt", "b.txt"]
                )
                self.assertEqual(list(workspace.rglob("*.tmp-*")), [])

    def test_out_of_order_import_is_report_and_state_deterministic(self) -> None:
        v1, fp1 = make_snapshot(
            1,
            [make_entry("a.txt", "one\n"), make_entry("b.txt", "one\n")],
        )
        bad = make_entry("b.txt", "two\n")
        bad["digest"] = "sha256:" + ("1" * 64)
        v2, fp2 = make_snapshot(
            2,
            [make_entry("a.txt", "two\n"), bad],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        v3, fp3 = make_snapshot(
            3,
            [make_entry("b.txt", "restored\n"), make_entry("c.txt", "three\n")],
            previous_fingerprint=fp2,
            previous_version=2,
        )
        v4, fp4 = make_snapshot(
            4,
            [make_entry("d.txt", "four\n")],
            previous_fingerprint=fp3,
            previous_version=3,
        )
        paths = [
            self.write_snapshot("order-01.json", v1),
            self.write_snapshot("order-02.json", v2),
            self.write_snapshot("order-03.json", v3),
            self.write_snapshot("order-04.json", v4),
        ]
        before = [self.sha256(path) for path in paths]
        orders = [(3, 1, 0, 2), (2, 0, 3, 1), (0, 1, 2, 3)]
        reports = []
        states = []
        for index, order in enumerate(orders):
            engine = BackupEngine(self.root / f"order-{index}")
            for position in order:
                imported = engine.import_snapshots([paths[position]])
                self.assertIn(imported["exit_code"], {0, 1})
            reports.append(engine.recover())
            states.append(engine.status()["state"])
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[1], reports[2])
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])
        self.assertEqual(
            [item["path"] for item in reports[0]["excluded_paths"]], []
        )
        self.assertEqual([self.sha256(path) for path in paths], before)

    def test_policy_and_schema_change_rebuild_without_mixing_old_state(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        v2, _ = make_snapshot(
            2,
            [make_entry("b.txt", "two\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        v1_path = self.write_snapshot("rebuild-01.json", v1)
        v2_path = self.write_snapshot("rebuild-02.json", v2)
        workspace = self.root / "ws"
        initial = BackupEngine(workspace)
        self.assertEqual(initial.recover([v1_path, v2_path])["exit_code"], 0)
        user_state = initial.status()["state"]

        narrowed = BackupEngine(workspace, algorithms=["sha256"])
        policy_report = narrowed.recover()
        self.assertIn("E_POLICY_REBUILD", {item["code"] for item in policy_report["warnings"]})
        self.assertTrue(policy_report["checkpoint"]["rebuilt"])
        self.assertEqual(policy_report["exit_code"], 0)
        rebuilt_state = narrowed.status()["state"]
        self.assertEqual(rebuilt_state["entries"], user_state["entries"])
        self.assertEqual(rebuilt_state["watermark"], 2)

        checkpoint = json.loads((workspace / "checkpoint.json").read_text())
        checkpoint["schema_version"] = 999
        (workspace / "checkpoint.json").write_bytes(canonical_json_bytes(checkpoint))
        schema_report = narrowed.recover()
        self.assertIn(
            "E_CHECKPOINT_REBUILD",
            {item["code"] for item in schema_report["warnings"]},
        )
        self.assertTrue(schema_report["checkpoint"]["rebuilt"])
        self.assertEqual(narrowed.status()["state"]["entries"], user_state["entries"])

    def test_checkpoint_validation_detects_corrupt_restore_then_rebuilds(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        v2, _ = make_snapshot(
            2,
            [make_entry("b.txt", "two\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        workspace = self.root / "ws"
        engine = BackupEngine(workspace)
        engine.recover(
            [
                self.write_snapshot("corrupt-01.json", v1),
                self.write_snapshot("corrupt-02.json", v2),
            ]
        )
        checkpoint = json.loads((workspace / "checkpoint.json").read_text())
        restore_path = workspace / checkpoint["restore_file"]
        restore = json.loads(restore_path.read_text())
        restore["entries"][0]["content_b64"] = base64.b64encode(b"tampered").decode()
        restore_path.write_bytes(canonical_json_bytes(restore))

        verification = engine.verify_checkpoint()
        self.assertFalse(verification["checkpoint_valid"])
        self.assertEqual(verification["exit_code"], 1)
        self.assertIn("E_CHECKPOINT", self.codes(verification))
        status = engine.status()
        self.assertFalse(status["checkpoint"]["valid"])

        recovered = engine.recover()
        self.assertEqual(recovered["exit_code"], 0)
        self.assertTrue(recovered["checkpoint"]["rebuilt"])
        repaired = engine.status()
        self.assertTrue(repaired["checkpoint"]["valid"])
        self.assertEqual(self.entry_paths(repaired), ["a.txt", "b.txt"])

    def test_unreadable_checkpoint_is_reported_and_rebuilt(self) -> None:
        snapshot, _ = make_snapshot(1, [make_entry("a.txt", "one\n")])
        workspace = self.root / "ws"
        engine = BackupEngine(workspace)
        engine.recover([self.write_snapshot("bad-cp-01.json", snapshot)])
        (workspace / "checkpoint.json").write_bytes(b"{not-json")

        status = engine.status()
        self.assertEqual(status["exit_code"], 1)
        self.assertTrue(status["checkpoint"]["present"])
        self.assertFalse(status["checkpoint"]["valid"])
        self.assertIn("E_CHECKPOINT", self.codes(status))

        recovered = engine.recover()
        self.assertEqual(recovered["exit_code"], 0)
        self.assertTrue(recovered["checkpoint"]["rebuilt"])
        self.assertEqual(self.entry_paths(engine.status()), ["a.txt"])

    def test_atomic_files_are_canonical_and_have_no_temp_residue(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        v2, _ = make_snapshot(
            2,
            [make_entry("b.txt", "two\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        workspace = self.root / "ws"
        engine = BackupEngine(workspace)
        engine.recover(
            [
                self.write_snapshot("atomic-01.json", v1),
                self.write_snapshot("atomic-02.json", v2),
            ]
        )
        self.assertEqual(list(workspace.rglob("*.tmp-*")), [])
        for path in [workspace / "checkpoint.json", *workspace.glob("restore/*.json")]:
            raw = path.read_bytes()
            self.assertEqual(raw, canonical_json_bytes(json.loads(raw)))
        report = engine.verify_checkpoint()
        self.assertTrue(report["checkpoint_valid"])

    def test_store_content_must_match_content_addressed_filename(self) -> None:
        v1, fp1 = make_snapshot(1, [make_entry("a.txt", "one\n")])
        v2, _ = make_snapshot(
            2,
            [make_entry("b.txt", "two\n")],
            previous_fingerprint=fp1,
            previous_version=1,
        )
        workspace = self.root / "ws"
        engine = BackupEngine(workspace)
        engine.recover(
            [
                self.write_snapshot("store-01.json", v1),
                self.write_snapshot("store-02.json", v2),
            ]
        )
        stored = sorted((workspace / "snapshots").glob("v0000000002-*.json"))[0]
        modified = json.loads(stored.read_text())
        modified["entries"][0]["tampered"] = True
        stored.write_bytes(canonical_json_bytes(modified))

        report = engine.recover()
        self.assertIn("E_FINGERPRINT", self.codes(report))
        self.assertEqual(report["checkpoint"]["watermark"], 2)
        self.assertEqual(self.entry_paths(engine.status()), ["a.txt", "b.txt"])

    def test_sample_generator_is_deterministic(self) -> None:
        from backup_engine.samples import generate_samples

        first = self.root / "samples-a"
        second = self.root / "samples-b"
        generate_samples(first)
        generate_samples(second)
        first_hashes = {
            str(path.relative_to(first)): self.sha256(path)
            for path in sorted(first.rglob("*.json"))
        }
        second_hashes = {
            str(path.relative_to(second)): self.sha256(path)
            for path in sorted(second.rglob("*.json"))
        }
        self.assertEqual(first_hashes, second_hashes)

    def test_cli_exit_codes_and_four_command_families(self) -> None:
        samples_dir = self.root / "samples"
        subprocess.run(
            [sys.executable, "-m", "backup_engine", "samples", "--output", str(samples_dir)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        workspace = self.root / "cli-ws"
        valid = samples_dir / "valid"
        report_path = self.root / "report.json"

        clean = subprocess.run(
            [
                sys.executable,
                "-m",
                "backup_engine",
                "recover",
                "--workspace",
                str(workspace),
                str(valid / "01.json"),
                str(valid / "02.json"),
                "--report",
                str(report_path),
            ],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(clean.returncode, 0)
        self.assertTrue(json.loads(report_path.read_text())["ok"])

        status = subprocess.run(
            [sys.executable, "-m", "backup_engine", "status", "--workspace", str(workspace)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(status.returncode, 0)
        self.assertEqual(json.loads(status.stdout)["command"], "status")
        verify = subprocess.run(
            [
                sys.executable,
                "-m",
                "backup_engine",
                "verify-checkpoint",
                "--workspace",
                str(workspace),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(verify.returncode, 0)
        self.assertTrue(json.loads(verify.stdout)["checkpoint_valid"])

        tamper_workspace = self.root / "cli-tamper"
        partial = subprocess.run(
            [
                sys.executable,
                "-m",
                "backup_engine",
                "recover",
                "--workspace",
                str(tamper_workspace),
                str(samples_dir / "tamper" / "01.json"),
                str(samples_dir / "tamper" / "02.json"),
                "--report",
                str(self.root / "tamper-report.json"),
            ],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(partial.returncode, 1)
        self.assertFalse(json.loads((self.root / "tamper-report.json").read_text())["ok"])

        parameter_error = subprocess.run(
            [
                sys.executable,
                "-m",
                "backup_engine",
                "recover",
                "--workspace",
                str(workspace),
                "--algorithms",
                "md5",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(parameter_error.returncode, 2)


if __name__ == "__main__":
    unittest.main()
