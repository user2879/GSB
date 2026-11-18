from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import migrate_sample

from dbml.database import connect, write_json_atomic
from dbml.errors import MigrationError
from dbml.migrations import perturb_scale
from dbml.replay import replay_snapshot
from dbml.snapshot import SnapshotManager, restore_snapshot
from dbml.verify import verify_drift


DRIFT_COLUMNS = ["discount_rate", "net_cents", "tax_cents", "total_cents", "unit_price_cents"]


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.base, self.db, self.runner, self.result = migrate_sample(self.directory)
        self.drift = verify_drift(
            self.base,
            self.db,
            table="orders",
            columns=DRIFT_COLUMNS,
            before_calibrations=self.runner.manifest,
            row_key=["id"],
            raise_on_drift=False,
        )
        self.snapshot_dir = self.directory / "snapshots"
        self.snapshot_dir.mkdir()
        manager = SnapshotManager(
            self.db,
            self.runner.manifest,
            self.snapshot_dir,
            base_db_path=self.base,
            verification_config={
                "table": "orders",
                "columns": DRIFT_COLUMNS,
                "row_key": ["id"],
                "batch_size": 7,
            },
        )
        self.snapshot_path = self.snapshot_dir / "session.json"
        manager.save(name="session.json", conclusions={"drift": self.drift}, base_db_path=self.base)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_snapshot_restore_replaces_on_calibration_change(self) -> None:
        same = restore_snapshot(self.snapshot_path, self.db)
        self.assertTrue(same["restored"])
        self.assertFalse(same["replaced"])
        perturb_scale(
            self.db,
            self.runner.manifest,
            table="orders",
            column="discount_rate",
            new_scale=2,
        )
        replaced = restore_snapshot(self.snapshot_path, self.db)
        self.assertFalse(replaced["restored"])
        self.assertTrue(replaced["replaced"])
        self.assertEqual(replaced["merge_policy"], "replace_all")
        self.assertEqual(
            replaced["affected_columns"],
            ["orders.discount_rate", "orders.net_cents", "orders.total_cents"],
        )
        self.assertEqual(replaced["affected_steps"], ["backfill-net", "backfill-total"])
        self.assertEqual(replaced["session_state"]["conclusions"], {})
        self.assertIn("calibration_version", {item["version"] for item in replaced["version_differences"]})

    def test_replay_reproduces_snapshot(self) -> None:
        replay = replay_snapshot(
            self.snapshot_path,
            self.base,
            self.directory / "replay",
            batch_size=5,
        )
        self.assertTrue(replay["reproducible"])
        self.assertEqual(replay["steps_compared"], 5)
        self.assertEqual(replay["drift"]["conclusion"], "precision_equivalent")

    def test_replay_reports_base_digest_mismatch(self) -> None:
        tampered = self.directory / "tampered_base.sqlite3"
        shutil.copy2(self.base, tampered)
        connection = connect(tampered)
        try:
            connection.execute("UPDATE orders SET total_cents = total_cents + 1 WHERE id = 1")
        finally:
            connection.close()
        with self.assertRaises(MigrationError) as raised:
            replay_snapshot(self.snapshot_path, tampered, self.directory / "replay-tampered")
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.step_id, "__base__")
        self.assertEqual(raised.exception.details["column"], "*")

    def test_replay_reports_first_column_calibration_mismatch(self) -> None:
        snapshot = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        first_step = snapshot["ledger_view"]["steps"][0]
        target = first_step["before_columns"][0]
        target["calibration_version"] = "tampered-version"
        tampered_path = self.directory / "tampered_snapshot.json"
        write_json_atomic(tampered_path, snapshot)
        with self.assertRaises(MigrationError) as raised:
            replay_snapshot(tampered_path, self.base, self.directory / "replay-bad-snapshot")
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.step_id, "widen-unit-price")
        self.assertEqual(raised.exception.details["column"], f"orders.{target['column']}")

    def test_perturbed_snapshot_replays_recalibration(self) -> None:
        perturb_scale(
            self.db,
            self.runner.manifest,
            table="orders",
            column="discount_rate",
            new_scale=2,
        )
        perturbed_drift = verify_drift(
            self.base,
            self.db,
            table="orders",
            columns=DRIFT_COLUMNS,
            before_calibrations=self.runner.manifest,
            row_key=["id"],
            raise_on_drift=False,
        )
        manager = SnapshotManager(
            self.db,
            self.runner.manifest,
            self.snapshot_dir,
            base_db_path=self.base,
            verification_config={
                "table": "orders",
                "columns": DRIFT_COLUMNS,
                "row_key": ["id"],
                "batch_size": 7,
            },
        )
        perturbed_path = self.snapshot_dir / "perturbed.json"
        manager.save(name="perturbed.json", conclusions={"drift": perturbed_drift})
        replay = replay_snapshot(
            perturbed_path, self.base, self.directory / "replay-perturbed", batch_size=7
        )
        self.assertTrue(replay["reproducible"])
        self.assertEqual(replay["drift"]["conclusion"], "precision_equivalent")


if __name__ == "__main__":
    unittest.main()
