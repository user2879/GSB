import tempfile
import unittest
from pathlib import Path

from pool_workbench.gui import PoolViewModel
from pool_workbench.app import WorkbenchSession
from pool_workbench.sample_db import build_sample_db

COLD = Path(__file__).resolve().parent.parent / "samples" / "tasks_cold.json"


class ViewModelTests(unittest.TestCase):
    def test_headless_view_model_reflects_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "sample.sqlite"
            build_sample_db(db)
            session = WorkbenchSession(tmp)
            session.load(COLD, db)
            session.execute()
            session.save_snapshot()
            view = PoolViewModel(session)
            self.assertTrue(view.connection_rows())
            self.assertEqual(
                {row["workload"] for row in view.quota_rows()},
                {"wl_a", "wl_b", "wl_c"},
            )
            self.assertEqual(view.queued_task_ids(), ["t09"])
            self.assertEqual(sum(view.outcome_distribution().values()), 12)
            self.assertTrue(view.snapshots())
            session.close()


if __name__ == "__main__":
    unittest.main()
