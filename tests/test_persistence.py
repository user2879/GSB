from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tactical_path.engine import TierEngine
from tactical_path.persistence import StateStore, verify_recovered_hashes
from tactical_path.strategy import StrategyPlanner

from .helpers import ROOT, bundle, config


class PersistenceTests(unittest.TestCase):
    def test_forced_termination_recovers_only_committed_state(self) -> None:
        cfg = config()
        loaded = bundle("small")
        planner = StrategyPlanner(cfg)
        plan = planner.plan(
            loaded.map_data, loaded.queries, cfg.budgets["small"]
        )
        engine = TierEngine(loaded, cfg)
        engine.solve_batch(loaded.queries, plan.decisions, cfg.budgets["small"])
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory))
            store.commit(engine, decisions=plan.decisions, label="initial")
            final = store.commit(engine, decisions=plan.decisions, label="final")
            expected = engine.route_hashes()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tactical_path.crash_worker",
                    "--workspace",
                    str(store.workspace),
                ],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 91)
            recovery = store.recover()
            self.assertEqual(
                recovery["committed_snapshot_id"], final["snapshot_id"]
            )
            self.assertEqual(
                recovery["orphans_recovered"],
                ["orphans/snap-000003", "orphans/tx-demo-crash.json"],
            )
            verification = verify_recovered_hashes(recovery, expected)
            self.assertTrue(verification["all_match"])
            self.assertTrue((store.state / "dependency_graph.json").exists())
            self.assertTrue((store.state / "content_hashes.json").exists())

