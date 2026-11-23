from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mixedmem.gui import _graph_layout, load_diagnostic_state
from mixedmem.ledger import Ledger
from mixedmem.runtime import InferenceRuntime
from mixedmem.schema import parse_plan

from .helpers import PLAN_PATH, plan_copy


class GuiStateTests(unittest.TestCase):
    def test_readonly_state_load_does_not_change_ledger(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root / "ledger.db")
            runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
            runtime.run(worker_count=2)
            from mixedmem.snapshot import save_snapshot

            save_snapshot(root / "snapshots", runtime.snapshot_payload())
            digest_before = ledger.content_digest()
            ledger.close()
            state = load_diagnostic_state(
                plan_path=str(PLAN_PATH),
                db_path=str(root / "ledger.db"),
                snapshot_dir=str(root / "snapshots"),
            )
            self.assertEqual(state.ledger_content_digest, digest_before)
            self.assertEqual(len(state.batches), 2)
            self.assertEqual(len(state.leases), 8)
            self.assertEqual(len(state.layer_results), 8)
            self.assertEqual(state.attribution["max_deviation_layer"], "compress")
            positions, edges = _graph_layout(state.graph)
            self.assertEqual(list(positions), list(state.graph.order))
            self.assertEqual(len(edges), 4)


if __name__ == "__main__":
    unittest.main()
