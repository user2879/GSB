from __future__ import annotations

import copy
import unittest

from forensic_workbench.errors import ForensicError
from forensic_workbench.pipeline import execute
from forensic_workbench.state import SnapshotStore
from tests.support import TemporaryWorkspace, make_workspace_source


class GraphErrorTests(unittest.TestCase):
    def test_missing_dependency_node_returns_business_code(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            execute(workspace, source.default_pipeline_config(), verify_full=False)
            graph = copy.deepcopy(SnapshotStore(workspace).load().graph)  # type: ignore[union-attr]
            del graph.nodes["frame:000000"]
            with self.assertRaises(ForensicError) as raised:
                graph.validate()
            self.assertEqual(raised.exception.code, "E_DEPENDENCY_STALE")
            self.assertIn("source", raised.exception.context)
            self.assertIn("target", raised.exception.context)


if __name__ == "__main__":
    unittest.main()

