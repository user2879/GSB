from __future__ import annotations

import unittest

from spectral_diag.closure import ConventionSession
from spectral_diag.config import RunConfig


class ClosureTests(unittest.TestCase):
    def test_switch_and_restore_recompute_exactly_the_closure(self) -> None:
        session = ConventionSession(RunConfig())
        first = session.run()
        same = session.run()
        switched = session.run(observer="cie1964_10")
        restored = session.run(observer="cie1931_2")

        self.assertEqual(same["report"]["recomputed"], [])
        self.assertEqual(restored["report"]["recomputed"], [])
        self.assertTrue(restored["report"]["cache_hits"])
        self.assertEqual(first["report"]["hashes"], restored["report"]["hashes"])
        self.assertEqual(first["scene_sha256"], switched["scene_sha256"])
        self.assertNotEqual(first["sample_source_xyz_sha256"], switched["sample_source_xyz_sha256"])
        self.assertNotIn("scene", switched["report"]["recomputed"])
        self.assertTrue(switched["report"]["closure_reconciled"])
        self.assertEqual(
            switched["report"]["recomputed"], switched["report"]["invalidation_closure"]
        )
        self.assertEqual(
            switched["attribution"]["reference"]["observer"], "cie1964_10"
        )
        self.assertEqual(
            switched["attribution"]["reference"]["reference_density_nm"],
            switched["report"]["reference_density_nm"],
        )

    def test_normalization_change_invalidates_matrix_and_adaptation(self) -> None:
        session = ConventionSession(RunConfig())
        session.run()
        changed = session.run(normalization="unity")
        recomputed = set(changed["report"]["recomputed"])
        self.assertIn("samples.source_xyz", recomputed)
        self.assertIn("matrix", recomputed)
        self.assertIn("materials.adapted_xyz", recomputed)
        self.assertNotIn("scene", recomputed)
        self.assertTrue(changed["report"]["closure_reconciled"])


if __name__ == "__main__":
    unittest.main()
