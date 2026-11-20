from __future__ import annotations

import unittest

from rope_extrap.analysis import diagnostic_consistency_view
from rope_extrap.engine import run_full_diagnosis
from rope_extrap.loader import load_commands
from rope_extrap.session import InteractiveSession
from tests.helpers import CONFIG_LARGE, load_config_and_samples


class IncrementalTests(unittest.TestCase):
    def test_command_cross_product_matches_blank_full_rebuild(self) -> None:
        config, samples, _ = load_config_and_samples(count=12)
        commands, command_failures = load_commands(
            "samples/commands.json",
            config=config,
        )
        self.assertEqual(command_failures, [])
        session = InteractiveSession(config, samples, commands)
        try:
            responses = session.run_all()
            incremental = session.final_report()
            final_config = session.config
            final_samples = dict(session.engine.samples)
        finally:
            session.close()
        full = run_full_diagnosis(final_config, final_samples)
        self.assertEqual(
            diagnostic_consistency_view(incremental),
            diagnostic_consistency_view(full),
        )
        self.assertTrue(incremental["summary"]["is_complete"])
        by_id = {response["command_id"]: response for response in responses}
        self.assertEqual(
            by_id["c05"]["reuse_and_recompute"]["summary"]["reused_stage_count"],
            12,
        )
        self.assertEqual(
            by_id["c05"]["reuse_and_recompute"]["summary"]["recomputed_stage_count"],
            24,
        )
        self.assertEqual(
            by_id["c12"]["reuse_and_recompute"]["summary"]["recomputed_stage_count"],
            3,
        )
        self.assertEqual(
            by_id["c12"]["reuse_and_recompute"]["summary"]["recomputed_samples"],
            ["s007"],
        )
        self.assertEqual(
            by_id["c01"]["reuse_and_recompute"]["summary"]["recomputed_stage_count"],
            36,
        )


if __name__ == "__main__":
    unittest.main()

