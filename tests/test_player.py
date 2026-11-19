import os
import tempfile
import unittest

from rts_replay.demo_clock import build_scripted_costs
from rts_replay.engine import TickEngine
from rts_replay.errors import ErrorCode
from rts_replay.generator import PROFILES, build_rules, generate_events
from rts_replay.offline import run_offline
from rts_replay.player import ReplayPlayer, ScriptedClock


class PlayerTests(unittest.TestCase):
    def setUp(self):
        self.profile = PROFILES[3]
        self.rules = build_rules(self.profile)
        self.events = generate_events(self.profile, self.rules)

    def test_online_matches_offline_hash_and_ledger(self):
        offline = run_offline(self.rules, self.events)
        costs = build_scripted_costs(self.profile.ticks, 4.0, 6.0)
        player = ReplayPlayer(
            self.rules,
            self.events,
            budget_ms=4.0,
            base_ticks_per_frame=4,
            clock=ScriptedClock(costs),
        )
        player.run_to_end()
        self.assertEqual(
            [r.state_hash for r in player.engine.records],
            [r["state_hash"] for r in offline.tick_hashes],
        )
        self.assertEqual(
            [entry.to_dict() for entry in player.engine.ledger], offline.ledger
        )

    def test_budget_degradation_halves_allowance_and_records_code(self):
        costs = build_scripted_costs(self.profile.ticks, 4.0, 6.0)
        player = ReplayPlayer(
            self.rules,
            self.events,
            budget_ms=4.0,
            base_ticks_per_frame=4,
            clock=ScriptedClock(costs),
        )
        degraded_seen = False
        while not player.finished:
            stat = player.run_frame()
            if stat.degraded:
                degraded_seen = True
                self.assertEqual(stat.note, ErrorCode.E_BUDGET_EXCEEDED.value)
                self.assertEqual(stat.allowance_after, max(1, stat.allowance_before // 2))
        self.assertTrue(degraded_seen)
        self.assertGreater(player.budget_report.over_budget_count, 0)

    def test_degradation_does_not_change_final_state(self):
        costs = build_scripted_costs(self.profile.ticks, 4.0, 6.0)
        strict = ReplayPlayer(
            self.rules, self.events, budget_ms=4.0, base_ticks_per_frame=4,
            clock=ScriptedClock(costs),
        )
        relaxed = ReplayPlayer(
            self.rules, self.events, budget_ms=1000.0, base_ticks_per_frame=4,
            clock=ScriptedClock(costs),
        )
        strict.run_to_end()
        relaxed.run_to_end()
        self.assertEqual(
            strict.engine.state.state_hash(), relaxed.engine.state.state_hash()
        )

    def test_jump_rebuilds_equivalent_state_without_full_replay(self):
        offline = run_offline(self.rules, self.events)
        baseline = {0: offline.engine.snapshots[0].state_hash}
        baseline.update({r["tick"]: r["state_hash"] for r in offline.tick_hashes})
        player = ReplayPlayer(self.rules, self.events)
        for target in (0, 1, 25, 60, self.profile.ticks):
            player.jump_to(target)
            self.assertEqual(player.engine.state.state_hash(), baseline[target])
            self.assertEqual(player.engine.state.tick, target)

    def test_snapshot_recovery_after_simulated_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot_dir = os.path.join(directory, "snap")
            first = ReplayPlayer(
                self.rules, self.events, snapshot_dir=snapshot_dir,
                snapshot_interval=5,
            )
            for _ in range(4):
                first.run_frame()
            persisted = sorted(os.listdir(snapshot_dir))
            self.assertTrue(persisted)

            restarted = ReplayPlayer(
                self.rules, self.events, snapshot_dir=snapshot_dir,
                snapshot_interval=5,
            )
            snapshot_tick = max(first.engine.snapshots)
            self.assertEqual(restarted.current_tick, snapshot_tick)
            self.assertEqual(
                restarted.engine.state.state_hash(),
                first.engine.snapshots[snapshot_tick].state_hash,
            )
            restarted.run_to_end()
            offline = run_offline(self.rules, self.events)
            self.assertEqual(
                restarted.engine.state.state_hash(),
                offline.engine.state.state_hash(),
            )

    def test_speed_multipliers_and_pause(self):
        player = ReplayPlayer(self.rules, self.events)
        player.set_speed(4)
        self.assertEqual(player.speed, 4)
        with self.assertRaises(Exception):
            player.set_speed(3)
        player.step_one()
        self.assertEqual(player.current_tick, 1)
        self.assertTrue(player.paused)


if __name__ == "__main__":
    unittest.main()
