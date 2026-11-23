"""业务要求 5/6：精确失效账本与增量/全量逐字段等价。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from retbench.config import Command, build_config, load_commands, load_config
from retbench.concurrency import BoundedPool
from retbench.engine import Engine
from retbench.fingerprint import digest_obj
from retbench.generator import default_config, generate_workspace
from retbench.runner import _normalize, equivalence_check


class LedgerEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp(prefix="retbench-")
        generate_workspace(self.temp, sample_count=12, inject_count=2)
        self.config_path = os.path.join(self.temp, "config.yaml")
        self.command_path = os.path.join(self.temp, "commands.json")
        self.sample_root = os.path.join(self.temp, "samples")

    def _engine(self, snap: str) -> Engine:
        config = load_config(self.config_path)
        engine = Engine(config, self.sample_root, os.path.join(self.temp, snap))
        engine.load_samples()
        return engine

    def test_weight_change_keeps_feature_cache(self) -> None:
        engine = self._engine("snap_a")
        pool = BoundedPool(2)
        engine.baseline_report(pool)
        cmd = Command(seq=0, op="set_weight",
                      payload={"channel": "color", "value": 0.8})
        outcome = engine.handle_command(cmd, 1, pool)
        self.assertTrue(any(x.startswith("visual:") for x in outcome.reused))
        self.assertTrue(any(x.startswith("text:") for x in outcome.reused))
        self.assertTrue(all(not x.startswith(("visual:", "text:"))
                            for x in outcome.recomputed))
        self.assertTrue(any(x.startswith("alignment:") for x in outcome.recomputed))

    def test_channel_toggle_only_invalidates_recall(self) -> None:
        engine = self._engine("snap_b")
        pool = BoundedPool(2)
        engine.baseline_report(pool)
        outcome = engine.handle_command(
            Command(seq=1, op="disable_channel", payload={"channel": "hybrid"}),
            1, pool,
        )
        self.assertTrue(all(not x.startswith(("visual:", "text:", "alignment:"))
                            for x in outcome.recomputed))
        self.assertTrue(any(x.startswith("recall:") for x in outcome.recomputed))

    def test_full_vs_incremental_field_equivalence(self) -> None:
        config = load_config(self.config_path)
        commands = load_commands(self.command_path, config)

        inc_engine = Engine(config, self.sample_root,
                            os.path.join(self.temp, "snap_inc"))
        inc_engine.load_samples()
        pool = BoundedPool(config.thread_pool_size)
        inc_engine.baseline_report(pool)
        for index, command in enumerate(commands):
            inc_engine.handle_command(command, index + 1, pool)
        incremental = inc_engine.last_report

        full_root = os.path.join(self.temp, "full_copy")
        shutil.copytree(self.sample_root, full_root)
        full_config = load_config(self.config_path)
        full_engine = Engine(full_config, full_root,
                             os.path.join(self.temp, "snap_full"))
        full_engine.load_samples()
        for command in commands:
            full_engine.apply_command(command)
        full_pool = BoundedPool(full_config.thread_pool_size)
        full_pipeline = full_engine.run_pipeline(full_pool)
        full = full_engine.build_report(
            full_pipeline, None, False, 0.0, [], 0
        )

        check = equivalence_check(full, incremental)
        self.assertTrue(check["equivalent"], msg=json.dumps(check, ensure_ascii=False))

    def test_quant_switch_invalidates_all_stages(self) -> None:
        engine = self._engine("snap_q")
        pool = BoundedPool(2)
        engine.baseline_report(pool)
        outcome = engine.handle_command(
            Command(seq=2, op="set_quant",
                    payload={"frac_bits": 16, "tier": "fine"}),
            1, pool,
        )
        stages = {item.split(":")[0] for item in outcome.recomputed}
        self.assertIn("visual", stages)
        self.assertIn("text", stages)
        self.assertIn("alignment", stages)
        self.assertIn("recall", stages)


if __name__ == "__main__":
    unittest.main()
