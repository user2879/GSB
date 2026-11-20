import json
import os
import shutil
import tempfile
import unittest

from shardbench.config import load_config
from shardbench.engine import Engine, compile_all, full_recompute
from shardbench.invalidation import compare_recomputations
from shardbench.injection import (
    trigger_config_injection,
    trigger_ring_injection,
    trigger_version_injection,
)
from shardbench.ledger import Ledger
from shardbench.replay import replay_round
from shardbench.snapshot import load_snapshot

from .test_validation import sample_paths


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmp, "config.json")
        self.units_path = os.path.join(self.tmp, "units.json")
        shutil.copy(*sample_paths(), self.config_path) if False else shutil.copy(
            sample_paths()[0], self.config_path
        )
        shutil.copy(sample_paths()[1], self.units_path)
        self.ledger_path = os.path.join(self.tmp, "ledger.sqlite3")
        self.snapshot_path = os.path.join(self.tmp, "snapshot.json")
        self.log_path = os.path.join(self.tmp, "events.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _engine(self):
        from shardbench.logging_utils import StructuredLogger

        logger = StructuredLogger(self.log_path)
        engine = Engine(
            self.config_path,
            self.units_path,
            self.ledger_path,
            self.snapshot_path,
            logger=logger,
        )
        return engine, logger

    def test_round_commit_and_replay_consistency(self):
        engine, logger = self._engine()
        result = engine.run_round()
        engine.close()
        logger.close()
        self.assertEqual(result.round_id, 1)
        self.assertTrue(result.scheduler_verification["global_ok"])
        replay = replay_round(self.ledger_path, 1)
        self.assertEqual(replay["status"], "ok")
        self.assertEqual(replay["units_checked"], 11)

    def test_version_monotonic_and_cache_replacement(self):
        engine, logger = self._engine()
        first = engine.run_round()
        self.assertEqual(first.versions.config_version, 1)

        payload = _read_json(self.config_path)
        payload["tables"][0]["params"]["base"] = 16
        _write_json(self.config_path, payload)
        second = engine.run_round(changed_tables=["t_user"])
        engine.close()
        logger.close()

        self.assertEqual(second.versions.config_version, 2)
        self.assertIsNotNone(second.replacement)
        self.assertGreater(second.replacement["cleared_entry_count"], 0)
        self.assertIn("config", second.replacement["changed_rules"])
        self.assertTrue(any(target.startswith("db0.") for target in second.replacement["affected_targets"]))

    def test_invalidation_only_recomputes_subset(self):
        engine, logger = self._engine()
        first = engine.run_round()
        before = dict(first.plans)

        payload = _read_json(self.config_path)
        ring_table = next(
            t for t in payload["tables"] if t["algorithm"] == "consistent_hash"
        )
        ring_table["params"]["nodes"].append({"id": "node-d", "weight": 2})
        _write_json(self.config_path, payload)
        second = engine.run_round(changed_tables=["t_cart_ring"])
        engine.close()
        logger.close()

        invalidated = set(second.invalidation["invalidated"])
        self.assertIn("u_ring_eq", invalidated)
        self.assertNotIn("u_user_eq", invalidated)
        untouched = [
            unit_id for unit_id in before if unit_id not in invalidated
        ]
        self.assertTrue(untouched)
        for unit_id in untouched:
            self.assertEqual(
                before[unit_id]["summary"], second.plans[unit_id]["summary"]
            )
            self.assertEqual(
                before[unit_id]["targets"], second.plans[unit_id]["targets"]
            )

    def test_incremental_matches_full_recompute(self):
        config = load_config(self.config_path, self.units_path)
        incremental = compile_all(config)
        full = full_recompute(config)
        differences = compare_recomputations(incremental, full)
        self.assertEqual(differences, [])

    def test_snapshot_recovery_and_version_mismatch(self):
        engine, logger = self._engine()
        engine.run_round()
        engine.close()
        logger.close()
        snapshot = load_snapshot(self.snapshot_path)
        self.assertEqual(snapshot.ledger_cursor, 1)

        error = trigger_version_injection(self.snapshot_path)
        self.assertEqual(error.code, "E_VERSION")
        self.assertIn("config", error.details["mismatches"])

    def test_force_kill_leaves_unknown_round_requiring_rerun(self):
        engine, logger = self._engine()
        round_id = engine.ledger.begin_round()
        # 模拟进程被强制终止：不提交事务、不写快照，直接释放连接。
        engine.ledger.close()
        logger.close()

        ledger = Ledger(self.ledger_path)
        try:
            unknown = ledger.unknown_rounds()
            self.assertIn(round_id, unknown)
            ledger.mark_unknown_for_rerun(unknown)
            self.assertEqual(ledger.last_committed_round_id(), None)
        finally:
            ledger.close()

        # 重新打开后只能基于已提交轮次恢复，这里没有任何已提交轮次。
        engine2, logger2 = self._engine()
        result = engine2.run_round()
        engine2.close()
        logger2.close()
        self.assertEqual(result.round_id, round_id + 1)

    def test_structured_injection_codes(self):
        config_payload = _read_json(self.config_path)
        units_payload = _read_json(self.units_path)
        self.assertEqual(
            trigger_config_injection(config_payload, units_payload).code,
            "E_CONFIG",
        )
        engine, logger = self._engine()
        engine.run_round()
        engine.close()
        logger.close()
        self.assertEqual(
            trigger_ring_injection(config_payload, units_payload).code, "E_RING"
        )


if __name__ == "__main__":
    unittest.main()
