from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tactical_path.canonical import atomic_write_json
from tactical_path.config import load_config
from tactical_path.engine import TierEngine
from tactical_path.errors import ErrorCode, TacticalError
from tactical_path.models import MapData, Unit
from tactical_path.persistence import StateStore

from .helpers import CONFIG_PATH, bundle, config


class ErrorCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = config()
        cls.bundle = bundle("small")

    def test_all_declared_codes_exist(self) -> None:
        self.assertEqual(
            {code.value for code in ErrorCode},
            {
                "E_MAP_SHAPE",
                "E_TERRAIN_RANGE",
                "E_DEPENDENCY_STALE",
                "E_CLOSURE_MISMATCH",
                "E_BUDGET_DENIED",
                "E_UNIT_INVALID",
                "E_SNAPSHOT_VERSION",
            },
        )

    def test_map_shape(self) -> None:
        bad = MapData(
            tier="bad",
            seed=1,
            width=4,
            height=4,
            region_size=2,
            region_shape=(2, 2),
            sector_shape=(1, 1),
            terrain=np.ones((3, 4), dtype=np.uint8),
            passable=np.ones((4, 4), dtype=np.uint8),
            regions=np.zeros((4, 4), dtype=np.uint16),
            sectors=np.zeros((4, 4), dtype=np.uint16),
            mobility_rules=dict(self.cfg.mobility),
            content_hash="0" * 64,
            map_version="bad",
        )
        with self.assertRaises(TacticalError) as raised:
            bad.validate()
        self.assertEqual(raised.exception.code, ErrorCode.MAP_SHAPE)

    def test_terrain_range(self) -> None:
        terrain = self.bundle.map_data.terrain.copy()
        terrain[0, 0] = 0
        bad = self.bundle.map_data.clone_with(
            terrain=terrain, passable=self.bundle.map_data.passable.copy()
        )
        with self.assertRaises(TacticalError) as raised:
            bad.validate()
        self.assertEqual(raised.exception.code, ErrorCode.TERRAIN_RANGE)

    def test_unit_invalid(self) -> None:
        impassable = int(np.flatnonzero(self.bundle.map_data.passable.ravel() == 0)[0])
        unit = Unit(
            id="bad-unit",
            tier="small",
            cell=impassable,
            faction="blue",
            mobility="foot",
            cost_multiplier=1,
            max_terrain=9,
        )
        with self.assertRaises(TacticalError) as raised:
            unit.validate(self.bundle.map_data)
        self.assertEqual(raised.exception.code, ErrorCode.UNIT_INVALID)

    def test_dependency_stale(self) -> None:
        engine = TierEngine(self.bundle, self.cfg)
        region_id = 0
        node_id = f"region:small:{region_id}"
        with self.assertRaises(TacticalError) as raised:
            engine.graph.assert_dependencies_current(
                node_id, {node_id: "stale-version"}
            )
        self.assertEqual(raised.exception.code, ErrorCode.DEPENDENCY_STALE)

    def test_budget_denied(self) -> None:
        from tactical_path.models import Budgets

        with self.assertRaises(TacticalError) as raised:
            Budgets(
                tier="bad",
                query_time_ms=-1,
                query_memory_mib=1,
                total_memory_mib=1,
                max_threads=1,
                precompute_time_ms=1,
            ).validate()
        self.assertEqual(raised.exception.code, ErrorCode.BUDGET_DENIED)

    def test_snapshot_version_and_cause_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory))
            atomic_write_json(
                store.pointer_path,
                {
                    "schema": "commit-pointer-v1",
                    "snapshot_id": "snap-000001",
                    "snapshot_path": "snapshots/snap-000001/snapshot.json",
                    "sha256": "0" * 64,
                    "commit_seq": 1,
                },
            )
            (store.state / "snapshots" / "snap-000001").mkdir(parents=True)
            (store.state / "snapshots" / "snap-000001" / "snapshot.json").write_bytes(
                b"{}\n"
            )
            with self.assertRaises(TacticalError) as raised:
                store.load_committed()
            self.assertEqual(raised.exception.code, ErrorCode.SNAPSHOT_VERSION)
        with self.assertRaises(TacticalError) as missing:
            load_config(Path(directory) / "missing.yaml")
        self.assertEqual(missing.exception.code, ErrorCode.SNAPSHOT_VERSION)
        self.assertIsNotNone(missing.exception.__cause__)

