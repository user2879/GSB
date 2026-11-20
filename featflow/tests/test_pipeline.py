"""End-to-end and unit tests for the featflow pipeline (stdlib unittest)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from featflow.common import (  # noqa: E402
    CODE_DEF_CYCLE,
    CODE_DEF_PARAM,
    CODE_EVENT_INVALID,
    FixedPoint,
    write_json,
    write_jsonl,
)
from featflow.compute import FeatureComputer  # noqa: E402
from featflow.config import PipelineConfig  # noqa: E402
from featflow.loader import load_events, load_feature_defs  # noqa: E402
from featflow.models import Event, FeatureDef  # noqa: E402
from featflow.storage import EventStore  # noqa: E402

DATA = os.path.join(ROOT, "data")


def write_tmp_tree(files: dict[str, object]) -> str:
    tmp = tempfile.mkdtemp()
    for rel, payload in files.items():
        path = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rel.endswith(".jsonl"):
            write_jsonl(path, payload)  # type: ignore[arg-type]
        elif rel.endswith(".yaml"):
            import yaml
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, sort_keys=True)
        else:
            write_json(path, payload)
    return tmp


BASE_CONFIG = {
    "thread_concurrency": 2,
    "global_event_budget": -1,
    "feature_point_quota": -1,
    "cache_block_limit": 2,
    "snapshot_commit_every": 10,
    "fixed_point_scale": 4,
    "quant_step": "0.25",
    "default_empty_value": "0",
    "behaviors": ["view", "cart", "buy"],
    "context_fields": ["price", "duration"],
    "business_codes": {},
}


def event(eid: str, uid="U1", iid="I1", behavior="view", ts=1000,
          **ctx) -> Event:
    return Event(eid, uid, iid, behavior, ts,
                 ctx or {"price": 10.0, "duration": 5.0})


class FixedPointTests(unittest.TestCase):
    def test_quantize_half_up(self) -> None:
        fp = FixedPoint.build(4, "0.25")
        self.assertEqual(fp.quantize(Decimal("0.3")), 2500)
        self.assertEqual(fp.quantize(Decimal("0.5")), 5000)
        self.assertEqual(fp.quantize(Decimal("-1.7")), -17500)
        self.assertEqual(fp.empty_value(None), 0)

    def test_invalid_scale_or_step(self) -> None:
        with self.assertRaises(ValueError):
            FixedPoint.build(-1, "1")
        with self.assertRaises(ValueError):
            FixedPoint.build(4, "0")
        with self.assertRaises(ValueError):
            FixedPoint.build(4, "-0.1")


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_event_validation(self) -> None:
        rows = [
            {"event_id": "E1", "user_id": "U1", "item_id": "I1",
             "behavior": "view", "ts_ms": 10, "context": {"price": 1.0}},
            {"event_id": "E1", "user_id": "U1", "item_id": "I1",
             "behavior": "view", "ts_ms": 11, "context": {}},          # dup id
            {"event_id": "E2", "user_id": "", "item_id": "I1",
             "behavior": "view", "ts_ms": 12, "context": {}},          # empty user
            {"event_id": "E3", "user_id": "U1", "item_id": "I1",
             "behavior": "view", "ts_ms": 10, "context": {}},          # not increasing
            {"event_id": "E4", "user_id": "U1", "item_id": "I1",
             "behavior": "explode", "ts_ms": 13, "context": {}},       # bad behavior
            {"event_id": "E5", "user_id": "U1", "item_id": "I1",
             "behavior": "view", "ts_ms": -1, "context": {}},          # negative ts
            {"event_id": "E6", "user_id": "U1", "item_id": "I1",
             "behavior": "view", "ts_ms": 14,
             "context": {"price": float("nan")}},                      # non-finite
        ]
        path = os.path.join(self.tmp, "events.jsonl")
        write_jsonl(path, rows)
        events, failures = load_events(path, BASE_CONFIG["behaviors"])
        self.assertEqual([e.event_id for e in events], ["E1"])
        codes = [f.code for f in failures]
        self.assertEqual(codes.count(CODE_EVENT_INVALID), 6)

    def test_def_cycle_and_param(self) -> None:
        defs = [
            {"name": "A", "side": "user", "window_ms": 100, "agg": "sum",
             "source_field": "price", "upstream": []},
            {"name": "B", "side": "user", "window_ms": 100, "agg": "sum",
             "source_field": None, "upstream": ["C"]},
            {"name": "C", "side": "user", "window_ms": 100, "agg": "sum",
             "source_field": None, "upstream": ["B"]},
            {"name": "D", "side": "user", "window_ms": 0, "agg": "sum",
             "source_field": "price", "upstream": []},
            {"name": "E", "side": "user", "window_ms": 10, "agg": "median",
             "source_field": "price", "upstream": []},
        ]
        path = os.path.join(self.tmp, "defs.json")
        write_json(path, defs)
        parsed, failures = load_feature_defs(path)
        codes = {f.code for f in failures}
        self.assertIn(CODE_DEF_CYCLE, codes)
        self.assertIn(CODE_DEF_PARAM, codes)
        self.assertTrue(all(d.valid for d in parsed if d.name == "A"))
        self.assertFalse(any(d.valid for d in parsed if d.name in ("B", "C", "D", "E")))


class ComputeTests(unittest.TestCase):
    def _computer(self, events: list[Event]) -> tuple[FeatureComputer, EventStore]:
        store = EventStore()
        store.rebuild(events)
        return FeatureComputer(store, FixedPoint.build(4, "0.25"), "0"), store

    def test_window_boundary_inclusive_and_order(self) -> None:
        events = [
            event("E2", ts=1000, price=10.0),
            event("E1", ts=900, price=20.0),
            event("E3", ts=1500, price=30.0),
        ]
        computer, _ = self._computer(events)
        feat = FeatureDef("f", "user", 600, "sum", "price", [], None)
        feat.topo_rank, feat.level = 0, 0
        res = computer.evaluate_base(feat, "U1", 0, 1500)
        # window [900,1500]; E1,E2,E3 all included; sum 60 -> 600000
        self.assertEqual(res.value, 600000)
        self.assertEqual(res.n_inputs, 3)

    def test_count_mean_last_and_empty_sources(self) -> None:
        events = [
            Event("E1", "U1", "I1", "view", 1000, {"price": 10.0}),
            Event("E2", "U1", "I1", "view", 1100, {"duration": 7.0}),
        ]
        computer, _ = self._computer(events)
        count = FeatureDef("c", "user", 5000, "count", "price", [], None)
        count.topo_rank, count.level = 0, 0
        mean = FeatureDef("m", "user", 5000, "mean", "price", [], None)
        mean.topo_rank, mean.level = 0, 0
        last = FeatureDef("l", "user", 5000, "last", "price", [], None)
        last.topo_rank, last.level = 0, 0
        missing = FeatureDef("x", "user", 5000, "sum", "nope", [], None)
        missing.topo_rank, missing.level = 0, 0
        self.assertEqual(computer.evaluate_base(count, "U1", 0, 1100).value, 10000)
        self.assertEqual(computer.evaluate_base(mean, "U1", 0, 1100).value, 100000)
        self.assertEqual(computer.evaluate_base(last, "U1", 0, 1100).value, 100000)
        empty = computer.evaluate_base(missing, "U1", 0, 1100)
        self.assertTrue(empty.empty)
        self.assertEqual(empty.empty_source, "no_source_value")
        no_entity = computer.evaluate_base(count, "U9", 0, 1100)
        self.assertTrue(no_entity.empty)
        self.assertEqual(no_entity.empty_source, "no_event")


if __name__ == "__main__":
    unittest.main(verbosity=2)
