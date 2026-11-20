"""End-to-end behaviour tests built on small handcrafted deterministic inputs."""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from featflow.common import (  # noqa: E402
    CODE_FEAT_BUDGET_EXHAUSTED,
    CODE_GLOBAL_BUDGET_EXHAUSTED,
    CODE_SNAPSHOT_STALE,
    read_json,
    write_json,
    write_jsonl,
)
from featflow.config import PipelineConfig  # noqa: E402
from featflow.engine import PipelineEngine  # noqa: E402
from featflow.snapshot import SnapshotStore  # noqa: E402
from featflow.verifier import IncrementalVerifier, descendants  # noqa: E402

BEHAVIORS = ["view", "cart", "buy"]

FEATURES = [
    {"name": "u_price_sum", "side": "user", "window_ms": 5000,
     "agg": "sum", "source_field": "price", "upstream": [], "empty_value": 0},
    {"name": "u_count", "side": "user", "window_ms": 8000,
     "agg": "count", "source_field": "duration", "upstream": [], "empty_value": 0},
    {"name": "u_score", "side": "user", "window_ms": 9000,
     "agg": "mean", "source_field": None,
     "upstream": ["u_price_sum", "u_count"], "empty_value": 0},
    {"name": "i_count", "side": "item", "window_ms": 7000,
     "agg": "count", "source_field": "price", "upstream": [], "empty_value": 0},
]

LABELS = [
    {"name": "hot", "trigger_behaviors": ["buy"],
     "decision_feature": "u_score", "threshold": 50, "positive_op": "ge",
     "observed_features": ["u_score", "u_price_sum", "u_count"], "delay_ms": 1000},
]


def make_tree(tmp: str, *, events: list[dict], features=None, labels=None,
              config=None, injections=None, queries=None) -> str:
    write_jsonl(os.path.join(tmp, "events.jsonl"), events)
    write_json(os.path.join(tmp, "feature_defs.json"), features or FEATURES)
    write_json(os.path.join(tmp, "label_defs.json"), labels or LABELS)
    import yaml
    with open(os.path.join(tmp, "pipeline.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(config or base_config(), fh, sort_keys=True)
    write_json(os.path.join(tmp, "injections.json"),
               injections or {"rules": []})
    write_json(os.path.join(tmp, "query_points.json"), queries or [])
    return tmp


def base_config(**overrides) -> dict:
    cfg = {
        "thread_concurrency": 1,
        "global_event_budget": -1,
        "feature_point_quota": -1,
        "cache_block_limit": 2,
        "snapshot_commit_every": 5,
        "fixed_point_scale": 4,
        "quant_step": "0.25",
        "default_empty_value": "0",
        "behaviors": BEHAVIORS,
        "context_fields": ["price", "duration"],
        "business_codes": {},
    }
    cfg.update(overrides)
    return cfg


def make_events(n: int = 12) -> list[dict]:
    rows = []
    ts = 1000
    for i in range(n):
        ts += 200
        rows.append({
            "event_id": f"E{i:03d}",
            "user_id": f"U{i % 3:02d}",
            "item_id": f"I{i % 4:02d}",
            "behavior": BEHAVIORS[i % 3],
            "ts_ms": ts,
            "context": {"price": 10.0 + i, "duration": 1.0 + (i % 5)},
        })
    return rows


def run_engine(tmp: str, out: str, concurrency: int | None = None,
               injections: bool = False):
    cfg = PipelineConfig.load(os.path.join(tmp, "pipeline.yaml"))
    if concurrency is not None:
        cfg.thread_concurrency = concurrency
    snap = SnapshotStore(os.path.join(out, "snap"))
    snap.reset()
    engine = PipelineEngine(cfg, snap)
    events, defs, labels, rules = engine.load_inputs(
        os.path.join(tmp, "events.jsonl"),
        os.path.join(tmp, "feature_defs.json"),
        os.path.join(tmp, "label_defs.json"),
        os.path.join(tmp, "injections.json"))
    if injections:
        events = engine._apply_load_injections(events, rules)
    queries = read_json(os.path.join(tmp, "query_points.json"))
    result = engine.run(events, defs, labels, query_points=queries)
    return cfg, events, defs, labels, result, engine


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        make_tree(self.tmp, events=make_events(),
                  config=base_config(global_event_budget=10,
                                     feature_point_quota=4,
                                     thread_concurrency=3))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)
        shutil.rmtree(self.out)

    def test_two_layer_budget_and_ledger(self) -> None:
        cfg, events, defs, labels, result, engine = run_engine(self.tmp, self.out)
        codes = [f["code"] for f in result["failures"]]
        self.assertIn(CODE_FEAT_BUDGET_EXHAUSTED, codes)
        self.assertIn(CODE_GLOBAL_BUDGET_EXHAUSTED, codes)
        # ledger is a contiguous audit trail with monotonic sequence numbers
        seqs = [e["seq"] for e in result["ledger"]]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))
        charged = [e for e in result["ledger"] if e["decision"] == "charged"]
        self.assertEqual(sum(e["charge"] for e in charged), 10)
        self.assertTrue(result["pending_queue"])
        # once aborted, a feature never charges again
        for name, used in result["feature_used"].items():
            self.assertLessEqual(used, 4)


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        make_tree(self.tmp, events=make_events(),
                  config=base_config(cache_block_limit=1,
                                     thread_concurrency=4))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)
        shutil.rmtree(self.out)

    def test_lru_eviction_and_recompute(self) -> None:
        _, _, _, _, result, _ = run_engine(self.tmp, self.out)
        cache = result["cache"]
        self.assertGreater(cache["evictions"], 0)
        self.assertGreater(cache["recomputes"], 0)
        # LRU tie-break is deterministic: evict event has the smallest access seq
        evicts = [e for e in cache["events"] if e["op"] == "evict"]
        for ev in evicts:
            self.assertIn(ev["block_key"], cache["registry"].get(
                str(ev["point_seq"]), []))


class SnapshotTests(unittest.TestCase):
    def test_stale_detection(self) -> None:
        snap = SnapshotStore(tempfile.mkdtemp())
        expected = {"definition_hash": "a", "config_hash": "b",
                    "cursor_event_id": "E1"}
        drift = dict(expected, definition_hash="z")
        failure = snap.check_stale(expected, drift)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.code, CODE_SNAPSHOT_STALE)
        self.assertIsNone(snap.check_stale(expected, dict(expected)))


class IncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        make_tree(self.tmp, events=make_events())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)
        shutil.rmtree(self.out)

    def test_invalidation_scope_and_equivalence(self) -> None:
        cfg, events, defs, labels, base, engine = run_engine(self.tmp, self.out)
        # Edit: change window of u_price_sum and correct one event.
        final_defs = copy.deepcopy(defs)
        for d in final_defs:
            if d.name == "u_price_sum":
                d.window_ms = 6500
        final_events = copy.deepcopy(events)
        final_events[2] = final_events[2].with_correction(
            ts_ms=final_events[2].ts_ms, context={"price": 99.0, "duration": 2.0})

        verifier = IncrementalVerifier(cfg)
        verifier._changed_fields = {
            final_events[2].event_id: {"price": True}}
        invalid = verifier.invalidation_set(
            final_defs, final_events, {final_events[2].event_id},
            {"u_price_sum"})
        # The definition change roots u_price_sum -> u_score; the backfilled
        # event shifts windows for every base feature on the touched user and
        # item entities, which covers all four base features in this fixture.
        self.assertEqual(invalid,
                         {"u_price_sum", "u_count", "u_score", "i_count"})

        snap_full = SnapshotStore(os.path.join(self.out, "full"))
        full = PipelineEngine(cfg, snap_full).run(
            final_events, final_defs, labels, query_points=[])
        snap_inc = SnapshotStore(os.path.join(self.out, "inc"))
        inc = verifier.replay_incremental(base, final_defs, final_events,
                                          labels, [], invalid, snap_inc)
        ok, diff, fails = verifier.compare_runs(
            full, inc, os.path.join(self.out, "eq.json"))
        self.assertTrue(ok, f"unexpected diff {diff}")
        self.assertEqual(inc["invalidation_ledger"]["invalidated_features"],
                         sorted({"u_price_sum", "u_count", "u_score", "i_count"}))

    def test_definition_only_change_preserves_unrelated_features(self) -> None:
        cfg, events, defs, labels, base, engine = run_engine(self.tmp, self.out)
        # Only change the window of u_price_sum. No event is backfilled, so
        # u_count and i_count have no reason to be recomputed.
        final_defs = copy.deepcopy(defs)
        for d in final_defs:
            if d.name == "u_price_sum":
                d.window_ms = 7777
        verifier = IncrementalVerifier(cfg)
        invalid = verifier.invalidation_set(
            final_defs, events, set(), {"u_price_sum"})
        self.assertEqual(invalid, {"u_price_sum", "u_score"})
        snap_full = SnapshotStore(os.path.join(self.out, "full2"))
        full = PipelineEngine(cfg, snap_full).run(
            events, final_defs, labels, query_points=[])
        snap_inc = SnapshotStore(os.path.join(self.out, "inc2"))
        inc = verifier.replay_incremental(base, final_defs, events,
                                          labels, [], invalid, snap_inc)
        ok, diff, _ = verifier.compare_runs(
            full, inc, os.path.join(self.out, "eq2.json"))
        self.assertTrue(ok, diff)
        self.assertEqual(
            set(inc["invalidation_ledger"]["reused_features"]),
            {"u_count", "i_count"})
        self.assertGreater(
            inc["invalidation_ledger"]["reused_block_count"], 0)

    def test_backfill_order_rejection(self) -> None:
        cfg, events, _, _, _, _ = run_engine(self.tmp, self.out)
        verifier = IncrementalVerifier(cfg)
        cursor_id = max(events, key=lambda e: e.ts_ms).event_id
        bad = events[0].with_correction(ts_ms=1)
        failure = verifier.check_backfill_order(events, bad, cursor_id)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.code, "E_BACKFILL_ORDER")


class ConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.out_a = tempfile.mkdtemp()
        self.out_b = tempfile.mkdtemp()
        make_tree(self.tmp, events=make_events(),
                  config=base_config(global_event_budget=20,
                                     feature_point_quota=8))

    def tearDown(self) -> None:
        for d in (self.tmp, self.out_a, self.out_b):
            shutil.rmtree(d)

    def test_serial_equals_concurrent(self) -> None:
        _, _, _, _, serial, _ = run_engine(self.tmp, self.out_a, concurrency=1)
        _, _, _, _, concurrent, _ = run_engine(self.tmp, self.out_b, concurrency=8)
        for artifact in ("features", "samples", "ledger", "failures"):
            self.assertEqual(serial[artifact], concurrent[artifact], artifact)
        self.assertEqual(serial["cache"]["registry"], concurrent["cache"]["registry"])


class InjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        events = make_events(8)
        injections = {"rules": [
            {"injection_id": "INJ-1", "event_id": "E002", "stage": "load",
             "kind": "non_finite_context", "context_field": "price",
             "bad_ts_ms": -1},
            {"injection_id": "INJ-2", "event_id": "E004", "stage": "load",
             "kind": "unknown_behavior", "context_field": "",
             "bad_ts_ms": -1},
        ]}
        make_tree(self.tmp, events=events, injections=injections)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)
        shutil.rmtree(self.out)

    def test_faults_are_isolated_and_recorded(self) -> None:
        _, events, _, _, result, _ = run_engine(self.tmp, self.out, injections=True)
        ids = {e.event_id for e in events}
        self.assertNotIn("E002", ids)
        self.assertNotIn("E004", ids)
        # other events survive
        self.assertIn("E000", ids)
        hit = {f["injection_id"] for f in result["failures"] if f["injection_id"]}
        self.assertEqual(hit, {"INJ-1", "INJ-2"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
