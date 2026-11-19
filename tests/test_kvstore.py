"""键值索引与过期组件的单元/集成测试（标准库 unittest，离线可跑）。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest

from kvstore import codec, inject as inject_mod
from kvstore.engine import Engine
from kvstore.errors import (
    ExpiredError,
    IndexError as KvIndexError,
    KvError,
    RecordError,
    ReplayError,
    SnapshotError,
    TraceError,
    VersionError,
    VisibleError,
)
from kvstore.replay import replay_authoritative, verify_replay_after_restore, verify_visible
from kvstore.session import Session
from kvstore.snapshot import open_snapshot, save_snapshot
from kvstore.storage import Storage
from kvstore.trace import load_trace, operations_digest, write_trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def tmp_db(name: str) -> str:
    workdir = tempfile.mkdtemp(prefix="kvtest_")
    return os.path.join(workdir, name)


def remove_db(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = path + suffix
        if os.path.exists(target):
            os.remove(target)


class CodecTests(unittest.TestCase):
    def test_valid_and_invalid_keys(self) -> None:
        self.assertEqual(codec.validate_key("user/1/a-b_c"), "user/1/a-b_c")
        for bad in ("", "/leading", "trailing/", "bad segment", "中文", "a//b"):
            with self.assertRaises(RecordError) as ctx:
                codec.validate_key(bad)
            self.assertEqual(ctx.exception.code, "E_RECORD")
            self.assertEqual(ctx.exception.details["field"], "key")

    def test_prefixes_are_segment_aligned(self) -> None:
        self.assertEqual(codec.prefixes_of("user/1/name"),
                         ["user", "user/1", "user/1/name"])

    def test_record_validation_negative_expire(self) -> None:
        record = {"key": "k/1", "value": "v", "expire_ms": -1, "write_seq": 1}
        with self.assertRaises(RecordError) as ctx:
            codec.validate_record(record)
        self.assertEqual(ctx.exception.details["field"], "expire_ms")

    def test_non_monotonic_write_seq(self) -> None:
        codec.check_write_seq_monotonic(None, 1, "k/1")
        with self.assertRaises(RecordError) as ctx:
            codec.check_write_seq_monotonic(5, 5, "k/1")
        self.assertEqual(ctx.exception.details["field"], "write_seq")


class StorageEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = tmp_db("engine.db")
        self.storage = Storage(self.db)
        self.engine = Engine(self.storage)

    def tearDown(self) -> None:
        self.storage.close()
        remove_db(self.db)

    def test_put_overwrite_does_not_duplicate_index(self) -> None:
        self.engine.put("user/1/name", "a", 0, 1, 10)
        self.engine.put("user/1/name", "b", 0, 2, 20)
        rows = self.storage.all_index()
        refs = [r["ref_key"] for r in rows if r["prefix"] == "user/1/name"]
        self.assertEqual(refs, ["user/1/name"])
        version = self.storage.get_index_version("user/1/name", "user/1/name")
        self.assertEqual(version, 2)
        self.assertEqual(self.storage.get_count("user"), 1)
        self.engine.verify_index_consistency()

    def test_put_rejects_non_monotonic_seq(self) -> None:
        self.engine.put("k/1", "v", 0, 5, 10)
        with self.assertRaises(RecordError):
            self.engine.put("k/1", "v2", 0, 5, 20)

    def test_expiration_uses_only_injected_clock(self) -> None:
        self.engine.put("k/1", "v", 500, 1, 100)
        self.assertEqual(self.engine.get("k/1", 499)["status"], "OK")
        expired = self.engine.get("k/1", 500)
        self.assertEqual(expired["status"], "E_EXPIRED")
        self.assertTrue(all(step.get("system_clock_used") is not True
                            for step in expired["basis"]))
        # 未 PURGE 前记录仍在但不可见、不计入计数
        self.assertEqual(self.engine.visible_count("k", 500), 0)
        self.assertIsNotNone(self.storage.get_record("k/1"))

    def test_purge_removes_record_and_index_and_bumps_version(self) -> None:
        self.engine.put("k/1", "v", 500, 1, 100)
        before = self.engine.keyspace_version
        result = self.engine.purge(500)
        self.assertEqual(result["purged_keys"], ["k/1"])
        self.assertEqual(self.engine.keyspace_version, before + 1)
        self.assertIsNone(self.storage.get_record("k/1"))
        self.assertEqual(self.storage.all_index(), [])
        self.assertEqual(self.storage.all_counts(), {})

    def test_delete_updates_counts(self) -> None:
        self.engine.put("user/1/a", "x", 0, 1, 10)
        self.engine.put("user/2/b", "y", 0, 2, 10)
        self.engine.delete("user/1/a", 20)
        self.assertEqual(self.storage.get_count("user"), 1)
        self.engine.verify_index_consistency()

    def test_eternal_key_expire_zero(self) -> None:
        self.engine.put("k/1", "v", 0, 1, 10)
        for clock in (0, 10 ** 9):
            self.assertEqual(self.engine.get("k/1", clock)["status"], "OK")


class InitialLoadTests(unittest.TestCase):
    def test_initial_load_all_or_nothing(self) -> None:
        db = tmp_db("initial.db")
        storage = Storage(db)
        engine = Engine(storage)
        good = [{"key": "a/1", "value": "v", "expire_ms": 0,
                 "write_seq": 1, "format_version": 1}]
        bad = [{"key": "a/1", "value": "v", "expire_ms": 0,
                "write_seq": 1, "format_version": 1},
               {"key": "BAD KEY", "value": "v", "expire_ms": 0,
                "write_seq": 1, "format_version": 1}]
        engine.load_initial_records(good)
        with self.assertRaises(RecordError):
            engine.load_initial_records(bad)
        storage.close()
        remove_db(db)


class TraceTests(unittest.TestCase):
    def test_sample_trace_deterministic_digest(self) -> None:
        path = os.path.join(DATA, "sample_trace.jsonl")
        first = operations_digest(load_trace(path))
        second = operations_digest(load_trace(path))
        self.assertEqual(first, second)

    def test_non_contiguous_seq_rejected(self) -> None:
        ops = [
            {"op_seq": 1, "op": "PURGE", "clock": 1},
            {"op_seq": 3, "op": "PURGE", "clock": 2},
        ]
        path = os.path.join(tempfile.mkdtemp(), "t.jsonl")
        write_trace(path, ops)
        with self.assertRaises(TraceError) as ctx:
            load_trace(path)
        self.assertEqual(ctx.exception.code, "E_TRACE")
        self.assertEqual(ctx.exception.details["op_seq"], 3)

    def test_clock_regression_and_missing_put_field(self) -> None:
        workdir = tempfile.mkdtemp()
        regress = [{"op_seq": 1, "op": "PURGE", "clock": 5},
                   {"op_seq": 2, "op": "PURGE", "clock": 4}]
        path = os.path.join(workdir, "r.jsonl")
        write_trace(path, regress)
        with self.assertRaises(TraceError):
            load_trace(path)
        missing_put = [{"op_seq": 1, "op": "PUT", "clock": 1, "key": "k/1"}]
        path2 = os.path.join(workdir, "p.jsonl")
        write_trace(path2, missing_put)
        with self.assertRaises(TraceError) as ctx:
            load_trace(path2)
        self.assertEqual(ctx.exception.details["field"], "value")


class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace_path = os.path.join(DATA, "sample_trace.jsonl")
        with open(os.path.join(DATA, "initial_records.json"), encoding="utf-8") as h:
            self.initial = json.load(h)
        self.operations = load_trace(self.trace_path)

    def test_full_run_consistency_and_snapshot_replay(self) -> None:
        db = tmp_db("full.db")
        session = Session(db, initial_records=self.initial)
        session.run(self.operations)
        session.engine.verify_index_consistency(clock=1300)
        snap = os.path.join(os.path.dirname(db), "snap.json")
        session.snapshot(snap)

        restored_db = tmp_db("restored.db")
        storage = Storage(restored_db)
        engine = Engine(storage)
        opened = open_snapshot(snap, engine)
        replay_db = tmp_db("replay.db")
        result = verify_replay_after_restore(
            engine, self.operations, self.initial,
            opened["conclusions"], replay_db,
        )
        self.assertTrue(result["ok"])
        remove_db(replay_db)
        storage.close()
        session.close()
        remove_db(db)
        remove_db(restored_db)

    def test_visible_matches_authority(self) -> None:
        db = tmp_db("vis.db")
        session = Session(db, initial_records=self.initial)
        session.run(self.operations)
        prefixes = sorted(session.engine.storage.all_counts().keys())
        replay_db = tmp_db("vis_replay.db")
        result = verify_visible(
            session.engine, self.operations, self.initial,
            replay_db, clock=1300, prefixes=prefixes,
        )
        self.assertTrue(result["ok"])
        remove_db(replay_db)
        session.close()
        remove_db(db)

    def test_concurrent_rebuild_never_shows_intermediate_state(self) -> None:
        db = tmp_db("rebuild.db")
        session = Session(db, initial_records=self.initial)
        pre = [op for op in self.operations if op["op_seq"] < 8]
        session.run(pre)
        barrier = threading.Event()
        session.rebuild_worker.start(800, barrier=barrier)
        while session.rebuild_worker.phase != "BUILDING":
            threading.Event().wait(0.005)
        # 重建挂起期间：正式表查询必须持续可用且自洽
        mid = session.engine.query_prefix("user", 750)
        self.assertEqual(mid["visible_count"], len(mid["keys"]))
        session.engine.put("user/5/name", "erin", 3000, 99, 760)
        barrier.set()
        result = session.finish_rebuild()
        self.assertIn("user/5/name", result["delta_keys"])
        session.engine.verify_index_consistency(clock=760)
        # 重建后条目版本不回退
        self.assertGreaterEqual(
            session.storage.get_index_version("user/5/name", "user/5/name"), 1
        )
        session.close()
        remove_db(db)

    def test_repeated_rebuild_is_stable(self) -> None:
        db = tmp_db("stable.db")
        session = Session(db, initial_records=self.initial)
        session.run(self.operations)
        before = session.engine.export_state()
        session.rebuild_worker.start(1300)
        session.finish_rebuild()
        after = session.engine.export_state()
        self.assertEqual(
            sorted((r["prefix"], r["ref_key"], r["index_value"], r["entry_version"])
                   for r in before["index_entries"]),
            sorted((r["prefix"], r["ref_key"], r["index_value"], r["entry_version"])
                   for r in after["index_entries"]),
        )
        session.close()
        remove_db(db)


class SnapshotAndInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = tempfile.mkdtemp(prefix="kvsnap_")
        self.trace_path = os.path.join(DATA, "sample_trace.jsonl")
        with open(os.path.join(DATA, "initial_records.json"), encoding="utf-8") as h:
            self.initial = json.load(h)
        self.operations = load_trace(self.trace_path)
        self.db = os.path.join(self.workdir, "main.db")
        session = Session(self.db, initial_records=self.initial)
        session.run(self.operations)
        self.snap = os.path.join(self.workdir, "snap.json")
        session.snapshot(self.snap)
        session.close()

    def test_snapshot_digest_mismatch(self) -> None:
        bad = inject_mod.inject_snapshot_digest(self.snap, self.workdir)
        storage = Storage(os.path.join(self.workdir, "baddb.db"))
        with self.assertRaises(SnapshotError) as ctx:
            open_snapshot(bad, Engine(storage))
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        storage.close()

    def test_index_version_rollback(self) -> None:
        rb = inject_mod.inject_index_rollback(self.snap, self.workdir)
        storage = Storage(os.path.join(self.workdir, "rbdb.db"))
        with self.assertRaises(VersionError) as ctx:
            open_snapshot(rb, Engine(storage))
        self.assertEqual(ctx.exception.code, "E_VERSION")
        storage.close()

    def test_trace_corruption(self) -> None:
        corrupt = inject_mod.inject_trace_corrupt(self.trace_path, self.workdir)
        with self.assertRaises(TraceError) as ctx:
            load_trace(corrupt)
        self.assertEqual(ctx.exception.code, "E_TRACE")


class VisibleDivergenceTests(unittest.TestCase):
    def test_e_visible_reports_prefix_and_conclusions(self) -> None:
        trace_path = os.path.join(DATA, "sample_trace.jsonl")
        with open(os.path.join(DATA, "initial_records.json"), encoding="utf-8") as h:
            initial = json.load(h)
        operations = load_trace(trace_path)
        db = tmp_db("div.db")
        session = Session(db, initial_records=initial)
        session.run(operations)
        # 人为破坏快速缓存计数，制造快速/权威分歧
        session.storage.set_count("user", 999)
        replay_db = tmp_db("div_replay.db")
        with self.assertRaises(VisibleError) as ctx:
            verify_visible(session.engine, operations, initial,
                           replay_db, clock=1300, prefixes=["user"])
        details = ctx.exception.details
        self.assertEqual(details["prefix"], "user")
        self.assertIn("quick", details)
        self.assertIn("authoritative", details)
        self.assertIn("divergence_op_seq", details)
        remove_db(replay_db)
        session.close()
        remove_db(db)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ConcurrentMutationStressTests(unittest.TestCase):
    def test_mixed_mutations_during_rebuild(self) -> None:
        db = tmp_db("stress.db")
        storage = Storage(db)
        engine = Engine(storage)
        from kvstore.rebuild import RebuildWorker

        worker = RebuildWorker(storage)
        engine.delta_listener = worker.on_delta
        for i in range(20):
            engine.put(f"root/{i % 5}/k{i}", f"v{i}", 0, i + 1, 100)
        barrier = threading.Event()
        worker.start(100, barrier=barrier)
        while worker.phase != "BUILDING":
            threading.Event().wait(0.005)
        # 重建挂起期间混合写入、覆盖、删除与逻辑过期清理
        engine.put("root/9/new", "x", 0, 100, 110)
        engine.put("root/0/k0", "overwritten", 0, 101, 110)
        engine.delete("root/1/k1", 110)
        engine.put("root/2/ephemeral", "e", 150, 102, 120)
        engine.purge(200)  # ephemeral 到期被物理移除
        barrier.set()
        worker.join()
        engine.verify_index_consistency(clock=200)
        # 不变式：存在的键都能在每个前缀路径上找到索引，反之亦然
        live_keys = {row["key"] for row in storage.all_records()}
        indexed = {(r["prefix"], r["ref_key"]) for r in storage.all_index()}
        expected = {(p, k) for k in live_keys for p in codec.prefixes_of(k)}
        self.assertEqual(indexed, expected)
        storage.close()
        remove_db(db)
