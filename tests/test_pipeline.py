import os
import shutil
import tempfile
import unittest

from gsb.engine import Engine, catch_up, rebuild
from gsb.equivalence import equivalence_report
from gsb.errors import (
    E_BASE_VERSION,
    E_FORMAT_FIELD,
    E_PEER_GAP,
    E_SCHEMA_TOO_NEW,
    PipelineError,
)
from gsb.formats import binary as fmt_binary
from gsb.formats import jsonl as fmt_jsonl
from gsb.formats import legacy as fmt_legacy
from gsb.generator import generate_all
from gsb.migrate import migrate_event
from gsb.model import Event
from gsb.pipeline import load_all_formats, load_format, load_mixed, normalize_stream
from gsb.store import Store
from gsb.verifier import verify


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        generate_all(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def scenario(self, name):
        return os.path.join(self.tmp, name)


class TestFormats(Base):
    def test_three_format_equivalence_all_scenarios(self):
        for name in ("normal", "mixed", "upgrade", "conflict", "kill"):
            streams = load_all_formats(self.scenario(name))
            report = equivalence_report(streams)
            self.assertTrue(report["all_equal"],
                            f"{name}: {report['pairs']}")

    def test_binary_endian_width_bitmap_errors_carry_offset(self):
        event = Event(0, "peer-1", 0, "move", {"entity": "e0", "dx": 1, "dy": 0}, 0).finalize()
        frame = bytearray(fmt_binary.encode_frame(event))
        frame[3] = 0x7F
        with self.assertRaises(PipelineError) as ctx:
            fmt_binary.decode_frame(bytes(frame))
        self.assertEqual(ctx.exception.code, E_FORMAT_FIELD)
        self.assertEqual(ctx.exception.offset, 3)

        frame = bytearray(fmt_binary.encode_frame(event))
        frame[4] = 3
        with self.assertRaises(PipelineError) as ctx:
            fmt_binary.decode_frame(bytes(frame))
        self.assertEqual(ctx.exception.offset, 4)

        frame = bytearray(fmt_binary.encode_frame(event))
        frame[5] = 0xC0
        with self.assertRaises(PipelineError) as ctx:
            fmt_binary.decode_frame(bytes(frame))
        self.assertEqual(ctx.exception.offset, 5)

    def test_binary_roundtrip_widths_and_endians(self):
        event = Event(3, "peer-2", 7, "set_attr",
                      {"entity": "e1", "field": "hp", "value": 42}, 2).finalize()
        for endian in (fmt_binary.ENDIAN_LITTLE, fmt_binary.ENDIAN_BIG):
            for width in (2, 4):
                frame = fmt_binary.encode_frame(event, endian=endian, width=width)
                decoded = normalize_stream([fmt_binary.decode_frame(frame)])[0]
                self.assertEqual(decoded.identity_dict(), event.identity_dict())

    def test_legacy_normalization(self):
        line = fmt_legacy.dump_line(
            Event(5, "peer-3", 0, "destroy", {"entity": "e9"}, 4).finalize())
        event = normalize_stream([fmt_legacy.parse_line(line)])[0]
        self.assertEqual(event.tick, 5)
        self.assertEqual(event.peer_id, "peer-3")
        self.assertEqual(event.local_seq, 0)
        self.assertEqual(event.time_base, "tick")


class TestMigration(Base):
    def test_idempotent(self):
        for event in load_format(self.scenario("upgrade"), "jsonl"):
            again, audit = migrate_event(Event.from_dict(event.to_dict()))
            self.assertEqual(audit, [])
            self.assertEqual(again.finalize().compute_event_id(), event.compute_event_id())

    def test_schema_too_new(self):
        event = Event(0, "peer-1", 0, "move", {"entity": "e0", "dx": 1, "dy": 0},
                      0, schema_version=99)
        with self.assertRaises(PipelineError) as ctx:
            migrate_event(event)
        self.assertEqual(ctx.exception.code, E_SCHEMA_TOO_NEW)

    def test_audit_records_filled_fields(self):
        audit = []
        load_mixed(self.scenario("mixed"), audit)
        self.assertTrue(audit)
        for entry in audit:
            self.assertIn(entry["field"],
                          ("local_seq", "payload_encoding", "time_base"))
            self.assertIn("from_version", entry)
            self.assertIn("source_format", entry)


class TestEngine(Base):
    def _run(self, scenario, events=None):
        store_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, store_dir, True)
        store = Store(store_dir)
        engine = Engine(store)
        engine.run(events if events is not None
                   else load_format(self.scenario(scenario), "jsonl"))
        return store_dir, store

    def test_peer_gap(self):
        store = Store(tempfile.mkdtemp())
        engine = Engine(store)
        with self.assertRaises(PipelineError) as ctx:
            engine.run([Event(0, "peer-1", 1, "move",
                              {"entity": "e0", "dx": 1, "dy": 0}, 0).finalize()])
        self.assertEqual(ctx.exception.code, E_PEER_GAP)

    def test_unknown_base_version(self):
        store = Store(tempfile.mkdtemp())
        engine = Engine(store)
        with self.assertRaises(PipelineError) as ctx:
            engine.run([Event(0, "peer-1", 0, "move",
                              {"entity": "e0", "dx": 1, "dy": 0}, 9).finalize()])
        self.assertEqual(ctx.exception.code, E_BASE_VERSION)

    def test_catch_up_matches_target(self):
        _, store = self._run("normal")
        head = store.load_manifest()["head"]
        base = store.load_manifest()["versions"][0]["version"]
        report = catch_up(store, max(base, head - 3), head)
        self.assertTrue(report["match"])

    def test_compaction_preserves_rebuild(self):
        store_dir, store = self._run("normal")
        manifest = store.load_manifest()
        self.assertIn("compacted_base", manifest)
        self.assertTrue(store.has("chain_ledger.jsonl"))
        report = verify(store_dir)
        self.assertTrue(report["consistent"])

    def test_conflict_adjudication_traceable(self):
        store_dir, store = self._run("conflict")
        records = []
        with open(os.path.join(store_dir, "adjudication.jsonl"), encoding="utf-8") as fh:
            import json
            records = [json.loads(line) for line in fh if line.strip()]
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record["rule"], "field-lww-by-commit-order")
            self.assertTrue(record["winner_event_id"])
            self.assertTrue(record["loser_event_id"])
            self.assertIn(record["winner_format"], ("jsonl", "binary", "legacy"))
        # 任意最终字段值可回溯到胜出操作
        head = store.load_manifest()["head"]
        snapshot = store.load_snapshot(head)
        by_id = {}
        for v in range(store.load_manifest()["versions"][0]["version"], head + 1):
            for op in store.load_ops(v):
                by_id[op["event_id"]] = op
        for record in records:
            self.assertIn(record["winner_event_id"], by_id)
        self.assertTrue(snapshot.state_hash)

    def test_format_carriers_same_outcome(self):
        """同一内容三种承载分别合成，状态哈希必须一致。"""
        hashes = {}
        for fmt in ("jsonl", "binary", "legacy"):
            store_dir = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, store_dir, True)
            engine = Engine(Store(store_dir))
            engine.run(load_format(self.scenario("mixed"), fmt))
            hashes[fmt] = engine.hashes[-1]["state_hash"]
        self.assertEqual(len(set(hashes.values())), 1)


class TestRecovery(Base):
    def test_recover_reclaims_uncommitted_tail(self):
        store_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, store_dir, True)
        store = Store(store_dir)
        engine = Engine(store)
        events = load_format(self.scenario("normal"), "jsonl")
        engine.run([e for e in events if e.tick < 3])
        os.remove(os.path.join(store_dir, "SHUTDOWN_OK"))
        # 伪造未提交尾部与临时文件
        with open(os.path.join(store_dir, "snapshots", "v99.json"), "w") as fh:
            fh.write("{}")
        with open(os.path.join(store_dir, "logs", "v99.jsonl.tmp"), "w") as fh:
            fh.write("[")
        report = store.recover()
        self.assertFalse(report["previous_clean_shutdown"])
        self.assertIn("snapshots/v99.json", report["reclaimed"])
        self.assertIn("logs/v99.jsonl.tmp", report["reclaimed"])
        self.assertFalse(store.has("snapshots/v99.json"))

    def test_adopt_snapshot_dir_by_version_and_hash(self):
        store_dir, store = self._run("normal")
        incoming_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, incoming_dir, True)
        incoming = Store(incoming_dir)
        Engine(incoming).run(load_format(self.scenario("conflict"), "jsonl"))
        # incoming 头版本更低：拒绝盲目录入
        decision = store.adopt_snapshot_dir(incoming_dir)
        self.assertFalse(decision["adopted"])
        # 反向：更高版本戳且哈希校验通过则接受
        decision = incoming.adopt_snapshot_dir(store_dir)
        self.assertTrue(decision["adopted"])
        self.assertEqual(incoming.load_manifest()["head"], store.load_manifest()["head"])

    def _run(self, scenario):
        store_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, store_dir, True)
        store = Store(store_dir)
        Engine(store).run(load_format(self.scenario(scenario), "jsonl"))
        return store_dir, store


if __name__ == "__main__":
    unittest.main()
