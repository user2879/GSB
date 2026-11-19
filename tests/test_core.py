from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shardrep.audit import StructuredLogger
from shardrep.canonical import canonical_json_bytes, sha256_file
from shardrep.content import content_digest
from shardrep.errors import (
    IncrementalMismatch,
    ReplayError,
    SnapshotError,
    SpecValidationError,
)
from shardrep.ledger import Ledger
from shardrep.packing import PackStore, PackTask, build_pack_tasks, compare_artifact_sets
from shardrep.propagation import propagate_invalidation
from shardrep.registry import ReplicaRegistry
from shardrep.repair import apply_and_report, block_replica_digest
from shardrep.sample import generate_sample
from shardrep.scheduler import PackScheduler
from shardrep.snapshot import SnapshotManager, compare_replay, tamper_snapshot_copy
from shardrep.spec import load_spec, spec_manifest_digest


ROOT = Path(__file__).resolve().parents[1]


class SampleCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.spec_path = generate_sample(self.root, force=True)
        self.raw = json.loads(self.spec_path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def load(self):
        return load_spec(self.spec_path)


class SpecValidationTests(SampleCase):
    def test_duplicate_shard_id_is_e_spec_with_field_path(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][0]["shards"][1]["shard_id"] = raw["nodes"][0]["shards"][0]["shard_id"]
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(self.spec_path, raw_override=raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("shards[1].shard_id", caught.exception.field_path)
        self.assertEqual(caught.exception.shard_id, raw["nodes"][0]["shards"][0]["shard_id"])

    def test_non_positive_bytes_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][2]["shards"][0]["bytes"] = 0
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(self.spec_path, raw_override=raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("bytes", caught.exception.field_path)

    def test_block_gap_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["blocks"][1]["seq_start"] = 4
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(self.spec_path, raw_override=raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("blocks[1].seq_start", caught.exception.field_path)

    def test_missing_relation_target_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["relations"][0]["target"] = "does-not-exist"
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(self.spec_path, raw_override=raw)
        self.assertEqual(caught.exception.shard_id, "does-not-exist")
        self.assertIn("relations[0].target", caught.exception.field_path)

    def test_cycle_is_e_spec_with_e_cycle_cause(self):
        raw = copy.deepcopy(self.raw)
        raw["relations"].append({"kind": "mirror", "source": "s7", "target": "s1"})
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(self.spec_path, raw_override=raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.cause_code, "E_CYCLE")

    def test_absolute_path_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][0]["directory"] = str((self.root / "samples" / "nodes" / "node-a").resolve())
        with self.assertRaises(SpecValidationError) as caught:
            load_spec(self.spec_path, raw_override=raw)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("directory", caught.exception.field_path)


class IntegrityAndPropagationTests(SampleCase):
    def test_digest_is_stable_and_injection_only_mutates_shard_file(self):
        spec = self.load()
        before_spec = self.spec_path.read_bytes()
        before = content_digest(spec.shard_by_id["node-a:s2"].path.read_bytes())
        self.assertEqual(before, content_digest(spec.shard_by_id["node-a:s2"].path.read_bytes()))
        registry = ReplicaRegistry.load(spec)
        injection = registry.inject(shard_id="node-a:s2", kind="truncate")
        self.assertEqual(injection.logical_id, "s2")
        self.assertEqual(len(registry.integrity_issues), 1)
        self.assertEqual(registry.integrity_issues[0]["code"], "E_INTEGRITY")
        self.assertEqual(registry.by_shard("node-a:s2").status, "truncated")
        self.assertEqual(self.spec_path.read_bytes(), before_spec)

    def test_three_injection_kinds_are_detected(self):
        for kind in ("truncate", "flip", "replace"):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as temporary:
                    path = generate_sample(temporary, force=True)
                    spec = load_spec(path)
                    registry = ReplicaRegistry.load(spec)
                    injection = registry.inject(shard_id="node-b:s4", kind=kind, offset=5)
                    self.assertNotEqual(injection.after_digest, injection.before_digest)
                    self.assertNotEqual(registry.by_shard("node-b:s4").status, "ok")

    def test_delete_is_registered_as_missing(self):
        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        registry.inject(shard_id="node-c:s6", kind="delete")
        self.assertEqual(registry.by_shard("node-c:s6").status, "missing")

    def test_propagation_and_trace_are_stable_across_declaration_order(self):
        spec = self.load()
        first = propagate_invalidation(spec, ["s2"])
        self.assertEqual(first.affected, ("s2", "s4", "s5", "s7"))
        self.assertTrue(first.entries[1].direct)
        self.assertFalse(first.entries[2].direct)
        self.assertEqual(first.entries[-1].cause[0].path, ("s2", "s4", "s5", "s7"))

        raw = copy.deepcopy(self.raw)
        raw["nodes"] = list(reversed(raw["nodes"]))
        raw["relations"] = list(reversed(raw["relations"]))
        reordered = load_spec(self.spec_path, raw_override=raw)
        second = propagate_invalidation(reordered, ["s2"])
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(
            canonical_json_bytes(first.as_dict()),
            canonical_json_bytes(second.as_dict()),
        )

    def test_affected_blocks_are_marked_for_all_affected_logicals(self):
        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        registry.inject(shard_id="node-b:s4", kind="flip", offset=3)
        propagation = propagate_invalidation(spec, ["s4"])
        registry.mark_affected(propagation.affected)
        self.assertEqual(propagation.affected, ("s4", "s5", "s7"))
        self.assertEqual(len(registry.pending_records()), 3 * len(propagation.affected))
        self.assertEqual(propagation.affected_blocks, ("block-beta", "block-gamma"))


class IncrementalRepackTests(SampleCase):
    def test_only_affected_tasks_and_untouched_artifacts_are_byte_stable(self):
        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        pack_dir = self.root / "packs"
        store = PackStore(pack_dir, spec)
        baseline = store.repack_all_direct()
        before_files = {
            logical_id: sha256_file(pack_dir / entry["artifact"])
            for logical_id, entry in baseline["artifacts"].items()
        }
        registry.inject(shard_id="node-b:s4", kind="flip", offset=7)
        propagation = propagate_invalidation(spec, ["s4"])
        quotas = {node.node_id: node.pack_quota for node in spec.nodes}
        tasks = build_pack_tasks(spec, propagation.affected, quotas=quotas)
        result = PackScheduler(
            store=store,
            global_slots=2,
            node_quotas=quotas,
            run_id="incremental",
        ).run(tasks)
        self.assertTrue(all(entry.outcome == "success" for entry in result.entries))
        self.assertEqual({task.logical_id for task in tasks}, {"s4", "s5", "s7"})
        after = store.manifest()
        for logical_id in ("s0", "s1", "s2", "s3", "s6"):
            self.assertEqual(baseline["artifacts"][logical_id]["version"], after["artifacts"][logical_id]["version"])
            self.assertEqual(baseline["artifacts"][logical_id]["content_digest"], after["artifacts"][logical_id]["content_digest"])
            self.assertEqual(before_files[logical_id], sha256_file(pack_dir / after["artifacts"][logical_id]["artifact"]))
        full_store = PackStore(self.root / "full", spec)
        full_store.repack_all_direct()
        digest = block_replica_digest(spec, registry)
        comparison = compare_artifact_sets(
            after,
            full_store.manifest(),
            incremental_registry_digest=digest,
            full_registry_digest=digest,
        )
        self.assertTrue(comparison["ok"])

    def test_e_incr_reports_first_shard_and_field(self):
        spec = self.load()
        left_store = PackStore(self.root / "left", spec)
        right_store = PackStore(self.root / "right", spec)
        left_store.repack_all_direct()
        right_store.repack_all_direct()
        left = left_store.manifest()
        right = right_store.manifest()
        left["artifacts"]["s0"]["bytes"] += 1
        with self.assertRaises(IncrementalMismatch) as caught:
            compare_artifact_sets(
                left,
                right,
                incremental_registry_digest="same",
                full_registry_digest="same",
            )
        self.assertEqual(caught.exception.code, "E_INCR")
        self.assertEqual(caught.exception.shard_id, "s0")
        self.assertEqual(caught.exception.field_path, "bytes")


class SchedulerTests(SampleCase):
    def _task(self, logical_id: str, node: str, **overrides) -> PackTask:
        spec = self.load()
        logical = spec.logical_shards[logical_id]
        values = dict(
            task_id=f"manual:{logical_id}:{node}",
            node_id=node,
            logical_id=logical_id,
            seq=logical.seq,
            bytes=logical.byte_count,
            timeout_ms=2_000,
            simulate_ms=20,
            max_retries=0,
        )
        values.update(overrides)
        return PackTask(**values)

    def test_global_slot_and_per_node_quota_are_never_exceeded(self):
        spec = self.load()
        store = PackStore(self.root / "packs", spec)
        tasks = [self._task(logical_id, "node-a") for logical_id in ("s0", "s1", "s2", "s3")]
        result = PackScheduler(
            store=store,
            global_slots=4,
            node_quotas={"node-a": 1},
            run_id="quota",
        ).run(tasks)
        self.assertEqual(result.max_per_node_observed["node-a"], 1)
        self.assertEqual(result.max_global_observed, 1)
        self.assertTrue(all(entry.queue_ms >= 0 for entry in result.entries))
        self.assertEqual(len(result.entries), 4)

        tasks = [self._task(logical_id, node) for logical_id, node in (("s0", "node-a"), ("s1", "node-b"), ("s2", "node-c"), ("s3", "node-a"))]
        result = PackScheduler(
            store=store,
            global_slots=2,
            node_quotas={"node-a": 2, "node-b": 2, "node-c": 2},
            run_id="global",
        ).run(tasks)
        self.assertLessEqual(result.max_global_observed, 2)
        self.assertGreaterEqual(result.max_global_observed, 1)

    def test_real_timeout_retries_and_never_bypasses_budget(self):
        spec = self.load()
        ledger = Ledger(self.root / "ledger.sqlite3")
        store = PackStore(self.root / "packs", spec)
        task = self._task("s0", "node-a", timeout_ms=8, simulate_ms=120, max_retries=1)
        result = PackScheduler(
            store=store,
            global_slots=1,
            node_quotas={"node-a": 1},
            ledger=ledger,
            run_id="timeout",
        ).run([task])
        self.assertEqual(len(result.entries), 2)
        self.assertTrue(all(entry.outcome == "timeout" for entry in result.entries))
        self.assertEqual(result.entries[-1].skip_reason, "timeout_retries_exhausted")
        self.assertLessEqual(result.max_global_observed, 1)
        commits = ledger.committed_results("timeout")
        self.assertEqual(len(commits), 2)
        self.assertTrue(all(row["outcome"] == "timeout" for row in commits))

    def test_zero_quota_is_skipped_with_reason(self):
        spec = self.load()
        store = PackStore(self.root / "packs", spec)
        result = PackScheduler(store=store, global_slots=1, node_quotas={"node-a": 0}, run_id="skip").run(
            [self._task("s0", "node-a")]
        )
        self.assertEqual(result.entries[0].outcome, "skipped")
        self.assertEqual(result.entries[0].skip_reason, "node_quota_zero_or_missing")


class RepairTests(SampleCase):
    def test_full_repair_counts_are_self_consistent(self):
        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        registry.inject(shard_id="node-b:s4", kind="flip", offset=4)
        report = apply_and_report(spec, registry)
        self.assertEqual(report["counts"], {"full": 3, "partial": 0, "unrecoverable": 0})
        self.assertTrue(report["self_consistent"])
        self.assertEqual(report["repair_task_count"], 1)
        self.assertEqual(report["repair_success_count"], 1)
        task = report["repair_tasks"][0]
        self.assertEqual(task["source_shard_id"], "node-a:s4")
        self.assertEqual(task["target_shard_id"], "node-b:s4")
        self.assertEqual(task["bytes"], spec.logical_shards["s4"].byte_count)
        self.assertEqual(task["expected_digest"], spec.logical_shards["s4"].digest)

    def test_disabled_target_produces_partial_repair(self):
        raw = copy.deepcopy(self.raw)
        for node in raw["nodes"]:
            if node["node_id"] == "node-b":
                node["repair_enabled"] = False
        spec = load_spec(self.spec_path, raw_override=raw)
        registry = ReplicaRegistry.load(spec)
        registry.inject(shard_id="node-b:s4", kind="flip", offset=9)
        report = apply_and_report(spec, registry)
        self.assertGreaterEqual(report["counts"]["partial"], 1)
        self.assertEqual(sum(report["counts"].values()), report["block_count"])
        self.assertEqual(report["repair_task_count"], 0)

    def test_no_matching_replica_is_unrecoverable_with_attempts(self):
        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        for node_id in ("node-a", "node-b", "node-c"):
            registry.inject(shard_id=f"{node_id}:s3", kind="flip", offset=1)
        report = apply_and_report(spec, registry)
        self.assertGreaterEqual(report["counts"]["unrecoverable"], 1)
        unrecoverable = [
            item for item in report["decisions"] if item["category"] == "unrecoverable"
        ]
        self.assertEqual(unrecoverable[0]["missing_seqs"], [3])
        self.assertEqual(len(unrecoverable[0]["attempted_replicas"]), 3)
        self.assertEqual(sum(report["counts"].values()), report["block_count"])


class SnapshotTests(SampleCase):
    def _state(self):
        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        store = PackStore(self.root / "packs", spec)
        store.repack_all_direct()
        registry.inject(shard_id="node-b:s4", kind="flip", offset=6)
        propagation = propagate_invalidation(spec, ["s4"])
        registry.mark_affected(propagation.affected)
        report = apply_and_report(spec, registry)
        registry.clear_pending()
        manager = SnapshotManager(self.root / "snapshots" / "session.json")
        snapshot = manager.save(
            spec=spec,
            registry=registry,
            pack_store=store,
            schedule={"entries": []},
            repair_report=report,
            injections=[],
            budget={"global_slots": 2, "node_quotas": {}},
            spec_digest=spec_manifest_digest(spec),
            replay={"affected": list(propagation.affected)},
            block_replica_digest=block_replica_digest(spec, registry),
        )
        return spec, registry, store, manager, snapshot

    def test_snapshot_restore_is_whole_replacement_and_backup_is_readonly(self):
        spec, registry, store, manager, snapshot = self._state()
        registry.inject(shard_id="node-c:s6", kind="replace", sequence=8)
        self.assertNotEqual(registry.disk_digest(), snapshot["disk_digest"])
        result = manager.restore(
            spec=spec,
            registry=registry,
            state_path=self.root / "state" / "current.json",
        )
        self.assertTrue(result["replaced"])
        self.assertEqual(result["registry_digest"], snapshot["registry_digest"])
        self.assertTrue((self.root / "state" / "current.json").exists())
        backup = Path(result["backup_path"])
        self.assertTrue(backup.exists())
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode) & 0o222, 0)

    def test_snapshot_overwrite_keeps_readonly_backup(self):
        spec, registry, store, manager, first = self._state()
        second = manager.save(
            spec=spec,
            registry=registry,
            pack_store=store,
            schedule={"entries": []},
            repair_report={"counts": {"full": 3, "partial": 0, "unrecoverable": 0}},
            injections=[],
            budget={"global_slots": 2},
            spec_digest=spec_manifest_digest(spec),
        )
        self.assertEqual(second["snapshot_version"], first["snapshot_version"] + 1)
        backups = list((self.root / "snapshots" / "backups").glob("*.json"))
        self.assertTrue(backups)
        self.assertEqual(stat.S_IMODE(backups[0].stat().st_mode) & 0o222, 0)

    def test_tampered_content_digest_is_e_snapshot(self):
        spec, registry, store, manager, snapshot = self._state()
        tampered = tamper_snapshot_copy(manager.path, self.root / "tampered.json")
        with self.assertRaises(SnapshotError) as caught:
            SnapshotManager(tampered).load()
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertEqual(caught.exception.field_path, "content_digest")

    def test_replay_mismatch_is_e_replay_with_field(self):
        spec, registry, store, manager, snapshot = self._state()
        loaded = manager.load()
        with self.assertRaises(ReplayError) as caught:
            compare_replay(
                loaded,
                {
                    "affected": ["s4"],
                    "pack_manifest": loaded["pack_manifest"],
                    "schedule": loaded["schedule"],
                    "repair_report": loaded["repair_report"],
                    "registry_digest": loaded["registry_digest"],
                    "block_replica_digest": loaded["block_replica_digest"],
                },
            )
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.field_path, "affected")


class CrashRecoveryTests(SampleCase):
    def test_forced_termination_recovers_only_committed_tasks(self):
        db = self.root / "crash.sqlite3"
        command = [
            sys.executable,
            "-m",
            "shardrep",
            "run",
            "--spec",
            str(self.spec_path),
            "--inject",
            "node-b:s4=flip",
            "--pack-dir",
            str(self.root / "packs"),
            "--db",
            str(db),
            "--log-dir",
            str(self.root / "logs"),
            "--global-slots",
            "2",
            "--simulate-ms",
            "100",
            "--timeout-ms",
            "2000",
            "--crash-after-commits",
            "1",
            "--run-id",
            "crash-run",
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 17)
        ledger = Ledger(db)
        report = ledger.recover("crash-run")
        self.assertFalse(report["exit_report_available"])
        self.assertGreaterEqual(len(report["committed"]), 1)
        self.assertGreaterEqual(len(report["unknown_requires_rerun"]), 1)
        self.assertTrue(all(row["outcome"] == "success" for row in report["committed"]))
        resume = subprocess.run(
            [
                sys.executable,
                "-m",
                "shardrep",
                "run",
                "--spec",
                str(self.spec_path),
                "--inject",
                "node-b:s4=flip",
                "--pack-dir",
                str(self.root / "packs"),
                "--db",
                str(db),
                "--log-dir",
                str(self.root / "logs"),
                "--global-slots",
                "2",
                "--simulate-ms",
                "1",
                "--timeout-ms",
                "2000",
                "--resume",
                "--run-id",
                "crash-run",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(resume.returncode, 0, resume.stderr)
        resumed = json.loads(resume.stdout)
        outcomes = {entry["outcome"] for entry in resumed["schedule"]["entries"]}
        self.assertIn("recovered", outcomes)
        for logical_id in ("s4", "s5", "s7"):
            self.assertTrue((self.root / "packs" / f"{logical_id}.json").exists())


class AnomalyLogTests(SampleCase):
    def test_cycle_anomaly_writes_structured_e_cycle(self):
        log_dir = self.root / "logs"
        db = self.root / "anomaly.sqlite3"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "shardrep",
                "anomaly",
                "--kind",
                "cycle",
                "--spec",
                str(self.spec_path),
                "--log-dir",
                str(log_dir),
                "--db",
                str(db),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(process.returncode, 2)
        events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text().splitlines()]
        cycle = [event for event in events if event["code"] == "E_CYCLE"]
        self.assertEqual(len(cycle), 1)
        self.assertIsNotNone(cycle[0]["shard_id"])
        self.assertEqual(cycle[0]["version"], 1)
        self.assertIn("timestamp", cycle[0])

    def test_integrity_and_snapshot_anomalies_write_expected_codes(self):
        db = self.root / "anomaly.sqlite3"
        log_dir = self.root / "logs"
        integrity = subprocess.run(
            [
                sys.executable,
                "-m",
                "shardrep",
                "anomaly",
                "--kind",
                "integrity",
                "--spec",
                str(self.spec_path),
                "--shard-id",
                "node-a:s2",
                "--log-dir",
                str(log_dir),
                "--db",
                str(db),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(integrity.returncode, 2)

        spec = self.load()
        registry = ReplicaRegistry.load(spec)
        store = PackStore(self.root / "packs", spec)
        store.repack_all_direct()
        manager = SnapshotManager(self.root / "snapshots" / "session.json")
        manager.save(
            spec=spec,
            registry=registry,
            pack_store=store,
            schedule={"entries": []},
            repair_report={"counts": {"full": 3, "partial": 0, "unrecoverable": 0}},
            injections=[],
            budget={"global_slots": 2},
            spec_digest=spec_manifest_digest(spec),
        )
        snapshot_anomaly = subprocess.run(
            [
                sys.executable,
                "-m",
                "shardrep",
                "anomaly",
                "--kind",
                "snapshot",
                "--snapshot",
                str(manager.path),
                "--log-dir",
                str(log_dir),
                "--db",
                str(db),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(snapshot_anomaly.returncode, 2)
        events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text().splitlines()]
        codes = {event["code"] for event in events}
        self.assertIn("E_INTEGRITY", codes)
        self.assertIn("E_SNAPSHOT", codes)
        for event in events:
            if event["code"] in {"E_INTEGRITY", "E_SNAPSHOT"}:
                self.assertIn("shard_id", event)
                self.assertIn("version", event)
                self.assertIn("timestamp", event)


class DemoRepeatabilityTests(unittest.TestCase):
    def test_demo_stable_digest_repeats(self):
        from shardrep.demo import run_demo

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = run_demo(root, simulate_ms=1, timeout_ms=1000)
            second = run_demo(root, simulate_ms=1, timeout_ms=1000)
            self.assertEqual(first["stable_digest"], second["stable_digest"])
            self.assertTrue(first["stage2_incremental"]["incremental_vs_full"]["ok"])


if __name__ == "__main__":
    unittest.main()
