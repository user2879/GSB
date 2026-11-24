from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from workbench.adjudication import choose_authority
from workbench.cli import main
from workbench.engine import Workbench
from workbench.report import build_report
from workbench.sample import build_sample, find_free_ports, write_sample_files
from workbench.snapshot import SnapshotStore
from workbench.validation import load_and_validate
from workbench.watermark import rounded_permille_limit


def sample_validation():
    sample = build_sample(find_free_ports(4))
    return sample, load_and_validate(config=sample["config"], events=sample["events"])


def run_report(validation, *, parallel=False, split=None, self_check=True):
    workbench = Workbench(validation, parallel=parallel)
    try:
        if split is None:
            workbench.run(close_cluster=False)
        else:
            workbench.run(max_events=split, close_cluster=False)
            workbench.run(resume=True, close_cluster=False)
        if self_check:
            workbench.protocol_self_check()
        return build_report(workbench)
    finally:
        pids = dict(workbench.cluster.pids())
        workbench.close()
    return pids


class WatermarkTests(unittest.TestCase):
    def test_fixed_point_half_up(self):
        self.assertEqual(rounded_permille_limit(10, 750), 8)
        self.assertEqual(rounded_permille_limit(3, 500), 2)
        self.assertEqual(rounded_permille_limit(10, 800), 8)
        self.assertEqual(rounded_permille_limit(10, 500), 5)

    def test_policy_switch_changes_numeric_limits(self):
        self.assertEqual(rounded_permille_limit(10, 800), 8)
        self.assertEqual(rounded_permille_limit(10, 500), 5)
        self.assertEqual(rounded_permille_limit(10, 600), 6)
        self.assertEqual(rounded_permille_limit(10, 300), 3)


class ValidationTests(unittest.TestCase):
    def test_invalid_inputs_have_separate_codes(self):
        sample = build_sample(find_free_ports(4))
        result = load_and_validate(config=sample["invalid"])
        codes = {item["code"] for item in result["failures"]}
        self.assertIn("E_NODE_INVALID", codes)
        self.assertIn("E_MANIFEST_INVALID", codes)
        self.assertIn("E_EVENT_INVALID", codes)
        messages = "\n".join(item["message"] for item in result["failures"])
        self.assertIn("strictly increasing", messages)
        self.assertIn("cycle", messages)
        self.assertIn("does not exist", messages)
        self.assertIn("duplicate loopback port", messages)


class AuthorityTieBreakTests(unittest.TestCase):
    def test_missing_digest_uses_node_id_lexicographic(self):
        decision = choose_authority([
            {"key": "k", "node_id": "z", "capacity_version": 1, "elapsed_ms": 2, "content_digest": None},
            {"key": "k", "node_id": "a", "capacity_version": 1, "elapsed_ms": 2, "content_digest": None},
        ])
        self.assertEqual(decision["winner_node_id"], "a")

    def test_capacity_version_dominates(self):
        decision = choose_authority([
            {"key": "k", "node_id": "a", "capacity_version": 1, "elapsed_ms": 1, "content_digest": "sha256:1"},
            {"key": "k", "node_id": "b", "capacity_version": 2, "elapsed_ms": 9, "content_digest": "sha256:2"},
        ])
        self.assertEqual(decision["winner_node_id"], "b")


class EndToEndSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample, cls.validation = sample_validation()
        cls.report = run_report(cls.validation)

    def test_01_all_sample_events_valid(self):
        self.assertTrue(self.validation["valid"], self.validation["failures"])
        self.assertEqual(len(self.validation["events"]), 25)

    def test_02_authority_and_overwrite(self):
        authority = {item["artifact_key"]: item for item in self.report["authority_replicas"]}
        self.assertEqual(authority["feat:user_profile"]["winner_node_id"], "n-core-b")
        overwrite = [x for x in self.report["overwrite_references"] if x["artifact_key"] == "feat:user_profile"]
        self.assertTrue(overwrite)
        self.assertIsNotNone(overwrite[0]["overwritten_digest"])
        self.assertNotEqual(overwrite[0]["overwritten_digest"], overwrite[0]["new_digest"])

    def test_03_lease_renewal_and_expiration_order(self):
        lease_evictions = [x for x in self.report["eviction_details"] if x["reason"] == "lease_expired"]
        self.assertEqual([x["key"] for x in lease_evictions], ["lease:high", "lease:middle", "lease:late"])
        renewed = [x for x in self.report["lease_renewals"] if x["key"] == "lease:renewed"]
        self.assertTrue(renewed and renewed[0]["renewed"])
        failed = [x for x in self.report["lease_renewals"] if x["key"] == "lease:late"]
        self.assertFalse(failed[0]["renewed"])
        lease_node = next(n for n in self.report["nodes"] if n["node_id"] == "n-lease")
        self.assertEqual(lease_node["active_leases"], 1)

    def test_04_policy_eviction_pending_and_capacity_code(self):
        policy_evictions = [x for x in self.report["eviction_details"] if x["reason"] == "policy_eviction"]
        self.assertEqual(
            [x["key"] for x in policy_evictions],
            ["block:04", "block:02", "block:01", "block:03"],
        )
        decisions = {x["artifact_key"]: x for x in self.report["pending_decisions"]}
        self.assertEqual(decisions["af:pending:a"]["decision"], "reserved")
        self.assertTrue(decisions["af:pending:a"]["admitted"])
        self.assertEqual(decisions["af:pending:b"]["decision"], "rejected")
        reconciliation = self.report["reconciliations"][-1]
        self.assertTrue(reconciliation["matched"])
        self.assertEqual(reconciliation["actual_eviction_keys"], reconciliation["planned_eviction_keys"])
        water = next(n for n in self.report["nodes"] if n["node_id"] == "n-water")
        self.assertEqual((water["soft_limit"], water["hard_limit"]), (3, 5))
        self.assertEqual(water["occupancy"], 5)
        self.assertEqual(water["watermark_state"], "hard")
        self.assertEqual(self.report["code_counts"]["E_CAPACITY_EXCEEDED"], 1)

    def test_05_protocol_codes_and_failure_order(self):
        self.assertEqual(self.report["code_counts"]["E_PROTOCOL_VIOLATION"], 2)
        codes = [item["code"] for item in self.report["failure_list"]]
        self.assertEqual(codes, ["E_CAPACITY_EXCEEDED", "E_PROTOCOL_VIOLATION", "E_PROTOCOL_VIOLATION"])

    def test_06_cache_reuse_ledger(self):
        rows = self.report["cache_ledger"]
        core_b_summary = next(x for x in rows if x["node_id"] == "n-core-b" and x["stage"] == "resident_summary")
        self.assertEqual(core_b_summary["status"], "hit")
        self.assertGreater(core_b_summary["hit_count"], 0)
        core_a_watermark = next(x for x in rows if x["node_id"] == "n-core-a" and x["stage"] == "watermark")
        self.assertGreaterEqual(core_a_watermark["recompute_count"], 2)
        sorted_rows = sorted(rows, key=lambda x: (x["node_id"], x["artifact_key"], x["stage"], x["scope_id"]))
        self.assertEqual(rows, sorted_rows)

    def test_07_budget_deduction_sequence(self):
        sequence = self.report["budget"]["deduction_sequence"]
        self.assertTrue(sequence)
        self.assertTrue(all(item["delta"] == -1 for item in sequence))
        self.assertEqual([item["seq"] for item in sequence], sorted(item["seq"] for item in sequence))


class ConsistencyTests(unittest.TestCase):
    def test_incremental_prefixes_match_full_field_by_field(self):
        _, validation = sample_validation()
        full = run_report(validation)
        event_count = len(validation["events"])
        for split in range(1, event_count + 1):
            with self.subTest(split=split):
                incremental = run_report(validation, split=split)
                self.assertEqual(incremental, full)

    def test_parallel_fanout_matches_serial(self):
        _, validation = sample_validation()
        serial = run_report(validation, parallel=False)
        parallel = run_report(validation, parallel=True)
        self.assertEqual(parallel, serial)


class ProcessLifecycleTests(unittest.TestCase):
    def test_child_processes_are_terminated(self):
        _, validation = sample_validation()
        workbench = Workbench(validation)
        workbench.cluster.start()
        pids = list(workbench.cluster.pids().values())
        self.assertEqual(len(pids), 4)
        workbench.close()
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_lease_limit_is_enforced_by_node(self):
        port = find_free_ports(1)[0]
        config = {
            "capacity_policies": [{"capacity_version": 1, "soft_permille": 800, "hard_permille": 950}],
            "nodes": [{
                "node_id": "lease-limited",
                "port": port,
                "capacity_slots": 5,
                "lease_limit": 1,
                "seed": 7,
                "capacity_version": 1,
            }],
            "manifest": {"stages": []},
        }
        validation = load_and_validate(config=config, events=[])
        workbench = Workbench(validation)
        workbench.cluster.start()
        try:
            first = workbench.cluster.request("lease-limited", {
                "request_id": "lease-1", "type": "write", "key": "k1",
                "content": "one", "ttl_ms": 1000, "now_ms": 1,
            })
            second = workbench.cluster.request("lease-limited", {
                "request_id": "lease-2", "type": "write", "key": "k2",
                "content": "two", "ttl_ms": 1000, "now_ms": 2,
            })
            self.assertTrue(first["ok"], first)
            self.assertFalse(second["ok"])
            self.assertEqual(second["error"]["code"], "E_LEASE_LIMIT_EXCEEDED")
        finally:
            workbench.close()


class RandomSequenceConsistencyTests(unittest.TestCase):
    def test_deterministic_generated_sequences(self):
        import random

        for seed in range(4):
            with self.subTest(seed=seed):
                rnd = random.Random(1000 + seed)
                ports = find_free_ports(2)
                config = {
                    "capacity_policies": [{"capacity_version": 1, "soft_permille": 500, "hard_permille": 800}],
                    "nodes": [
                        {"node_id": "a", "port": ports[0], "capacity_slots": 6, "lease_limit": 4, "seed": seed, "capacity_version": 1},
                        {"node_id": "b", "port": ports[1], "capacity_slots": 6, "lease_limit": 4, "seed": seed + 1, "capacity_version": 1},
                    ],
                    "manifest": {"stages": [{"stage_id": "s", "artifact_key": "k0", "needs": [], "concurrency_quota": 9}]},
                }
                keys = [f"k{i}" for i in range(4)]
                events = []
                now = 1000
                for seq in range(1, 19):
                    now += rnd.choice([0, 10, 20])
                    kind = rnd.choice(["both", "write_a", "write_b", "read", "renew", "release", "tick"])
                    key = rnd.choice(keys)
                    event = {"seq": seq}
                    if kind == "both":
                        event.update({"type": "WRITE", "node_ids": ["a", "b"], "key": key, "stage_id": "s",
                                      "content": seq, "priority": rnd.randrange(4),
                                      "ttl_ms": rnd.choice([80, 200, 1000]), "now_ms": now})
                    elif kind.startswith("write"):
                        event.update({"type": "WRITE", "node_id": kind[-1], "key": key, "stage_id": "s",
                                      "content": seq, "priority": rnd.randrange(4),
                                      "ttl_ms": rnd.choice([80, 200, 1000]), "now_ms": now})
                    elif kind == "read":
                        event.update({"type": "READ", "node_id": rnd.choice(["a", "b"]), "key": key, "now_ms": now})
                    elif kind == "renew":
                        event.update({"type": "LEASE_RENEW", "node_id": rnd.choice(["a", "b"]), "key": key,
                                      "ttl_ms": 300, "now_ms": now})
                    elif kind == "release":
                        event.update({"type": "RELEASE", "node_id": rnd.choice(["a", "b"]), "key": key, "now_ms": now})
                    else:
                        event.update({"type": "TICK", "now_ms": now})
                    events.append(event)
                validation = load_and_validate(config=config, events=events)
                self.assertTrue(validation["valid"], validation["failures"])
                full = run_report(validation, self_check=False)
                incremental = run_report(validation, split=7 + seed * 3, self_check=False)
                parallel = run_report(validation, parallel=True, self_check=False)
                self.assertEqual(incremental, full)
                self.assertEqual(parallel, full)


class SnapshotCliTests(unittest.TestCase):
    def test_cli_prefix_resume_report_matches_full(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_sample_files(root, ports=find_free_ports(4))
            full_out = root / "full"
            incremental_out = root / "incremental"

            self.assertEqual(main([
                "run", "--config", str(paths["config"]), "--events", str(paths["events"]),
                "--out", str(full_out), "--protocol-self-check",
            ]), 0)
            self.assertEqual(main([
                "run", "--config", str(paths["config"]), "--events", str(paths["events"]),
                "--out", str(incremental_out), "--max-events", "16",
            ]), 0)
            self.assertEqual(main([
                "run", "--config", str(paths["config"]), "--events", str(paths["events"]),
                "--out", str(incremental_out), "--protocol-self-check",
            ]), 0)

            full_report = SnapshotStore(full_out).read_current_report()
            incremental_report = SnapshotStore(incremental_out).read_current_report()
            self.assertEqual(incremental_report, full_report)
            recovery = json.loads((incremental_out / "recovery.json").read_text(encoding="utf-8"))
            self.assertTrue(recovery["resume"])
            self.assertEqual(recovery["event_cursor"], 24)

    def test_config_hash_change_bumps_global_version(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            out = Path(tmp)
            store = SnapshotStore(out, config_hash="hash-a")
            pointer = {
                "global_version": store.global_version,
                "config_hash": "hash-a",
                "event_cursor": 3,
                "snapshot_dir": "x",
            }
            store.current_pointer.write_text(json.dumps(pointer), encoding="utf-8")
            changed = SnapshotStore(out, config_hash="hash-b")
            plan = changed.recovery_plan([])
            self.assertFalse(plan["resume"])
            self.assertEqual(plan["reason"], "config_hash_changed")
            self.assertEqual(changed.global_version, 2)


if __name__ == "__main__":
    unittest.main()
