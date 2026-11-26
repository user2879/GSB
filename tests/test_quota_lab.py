from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from quota_lab.accounting import candidate_error_vector
from quota_lab.canonical import canonical_dumps
from quota_lab.errors import LeaseError, SpecError
from quota_lab.fixtures import demo_declaration, demo_shards, write_demo_files
from quota_lab.fixedpoint import within_integer_bound
from quota_lab.invalidation import compare_specs
from quota_lab.leases import LeaseRegistry, new_lease
from quota_lab.ledger import Ledger, replace_json_snapshot_if_stale
from quota_lab.processes import Cluster, NodeProcess
from quota_lab.scheduler import Scheduler
from quota_lab.spec import NodeSpec, parse_session


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def minimal_spec(port: int | None = None) -> object:
    declaration = {
        "schema_version": 1,
        "session_id": "unit",
        "scale_power": 1,
        "global_budget_units": 3,
        "lease_ttl_ms": 40,
        "lease_max_renewals": 1,
        "socket_timeout_ms": 500,
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": port or free_port(),
                "concurrency_quota": 2,
                "lease_limit": 2,
                "seed": 42,
                "capacity_version": 1,
            }
        ],
    }
    shards = {
        "shards": [
            {
                "shard_id": "a",
                "node_id": "n1",
                "depends_on": [],
                "sample_rows": 8,
                "resource_units": 1,
                "priority": 1,
                "planned_renewals": 1,
            },
            {
                "shard_id": "b",
                "node_id": "n1",
                "depends_on": ["a"],
                "sample_rows": 6,
                "resource_units": 2,
                "priority": 0,
            },
        ]
    }
    return parse_session(declaration, shards)


class SpecValidationTests(unittest.TestCase):
    def test_valid_spec_and_topo(self) -> None:
        spec = minimal_spec()
        self.assertEqual(list(spec.topo_order), ["a", "b"])
        self.assertEqual(spec.q, 10)

    def test_bad_host_reports_field(self) -> None:
        decl = demo_declaration()
        decl["nodes"][0]["host"] = "localhost"
        with self.assertRaises(SpecError) as ctx:
            parse_session(decl, demo_shards())
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "declaration.nodes[0].host")

    def test_duplicate_port_reports_field(self) -> None:
        decl = demo_declaration()
        decl["nodes"][1]["port"] = decl["nodes"][0]["port"]
        with self.assertRaises(SpecError) as ctx:
            parse_session(decl, demo_shards())
        self.assertEqual(ctx.exception.field, "declaration.nodes[1].port")

    def test_cycle_reports_dependency_field(self) -> None:
        decl = demo_declaration()
        shards = demo_shards()
        by_id = {item["shard_id"]: item for item in shards["shards"]}
        by_id["s1"]["depends_on"] = ["s2"]
        by_id["s2"]["depends_on"] = ["s1"]
        with self.assertRaises(SpecError) as ctx:
            parse_session(decl, shards)
        self.assertEqual(ctx.exception.field, "shards.depends_on")


class FixedPointTests(unittest.TestCase):
    def test_integer_boundary_decision(self) -> None:
        self.assertTrue(within_integer_bound(10, 10))
        self.assertFalse(within_integer_bound(11, 10))

    def test_candidate_vector_is_integer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_demo_files(root)
            from quota_lab.samples import load_event_stream
            from quota_lab.spec import load_session_files

            spec = load_session_files(root / "configs/declaration.json", root / "configs/shards.json")
            samples = load_event_stream(paths["events"], spec)
            accounting = {
                shard.shard_id: __import__("quota_lab.accounting", fromlist=["build_shard_accounting"]).build_shard_accounting(
                    spec, shard.shard_id, samples[shard.shard_id]
                )
                for shard in spec.shards
            }
            vector = candidate_error_vector(spec, accounting, set(), "s1")
            self.assertIsInstance(vector["composite_bound_q"], int)
            self.assertGreater(vector["composite_bound_q"], 0)


class InvalidationTests(unittest.TestCase):
    def test_capacity_version_descendants(self) -> None:
        old = minimal_spec(19001)
        new = replace(
            old,
            nodes=tuple(replace(node, capacity_version=2) if node.node_id == "n1" else node for node in old.nodes),
        )
        plan = compare_specs(old, new)
        self.assertEqual(plan.invalid_shard_ids, ("a", "b"))
        self.assertEqual(plan.unchanged_shard_ids, ())

    def test_resource_demand_change_invalidates_shard_and_descendant(self) -> None:
        old = minimal_spec(19002)
        new_shards = tuple(
            replace(shard, resource_units=shard.resource_units + 1) if shard.shard_id == "a" else shard
            for shard in old.shards
        )
        new = replace(old, shards=new_shards)
        plan = compare_specs(old, new)
        self.assertEqual(plan.direct_shard_ids, ("a",))
        self.assertEqual(plan.invalid_shard_ids, ("a", "b"))

    def test_quota_change_invalidates_all_node_shards(self) -> None:
        old = minimal_spec(19003)
        new_nodes = tuple(
            replace(node, concurrency_quota=node.concurrency_quota + 1) if node.node_id == "n1" else node
            for node in old.nodes
        )
        new = replace(old, nodes=new_nodes)
        plan = compare_specs(old, new)
        self.assertEqual(plan.invalid_shard_ids, ("a", "b"))
        self.assertTrue(any(reason["field"].endswith("concurrency_quota") for reason in plan.direct_reasons))


class LeaseRaceTests(unittest.TestCase):
    def test_expiry_and_version_void_release_once(self) -> None:
        spec = minimal_spec()
        registry = LeaseRegistry(spec)
        lease = new_lease(spec, "a")
        registry.add(lease)
        time.sleep(0.05)
        barrier = threading.Barrier(2)
        outcomes: list[int] = []

        def expire() -> None:
            barrier.wait()
            outcomes.append(len(registry.reap_expired()))

        def void() -> None:
            barrier.wait()
            outcomes.append(len(registry.void_capacity_version("n1", 2)))

        threads = [threading.Thread(target=expire), threading.Thread(target=void)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(outcomes), 1)
        self.assertEqual(registry.active_count("n1"), 0)

    def test_renew_limit_is_lease_error(self) -> None:
        spec = minimal_spec()
        registry = LeaseRegistry(spec)
        lease = new_lease(spec, "a")
        lease["renew_count"] = 1
        with self.assertRaises(LeaseError):
            registry.renew_optimistic(object(), lease)


class LedgerTests(unittest.TestCase):
    def test_mismatch_replaces_snapshot_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            first = replace_json_snapshot_if_stale(
                snapshot, expected_version=1, spec_digest="a", fresh_payload={"v": 1}
            )
            second = replace_json_snapshot_if_stale(
                snapshot, expected_version=1, spec_digest="b", fresh_payload={"v": 2}
            )
            current = json.loads(snapshot.read_text())
            self.assertTrue(first["replaced"])
            self.assertTrue(second["replaced"])
            self.assertEqual(current["spec_digest"], "b")
            self.assertEqual(current["payload"]["v"], 2)

            ledger_path = root / "state.sqlite3"
            ledger = Ledger.open_for_session(ledger_path, spec_digest="a")
            ledger.close()
            replaced = Ledger.open_for_session(ledger_path, spec_digest="b")
            self.assertTrue(replaced.replaced)
            header = Ledger.read_header(ledger_path)
            self.assertEqual(header["spec_digest"], "b")
            replaced.close()

    def test_recovery_marks_running_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "l.sqlite3", __import__("sqlite3").connect(Path(tmp) / "l.sqlite3"))
            ledger.conn.row_factory = __import__("sqlite3").Row
            ledger.conn.executescript(__import__("quota_lab.ledger", fromlist=["SCHEMA_SQL"]).SCHEMA_SQL)
            ledger._initialize_meta("x")
            lease = new_lease(minimal_spec(), "a")
            ledger.begin_attempt(
                shard_id="a",
                node_id="n1",
                topo_rank=0,
                attempt=1,
                demand_units=1,
                demand_q=10,
                wait_started_ms=1,
                wait_ended_ms=2,
                started_ms=2,
                lease=lease,
                lease_ownership={"lease_id": lease["lease_id"]},
            )
            reclaimed = ledger.recover_incomplete()
            self.assertEqual([item["shard_id"] for item in reclaimed], ["a"])
            row = ledger.conn.execute("SELECT state FROM shard_records WHERE shard_id='a'").fetchone()
            self.assertEqual(row["state"], "unknown")
            ledger.close()


class NodeProtocolTests(unittest.TestCase):
    def test_real_tcp_lease_cas_and_version_advance(self) -> None:
        port = free_port()
        spec = minimal_spec(port)
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
            runner = NodeProcess(spec.nodes[0], spec_path, Path(tmp) / "node.log").start()
            try:
                granted = runner.request(
                    {
                        "op": "lease_grant",
                        "node_id": "n1",
                        "lease_id": "lease-test",
                        "shard_id": "a",
                    },
                    timeout_ms=500,
                )
                self.assertEqual(granted["lease"]["lease_version"], 1)
                renewed = runner.request(
                    {
                        "op": "lease_renew",
                        "node_id": "n1",
                        "lease_id": "lease-test",
                        "expected_version": 1,
                    },
                    timeout_ms=500,
                )
                self.assertEqual(renewed["lease"]["lease_version"], 2)
                with self.assertRaises(LeaseError):
                    runner.request(
                        {
                            "op": "lease_renew",
                            "node_id": "n1",
                            "lease_id": "lease-test",
                            "expected_version": 1,
                        },
                        timeout_ms=500,
                    )
                advanced = runner.request(
                    {"op": "advance_version", "node_id": "n1", "new_capacity_version": 2},
                    timeout_ms=500,
                )
                self.assertEqual(advanced["capacity_version"], 2)
                self.assertEqual(len(advanced["revoked_leases"]), 1)
            finally:
                runner.stop()


class EndToEndSchedulerTests(unittest.TestCase):
    def test_demo_scheduler_over_real_tcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_demo_files(root)
            from quota_lab.spec import load_session_files
            from quota_lab.samples import load_event_stream

            loaded = load_session_files(
                root / "configs/declaration.json",
                root / "configs/shards.json",
                capacity_path=root / "configs/capacity.json",
            )
            nodes = tuple(replace(node, port=free_port()) for node in loaded.nodes)
            spec = replace(loaded, nodes=nodes)
            samples = load_event_stream(root / "samples/event_stream.jsonl", loaded)
            spec_path = root / "runtime-spec.json"
            spec_path.write_text(canonical_dumps(spec.to_dict()) + "\n", encoding="utf-8")
            ledger = Ledger.open_for_session(root / "state.sqlite3", spec_digest=spec.digest())
            with Cluster(spec, spec_path, root / "logs") as cluster:
                result = Scheduler(spec, samples, cluster, ledger=ledger).run()
            ledger.close()
            self.assertTrue(result["ok"], result["run_errors"])
            self.assertTrue(result["constraint_check"]["budget_never_exceeded_exact_integer"])
            self.assertEqual(len(result["completed_shard_ids"]), 6)
            self.assertTrue(any(decision.get("error_affected") for decision in result["decisions"]))


if __name__ == "__main__":
    unittest.main()
