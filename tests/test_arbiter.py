from __future__ import annotations

import contextlib
import io
import json
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from gray_arbiter import cli
from gray_arbiter.config import parse_config
from gray_arbiter.engine import ArbiterEngine
from gray_arbiter.errors import BusinessError, StructuredLogger
from gray_arbiter.hashing import BUCKET_SPACE, gray_destination, stable_bucket
from gray_arbiter.ledger import Ledger, ReplayMismatch
from gray_arbiter.model import PoolBatchStats, RequestSpec
from gray_arbiter.pool import PoolManager
from gray_arbiter.stats import build_error_report, numpy_contribution_matrix
from tools.target_common import make_server


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


@contextlib.contextmanager
def target_servers(delay_ms: int = 5):
    servers = []
    ports: dict[str, int] = {}
    threads = []
    for role in ("stable", "gray", "mirror"):
        server = make_server("127.0.0.1", 0, role, delay_ms)
        servers.append(server)
        ports[role] = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
    try:
        yield ports
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=1)


def raw_config(
    ports: dict[str, int],
    *,
    stable_max: int = 2,
    gray_max: int = 1,
    mirror_max: int = 2,
    gray_weight: int = 5000,
    mirror_sample: int = 0,
    stable_reuse: int = 4,
    gray_reuse: int = 4,
    mirror_reuse: int = 3,
    idle_ms: int = 5000,
    mainstream_lends_gray: bool = True,
) -> dict[str, Any]:
    return {
        "version": 1,
        "borrow_ttl_ms": 3000,
        "queue_timeout_ms": 5000,
        "main_timeout_ms": 1000,
        "mirror_timeout_ms": 300,
        "pools": [
            {
                "pool_id": "stable",
                "host": "127.0.0.1",
                "port": ports["stable"],
                "max_connections": stable_max,
                "max_requests_per_connection": stable_reuse,
                "idle_timeout_ms": idle_ms,
                "lendable_to": ["gray"] if mainstream_lends_gray else [],
            },
            {
                "pool_id": "gray",
                "host": "127.0.0.1",
                "port": ports["gray"],
                "max_connections": gray_max,
                "max_requests_per_connection": gray_reuse,
                "idle_timeout_ms": idle_ms,
                "lendable_to": [],
            },
            {
                "pool_id": "mirror",
                "host": "127.0.0.1",
                "port": ports["mirror"],
                "max_connections": mirror_max,
                "max_requests_per_connection": mirror_reuse,
                "idle_timeout_ms": idle_ms,
                "lendable_to": [],
            },
        ],
        "rules": [
            {
                "rule_id": "r1",
                "prefix": "/api",
                "stable_pool": "stable",
                "gray_pool": "gray",
                "mirror_pool": "mirror",
                "gray_weight_bp": gray_weight,
                "mirror_sample_bp": mirror_sample,
            }
        ],
    }


class ConfigTests(unittest.TestCase):
    def test_duplicate_reference_weight_port_and_mirror_validation(self) -> None:
        ports = {"stable": free_port(), "gray": free_port(), "mirror": free_port()}
        base = raw_config(ports)
        cfg = parse_config(base)
        self.assertEqual(set(cfg.pools), {"stable", "gray", "mirror"})

        mapped = {
            "pools": {
                "stable": {
                    "endpoint": f"http://127.0.0.1:{ports['stable']}",
                    "capacity": 2,
                    "reuse_limit": 4,
                    "idle_ms": 1000,
                    "lendable_pool_ids": ["gray"],
                },
                "gray": {
                    "target": f"http://127.0.0.1:{ports['gray']}",
                    "capacity": 1,
                    "reuse_limit": 4,
                    "idle_ms": 1000,
                },
                "mirror": {
                    "port": ports["mirror"],
                    "capacity": 1,
                    "reuse_limit": 4,
                    "idle_ms": 1000,
                },
            },
            "rules": {
                "r1": {
                    "prefix": "/api",
                    "stable_pool": "stable",
                    "gray_pool": "gray",
                    "mirror_pool": "mirror",
                    "gray_ratio_bp": 5000,
                    "mirror_sampling_bp": 2500,
                }
            },
        }
        mapped_cfg = parse_config(mapped)
        self.assertEqual(mapped_cfg.rules["r1"].gray_weight_bp, 5000)
        self.assertEqual(mapped_cfg.pools["stable"].max_connections, 2)

        duplicate = json.loads(json.dumps(base))
        duplicate["pools"][1]["pool_id"] = "stable"
        with self.assertRaises(BusinessError) as ctx:
            parse_config(duplicate)
        self.assertEqual(ctx.exception.code, "E_POOL")
        self.assertIn("pool_id", ctx.exception.field)

        missing = json.loads(json.dumps(base))
        missing["rules"][0]["gray_pool"] = "missing"
        with self.assertRaises(BusinessError) as ctx:
            parse_config(missing)
        self.assertEqual(ctx.exception.code, "E_POOL")
        self.assertIn("gray_pool", ctx.exception.field)

        low_port = json.loads(json.dumps(base))
        low_port["pools"][0]["port"] = 80
        with self.assertRaises(BusinessError) as ctx:
            parse_config(low_port)
        self.assertEqual(ctx.exception.code, "E_POOL")
        self.assertIn("port", ctx.exception.field)

        bad_rate = json.loads(json.dumps(base))
        bad_rate["rules"][0]["gray_weight_bp"] = 10001
        with self.assertRaises(BusinessError) as ctx:
            parse_config(bad_rate)
        self.assertEqual(ctx.exception.code, "E_POOL")
        self.assertIn("gray_weight_bp", ctx.exception.field)

        mirror_cross = json.loads(json.dumps(base))
        mirror_cross["pools"][0]["lendable_to"] = ["mirror"]
        with self.assertRaises(BusinessError) as ctx:
            parse_config(mirror_cross)
        self.assertEqual(ctx.exception.code, "E_POOL")
        self.assertIn("lendable_to", ctx.exception.field)

        self_loop = json.loads(json.dumps(base))
        self_loop["pools"][0]["lendable_to"] = ["stable"]
        with self.assertRaises(BusinessError) as ctx:
            parse_config(self_loop)
        self.assertEqual(ctx.exception.code, "E_POOL")
        self.assertIn("lendable_to", ctx.exception.field)

    def test_hash_is_stable_and_inclusive_bucket_range(self) -> None:
        buckets = {stable_bucket(f"client-{i}", "r1") for i in range(1000)}
        self.assertTrue(all(0 <= value < BUCKET_SPACE for value in buckets))
        self.assertEqual(stable_bucket("fixed", "rule"), stable_bucket("fixed", "rule"))
        destination, bucket = gray_destination("fixed", "rule", 10000)
        self.assertEqual(destination, "gray")
        self.assertTrue(0 <= bucket <= 10000)
        self.assertEqual(gray_destination("fixed", "rule", 0)[0], "stable")
        self.assertTrue(all(gray_destination(f"c{i}", "rule", 10000)[0] == "gray" for i in range(100)))


class PoolTests(unittest.TestCase):
    def test_connection_reuse_limit_and_idle_reap(self) -> None:
        with target_servers() as ports:
            cfg = parse_config(
                raw_config(
                    ports,
                    stable_max=1,
                    gray_max=1,
                    idle_ms=40,
                    stable_reuse=2,
                    mirror_sample=0,
                )
            )
            manager = PoolManager(cfg, start_reaper=False)
            manager.begin_batch("b1")
            first = manager.acquire(
                "stable", "q1", "b1", allow_borrow=False, block=True, http_timeout_ms=500
            )
            self.assertEqual(first.reuse_sequence, 0)
            manager.execute_http(first, "/api/one", "q1", timeout_ms=500, phase="main")
            manager.release(first, "q1")
            self.assertEqual(first.connection.state, "idle")

            second = manager.acquire(
                "stable", "q2", "b1", allow_borrow=False, block=True, http_timeout_ms=500
            )
            self.assertEqual(second.connection.connection_id, first.connection.connection_id)
            self.assertEqual(second.reuse_sequence, 1)
            self.assertEqual(second.connection.state, "reused")
            manager.execute_http(second, "/api/two", "q2", timeout_ms=500, phase="main")
            manager.release(second, "q2")
            self.assertEqual(second.connection.state, "closed")

            third = manager.acquire(
                "stable", "q3", "b1", allow_borrow=False, block=True, http_timeout_ms=500
            )
            self.assertNotEqual(third.connection.connection_id, first.connection.connection_id)
            manager.execute_http(third, "/api/three", "q3", timeout_ms=500, phase="main")
            manager.release(third, "q3")
            self.assertEqual(third.connection.state, "idle")
            time.sleep(0.08)
            manager.reap_idle()
            self.assertEqual(manager.states["stable"].active, 0)
            self.assertEqual(manager.invariant_violations(), [])
            manager.shutdown()

    def test_borrow_watermark_and_queue_are_bounded(self) -> None:
        with target_servers(delay_ms=3) as ports:
            cfg = parse_config(
                raw_config(
                    ports,
                    stable_max=2,
                    gray_max=1,
                    gray_weight=10000,
                    mirror_sample=0,
                )
            )
            manager = PoolManager(cfg, start_reaper=False)
            manager.begin_batch("borrow-batch")
            barrier = threading.Barrier(6)
            acquired_results: list[Any] = []
            lock = threading.Lock()

            def worker(index: int) -> None:
                barrier.wait()
                acquired = manager.acquire(
                    "gray",
                    f"q{index}",
                    "borrow-batch",
                    allow_borrow=True,
                    block=True,
                    timeout_ms=5000,
                    http_timeout_ms=500,
                )
                with lock:
                    acquired_results.append(acquired)
                time.sleep(0.18)
                manager.release(acquired, f"q{index}")

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=6)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(acquired_results), 6)

            stats, _ = manager.finalize_batch()
            stats_by_id = {stat.pool_id: stat for stat in stats}
            stable = stats_by_id["stable"]
            gray = stats_by_id["gray"]
            self.assertLessEqual(stable.peak_watermark, stable.max_connections)
            self.assertEqual(stable.lent_out, 0)
            self.assertEqual(gray.borrowed_in, 0)
            self.assertGreaterEqual(gray.peak_watermark, 2)
            self.assertGreaterEqual(stable.borrow_count, 2)
            self.assertEqual(manager.invariant_violations(), [])
            self.assertTrue(
                any(item.wait_ms > 0 for item in acquired_results),
                "at least one request must have waited in the pool queue",
            )
            borrows = manager.batch_borrows()
            self.assertGreaterEqual(len(borrows), 2)
            self.assertTrue(all(item.lender_pool == "stable" for item in borrows))
            self.assertTrue(all(item.borrower_pool == "gray" for item in borrows))
            self.assertTrue(all(item.amount == 1 for item in borrows))
            manager.shutdown()


class EngineTests(unittest.TestCase):
    def test_mirror_unreachable_does_not_change_main_outcome(self) -> None:
        with target_servers() as ports:
            closed = free_port()
            ports["mirror"] = closed
            cfg = parse_config(
                raw_config(
                    ports,
                    gray_weight=0,
                    mirror_sample=10000,
                    mirror_max=1,
                    stable_reuse=10,
                )
            )
            with tempfile.TemporaryDirectory() as temp:
                logger = StructuredLogger(Path(temp) / "logs.ndjson")
                ledger = Ledger(Path(temp) / "ledger.sqlite", logger)
                engine = ArbiterEngine(
                    cfg,
                    ledger=ledger,
                    logger=logger,
                    snapshot_dir=Path(temp) / "snapshots",
                    worker_count=4,
                )
                requests = [
                    RequestSpec(f"m{i}", "/api/item", f"c{i}", 1000 + i) for i in range(4)
                ]
                result = engine.run(requests, batch_size=4)
                batch = result["batches"][0]
                self.assertEqual([item["main_status"] for item in batch["requests"]], ["ok"] * 4)
                self.assertTrue(all(item["mirrored"] for item in batch["requests"]))
                self.assertTrue(
                    all(item["mirror_error_code"] == "E_TARGET" for item in batch["requests"])
                )
                self.assertTrue(
                    all(item["main_pool_id"] == "stable" for item in batch["requests"])
                )
                mirror_stats = next(stat for stat in batch["pool_stats"] if stat["pool_id"] == "mirror")
                main_stats = next(stat for stat in batch["pool_stats"] if stat["pool_id"] == "stable")
                self.assertEqual(mirror_stats["active_connections_scaled"], 0)
                self.assertGreaterEqual(main_stats["reuse_count_scaled"], 0)
                events = logger.sorted_events()
                target_events = [event for event in events if event["business_code"] == "E_TARGET"]
                self.assertEqual(len(target_events), 4)
                self.assertEqual(
                    [event["request_id"] for event in target_events],
                    sorted(event["request_id"] for event in target_events),
                )
                self.assertTrue(all(event["phase"] == "mirror" for event in target_events))
                self.assertTrue(all(event["pool_id"] == "mirror" for event in target_events))
                self.assertEqual(ledger.replay_self_check()["status"], "ok")
                engine.close()

    def test_mirror_timeout_is_real_socket_timeout(self) -> None:
        with target_servers(delay_ms=3) as ports:
            slow = make_server("127.0.0.1", 0, "mirror-slow", 1200)
            thread = threading.Thread(target=slow.serve_forever, daemon=True)
            thread.start()
            ports["mirror"] = int(slow.server_address[1])
            raw = raw_config(ports, gray_weight=0, mirror_sample=10000, mirror_max=2)
            raw["mirror_timeout_ms"] = 80
            cfg = parse_config(raw)
            with tempfile.TemporaryDirectory() as temp:
                engine = ArbiterEngine(cfg, snapshot_dir=Path(temp))
                try:
                    result = engine.run(
                        [RequestSpec("slow-mirror", "/api/__mirror_slow__", "client", 1)],
                        batch_size=1,
                    )
                    outcome = result["batches"][0]["requests"][0]
                    self.assertEqual(outcome["main_status"], "ok")
                    self.assertEqual(outcome["mirror_status"], "timeout")
                    self.assertEqual(outcome["mirror_error_code"], "E_TIMEOUT")
                    self.assertGreaterEqual(outcome["mirror_elapsed_ms"], 60)
                finally:
                    engine.close()
                    slow.shutdown()
                    slow.server_close()

    def test_ledger_replay_tamper_and_snapshot_digest(self) -> None:
        with target_servers() as ports:
            cfg = parse_config(raw_config(ports, gray_weight=0, mirror_sample=0))
            with tempfile.TemporaryDirectory() as temp:
                ledger_path = Path(temp) / "ledger.sqlite"
                ledger = Ledger(ledger_path)
                engine = ArbiterEngine(cfg, ledger=ledger, snapshot_dir=Path(temp) / "snapshots")
                try:
                    result = engine.run(
                        [RequestSpec(f"l{i}", "/api/item", f"k{i}", i) for i in range(3)],
                        batch_size=2,
                    )
                    self.assertEqual(len(result["batches"]), 2)
                    replay = ledger.replay_self_check()
                    self.assertEqual(replay["batches_checked"], 2)
                    self.assertEqual(replay["requests_checked"], 3)
                    batch_id = ledger.list_batches()[0]["batch_id"]
                    with sqlite3.connect(ledger_path) as connection:
                        connection.execute(
                            "UPDATE request_ledger SET destination = 'tampered' "
                            "WHERE batch_id = ? AND ordinal = 1",
                            (batch_id,),
                        )
                        connection.commit()
                    with self.assertRaises(ReplayMismatch) as mismatch:
                        ledger.replay_self_check()
                    self.assertEqual(mismatch.exception.code, "E_REPLAY")
                    self.assertEqual(mismatch.exception.field, "destination")
                    self.assertEqual(mismatch.exception.request_id, "l0")
                    with self.assertRaises(BusinessError) as snapshot:
                        ledger.verify_batch(batch_id, force_digest_mismatch=True)
                    self.assertEqual(snapshot.exception.code, "E_SNAPSHOT")
                    self.assertEqual(snapshot.exception.field, "content_digest")
                finally:
                    engine.close()

    def test_recovery_marks_uncommitted_resources_unknown(self) -> None:
        with target_servers() as ports:
            cfg = parse_config(raw_config(ports, gray_weight=0, mirror_sample=0))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "ledger.sqlite"
                ledger = Ledger(path)
                engine = ArbiterEngine(cfg, ledger=ledger, snapshot_dir=Path(temp) / "snapshots")
                engine.run(
                    [RequestSpec(f"c{i}", "/api/item", f"k{i}", i) for i in range(2)],
                    batch_size=2,
                )
                ledger.begin_batch(
                    "v1-b000003-000004",
                    config_version=1,
                    start_cursor=3,
                    end_cursor=4,
                    resource_snapshot={
                        "connections": [
                            {
                                "connection_id": "stable-v1-c999",
                                "pool_id": "stable",
                                "state": "reused",
                                "in_use": True,
                            }
                        ],
                        "borrows": [
                            {
                                "borrow_id": "b-v1-999999",
                                "borrower_pool": "gray",
                                "lender_pool": "stable",
                                "state": "active",
                            }
                        ],
                    },
                )
                engine.manager.shutdown()
                ledger.close()

                recovered = Ledger(path)
                try:
                    recovery = recovered.recover()
                    self.assertEqual(len(recovery["committed_batches"]), 1)
                    self.assertEqual(len(recovery["interrupted_batches"]), 1)
                    self.assertFalse(recovery["exit_report_available"])
                    self.assertEqual(recovery["rerun_from_cursor"], 3)
                    self.assertEqual(len(recovery["unknown_connections"]), 1)
                    self.assertEqual(len(recovery["unknown_borrows"]), 1)
                    self.assertTrue(
                        all(item["state"] == "unknown" for item in recovery["unknown_resources"])
                    )
                    second_recovery = recovered.recover()
                    self.assertEqual(len(second_recovery["unknown_resources"]), 2)
                    self.assertFalse(second_recovery["exit_report_available"])
                    self.assertEqual(recovered.replay_self_check()["batches_checked"], 1)
                finally:
                    recovered.close()

    def test_resume_reruns_only_uncommitted_cursor(self) -> None:
        with target_servers() as ports:
            cfg = parse_config(raw_config(ports, gray_weight=0, mirror_sample=0))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "ledger.sqlite"
                ledger = Ledger(path)
                engine = ArbiterEngine(cfg, ledger=ledger, snapshot_dir=Path(temp) / "snapshots")
                requests = [
                    RequestSpec(f"resume-{i}", "/api/item", f"k{i}", i) for i in range(4)
                ]
                engine.run(requests[:2], batch_size=2)
                ledger.begin_batch(
                    "v1-b000003-000004",
                    config_version=1,
                    start_cursor=3,
                    end_cursor=4,
                    resource_snapshot={
                        "connections": [
                            {
                                "connection_id": "stable-v1-c888",
                                "pool_id": "stable",
                                "state": "reused",
                                "in_use": True,
                            }
                        ],
                        "borrows": [],
                    },
                )
                engine.manager.shutdown()
                ledger.close()

                resumed_ledger = Ledger(path)
                resumed_engine = ArbiterEngine(
                    cfg, ledger=resumed_ledger, snapshot_dir=Path(temp) / "snapshots"
                )
                try:
                    result = resumed_engine.run(requests, batch_size=2, resume=True)
                    self.assertEqual(result["recovery"]["resume_cursor"], 3)
                    self.assertEqual(len(result["batches"]), 1)
                    self.assertEqual(result["batches"][0]["start_cursor"], 3)
                    self.assertEqual(len(resumed_ledger.list_batches()), 2)
                    self.assertEqual(resumed_ledger.replay_self_check()["requests_checked"], 4)
                finally:
                    resumed_engine.close()

    def test_version_replacement_is_monotonic_and_stales_old_epoch(self) -> None:
        with target_servers() as ports:
            cfg = parse_config(
                raw_config(ports, stable_max=2, gray_max=1, gray_weight=0, mirror_sample=0)
            )
            with tempfile.TemporaryDirectory() as temp:
                ledger = Ledger(Path(temp) / "ledger.sqlite")
                engine = ArbiterEngine(cfg, ledger=ledger, snapshot_dir=Path(temp) / "snapshots")
                try:
                    engine.manager.begin_batch("manual")
                    engine.manager.acquire(
                        "gray", "own", "manual", allow_borrow=True, block=True, http_timeout_ms=500
                    )
                    second = engine.manager.acquire(
                        "gray", "borrow-1", "manual", allow_borrow=True, block=True, http_timeout_ms=500
                    )
                    third = engine.manager.acquire(
                        "gray", "borrow-2", "manual", allow_borrow=True, block=True, http_timeout_ms=500
                    )
                    self.assertIsNotNone(second.borrow_id)
                    self.assertIsNotNone(third.borrow_id)
                    new_pools = dict(cfg.pools)
                    new_pools["stable"] = replace(new_pools["stable"], max_connections=1)
                    new_pools["gray"] = replace(new_pools["gray"], max_connections=3)
                    new_rule = replace(cfg.rules["r1"], gray_weight_bp=6000, mirror_sample_bp=1000)
                    proposed = replace(
                        cfg,
                        version=1,
                        pools=new_pools,
                        rules={"r1": new_rule},
                    )
                    report = engine.replace_config(proposed, borrow_policy="force_return")
                    self.assertEqual(report["new_version"], 2)
                    self.assertTrue(report["version_monotonic"])
                    self.assertEqual(report["stale_versions"], [1])
                    self.assertEqual(len(report["borrow_dispositions"]), 2)
                    self.assertTrue(
                        all(item["action"] == "force_return" for item in report["borrow_dispositions"])
                    )
                    self.assertTrue(report["changed_pools"])
                    self.assertTrue(report["changed_rules"])
                    epochs = {epoch["version"]: epoch for epoch in ledger.ledger_snapshot()["epochs"]}
                    self.assertEqual(epochs[1]["status"], "stale")
                    self.assertEqual(epochs[2]["status"], "active")
                    self.assertEqual(engine.current_view()["stale_versions"], [1])
                finally:
                    engine.close()

    def test_replacement_then_resume_accepts_stale_old_epoch(self) -> None:
        with target_servers() as ports:
            cfg = parse_config(raw_config(ports, gray_weight=0, mirror_sample=0))
            with tempfile.TemporaryDirectory() as temp:
                ledger = Ledger(Path(temp) / "ledger.sqlite")
                engine = ArbiterEngine(cfg, ledger=ledger, snapshot_dir=Path(temp) / "snapshots")
                try:
                    requests = [
                        RequestSpec(f"epoch-{index}", "/api/item", f"k{index}", index)
                        for index in range(4)
                    ]
                    engine.run(requests[:2], batch_size=2)
                    new_pools = dict(cfg.pools)
                    new_rule = replace(cfg.rules["r1"], gray_weight_bp=2500, mirror_sample_bp=500)
                    report = engine.replace_config(
                        replace(cfg, pools=new_pools, rules={"r1": new_rule}),
                        borrow_policy="recompute",
                    )
                    self.assertEqual(report["new_version"], 2)
                    result = engine.run(requests, batch_size=2, resume=True)
                    self.assertEqual(len(result["batches"]), 1)
                    self.assertEqual(result["batches"][0]["config_version"], 2)
                    batches = ledger.list_batches()
                    self.assertEqual([batch["config_version"] for batch in batches], [1, 2])
                    self.assertTrue(batches[0]["stale"])
                    self.assertFalse(batches[1]["stale"])
                    self.assertEqual(ledger.replay_self_check()["requests_checked"], 4)
                    recovery_after = ledger.recover()
                    self.assertTrue(recovery_after["exit_report_available"])
                    self.assertEqual(recovery_after["unknown_resources"], [])
                finally:
                    engine.close()


class StatisticsTests(unittest.TestCase):
    def test_error_report_is_order_independent_and_stable_tiebreak(self) -> None:
        ports = {"stable": free_port(), "gray": free_port(), "mirror": free_port()}
        cfg = parse_config(raw_config(ports, gray_weight=5000, mirror_sample=2500))
        outcomes = [
            {
                "rule_id": "r1",
                "main_destination": "gray" if index % 2 else "stable",
                "mirrored": index % 4 == 0,
            }
            for index in range(40)
        ]
        stats = [
            PoolBatchStats(
                pool_id="stable",
                max_connections=3,
                active_connections=1,
                in_use_connections=1,
                idle_connections=0,
                lent_out=0,
                borrowed_in=0,
                reuse_count=3,
                borrow_count=0,
                closed_count=0,
                sample_count=10,
                average_watermark_scaled=1234,
                peak_watermark=2,
            ),
            PoolBatchStats(
                pool_id="gray",
                max_connections=2,
                active_connections=1,
                in_use_connections=1,
                idle_connections=0,
                lent_out=0,
                borrowed_in=0,
                reuse_count=2,
                borrow_count=1,
                closed_count=0,
                sample_count=10,
                average_watermark_scaled=1234,
                peak_watermark=2,
            ),
            PoolBatchStats(
                pool_id="mirror",
                max_connections=2,
                active_connections=0,
                in_use_connections=0,
                idle_connections=0,
                lent_out=0,
                borrowed_in=0,
                reuse_count=0,
                borrow_count=0,
                closed_count=0,
                sample_count=10,
                average_watermark_scaled=0,
                peak_watermark=0,
            ),
        ]
        first = build_error_report(cfg, outcomes, stats)
        second = build_error_report(cfg, list(reversed(outcomes)), stats)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        comparison = first["watermark_comparisons"][0]
        self.assertTrue(comparison["within_error_bound"])
        self.assertEqual(comparison["winner_by_stable_tiebreak"], "stable")
        self.assertEqual(comparison["stable_order"], ["stable", "gray"])
        for pool in first["pool_errors"].values():
            for contribution in pool["contributions"]:
                self.assertGreaterEqual(contribution["share_bp"], 0)
                self.assertLessEqual(contribution["share_bp"], 10000)
        matrix = numpy_contribution_matrix(first)
        self.assertEqual(matrix["pool_ids"], ["gray", "stable"])

    def test_observed_rates_and_quantization_components_are_present(self) -> None:
        ports = {"stable": free_port(), "gray": free_port(), "mirror": free_port()}
        cfg = parse_config(raw_config(ports, gray_weight=6500, mirror_sample=3500))
        outcomes = [
            {"rule_id": "r1", "main_destination": "gray" if i < 7 else "stable", "mirrored": i < 3}
            for i in range(10)
        ]
        report = build_error_report(cfg, outcomes, [])
        entry = report["pool_errors"]["gray"]
        components = {item["component"] for item in entry["contributions"]}
        self.assertIn("rate_quantization_gray", components)
        self.assertIn("hash_bucket_discretization_gray", components)
        mirror_components = {
            item["component"]
            for item in report["mirror_pool_errors"]["mirror"]["contributions"]
        }
        self.assertIn("rate_quantization_mirror_sample", mirror_components)
        self.assertEqual(report["observed_rates"][0]["observed_gray_bp"], 7000)


class CliTests(unittest.TestCase):
    def test_borrow_cycle_injection_reports_e_borrow(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            ports = {"stable": free_port(), "gray": free_port(), "mirror": free_port()}
            config_path = temp_path / "config.json"
            config_path.write_text(json.dumps(raw_config(ports)), encoding="utf-8")
            requests_path = temp_path / "requests.ndjson"
            requests_path.write_text(
                json.dumps({"request_id": "x", "path": "/api/x", "client_key": "c", "timestamp_ms": 1})
                + "\n",
                encoding="utf-8",
            )
            log_path = temp_path / "log.ndjson"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli.main(
                    [
                        "run",
                        "--config",
                        str(config_path),
                        "--requests",
                        str(requests_path),
                        "--ledger",
                        str(temp_path / "ledger.sqlite"),
                        "--snapshots",
                        str(temp_path / "snapshots"),
                        "--log",
                        str(log_path),
                        "--inject",
                        "borrow-cycle",
                    ]
                )
            self.assertEqual(code, 3)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["status"], "error")
            self.assertEqual(output["error"]["code"], "E_BORROW")
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["business_code"], "E_BORROW")
            for key in ("request_id", "pool_id", "phase"):
                self.assertIn(key, events[-1])

    def test_cli_run_and_replay_end_to_end(self) -> None:
        with target_servers() as ports, tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "config.json"
            config_path.write_text(
                json.dumps(raw_config(ports, gray_weight=7000, mirror_sample=3000)),
                encoding="utf-8",
            )
            requests = []
            for index in range(8):
                requests.append(
                    json.dumps(
                        {
                            "request_id": f"cli-{index}",
                            "path": "/api/items",
                            "client_key": f"client-{index}",
                            "timestamp_ms": 1000 + index,
                        }
                    )
                )
            request_path = temp_path / "requests.ndjson"
            request_path.write_text("\n".join(requests) + "\n", encoding="utf-8")
            ledger_path = temp_path / "ledger.sqlite"
            snapshot_path = temp_path / "snapshots"
            log_path = temp_path / "log.ndjson"
            run_stdout = io.StringIO()
            with contextlib.redirect_stdout(run_stdout):
                run_code = cli.main(
                    [
                        "run",
                        "--config",
                        str(config_path),
                        "--requests",
                        str(request_path),
                        "--ledger",
                        str(ledger_path),
                        "--snapshots",
                        str(snapshot_path),
                        "--log",
                        str(log_path),
                        "--batch-size",
                        "3",
                    ]
                )
            self.assertEqual(run_code, 0)
            run_result = json.loads(run_stdout.getvalue())
            self.assertEqual(run_result["summary"]["processed_requests"], 8)
            self.assertEqual(len(run_result["batches"]), 3)
            replay_stdout = io.StringIO()
            with contextlib.redirect_stdout(replay_stdout):
                replay_code = cli.main(
                    ["replay", "--ledger", str(ledger_path), "--log", str(log_path)]
                )
            self.assertEqual(replay_code, 0)
            replay_result = json.loads(replay_stdout.getvalue())
            self.assertEqual(replay_result["replay"]["requests_checked"], 8)
            self.assertTrue((snapshot_path / "latest.json").exists())

    def test_cli_target_unreachable_and_snapshot_digest_codes(self) -> None:
        with target_servers() as ports, tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "config.json"
            config_path.write_text(
                json.dumps(raw_config(ports, gray_weight=0, mirror_sample=0)),
                encoding="utf-8",
            )
            request_path = temp_path / "requests.ndjson"
            request_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "request_id": f"fault-{index}",
                            "path": "/api/items",
                            "client_key": f"c{index}",
                            "timestamp_ms": 100 + index,
                        }
                    )
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            ledger_path = temp_path / "ledger.sqlite"
            log_path = temp_path / "log.ndjson"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli.main(
                    [
                        "run",
                        "--config",
                        str(config_path),
                        "--requests",
                        str(request_path),
                        "--ledger",
                        str(ledger_path),
                        "--snapshots",
                        str(temp_path / "snapshots"),
                        "--log",
                        str(log_path),
                        "--inject-target-unreachable",
                        "stable",
                    ]
                )
            self.assertEqual(code, 3)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["injection"]["kind"], "target-unreachable")
            self.assertTrue(
                all(
                    request["main_error_code"] == "E_TARGET"
                    for request in result["batches"][0]["requests"]
                )
            )
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["business_code"] == "E_TARGET" for event in events))

            digest_stdout = io.StringIO()
            with contextlib.redirect_stdout(digest_stdout):
                digest_code = cli.main(
                    [
                        "replay",
                        "--ledger",
                        str(ledger_path),
                        "--log",
                        str(log_path),
                        "--inject-snapshot-digest",
                    ]
                )
            self.assertEqual(digest_code, 4)
            digest_result = json.loads(digest_stdout.getvalue())
            self.assertEqual(digest_result["error"]["code"], "E_SNAPSHOT")
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["business_code"], "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
