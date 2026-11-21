from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from txn_mw.engine import Engine, EngineConfig
from txn_mw.errors import ConflictError, ProtocolError
from txn_mw.logging import StructuredLogger
from txn_mw.scenario import QueryStubServer
from txn_mw.verify import verify_database


def wait_for(func, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = func()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"condition not met; last={last}")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log_path = self.root / "structured.jsonl"
        self.logger = StructuredLogger(self.log_path)
        self.stubs: list[QueryStubServer] = []
        self.engines: list[Engine] = []

    def tearDown(self):
        for engine in self.engines:
            try:
                engine.shutdown(report=False)
            except Exception:
                pass
        for stub in self.stubs:
            stub.close()
        self.temp.cleanup()

    def make_engine(self, **overrides):
        config = EngineConfig(
            query_budget_ms=overrides.pop("query_budget_ms", 1000),
            pull_budget_ms=overrides.pop("pull_budget_ms", 200),
            query_retry_ms=overrides.pop("query_retry_ms", 20),
            query_tick_ms=overrides.pop("query_tick_ms", 5),
            **overrides,
        )
        engine = Engine(self.root / "mw.db", config=config, logger=self.logger, start_worker=False)
        self.engines.append(engine)
        return engine

    def stub(self, scripts):
        stub = QueryStubServer({"scripts": scripts})
        self.stubs.append(stub)
        return stub

    def half(self, engine, txn_id, port, timeout_ms=60, query_max=3, body=None):
        return engine.persist_half(
            {
                "txn_id": txn_id,
                "version": 1,
                "body": body or {"txn": txn_id},
                "txn_timeout_ms": timeout_ms,
                "query": {"host": "127.0.0.1", "port": port},
                "query_max": query_max,
            }
        )

    def test_half_is_not_query_eligible_before_ack_marker(self):
        stub = self.stub({"txn-a": [{"result": "committed"}]})
        engine = self.make_engine()
        self.half(engine, "txn-a", stub.port, timeout_ms=30)
        time.sleep(0.08)
        trace = engine.transaction_trace("txn-a")
        self.assertEqual(trace["query_count"], 0)
        self.assertFalse(trace["ack_sent"])
        engine.mark_half_ack("txn-a")
        engine.start()
        committed = wait_for(lambda: (lambda t: t if t["state"] == "committable" else None)(engine.transaction_trace("txn-a")))
        self.assertEqual(committed["query_count"], 1)
        half_event = committed["events"][0]
        query_event = next(event for event in committed["events"] if event["type"] == "query_started")
        self.assertGreaterEqual(query_event["data"]["started_mono_ns"], half_event["data"]["deadline_mono_ns"])
        self.assertTrue(verify_database(engine.store)["ok"])

    def test_query_committed_vs_rollback_conflict_uses_commit_precedence(self):
        stub = self.stub({"txn-conflict": [{"result": "committed", "delay_ms": 180}]})
        engine = self.make_engine(query_budget_ms=1500)
        self.half(engine, "txn-conflict", stub.port, timeout_ms=20)
        engine.mark_half_ack("txn-conflict")
        engine.start()
        wait_for(lambda: engine.transaction_trace("txn-conflict")["query_inflight"])
        deferred = engine.producer_command({"txn_id": "txn-conflict", "version": 1}, "rollback")
        self.assertTrue(deferred["deferred"])
        final = wait_for(
            lambda: (
                lambda t: t
                if t["state"] == "committable" and t["conflict_reason"] == "commit_precedence"
                else None
            )(engine.transaction_trace("txn-conflict"))
        )
        self.assertEqual(final["disposition"], "committable")
        with self.assertRaises(ConflictError):
            engine.producer_command({"txn_id": "txn-conflict"}, "rollback")
        self.assertEqual(engine.transaction_trace("txn-conflict")["state"], "committable")

    def test_idempotent_pull_and_capacity_eviction_do_not_duplicate_effect(self):
        stub = self.stub({"unused": [{"result": "unknown"}]})
        engine = self.make_engine(idem_max_entries=1, idem_ttl_ms=60_000)
        for txn_id in ("idem-a", "idem-b"):
            self.half(engine, txn_id, stub.port)
            engine.mark_half_ack(txn_id)
            engine.producer_command({"txn_id": txn_id, "version": 1}, "commit")
        first = engine.pull({"txn_id": "idem-a", "consumer_id": "c1", "request_id": "p1"})
        duplicate = engine.pull({"txn_id": "idem-a", "consumer_id": "c2", "request_id": "p2"})
        self.assertEqual(first["effect_id"], duplicate["effect_id"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(first["consumption_count"], duplicate["consumption_count"])
        second = engine.pull({"txn_id": "idem-b", "consumer_id": "c3", "request_id": "p3"})
        self.assertFalse(second["duplicate"])
        after_eviction = engine.pull({"txn_id": "idem-a", "consumer_id": "c4", "request_id": "p4"})
        self.assertTrue(after_eviction["duplicate"])
        self.assertEqual(after_eviction["effect_id"], first["effect_id"])
        self.assertEqual(after_eviction["consumption_count"], 1)
        trace = engine.transaction_trace("idem-a")
        self.assertEqual(trace["idempotency_evictions"][0]["reason"], "capacity_exceeded")
        inventory = engine.inventory()
        self.assertEqual(inventory["counts"]["consumed_transactions"], 2)
        self.assertEqual(inventory["counts"]["idempotency_evictions"], 1)
        self.assertTrue(verify_database(engine.store)["ok"])

    def test_query_exhaustion_marks_manual_and_logs_e_query(self):
        stub = self.stub({"txn-exhaust": [{"result": "unknown"}]})
        engine = self.make_engine()
        self.half(engine, "txn-exhaust", stub.port, timeout_ms=20, query_max=2)
        engine.mark_half_ack("txn-exhaust")
        engine.start()
        final = wait_for(
            lambda: (lambda t: t if t["disposition"] == "manual_review" else None)(engine.transaction_trace("txn-exhaust"))
        )
        self.assertEqual(final["state"], "half")
        self.assertEqual(final["query_count"], 2)
        lines = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        query_errors = [line for line in lines if line.get("code") == "E_QUERY"]
        self.assertTrue(query_errors)
        self.assertEqual(query_errors[-1]["txn_id"], "txn-exhaust")
        self.assertEqual(query_errors[-1]["query_count"], 2)

    def test_query_budget_uses_real_socket_wait_and_records_latency(self):
        stub = self.stub({"txn-budget": [{"result": "committed", "delay_ms": 80}]})
        engine = self.make_engine(query_budget_ms=20)
        self.half(engine, "txn-budget", stub.port, timeout_ms=20, query_max=3)
        engine.mark_half_ack("txn-budget")
        engine.start()
        wait_for(lambda: engine.transaction_trace("txn-budget")["query_count"] >= 1)
        time.sleep(0.06)
        stats = engine.latency_stats(txn_id="txn-budget", operation="query")
        self.assertGreaterEqual(stats["overall"]["query"]["sample_count"], 1)
        self.assertGreaterEqual(stats["overall"]["query"]["budget_exceeded_count"], 1)
        lines = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        latency_errors = [line for line in lines if line.get("code") == "E_LATENCY"]
        self.assertTrue(latency_errors)
        self.assertEqual(latency_errors[0]["operation"], "query")
        self.assertEqual(latency_errors[0]["txn_id"], "txn-budget")

    def test_multiple_due_queries_run_concurrently_and_converge_once(self):
        scripts = {
            f"txn-concurrent-{index}": [{"result": "committed", "delay_ms": 160}]
            for index in range(4)
        }
        stub = self.stub(scripts)
        engine = self.make_engine(query_budget_ms=1500, query_workers=4)
        for index in range(4):
            txn_id = f"txn-concurrent-{index}"
            self.half(engine, txn_id, stub.port, timeout_ms=20)
            engine.mark_half_ack(txn_id)
        engine.start()
        for index in range(4):
            txn_id = f"txn-concurrent-{index}"
            final = wait_for(
                lambda txn_id=txn_id: (
                    lambda t: t if t["state"] == "committable" else None
                )(engine.transaction_trace(txn_id))
            )
            self.assertEqual(final["disposition"], "committable")
            self.assertEqual(final["query_count"], 1)
        self.assertEqual(len(stub.observed), 4)
        self.assertTrue(verify_database(engine.store)["ok"])

    def test_late_query_result_cannot_rewrite_rolled_back_terminal(self):
        stub = self.stub({"unused": [{"result": "unknown"}]})
        engine = self.make_engine()
        self.half(engine, "txn-late", stub.port, timeout_ms=500)
        engine.mark_half_ack("txn-late")
        engine.producer_command({"txn_id": "txn-late"}, "rollback")
        start = time.monotonic_ns()
        engine._resolve_query(
            "txn-late",
            1,
            start,
            start + 2_000_000,
            "committed",
            None,
            "txn-late",
        )
        trace = engine.transaction_trace("txn-late")
        self.assertEqual(trace["state"], "half")
        self.assertEqual(trace["disposition"], "rolled_back")
        self.assertEqual(trace["conflicts"][-1]["winner"], "terminal_locked")
        self.assertTrue(any(event["type"] == "conflict" for event in trace["events"]))
        self.assertTrue(any(event["type"] == "late_query_result" for event in trace["events"]))
        self.assertTrue(verify_database(engine.store)["ok"])
        lines = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(line.get("code") == "E_CONFLICT" for line in lines))

    def test_nonstandard_nan_body_is_rejected_as_e_txn(self):
        stub = self.stub({"unused": [{"result": "unknown"}]})
        engine = self.make_engine()
        with self.assertRaises(ProtocolError) as caught:
            engine.persist_half(
                {
                    "txn_id": "txn-nan",
                    "version": 1,
                    "body": {"value": float("nan")},
                    "txn_timeout_ms": 50,
                    "query": {"host": "127.0.0.1", "port": stub.port},
                    "query_max": 3,
                }
            )
        self.assertEqual(caught.exception.code, "E_TXN")
        self.assertEqual(caught.exception.path, "$.body")

    def test_ttl_eviction_records_reason_and_keeps_single_effect(self):
        stub = self.stub({"unused": [{"result": "unknown"}]})
        engine = self.make_engine(idem_max_entries=10, idem_ttl_ms=1000)
        for txn_id in ("ttl-a", "ttl-b"):
            self.half(engine, txn_id, stub.port)
            engine.mark_half_ack(txn_id)
            engine.producer_command({"txn_id": txn_id}, "commit")
        first = engine.pull({"txn_id": "ttl-a", "consumer_id": "c1", "request_id": "ttl-p1"})
        with engine.store.transaction() as conn:
            conn.execute(
                "UPDATE idempotency SET last_access_mono_ns=? WHERE idem_key='ttl-a'",
                (time.monotonic_ns() - 2_000_000_000,),
            )
            engine.store.refresh_digest(conn)
        engine.pull({"txn_id": "ttl-b", "consumer_id": "c2", "request_id": "ttl-p2"})
        trace = engine.transaction_trace("ttl-a")
        self.assertEqual(trace["idempotency_evictions"][-1]["reason"], "ttl_exceeded")
        again = engine.pull({"txn_id": "ttl-a", "consumer_id": "c3", "request_id": "ttl-p3"})
        self.assertTrue(again["duplicate"])
        self.assertEqual(again["effect_id"], first["effect_id"])
        self.assertEqual(again["consumption_count"], 1)


if __name__ == "__main__":
    unittest.main()
