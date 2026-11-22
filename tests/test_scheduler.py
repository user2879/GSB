"""Requirement 6: dual concurrency limits, queue timing, retry, E_EXHAUST."""

import threading
import time

import pytest

from config_regression.errors import E_EXHAUST
from config_regression.model import NodeEntry
from config_regression.scheduler import CollectionJob, Scheduler, now_ms
from config_regression.transport import NodeProcess


def _job(node_id, port, version):
    return CollectionJob(
        job_id=f"job-{version}-{node_id}", node_id=node_id, host="127.0.0.1",
        port=port, manifest_version=version,
        payload={"request_id": f"r{version}{node_id}", "node_id": node_id,
                 "manifest_version": version, "base_seed": 1,
                 "sample_count": 100, "group_count": 2,
                 "aggregations": ["sum"],
                 "items": [{"key": "k", "value": 1.0,
                            "scale_value": 1.0, "quantized_value": 1}]})


def _start(node_id, fault_mode="none", fault_arg=0):
    proc = NodeProcess(NodeEntry(node_id, "127.0.0.1", 0, 1, 2,
                                 fault_mode, fault_arg))
    proc.start()
    return proc


def test_successful_collection_records_queue_window_and_attempts():
    procs = [_start("node-a"), _start("node-b")]
    try:
        jobs = [_job(p.entry.node_id, p.port, v)
                for p in procs for v in (1, 2)]
        scheduler = Scheduler(global_concurrency=2, node_concurrency=1,
                              socket_timeout_ms=2000, max_attempts=2,
                              retry_backoff_ms=5)
        results = scheduler.run_all(jobs)
        assert all(r.status == "ok" for r in results)
        for result in results:
            assert result.wait_ended_monotonic_ms >= \
                result.wait_started_monotonic_ms
            assert result.attempts[0].outcome == "ok"
    finally:
        for proc in procs:
            proc.stop()


def test_concurrency_caps_are_never_exceeded():
    procs = [_start(f"node-{i}") for i in range(3)]
    in_flight = {"global": 0, "per_node": {}}
    lock = threading.Lock()
    max_seen = {"global": 0, "node": 0}
    original = scheduler_run = Scheduler._execute

    def tracked(self, job, wait_started, wait_ended):
        with lock:
            in_flight["global"] += 1
            in_flight["per_node"][job.node_id] = \
                in_flight["per_node"].get(job.node_id, 0) + 1
            max_seen["global"] = max(max_seen["global"],
                                     in_flight["global"])
            max_seen["node"] = max(max_seen["node"],
                                   in_flight["per_node"][job.node_id])
        try:
            return original(self, job, wait_started, wait_ended)
        finally:
            with lock:
                in_flight["global"] -= 1
                in_flight["per_node"][job.node_id] -= 1

    Scheduler._execute = tracked
    try:
        jobs = [_job(p.entry.node_id, p.port, v)
                for p in procs for v in (1, 2, 3)]
        scheduler = Scheduler(global_concurrency=2, node_concurrency=1,
                              socket_timeout_ms=2000, max_attempts=1,
                              retry_backoff_ms=1)
        scheduler.run_all(jobs)
        assert max_seen["global"] <= 2
        assert max_seen["node"] <= 1
    finally:
        Scheduler._execute = original
        for proc in procs:
            proc.stop()


def test_retry_semantics_unchanged_then_exhausted():
    # drop_first: the first accepted request is closed silently, the next
    # identical request succeeds, proving retry preserves collection semantics.
    proc = _start("node-a", fault_mode="drop_first", fault_arg=1)
    try:
        scheduler = Scheduler(1, 1, 1500, max_attempts=3, retry_backoff_ms=5)
        result = scheduler.run_all([_job("node-a", proc.port, 1)])[0]
        assert result.status == "ok"
        assert [a.attempt for a in result.attempts] == [1, 2]
        assert result.attempts[0].outcome == "failed"
        assert result.attempts[1].outcome == "ok"
    finally:
        proc.stop()

    dead = _start("node-b", fault_mode="drop_all")
    try:
        scheduler = Scheduler(1, 1, 500, max_attempts=2, retry_backoff_ms=5)
        result = scheduler.run_all([_job("node-b", dead.port, 1)])[0]
        assert result.status == E_EXHAUST
        assert result.error_code == E_EXHAUST
        assert len(result.attempts) == 2
    finally:
        dead.stop()
