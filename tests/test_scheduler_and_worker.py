from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from inference_attribution.errors import DiagnosticError
from inference_attribution.jsonio import canonical_json
from inference_attribution.planner import build_plan
from inference_attribution.scheduler import QuotaScheduler
from inference_attribution.spec import load_requests, load_workers, parse_workers
from inference_attribution.workers import WorkerClient, WorkerProcess
from helpers import SAMPLES, free_port, load_sample


def test_quota_scheduler_uses_thread_pool_and_records_queue_waits():
    scheduler = QuotaScheduler(
        global_limit=2,
        worker_quotas={"worker-a": 1, "worker-b": 1},
    )
    release = threading.Event()
    two_started = threading.Event()
    started: list[tuple[str, int]] = []
    lock = threading.Lock()
    items = [("worker-a", 0), ("worker-a", 1), ("worker-b", 0), ("worker-b", 1)]

    def operation(item):
        with lock:
            started.append(item)
            if len(started) >= 2:
                two_started.set()
        assert release.wait(timeout=5)
        return item

    holder: dict[str, object] = {}

    def run_scheduler():
        try:
            holder["report"] = scheduler.run(
                items,
                operation,
                batch_id=lambda item: f"batch-{item[0]}-{item[1]}",
                worker_id=lambda item: item[0],
            )
        except BaseException as exc:
            holder["error"] = exc

    thread = threading.Thread(target=run_scheduler)
    thread.start()
    assert two_started.wait(timeout=5)
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in holder
    report = holder["report"]
    assert report.max_global_running == 2
    assert report.max_worker_running == {"worker-a": 1, "worker-b": 1}
    assert all(event.thread_name.startswith("ia-batch") for event in report.events())
    assert all(event.run_started_ms >= event.queue_started_ms for event in report.events())
    assert max(event.wait_ms for event in report.events()) > 0


def test_real_worker_loopback_round_trip():
    port = free_port()
    document = load_sample("workers.json")
    document["workers"][0]["port"] = port
    workers = parse_workers(document, object_name="workers.json")
    requests = load_requests(SAMPLES / "requests.json", workers)
    plan = build_plan(workers, requests)
    batch = next(
        batch for batch in plan.batches if batch.worker_id == "worker-alpha"
    )
    worker = workers.by_id()["worker-alpha"]
    with WorkerProcess(worker) as process:
        assert process.process is not None
        response = WorkerClient(worker).dispatch_batch(
            batch, requests.by_id()
        )
    assert response.batch_id == batch.batch_id
    assert len(response.batch_output_digest) == 64
    assert len(response.receipts) == len(batch.request_ids)
    assert {row["request_id"] for row in response.receipts} == set(batch.request_ids)


def test_occupied_port_reports_e_port():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        document = load_sample("workers.json")
        document["workers"][0]["port"] = port
        worker = parse_workers(document, object_name="occupied.json").workers[0]
        with pytest.raises(DiagnosticError) as caught:
            WorkerProcess(worker).start()
        assert caught.value.code == "E_PORT"
        assert caught.value.field == "port"
    finally:
        listener.close()


def test_missing_protocol_field_reports_e_proto():
    port = free_port()
    document = load_sample("workers.json")
    document["workers"][0]["port"] = port
    workers = parse_workers(document, object_name="protocol.json")
    requests = load_requests(SAMPLES / "requests.json", workers)
    plan = build_plan(workers, requests)
    batch = next(
        batch for batch in plan.batches if batch.worker_id == "worker-alpha"
    )
    worker = workers.by_id()["worker-alpha"]
    response = {
        "ok": True,
        "protocol_version": 1,
        "batch_id": batch.batch_id,
        "worker_id": batch.worker_id,
        "param_version_seq": batch.param_version_seq,
        "param_digest": batch.param_digest,
        "latency_ms": 0,
        "batch_output_digest": "0" * 64,
    }

    def serve_twice():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", port))
            server.listen(2)
            with server.accept()[0] as readiness:
                readiness.makefile("rb").readline()
            with server.accept()[0] as connection:
                connection.makefile("rb").readline()
                connection.sendall((canonical_json(response) + "\n").encode("utf-8"))

    thread = threading.Thread(target=serve_twice)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        with pytest.raises(DiagnosticError) as caught:
            WorkerClient(worker).dispatch_batch(batch, requests.by_id())
    finally:
        thread.join(timeout=5)
    assert caught.value.code == "E_PROTO"
    assert caught.value.field == "receipts"
