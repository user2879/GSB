"""Deterministic loopback acceptance scenario runner and query stub."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .engine import Engine, EngineConfig
from .logging import StructuredLogger
from .server import RunningServer, request
from .verify import verify_database, verify_ledger_file


class QueryStubServer:
    """Loopback producer transaction-status endpoint driven by a JSON script."""

    def __init__(self, answers: dict[str, Any], host: str = "127.0.0.1", port: int = 0):
        self.answers = answers
        self.host = host
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(64)
        self.port = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.attempts: dict[str, int] = {}
        self.observed: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._serve, name="query-stub", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, address = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn, address), daemon=True).start()

    def _handle(self, conn: socket.socket, address: tuple[str, int]) -> None:
        with conn:
            conn.settimeout(10)
            buffer = bytearray()
            try:
                while b"\n" not in buffer:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buffer.extend(chunk)
                request_value = json.loads(bytes(buffer).split(b"\n", 1)[0].decode("utf-8"))
                txn_id = request_value.get("txn_id")
                with self._lock:
                    attempt = self.attempts.get(txn_id, 0) + 1
                    self.attempts[txn_id] = attempt
                    script = self.answers.get("scripts", {}).get(txn_id)
                    if script is None:
                        script = [{"result": self.answers.get("default", "unknown")}]
                    item = script[min(attempt - 1, len(script) - 1)]
                    self.observed.append(
                        {
                            "txn_id": txn_id,
                            "attempt": attempt,
                            "result": item.get("result", "unknown"),
                            "delay_ms": int(item.get("delay_ms", 0)),
                            "remote": address,
                        }
                    )
                delay_ms = int(item.get("delay_ms", 0))
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000)
                if item.get("close_without_reply"):
                    return
                reply = {
                    "ok": True,
                    "txn_id": txn_id,
                    "result": item.get("result", "unknown"),
                    "attempt": attempt,
                }
                conn.sendall((canonical_json(reply) + "\n").encode("utf-8"))
            except Exception as exc:
                with self._lock:
                    self.observed.append({"error": repr(exc), "remote": address})

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=2)


def _half_request(tx: dict[str, Any], stub_port: int) -> dict[str, Any]:
    return {
        "op": "half",
        "txn_id": tx["txn_id"],
        "version": int(tx.get("version", 1)),
        "body": tx["body"],
        "txn_timeout_ms": int(tx["txn_timeout_ms"]),
        "query": {"host": "127.0.0.1", "port": stub_port},
        "query_max": int(tx.get("query_max", 3)),
    }


def _trace(address: tuple[str, int], txn_id: str) -> dict[str, Any]:
    return request(*address, {"op": "trace", "txn_id": txn_id})


def _wait_for(address: tuple[str, int], txn_id: str, predicate, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = _trace(address, txn_id)
        if predicate(last):
            return last
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {txn_id}; last trace={canonical_json(last)}")


def _wait_query_inflight(address: tuple[str, int], txn_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
    return _wait_for(address, txn_id, lambda t: bool(t.get("query_inflight")), timeout_s)


def run_scenario(
    scenario_path: str | os.PathLike[str],
    *,
    artifacts_dir: str | os.PathLike[str],
    log_path: str | os.PathLike[str],
) -> dict[str, Any]:
    scenario_file = Path(scenario_path).resolve()
    scenario = json.loads(scenario_file.read_text(encoding="utf-8"))
    base = scenario_file.parent
    half_sequence = json.loads((base / scenario["producer_sequence"]).read_text(encoding="utf-8"))
    answers = json.loads((base / scenario["query_answers"]).read_text(encoding="utf-8"))
    pulls = json.loads((base / scenario["consumer_pulls"]).read_text(encoding="utf-8"))
    faults = json.loads((base / scenario["fault_injection"]).read_text(encoding="utf-8"))
    artifacts = Path(artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    log = StructuredLogger(log_path)
    db_path = artifacts / "scenario.db"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    config = EngineConfig(
        query_budget_ms=int(scenario.get("query_budget_ms", 1000)),
        pull_budget_ms=int(scenario.get("pull_budget_ms", 1)),
        query_default_max=3,
        query_retry_ms=int(scenario.get("query_retry_ms", 25)),
        query_tick_ms=int(scenario.get("query_tick_ms", 5)),
        query_workers=8,
        idem_max_entries=int(scenario.get("idempotency", {}).get("max_entries", 2)),
        idem_ttl_ms=int(scenario.get("idempotency", {}).get("ttl_ms", 60_000)),
    )
    stub = QueryStubServer(answers)
    engine = Engine(db_path, config=config, logger=log)
    server = RunningServer(engine)
    server.start()
    address = server.address
    results: dict[str, Any] = {
        "ok": False,
        "scenario": scenario.get("name"),
        "address": {"host": address[0], "port": address[1]},
        "stub_port": stub.port,
        "fault_parameters": faults,
        "transactions": {},
        "errors": [],
    }
    try:
        for tx in half_sequence["transactions"]:
            txn_id = tx["txn_id"]
            half = _half_request(tx, stub.port)
            half_result = request(*address, half)
            results["transactions"][txn_id] = {"half": half_result}
            if tx.get("duplicate_register"):
                duplicate = request(*address, half)
                results["transactions"][txn_id]["duplicate_register"] = duplicate
                if duplicate.get("code") != "E_DUP":
                    results["errors"].append(f"{txn_id}: duplicate registration did not return E_DUP")
            end = tx.get("end")
            if end:
                if end.get("when") == "query_inflight":
                    _wait_query_inflight(address, txn_id)
                elif int(end.get("delay_ms", 0)):
                    time.sleep(int(end["delay_ms"]) / 1000)
                command = {"op": end["op"], "txn_id": txn_id, "version": int(tx.get("version", 1))}
                results["transactions"][txn_id]["command"] = request(*address, command)

        for tx in half_sequence["transactions"]:
            txn_id = tx["txn_id"]
            expected = tx.get("expect", {})
            state = expected.get("state")
            disposition = expected.get("disposition")
            if state is None:
                continue
            try:
                trace = _wait_for(
                    address,
                    txn_id,
                    lambda t: t.get("state") == state and t.get("disposition") == disposition,
                    timeout_s=float(scenario.get("settle_timeout_s", 8)),
                )
                results["transactions"][txn_id]["trace"] = trace
                if expected.get("conflict_reason") and trace.get("conflict_reason") != expected["conflict_reason"]:
                    results["errors"].append(
                        f"{txn_id}: conflict_reason={trace.get('conflict_reason')} expected={expected['conflict_reason']}"
                    )
            except Exception as exc:
                results["errors"].append(f"{txn_id}: {exc}")
                try:
                    results["transactions"][txn_id]["trace"] = _trace(address, txn_id)
                except Exception:
                    pass

        pull_results: list[dict[str, Any]] = []
        concurrent = pulls.get("concurrent", [])
        for group in concurrent:
            txn_id = group["txn_id"]
            consumers = group["consumers"]
            with ThreadPoolExecutor(max_workers=len(consumers)) as pool:
                futures = [
                    pool.submit(
                        request,
                        *address,
                        {"op": "pull", "txn_id": txn_id, "consumer_id": consumer, "request_id": f"conc-{txn_id}-{consumer}"},
                    )
                    for consumer in consumers
                ]
                group_results = [future.result() for future in as_completed(futures)]
            pull_results.extend(group_results)
            effects = {item.get("effect_id") for item in group_results if item.get("ok")}
            if len(effects) != 1:
                results["errors"].append(f"{txn_id}: concurrent pulls produced {len(effects)} effects")
            if sum(1 for item in group_results if not item.get("duplicate")) != 1:
                results["errors"].append(f"{txn_id}: concurrent pulls did not produce exactly one first effect")

        sequential_results: list[dict[str, Any]] = []
        for item in pulls.get("sequential", []):
            response = request(
                *address,
                {
                    "op": "pull",
                    "txn_id": item["txn_id"],
                    "consumer_id": item["consumer_id"],
                    "request_id": item.get("request_id"),
                },
            )
            sequential_results.append(response)
            if item.get("expect_code") and response.get("code") != item["expect_code"]:
                results["errors"].append(
                    f"{item['txn_id']}: pull code={response.get('code')} expected={item['expect_code']}"
                )
        results["pulls"] = {"concurrent": pull_results, "sequential": sequential_results}

        inventory = request(*address, {"op": "inventory"})
        stats = request(*address, {"op": "stats"})
        verify = verify_database(engine.store, logger=log)
        ledger_path = artifacts / "ledger.json"
        export = engine.store.export_ledger(ledger_path)
        verify_file = verify_ledger_file(ledger_path, logger=log)
        results.update(
            {
                "inventory": inventory,
                "latency": stats,
                "verify": verify,
                "ledger_export": export,
                "ledger_verify": verify_file,
                "query_stub_observed": stub.observed,
            }
        )
        if not verify.get("ok"):
            results["errors"].append("independent transaction replay failed")
        if not verify_file.get("ok"):
            results["errors"].append("ledger-file verification failed")
        if stats.get("overall", {}).get("pull", {}).get("budget_exceeded_count", 0) < 1:
            results["errors"].append("pull budget was not observed being exceeded")
        if stats.get("overall", {}).get("query", {}).get("sample_count", 0) < 1:
            results["errors"].append("no query latency samples were recorded")
        trace_map = {}
        for tx in half_sequence["transactions"]:
            trace_map[tx["txn_id"]] = _trace(address, tx["txn_id"])
        results["final_traces"] = trace_map
        results["ok"] = not results["errors"]
    except Exception as exc:
        results["errors"].append(f"{type(exc).__name__}: {exc}")
        results["traceback"] = traceback.format_exc()
    finally:
        try:
            shutdown = server.stop(report=True)
            if shutdown:
                (artifacts / "shutdown_report.json").write_text(canonical_json(shutdown) + "\n", encoding="utf-8")
        finally:
            stub.close()
    (artifacts / "scenario-result.json").write_text(canonical_json(results) + "\n", encoding="utf-8")
    return results


def stub_main(path: str, host: str = "127.0.0.1", port: int = 0) -> QueryStubServer:
    answers = json.loads(Path(path).read_text(encoding="utf-8"))
    return QueryStubServer(answers, host=host, port=port)
