from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .errors import ShaperError


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class ConcurrencyGovernor:
    def __init__(self, global_limit: int, pipeline_limit: int):
        self.global_limit = max(1, int(global_limit))
        self.pipeline_limit = max(1, int(pipeline_limit))
        self._condition = threading.Condition()
        self._global_inflight = 0
        self._pipeline_inflight: dict[str, int] = {}
        self.max_global_seen = 0
        self.max_pipeline_seen: dict[str, int] = {}
        self.queue_entries = 0
        self.total_wait_ms = 0
        self.max_wait_ms = 0

    def acquire(self, pipeline_id: str) -> "Permit":
        wait_start = monotonic_ms()
        with self._condition:
            self.queue_entries += 1
            while (
                self._global_inflight >= self.global_limit
                or self._pipeline_inflight.get(pipeline_id, 0) >= self.pipeline_limit
            ):
                self._condition.wait()
            self._global_inflight += 1
            self._pipeline_inflight[pipeline_id] = self._pipeline_inflight.get(pipeline_id, 0) + 1
            self.max_global_seen = max(self.max_global_seen, self._global_inflight)
            self.max_pipeline_seen[pipeline_id] = max(
                self.max_pipeline_seen.get(pipeline_id, 0),
                self._pipeline_inflight[pipeline_id],
            )
        wait_end = monotonic_ms()
        wait_ms = max(0, wait_end - wait_start)
        with self._condition:
            self.total_wait_ms += wait_ms
            self.max_wait_ms = max(self.max_wait_ms, wait_ms)
        return Permit(self, pipeline_id, wait_start, wait_end)

    def _release(self, pipeline_id: str) -> None:
        with self._condition:
            self._global_inflight -= 1
            self._pipeline_inflight[pipeline_id] -= 1
            if self._pipeline_inflight[pipeline_id] <= 0:
                self._pipeline_inflight.pop(pipeline_id, None)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "global_limit": self.global_limit,
                "pipeline_limit": self.pipeline_limit,
                "max_global_inflight_seen": self.max_global_seen,
                "max_pipeline_inflight_seen": dict(sorted(self.max_pipeline_seen.items())),
                "queue_entries": self.queue_entries,
                "total_wait_ms": self.total_wait_ms,
                "max_wait_ms": self.max_wait_ms,
            }


class Permit:
    def __init__(self, governor: ConcurrencyGovernor, pipeline_id: str, wait_start_ms: int, wait_end_ms: int):
        self.governor = governor
        self.pipeline_id = pipeline_id
        self.wait_start_ms = wait_start_ms
        self.wait_end_ms = wait_end_ms
        self.wait_ms = max(0, wait_end_ms - wait_start_ms)
        self._released = False

    def __enter__(self) -> "Permit":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.governor._release(self.pipeline_id)


class ShaperClient:
    def __init__(self, host: str, port: int, timeout_ms: int):
        self.host = host
        self.port = int(port)
        self.timeout_ms = max(1, int(timeout_ms))

    def judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout_seconds = self.timeout_ms / 1000.0
        request_bytes = canonical_bytes(payload) + b"\n"
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout_seconds) as sock:
                sock.settimeout(timeout_seconds)
                sock.sendall(request_bytes)
                buffer = bytearray()
                while b"\n" not in buffer:
                    chunk = sock.recv(65536)
                    if not chunk:
                        raise ShaperError("shaper closed connection before a receipt", object_id=f"{self.host}:{self.port}")
                    buffer.extend(chunk)
                    if len(buffer) > 16 * 1024 * 1024:
                        raise ShaperError("shaper response exceeded 16 MiB", object_id=f"{self.host}:{self.port}")
                line = bytes(buffer.split(b"\n", 1)[0])
                response = json.loads(line.decode("utf-8"))
        except (socket.timeout, TimeoutError) as exc:
            raise ShaperError(
                f"real socket timeout after {self.timeout_ms} ms",
                object_id=f"{self.host}:{self.port}",
                field="socket_timeout",
            ) from exc
        except OSError as exc:
            raise ShaperError(
                f"socket call failed: {exc}",
                object_id=f"{self.host}:{self.port}",
                field="socket",
            ) from exc
        if not isinstance(response, dict) or not response.get("ok"):
            raise ShaperError(f"shaper returned failure: {response!r}", object_id=f"{self.host}:{self.port}")
        return response


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class ShaperProcessManager:
    def __init__(self, bundle: dict[str, Any], run_dir: str | Path, *, server_delay_ms: int | None = None):
        self.bundle = bundle
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.server_delay_ms = server_delay_ms
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_files: dict[str, Any] = {}

    def _paths(self, shaper_id: str) -> tuple[Path, Path, Path]:
        safe = shaper_id.replace("/", "_").replace("\\", "_")
        return (
            self.run_dir / f"{safe}.pid",
            self.run_dir / f"{safe}.ready",
            self.run_dir / f"{safe}.log",
        )

    def start_one(self, shaper: dict[str, Any], *, ready_timeout_s: float = 10.0) -> subprocess.Popen[bytes]:
        shaper_id = shaper["shaper_id"]
        pid_path, ready_path, log_path = self._paths(shaper_id)
        ready_path.unlink(missing_ok=True)
        log_handle = open(log_path, "ab")
        self.log_files[shaper_id] = log_handle
        spec_dir = self.bundle.get("spec_dir")
        if not spec_dir:
            raise ShaperError("bundle is missing spec_dir", object_id=shaper_id)
        command = [
            sys.executable,
            "-m",
            "qos_diag.shaper_server",
            "--spec-dir",
            str(spec_dir),
            "--shaper-id",
            shaper_id,
            "--ready-file",
            str(ready_path),
        ]
        if self.server_delay_ms is not None:
            command.extend(["--delay-ms", str(int(self.server_delay_ms))])
        env = dict(os.environ)
        root = str(_project_root())
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self.processes[shaper_id] = process
        pid_path.write_text(str(process.pid), encoding="utf-8")
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ShaperError(
                    f"shaper subprocess exited early with code {process.returncode}; see {log_path}",
                    object_id=shaper_id,
                )
            if ready_path.exists():
                try:
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    if int(ready.get("port", -1)) == int(shaper["port"]):
                        return process
                except (json.JSONDecodeError, OSError, ValueError):
                    pass
            time.sleep(0.02)
        raise ShaperError(f"timed out waiting for shaper readiness: {shaper_id}", object_id=shaper_id)

    def start_all(self) -> None:
        for shaper in sorted(self.bundle["shapers"], key=lambda item: item["shaper_id"]):
            self.start_one(shaper)

    def kill(self, shaper_id: str) -> None:
        process = self.processes.get(shaper_id)
        if process is None:
            return
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)

    def restart(self, shaper_id: str) -> None:
        shaper = next(item for item in self.bundle["shapers"] if item["shaper_id"] == shaper_id)
        self.kill(shaper_id)
        self.start_one(shaper)

    def stop_all(self) -> None:
        for shaper_id in list(self.processes):
            process = self.processes[shaper_id]
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 3
        for shaper_id, process in list(self.processes.items()):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for handle in self.log_files.values():
            try:
                handle.close()
            except OSError:
                pass

    def __enter__(self) -> "ShaperProcessManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()


def kill_stale_pid_files(run_dir: str | Path) -> list[int]:
    killed: list[int] = []
    root = Path(run_dir)
    if not root.exists():
        return killed
    pid_files = list(root.rglob("*.pid")) if root.is_dir() else [root]
    for pid_file in pid_files:
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)
            continue
        if _pid_alive(pid):
            _kill_pid(pid)
            killed.append(pid)
        pid_file.unlink(missing_ok=True)
    return killed


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def build_call_payload(bundle: dict[str, Any], result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    arrival = int(result["arrival_ms"])
    elapsed = int(record["elapsed_ms"])
    return {
        "type": "judge",
        "request_id": f"{int(result['event_seq']):06d}:{record['stage']}",
        "scale": int(bundle["config"]["scale"]),
        "window_ms": int(bundle["config"]["window_ms"]),
        "slide_ms": int(bundle["config"]["slide_ms"]),
        "event": {
            "event_seq": int(result["event_seq"]),
            "event_id": result["event_id"],
            "pipeline_id": result["pipeline_id"],
            "arrival_ms": arrival,
            "packet_length": int(result["packet_length"]),
        },
        "previous_state": {
            "balance_q": int(record["balance_before_refill_q"]),
            "state_clock_ms": arrival - elapsed,
        },
        "window_start_ms": record.get("window_start_ms"),
    }


def execute_stage_calls(
    bundle: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    timeout_ms: int | None = None,
    governor: ConcurrencyGovernor | None = None,
    only_stage_keys: set[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    config = bundle["config"]
    timeout_ms = int(timeout_ms or config["call_timeout_ms"])
    governor = governor or ConcurrencyGovernor(
        int(config["global_max_inflight"]), int(config["pipeline_max_inflight"])
    )
    tasks: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], ShaperClient]] = []
    for result in results:
        for record in result["stages"]:
            if not record["evaluated"]:
                continue
            if only_stage_keys is not None and (int(result["event_seq"]), record["stage"]) not in only_stage_keys:
                continue
            payload = build_call_payload(bundle, result, record)
            client = ShaperClient(record["host"], int(record["port"]), timeout_ms)
            tasks.append((result, record, payload, client))

    call_results: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    def run_one(result: dict[str, Any], record: dict[str, Any], payload: dict[str, Any], client: ShaperClient):
        key = f"{result['event_seq']}:{record['stage']}"
        permit = governor.acquire(result["pipeline_id"])
        try:
            call_started = monotonic_ms()
            response = client.judge(payload)
            call_ended = monotonic_ms()
        finally:
            permit.release()
        expected_bytes = canonical_bytes(record)
        actual_bytes = canonical_bytes(response.get("result"))
        if expected_bytes != actual_bytes:
            raise ShaperError(
                "shaper receipt does not match deterministic stage record",
                object_id=record["shaper_id"],
                field="receipt",
            )
        return key, {
            "event_seq": int(result["event_seq"]),
            "pipeline_id": result["pipeline_id"],
            "stage": record["stage"],
            "shaper_id": record["shaper_id"],
            "wait_start_ms": permit.wait_start_ms,
            "wait_end_ms": permit.wait_end_ms,
            "wait_ms": permit.wait_ms,
            "call_started_ms": call_started,
            "call_ended_ms": call_ended,
            "call_ms": max(0, call_ended - call_started),
            "receipt": response,
            "receipt_digest": response.get("record_sha256"),
        }

    max_workers = max(1, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qos-call") as pool:
        futures = {pool.submit(run_one, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result, record, _, _ = task
            key = f"{result['event_seq']}:{record['stage']}"
            try:
                returned_key, call_result = future.result()
                call_results[returned_key] = call_result
            except Exception as exc:  # deterministic batch boundary records failure
                errors.append(
                    {
                        "key": key,
                        "event_seq": int(result["event_seq"]),
                        "pipeline_id": result["pipeline_id"],
                        "stage": record["stage"],
                        "shaper_id": record["shaper_id"],
                        "error": str(exc),
                        "code": getattr(exc, "code", "E_SHAPER"),
                    }
                )
    errors.sort(key=lambda item: item["key"])
    return {
        "calls": call_results,
        "errors": errors,
        "governor": governor.snapshot(),
        "task_count": len(tasks),
        "ok": not errors and len(call_results) == len(tasks),
    }
