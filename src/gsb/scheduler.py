from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .canonical import digest_json
from .executor import MAX_LINE_BYTES, PROTOCOL
from .receipts import normalize_receipt
from .schema import ExecutorSpec


class ExecutorClientError(RuntimeError):
    pass


class RemoteExecutorError(ExecutorClientError):
    def __init__(self, code: str, message: str, field: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.remote_message = message
        self.field = field


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    batch_id: str
    executor_id: str
    shard_id: str
    acquired_mono_ns: int
    deadline_mono_ns: int
    owner_reason: str | None = None
    reclaimed_mono_ns: int | None = None
    receipt_digest: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "batch_id": self.batch_id,
            "executor_id": self.executor_id,
            "shard_id": self.shard_id,
            "acquired_mono_ns": self.acquired_mono_ns,
            "deadline_mono_ns": self.deadline_mono_ns,
            "owner_reason": self.owner_reason,
            "reclaimed_mono_ns": self.reclaimed_mono_ns,
            "receipt_digest": self.receipt_digest,
        }


class LeaseManager:
    """Capacity and lease ownership manager.

    A lease consumes one global slot and one per-executor slot. A lease is
    released by exactly one owner: completion, cancellation, expiry, or a
    transport failure. The first terminal transition wins under the lock.
    """

    def __init__(
        self,
        executors: Iterable[ExecutorSpec],
        *,
        global_inflight: int,
        max_lease_seconds: float,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.executors = tuple(sorted(executors, key=lambda spec: spec.id))
        self.global_inflight = global_inflight
        self.max_lease_seconds = float(max_lease_seconds)
        self._clock_ns = clock_ns
        self._lock = __import__("threading").Lock()
        self._active: dict[str, LeaseRecord] = {}
        self._records: dict[str, LeaseRecord] = {}
        self._global_active = 0
        self._per_executor_active = {spec.id: 0 for spec in self.executors}
        self._sequence = 0
        self._executors_by_id = {spec.id: spec for spec in self.executors}

    @property
    def global_active(self) -> int:
        with self._lock:
            return self._global_active

    def per_executor_active(self) -> dict[str, int]:
        with self._lock:
            return dict(self._per_executor_active)

    def has_capacity(self, executor_id: str) -> bool:
        with self._lock:
            spec = self._executors_by_id.get(executor_id)
            if spec is None:
                return False
            return (
                self._global_active < self.global_inflight
                and self._per_executor_active[executor_id] < spec.concurrency
            )

    def acquire(
        self,
        *,
        executor_id: str,
        batch_id: str,
        shard_id: str,
        lease_seconds: float,
    ) -> LeaseRecord | None:
        if not 0 < lease_seconds <= self.max_lease_seconds:
            raise ValueError("lease_seconds must be positive and <= max_lease_seconds")
        with self._lock:
            spec = self._executors_by_id.get(executor_id)
            if spec is None:
                return None
            if self._global_active >= self.global_inflight:
                return None
            if self._per_executor_active[executor_id] >= spec.concurrency:
                return None
            self._sequence += 1
            now_ns = self._clock_ns()
            lease_id = "lease-" + digest_json(
                {
                    "batch_id": batch_id,
                    "executor_id": executor_id,
                    "shard_id": shard_id,
                    "sequence": self._sequence,
                }
            )[:48]
            record = LeaseRecord(
                lease_id=lease_id,
                batch_id=batch_id,
                executor_id=executor_id,
                shard_id=shard_id,
                acquired_mono_ns=now_ns,
                deadline_mono_ns=now_ns + int(lease_seconds * 1_000_000_000),
            )
            self._active[lease_id] = record
            self._records[lease_id] = record
            self._global_active += 1
            self._per_executor_active[executor_id] += 1
            return record

    def get(self, lease_id: str) -> LeaseRecord | None:
        with self._lock:
            return self._records.get(lease_id)

    def active_records(self) -> tuple[LeaseRecord, ...]:
        with self._lock:
            return tuple(sorted(self._active.values(), key=lambda record: record.lease_id))

    def records(self) -> tuple[LeaseRecord, ...]:
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda record: record.lease_id))

    def complete(self, lease_id: str, *, receipt_digest: str | None = None) -> LeaseRecord:
        with self._lock:
            record = self._records.get(lease_id)
            if record is None:
                raise KeyError(lease_id)
            if record.owner_reason is not None:
                return record
            now_ns = self._clock_ns()
            if now_ns > record.deadline_mono_ns:
                return self._reclaim_locked(record, "expired", now_ns, receipt_digest)
            return self._reclaim_locked(record, "completed", now_ns, receipt_digest)

    def fail(self, lease_id: str, message: str = "transport_failure") -> LeaseRecord:
        with self._lock:
            record = self._records.get(lease_id)
            if record is None:
                raise KeyError(lease_id)
            if record.owner_reason is not None:
                return record
            now_ns = self._clock_ns()
            reason = "expired" if now_ns >= record.deadline_mono_ns else "failed"
            return self._reclaim_locked(record, reason, now_ns, None)

    def reclaim(self, lease_id: str, reason: str) -> LeaseRecord:
        if reason not in ("expired", "cancelled", "failed"):
            raise ValueError("invalid reclaim reason")
        with self._lock:
            record = self._records.get(lease_id)
            if record is None:
                raise KeyError(lease_id)
            if record.owner_reason is not None:
                return record
            now_ns = self._clock_ns()
            return self._reclaim_locked(record, reason, now_ns, None)

    def cancel_batch(self, batch_id: str) -> tuple[LeaseRecord, ...]:
        with self._lock:
            now_ns = self._clock_ns()
            changed: list[LeaseRecord] = []
            for record in sorted(self._active.values(), key=lambda item: item.lease_id):
                if record.batch_id != batch_id:
                    continue
                reason = "expired" if now_ns >= record.deadline_mono_ns else "cancelled"
                changed.append(self._reclaim_locked(record, reason, now_ns, None))
            return tuple(changed)

    def sweep_expired(self) -> tuple[LeaseRecord, ...]:
        with self._lock:
            now_ns = self._clock_ns()
            expired: list[LeaseRecord] = []
            for record in sorted(self._active.values(), key=lambda item: item.lease_id):
                if now_ns >= record.deadline_mono_ns:
                    expired.append(self._reclaim_locked(record, "expired", now_ns, None))
            return tuple(expired)

    def _reclaim_locked(
        self,
        record: LeaseRecord,
        reason: str,
        now_ns: int,
        receipt_digest: str | None,
    ) -> LeaseRecord:
        if record.owner_reason is not None:
            return record
        updated = LeaseRecord(
            lease_id=record.lease_id,
            batch_id=record.batch_id,
            executor_id=record.executor_id,
            shard_id=record.shard_id,
            acquired_mono_ns=record.acquired_mono_ns,
            deadline_mono_ns=record.deadline_mono_ns,
            owner_reason=reason,
            reclaimed_mono_ns=now_ns,
            receipt_digest=receipt_digest,
        )
        self._records[record.lease_id] = updated
        self._active.pop(record.lease_id, None)
        self._global_active -= 1
        self._per_executor_active[record.executor_id] -= 1
        return updated


@dataclass(frozen=True)
class DispatchOutcome:
    shard_id: str
    executor_id: str
    status: str
    lease: LeaseRecord
    receipt_payload: dict[str, Any] | None = None
    adoption_digest: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "executor_id": self.executor_id,
            "status": self.status,
            "lease": self.lease.as_dict(),
            "receipt": self.receipt_payload,
            "adoption_digest": self.adoption_digest,
            "error": self.error,
        }


def _recv_json_line(stream: Any, sock: socket.socket, deadline_mono_ns: int) -> dict[str, Any]:
    remaining_ns = deadline_mono_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise TimeoutError("lease expired before socket read")
    timeout_seconds = remaining_ns / 1_000_000_000
    sock.settimeout(timeout_seconds)
    line = stream.readline(MAX_LINE_BYTES + 1)
    if not line:
        raise ExecutorClientError("executor closed the connection")
    if len(line) > MAX_LINE_BYTES:
        raise ExecutorClientError("executor response line is too wide")
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutorClientError(f"invalid executor JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExecutorClientError("executor response must be an object")
    return payload


def _send_json_line(stream: Any, sock: socket.socket, payload: dict[str, Any], deadline_mono_ns: int) -> None:
    remaining_ns = deadline_mono_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise TimeoutError("lease expired before socket write")
    sock.settimeout(remaining_ns / 1_000_000_000)
    data = (json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    stream.write(data)
    stream.flush()


def send_shard(
    spec: ExecutorSpec,
    request: dict[str, Any],
    *,
    deadline_mono_ns: int,
) -> dict[str, Any]:
    request = dict(request)
    request["type"] = "execute_shard"
    request["protocol"] = PROTOCOL
    remaining_ns = deadline_mono_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise TimeoutError("lease expired before connect")
    sock = socket.create_connection((spec.host, spec.port), timeout=remaining_ns / 1_000_000_000)
    sock.settimeout(remaining_ns / 1_000_000_000)
    try:
        stream = sock.makefile("rwb")
        _send_json_line(
            stream,
            sock,
            {
                "type": "hello",
                "protocol": PROTOCOL,
                "client": "gsb-planner",
                "nonce": request.get("request_id", ""),
            },
            deadline_mono_ns,
        )
        hello = _recv_json_line(stream, sock, deadline_mono_ns)
        if hello.get("type") == "error":
            raise RemoteExecutorError(
                str(hello.get("code", "E_EXEC")),
                str(hello.get("message", "executor handshake rejected")),
                hello.get("field"),
            )
        if hello.get("type") != "hello" or hello.get("protocol") != PROTOCOL:
            raise ExecutorClientError("invalid executor handshake")
        if hello.get("executor_id") != spec.id:
            raise ExecutorClientError(
                f"connected executor id {hello.get('executor_id')} does not match declared {spec.id}"
            )
        _send_json_line(stream, sock, request, deadline_mono_ns)
        response = _recv_json_line(stream, sock, deadline_mono_ns)
        if response.get("type") == "error":
            raise RemoteExecutorError(
                str(response.get("code", "E_EXEC")),
                str(response.get("message", "executor rejected shard")),
                response.get("field"),
            )
        if response.get("type") != "executor_receipt" or response.get("protocol") != PROTOCOL:
            raise ExecutorClientError("invalid executor receipt envelope")
        receipt = response.get("receipt")
        if not isinstance(receipt, dict):
            raise ExecutorClientError("executor receipt is missing")
        if receipt.get("executor_id") != spec.id:
            raise ExecutorClientError("receipt executor_id does not match the connected executor")
        return receipt
    finally:
        try:
            sock.close()
        except OSError:
            pass


class ParallelDispatcher:
    def __init__(
        self,
        executors: Iterable[ExecutorSpec],
        lease_manager: LeaseManager,
        *,
        lease_seconds: float,
    ) -> None:
        self.executors = tuple(sorted(executors, key=lambda spec: spec.id))
        self.lease_manager = lease_manager
        self.lease_seconds = lease_seconds
        self._executors_by_id = {spec.id: spec for spec in executors}
        self._cancel_lock = __import__("threading").Lock()
        self._cancelled_batches: set[str] = set()

    def cancel_batch(self, batch_id: str) -> tuple[LeaseRecord, ...]:
        with self._cancel_lock:
            self._cancelled_batches.add(batch_id)
        return self.lease_manager.cancel_batch(batch_id)

    def dispatch(self, requests: Iterable[dict[str, Any]], batch_id: str) -> tuple[DispatchOutcome, ...]:
        pending: dict[str, list[dict[str, Any]]] = {spec.id: [] for spec in self.executors}
        for request in sorted(requests, key=lambda item: (str(item.get("shard_id")), int(item.get("shard_index", 0)))):
            executor_id = request["executor_id"]
            if executor_id not in pending:
                raise ValueError(f"request names unknown executor {executor_id}")
            pending[executor_id].append(request)

        outcomes: dict[str, DispatchOutcome] = {}
        in_flight: dict[Future[DispatchOutcome], str] = {}
        with ThreadPoolExecutor(max_workers=self.lease_manager.global_inflight) as pool:
            while any(pending.values()) or in_flight:
                self.lease_manager.sweep_expired()
                with self._cancel_lock:
                    cancelled = batch_id in self._cancelled_batches
                if cancelled:
                    for executor_id, requests_queue in pending.items():
                        for request in requests_queue:
                            outcomes[request["shard_id"]] = DispatchOutcome(
                                shard_id=request["shard_id"],
                                executor_id=executor_id,
                                status="cancelled",
                                lease=LeaseRecord(
                                    lease_id="not-issued",
                                    batch_id=batch_id,
                                    executor_id=executor_id,
                                    shard_id=request["shard_id"],
                                    acquired_mono_ns=time.monotonic_ns(),
                                    deadline_mono_ns=time.monotonic_ns(),
                                    owner_reason="cancelled",
                                ),
                            )
                        pending[executor_id] = []

                for executor_id in sorted(pending):
                    while pending[executor_id] and self.lease_manager.has_capacity(executor_id):
                        request = pending[executor_id].pop(0)
                        lease = self.lease_manager.acquire(
                            executor_id=executor_id,
                            batch_id=batch_id,
                            shard_id=request["shard_id"],
                            lease_seconds=self.lease_seconds,
                        )
                        if lease is None:
                            pending[executor_id].insert(0, request)
                            break
                        request = dict(request)
                        request["request_id"] = lease.lease_id
                        request["lease"] = {
                            "lease_id": lease.lease_id,
                            "deadline_mono_ns": lease.deadline_mono_ns,
                            "lease_seconds": self.lease_seconds,
                        }
                        future = pool.submit(self._run_one, executor_id, request, lease)
                        in_flight[future] = executor_id

                if in_flight:
                    done, _ = wait(in_flight, timeout=0.01, return_when=FIRST_COMPLETED)
                    for future in done:
                        executor_id = in_flight.pop(future)
                        try:
                            outcome = future.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            lease_id = getattr(future, "_gsb_lease_id", None)
                            raise RuntimeError(f"dispatch worker failed for {executor_id}: {exc}") from exc
                        outcomes[outcome.shard_id] = outcome
                else:
                    time.sleep(0.005)
        ordered = tuple(outcomes[shard_id] for shard_id in sorted(outcomes))
        return ordered

    def _run_one(self, executor_id: str, request: dict[str, Any], lease: LeaseRecord) -> DispatchOutcome:
        spec = self._executors_by_id[executor_id]
        try:
            receipt = send_shard(spec, request, deadline_mono_ns=lease.deadline_mono_ns)
            normalized = normalize_receipt(receipt)
            completed = self.lease_manager.complete(lease.lease_id, receipt_digest=normalized.adoption_digest)
            if completed.owner_reason == "completed":
                return DispatchOutcome(
                    shard_id=request["shard_id"],
                    executor_id=executor_id,
                    status="completed",
                    lease=completed,
                    receipt_payload=receipt,
                    adoption_digest=normalized.adoption_digest,
                )
            return DispatchOutcome(
                shard_id=request["shard_id"],
                executor_id=executor_id,
                status=str(completed.owner_reason),
                lease=completed,
                receipt_payload=receipt,
                adoption_digest=normalized.adoption_digest,
                error=f"lease reclaimed by {completed.owner_reason}",
            )
        except (TimeoutError, socket.timeout) as exc:
            failed = self.lease_manager.fail(lease.lease_id, f"timeout: {exc}")
            return DispatchOutcome(
                shard_id=request["shard_id"],
                executor_id=executor_id,
                status=str(failed.owner_reason),
                lease=failed,
                error=str(exc),
            )
        except Exception as exc:
            failed = self.lease_manager.fail(lease.lease_id, str(exc))
            return DispatchOutcome(
                shard_id=request["shard_id"],
                executor_id=executor_id,
                status=str(failed.owner_reason),
                lease=failed,
                error=str(exc),
            )


@dataclass(frozen=True)
class RunningExecutor:
    spec: ExecutorSpec
    process: subprocess.Popen[bytes]
    ready: dict[str, Any]


@contextmanager
def launch_executors(
    executors: Iterable[ExecutorSpec],
    runtime_dir: str | Path,
    *,
    startup_timeout: float = 5.0,
) -> Iterator[tuple[ExecutorSpec, ...]]:
    directory = Path(runtime_dir)
    directory.mkdir(parents=True, exist_ok=True)
    processes: list[RunningExecutor] = []
    src_path = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    try:
        for spec in sorted(executors, key=lambda item: item.id):
            ready_file = directory / f"executor-{spec.id}.ready.json"
            if ready_file.exists():
                ready_file.unlink()
            command = [
                sys.executable,
                "-m",
                "gsb.executor",
                "--executor-id",
                spec.id,
                "--host",
                spec.host,
                "--port",
                str(spec.port),
                "--max-concurrency",
                str(spec.concurrency),
                "--ready-file",
                str(ready_file),
            ]
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            deadline = time.monotonic() + startup_timeout
            ready: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    _, stderr = process.communicate(timeout=1)
                    raise ExecutorClientError(f"executor {spec.id} exited before readiness: {stderr.decode('utf-8', 'replace')}")
                if ready_file.exists():
                    try:
                        ready = json.loads(ready_file.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                    if ready is not None:
                        break
                time.sleep(0.02)
            if ready is None:
                process.terminate()
                process.wait(timeout=2)
                raise ExecutorClientError(f"executor {spec.id} did not become ready")
            actual_spec = ExecutorSpec(
                id=spec.id,
                host=str(ready["host"]),
                port=int(ready["port"]),
                concurrency=spec.concurrency,
            )
            processes.append(RunningExecutor(spec=actual_spec, process=process, ready=ready))
        yield tuple(item.spec for item in processes)
    finally:
        for item in processes:
            if item.process.poll() is None:
                item.process.terminate()
        for item in processes:
            try:
                item.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                item.process.kill()
                item.process.wait(timeout=1)
            if item.process.stdout is not None:
                item.process.stdout.close()
            if item.process.stderr is not None:
                item.process.stderr.close()
