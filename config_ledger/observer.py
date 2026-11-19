"""Budgeted parallel observation pulling over real loopback sockets."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlencode

from .spec import LOOPBACK_HOST, NodeSpec, Spec
from .structured_log import StructuredLogger


MAX_RESPONSE_BYTES = 1_000_000


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def duration_us(start_ns: int, end_ns: int) -> int:
    return max(0, (end_ns - start_ns) // 1_000)


@dataclass(frozen=True)
class Observation:
    node_id: str
    key: str
    value: int | str
    observation_seq: int
    timestamp_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "key": self.key,
            "value": self.value,
            "observation_seq": self.observation_seq,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass(frozen=True)
class FetchOutcome:
    node_id: str
    key: str
    status: str
    observations: tuple[Observation, ...] = ()
    error_reason: str | None = None
    diagnostic_code: str | None = None
    http_status: int | None = None
    request_start_ms: int = 0
    request_end_ms: int = 0
    request_latency_us: int = 0
    queue_wait_start_ms: int = 0
    queue_wait_end_ms: int = 0
    queued: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def queue_wait_ms(self) -> int:
        return max(0, self.queue_wait_end_ms - self.queue_wait_start_ms)

    def observations_as_list(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.observations]

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "key": self.key,
            "status": self.status,
            "observations": self.observations_as_list(),
            "error_reason": self.error_reason,
            "diagnostic_code": self.diagnostic_code,
            "http_status": self.http_status,
            "request_start_ms": self.request_start_ms,
            "request_end_ms": self.request_end_ms,
            "request_latency_us": self.request_latency_us,
            "queue_wait_start_ms": self.queue_wait_start_ms,
            "queue_wait_end_ms": self.queue_wait_end_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "queued": self.queued,
        }


@dataclass(frozen=True)
class FetchBatch:
    outcomes: tuple[FetchOutcome, ...]
    peak_global: int
    peak_by_node: dict[str, int]
    global_limit: int
    node_quotas: dict[str, int]

    def ordered(self) -> tuple[FetchOutcome, ...]:
        return tuple(sorted(self.outcomes, key=lambda item: (item.node_id, item.key)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_global": self.peak_global,
            "peak_by_node": {node: self.peak_by_node[node] for node in sorted(self.peak_by_node)},
            "global_limit": self.global_limit,
            "node_quotas": {node: self.node_quotas[node] for node in sorted(self.node_quotas)},
            "outcomes": [item.as_dict() for item in self.ordered()],
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_value(value: Any) -> bool:
    return _is_int(value) or isinstance(value, str)


class ObservationFetcher:
    """Pulls key observations while enforcing two independent concurrency budgets."""

    def __init__(self, logger: StructuredLogger | None = None):
        self.logger = logger or StructuredLogger()

    def fetch_all(
        self,
        spec: Spec,
        *,
        window_id: str | None = None,
        parameter_version: int | None = None,
        keys: Iterable[str] | None = None,
        node_ids: Iterable[str] | None = None,
    ) -> FetchBatch:
        node_map = spec.nodes_by_id
        selected_nodes = sorted(set(node_ids)) if node_ids is not None else list(spec.sorted_node_ids)
        selected_keys = sorted(set(keys)) if keys is not None else list(spec.sorted_keys)
        for node_id in selected_nodes:
            if node_id not in node_map:
                raise ValueError(f"unknown node id in fetch selection: {node_id}")
        for key in selected_keys:
            if key not in spec.keys:
                raise ValueError(f"unknown key in fetch selection: {key}")

        jobs = [(node_id, key) for node_id in selected_nodes for key in selected_keys]
        if not jobs:
            return FetchBatch((), 0, {}, spec.global_concurrency, {
                node_id: node_map[node_id].quota for node_id in selected_nodes
            })

        effective_version = parameter_version or spec.parameter_version
        effective_window = window_id or spec.observation_window
        enqueued_ms = monotonic_ms()
        condition = threading.Condition()
        pending: list[tuple[str, str]] = list(jobs)
        results: dict[tuple[str, str], FetchOutcome] = {}
        active_global = 0
        active_by_node: dict[str, int] = {node_id: 0 for node_id in selected_nodes}
        peak_by_node: dict[str, int] = {node_id: 0 for node_id in selected_nodes}
        peak_global = 0
        blocked_at_least_once: set[tuple[str, str]] = set()
        worker_threads: list[threading.Thread] = []

        def worker(job: tuple[str, str], wait_start_ms: int, queued: bool) -> None:
            nonlocal active_global
            node_id, key = job
            request_start_ms = monotonic_ms()
            outcome = self._fetch_one(
                node_map[node_id],
                key=key,
                timeout_ms=spec.observation_timeout_ms,
                window_id=effective_window,
                parameter_version=effective_version,
            )
            request_end_ms = monotonic_ms()
            wait_end_ms = request_start_ms
            completed = FetchOutcome(
                node_id=outcome.node_id,
                key=outcome.key,
                status=outcome.status,
                observations=outcome.observations,
                error_reason=outcome.error_reason,
                diagnostic_code=outcome.diagnostic_code,
                http_status=outcome.http_status,
                request_start_ms=request_start_ms,
                request_end_ms=request_end_ms,
                request_latency_us=outcome.request_latency_us,
                queue_wait_start_ms=wait_start_ms,
                queue_wait_end_ms=wait_end_ms,
                queued=queued,
            )
            with condition:
                active_global -= 1
                active_by_node[node_id] -= 1
                results[job] = completed
                condition.notify_all()
            self._log_outcome(completed)

        with condition:
            while pending or active_global:
                launched = False
                if active_global < spec.global_concurrency:
                    for index, job in enumerate(pending):
                        node_id = job[0]
                        if active_by_node[node_id] >= node_map[node_id].quota:
                            blocked_at_least_once.add(job)
                            continue
                        pending.pop(index)
                        active_global += 1
                        active_by_node[node_id] += 1
                        peak_global = max(peak_global, active_global)
                        peak_by_node[node_id] = max(peak_by_node[node_id], active_by_node[node_id])
                        thread = threading.Thread(
                            target=worker,
                            args=(job, enqueued_ms, job in blocked_at_least_once),
                            name=f"observe-{node_id}-{job[1]}",
                            daemon=True,
                        )
                        worker_threads.append(thread)
                        thread.start()
                        launched = True
                        break
                if launched:
                    continue
                if active_global == 0:
                    # A validated spec always has positive per-node quotas, so
                    # reaching this branch indicates a scheduler invariant loss.
                    raise RuntimeError("observer scheduler made no progress")
                condition.wait(timeout=0.05)

        ordered_outcomes = tuple(results[job] for job in sorted(results))
        for thread in worker_threads:
            thread.join(timeout=2)
        return FetchBatch(
            outcomes=ordered_outcomes,
            peak_global=peak_global,
            peak_by_node={node_id: peak_by_node[node_id] for node_id in sorted(peak_by_node)},
            global_limit=spec.global_concurrency,
            node_quotas={node_id: node_map[node_id].quota for node_id in selected_nodes},
        )

    def _fetch_one(
        self,
        node: NodeSpec,
        *,
        key: str,
        timeout_ms: int,
        window_id: str,
        parameter_version: int,
    ) -> FetchOutcome:
        if node.host != LOOPBACK_HOST:
            raise ValueError("observation fetcher only permits the 127.0.0.1 literal")
        query = urlencode(
            {
                "key": key,
                "window": window_id,
                "parameter_version": str(parameter_version),
            }
        )
        path = f"/observe?{query}"
        connection: http.client.HTTPConnection | None = None
        start_ns = time.perf_counter_ns()
        try:
            connection = http.client.HTTPConnection(
                node.host, node.port, timeout=max(0.001, timeout_ms / 1000.0)
            )
            connection.request(
                "GET",
                path,
                headers={"Host": f"{node.host}:{node.port}", "Connection": "close"},
            )
            response = connection.getresponse()
            body = response.read(MAX_RESPONSE_BYTES + 1)
            http_status = int(response.status)
            if len(body) > MAX_RESPONSE_BYTES:
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    "response_too_large",
                    http_status,
                    start_ns,
                )
            if http_status == 404:
                return self._failure(
                    node,
                    key,
                    "MISSING_KEY",
                    "http_404_missing_key",
                    http_status,
                    start_ns,
                )
            if http_status < 200 or http_status >= 300:
                return self._failure(
                    node,
                    key,
                    "HTTP_ERROR",
                    f"http_{http_status}",
                    http_status,
                    start_ns,
                )
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._failure(
                    node,
                    key,
                    "INVALID_JSON",
                    "invalid_json_payload",
                    http_status,
                    start_ns,
                )
            parsed = self._parse_payload(node, key, payload, http_status, start_ns)
            return parsed
        except (socket.timeout, TimeoutError):
            return self._failure(
                node,
                key,
                "TIMEOUT",
                f"socket_timeout_{timeout_ms}ms",
                None,
                start_ns,
            )
        except ConnectionRefusedError:
            return self._failure(
                node,
                key,
                "CONNECTION_FAILED",
                "connection_refused",
                None,
                start_ns,
                diagnostic_code="E_PORT",
            )
        except (ConnectionError, OSError) as exc:
            reason = f"{exc.__class__.__name__}:{exc}"
            return self._failure(
                node,
                key,
                "CONNECTION_FAILED",
                reason,
                None,
                start_ns,
                diagnostic_code="E_PORT",
            )
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def _parse_payload(
        self,
        node: NodeSpec,
        key: str,
        payload: Any,
        http_status: int,
        start_ns: int,
    ) -> FetchOutcome:
        if not isinstance(payload, dict):
            return self._failure(
                node, key, "INVALID_PAYLOAD", "payload_not_object", http_status, start_ns
            )
        if payload.get("node_id", node.id) != node.id:
            return self._failure(
                node, key, "INVALID_PAYLOAD", "node_id_mismatch", http_status, start_ns
            )
        if payload.get("key", key) != key:
            return self._failure(
                node, key, "INVALID_PAYLOAD", "key_mismatch", http_status, start_ns
            )
        raw_records = payload.get("observations")
        if raw_records is None:
            # Accept a single record as a convenience, but never accept a map
            # with arbitrary keys.
            if all(field in payload for field in ("value", "observation_seq", "timestamp_ms")):
                raw_records = [payload]
            else:
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    "observations_missing",
                    http_status,
                    start_ns,
                )
        if not isinstance(raw_records, list):
            return self._failure(
                node,
                key,
                "INVALID_PAYLOAD",
                "observations_not_array",
                http_status,
                start_ns,
            )
        if not raw_records:
            return self._failure(
                node, key, "EMPTY", "empty_observations", http_status, start_ns
            )
        parsed: list[Observation] = []
        for index, record in enumerate(raw_records):
            if not isinstance(record, dict):
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    f"observation_{index}_not_object",
                    http_status,
                    start_ns,
                )
            record_node = record.get("node_id", node.id)
            record_key = record.get("key", key)
            value = record.get("value")
            seq = record.get("observation_seq")
            timestamp_ms = record.get("timestamp_ms")
            if record_node != node.id:
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    f"observation_{index}_node_id_mismatch",
                    http_status,
                    start_ns,
                )
            if record_key != key:
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    f"observation_{index}_key_mismatch",
                    http_status,
                    start_ns,
                )
            if not _is_value(value):
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    f"observation_{index}_value_type",
                    http_status,
                    start_ns,
                )
            if not _is_int(seq) or seq < 0:
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    f"observation_{index}_observation_seq",
                    http_status,
                    start_ns,
                )
            if not _is_int(timestamp_ms) or timestamp_ms < 0:
                return self._failure(
                    node,
                    key,
                    "INVALID_PAYLOAD",
                    f"observation_{index}_timestamp_ms",
                    http_status,
                    start_ns,
                )
            parsed.append(
                Observation(
                    node_id=node.id,
                    key=key,
                    value=value,
                    observation_seq=seq,
                    timestamp_ms=timestamp_ms,
                )
            )
        end_ns = time.perf_counter_ns()
        return FetchOutcome(
            node_id=node.id,
            key=key,
            status="OK",
            observations=tuple(parsed),
            http_status=http_status,
            request_latency_us=duration_us(start_ns, end_ns),
        )

    def _failure(
        self,
        node: NodeSpec,
        key: str,
        status: str,
        reason: str,
        http_status: int | None,
        start_ns: int,
        *,
        diagnostic_code: str | None = None,
    ) -> FetchOutcome:
        end_ns = time.perf_counter_ns()
        return FetchOutcome(
            node_id=node.id,
            key=key,
            status=status,
            error_reason=reason,
            diagnostic_code=diagnostic_code,
            http_status=http_status,
            request_latency_us=duration_us(start_ns, end_ns),
        )

    def _log_outcome(self, outcome: FetchOutcome) -> None:
        if outcome.ok:
            return
        fields: dict[str, Any] = {
            "node_id": outcome.node_id,
            "key": outcome.key,
            "status": outcome.status,
            "reason": outcome.error_reason,
            "http_status": outcome.http_status,
            "request_latency_us": outcome.request_latency_us,
            "queue_wait_start_ms": outcome.queue_wait_start_ms,
            "queue_wait_end_ms": outcome.queue_wait_end_ms,
            "queued": outcome.queued,
        }
        self.logger.warning(
            "E_OBSERVE",
            "observation missing for this node/key; it remains in the ledger",
            **fields,
        )
        if outcome.diagnostic_code == "E_PORT":
            self.logger.error(
                "E_PORT",
                "observation port is unreachable or refused the connection",
                node_id=outcome.node_id,
                key=outcome.key,
                reason=outcome.error_reason,
                status=outcome.status,
            )
