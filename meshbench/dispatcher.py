"""Real loopback request dispatch with budgets, retries and breakers.

Timeouts can only originate from genuine socket timeouts, and connection
failures only from genuine socket connection errors.  The global concurrency
budget and each service's concurrency quota are enforced by real semaphores.
"""

import json
import socket
import threading

from .budget import FAILURE, SLOW, SUCCESS, TIMEOUT
from .breaker import BreakerRegistry
from .clock import Clock, mono_ms
from .config import LOOPBACK, MeshConfig
from .hashing import stable_bucket, select_instance
from .protocol import AttemptRecord, Request, RequestResult
from .errors import CircuitError, UpstreamError


class Dispatcher:
    def __init__(self, config: MeshConfig, registry: BreakerRegistry,
                 clock: Clock = None):
        self.config = config
        self.registry = registry
        self.clock = clock or Clock()
        self.global_sem = threading.BoundedSemaphore(config.global_concurrency)
        self.service_sems = {
            service_id: threading.BoundedSemaphore(service.concurrency_quota)
            for service_id, service in config.services.items()
        }
        self._migration_lock = threading.Lock()
        self.migrations = []

    def update_config(self, config: MeshConfig):
        self.config = config
        self.global_sem = threading.BoundedSemaphore(config.global_concurrency)
        self.service_sems = {
            service_id: threading.BoundedSemaphore(service.concurrency_quota)
            for service_id, service in config.services.items()
        }

    def send_batch(self, requests):
        """Dispatch concurrently; return results sorted by request sequence."""
        results = [None] * len(requests)
        threads = []

        def worker(index, request):
            results[index] = self.send_request(request)

        for index, request in enumerate(requests):
            thread = threading.Thread(
                target=worker, args=(index, request),
                name=f"dispatch-{request.sequence}", daemon=True,
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        return sorted(results, key=lambda item: item.sequence)

    def send_request(self, request: Request) -> RequestResult:
        route = self.config.routes[request.route_id]
        service = self.config.services[route.target_service]
        bucket = stable_bucket(self.config.version, route.route_id, request.client_key)
        # Instance choice is fixed for the whole request; retries reuse it.
        instance = select_instance(service.instances, bucket)
        breaker = self.registry.get(route.target_service, instance)
        started_clock = self.clock.now_ms()
        start_mono = mono_ms()
        attempts = []
        final_outcome = FAILURE

        with self.global_sem, self.service_sems[route.target_service]:
            for attempt in range(1, route.retries + 2):
                record, outcome = self._attempt(
                    request, route, instance, breaker, attempt
                )
                attempts.append(record)
                final_outcome = outcome
                if outcome in (SUCCESS, SLOW):
                    break
                # failure/timeout -> retry while attempts remain.

        result = RequestResult(
            request_id=request.request_id,
            route_id=route.route_id,
            client_key=request.client_key,
            sequence=request.sequence,
            service_id=route.target_service,
            instance_id=instance.instance_id,
            final_outcome=final_outcome,
            attempts=attempts,
            total_elapsed_ms=mono_ms() - start_mono,
            started_clock_ms=started_clock,
            finished_clock_ms=self.clock.now_ms(),
        )
        return result

    def _attempt(self, request, route, instance, breaker, attempt_index):
        allowed, breaker_state = breaker.allow_request()
        if not allowed:
            # Fast fail with no real connection while the breaker is open/full.
            now = mono_ms()
            error = CircuitError(
                "circuit open; fast fail without connection",
                instance_id=instance.instance_id, breaker_state=breaker_state,
            )
            record = AttemptRecord(
                attempt=attempt_index, instance_id=instance.instance_id,
                started_mono_ms=now, elapsed_ms=0, outcome=FAILURE,
                error_code=error.code, detail=str(error),
            )
            return record, FAILURE

        payload = {
            "request_id": request.request_id,
            "route_id": request.route_id,
            "client_key": request.client_key,
            "sequence": request.sequence,
            "attempt": attempt_index,
            "rule_version": self.config.version,
        }
        started = mono_ms()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # The single-attempt timeout budget is fixed by the route and is never
        # widened by retries.
        sock.settimeout(route.attempt_timeout_ms / 1000.0)
        try:
            try:
                sock.connect((LOOPBACK, instance.port))
                sock.sendall(
                    json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
                )
                chunks = []
                while b"\n" not in b"".join(chunks):
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("upstream closed without response")
                    chunks.append(chunk)
            except socket.timeout as exc:
                elapsed = mono_ms() - started
                migration = breaker.record_failure()
                self._note_migration(migration)
                record = AttemptRecord(
                    attempt=attempt_index, instance_id=instance.instance_id,
                    started_mono_ms=started, elapsed_ms=elapsed, outcome=TIMEOUT,
                    detail=f"socket timeout: {exc}",
                )
                return record, TIMEOUT
            except (ConnectionError, OSError) as exc:
                elapsed = mono_ms() - started
                migration = breaker.record_failure()
                self._note_migration(migration)
                error = UpstreamError(
                    "real connection error to loopback upstream",
                    instance_id=instance.instance_id, detail=str(exc),
                )
                record = AttemptRecord(
                    attempt=attempt_index, instance_id=instance.instance_id,
                    started_mono_ms=started, elapsed_ms=elapsed, outcome=FAILURE,
                    error_code=error.code, detail=str(exc),
                )
                return record, FAILURE

            raw = b"".join(chunks).split(b"\n", 1)[0]
            response = json.loads(raw.decode("utf-8"))
            elapsed = mono_ms() - started
            if response.get("status") == "error":
                migration = breaker.record_failure()
                self._note_migration(migration)
                record = AttemptRecord(
                    attempt=attempt_index, instance_id=instance.instance_id,
                    started_mono_ms=started, elapsed_ms=elapsed, outcome=FAILURE,
                    detail="upstream deterministic error",
                )
                return record, FAILURE

            migration = breaker.record_success()
            self._note_migration(migration)
            outcome = SLOW if elapsed >= route.slow_threshold_ms else SUCCESS
            record = AttemptRecord(
                attempt=attempt_index, instance_id=instance.instance_id,
                started_mono_ms=started, elapsed_ms=elapsed, outcome=outcome,
                detail=response.get("status", ""),
            )
            return record, outcome
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _note_migration(self, migration):
        if migration is None:
            return
        with self._migration_lock:
            self.migrations.append(migration)

    def drain_migrations(self):
        with self._migration_lock:
            pending = list(getattr(self, "migrations", []))
            self.migrations = []
            return pending
