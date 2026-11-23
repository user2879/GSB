"""Pipeline stage execution over a real thread pool."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from .arbiter import LeaseArbiter, RequestOutcome
from .errors import AppError
from .ledger import Ledger
from .spec import ArtifactSpec, Spec, StageSpec
from .util import monotonic_ms


def _wait_until_expired(arbiter: LeaseArbiter, ledger: Ledger, timeout_s: float) -> dict[str, Any]:
    started = monotonic_ms()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        active = ledger.get_active_leases()
        if not active:
            return {
                "expired": True,
                "wait_ms": monotonic_ms() - started,
                "active_leases": 0,
            }
        time.sleep(0.01)
    active = ledger.get_active_leases()
    return {
        "expired": not active,
        "wait_ms": monotonic_ms() - started,
        "active_leases": len(active),
        "remaining_lease_ids": [lease["lease_id"] for lease in active],
    }


def run_pipeline(
    spec: Spec,
    ledger: Ledger,
    arbiter: LeaseArbiter,
    *,
    wait_for_expiry: bool = True,
) -> dict[str, Any]:
    started_ms = monotonic_ms()
    stage_reports: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    stage_order = spec.topological_stages()
    for stage_index, stage in enumerate(stage_order, start=1):
        stage_started = monotonic_ms()
        stage_workers = min(spec.pipeline.global_concurrency, stage.concurrency)
        stage_gate = threading.BoundedSemaphore(stage.concurrency)
        stage_active = 0
        stage_active_max = 0
        stage_lock = threading.Lock()
        futures: dict[Future, ArtifactSpec] = {}

        def execute(artifact: ArtifactSpec, submitted_monotonic_ms: int) -> dict[str, Any]:
            nonlocal stage_active, stage_active_max
            with stage_gate:
                with stage_lock:
                    stage_active += 1
                    stage_active_max = max(stage_active_max, stage_active)
                request_id = f"{spec.pipeline.pipeline_id}:{stage.stage_id}:{artifact.key}"
                try:
                    outcome = arbiter.request_lease(
                        artifact.key,
                        request_id,
                        stage_id=stage.stage_id,
                        extra={
                            "stage_index": stage_index,
                            "worker_thread": threading.current_thread().name,
                        },
                    )
                    payload = outcome.to_dict()
                    stage_wait_ms = max(
                        0,
                        int(payload["queue_started_monotonic_ms"]) - submitted_monotonic_ms,
                    )
                    payload["stage_id"] = stage.stage_id
                    payload["worker_thread"] = threading.current_thread().name
                    payload["stage_queue_ms"] = stage_wait_ms
                    payload["total_queue_ms"] = stage_wait_ms + int(payload["queue_ms"])
                    return payload
                except AppError as exc:
                    return {
                        "ok": False,
                        "stage_id": stage.stage_id,
                        "artifact_key": artifact.key,
                        "request_id": request_id,
                        "worker_thread": threading.current_thread().name,
                        "stage_queue_ms": max(0, monotonic_ms() - submitted_monotonic_ms),
                        "code": exc.code,
                        "error": exc.to_dict(),
                    }
                except Exception as exc:  # Keep the stage report complete.
                    return {
                        "ok": False,
                        "stage_id": stage.stage_id,
                        "artifact_key": artifact.key,
                        "request_id": request_id,
                        "worker_thread": threading.current_thread().name,
                        "stage_queue_ms": max(0, monotonic_ms() - submitted_monotonic_ms),
                        "code": "E_INTERNAL",
                        "error": {"ok": False, "code": "E_INTERNAL", "message": repr(exc)},
                    }
                finally:
                    with stage_lock:
                        stage_active -= 1

        with ThreadPoolExecutor(
            max_workers=max(1, stage_workers),
            thread_name_prefix=f"stage-{stage.stage_id}",
        ) as pool:
            for artifact in (spec.artifacts[key] for key in stage.artifact_keys):
                submitted_monotonic_ms = monotonic_ms()
                futures[pool.submit(execute, artifact, submitted_monotonic_ms)] = artifact
            stage_results = [future.result() for future in as_completed(futures)]
        stage_results.sort(
            key=lambda item: stage.artifact_keys.index(str(item.get("artifact_key")))
        )
        for result in stage_results:
            requests.append(result)
            if not result.get("ok"):
                errors.append(result)
        stage_reports.append(
            {
                "stage_id": stage.stage_id,
                "stage_index": stage_index,
                "concurrency_quota": stage.concurrency,
                "global_concurrency_quota": spec.pipeline.global_concurrency,
                "pool_workers": stage_workers,
                "observed_stage_inflight_max": stage_active_max,
                "started_monotonic_ms": stage_started,
                "ended_monotonic_ms": monotonic_ms(),
                "duration_ms": max(0, monotonic_ms() - stage_started),
                "artifact_keys": list(stage.artifact_keys),
                "results": stage_results,
            }
        )

    expiry_report: dict[str, Any] | None = None
    if wait_for_expiry:
        # Every lease in the acceptance run is intentionally short-lived.  The
        # wait is real wall/monotonic time, not a synthetic event.
        expiry_report = _wait_until_expired(
            arbiter,
            ledger,
            timeout_s=max(
                5.0,
                len(spec.all_artifact_keys) * arbiter.lease_ms / 1000.0 * 2.0 + 2.0,
            ),
        )
    metrics = arbiter.metrics()
    return {
        "ok": not errors and (expiry_report is None or expiry_report.get("expired", False)),
        "pipeline_id": spec.pipeline.pipeline_id,
        "global_concurrency_quota": spec.pipeline.global_concurrency,
        "lease_ms": arbiter.lease_ms,
        "stage_order": [stage.stage_id for stage in stage_order],
        "stages": stage_reports,
        "requests": requests,
        "errors": errors,
        "expiry": expiry_report,
        "scheduler_metrics": metrics,
        "started_monotonic_ms": started_ms,
        "ended_monotonic_ms": monotonic_ms(),
        "duration_ms": max(0, monotonic_ms() - started_ms),
    }
