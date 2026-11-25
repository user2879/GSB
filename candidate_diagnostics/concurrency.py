from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .config import Config
from .fingerprinting import stable_sort_key
from .models import Request
from .pipeline import RequestResult, analyze_request


class ConcurrencyMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.observed_max = 0
        self.started: list[str] = []

    def enter(self, key: str) -> None:
        with self._lock:
            self.active += 1
            self.observed_max = max(self.observed_max, self.active)
            self.started.append(key)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


def analyze_many_concurrent(
    requests: dict[str, Request] | list[Request],
    config: Config,
    *,
    max_workers: int | None = None,
) -> tuple[dict[str, RequestResult], dict[str, Any]]:
    if isinstance(requests, list):
        request_map = {item.request_id: item for item in requests}
    else:
        request_map = dict(requests)
    worker_count = min(config.max_workers, max_workers or config.max_workers)
    if worker_count < 1:
        worker_count = 1
    monitor = ConcurrencyMonitor()
    results: dict[str, RequestResult] = {}
    errors: list[dict[str, str]] = []

    def run_one(request: Request) -> tuple[str, RequestResult | None, str | None]:
        key = stable_sort_key(request.request_id, "normalization_truncation")
        monitor.enter(key)
        try:
            return request.request_id, analyze_request(request, config), key
        except Exception as exc:  # isolation: caller receives deterministic error list
            return request.request_id, None, f"{type(exc).__name__}: {exc}"
        finally:
            monitor.exit()

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="candidate") as pool:
        future_map = {
            pool.submit(run_one, request_map[rid]): stable_sort_key(
                rid, "normalization_truncation"
            )
            for rid in sorted(request_map)
        }
        for future in as_completed(future_map):
            rid, result, error = future.result()
            if result is not None:
                results[rid] = result
            else:
                errors.append({"request_id": rid, "error": error or "unknown"})

    ordered = {rid: results[rid] for rid in sorted(results)}
    report = {
        "configured_max_workers": config.max_workers,
        "actual_max_workers_argument": worker_count,
        "observed_max_concurrency": monitor.observed_max,
        "never_exceeds_config": monitor.observed_max <= config.max_workers,
        "stable_sort_keys": {
            rid: stable_sort_key(rid, "normalization_truncation") for rid in sorted(request_map)
        },
        "start_order_is_scheduler_dependent": monitor.started,
        "errors": sorted(errors, key=lambda item: item["request_id"]),
    }
    return ordered, report
