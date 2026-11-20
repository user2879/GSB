"""Serial and thread-pool execution with deterministic result ordering."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

from .config import stable_json
from .loader import QuerySample
from .pipeline import QueryPipeline, QueryResult


def _run_serial(pipeline: QueryPipeline,
                samples: Sequence[QuerySample]) -> List[QueryResult]:
    return [pipeline.process(s.query_id, s.query, s.context, s.labels)
            for s in samples]


def _run_parallel(pipeline: QueryPipeline, samples: Sequence[QuerySample],
                  max_workers: int) -> List[QueryResult]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(pipeline.process, s.query_id, s.query, s.context, s.labels)
            for s in samples
        ]
        return [future.result() for future in futures]


def _canonical(result: QueryResult) -> str:
    payload = result.to_dict()
    # Timings depend on scheduling; compare every semantic field instead.
    payload.pop("stage_timings_ms", None)
    payload.pop("total_latency_ms", None)
    return stable_json(payload)


def compare_executions(serial: Sequence[QueryResult],
                       parallel: Sequence[QueryResult]) -> dict:
    """Field-by-field comparison of serial vs parallel outputs."""
    mismatches: List[dict] = []
    serial_by_id = {item.query_id: item for item in serial}
    parallel_by_id = {item.query_id: item for item in parallel}
    all_ids = sorted(set(serial_by_id) | set(parallel_by_id))
    for query_id in all_ids:
        left = serial_by_id.get(query_id)
        right = parallel_by_id.get(query_id)
        if left is None or right is None:
            mismatches.append({"query_id": query_id, "reason": "missing_in_one_run"})
            continue
        if _canonical(left) != _canonical(right):
            mismatches.append({"query_id": query_id, "reason": "field_mismatch"})

    ordering_equal = [item.query_id for item in parallel] == \
        [item.query_id for item in serial]
    # Every emitted list must keep the documented deterministic ordering.
    list_order_checks = []
    for left, right in zip(serial, parallel):
        for section, key in (
            ("entity.slots", lambda r: [stable_json(x) for x in (r.entity or {}).get("slots", [])]),
            ("entity.suppressed", lambda r: [stable_json(x) for x in (r.entity or {}).get("suppressed", [])]),
            ("entity.resolutions", lambda r: [stable_json(x) for x in (r.entity or {}).get("resolutions", [])]),
            ("intent.contributions", lambda r: [stable_json(x) for x in (r.intent or {}).get("contributions", [])]),
            ("rewrite.suggestions", lambda r: [stable_json(x) for x in (r.rewrite or {}).get("suggestions", [])]),
            ("failures", lambda r: [stable_json(x) for x in r.failures]),
        ):
            if key(left) != key(right):
                list_order_checks.append({"query_id": left.query_id,
                                          "section": section})
    return {
        "serial_count": len(serial),
        "parallel_count": len(parallel),
        "ordering_equal": ordering_equal,
        "field_mismatch_count": len(mismatches),
        "list_order_mismatches": list_order_checks,
        "equal": not mismatches and ordering_equal and not list_order_checks,
    }


class TierRunner:
    """Executes one scale tier and retains both runs for the audit summary."""

    def __init__(self, pipeline: QueryPipeline, max_workers: int):
        self.pipeline = pipeline
        self.max_workers = max_workers

    def run(self, samples: Sequence[QuerySample]) -> Dict[str, object]:
        serial = _run_serial(self.pipeline, samples)
        parallel = _run_parallel(self.pipeline, samples, self.max_workers)
        # Parallel completion order follows submission order; sorting by the
        # documented stable key makes ordering independent of scheduling.
        parallel_sorted = sorted(parallel, key=lambda r: r.sort_key)
        comparison = compare_executions(serial, parallel_sorted)
        failures = [failure for result in serial for failure in result.failures]
        failures.sort(key=lambda item: (item["sort_key"], item["code"]))
        return {
            "results": serial,
            "parallel_results": parallel_sorted,
            "comparison": comparison,
            "failures": failures,
        }
