"""Dependency-guided invalidation and selective recomputation.

Nodes in the graph are worker parameter identities, batches, and window
instances.  Changes enter at a worker or batch/window-layout node; invalidation
then follows batch -> window and duplicate window -> consuming batch edges.
The latter is deliberately conservative: a shared/retried window makes every
consumer a potential invalidation boundary.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .canonical import canonical_bytes, canonical_sha256
from .engine import RawBatchResult, infer_local_batch
from .errors import DiagnosticError
from .model import ParameterCache
from .receipts import NormalizedReceipt, context_for, normalize_receipt
from .spec import RequestSpec, WorkloadSpec
from .synthetic import SyntheticBundle


REPLAY_IGNORED_FIELDS = frozenset(
    {
        "schema",
        "source_format",
        "source_format_version",
        "filled_fields",
        "fill_basis",
        "elapsed_ms",
        "scores",
        "confidence_q12",
        "score_margin",
        "tie_break_used",
    }
)


def _worker_signature(spec: WorkloadSpec, worker_id: str) -> dict[str, Any]:
    worker = spec.worker_by_id(worker_id)
    return spec.parameter_identity(worker)


def _request_signature(request: RequestSpec) -> dict[str, Any]:
    return {
        "batch_id": request.batch_id,
        "worker_id": request.worker_id,
        "window_ids": list(request.window_ids),
    }


def _layout_changes(old_spec: WorkloadSpec, new_spec: WorkloadSpec) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    if old_spec.seed != new_spec.seed:
        triggers.append({"node": "audio.seed", "old": old_spec.seed, "new": new_spec.seed})
    if old_spec.audio != new_spec.audio:
        triggers.append(
            {
                "node": "audio",
                "old": old_spec.audio.to_dict(),
                "new": new_spec.audio.to_dict(),
            }
        )
    if old_spec.window != new_spec.window:
        triggers.append(
            {
                "node": "window",
                "old": old_spec.window.to_dict(),
                "new": new_spec.window.to_dict(),
            }
        )
    if old_spec.batch_size != new_spec.batch_size:
        triggers.append({"node": "batch.batch_size", "old": old_spec.batch_size, "new": new_spec.batch_size})
    return triggers


def compute_invalidation(old_spec: WorkloadSpec, new_spec: WorkloadSpec) -> dict[str, Any]:
    old_requests = {request.batch_id: request for request in old_spec.requests}
    new_requests = {request.batch_id: request for request in new_spec.requests}
    layout_triggers = _layout_changes(old_spec, new_spec)
    global_construction_change = any(
        trigger["node"] in ("audio.seed", "audio", "window") for trigger in layout_triggers
    )
    global_batch_layout_change = any(
        trigger["node"] in ("batch.batch_size",) for trigger in layout_triggers
    )

    old_worker_ids = {worker.worker_id for worker in old_spec.workers}
    new_worker_ids = {worker.worker_id for worker in new_spec.workers}
    changed_workers: dict[str, dict[str, Any]] = {}
    for worker_id in sorted(old_worker_ids & new_worker_ids):
        old_signature = _worker_signature(old_spec, worker_id)
        new_signature = _worker_signature(new_spec, worker_id)
        if old_signature != new_signature:
            changed_fields = sorted(
                key for key in old_signature if old_signature.get(key) != new_signature.get(key)
            )
            changed_workers[worker_id] = {
                "worker_id": worker_id,
                "old": old_signature,
                "new": new_signature,
                "changed_fields": changed_fields,
            }

    direct_batch_basis: dict[str, list[str]] = {batch_id: [] for batch_id in new_requests}
    for trigger in layout_triggers:
        for batch_id in new_requests:
            direct_batch_basis[batch_id].append(f"global:{trigger['node']}")
    if global_batch_layout_change:
        for batch_id in new_requests:
            if "global:batch.batch_size" not in direct_batch_basis[batch_id]:
                direct_batch_basis[batch_id].append("global:batch.batch_size")

    for batch_id in sorted(set(old_requests) | set(new_requests)):
        if batch_id not in new_requests:
            continue
        new_request = new_requests[batch_id]
        old_request = old_requests.get(batch_id)
        if old_request is None:
            direct_batch_basis[batch_id].append("request:new_batch")
        elif _request_signature(old_request) != _request_signature(new_request):
            reasons = []
            if old_request.worker_id != new_request.worker_id:
                reasons.append("worker_assignment")
            if old_request.window_ids != new_request.window_ids:
                reasons.append("window_composition")
            direct_batch_basis[batch_id].append("request:" + "+".join(reasons or ["changed"]))
        if new_request.worker_id in changed_workers:
            details = changed_workers[new_request.worker_id]
            direct_batch_basis[batch_id].append(
                f"worker_param:{new_request.worker_id}:" + "+".join(details["changed_fields"])
            )

    direct_batches = sorted(
        [batch_id for batch_id, reasons in direct_batch_basis.items() if reasons],
        key=lambda batch_id: (new_requests[batch_id].worker_id, new_requests[batch_id].request_index),
    )
    direct_batch_set = set(direct_batches)
    direct_windows: set[str] = set(new_spec.window_ids) if global_construction_change else set()
    invalid_windows: dict[str, list[str]] = {}
    invalid_batches: set[str] = set(direct_batches)
    indirect_batch_basis: dict[str, list[str]] = {}

    # All direct batches expose their window instances immediately.
    for batch_id in direct_batches:
        for window_id in new_requests[batch_id].window_ids:
            invalid_windows.setdefault(window_id, []).append(f"batch:{batch_id}->window:{window_id}")
    for window_id in sorted(direct_windows):
        invalid_windows.setdefault(window_id, []).append("global:window_construction")

    # Repeated/shared windows conservatively pull their other consuming batches
    # across the invalidation boundary.
    consumers: dict[str, list[str]] = {}
    for request in new_spec.requests:
        for window_id in request.window_ids:
            consumers.setdefault(window_id, []).append(request.batch_id)
    changed = True
    while changed:
        changed = False
        for window_id in sorted(invalid_windows):
            for batch_id in sorted(
                consumers.get(window_id, []),
                key=lambda item: (new_requests[item].worker_id, new_requests[item].request_index),
            ):
                if batch_id not in invalid_batches:
                    invalid_batches.add(batch_id)
                    indirect_batch_basis.setdefault(batch_id, []).append(
                        f"window:{window_id}->batch:{batch_id}"
                    )
                    changed = True
                if batch_id not in direct_batch_set:
                    for other_window in new_requests[batch_id].window_ids:
                        if other_window not in invalid_windows:
                            invalid_windows[other_window] = [
                                f"batch:{batch_id}->window:{other_window}"
                            ]
                            changed = True

    invalid_batches_sorted = sorted(
        invalid_batches,
        key=lambda batch_id: (new_requests[batch_id].worker_id, new_requests[batch_id].request_index),
    )
    invalid_windows_sorted = sorted(
        invalid_windows,
        key=lambda window_id: new_spec.window_ids.index(window_id) if window_id in new_spec.window_ids else 10**12,
    )
    indirect_batches = [batch_id for batch_id in invalid_batches_sorted if batch_id not in direct_batch_set]
    indirect_windows = [window_id for window_id in invalid_windows_sorted if window_id not in direct_windows]

    triggers = list(layout_triggers)
    for worker_id, change in changed_workers.items():
        triggers.append(
            {
                "node": f"worker_param:{worker_id}",
                "old": change["old"],
                "new": change["new"],
                "changed_fields": change["changed_fields"],
            }
        )
    for batch_id in sorted(set(old_requests) | set(new_requests)):
        if batch_id not in old_requests:
            triggers.append({"node": f"request:{batch_id}", "old": None, "new": _request_signature(new_requests[batch_id])})
        elif batch_id not in new_requests:
            triggers.append({"node": f"request:{batch_id}", "old": _request_signature(old_requests[batch_id]), "new": None})
        elif _request_signature(old_requests[batch_id]) != _request_signature(new_requests[batch_id]):
            triggers.append(
                {
                    "node": f"request:{batch_id}",
                    "old": _request_signature(old_requests[batch_id]),
                    "new": _request_signature(new_requests[batch_id]),
                }
            )

    steps: list[dict[str, Any]] = []
    sequence = 0
    for trigger in triggers:
        sequence += 1
        steps.append({"seq": sequence, "edge": "trigger", "node": trigger["node"], "basis": trigger})
    for batch_id in direct_batches:
        request = new_requests[batch_id]
        sequence += 1
        steps.append(
            {
                "seq": sequence,
                "edge": "trigger->batch",
                "node": f"batch:{batch_id}",
                "worker_id": request.worker_id,
                "batch_order": request.request_index,
                "basis": sorted(direct_batch_basis[batch_id]),
            }
        )
        for window_id in request.window_ids:
            sequence += 1
            steps.append(
                {
                    "seq": sequence,
                    "edge": "batch->window",
                    "node": f"window:{window_id}",
                    "basis": f"batch:{batch_id}",
                }
            )
    for batch_id in indirect_batches:
        request = new_requests[batch_id]
        sequence += 1
        steps.append(
            {
                "seq": sequence,
                "edge": "window->batch",
                "node": f"batch:{batch_id}",
                "worker_id": request.worker_id,
                "batch_order": request.request_index,
                "basis": sorted(indirect_batch_basis.get(batch_id, [])),
            }
        )

    all_old_batches = sorted(old_requests)
    all_new_batches = sorted(new_requests)
    return {
        "schema": "speaker-diag-invalidation-v1",
        "old_spec_digest": old_spec.spec_digest,
        "new_spec_digest": new_spec.spec_digest,
        "triggers": triggers,
        "changed_workers": [changed_workers[key] for key in sorted(changed_workers)],
        "direct_batches": [
            {
                "batch_id": batch_id,
                "worker_id": new_requests[batch_id].worker_id,
                "batch_order": new_requests[batch_id].request_index,
                "basis": sorted(direct_batch_basis[batch_id]),
            }
            for batch_id in direct_batches
        ],
        "indirect_batches": [
            {
                "batch_id": batch_id,
                "worker_id": new_requests[batch_id].worker_id,
                "batch_order": new_requests[batch_id].request_index,
                "basis": sorted(indirect_batch_basis.get(batch_id, [])),
            }
            for batch_id in indirect_batches
        ],
        "invalid_batches": [
            {
                "batch_id": batch_id,
                "worker_id": new_requests[batch_id].worker_id,
                "batch_order": new_requests[batch_id].request_index,
                "direct": batch_id in direct_batch_set,
                "basis": sorted(
                    direct_batch_basis.get(batch_id, []) + indirect_batch_basis.get(batch_id, [])
                ),
            }
            for batch_id in invalid_batches_sorted
        ],
        "direct_windows": [
            {"window_id": window_id, "basis": sorted(invalid_windows[window_id])}
            for window_id in sorted(direct_windows)
        ],
        "indirect_windows": [
            {"window_id": window_id, "basis": sorted(invalid_windows[window_id])}
            for window_id in indirect_windows
        ],
        "invalid_windows": [
            {
                "window_id": window_id,
                "direct": window_id in direct_windows,
                "basis": sorted(invalid_windows[window_id]),
            }
            for window_id in invalid_windows_sorted
        ],
        "unchanged_batches": [batch_id for batch_id in all_new_batches if batch_id not in invalid_batches],
        "removed_batches": [batch_id for batch_id in all_old_batches if batch_id not in new_requests],
        "steps": steps,
        "stable_order": "worker_id,batch_order",
        "graph_digest": canonical_sha256(
            {
                "direct_batches": direct_batches,
                "indirect_batches": indirect_batches,
                "invalid_windows": invalid_windows_sorted,
                "steps": steps,
            }
        ),
    }


def find_first_receipt_difference(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    ignore_fields: Iterable[str] = REPLAY_IGNORED_FIELDS,
) -> dict[str, Any] | None:
    ignored = set(ignore_fields)
    batch_id = expected.get("batch_id", actual.get("batch_id"))

    def walk(left: Any, right: Any, path: tuple[Any, ...]) -> dict[str, Any] | None:
        if path and path[-1] in ignored:
            return None
        if type(left) is not type(right):
            return {"batch_id": batch_id, "window_id": None, "field": _path(path), "expected": left, "actual": right}
        if isinstance(left, dict):
            all_keys = sorted(set(left) | set(right), key=str)
            for key in all_keys:
                if key in ignored:
                    continue
                if key not in left:
                    return {"batch_id": batch_id, "window_id": None, "field": _path(path + (key,)), "expected": "<missing>", "actual": right[key]}
                if key not in right:
                    return {"batch_id": batch_id, "window_id": None, "field": _path(path + (key,)), "expected": left[key], "actual": "<missing>"}
                difference = walk(left[key], right[key], path + (key,))
                if difference is not None:
                    return difference
            return None
        if isinstance(left, list):
            if len(left) != len(right):
                return {"batch_id": batch_id, "window_id": None, "field": _path(path) + ".length", "expected": len(left), "actual": len(right)}
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                difference = walk(left_item, right_item, path + (index,))
                if difference is not None:
                    return difference
            return None
        if left != right:
            window_id = None
            if "windows" in path:
                window_index = path.index("windows") + 1
                if window_index < len(path) and isinstance(path[window_index], int):
                    source = expected if path[:1] == ("windows",) else expected
                    windows = source.get("windows", [])
                    if isinstance(windows, list) and path[window_index] < len(windows):
                        candidate = windows[path[window_index]]
                        if isinstance(candidate, dict):
                            window_id = candidate.get("window_id")
            return {"batch_id": batch_id, "window_id": window_id, "field": _path(path), "expected": left, "actual": right}
        return None

    return walk(expected, actual, ())


def _path(parts: tuple[Any, ...]) -> str:
    output = ""
    for part in parts:
        if isinstance(part, int):
            output += f"[{part}]"
        elif output:
            output += f".{part}"
        else:
            output = str(part)
    return output or "$"


def verify_unchanged_receipts(
    previous: Mapping[str, NormalizedReceipt],
    recomputed: Mapping[str, NormalizedReceipt],
    invalid_batches: Iterable[str],
) -> dict[str, Any]:
    invalid = set(invalid_batches)
    unchanged = sorted(set(previous) - invalid)
    checked: list[dict[str, Any]] = []
    for batch_id in unchanged:
        if batch_id not in recomputed:
            raise DiagnosticError(
                "E_REPLAY",
                "an unaffected batch disappeared during selective recomputation",
                field="batch_id",
                details={"batch_id": batch_id},
            )
        left = previous[batch_id].data
        right = recomputed[batch_id].data
        if canonical_bytes(left) != canonical_bytes(right):
            difference = find_first_receipt_difference(left, right, ignore_fields=())
            raise DiagnosticError(
                "E_REPLAY",
                "an unaffected batch changed during selective recomputation",
                field=(difference or {}).get("field"),
                details={
                    "reason": "unaffected_batch_changed",
                    **({"batch_id": batch_id, "window_id": (difference or {}).get("window_id")} if difference else {"batch_id": batch_id}),
                    "difference": difference,
                },
            )
        checked.append(
            {
                "batch_id": batch_id,
                "sha256": previous[batch_id].sha256,
                "attribution_digest": previous[batch_id].attribution_digest,
                "byte_identical": True,
            }
        )
    return {
        "schema": "speaker-diag-unchanged-check-v1",
        "checked_count": len(checked),
        "all_byte_identical": True,
        "batches": checked,
    }


def recompute_invalid_subset(
    old_spec: WorkloadSpec,
    new_spec: WorkloadSpec,
    new_bundle: SyntheticBundle,
    previous: Mapping[str, NormalizedReceipt],
    invalidation: Mapping[str, Any],
) -> dict[str, Any]:
    invalid = [entry["batch_id"] for entry in invalidation["invalid_batches"]]
    cache = ParameterCache()
    combined: dict[str, NormalizedReceipt] = dict(previous)
    recomputed_batches: list[dict[str, Any]] = []
    for batch_id in invalid:
        request = new_spec.request_by_batch(batch_id)
        raw: RawBatchResult = infer_local_batch(
            new_spec,
            new_bundle,
            request,
            parameter_cache=cache,
            elapsed_ms=0,
        )
        normalized = normalize_receipt(raw.receipt, context=context_for(new_spec, batch_id, cache))
        old_sha = previous[batch_id].sha256 if batch_id in previous else None
        combined[batch_id] = normalized
        recomputed_batches.append(
            {
                "batch_id": batch_id,
                "worker_id": request.worker_id,
                "old_sha256": old_sha,
                "new_sha256": normalized.sha256,
                "old_attribution_digest": previous[batch_id].attribution_digest if batch_id in previous else None,
                "new_attribution_digest": normalized.attribution_digest,
                "changed": old_sha != normalized.sha256,
            }
        )
    unchanged_check = verify_unchanged_receipts(previous, combined, invalid)
    return {
        "schema": "speaker-diag-selective-recompute-v1",
        "invalid_batch_count": len(invalid),
        "unchanged_batch_count": len(unchanged_check["batches"]),
        "recomputed": recomputed_batches,
        "unchanged_check": unchanged_check,
        "receipts": combined,
    }
