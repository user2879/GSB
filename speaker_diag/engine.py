"""Batch payload construction and the shared local/worker inference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import canonical_sha256
from .errors import DiagnosticError
from .model import InferenceResult, ParameterCache, integer_forward
from .spec import RequestSpec, WorkerSpec, WorkloadSpec
from .synthetic import SyntheticBundle


@dataclass(frozen=True)
class BatchPayload:
    batch_id: str
    worker_id: str
    param_version: int
    param_digest: str
    participant_window_ids: tuple[str, ...]
    windows: tuple[dict[str, Any], ...]
    payload_digest: str

    def to_message(self) -> dict[str, Any]:
        return {
            "type": "batch",
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "param_digest": self.param_digest,
            "participant_window_ids": list(self.participant_window_ids),
            "windows": list(self.windows),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "param_digest": self.param_digest,
            "participant_window_ids": list(self.participant_window_ids),
            "windows": list(self.windows),
            "payload_digest": self.payload_digest,
        }


@dataclass(frozen=True)
class RawBatchResult:
    receipt: dict[str, Any]
    inference: tuple[InferenceResult, ...]
    payload: BatchPayload


def make_batch_payloads(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    parameter_cache: ParameterCache | None = None,
) -> tuple[BatchPayload, ...]:
    cache = parameter_cache or ParameterCache()
    payloads: list[BatchPayload] = []
    for request in spec.requests:
        worker = spec.worker_by_id(request.worker_id)
        parameters = cache.get(worker, spec.audio.speaker_count)
        windows = []
        for window_id in request.window_ids:
            windows.append(
                {
                    "window_id": window_id,
                    "feature": list(bundle.feature_for(window_id, worker.model_dim)),
                }
            )
        core = {
            "batch_id": request.batch_id,
            "worker_id": request.worker_id,
            "param_version": worker.parameter_version,
            "param_digest": parameters.digest,
            "participant_window_ids": list(request.window_ids),
            "windows": windows,
        }
        payloads.append(
            BatchPayload(
                batch_id=request.batch_id,
                worker_id=request.worker_id,
                param_version=worker.parameter_version,
                param_digest=parameters.digest,
                participant_window_ids=request.window_ids,
                windows=tuple(windows),
                payload_digest=canonical_sha256(core),
            )
        )
    return tuple(payloads)


def resolve_payload_features(
    worker: WorkerSpec,
    windows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    window_ids: list[str] = []
    features: list[tuple[int, ...]] = []
    for index, window in enumerate(windows):
        window_id = window.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise DiagnosticError("E_FIELD", "window_id must be a non-empty string", field=f"windows[{index}].window_id")
        if window_id in window_ids:
            raise DiagnosticError("E_FIELD", "duplicate window_id in payload", field=f"windows[{index}].window_id")
        feature = window.get("feature")
        if not isinstance(feature, list) or len(feature) != worker.model_dim:
            raise DiagnosticError(
                "E_FIELD",
                f"feature must contain exactly model_dim={worker.model_dim} integers",
                field=f"windows[{index}].feature",
            )
        converted: list[int] = []
        for feature_index, value in enumerate(feature):
            if type(value) is not int:
                raise DiagnosticError(
                    "E_FIELD",
                    "feature values must be integers",
                    field=f"windows[{index}].feature[{feature_index}]",
                )
            converted.append(value)
        window_ids.append(window_id)
        features.append(tuple(converted))
    if not window_ids:
        raise DiagnosticError("E_FIELD", "batch must contain at least one window", field="windows")
    return tuple(window_ids), tuple(features)


def infer_batch(
    spec: WorkloadSpec,
    worker: WorkerSpec,
    batch_id: str,
    window_ids: Iterable[str],
    features_by_window: Mapping[str, Iterable[int]],
    *,
    parameter_cache: ParameterCache | None = None,
    elapsed_ms: int = 0,
    duplicate_of: Mapping[str, str | None] | None = None,
) -> RawBatchResult:
    cache = parameter_cache or ParameterCache()
    parameters = cache.get(worker, spec.audio.speaker_count)
    ordered_ids = tuple(window_ids)
    if not ordered_ids:
        raise DiagnosticError("E_FIELD", "batch must contain at least one window", field="windows")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise DiagnosticError("E_FIELD", "participant windows must be unique", field="participant_window_ids")
    if type(elapsed_ms) is not int or elapsed_ms < 0:
        raise DiagnosticError("E_FIELD", "elapsed_ms must be a non-negative integer", field="elapsed_ms")
    results: list[InferenceResult] = []
    receipt_windows: list[dict[str, Any]] = []
    duplicate_map = duplicate_of or {}
    for rank, window_id in enumerate(ordered_ids, start=1):
        if window_id not in features_by_window:
            raise DiagnosticError("E_FIELD", "missing feature for participant window", field="windows", details={"window_id": window_id})
        feature = tuple(int(value) for value in features_by_window[window_id])
        if len(feature) != worker.model_dim:
            raise DiagnosticError(
                "E_FIELD",
                f"feature length must equal model_dim={worker.model_dim}",
                field="windows",
                details={"window_id": window_id, "length": len(feature)},
            )
        result = integer_forward(parameters, feature)
        results.append(result)
        receipt_windows.append(
            {
                "window_id": window_id,
                "attribution": result.attribution,
                "rank": rank,
                "duplicate_of": duplicate_map.get(window_id),
                "scores": list(result.scores),
                "confidence_q12": result.confidence_q12,
                "score_margin": result.margin,
                "tie_break_used": result.tie_break_used,
            }
        )
    receipt = {
        "format": "r3",
        "format_version": 3,
        "batch_id": batch_id,
        "worker_id": worker.worker_id,
        "param_version": worker.parameter_version,
        "elapsed_ms": elapsed_ms,
        "parameter_digest": parameters.digest,
        "parameter_snapshot": parameters.snapshot,
        "participant_window_ids": list(ordered_ids),
        "windows": receipt_windows,
    }
    payload_core = {
        "batch_id": batch_id,
        "worker_id": worker.worker_id,
        "param_version": worker.parameter_version,
        "param_digest": parameters.digest,
        "participant_window_ids": list(ordered_ids),
        "windows": [
            {"window_id": window_id, "feature": list(features_by_window[window_id])}
            for window_id in ordered_ids
        ],
    }
    payload = BatchPayload(
        batch_id=batch_id,
        worker_id=worker.worker_id,
        param_version=worker.parameter_version,
        param_digest=parameters.digest,
        participant_window_ids=ordered_ids,
        windows=tuple(payload_core["windows"]),
        payload_digest=canonical_sha256(payload_core),
    )
    return RawBatchResult(receipt=receipt, inference=tuple(results), payload=payload)


def infer_local_batch(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    request: RequestSpec,
    *,
    parameter_cache: ParameterCache | None = None,
    elapsed_ms: int = 0,
) -> RawBatchResult:
    worker = spec.worker_by_id(request.worker_id)
    features = {
        window_id: bundle.feature_for(window_id, worker.model_dim)
        for window_id in request.window_ids
    }
    return infer_batch(
        spec,
        worker,
        request.batch_id,
        request.window_ids,
        features,
        parameter_cache=parameter_cache,
        elapsed_ms=elapsed_ms,
    )


def infer_all_local(
    spec: WorkloadSpec,
    bundle: SyntheticBundle,
    *,
    batch_ids: Iterable[str] | None = None,
) -> dict[str, RawBatchResult]:
    cache = ParameterCache()
    wanted = set(batch_ids) if batch_ids is not None else None
    results: dict[str, RawBatchResult] = {}
    for request in spec.requests:
        if wanted is not None and request.batch_id not in wanted:
            continue
        results[request.batch_id] = infer_local_batch(
            spec,
            bundle,
            request,
            parameter_cache=cache,
            elapsed_ms=0,
        )
    return results


def combine_payload_digest(payloads: Iterable[BatchPayload]) -> str:
    return canonical_sha256(sorted(payload.payload_digest for payload in payloads))
