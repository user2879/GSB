"""Deterministic staged detection pipeline with exact incremental reuse."""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .algorithms import (
    HistogramState,
    MahalanobisState,
    PartitionState,
    algorithm_ranking,
    fit_histogram_density,
    fit_random_partition_depth,
    fit_robust_mahalanobis,
    score_histogram_density,
    score_random_partition_depth,
    score_robust_mahalanobis,
)
from .canonical import content_key, fsum, quantize_score
from .config import (
    ALGORITHM_NAMES,
    ALGORITHM_VERSIONS,
    DECISION_VERSION,
    PREPROCESS_VERSION,
    algorithm_fingerprint,
    algorithm_weight,
    config_hash,
    enabled_algorithms,
    preprocess_fingerprint,
)
from .data import Dataset, feature_data_fingerprint
from .errors import Failure, sample_failure, sort_failures
from .injection import InjectionController
from .preprocess import PreprocessedData, preprocess_dataset
from .storage import CacheStats, ContentCache, SnapshotManager


ProgressCallback = Callable[[str, int, int, dict[str, Any]], None]


@dataclass
class RunOutcome:
    status: str
    payload: dict[str, Any] | None
    diagnostics: dict[str, Any]
    stopped: bool = False
    stop_stage: str | None = None
    error: str | None = None


@dataclass
class TaskSpec:
    key: tuple[Any, ...]
    operation: Callable[[], Any]
    fallback: Any = None


def _algorithm_order(name: str) -> int:
    return ALGORITHM_NAMES.index(name)


def _score_stats(values: Iterable[float]) -> dict[str, Any]:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p95": None}
    count = len(finite)

    def quantile(q: float) -> float:
        position = (count - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return quantize_score(finite[lower])
        fraction = position - lower
        return quantize_score(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)

    return {
        "count": count,
        "min": quantize_score(finite[0]),
        "max": quantize_score(finite[-1]),
        "mean": quantize_score(fsum(finite) / count),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
    }


def _ranking_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    score = record.get("anomaly_score")
    if score is None or not math.isfinite(float(score)):
        return (1, 0.0, str(record["sample_id"]))
    return (0, -float(score), str(record["sample_id"]))


class DetectionPipeline:
    def __init__(
        self,
        dataset: Dataset,
        config: dict[str, Any],
        *,
        cache: ContentCache | None = None,
        snapshots: SnapshotManager | None = None,
        injection_items: list[dict[str, Any]] | None = None,
        affected_algorithms: list[str] | None = None,
        force_serial: bool = False,
        stop_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ):
        self.dataset = dataset
        self.config = config
        self.cache = cache or ContentCache(None, enabled=False)
        self.snapshots = snapshots
        self.injection_items = list(injection_items or [])
        self.affected_algorithms = sorted(affected_algorithms or [], key=_algorithm_order)
        self.force_serial = force_serial
        self.stop_event = stop_event or threading.Event()
        self.progress = progress
        self.stats = CacheStats()
        self.feature_columns = list(config["data"]["feature_columns"])
        self.data_fingerprint = feature_data_fingerprint(dataset, self.feature_columns)
        self.config_hash = config_hash(config)
        self.concurrency = 1 if force_serial else int(config["interaction"]["concurrency"])
        self.snapshot_interval = int(config["interaction"].get("snapshot_interval", 1))
        self.snapshot_writes: list[dict[str, Any]] = []
        self.recalculator_calls: list[str] = []
        self.reused_algorithms: list[str] = []
        self.recalculated_algorithms: list[str] = []
        self.preprocess_reused = False
        self.decision_reused = False
        self.started_at = time.perf_counter()

    def _emit(self, stage: str, completed: int, total: int, **detail: Any) -> None:
        if self.progress is not None:
            self.progress(stage, int(completed), int(total), detail)

    def _run_tasks(self, specs: list[TaskSpec]) -> list[Any]:
        ordered = sorted(specs, key=lambda item: item.key)
        if not ordered:
            return []
        if self.force_serial or self.concurrency <= 1 or len(ordered) == 1:
            results: list[Any] = []
            for spec in ordered:
                try:
                    results.append(spec.operation())
                except BaseException as exc:  # task failures are converted deterministically
                    results.append(("__task_error__", type(exc).__name__, str(exc), spec.fallback))
            return results
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(ordered)), thread_name_prefix="adw-score") as pool:
            future_map = {pool.submit(spec.operation): spec for spec in ordered}
            completed: dict[tuple[Any, ...], Any] = {}
            for future in as_completed(future_map):
                spec = future_map[future]
                try:
                    completed[spec.key] = future.result()
                except BaseException as exc:
                    completed[spec.key] = ("__task_error__", type(exc).__name__, str(exc), spec.fallback)
        return [completed[spec.key] for spec in ordered]

    def _snapshot(
        self,
        stage: str,
        completed_samples: int,
        artifact_version: str,
        payload: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.snapshots is None:
            return
        info = self.snapshots.write_stage(
            stage=stage,
            config_hash=self.config_hash,
            data_fingerprint=self.data_fingerprint,
            completed_samples=completed_samples,
            artifact_version=artifact_version,
            payload=payload,
            extra=extra,
        )
        self.snapshot_writes.append(info)

    def _stopped(self, stage: str, prepared: PreprocessedData | None = None) -> RunOutcome:
        diagnostics = self._diagnostics(None, prepared)
        diagnostics["stop_stage"] = stage
        return RunOutcome(status="stopped", payload=None, diagnostics=diagnostics, stopped=True, stop_stage=stage)

    def run(self) -> RunOutcome:
        injection = InjectionController(self.injection_items, dataset=self.dataset, config=self.config)
        injection.run_data_size_checks()
        preprocess_fp = preprocess_fingerprint(self.config)
        preprocess_key = content_key(
            "artifact:preprocess:v1",
            {
                "version": PREPROCESS_VERSION,
                "data_fingerprint": self.data_fingerprint,
                "preprocess_fingerprint": preprocess_fp,
            },
        )
        self._emit("preprocess", 0, len(self.dataset.sample_ids), reused=False)
        artifact = self.cache.get("preprocess", preprocess_key, self.stats)
        if artifact is not None:
            try:
                prepared = PreprocessedData.from_payload(artifact["data"])
                self.preprocess_reused = True
            except (KeyError, TypeError, ValueError):
                self.stats.corrupt += 1
                artifact = None
        if artifact is None:
            prepared = preprocess_dataset(self.dataset, self.config)
            self.cache.put(
                "preprocess",
                preprocess_key,
                {"data": prepared.to_payload()},
                {"data_fingerprint": self.data_fingerprint, "preprocess_fingerprint": preprocess_fp},
                self.stats,
            )
        self._emit("preprocess", len(prepared.sample_ids), len(prepared.sample_ids), reused=self.preprocess_reused)
        preprocess_failures = injection.preprocess_failures(prepared)
        self._snapshot(
            "preprocess",
            len(prepared.sample_ids),
            preprocess_key,
            {
                "preprocess_key": preprocess_key,
                "data_fingerprint": self.data_fingerprint,
                "artifact": {"data": prepared.to_payload()},
            },
            {"stage": "preprocess", "reused": self.preprocess_reused},
        )
        if self.stop_event.is_set():
            return self._stopped("preprocess", prepared)

        selected = [name for name in ALGORITHM_NAMES if name in enabled_algorithms(self.config)]
        algorithm_keys: dict[str, str] = {}
        algorithm_artifacts: dict[str, dict[str, Any]] = {}
        missed: list[str] = []
        for name in selected:
            algorithm_fp = algorithm_fingerprint(self.config, name)
            key = content_key(
                "artifact:algorithm-score:v1",
                {
                    "version": ALGORITHM_VERSIONS[name],
                    "data_fingerprint": self.data_fingerprint,
                    "preprocess_key": preprocess_key,
                    "algorithm_fingerprint": algorithm_fp,
                },
            )
            algorithm_keys[name] = key
            cached = self.cache.get("algorithm", key, self.stats)
            if cached is not None and isinstance(cached.get("scores"), dict) and isinstance(cached.get("failures"), list):
                algorithm_artifacts[name] = cached
                self.reused_algorithms.append(name)
            else:
                missed.append(name)

        self._emit("score", 0, len(selected) * len(prepared.sample_ids), missed=missed)
        if missed:
            fits: dict[str, Any] = {}
            fit_failures: dict[str, Failure] = {}
            for name in missed:
                self.recalculator_calls.append(name)
                params = self.config["algorithms"][name]["params"]
                try:
                    if name == "robust_mahalanobis":
                        fits[name] = fit_robust_mahalanobis(prepared.matrix, params)
                    elif name == "histogram_density":
                        fits[name] = fit_histogram_density(prepared.matrix, params)
                    elif name == "random_partition_depth":
                        seed_material = algorithm_keys[name] + "|" + self.data_fingerprint
                        fits[name] = fit_random_partition_depth(prepared.matrix, params, seed_material)
                except BaseException as exc:
                    for sample_id in prepared.sample_ids:
                        fit_failures[sample_id] = Failure(
                            code="E_SAMPLE_STAGE",
                            stage="score",
                            sample_id=sample_id,
                            algorithm=name,
                            exception_type=type(exc).__name__,
                            message=f"algorithm fit failed: {exc}",
                        )
            n = len(prepared.sample_ids)
            chunk_size = max(8, int(math.ceil(n / float(max(1, self.concurrency * 4)))))
            task_specs: list[TaskSpec] = []
            task_metadata: dict[tuple[Any, ...], tuple[str, list[int], list[str]]] = {}
            for name in missed:
                if name not in fits:
                    continue
                for start in range(0, n, chunk_size):
                    indices = list(range(start, min(n, start + chunk_size)))
                    sample_ids = [prepared.sample_ids[i] for i in indices]
                    key = (_algorithm_order(name), sample_ids[0], start)
                    operation = self._make_score_task(name, fits[name], prepared.matrix, indices, sample_ids, algorithm_keys[name])
                    task_specs.append(TaskSpec(key=key, operation=operation))
                    task_metadata[key] = (name, indices, sample_ids)
            task_results = self._run_tasks(task_specs)
            raw_scores: dict[str, dict[str, float]] = {name: {} for name in missed}
            raw_failures: dict[str, list[dict[str, Any]]] = {name: [] for name in missed}
            for spec, result in zip(sorted(task_specs, key=lambda x: x.key), task_results):
                name, indices, sample_ids = task_metadata[spec.key]
                if isinstance(result, tuple) and result and result[0] == "__task_error__":
                    exc_name, message = result[1], result[2]
                    for sample_id in sample_ids:
                        raw_failures[name].append(
                            Failure(
                                code="E_SAMPLE_STAGE",
                                stage="score",
                                sample_id=sample_id,
                                algorithm=name,
                                exception_type=exc_name,
                                message=message,
                            ).to_dict()
                        )
                    continue
                scores_part, failures_part = result
                raw_scores[name].update(scores_part)
                raw_failures[name].extend(failures_part)
            for name in missed:
                scores = raw_scores.get(name, {})
                failures = raw_failures.get(name, [])
                for sample_id, failure in fit_failures.items():
                    failures.append(failure.to_dict())
                artifact = {
                    "algorithm": name,
                    "scores": dict(sorted(scores.items())),
                    "failures": sort_failures(failures),
                }
                algorithm_artifacts[name] = artifact
                self.cache.put(
                    "algorithm",
                    algorithm_keys[name],
                    artifact,
                    {
                        "algorithm": name,
                        "data_fingerprint": self.data_fingerprint,
                        "preprocess_key": preprocess_key,
                    },
                    self.stats,
                )
                self._emit(
                    "score",
                    (missed.index(name) + 1) * n,
                    len(selected) * n,
                    algorithm=name,
                    recomputed=True,
                )
        self.recalculated_algorithms = sorted(self.recalculator_calls, key=_algorithm_order)
        injection.score_failures()
        # Ensure artifacts exist for every selected algorithm and preserve canonical insertion.
        algorithm_artifacts = {name: algorithm_artifacts[name] for name in selected}
        score_payload = {
            "preprocess_key": preprocess_key,
            "algorithm_keys": algorithm_keys,
            "artifacts": algorithm_artifacts,
        }
        score_version = content_key("snapshot:score:v1", algorithm_keys)
        # snapshot_interval controls how many completed algorithm artifacts are committed.
        if selected:
            for ordinal, name in enumerate(selected, 1):
                if ordinal % self.snapshot_interval == 0 or ordinal == len(selected):
                    partial_names = selected[:ordinal]
                    self._snapshot(
                        "score",
                        len(prepared.sample_ids),
                        content_key("snapshot:score-partial:v1", {k: algorithm_keys[k] for k in partial_names}),
                        {
                            "preprocess_key": preprocess_key,
                            "algorithm_keys": {k: algorithm_keys[k] for k in partial_names},
                            "artifacts": {k: algorithm_artifacts[k] for k in partial_names},
                        },
                        {"stage": "score", "completed_algorithms": partial_names},
                    )
        if self.stop_event.is_set():
            return self._stopped("score", prepared)

        weights = {name: algorithm_weight(self.config, name) for name in selected}
        decision_key = content_key(
            "artifact:decision:v1",
            {
                "version": DECISION_VERSION,
                "data_fingerprint": self.data_fingerprint,
                "preprocess_key": preprocess_key,
                "algorithms": [{"name": name, "key": algorithm_keys[name], "weight": weights[name]} for name in selected],
                "threshold": quantize_score(float(self.config["threshold"])),
            },
        )
        self._emit("decision", 0, len(prepared.sample_ids), reused=False)
        decision_artifact = self.cache.get("decision", decision_key, self.stats)
        if decision_artifact is not None and isinstance(decision_artifact.get("records"), dict):
            records = decision_artifact["records"]
            if set(records) == set(prepared.sample_ids):
                self.decision_reused = True
            else:
                decision_artifact = None
                records = {}
        else:
            decision_artifact = None
            records = {}
        if decision_artifact is None:
            records = self._compute_decisions(prepared, selected, algorithm_artifacts, weights)
            self.cache.put(
                "decision",
                decision_key,
                {"records": records},
                {
                    "data_fingerprint": self.data_fingerprint,
                    "preprocess_key": preprocess_key,
                    "algorithm_keys": algorithm_keys,
                },
                self.stats,
            )
        self._emit("decision", len(prepared.sample_ids), len(prepared.sample_ids), reused=self.decision_reused)

        decision_failures = injection.decision_failures(records, float(self.config["threshold"]))
        # The controller accumulates each injected stage exactly once.
        all_failures: list[Failure | dict[str, Any]] = list(injection.failures)
        for name in selected:
            all_failures.extend(algorithm_artifacts[name].get("failures", []))
        failures = sort_failures(all_failures)
        failed_ids = {str(item["sample_id"]) for item in failures if item.get("sample_id")}

        samples: list[dict[str, Any]] = []
        for sample_id in prepared.sample_ids:
            base = dict(records[sample_id])
            base["sample_id"] = sample_id
            if sample_id in failed_ids:
                base["failed"] = True
                base["is_anomaly"] = False
                base["anomaly_score"] = None
            else:
                base["failed"] = bool(base.get("failed", False))
            samples.append(base)
        ranking = sorted((dict(item) for item in samples), key=_ranking_sort_key)
        anomalies = [str(item["sample_id"]) for item in ranking if bool(item.get("is_anomaly"))]
        algorithm_rankings = {
            name: algorithm_ranking(algorithm_artifacts[name].get("scores", {}), failed_ids)
            for name in selected
        }
        score_table = [
            {"sample_id": item["sample_id"], "algorithm": name, "score": item["score"]}
            for name in selected
            for item in algorithm_rankings[name]
        ]
        summary = {
            "total_samples": len(prepared.sample_ids),
            "completed_samples": len(prepared.sample_ids),
            "anomaly_count": len(anomalies),
            "failed_sample_count": len(failed_ids),
            "selected_algorithms": selected,
            "affected_algorithms": list(self.affected_algorithms),
        }
        payload = {
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "preprocess_key": preprocess_key,
            "algorithm_keys": algorithm_keys,
            "decision_key": decision_key,
            "summary": summary,
            "samples": samples,
            "ranking": [
                {
                    "sample_id": item["sample_id"],
                    "anomaly_score": item["anomaly_score"],
                    "is_anomaly": bool(item["is_anomaly"]),
                    "failed": bool(item.get("failed", False)),
                }
                for item in ranking
            ],
            "anomalies": anomalies,
            "algorithm_rankings": algorithm_rankings,
            "score_table": score_table,
            "failures": failures,
        }
        self._snapshot(
            "decision",
            len(prepared.sample_ids),
            decision_key,
            {
                "decision_key": decision_key,
                "records": records,
                "failures": failures,
                "summary": summary,
            },
            {"stage": "decision", "reused": self.decision_reused},
        )
        diagnostics = self._diagnostics(payload, prepared, injection)
        diagnostics["decision_reused"] = self.decision_reused
        if self.stop_event.is_set():
            return RunOutcome(status="stopped", payload=payload, diagnostics=diagnostics, stopped=True, stop_stage="decision")
        return RunOutcome(status="ok", payload=payload, diagnostics=diagnostics)

    def _make_score_task(
        self,
        name: str,
        state: Any,
        matrix: np.ndarray,
        indices: list[int],
        sample_ids: list[str],
        algorithm_key: str,
    ) -> Callable[[], Any]:
        def operation() -> Any:
            scores: dict[str, float] = {}
            failures: list[dict[str, Any]] = []
            for sample_index, sample_id in zip(indices, sample_ids):
                x = matrix[sample_index]
                try:
                    if name == "robust_mahalanobis":
                        score = score_robust_mahalanobis(state, x)
                    elif name == "histogram_density":
                        score = score_histogram_density(state, sample_index, x)
                    elif name == "random_partition_depth":
                        score = score_random_partition_depth(state, x)
                    else:
                        raise ValueError(f"unknown algorithm: {name}")
                    scores[sample_id] = quantize_score(score)
                except BaseException as exc:
                    failures.append(sample_failure(exc, stage="score", sample_id=sample_id, algorithm=name).to_dict())
            return scores, failures

        return operation

    def _normalization_ranges(self, selected: list[str], artifacts: dict[str, dict[str, Any]]) -> dict[str, tuple[float, float]]:
        ranges: dict[str, tuple[float, float]] = {}
        for name in selected:
            values = [float(x) for x in artifacts[name].get("scores", {}).values() if math.isfinite(float(x))]
            if not values:
                ranges[name] = (0.0, 0.0)
            else:
                ranges[name] = (min(values), max(values))
        return ranges

    def _compute_decisions(
        self,
        prepared: PreprocessedData,
        selected: list[str],
        artifacts: dict[str, dict[str, Any]],
        weights: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        ranges = self._normalization_ranges(selected, artifacts)
        threshold = quantize_score(float(self.config["threshold"]))
        n = len(prepared.sample_ids)
        chunk_size = max(8, int(math.ceil(n / float(max(1, self.concurrency * 8)))))
        specs: list[TaskSpec] = []
        chunk_sample_ids: dict[tuple[Any, ...], list[str]] = {}
        for start in range(0, n, chunk_size):
            ids = prepared.sample_ids[start : start + chunk_size]
            key = (_algorithm_order(selected[0]), ids[0], start)
            specs.append(
                TaskSpec(
                    key=key,
                    operation=self._make_decision_task(ids, selected, artifacts, weights, ranges, threshold),
                )
            )
            chunk_sample_ids[key] = ids
        results = self._run_tasks(specs)
        records: dict[str, dict[str, Any]] = {}
        for spec, result in zip(sorted(specs, key=lambda x: x.key), results):
            ids = chunk_sample_ids[spec.key]
            if isinstance(result, tuple) and result and result[0] == "__task_error__":
                exc_name, message = result[1], result[2]
                for sample_id in ids:
                    records[sample_id] = {
                        "sample_id": sample_id,
                        "scores": {},
                        "normalized_scores": {},
                        "contributions": [],
                        "anomaly_score": None,
                        "is_anomaly": False,
                        "failed": True,
                        "failure_exception_type": exc_name,
                        "failure_message": message,
                    }
                continue
            records.update(result)
        return {sample_id: records[sample_id] for sample_id in prepared.sample_ids}

    def _make_decision_task(
        self,
        sample_ids: list[str],
        selected: list[str],
        artifacts: dict[str, dict[str, Any]],
        weights: dict[str, float],
        ranges: dict[str, tuple[float, float]],
        threshold: float,
    ) -> Callable[[], dict[str, dict[str, Any]]]:
        def operation() -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            total_weight = fsum(weights[name] for name in selected)
            for sample_id in sample_ids:
                raw_scores: dict[str, float] = {}
                normalized: dict[str, float] = {}
                contributions: list[dict[str, Any]] = []
                missing = False
                for name in selected:
                    value = artifacts[name].get("scores", {}).get(sample_id)
                    if value is None:
                        missing = True
                        break
                    raw = quantize_score(float(value))
                    low, high = ranges[name]
                    span = high - low
                    normalized_value = 0.0 if span <= 1.0e-15 else quantize_score((raw - low) / span)
                    contribution = quantize_score(normalized_value * weights[name])
                    raw_scores[name] = raw
                    normalized[name] = normalized_value
                    contributions.append(
                        {
                            "algorithm": name,
                            "raw_score": raw,
                            "normalized_score": normalized_value,
                            "weight": weights[name],
                            "weighted_contribution": contribution,
                        }
                    )
                if missing:
                    out[sample_id] = {
                        "sample_id": sample_id,
                        "scores": raw_scores,
                        "normalized_scores": normalized,
                        "contributions": contributions,
                        "anomaly_score": None,
                        "is_anomaly": False,
                        "failed": True,
                    }
                    continue
                composite = quantize_score(fsum(item["weighted_contribution"] for item in contributions) / total_weight)
                out[sample_id] = {
                    "sample_id": sample_id,
                    "scores": raw_scores,
                    "normalized_scores": normalized,
                    "contributions": contributions,
                    "anomaly_score": composite,
                    "is_anomaly": bool(composite >= threshold),
                    "failed": False,
                }
            return out

        return operation

    def _diagnostics(
        self,
        payload: dict[str, Any] | None,
        prepared: PreprocessedData | None,
        injection: InjectionController | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = quantize_score((time.perf_counter() - self.started_at) * 1000.0)
        distributions: dict[str, Any] = {}
        normalized_distributions: dict[str, Any] = {}
        composite_stats = _score_stats([])
        if payload is not None:
            for name in payload["algorithm_rankings"]:
                distributions[name] = _score_stats(item["score"] for item in payload["algorithm_rankings"][name])
            for name in payload.get("summary", {}).get("selected_algorithms", []):
                normalized_distributions[name] = _score_stats(
                    item.get("normalized_scores", {}).get(name)
                    for item in payload["samples"]
                    if item.get("normalized_scores", {}).get(name) is not None and not item.get("failed", False)
                )
            composite_stats = _score_stats(
                item["anomaly_score"] for item in payload["samples"] if item.get("anomaly_score") is not None
            )
        return {
            "status": "ok" if payload is not None else "incomplete",
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "elapsed_ms": elapsed_ms,
            "concurrency": self.concurrency,
            "force_serial": self.force_serial,
            "cache": self.stats.to_dict(),
            "preprocess_reused": self.preprocess_reused,
            "decision_reused": getattr(self, "decision_reused", False),
            "reused_algorithms": sorted(self.reused_algorithms, key=_algorithm_order),
            "recalculated_algorithms": sorted(self.recalculated_algorithms, key=_algorithm_order),
            "invalidated_artifact_count": self.stats.misses,
            "affected_algorithms": list(self.affected_algorithms),
            "snapshot_writes": list(self.snapshot_writes),
            "injection": injection.report() if injection is not None else {"items": [], "hits": 0, "total": 0},
            "distributions": {
                "raw": distributions,
                "normalized": normalized_distributions,
                "anomaly_score": composite_stats,
            },
        }


def canonical_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be exactly equal between incremental and full runs."""

    return {
        "algorithm_rankings": payload["algorithm_rankings"],
        "anomalies": payload["anomalies"],
        "failures": payload["failures"],
        "ranking": payload["ranking"],
        "samples": payload["samples"],
        "score_table": payload["score_table"],
        "summary": {
            "selected_algorithms": payload["summary"]["selected_algorithms"],
            "total_samples": payload["summary"]["total_samples"],
            "completed_samples": payload["summary"]["completed_samples"],
            "anomaly_count": payload["summary"]["anomaly_count"],
            "failed_sample_count": payload["summary"]["failed_sample_count"],
        },
    }


def run_full(
    dataset: Dataset,
    config: dict[str, Any],
    *,
    injection_items: list[dict[str, Any]] | None = None,
    affected_algorithms: list[str] | None = None,
) -> RunOutcome:
    pipeline = DetectionPipeline(
        dataset,
        config,
        cache=ContentCache(None, enabled=False),
        snapshots=None,
        injection_items=injection_items,
        affected_algorithms=affected_algorithms,
        force_serial=True,
    )
    return pipeline.run()
