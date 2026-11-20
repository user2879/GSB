from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .algorithms import (
    AlgorithmContext,
    RawScore,
    algorithm_dependency_report,
    build_algorithm_contexts,
    raw_score_cache_key,
    score_segment,
)
from .config import ConfigIssue, parse_config
from .data import DeviceData, LoadedData, Segment, load_data
from .errors import (
    E_BAND_INVALID,
    E_CONFIG_INVALID,
    E_FEATURE_DEPENDENCY,
    E_NONFINITE_SAMPLE,
    Failure,
    PipelineError,
    failure_from_error,
)
from .features import (
    FeatureBundle,
    band_feature_key,
    compose_feature_bundle,
    extract_band_energies,
    extract_time_features,
    time_feature_key,
)
from .injections import InjectionTracker, injection_hash, parse_injections
from .quantization import (
    Decision,
    QuantizedScore,
    fused_score,
    make_decision,
    quantize_score,
    quantized_score_cache_key,
)
from .snapshot import SnapshotState, SnapshotStore
from .util import ConcurrencyProbe, atomic_write_json, hash_object, stable_key


ENGINE_VERSION = 1


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def request(self, signal_number: int | None = None) -> None:
        self.requested = True
        if signal_number is not None:
            self.signal_number = int(signal_number)


@dataclass(frozen=True)
class Action:
    sort_key: str
    device_id: str
    segment_index: int
    layer: str
    action: str
    cache_key: str
    basis: str
    algorithm: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sort_key": self.sort_key,
            "device_id": self.device_id,
            "segment_index": self.segment_index,
            "layer": self.layer,
            "algorithm": self.algorithm,
            "action": self.action,
            "cache_key": self.cache_key,
            "basis": self.basis,
        }


@dataclass
class FeatureOutcome:
    segment: Segment
    bundle: FeatureBundle | None = None
    error: PipelineError | None = None
    failure: Failure | None = None
    actions: list[Action] = field(default_factory=list)
    time_record: dict[str, Any] | None = None
    band_record: dict[str, Any] | None = None
    band_edges: tuple[float, ...] = ()


@dataclass
class ScoreOutcome:
    segment: Segment
    algorithm: str
    result: RawScore | None = None
    action: Action | None = None
    failure: Failure | None = None


@dataclass
class QuantizedOutcome:
    segment: Segment
    algorithm: str
    result: QuantizedScore | None = None
    action: Action | None = None
    failure: Failure | None = None


@dataclass
class DecisionOutcome:
    segment: Segment
    result: Decision | None = None
    action: Action | None = None
    failure: Failure | None = None


def _short(value: str) -> str:
    return value[:12]


def _issue_failure(issue: ConfigIssue) -> Failure:
    return Failure(None, None, issue.field_name, "load", issue.code, issue.detail, None)


class PipelineRun:
    def __init__(
        self,
        config_spec: dict[str, Any],
        data_spec: dict[str, Any],
        injection_spec: dict[str, Any] | None,
        *,
        snapshot_dir: str | Path,
        mode: str = "incremental",
        edits_spec: dict[str, Any] | None = None,
        stop_controller: StopController | None = None,
    ) -> None:
        if mode not in ("incremental", "full"):
            raise ValueError("mode must be incremental or full")
        self.config = parse_config(config_spec)
        self.data_spec = data_spec
        self.injection_spec = injection_spec or {}
        self.snapshot_dir = Path(snapshot_dir)
        self.mode = mode
        self.edits_spec = edits_spec
        self.stop = stop_controller or StopController()
        self.probe = ConcurrencyProbe()
        self.failures: list[Failure] = [_issue_failure(issue) for issue in self.config.issues]
        self.tracker = InjectionTracker([])
        self.injection_errors: list[str] = []
        self.stage_timings: dict[str, float] = {}
        self.stage_artifacts: dict[str, dict[str, Any]] = {}
        self.feature_outcomes: dict[tuple[str, int], FeatureOutcome] = {}
        self.score_outcomes: dict[tuple[str, int, str], ScoreOutcome] = {}
        self.quantized_outcomes: dict[tuple[str, int, str], QuantizedOutcome] = {}
        self.decision_outcomes: dict[tuple[str, int], DecisionOutcome] = {}
        self.actions: list[Action] = []
        self.contexts: dict[str, dict[str, AlgorithmContext]] = {}
        self.loaded: LoadedData | None = None
        self.store: SnapshotStore | None = None
        self.snapshot: SnapshotState | None = None
        self.previous_cache: dict[str, Any] = {}
        self.current_cache: dict[str, Any] = {}
        self.previous_index: dict[str, Any] = {}
        self.current_index: dict[str, Any] = {"segments": {}, "algorithm_params": {}, "global": {}}
        self.cache: dict[str, Any] = {}
        self.interrupted = False
        self.effective_threshold = self.config.threshold
        self.threshold_injection_ids: set[str] = set()
        self.work_segments: list[Segment] = []
        self.device_map: dict[str, DeviceData] = {}
        self.processed_cursors: dict[str, int] = {}
        self.completed_devices: set[str] = set()

    def _prepare(self) -> None:
        injections, injection_errors = parse_injections(self.injection_spec)
        self.tracker = InjectionTracker(injections)
        self.injection_errors = injection_errors
        for detail in injection_errors:
            self.failures.append(Failure(None, None, None, "load", E_CONFIG_INVALID, detail, None))
        self.loaded = load_data(self.data_spec, self.config, edits_spec=self.edits_spec)
        self.failures.extend(self.loaded.failures)
        self.device_map = self.loaded.devices
        self.work_segments = [
            segment
            for device_id in sorted(self.device_map)
            for segment in sorted(self.device_map[device_id].segments, key=lambda item: item.segment_index)
        ]
        self.store = SnapshotStore(
            self.snapshot_dir,
            schema_version=self.config.snapshot_schema_version,
            config_hash=self.config.config_hash,
            waveform_fingerprint=self.loaded.waveform_fingerprint,
            segment_length=self.config.segment_length,
            quantization_fingerprint=self.config.quantization.fingerprint,
        )
        self.snapshot = self.store.open()
        self.previous_cache = copy.deepcopy(self.snapshot.cache)
        self.previous_index = copy.deepcopy(self.snapshot.previous_index)
        if self.mode == "full":
            self.current_cache = {
                "schema_version": 1,
                "feature_subcache": {},
                "algorithm_raw_cache": {},
                "quantized_score_cache": {},
                "decision_cache": {},
                "provenance_index": {"segments": {}, "algorithm_params": {}, "global": {}},
            }
            self.previous_index = {"segments": {}, "algorithm_params": {}, "global": {}}
        else:
            self.current_cache = copy.deepcopy(self.previous_cache)
            if not isinstance(self.current_cache.get("feature_subcache"), dict):
                self.current_cache["feature_subcache"] = {}
            if not isinstance(self.current_cache.get("algorithm_raw_cache"), dict):
                self.current_cache["algorithm_raw_cache"] = {}
            if not isinstance(self.current_cache.get("quantized_score_cache"), dict):
                self.current_cache["quantized_score_cache"] = {}
            if not isinstance(self.current_cache.get("decision_cache"), dict):
                self.current_cache["decision_cache"] = {}
        self.current_cache["schema_version"] = 1
        self.current_index = {"segments": {}, "algorithm_params": {}, "global": {
            "band_edges_hz": list(self.config.features.band_edges_hz),
            "quantization_fingerprint": self.config.quantization.fingerprint,
        }}
        self.current_index["algorithm_params"] = {
            name: self.config.algorithms[name].params_hash for name in sorted(self.config.algorithms)
        }

    def _segment_key(self, segment: Segment) -> str:
        return f"{segment.device_id}|{segment.segment_index}"

    def _make_action(
        self,
        segment: Segment,
        *,
        layer: str,
        action: str,
        cache_key: str,
        basis: str,
        algorithm: str | None = None,
        stage: str,
    ) -> Action:
        value = Action(
            sort_key=stable_key(segment.device_id, segment.segment_index, algorithm or "_segment", stage),
            device_id=segment.device_id,
            segment_index=segment.segment_index,
            layer=layer,
            action=action,
            cache_key=cache_key,
            basis=basis,
            algorithm=algorithm,
        )
        self.actions.append(value)
        return value

    def _wrap(self, function: Callable[[], Any]) -> Any:
        with self.probe.track():
            return function()

    def _run_parallel(
        self,
        stage_name: str,
        tasks: list[dict[str, Any]],
        worker: Callable[[dict[str, Any]], Any],
        checkpoint: Callable[[list[Any], bool], None],
    ) -> list[Any]:
        started = time.perf_counter()
        results: list[Any] = []
        if not tasks:
            self.stage_timings[stage_name] = self.stage_timings.get(stage_name, 0.0) + 0.0
            checkpoint([], True)
            return results
        workers = max(1, min(self.config.concurrency, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"vib-{stage_name}") as executor:
            futures = [executor.submit(self._wrap, lambda task=task: worker(task)) for task in tasks]
            chunk_start = 0
            for index, future in enumerate(futures, 1):
                try:
                    result = future.result()
                except Exception as exc:  # Defensive: workers normally translate errors into outcomes.
                    task = tasks[index - 1]
                    segment = task["segment"]
                    result = task.get("fallback")(exc) if "fallback" in task else {
                        "failure": failure_from_error(
                            exc,
                            device_id=segment.device_id,
                            segment_index=segment.segment_index,
                            algorithm=task.get("algorithm"),
                            stage=stage_name,
                        )
                    }
                results.append(result)
                interval = self.config.commit_interval
                if index % interval == 0 or index == len(futures):
                    checkpoint(results[chunk_start:], index == len(futures))
                    chunk_start = len(results)
        elapsed = time.perf_counter() - started
        self.stage_timings[stage_name] = self.stage_timings.get(stage_name, 0.0) + elapsed
        return results

    def _feature_basis(self, segment: Segment, kind: str, previous: dict[str, Any] | None) -> str:
        if self.mode == "full":
            return "full_rebuild_mode"
        if previous is None:
            return "new_segment_cache_miss"
        if previous.get("waveform_fingerprint") != segment.waveform_fingerprint:
            return "waveform_changed"
        if kind == "band" and previous.get("band_edges_hz") != list(
            self.device_map[segment.device_id].band_edges_hz
        ):
            return "band_configuration_changed"
        if kind == "time" and previous.get("preprocess_fingerprint") != segment.preprocess_fingerprint:
            return "preprocessing_changed"
        return "cache_key_miss"

    def _feature_worker(self, task: dict[str, Any]) -> FeatureOutcome:
        segment: Segment = task["segment"]
        device = self.device_map[segment.device_id]
        mode = self.config.features.mode
        need_time = mode in ("time_domain", "combined")
        need_band = mode in ("band_energy", "combined")
        band_edges = device.band_edges_hz
        samples = segment.samples
        feature_injections = self.tracker.matching(
            stage="feature_extraction", device_id=segment.device_id, segment_index=segment.segment_index
        )
        injected_nonfinite = False
        injected_band = False
        for injection in feature_injections:
            if injection.kind == "non_finite_sample" and not injected_nonfinite:
                mutable = np.asarray(samples, dtype=np.float64).copy()
                mutable[0] = np.nan
                samples = mutable
                injected_nonfinite = True
                self.tracker.mark(injection, effect="sample[0] replaced with NaN", code=E_NONFINITE_SAMPLE)
            elif injection.kind == "band_edge_out_of_range" and not injected_band:
                band_edges = tuple(list(band_edges) + [device.sample_rate_hz / 2.0 + 1.0])
                injected_band = True
                self.tracker.mark(injection, effect="last band boundary moved beyond Nyquist", code=E_BAND_INVALID)

        previous = self.previous_index.get("segments", {}).get(self._segment_key(segment))
        time_key = time_feature_key(segment.waveform_fingerprint, segment.preprocess_fingerprint, segment.segment_length)
        band_key = band_feature_key(
            segment.waveform_fingerprint,
            segment.preprocess_fingerprint,
            segment.segment_length,
            segment.sample_rate_hz,
            band_edges,
        )
        time_values: tuple[float, ...] = ()
        band_values: tuple[float, ...] = ()
        actions: list[Action] = []
        time_record: dict[str, Any] | None = None
        band_record: dict[str, Any] | None = None
        cache_usable = self.mode == "incremental" and not feature_injections
        try:
            if need_time:
                cached = self.current_cache["feature_subcache"].get(time_key) if cache_usable else None
                if cached is not None:
                    time_values = tuple(stable_float_value for stable_float_value in cached.get("values", []))
                    action = self._make_action(
                        segment,
                        layer="feature.time",
                        action="reused",
                        cache_key=time_key,
                        basis=f"cache_key_hit:{_short(time_key)}",
                        stage="feature_extraction",
                    )
                else:
                    time_values = extract_time_features(samples)
                    time_record = {"kind": "time", "cache_key": time_key, "values": list(time_values)}
                    self.current_cache["feature_subcache"][time_key] = time_record
                    action = self._make_action(
                        segment,
                        layer="feature.time",
                        action="recomputed",
                        cache_key=time_key,
                        basis=self._feature_basis(segment, "time", previous),
                        stage="feature_extraction",
                    )
                actions.append(action)
            if need_band:
                cached = self.current_cache["feature_subcache"].get(band_key) if cache_usable else None
                if cached is not None:
                    band_values = tuple(stable_float_value for stable_float_value in cached.get("values", []))
                    action = self._make_action(
                        segment,
                        layer="feature.band",
                        action="reused",
                        cache_key=band_key,
                        basis=f"cache_key_hit:{_short(band_key)}",
                        stage="feature_extraction",
                    )
                else:
                    band_values = extract_band_energies(samples, segment.sample_rate_hz, band_edges)
                    band_record = {"kind": "band", "cache_key": band_key, "values": list(band_values)}
                    self.current_cache["feature_subcache"][band_key] = band_record
                    action = self._make_action(
                        segment,
                        layer="feature.band",
                        action="recomputed",
                        cache_key=band_key,
                        basis=self._feature_basis(segment, "band", previous),
                        stage="feature_extraction",
                    )
                actions.append(action)
            bundle = compose_feature_bundle(
                time_values=time_values,
                band_values=band_values,
                time_key_value=time_key,
                band_key_value=band_key,
                segment_length=segment.segment_length,
                band_edges_hz=band_edges,
                waveform_fingerprint=segment.waveform_fingerprint,
                preprocessing_fingerprint=segment.preprocess_fingerprint,
                mode=mode,
            )
            return FeatureOutcome(
                segment=segment,
                bundle=bundle,
                actions=actions,
                time_record=time_record,
                band_record=band_record,
                band_edges=band_edges,
            )
        except Exception as error:
            failure = failure_from_error(
                error,
                device_id=segment.device_id,
                segment_index=segment.segment_index,
                algorithm=None,
                stage="feature_extraction",
            )
            if feature_injections and failure.injected_id is None:
                failure = Failure(
                    failure.device_id,
                    failure.segment_index,
                    failure.algorithm,
                    failure.stage,
                    failure.code,
                    failure.detail,
                    feature_injections[0].injection_id,
                )
            return FeatureOutcome(
                segment=segment,
                error=error if isinstance(error, PipelineError) else PipelineError(failure.code, failure.detail),
                failure=failure,
                actions=actions,
                time_record=time_record,
                band_record=band_record,
                band_edges=band_edges,
            )

    def _feature_record(self, outcome: FeatureOutcome) -> dict[str, Any]:
        segment = outcome.segment
        record: dict[str, Any] = {
            "device_id": segment.device_id,
            "segment_index": segment.segment_index,
            "start_sample": segment.start_sample,
            "end_sample": segment.end_sample,
            "status": "ok" if outcome.bundle is not None else "failed",
            "actions": [action.as_dict() for action in outcome.actions],
        }
        if outcome.bundle is not None:
            record["content_fingerprint"] = outcome.bundle.content_fingerprint
            record["feature_names"] = list(outcome.bundle.feature_names)
            record["feature_values"] = list(outcome.bundle.vector)
            record["time_key"] = outcome.bundle.time_key
            record["band_key"] = outcome.bundle.band_key
        if outcome.failure is not None:
            record["failure"] = outcome.failure.as_dict()
        return record

    def _commit_feature_checkpoint(self, results: list[FeatureOutcome], final: bool) -> None:
        for outcome in results:
            self.feature_outcomes[(outcome.segment.device_id, outcome.segment.segment_index)] = outcome
            if outcome.failure is not None:
                self.failures.append(outcome.failure)
            self.processed_cursors[outcome.segment.device_id] = max(
                self.processed_cursors.get(outcome.segment.device_id, -1), outcome.segment.segment_index
            )
        payload = {
            "stage": "feature_extraction",
            "final": final,
            "segments": [
                self._feature_record(self.feature_outcomes[key])
                for key in sorted(self.feature_outcomes)
            ],
        }
        artifact = self.store.commit_stage(  # type: ignore[union-attr]
            "feature_extraction",
            payload,
            metadata={"cursor": dict(sorted(self.processed_cursors.items())), "final": final},
        )
        self.stage_artifacts["feature_extraction"] = artifact
        self.store.save_cache(self.current_cache)  # type: ignore[union-attr]
        self._write_manifest("running", "feature_extraction")

    def _run_feature_stage(self) -> None:
        tasks = [
            {
                "segment": segment,
                "sort_key": stable_key(segment.device_id, segment.segment_index, "_segment", "feature_extraction"),
            }
            for segment in self.work_segments
        ]
        tasks.sort(key=lambda task: task["sort_key"])
        self._run_parallel("feature_extraction", tasks, self._feature_worker, self._commit_feature_checkpoint)
        if not tasks:
            self._commit_feature_checkpoint([], True)

    def _build_contexts(self) -> None:
        for device_id in sorted(self.device_map):
            bundles = {
                segment.segment_index: self.feature_outcomes[(device_id, segment.segment_index)].bundle
                for segment in self.device_map[device_id].segments
                if self.feature_outcomes[(device_id, segment.segment_index)].bundle is not None
            }
            self.contexts[device_id] = build_algorithm_contexts(self.device_map[device_id], bundles, self.config)  # type: ignore[arg-type]

    def _score_basis(self, segment: Segment, algorithm: str, previous: dict[str, Any] | None) -> str:
        if self.mode == "full":
            return "full_rebuild_mode"
        if previous is None:
            return "new_segment_cache_miss"
        if previous.get("waveform_fingerprint") != segment.waveform_fingerprint:
            return "waveform_changed"
        previous_params = self.previous_index.get("algorithm_params", {}).get(algorithm)
        if previous_params != self.config.algorithms[algorithm].params_hash:
            return "algorithm_parameter_changed"
        previous_raw = previous.get("raw_keys", {}).get(algorithm)
        if previous_raw is not None:
            return "upstream_feature_or_context_dependency_changed"
        return "cache_key_miss"

    def _score_worker(self, task: dict[str, Any]) -> ScoreOutcome:
        segment: Segment = task["segment"]
        algorithm: str = task["algorithm"]
        feature_outcome = self.feature_outcomes[(segment.device_id, segment.segment_index)]
        if feature_outcome.bundle is None:
            return ScoreOutcome(segment=segment, algorithm=algorithm)
        context = self.contexts.get(segment.device_id, {}).get(algorithm)
        if context is None:
            failure = Failure(
                segment.device_id,
                segment.segment_index,
                algorithm,
                "scoring",
                E_FEATURE_DEPENDENCY,
                "algorithm context could not be constructed",
                None,
            )
            return ScoreOutcome(segment=segment, algorithm=algorithm, failure=failure)
        cache_key = raw_score_cache_key(segment=segment, bundle=feature_outcome.bundle, context=context)
        injections = self.tracker.matching(
            stage="scoring", device_id=segment.device_id, segment_index=segment.segment_index, algorithm=algorithm
        )
        previous = self.previous_index.get("segments", {}).get(self._segment_key(segment))
        cache_usable = self.mode == "incremental" and not injections
        cached = self.current_cache["algorithm_raw_cache"].get(cache_key) if cache_usable else None
        if cached is not None:
            details = dict(cached.get("details", {}))
            context_fingerprint = context.segment_context_fingerprints.get(
                segment.segment_index, context.baseline_fingerprint
            )
            details["baseline_fingerprint"] = context_fingerprint
            details["segment_context_fingerprint"] = context_fingerprint
            result = RawScore(
                algorithm=cached["algorithm"],
                raw_score=float(cached["raw_score"]),
                cache_key=cached["cache_key"],
                params_hash=cached["params_hash"],
                referenced_features=tuple(cached.get("referenced_features", [])),
                referenced_segments=tuple(
                    (item["device_id"], int(item["segment_index"])) for item in cached.get("referenced_segments", [])
                ),
                details=details,
            )
            action = self._make_action(
                segment,
                layer="algorithm.raw",
                algorithm=algorithm,
                action="reused",
                cache_key=cache_key,
                basis=f"cache_key_hit:{_short(cache_key)}",
                stage="scoring",
            )
            return ScoreOutcome(segment=segment, algorithm=algorithm, result=result, action=action)
        injected_window: int | None = None
        for injection in injections:
            if injection.kind == "peak_factor_window_zero":
                injected_window = 0
                self.tracker.mark(injection, effect="crest_factor window_length set to 0", code="E_ALGO_PARAM_INVALID")
        try:
            result = score_segment(
                segment=segment,
                bundle=feature_outcome.bundle,
                context=context,
                injected_window_length=injected_window,
            )
            self.current_cache["algorithm_raw_cache"][cache_key] = {
                "algorithm": result.algorithm,
                "raw_score": result.raw_score,
                "cache_key": result.cache_key,
                "params_hash": result.params_hash,
                "referenced_features": list(result.referenced_features),
                "referenced_segments": [
                    {"device_id": device_id, "segment_index": index}
                    for device_id, index in result.referenced_segments
                ],
                "details": result.details,
            }
            action = self._make_action(
                segment,
                layer="algorithm.raw",
                algorithm=algorithm,
                action="recomputed",
                cache_key=cache_key,
                basis=self._score_basis(segment, algorithm, previous),
                stage="scoring",
            )
            return ScoreOutcome(segment=segment, algorithm=algorithm, result=result, action=action)
        except Exception as error:
            failure = failure_from_error(
                error,
                device_id=segment.device_id,
                segment_index=segment.segment_index,
                algorithm=algorithm,
                stage="scoring",
            )
            if injections and failure.injected_id is None:
                failure = Failure(
                    failure.device_id,
                    failure.segment_index,
                    failure.algorithm,
                    failure.stage,
                    failure.code,
                    failure.detail,
                    injections[0].injection_id,
                )
            return ScoreOutcome(segment=segment, algorithm=algorithm, failure=failure)

    def _commit_score_checkpoint(self, results: list[ScoreOutcome], final: bool) -> None:
        for outcome in results:
            key = (outcome.segment.device_id, outcome.segment.segment_index, outcome.algorithm)
            self.score_outcomes[key] = outcome
            if outcome.failure is not None:
                self.failures.append(outcome.failure)
        payload = {
            "stage": "scoring",
            "final": final,
            "scores": [
                {
                    "device_id": outcome.segment.device_id,
                    "segment_index": outcome.segment.segment_index,
                    "algorithm": outcome.algorithm,
                    "status": "ok" if outcome.result is not None else "failed",
                    "raw_score": outcome.result.raw_score if outcome.result is not None else None,
                    "cache_key": outcome.result.cache_key if outcome.result is not None else None,
                    "params_hash": outcome.result.params_hash if outcome.result is not None else None,
                    "failure": outcome.failure.as_dict() if outcome.failure is not None else None,
                    "action": outcome.action.as_dict() if outcome.action is not None else None,
                }
                for outcome in sorted(
                    self.score_outcomes.values(),
                    key=lambda item: (item.segment.device_id, item.segment.segment_index, item.algorithm),
                )
            ],
        }
        artifact = self.store.commit_stage(  # type: ignore[union-attr]
            "scoring",
            payload,
            metadata={"final": final, "cursor": dict(sorted(self.processed_cursors.items()))},
        )
        self.stage_artifacts["scoring"] = artifact
        self.store.save_cache(self.current_cache)  # type: ignore[union-attr]
        self._write_manifest("running", "scoring")

    def _run_scoring_stage(self) -> None:
        tasks: list[dict[str, Any]] = []
        for segment in self.work_segments:
            outcome = self.feature_outcomes[(segment.device_id, segment.segment_index)]
            if outcome.bundle is None:
                continue
            for algorithm in sorted(self.config.enabled_algorithms):
                tasks.append(
                    {
                        "segment": segment,
                        "algorithm": algorithm,
                        "sort_key": stable_key(segment.device_id, segment.segment_index, algorithm, "scoring"),
                    }
                )
        tasks.sort(key=lambda task: task["sort_key"])
        self._run_parallel("scoring", tasks, self._score_worker, self._commit_score_checkpoint)
        if not tasks:
            self._commit_score_checkpoint([], True)

    def _quantized_basis(self, segment: Segment, algorithm: str, previous: dict[str, Any] | None) -> str:
        if self.mode == "full":
            return "full_rebuild_mode"
        if previous is None:
            return "new_segment_cache_miss"
        if previous.get("waveform_fingerprint") != segment.waveform_fingerprint:
            return "waveform_changed"
        if previous.get("quantization_fingerprint") != self.config.quantization.fingerprint:
            return "quantization_configuration_changed"
        if previous.get("normalization_hashes", {}).get(algorithm) != self.config.algorithms[algorithm].normalization_hash:
            return "normalization_configuration_changed"
        if previous.get("raw_keys", {}).get(algorithm) != self.raw_key_for(segment, algorithm):
            return "upstream_raw_score_changed"
        return "cache_key_miss"

    def raw_key_for(self, segment: Segment, algorithm: str) -> str | None:
        outcome = self.score_outcomes.get((segment.device_id, segment.segment_index, algorithm))
        if outcome is not None and outcome.result is not None:
            return outcome.result.cache_key
        return None

    def _quantize_worker(self, task: dict[str, Any]) -> QuantizedOutcome:
        segment: Segment = task["segment"]
        algorithm: str = task["algorithm"]
        raw_outcome = self.score_outcomes.get((segment.device_id, segment.segment_index, algorithm))
        if raw_outcome is None or raw_outcome.result is None:
            return QuantizedOutcome(segment=segment, algorithm=algorithm)
        raw = raw_outcome.result
        spec = self.config.algorithms[algorithm]
        cache_key = quantized_score_cache_key(raw, spec.normalization_hash, self.config.quantization)
        previous = self.previous_index.get("segments", {}).get(self._segment_key(segment))
        cached = self.current_cache["quantized_score_cache"].get(cache_key) if self.mode == "incremental" else None
        if cached is not None:
            result = QuantizedScore(
                algorithm=cached["algorithm"],
                raw_score=float(cached["raw_score"]),
                normalized_score=float(cached["normalized_score"]),
                normalized_ticks=int(cached["normalized_ticks"]),
                cache_key=cached["cache_key"],
                raw_cache_key=cached["raw_cache_key"],
                normalization_hash=cached["normalization_hash"],
                quantization_fingerprint=cached["quantization_fingerprint"],
                details=cached.get("details", {}),
            )
            action = self._make_action(
                segment,
                layer="algorithm.quantized",
                algorithm=algorithm,
                action="reused",
                cache_key=cache_key,
                basis=f"cache_key_hit:{_short(cache_key)}",
                stage="threshold",
            )
            return QuantizedOutcome(segment=segment, algorithm=algorithm, result=result, action=action)
        try:
            result = quantize_score(raw, self.config)
            self.current_cache["quantized_score_cache"][cache_key] = {
                "algorithm": result.algorithm,
                "raw_score": result.raw_score,
                "normalized_score": result.normalized_score,
                "normalized_ticks": result.normalized_ticks,
                "cache_key": result.cache_key,
                "raw_cache_key": result.raw_cache_key,
                "normalization_hash": result.normalization_hash,
                "quantization_fingerprint": result.quantization_fingerprint,
                "details": result.details,
            }
            action = self._make_action(
                segment,
                layer="algorithm.quantized",
                algorithm=algorithm,
                action="recomputed",
                cache_key=cache_key,
                basis=self._quantized_basis(segment, algorithm, previous),
                stage="threshold",
            )
            return QuantizedOutcome(segment=segment, algorithm=algorithm, result=result, action=action)
        except Exception as error:
            failure = failure_from_error(
                error,
                device_id=segment.device_id,
                segment_index=segment.segment_index,
                algorithm=algorithm,
                stage="threshold",
            )
            return QuantizedOutcome(segment=segment, algorithm=algorithm, failure=failure)

    def _decision_basis(self, segment: Segment, previous: dict[str, Any] | None) -> str:
        if self.mode == "full":
            return "full_rebuild_mode"
        if previous is None:
            return "new_segment_cache_miss"
        if previous.get("waveform_fingerprint") != segment.waveform_fingerprint:
            return "waveform_changed"
        if previous.get("threshold_ticks") != self.threshold_ticks():
            return "threshold_changed"
        if previous.get("quantization_fingerprint") != self.config.quantization.fingerprint:
            return "quantization_configuration_changed"
        if previous.get("quant_keys") != self.quant_keys_for(segment):
            return "upstream_quantized_score_changed"
        return "cache_key_miss"

    def _decision_worker(self, task: dict[str, Any]) -> DecisionOutcome:
        segment: Segment = task["segment"]
        quantized = [
            outcome.result
            for (device_id, segment_index, _algorithm), outcome in sorted(self.quantized_outcomes.items())
            if device_id == segment.device_id and segment_index == segment.segment_index and outcome.result is not None
        ]
        previous = self.previous_index.get("segments", {}).get(self._segment_key(segment))
        if not quantized:
            failure = Failure(
                segment.device_id,
                segment.segment_index,
                None,
                "threshold",
                E_FEATURE_DEPENDENCY,
                "no algorithm score is available for fusion",
                None,
            )
            return DecisionOutcome(segment=segment, failure=failure)
        probe = make_decision(
            device_id=segment.device_id,
            segment_index=segment.segment_index,
            waveform_fingerprint=segment.waveform_fingerprint,
            preprocessing_fingerprint=segment.preprocess_fingerprint,
            quantized_scores=quantized,
            config=self.config,
            effective_threshold=self.effective_threshold,
        )
        cached = self.current_cache["decision_cache"].get(probe.cache_key) if self.mode == "incremental" else None
        if cached is not None:
            result = Decision(
                status=cached["status"],
                score=float(cached["score"]),
                score_ticks=int(cached["score_ticks"]),
                threshold=float(cached["threshold"]),
                threshold_ticks=int(cached["threshold_ticks"]),
                normalized_total=float(cached["normalized_total"]),
                anomaly=bool(cached["anomaly"]),
                boundary=bool(cached["boundary"]),
                boundary_reason=cached.get("boundary_reason"),
                cache_key=cached["cache_key"],
                algorithms_used=tuple(cached.get("algorithms_used", [])),
                reference=cached.get("reference", {}),
            )
            action = self._make_action(
                segment,
                layer="decision",
                action="reused",
                cache_key=probe.cache_key,
                basis=f"cache_key_hit:{_short(probe.cache_key)}",
                stage="threshold",
            )
            return DecisionOutcome(segment=segment, result=result, action=action)
        self.current_cache["decision_cache"][probe.cache_key] = {
            "status": probe.status,
            "score": probe.score,
            "score_ticks": probe.score_ticks,
            "threshold": probe.threshold,
            "threshold_ticks": probe.threshold_ticks,
            "normalized_total": probe.normalized_total,
            "anomaly": probe.anomaly,
            "boundary": probe.boundary,
            "boundary_reason": probe.boundary_reason,
            "cache_key": probe.cache_key,
            "algorithms_used": list(probe.algorithms_used),
            "reference": probe.reference,
        }
        action = self._make_action(
            segment,
            layer="decision",
            action="recomputed",
            cache_key=probe.cache_key,
            basis=self._decision_basis(segment, previous),
            stage="threshold",
        )
        return DecisionOutcome(segment=segment, result=probe, action=action)

    def threshold_ticks(self) -> int:
        from .util import quantize_half_away

        return quantize_half_away(
            min(self.config.score_max, max(self.config.score_min, self.effective_threshold)),
            self.config.quantization.scale,
            self.config.quantization.quantum,
        )

    def quant_keys_for(self, segment: Segment) -> dict[str, str]:
        result: dict[str, str] = {}
        for (device_id, segment_index, algorithm), outcome in sorted(self.quantized_outcomes.items()):
            if device_id == segment.device_id and segment_index == segment.segment_index and outcome.result is not None:
                result[algorithm] = outcome.result.cache_key
        return result

    def _commit_quantized_checkpoint(self, results: list[QuantizedOutcome], final: bool) -> None:
        for outcome in results:
            key = (outcome.segment.device_id, outcome.segment.segment_index, outcome.algorithm)
            self.quantized_outcomes[key] = outcome
            if outcome.failure is not None:
                self.failures.append(outcome.failure)
        payload = {
            "stage": "threshold.quantized_scores",
            "final": final,
            "scores": [
                {
                    "device_id": outcome.segment.device_id,
                    "segment_index": outcome.segment.segment_index,
                    "algorithm": outcome.algorithm,
                    "status": "ok" if outcome.result is not None else "failed",
                    "normalized_score": outcome.result.normalized_score if outcome.result is not None else None,
                    "normalized_ticks": outcome.result.normalized_ticks if outcome.result is not None else None,
                    "cache_key": outcome.result.cache_key if outcome.result is not None else None,
                    "failure": outcome.failure.as_dict() if outcome.failure is not None else None,
                    "action": outcome.action.as_dict() if outcome.action is not None else None,
                }
                for outcome in sorted(
                    self.quantized_outcomes.values(),
                    key=lambda item: (item.segment.device_id, item.segment.segment_index, item.algorithm),
                )
            ],
        }
        artifact = self.store.commit_stage(  # type: ignore[union-attr]
            "threshold",
            payload,
            metadata={"final": final, "cursor": dict(sorted(self.processed_cursors.items()))},
        )
        self.stage_artifacts["threshold"] = artifact
        self.store.save_cache(self.current_cache)  # type: ignore[union-attr]
        self._write_manifest("running", "threshold")

    def _apply_threshold_injections(self) -> None:
        threshold_injections = [
            injection
            for injection in self.tracker.injections.values()
            if injection.kind == "threshold_equal_boundary"
        ]
        for injection in sorted(threshold_injections, key=lambda item: item.injection_id):
            segment = next(
                (
                    item
                    for item in self.work_segments
                    if item.device_id == injection.device_id and item.segment_index == injection.segment_index
                ),
                None,
            )
            if segment is None:
                continue
            scores = [
                result
                for (device_id, segment_index, _algorithm), outcome in sorted(self.quantized_outcomes.items())
                if device_id == segment.device_id and segment_index == segment.segment_index and outcome.result is not None
                for result in [outcome.result]
            ]
            if not scores:
                continue
            try:
                score, _ticks, _normalized, _reference = fused_score(scores, self.config)
            except PipelineError:
                continue
            self.effective_threshold = score
            self.threshold_injection_ids.add(injection.injection_id)
            self.tracker.mark(
                injection,
                effect=f"effective threshold set to target fused score {score}",
                code=None,
            )

    def _commit_decision_checkpoint(self, results: list[DecisionOutcome], final: bool) -> None:
        for outcome in results:
            key = (outcome.segment.device_id, outcome.segment.segment_index)
            self.decision_outcomes[key] = outcome
            if outcome.failure is not None:
                self.failures.append(outcome.failure)
        payload = {
            "stage": "threshold.decisions",
            "final": final,
            "effective_threshold": self.effective_threshold,
            "threshold_ticks": self.threshold_ticks(),
            "decisions": [
                {
                    "device_id": outcome.segment.device_id,
                    "segment_index": outcome.segment.segment_index,
                    "status": "ok" if outcome.result is not None else "failed",
                    "score": outcome.result.score if outcome.result is not None else None,
                    "score_ticks": outcome.result.score_ticks if outcome.result is not None else None,
                    "anomaly": outcome.result.anomaly if outcome.result is not None else None,
                    "boundary": outcome.result.boundary if outcome.result is not None else None,
                    "cache_key": outcome.result.cache_key if outcome.result is not None else None,
                    "failure": outcome.failure.as_dict() if outcome.failure is not None else None,
                    "action": outcome.action.as_dict() if outcome.action is not None else None,
                }
                for outcome in sorted(
                    self.decision_outcomes.values(), key=lambda item: (item.segment.device_id, item.segment.segment_index)
                )
            ],
        }
        artifact = self.store.commit_stage(  # type: ignore[union-attr]
            "threshold",
            payload,
            metadata={"final": final, "cursor": dict(sorted(self.processed_cursors.items()))},
        )
        self.stage_artifacts["threshold"] = artifact
        self.store.save_cache(self.current_cache)  # type: ignore[union-attr]
        self._write_manifest("running", "threshold")

    def _run_threshold_stage(self) -> None:
        quant_tasks: list[dict[str, Any]] = []
        for segment in self.work_segments:
            for algorithm in sorted(self.config.enabled_algorithms):
                raw = self.score_outcomes.get((segment.device_id, segment.segment_index, algorithm))
                if raw is None or raw.result is None:
                    continue
                quant_tasks.append(
                    {
                        "segment": segment,
                        "algorithm": algorithm,
                        "sort_key": stable_key(segment.device_id, segment.segment_index, algorithm, "threshold.quantized"),
                    }
                )
        quant_tasks.sort(key=lambda task: task["sort_key"])
        self._run_parallel(
            "threshold", quant_tasks, self._quantize_worker, self._commit_quantized_checkpoint
        )
        if not quant_tasks:
            self._commit_quantized_checkpoint([], True)
        self._apply_threshold_injections()
        decision_tasks = [
            {
                "segment": segment,
                "sort_key": stable_key(segment.device_id, segment.segment_index, "_fusion", "threshold.decision"),
            }
            for segment in self.work_segments
        ]
        decision_tasks.sort(key=lambda task: task["sort_key"])
        decisions = self._run_parallel(
            "threshold", decision_tasks, self._decision_worker, self._commit_decision_checkpoint
        )
        if not decision_tasks:
            self._commit_decision_checkpoint([], True)
        _ = decisions

    def _write_manifest(self, status: str, current_stage: str) -> None:
        if self.snapshot is None or self.loaded is None:
            return
        for device_id, device in self.device_map.items():
            if device.segments and all(
                (device_id, segment.segment_index) in self.feature_outcomes for segment in device.segments
            ):
                self.completed_devices.add(device_id)
        manifest = {
            "snapshot_version": self.config.snapshot_schema_version,
            "generation": self.snapshot.generation,
            "engine_version": ENGINE_VERSION,
            "run_mode": self.mode,
            "status": status,
            "current_stage": current_stage,
            "config_hash": self.config.config_hash,
            "waveform_fingerprint": self.loaded.waveform_fingerprint,
            "injection_hash": injection_hash(self.injection_spec),
            "segment_length": self.config.segment_length,
            "quantization_fingerprint": self.config.quantization.fingerprint,
            "commit_interval_segments": self.config.commit_interval,
            "completed_devices": sorted(self.completed_devices),
            "segment_cursors": dict(sorted(self.processed_cursors.items())),
            "stages": {name: dict(value) for name, value in sorted(self.stage_artifacts.items())},
        }
        self.store.write_manifest(manifest)

    def _injection_ids_for_segment(self, segment: Segment) -> list[str]:
        return sorted(
            injection.injection_id
            for injection in self.tracker.injections.values()
            if injection.device_id == segment.device_id and injection.segment_index == segment.segment_index
        )

    def _segment_report(self, segment: Segment) -> dict[str, Any]:
        feature = self.feature_outcomes.get((segment.device_id, segment.segment_index))
        score_outcomes = sorted(
            (
                outcome
                for (device_id, segment_index, _algorithm), outcome in self.score_outcomes.items()
                if device_id == segment.device_id and segment_index == segment.segment_index
            ),
            key=lambda item: item.algorithm,
        )
        quant_outcomes = sorted(
            (
                outcome
                for (device_id, segment_index, _algorithm), outcome in self.quantized_outcomes.items()
                if device_id == segment.device_id and segment_index == segment.segment_index
            ),
            key=lambda item: item.algorithm,
        )
        decision = self.decision_outcomes.get((segment.device_id, segment.segment_index))
        report: dict[str, Any] = {
            "sort_key": stable_key(segment.device_id, segment.segment_index, "_segment", "summary"),
            "device_id": segment.device_id,
            "segment_index": segment.segment_index,
            "start_sample": segment.start_sample,
            "end_sample": segment.end_sample,
            "source": segment.source,
            "anomaly_generation_hint": segment.anomaly_hint,
            "missing_sample_count": len(segment.missing_indices),
            "resampled_sample_count": len(segment.resampled_indices),
            "waveform_fingerprint": segment.waveform_fingerprint,
            "preprocessing_fingerprint": segment.preprocess_fingerprint,
        }
        if feature is not None and feature.bundle is not None:
            report["feature"] = {
                "status": "ok",
                "content_fingerprint": feature.bundle.content_fingerprint,
                "time_cache_key": feature.bundle.time_key,
                "band_cache_key": feature.bundle.band_key,
                "summary": feature.bundle.summary(mode=self.config.features.mode),
            }
        else:
            report["feature"] = {
                "status": "failed",
                "error": feature.failure.as_dict() if feature is not None and feature.failure is not None else None,
            }
        algorithm_scores: list[dict[str, Any]] = []
        quant_map = {outcome.algorithm: outcome for outcome in quant_outcomes}
        for outcome in score_outcomes:
            quantized = quant_map.get(outcome.algorithm)
            entry: dict[str, Any] = {
                "algorithm": outcome.algorithm,
                "status": "ok" if outcome.result is not None else "failed",
            }
            if outcome.result is not None:
                entry.update(
                    {
                        "raw_score": outcome.result.raw_score,
                        "raw_cache_key": outcome.result.cache_key,
                        "params_hash": outcome.result.params_hash,
                        "referenced_features": list(outcome.result.referenced_features),
                        "referenced_segments": [
                            {"device_id": device_id, "segment_index": index}
                            for device_id, index in outcome.result.referenced_segments
                        ],
                        "detail": outcome.result.details,
                    }
                )
            else:
                entry["error"] = outcome.failure.as_dict() if outcome.failure is not None else None
            if quantized is not None and quantized.result is not None:
                entry.update(
                    {
                        "normalized_score": quantized.result.normalized_score,
                        "normalized_ticks": quantized.result.normalized_ticks,
                        "quantized_cache_key": quantized.result.cache_key,
                        "normalization_hash": quantized.result.normalization_hash,
                    }
                )
            elif quantized is not None and quantized.failure is not None:
                entry["quantization_error"] = quantized.failure.as_dict()
            algorithm_scores.append(entry)
        report["algorithm_scores"] = algorithm_scores
        report["segment_total_score"] = None
        report["segment_anomaly"] = None
        report["decision"] = None
        if decision is not None and decision.result is not None:
            result = decision.result
            report["segment_total_score"] = result.score
            report["segment_total_score_ticks"] = result.score_ticks
            report["segment_anomaly"] = result.anomaly
            report["decision"] = {
                "status": result.status,
                "score": result.score,
                "score_ticks": result.score_ticks,
                "threshold": result.threshold,
                "threshold_ticks": result.threshold_ticks,
                "normalized_total": result.normalized_total,
                "anomaly": result.anomaly,
                "boundary": result.boundary,
                "boundary_reason": result.boundary_reason,
                "cache_key": result.cache_key,
                "algorithms_used": list(result.algorithms_used),
                "reference": result.reference,
            }
        elif decision is not None and decision.failure is not None:
            report["decision"] = {"status": "failed", "error": decision.failure.as_dict()}
        report["injection_ids"] = self._injection_ids_for_segment(segment)
        return report

    def _boundary_report(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        boundaries: list[dict[str, Any]] = []
        for segment in segments:
            decision = segment.get("decision")
            if not decision or not decision.get("boundary"):
                continue
            boundaries.append(
                {
                    "sort_key": stable_key(segment["device_id"], segment["segment_index"], "_fusion", "boundary"),
                    "device_id": segment["device_id"],
                    "segment_index": segment["segment_index"],
                    "score": decision["score"],
                    "score_ticks": decision["score_ticks"],
                    "threshold": decision["threshold"],
                    "threshold_ticks": decision["threshold_ticks"],
                    "anomaly": decision["anomaly"],
                    "reference": decision["reference"],
                    "reason": decision["boundary_reason"],
                    "injection_ids": [
                        injection_id
                        for injection_id in self.threshold_injection_ids
                        if injection_id in self._injection_ids_for_segment(
                            next(
                                item
                                for item in self.work_segments
                                if item.device_id == segment["device_id"]
                                and item.segment_index == segment["segment_index"]
                            )
                        )
                    ],
                }
            )
        return sorted(boundaries, key=lambda item: item["sort_key"])

    def _finalize_cache_index(self) -> None:
        for segment in self.work_segments:
            key = self._segment_key(segment)
            feature = self.feature_outcomes.get((segment.device_id, segment.segment_index))
            raw_keys: dict[str, str] = {}
            quant_keys: dict[str, str] = {}
            for (device_id, segment_index, algorithm), outcome in sorted(self.score_outcomes.items()):
                if device_id == segment.device_id and segment_index == segment.segment_index and outcome.result is not None:
                    raw_keys[algorithm] = outcome.result.cache_key
            quant_keys = self.quant_keys_for(segment)
            decision = self.decision_outcomes.get((segment.device_id, segment.segment_index))
            current = {
                "waveform_fingerprint": segment.waveform_fingerprint,
                "preprocess_fingerprint": segment.preprocess_fingerprint,
                "band_edges_hz": list(self.device_map[segment.device_id].band_edges_hz),
                "time_key": feature.bundle.time_key if feature is not None and feature.bundle is not None else None,
                "band_key": feature.bundle.band_key if feature is not None and feature.bundle is not None else None,
                "raw_keys": raw_keys,
                "quant_keys": quant_keys,
                "normalization_hashes": {
                    name: self.config.algorithms[name].normalization_hash for name in sorted(self.config.algorithms)
                },
                "quantization_fingerprint": self.config.quantization.fingerprint,
                "threshold_ticks": self.threshold_ticks(),
                "decision_key": decision.result.cache_key if decision is not None and decision.result is not None else None,
            }
            self.current_index["segments"][key] = current
        self.current_cache["provenance_index"] = self.current_index

    def _business_projection(self, report: dict[str, Any]) -> dict[str, Any]:
        segments = []
        for segment in report.get("segments", []):
            segments.append(
                {
                    "device_id": segment["device_id"],
                    "segment_index": segment["segment_index"],
                    "start_sample": segment["start_sample"],
                    "end_sample": segment["end_sample"],
                    "source": segment["source"],
                    "waveform_fingerprint": segment["waveform_fingerprint"],
                    "preprocessing_fingerprint": segment["preprocessing_fingerprint"],
                    "feature": segment["feature"],
                    "algorithm_scores": segment["algorithm_scores"],
                    "segment_total_score": segment["segment_total_score"],
                    "segment_total_score_ticks": segment.get("segment_total_score_ticks"),
                    "segment_anomaly": segment["segment_anomaly"],
                    "decision": segment["decision"],
                }
            )
        return {
            "waveform_fingerprint": report["waveform_fingerprint"],
            "segments": segments,
            "boundary_segments": report["boundary_segments"],
            "failures": report["failures"],
            "injection_hits": report["injection_hits"],
        }

    def _build_report(self) -> dict[str, Any]:
        assert self.loaded is not None
        segment_reports = [self._segment_report(segment) for segment in self.work_segments]
        boundaries = self._boundary_report(segment_reports)
        failure_map = {
            (
                failure.device_id,
                failure.segment_index,
                failure.algorithm,
                failure.stage,
                failure.code,
                failure.detail,
                failure.injected_id,
            ): failure
            for failure in self.failures
        }
        failures = sorted(failure_map.values(), key=Failure.sort_key)
        action_dicts = sorted(
            (action.as_dict() for action in self.actions),
            key=lambda item: (
                item["sort_key"],
                item["layer"],
                item["algorithm"] or "",
                item["action"],
            ),
        )
        reuse = [item for item in action_dicts if item["action"] == "reused"]
        recompute = [item for item in action_dicts if item["action"] == "recomputed"]
        algorithm_dependencies = {
            device_id: algorithm_dependency_report(self.contexts.get(device_id, {}), device_id)
            for device_id in sorted(self.contexts)
        }
        manifest = None
        if self.store is not None and self.store.manifest_path.exists():
            try:
                from .util import read_json

                manifest = read_json(self.store.manifest_path)
            except Exception:
                manifest = None
        report: dict[str, Any] = {
            "schema_version": 1,
            "engine_version": ENGINE_VERSION,
            "status": "interrupted" if self.interrupted or self.stop.requested else "completed",
            "config_hash": self.config.config_hash,
            "injection_hash": injection_hash(self.injection_spec),
            "synthetic_description_hash": self.loaded.synthetic_description_hash,
            "waveform_fingerprint": self.loaded.waveform_fingerprint,
            "snapshot_version": self.config.snapshot_schema_version,
            "snapshot": {
                "dir": str(self.snapshot_dir),
                "generation": self.snapshot.generation if self.snapshot is not None else None,
                "stale": self.snapshot.stale if self.snapshot is not None else None,
                "stale_code": self.snapshot.stale_code if self.snapshot is not None else None,
                "stale_reasons": self.snapshot.stale_reasons if self.snapshot is not None else [],
                "manifest": manifest,
            },
            "run": {
                "mode": self.mode,
                "configured_concurrency": self.config.concurrency,
                "max_observed_concurrency": self.probe.max_observed,
                "snapshot_commit_interval_segments": self.config.commit_interval,
                "exit_code": 0,
                "stop_signal": self.stop.signal_number,
            },
            "config_issues": [issue.as_dict() for issue in self.config.issues],
            "injection_config_errors": list(self.injection_errors),
            "algorithms": {
                name: self.config.algorithms[name].definition() for name in sorted(self.config.algorithms)
            },
            "algorithm_dependencies": algorithm_dependencies,
            "segment_count": len(segment_reports),
            "segments": segment_reports,
            "boundary_segments": boundaries,
            "injection_hits": self.tracker.report(),
            "failures": [failure.as_dict() for failure in failures],
            "failure_count": len(failures),
            "reuse": reuse,
            "recompute": recompute,
            "reuse_count": len(reuse),
            "recompute_count": len(recompute),
            "stage_timings_seconds": {name: round(value, 6) for name, value in self.stage_timings.items()},
        }
        report["result_fingerprint"] = hash_object(self._business_projection(report), namespace="business-result-v1")
        return report

    def _commit_summary(self, report: dict[str, Any]) -> None:
        if self.snapshot is None:
            return
        payload = {
            "stage": "summary",
            "status": report["status"],
            "result_fingerprint": report["result_fingerprint"],
            "failure_count": report["failure_count"],
            "segment_count": report["segment_count"],
        }
        artifact = self.store.commit_stage(
            "summary",
            payload,
            metadata={"result_fingerprint": report["result_fingerprint"]},
        )
        self.stage_artifacts["summary"] = artifact
        self._write_manifest(report["status"], "summary")

    def run(self) -> dict[str, Any]:
        self._prepare()
        if not self.stop.requested:
            self._run_feature_stage()
        if self.stop.requested:
            self.interrupted = True
        if not self.interrupted:
            self._build_contexts()
            self._run_scoring_stage()
        if self.stop.requested:
            self.interrupted = True
        if not self.interrupted:
            self._run_threshold_stage()
        if self.stop.requested:
            self.interrupted = True
        self._finalize_cache_index()
        if self.snapshot is not None:
            self.store.save_cache(self.current_cache)
        summary_started = time.perf_counter()
        report = self._build_report()
        self._commit_summary(report)
        report["stage_timings_seconds"]["summary"] = round(time.perf_counter() - summary_started, 6)
        # Refresh summary-stage artifact metadata into the report.
        if self.store is not None and self.store.manifest_path.exists():
            from .util import read_json

            report["snapshot"]["manifest"] = read_json(self.store.manifest_path)
        return report


def run_pipeline(
    config_spec: dict[str, Any],
    data_spec: dict[str, Any],
    injection_spec: dict[str, Any] | None = None,
    *,
    snapshot_dir: str | Path,
    mode: str = "incremental",
    edits_spec: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    stop_controller: StopController | None = None,
) -> dict[str, Any]:
    run = PipelineRun(
        config_spec,
        data_spec,
        injection_spec,
        snapshot_dir=snapshot_dir,
        mode=mode,
        edits_spec=edits_spec,
        stop_controller=stop_controller,
    )
    report = run.run()
    if report_path is not None:
        atomic_write_json(Path(report_path), report)
    return report
