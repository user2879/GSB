"""Pipeline orchestration, exact stage reuse and diagnostic assembly."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import RenderConfig
from .samples import SequenceSpec
from .snapshot import SnapshotStore
from .stable import digest_int_array, sha256_json, sort_failures
from .stages import (
    STAGE_ORDER,
    compute_prediction_error,
    decode_step,
    encode_frames,
    estimate_motion_pair,
    params_fingerprint,
    predict_from_motion,
)
from .strategy import ScaledRunInput, select_scale_input


@dataclass(frozen=True)
class RunRequest:
    config: RenderConfig
    sequences: list[SequenceSpec]
    failures: list[dict[str, Any]]
    scale_tier: int
    state_dir: str | Path
    output_dir: str | Path
    fresh: bool = False
    expected_version: int | None = None


@dataclass
class RunResult:
    predictions: dict[str, Any]
    decision_record: dict[str, Any]
    cache_ledger: dict[str, Any]
    cache_execution: dict[str, Any]
    diagnostic_report: dict[str, Any]
    failures: list[dict[str, Any]]
    output_files: dict[str, str]


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


class PipelineRunner:
    def __init__(self, request: RunRequest) -> None:
        self.request = request
        self.config = request.config
        self.tracker = _ConcurrencyTracker()
        self.stage_elapsed_ns: dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
        self.cache_counts: dict[str, dict[str, int]] = {
            stage: {"hit": 0, "recompute": 0} for stage in STAGE_ORDER
        }
        self.execution_events: list[dict[str, Any]] = []
        self.canonical_entries: list[dict[str, Any]] = []

    def run(self) -> RunResult:
        run_start_ns = time.monotonic_ns()
        scaled = select_scale_input(
            self.config, self.request.sequences, self.request.scale_tier
        )
        store = SnapshotStore(
            self.request.state_dir,
            config_hash=self.config.config_hash,
            scale_tier=self.request.scale_tier,
            expected_version=self.request.expected_version,
            fresh=self.request.fresh,
        )
        store.open()

        completed_sequences: list[str] = []
        results_by_sequence: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=self.config.thread_pool_workers,
            thread_name_prefix="motion-pipeline",
        ) as executor:
            for sequence in scaled.sequences:
                result, error_row = self._process_sequence(
                    sequence=sequence,
                    scaled=scaled,
                    store=store,
                    executor=executor,
                )
                results_by_sequence.append(result)
                if error_row is not None:
                    error_rows.append(error_row)
                completed_sequences.append(sequence.sequence_id)
                if (
                    len(completed_sequences)
                    % self.config.snapshot_commit_interval_batches
                    == 0
                ):
                    store.commit(completed_sequences)
            store.commit(completed_sequences)

        failures = sort_failures([dict(item) for item in self.request.failures])
        predictions = {
            "schema_version": 1,
            "scale_tier": scaled.tier,
            "selected_strategy": scaled.decision["selected"],
            "sequences": results_by_sequence,
            "failures": failures,
        }
        cache_ledger = {
            "schema_version": 1,
            "scale_tier": scaled.tier,
            "execution_event_log": "cache_execution.json",
            "entries": sorted(
                self.canonical_entries,
                key=lambda item: (
                    STAGE_ORDER.index(item["stage"]),
                    item["sequence_id"],
                    item["cache_key"],
                ),
            ),
        }
        execution_events = sorted(
            self.execution_events,
            key=lambda item: (
                STAGE_ORDER.index(item["stage"]),
                item["sequence_id"],
                item["cache_key"],
            ),
        )
        cache_execution = {
            "schema_version": 1,
            "scale_tier": scaled.tier,
            "events": execution_events,
            "hit_set": [
                {
                    "stage": event["stage"],
                    "sequence_id": event["sequence_id"],
                    "cache_key": event["cache_key"],
                    "reason": event["reason"],
                }
                for event in execution_events
                if event["action"] == "HIT"
            ],
            "recompute_set": [
                {
                    "stage": event["stage"],
                    "sequence_id": event["sequence_id"],
                    "cache_key": event["cache_key"],
                    "reason": event["reason"],
                }
                for event in execution_events
                if event["action"] == "RECOMPUTE"
            ],
        }
        run_end_ns = time.monotonic_ns()
        elapsed_ms = (run_end_ns - run_start_ns) / 1_000_000.0
        business_codes = sorted({item["business_code"] for item in failures})
        stale_codes = sorted(
            {item["business_code"] for item in store.stale_events if "business_code" in item}
        )
        decision_record = {
            "schema_version": 1,
            "scale_tier": scaled.tier,
            **scaled.decision,
        }
        diagnostic_report = {
            "schema_version": 1,
            "selected_strategy": scaled.decision["selected"],
            "quantization_level": scaled.selected.quantization_level,
            "decision_reason": scaled.decision["reason"],
            "scale_metrics": scaled.decision["metrics"],
            "stage_cache": self.cache_counts,
            "stage_cache_totals": {
                "hit": sum(value["hit"] for value in self.cache_counts.values()),
                "recompute": sum(
                    value["recompute"] for value in self.cache_counts.values()
                ),
            },
            "per_sequence_prediction_error": sorted(
                error_rows, key=lambda item: item["sequence_id"]
            ),
            "monotonic_clock": {
                "start_ns": run_start_ns,
                "end_ns": run_end_ns,
                "elapsed_ms": round(elapsed_ms, 6),
                "stage_elapsed_ms": {
                    stage: round(self.stage_elapsed_ns[stage] / 1_000_000.0, 6)
                    for stage in STAGE_ORDER
                },
            },
            "thread_pool": {
                "configured_workers": self.config.thread_pool_workers,
                "observed_max_concurrency": self.tracker.maximum,
            },
            "latency_budget_ms": self.config.latency_budget_ms,
            "latency_budget_met": elapsed_ms <= self.config.latency_budget_ms,
            "failure_business_codes": business_codes,
            "failures": failures,
            "snapshot": {
                "global_version": store.global_version,
                "config_hash": self.config.config_hash,
                "completed_sequences": store.completed_sequences,
                "stage_versions": store.stage_versions,
                "stale_events": store.stale_events,
                "stale_business_codes": stale_codes,
            },
        }
        output_files = self._write_outputs(
            predictions=predictions,
            decision_record=decision_record,
            cache_ledger=cache_ledger,
            cache_execution=cache_execution,
            diagnostic_report=diagnostic_report,
            failures=failures,
        )
        return RunResult(
            predictions=predictions,
            decision_record=decision_record,
            cache_ledger=cache_ledger,
            cache_execution=cache_execution,
            diagnostic_report=diagnostic_report,
            failures=failures,
            output_files=output_files,
        )

    def _process_sequence(
        self,
        *,
        sequence: SequenceSpec,
        scaled: ScaledRunInput,
        store: SnapshotStore,
        executor: ThreadPoolExecutor,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        strategy = scaled.selected
        history_frames = sequence.history_frames(
            self.config.history_frames, scaled.trajectory_limit
        )
        content_hash = sequence.history_content_hash(
            history_limit=self.config.history_frames,
            trajectory_limit=scaled.trajectory_limit,
        )
        encode_params = {
            "frame_width": self.config.frame_width,
            "frame_height": self.config.frame_height,
            "history_frames": len(history_frames),
            "fixed_point_bits": self.config.fixed_point_bits,
            "encoding_precision": strategy.encoding_precision,
        }
        encode_param_fp = params_fingerprint("encode", encode_params, None)
        encode_key = sha256_json(
            {
                "stage": "encode",
                "content_hash": content_hash,
                "parameters": encode_param_fp,
                "quantization_level": None,
                "upstream": {},
            }
        )
        encode_entry, encode_action = self._obtain(
            store=store,
            stage="encode",
            sequence_id=sequence.sequence_id,
            cache_key=encode_key,
            upstream_keys={},
            parameter_fingerprint=encode_param_fp,
            reason_if_miss=lambda: self._miss_reason(
                store, "encode", sequence.sequence_id, upstream_miss=False
            ),
            compute=lambda: encode_frames(
                history_frames,
                width=self.config.frame_width,
                height=self.config.frame_height,
                bits=strategy.encoding_precision,
                sequence_id=sequence.sequence_id,
            ),
        )

        motion_params = {
            "block_size": strategy.block_size,
            "search_radius": self.config.motion_model.search_radius,
            "fixed_point_bits": self.config.fixed_point_bits,
            "frame_width": self.config.frame_width,
            "frame_height": self.config.frame_height,
        }
        motion_param_fp = params_fingerprint("motion", motion_params, None)
        motion_key = sha256_json(
            {
                "stage": "motion",
                "parameters": motion_param_fp,
                "quantization_level": None,
                "upstream": {"encode": encode_key},
            }
        )

        def compute_motion() -> dict[str, Any]:
            pixels = [np.asarray(frame, dtype=np.int16) for frame in encode_entry["payload"]["frame_pixels"]]
            pair_tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
            for pair_index in range(max(0, len(pixels) - 1)):
                task_key = f"{sequence.sequence_id}|motion|{pair_index:08d}"
                pair_tasks.append(
                    (
                        task_key,
                        lambda pair_index=pair_index: estimate_motion_pair(
                            pixels[pair_index],
                            pixels[pair_index + 1],
                            bits=strategy.encoding_precision,
                            block_size=strategy.block_size,
                            search_radius=self.config.motion_model.search_radius,
                            pair_index=pair_index,
                            sequence_id=sequence.sequence_id,
                        ),
                    )
                )
            pairs = self._run_parallel_tasks(executor, pair_tasks)
            blocks_y = max(1, (self.config.frame_height + strategy.block_size - 1) // strategy.block_size)
            blocks_x = max(1, (self.config.frame_width + strategy.block_size - 1) // strategy.block_size)
            return {
                "sequence_id": sequence.sequence_id,
                "pairs": pairs,
                "blocks_y": blocks_y,
                "blocks_x": blocks_x,
            }

        motion_entry, motion_action = self._obtain(
            store=store,
            stage="motion",
            sequence_id=sequence.sequence_id,
            cache_key=motion_key,
            upstream_keys={"encode": encode_key},
            parameter_fingerprint=motion_param_fp,
            reason_if_miss=lambda: self._miss_reason(
                store,
                "motion",
                sequence.sequence_id,
                upstream_miss=encode_action == "RECOMPUTE",
            ),
            compute=compute_motion,
        )

        predict_params = {
            "prediction_path": strategy.prediction_path,
            "horizon": scaled.horizon,
            "linear_gain": self.config.motion_model.linear_gain,
            "damping": self.config.motion_model.damping,
            "quantization_level": strategy.quantization_level,
            "fixed_point_bits": self.config.fixed_point_bits,
            "block_size": strategy.block_size,
        }
        predict_param_fp = params_fingerprint(
            "predict", predict_params, strategy.quantization_level
        )
        predict_key = sha256_json(
            {
                "stage": "predict",
                "parameters": predict_param_fp,
                "quantization_level": strategy.quantization_level,
                "upstream": {"motion": motion_key},
            }
        )
        predict_entry, predict_action = self._obtain(
            store=store,
            stage="predict",
            sequence_id=sequence.sequence_id,
            cache_key=predict_key,
            upstream_keys={"motion": motion_key},
            parameter_fingerprint=predict_param_fp,
            reason_if_miss=lambda: self._miss_reason(
                store,
                "predict",
                sequence.sequence_id,
                upstream_miss=motion_action == "RECOMPUTE",
            ),
            compute=lambda: predict_from_motion(
                motion_entry["payload"],
                path=strategy.prediction_path,
                horizon=scaled.horizon,
                bits=strategy.encoding_precision,
                linear_gain=self.config.motion_model.linear_gain,
                damping=self.config.motion_model.damping,
                quantization_level=strategy.quantization_level,
            ),
        )

        decode_params = {
            "noise_amplitude": self.config.noise_layer.amplitude,
            "noise_salt": self.config.noise_layer.seed_salt,
            "quantization_level": strategy.quantization_level,
            "fixed_point_bits": self.config.fixed_point_bits,
            "block_size": strategy.block_size,
            "frame_width": self.config.frame_width,
            "frame_height": self.config.frame_height,
        }
        decode_param_fp = params_fingerprint(
            "decode", decode_params, strategy.quantization_level
        )
        decode_key = sha256_json(
            {
                "stage": "decode",
                "parameters": decode_param_fp,
                "quantization_level": strategy.quantization_level,
                "upstream": {"predict": predict_key},
            }
        )

        def compute_decode() -> dict[str, Any]:
            last_pixels = np.asarray(
                encode_entry["payload"]["frame_pixels"][-1], dtype=np.int16
            )
            noise_seed = int(
                encode_entry["payload"]["frame_metadata"][-1]["noise_seed"]
            )
            step_tasks: list[tuple[str, Callable[[], tuple[np.ndarray, dict[str, Any]]]]] = []
            for step_data in predict_entry["payload"]["steps"]:
                step_number = int(step_data["step"])
                vectors = np.asarray(step_data["vectors"], dtype=np.int64)
                task_key = f"{sequence.sequence_id}|decode|{step_number:08d}"
                step_tasks.append(
                    (
                        task_key,
                        lambda step_number=step_number, vectors=vectors: decode_step(
                            last_pixels,
                            vectors,
                            step=step_number,
                            block_size=strategy.block_size,
                            bits=strategy.encoding_precision,
                            sequence_id=sequence.sequence_id,
                            noise_seed=noise_seed,
                            noise_amplitude=self.config.noise_layer.amplitude,
                            noise_salt=self.config.noise_layer.seed_salt,
                        ),
                    )
                )
            decoded = self._run_parallel_tasks(executor, step_tasks)
            predictions = []
            for index, step_data in enumerate(predict_entry["payload"]["steps"]):
                pixels, decode_metadata = decoded[index]
                predictions.append(
                    {
                        "step": int(step_data["step"]),
                        "pixels": pixels.tolist(),
                        "motion_vectors": step_data["vectors"],
                        "motion_summary": step_data["motion_summary"],
                        "coefficients": step_data["coefficients"],
                        "decode_metadata": decode_metadata,
                    }
                )
            return {"sequence_id": sequence.sequence_id, "predictions": predictions}

        decode_entry, decode_action = self._obtain(
            store=store,
            stage="decode",
            sequence_id=sequence.sequence_id,
            cache_key=decode_key,
            upstream_keys={"predict": predict_key},
            parameter_fingerprint=decode_param_fp,
            reason_if_miss=lambda: self._miss_reason(
                store,
                "decode",
                sequence.sequence_id,
                upstream_miss=predict_action == "RECOMPUTE",
            ),
            compute=compute_decode,
        )

        self._record_canonical_entries(
            sequence_id=sequence.sequence_id,
            keys={
                "encode": encode_key,
                "motion": motion_key,
                "predict": predict_key,
                "decode": decode_key,
            },
            upstream={
                "encode": {},
                "motion": {"encode": encode_key},
                "predict": {"motion": motion_key},
                "decode": {"predict": predict_key},
            },
        )

        motion_payload = motion_entry["payload"]
        decode_payload = decode_entry["payload"]
        truth_frames = sequence.future_frames(
            len(history_frames), scaled.horizon, scaled.trajectory_limit
        )
        prediction_error = compute_prediction_error(
            decode_payload["predictions"],
            truth_frames,
            width=self.config.frame_width,
            height=self.config.frame_height,
            bits=strategy.encoding_precision,
        )
        result = {
            "sequence_id": sequence.sequence_id,
            "task_sort_key": f"{sequence.sequence_id}|sequence|00000000",
            "history_content_hash": content_hash,
            "history_frame_count": len(history_frames),
            "trajectory_limit": scaled.trajectory_limit,
            "encoded_frames": [
                {
                    "frame_index": frame_index,
                    "pixel_digest": digest_int_array(frame),
                    "pixels": frame,
                    "metadata": encode_entry["payload"]["frame_metadata"][frame_index],
                }
                for frame_index, frame in enumerate(
                    encode_entry["payload"]["frame_pixels"]
                )
            ],
            "encoding_metadata": encode_entry["payload"]["frame_metadata"],
            "motion_vectors": [
                {
                    "pair_index": pair["pair_index"],
                    "block_size": pair["block_size"],
                    "blocks_y": pair["blocks_y"],
                    "blocks_x": pair["blocks_x"],
                    "vectors": pair["vectors"],
                }
                for pair in motion_payload["pairs"]
            ],
            "motion_summaries": [
                pair["summary"] for pair in motion_payload["pairs"]
            ],
            "predictions": decode_payload["predictions"],
            "prediction_error": prediction_error,
        }
        error_row = {
            "sequence_id": sequence.sequence_id,
            "prediction_error": prediction_error,
        }
        return result, error_row

    def _obtain(
        self,
        *,
        store: SnapshotStore,
        stage: str,
        sequence_id: str,
        cache_key: str,
        upstream_keys: dict[str, str],
        parameter_fingerprint: str,
        reason_if_miss: Callable[[], str],
        compute: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        start_ns = time.monotonic_ns()
        cached = store.lookup(stage, cache_key)
        if cached is not None:
            self.cache_counts[stage]["hit"] += 1
            action = "HIT"
            reason = "CONTENT_KEY_MATCH"
            entry = cached
        else:
            self.cache_counts[stage]["recompute"] += 1
            action = "RECOMPUTE"
            reason = reason_if_miss()
            payload = compute()
            entry = {
                "stage": stage,
                "sequence_id": sequence_id,
                "sequence_ids": [sequence_id],
                "cache_key": cache_key,
                "upstream_keys": dict(upstream_keys),
                "parameter_fingerprint": parameter_fingerprint,
                "payload": payload,
            }
            store.put(stage, cache_key, entry)
        elapsed = time.monotonic_ns() - start_ns
        self.stage_elapsed_ns[stage] += elapsed
        self.execution_events.append(
            {
                "stage": stage,
                "sequence_id": sequence_id,
                "cache_key": cache_key,
                "upstream_keys": dict(upstream_keys),
                "parameter_fingerprint": parameter_fingerprint,
                "action": action,
                "reason": reason,
            }
        )
        return entry, action

    def _miss_reason(
        self,
        store: SnapshotStore,
        stage: str,
        sequence_id: str,
        *,
        upstream_miss: bool,
    ) -> str:
        if not store.previous_keys_for_sequence(stage, sequence_id):
            return "NEW_SEQUENCE_OR_STAGE"
        if upstream_miss:
            return "UPSTREAM_STAGE_RECOMPUTED"
        if stage == "encode":
            return "SEQUENCE_CONTENT_OR_ENCODING_PARAMETERS_CHANGED"
        if stage == "motion":
            return "MOTION_PARAMETERS_CHANGED"
        if stage == "predict":
            return "PREDICTION_PARAMETERS_OR_QUANTIZATION_CHANGED"
        return "DECODE_PARAMETERS_OR_QUANTIZATION_CHANGED"

    def _record_canonical_entries(
        self,
        *,
        sequence_id: str,
        keys: dict[str, str],
        upstream: dict[str, dict[str, str]],
    ) -> None:
        for stage in STAGE_ORDER:
            self.canonical_entries.append(
                {
                    "stage": stage,
                    "sequence_id": sequence_id,
                    "sequence_ids": [sequence_id],
                    "cache_key": keys[stage],
                    "upstream_keys": upstream[stage],
                    "final_status": "FINAL_CONTENT_KEY_READY",
                    "reuse_reason": "CONTENT_ADDRESSED_STAGE_PRODUCT_AVAILABLE",
                }
            )

    def _run_parallel_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: list[tuple[str, Callable[[], Any]]],
    ) -> list[Any]:
        if not tasks:
            return []

        def wrapped(task: Callable[[], Any]) -> Any:
            self.tracker.enter()
            try:
                return task()
            finally:
                self.tracker.exit()

        futures = [executor.submit(wrapped, task) for _, task in tasks]
        return [future.result() for future in futures]

    def _write_outputs(
        self,
        *,
        predictions: dict[str, Any],
        decision_record: dict[str, Any],
        cache_ledger: dict[str, Any],
        cache_execution: dict[str, Any],
        diagnostic_report: dict[str, Any],
        failures: list[dict[str, Any]],
    ) -> dict[str, str]:
        output_dir = Path(self.request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        documents = {
            "predictions.json": predictions,
            "decision_record.json": decision_record,
            "cache_ledger.json": cache_ledger,
            "cache_execution.json": cache_execution,
            "diagnostic_report.json": diagnostic_report,
            "failures.json": failures,
        }
        paths: dict[str, str] = {}
        for name, document in documents.items():
            path = output_dir / name
            tmp_path = output_dir / f".{name}.tmp-{os.getpid()}"
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    document,
                    handle,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            paths[name] = str(path)
        return paths
