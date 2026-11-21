"""Four-stage evaluation pipeline, cache ledger, diagnostics and reporting."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .audio import AudioData, frame_ranges, load_audio
from .cache import ArtifactCache
from .config import DIMENSIONS, Config
from .conversion import convert_one
from .diagnostics import (
    consistency_report,
    degradation_statistics,
    deviation_report,
    injection_hits,
)
from .errors import (
    E_AUDIO_FORMAT,
    E_AUDIO_MISSING,
    E_EVAL_FORMAT,
    E_FRAME_FEATURE,
    E_SCORE,
    E_SERIALIZE,
    E_SNAPSHOT_STALE,
    Failure,
    InterruptedRun,
    LevelMapError,
    sorted_failures,
)
from .formats import (
    ParseOutcome,
    parse_bytes,
    read_exchange,
    write_format_bytes,
)
from .levels import LevelMap
from .models import UnifiedResult, sorted_degradations
from .scoring import (
    MICRO_SCALE,
    combine_fixed_total,
    compute_frame_features,
    frame_set_digest,
    frame_summaries_from_features,
    grade_total_micro,
    score_dimensions,
    summarize_frames,
)
from .snapshot import SnapshotStore
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    micro_to_float,
    pretty_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)


EXCHANGE_PATHS = {
    1: "eval/v1/results_v1.txt",
    2: "eval/v2/results_v2.json",
    3: "eval/v3/results_v3.bin",
}
OUTPUT_EXTENSIONS = {1: "txt", 2: "json", 3: "bin"}
STAGE_INDEX = {"features": 0, "dimensions": 1, "total": 2, "serialization": 3}
SERIALIZATION_CODEC = "exchange-codec-v1"


def _outcome_to_payload(outcome: ParseOutcome) -> dict[str, Any]:
    return {
        "version": outcome.version,
        "records": [record.to_dict() for record in outcome.records],
        "failures": [failure.to_dict() for failure in sorted_failures(outcome.failures)],
    }


def _outcome_from_payload(payload: dict[str, Any]) -> ParseOutcome:
    return ParseOutcome(
        version=int(payload["version"]),
        records=[UnifiedResult.from_dict(item) for item in payload.get("records", [])],
        failures=[Failure.from_dict(item) for item in payload.get("failures", [])],
    )


def _hash_or_missing(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError:
        return "missing"


def _safe_task(task_key: str, function: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"task_key": task_key, "ok": True, "value": function()}
    except Exception as exc:  # noqa: BLE001 - converted to a stable failure record
        return {
            "task_key": task_key,
            "ok": False,
            "error": str(exc),
            "exception_type": type(exc).__name__,
        }


def _reason_for_miss(
    artifact: str,
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> str:
    if not previous:
        return "cold_cache"
    if artifact == "features":
        if previous.get("audio_fingerprint") != current.get("audio_fingerprint"):
            return "audio_changed"
        if previous.get("frame_fingerprint") != current.get("frame_fingerprint"):
            return "frame_config_changed"
        return "feature_key_changed"
    if artifact == "dimensions":
        if previous.get("threshold_fingerprint") != current.get("threshold_fingerprint"):
            if previous.get("quantization_levels") != current.get("quantization_levels"):
                return "quantization_changed"
            return "threshold_changed"
        return "upstream_features_changed"
    if artifact == "total":
        if previous.get("weights_fingerprint") != current.get("weights_fingerprint"):
            return "weights_changed"
        return "upstream_dimensions_changed"
    if artifact == "serialization":
        if previous.get("target_version") != current.get("target_version"):
            return "version_switch"
        if previous.get("fixed_decimal_bits") != current.get("fixed_decimal_bits"):
            return "fixed_width_changed"
        if previous.get("level_entry_fingerprint") != current.get("level_entry_fingerprint"):
            return "level_mapping_changed"
        return "upstream_total_changed"
    return "cache_key_changed"


class EvaluationPipeline:
    def __init__(
        self,
        workspace: str | Path,
        *,
        rebuild_cache: bool = False,
        rebuild_snapshot: bool = False,
        stop_after_stage: str | None = None,
        serial_mode: bool = False,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.rebuild_cache = rebuild_cache
        self.rebuild_snapshot = rebuild_snapshot
        self.stop_after_stage = stop_after_stage
        self.serial_mode = serial_mode
        if self.stop_after_stage not in (None, "parse", "features", "scores", "conversion"):
            raise ValueError("stop_after_stage must be parse, features, scores or conversion")
        if rebuild_snapshot:
            import shutil

            snapshots = self.workspace / "snapshots"
            if snapshots.exists():
                shutil.rmtree(snapshots)

    def _load_state(self) -> dict[str, Any]:
        path = self.workspace / "state.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("state is not an object")
            return raw
        except (OSError, ValueError):
            return {
                "schema": "aqeval.state.v1",
                "active_version": 2,
                "config_revision": 0,
                "global_run": 0,
                "last_edits": [],
            }

    def _interrupt_if_requested(self, stage: str, cache: ArtifactCache) -> None:
        if self.stop_after_stage == stage:
            cache.save()
            raise InterruptedRun(
                f"run intentionally interrupted after committed stage {stage}",
                code="E_RUN_INTERRUPTED",
            )

    def run(self) -> dict[str, Any]:
        run_started = time.perf_counter()
        self.workspace.mkdir(parents=True, exist_ok=True)
        config = Config.load(self.workspace / "config.json")
        if self.serial_mode:
            config = config.with_updates(max_concurrency=1, concurrency_cap=1)
        level_map = LevelMap.load(self.workspace / "level_map.json")
        state = self._load_state()
        active_version = int(state.get("active_version", 2))
        if active_version not in (1, 2, 3):
            raise ValueError("state.active_version must be 1, 2 or 3")
        config_revision = int(state.get("config_revision", 0))
        global_run = int(state.get("global_run", 0))
        cache = ArtifactCache(self.workspace / "cache", clear=self.rebuild_cache)
        cache_fingerprints = {
            "config_hash": config.config_hash,
            "level_map_hash": level_map.digest,
            "frame_fingerprint": config.frame_fingerprint,
            "threshold_fingerprint": config.threshold_fingerprint,
            "weights_fingerprint": config.weights_fingerprint,
            "quantization_levels": list(config.quantization_levels),
        }
        cache.set_run_fingerprints(cache_fingerprints)
        snapshot = SnapshotStore(
            self.workspace,
            job_version=config_revision,
            config_hash=config.config_hash,
            level_map_hash=level_map.digest,
            active_version=active_version,
            global_run=global_run,
        )
        failures: list[Failure] = []
        exchange_hashes = {
            str(version): _hash_or_missing(self.workspace / relative)
            for version, relative in EXCHANGE_PATHS.items()
        }
        audio_paths = sorted((self.workspace / "samples").glob("*.wav"), key=lambda path: path.name)
        audio_hashes = {
            path.stem: _hash_or_missing(path) for path in audio_paths if path.is_file()
        }
        content_fingerprint = sha256_json(
            {"exchange": exchange_hashes, "audio": audio_hashes}
        )
        snapshot.set_content_fingerprint(content_fingerprint)
        parse_input = sha256_json(
            {"exchange_hashes": exchange_hashes, "level_map": level_map.digest, "parser": "exchange-parser-v1"}
        )
        feature_input = sha256_json(
            {"audio": audio_hashes, "frame_fingerprint": config.frame_fingerprint}
        )
        score_input = sha256_json(
            {
                "features": feature_input,
                "threshold_fingerprint": config.threshold_fingerprint,
                "quantization_levels": config.quantization_levels,
                "weights_fingerprint": config.weights_fingerprint,
                "scorer_input_version": "score-stage-v1",
            }
        )
        conversion_input = sha256_json(
            {
                "scores": score_input,
                "level_map": level_map.digest,
                "active_version": active_version,
                "fixed_decimal_bits": config.fixed_decimal_bits,
                "codec": SERIALIZATION_CODEC,
            }
        )
        timings: dict[str, dict[str, Any]] = {}
        decisions: dict[str, dict[str, str]] = {
            name: {} for name in ("features", "dimensions", "total", "serialization")
        }

        # Stage 1: parse and validate exchange inputs.
        stage_started = time.perf_counter()
        parse_resumed = snapshot.can_resume("parse", parse_input)
        if parse_resumed:
            parse_payload = snapshot.load_stage("parse")
            outcomes = {
                int(version): _outcome_from_payload(payload)
                for version, payload in parse_payload["by_version"].items()
            }
        else:
            outcomes = {}
            for version, relative in EXCHANGE_PATHS.items():
                outcomes[version] = read_exchange(version, self.workspace / relative, level_map)
            parse_payload = {
                "exchange_hashes": exchange_hashes,
                "by_version": {
                    str(version): _outcome_to_payload(outcome)
                    for version, outcome in sorted(outcomes.items())
                },
            }
            snapshot.commit_stage("parse", parse_payload, parse_input)
        parse_failures = [
            failure for outcome in outcomes.values() for failure in outcome.failures
        ]
        failures.extend(parse_failures)
        parse_artifact = snapshot.stage_entry("parse") or {}
        timings["parse"] = {
            "elapsed_ms": round((time.perf_counter() - stage_started) * 1000.0, 3),
            "resumed": parse_resumed,
            "artifact_version": parse_artifact.get("artifact_version"),
        }
        self._interrupt_if_requested("parse", cache)

        audio_data: dict[str, AudioData] = {}
        for path in audio_paths:
            file_id = path.stem
            try:
                audio_data[file_id] = load_audio(path, file_id, config)
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", E_AUDIO_FORMAT)
                failures.append(
                    Failure(
                        code=code,
                        message=str(exc),
                        file_id=file_id,
                        stage="audio",
                        location=str(path),
                        task_key=f"{file_id}|audio|read",
                    )
                )
        parsed_file_ids = {
            record.file_id for outcome in outcomes.values() for record in outcome.records
        }
        for file_id in sorted(parsed_file_ids - set(audio_data)):
            if not any(failure.file_id == file_id and failure.stage == "audio" for failure in failures):
                failures.append(
                    Failure(
                        code=E_AUDIO_MISSING,
                        message="exchange result has no corresponding audio sample",
                        file_id=file_id,
                        stage="audio",
                        location=str(self.workspace / "samples" / f"{file_id}.wav"),
                        task_key=f"{file_id}|audio|missing",
                    )
                )

        # Stage 2: frame features, computed as independently ordered frame tasks.
        stage_started = time.perf_counter()
        features_resumed = snapshot.can_resume("features", feature_input)
        batch_failures: list[Failure] = []
        feature_serial_verified = True
        feature_verification_mode = "not_run"
        if features_resumed:
            feature_payload = snapshot.load_stage("features")
            feature_files = {
                str(file_id): dict(value)
                for file_id, value in feature_payload.get("files", {}).items()
            }
            batch_failures.extend(
                Failure.from_dict(item) for item in feature_payload.get("failures", [])
            )
            feature_serial_verified = bool(
                feature_payload.get("serial_verification", {}).get("matches", True)
            )
            feature_verification_mode = "snapshot_resume"
            for file_id, info in feature_files.items():
                decisions["features"][file_id] = "reuse"
                info["_reason"] = "snapshot_stage_resume"
                if cache.get("features", str(info["feature_key"])) is None:
                    cache.put("features", str(info["feature_key"]), info)
        else:
            feature_files = {}
            pending_frames: dict[str, list[tuple[int, int, int]]] = {}
            frame_jobs: dict[Any, tuple[str, int]] = {}
            computed_any = False
            for file_id in sorted(audio_data):
                audio = audio_data[file_id]
                feature_key = sha256_json(
                    {
                        "schema": "features-v1",
                        "audio_fingerprint": audio.fingerprint,
                        "frame_fingerprint": config.frame_fingerprint,
                    }
                )
                cached = cache.get("features", feature_key)
                if cached is not None:
                    feature_files[file_id] = cached
                    decisions["features"][file_id] = "reuse"
                    continue
                pending_frames[file_id] = []
                for frame_index, (start, end) in enumerate(
                    frame_ranges(
                        audio.sample_count, config.frame_length_samples, config.hop_samples
                    )
                ):
                    pending_frames[file_id].append((frame_index, start, end))
            if pending_frames:
                with ThreadPoolExecutor(
                    max_workers=config.worker_count,
                    thread_name_prefix="aqeval-features",
                ) as executor:
                    for file_id in sorted(pending_frames):
                        audio = audio_data[file_id]
                        for frame_index, start, end in pending_frames[file_id]:
                            task_key = f"{file_id}|frame|{frame_index:06d}"
                            frame_samples = audio.samples[start:end].copy()
                            future = executor.submit(
                                _safe_task,
                                task_key,
                                lambda fs=frame_samples, fi=frame_index, fid=file_id: compute_frame_features(
                                    fid,
                                    fi,
                                    fs,
                                    config.sample_rate,
                                    config.clipping_amplitude_threshold,
                                ),
                            )
                            frame_jobs[future] = (file_id, frame_index)
                    frame_results: dict[str, list[dict[str, Any]]] = {
                        file_id: [] for file_id in pending_frames
                    }
                    for future in as_completed(frame_jobs):
                        file_id, frame_index = frame_jobs[future]
                        result = future.result()
                        if not result["ok"]:
                            batch_failures.append(
                                Failure(
                                    code=E_FRAME_FEATURE,
                                    message=f"frame feature calculation failed: {result['error']}",
                                    file_id=file_id,
                                    stage="features",
                                    location=audio_data[file_id].path,
                                    task_key=result["task_key"],
                                    frame_index=frame_index,
                                )
                            )
                            continue
                        frame_results[file_id].append(result["value"])
                        computed_any = True
                for file_id in sorted(pending_frames):
                    frames = sorted(
                        frame_results.get(file_id, []), key=lambda item: item["frame_index"]
                    )
                    if not frames:
                        continue
                    summary = summarize_frames(file_id, frames, config)
                    audio = audio_data[file_id]
                    feature_key = sha256_json(
                        {
                            "schema": "features-v1",
                            "audio_fingerprint": audio.fingerprint,
                            "frame_fingerprint": config.frame_fingerprint,
                        }
                    )
                    info = {
                        "file_id": file_id,
                        "path": audio.path,
                        "audio_fingerprint": audio.fingerprint,
                        "feature_key": feature_key,
                        "frame_fingerprint": config.frame_fingerprint,
                        "frame_count": len(frames),
                        "frame_set_digest": frame_set_digest(frames),
                        "summary": summary,
                        "frames": frames,
                    }
                    feature_files[file_id] = info
                    cache.put("features", feature_key, info)
                    previous = cache.previous_file_state(file_id)
                    decisions["features"][file_id] = "recompute"
                    info["_reason"] = _reason_for_miss(
                        "features",
                        previous=previous,
                        current={
                            "audio_fingerprint": audio.fingerprint,
                            "frame_fingerprint": config.frame_fingerprint,
                        },
                    )
            if computed_any and config.verify_serial_consistency:
                feature_verification_mode = "serial_recompute"
                recomputed_feature_ids = [
                    file_id
                    for file_id in sorted(feature_files)
                    if decisions["features"].get(file_id) == "recompute"
                ]
                for file_id in recomputed_feature_ids:
                    audio = audio_data.get(file_id)
                    if audio is None:
                        feature_serial_verified = False
                        continue
                    serial_frames: list[dict[str, Any]] = []
                    for frame_index, (start, end) in enumerate(
                        frame_ranges(
                            audio.sample_count, config.frame_length_samples, config.hop_samples
                        )
                    ):
                        serial_frames.append(
                            compute_frame_features(
                                file_id,
                                frame_index,
                                audio.samples[start:end].copy(),
                                config.sample_rate,
                                config.clipping_amplitude_threshold,
                            )
                        )
                    if sha256_json(serial_frames) != sha256_json(feature_files[file_id]["frames"]):
                        feature_serial_verified = False
                        batch_failures.append(
                            Failure(
                                code=E_FRAME_FEATURE,
                                message="parallel frame features differ from serial recomputation",
                                file_id=file_id,
                                stage="features",
                                task_key=f"{file_id}|frame|serial-verify",
                            )
                        )
            elif feature_files and all(
                value == "reuse" for value in decisions["features"].values()
            ):
                feature_verification_mode = "cache_reuse"
            feature_payload = {
                "files": {
                    file_id: {
                        key: value
                        for key, value in info.items()
                        if not key.startswith("_")
                    }
                    for file_id, info in sorted(feature_files.items())
                },
                "failures": [failure.to_dict() for failure in sorted_failures(batch_failures)],
                "serial_verification": {
                    "matches": feature_serial_verified,
                    "mode": feature_verification_mode,
                    "max_concurrency": config.worker_count,
                },
            }
            snapshot.commit_stage("features", feature_payload, feature_input)
        failures.extend(batch_failures)
        cache.save()
        feature_artifact = snapshot.stage_entry("features") or {}
        timings["features"] = {
            "elapsed_ms": round((time.perf_counter() - stage_started) * 1000.0, 3),
            "resumed": features_resumed,
            "artifact_version": feature_artifact.get("artifact_version"),
            "serial_verification": {
                "matches": feature_serial_verified,
                "mode": feature_verification_mode,
            },
        }
        self._interrupt_if_requested("features", cache)

        # Stage 3: dimension scoring and exact fixed-point aggregation.
        stage_started = time.perf_counter()
        scores_resumed = snapshot.can_resume("scores", score_input)
        dimension_batch_failures: list[Failure] = []
        score_serial_verified = True
        score_verification_mode = "not_run"
        if scores_resumed:
            score_payload = snapshot.load_stage("scores")
            score_files = {
                str(file_id): dict(value)
                for file_id, value in score_payload.get("files", {}).items()
            }
            dimension_batch_failures.extend(
                Failure.from_dict(item) for item in score_payload.get("failures", [])
            )
            score_serial_verified = bool(
                score_payload.get("serial_verification", {}).get("matches", True)
            )
            score_verification_mode = "snapshot_resume"
            for file_id, info in score_files.items():
                decisions["dimensions"][file_id] = "reuse"
                decisions["total"][file_id] = "reuse"
                info["dimension_payload"]["_dimension_reason"] = "snapshot_stage_resume"
                info["total_payload"]["_total_reason"] = "snapshot_stage_resume"
                dim_key = str(info["dimension_key"])
                if cache.get("dimensions", dim_key) is None:
                    cache.put("dimensions", dim_key, info["dimension_payload"])
                total_key = str(info["total_key"])
                if cache.get("totals", total_key) is None:
                    cache.put("totals", total_key, info["total_payload"])
        else:
            score_files = {}
            score_results: dict[str, dict[str, Any]] = {}
            score_jobs: dict[Any, str] = {}
            computed_dimensions = False
            for file_id in sorted(feature_files):
                info = feature_files[file_id]
                dimension_key = sha256_json(
                    {
                        "schema": "dimensions-v1",
                        "feature_key": info["feature_key"],
                        "threshold_fingerprint": config.threshold_fingerprint,
                        "quantization_levels": config.quantization_levels,
                        "scorer": "dimension-bands-v1",
                    }
                )
                cached = cache.get("dimensions", dimension_key)
                if cached is not None:
                    score_results[file_id] = dict(cached)
                    decisions["dimensions"][file_id] = "reuse"
                else:
                    score_jobs[
                        file_id
                    ] = None  # type: ignore[assignment]
            pending_dimensions = sorted(score_jobs)
            if pending_dimensions:
                with ThreadPoolExecutor(
                    max_workers=config.worker_count,
                    thread_name_prefix="aqeval-score",
                ) as executor:
                    future_map: dict[Any, str] = {}
                    for file_id in pending_dimensions:
                        info = feature_files[file_id]
                        task_key = f"{file_id}|score|dimensions"
                        future = executor.submit(
                            _safe_task,
                            task_key,
                            lambda fid=file_id, summary=info["summary"]: score_dimensions(
                                fid, summary, config
                            ),
                        )
                        future_map[future] = file_id
                    for future in as_completed(future_map):
                        file_id = future_map[future]
                        result = future.result()
                        if not result["ok"]:
                            dimension_batch_failures.append(
                                Failure(
                                    code=E_SCORE,
                                    message=f"dimension scoring failed: {result['error']}",
                                    file_id=file_id,
                                    stage="score",
                                    task_key=result["task_key"],
                                )
                            )
                            continue
                        raw = result["value"]
                        payload = {
                            "file_id": file_id,
                            "dimensions_micro": raw["dimensions_micro"],
                            "dimensions_permille": raw["dimensions_permille"],
                            "raw_scores_permille": raw["raw_scores_permille"],
                            "metrics": raw["metrics"],
                            "boundaries": raw["boundaries"],
                            "scorer_version": raw["scorer_version"],
                        }
                        score_results[file_id] = payload
                        dimension_key = sha256_json(
                            {
                                "schema": "dimensions-v1",
                                "feature_key": feature_files[file_id]["feature_key"],
                                "threshold_fingerprint": config.threshold_fingerprint,
                                "quantization_levels": config.quantization_levels,
                                "scorer": "dimension-bands-v1",
                            }
                        )
                        cache.put("dimensions", dimension_key, payload)
                        decisions["dimensions"][file_id] = "recompute"
                        computed_dimensions = True
            for file_id in sorted(score_results):
                payload = score_results[file_id]
                dimension_key = sha256_json(
                    {
                        "schema": "dimensions-v1",
                        "feature_key": feature_files[file_id]["feature_key"],
                        "threshold_fingerprint": config.threshold_fingerprint,
                        "quantization_levels": config.quantization_levels,
                        "scorer": "dimension-bands-v1",
                    }
                )
                total_key = sha256_json(
                    {
                        "schema": "total-v1",
                        "dimension_key": dimension_key,
                        "weights_fingerprint": config.weights_fingerprint,
                        "combiner": "weighted-fixed-v1",
                    }
                )
                total_payload = cache.get("totals", total_key)
                if total_payload is not None:
                    decisions["total"][file_id] = "reuse"
                else:
                    combined = combine_fixed_total(payload["dimensions_micro"], config)
                    total_payload = {
                        "file_id": file_id,
                        "total_micro": combined["total_micro"],
                        "total_permille": combined["total_permille"],
                        "weighted_contributions_micro": combined["weighted_contributions_micro"],
                    }
                    cache.put("totals", total_key, total_payload)
                    decisions["total"][file_id] = "recompute"
                previous = cache.previous_file_state(file_id)
                if decisions["dimensions"].get(file_id) == "recompute":
                    payload["_dimension_reason"] = _reason_for_miss(
                        "dimensions",
                        previous=previous,
                        current={
                            "threshold_fingerprint": config.threshold_fingerprint,
                            "quantization_levels": list(config.quantization_levels),
                        },
                    )
                if decisions["total"].get(file_id) == "recompute":
                    total_payload["_total_reason"] = _reason_for_miss(
                        "total",
                        previous=previous,
                        current={"weights_fingerprint": config.weights_fingerprint},
                    )
                dimension_sums = sum(payload["dimensions_micro"].values())
                score_files[file_id] = {
                    "file_id": file_id,
                    "feature_key": feature_files[file_id]["feature_key"],
                    "dimension_key": dimension_key,
                    "total_key": total_key,
                    "dimension_payload": payload,
                    "total_payload": total_payload,
                    "frame_refs": [
                        frame["frame_fingerprint"] for frame in feature_files[file_id]["frames"]
                    ],
                    "dimension_sums_micro": dimension_sums,
                }
            if computed_dimensions and config.verify_serial_consistency:
                score_verification_mode = "serial_recompute"
                recomputed_dimension_ids = [
                    file_id
                    for file_id in sorted(score_results)
                    if decisions["dimensions"].get(file_id) == "recompute"
                ]
                for file_id in recomputed_dimension_ids:
                    serial_payload = score_dimensions(
                        file_id, feature_files[file_id]["summary"], config
                    )
                    serial_compare = {
                        "dimensions_micro": serial_payload["dimensions_micro"],
                        "dimensions_permille": serial_payload["dimensions_permille"],
                        "raw_scores_permille": serial_payload["raw_scores_permille"],
                        "metrics": serial_payload["metrics"],
                        "boundaries": serial_payload["boundaries"],
                    }
                    parallel_payload = score_results.get(file_id, {})
                    parallel_compare = {
                        key: parallel_payload.get(key)
                        for key in (
                            "dimensions_micro",
                            "dimensions_permille",
                            "raw_scores_permille",
                            "metrics",
                            "boundaries",
                        )
                    }
                    if sha256_json(serial_compare) != sha256_json(parallel_compare):
                        score_serial_verified = False
                        dimension_batch_failures.append(
                            Failure(
                                code=E_SCORE,
                                message="parallel dimension scores differ from serial recomputation",
                                file_id=file_id,
                                stage="score",
                                task_key=f"{file_id}|score|serial-verify",
                            )
                        )
            elif score_results and all(
                value == "reuse" for value in decisions["dimensions"].values()
            ):
                score_verification_mode = "cache_reuse"
            clean_score_files: dict[str, dict[str, Any]] = {}
            for file_id, value in sorted(score_files.items()):
                clean_value = dict(value)
                dimension_payload = {
                    key: item
                    for key, item in value["dimension_payload"].items()
                    if not key.startswith("_")
                }
                total_payload = {
                    key: item
                    for key, item in value["total_payload"].items()
                    if not key.startswith("_")
                }
                clean_value["dimension_payload"] = dimension_payload
                clean_value["total_payload"] = total_payload
                clean_score_files[file_id] = clean_value
            score_payload = {
                "files": clean_score_files,
                "failures": [
                    failure.to_dict()
                    for failure in sorted_failures(dimension_batch_failures)
                ],
                "serial_verification": {
                    "matches": score_serial_verified,
                    "mode": score_verification_mode,
                    "max_concurrency": config.worker_count,
                },
            }
            snapshot.commit_stage("scores", score_payload, score_input)
        failures.extend(dimension_batch_failures)
        cache.save()
        scores_artifact = snapshot.stage_entry("scores") or {}
        timings["scores"] = {
            "elapsed_ms": round((time.perf_counter() - stage_started) * 1000.0, 3),
            "resumed": scores_resumed,
            "artifact_version": scores_artifact.get("artifact_version"),
            "serial_verification": {
                "matches": score_serial_verified,
                "mode": score_verification_mode,
            },
        }
        self._interrupt_if_requested("scores", cache)

        # Stage 4: grading, version conversion and serialized outputs.
        stage_started = time.perf_counter()
        conversion_resumed = snapshot.can_resume("conversion", conversion_input)
        boundaries: list[dict[str, Any]] = []
        canonical: dict[str, UnifiedResult] = {}
        for file_id in sorted(score_files):
            info = score_files[file_id]
            dimension_payload = info["dimension_payload"]
            total_payload = info["total_payload"]
            total_micro = int(total_payload["total_micro"])
            total_permille = int(total_payload["total_permille"])
            code, name, at_threshold = grade_total_micro(total_micro, level_map)
            boundaries.extend(dimension_payload.get("boundaries", []))
            if at_threshold:
                boundaries.append(
                    {
                        "file_id": file_id,
                        "dimension": "total",
                        "kind": "level_threshold",
                        "value": total_permille,
                        "threshold": total_permille,
                        "rule": "total equals the selected level threshold",
                    }
                )
            feature_info = feature_files.get(file_id)
            frames = (
                frame_summaries_from_features(feature_info["frames"])
                if feature_info is not None
                else []
            )
            canonical[file_id] = UnifiedResult(
                file_id=file_id,
                source_version=0,
                total_micro=total_micro,
                dimensions_micro={
                    dimension: int(dimension_payload["dimensions_micro"][dimension])
                    for dimension in DIMENSIONS
                },
                level_code=code,
                level_name=name,
                precision_tag="reeval_float6",
                frame_summaries=frames,
                bypass={"computed_from": "reevaluation"},
                location=str(feature_info.get("path", "")) if feature_info else "",
            )
        if conversion_resumed:
            conversion_payload = snapshot.load_stage("conversion")
            serialization_failures = [
                Failure.from_dict(item)
                for item in conversion_payload.get("failures", [])
            ]
            degradation_dicts = list(conversion_payload.get("degradations", []))
            output_info = dict(conversion_payload["output"])
            serialized_payloads = {
                str(file_id): dict(value)
                for file_id, value in conversion_payload.get("serialized", {}).items()
            }
            for file_id in serialized_payloads:
                decisions["serialization"][file_id] = "reuse"
                serialized_payloads[file_id]["_reason"] = "snapshot_stage_resume"
        else:
            serialization_failures = []
            degradation_dicts = []
            serialized_payloads: dict[str, dict[str, Any]] = {}
            level_entry_fingerprints = {
                file_id: level_map.entry_fingerprint(record.level_code or "")
                for file_id, record in canonical.items()
            }
            serialized_cache_keys: dict[str, str] = {}
            for file_id in sorted(canonical):
                record = canonical[file_id]
                serial_key = sha256_json(
                    {
                        "schema": "serialization-v1",
                        "codec": SERIALIZATION_CODEC,
                        "target_version": active_version,
                        "total_micro": record.total_micro,
                        "dimensions_micro": record.dimensions_micro,
                        "level_code": record.level_code,
                        "level_name": record.level_name,
                        "precision_tag": record.precision_tag,
                        "level_entry_fingerprint": level_entry_fingerprints[file_id],
                        "audio_fingerprint": feature_files.get(file_id, {}).get(
                            "audio_fingerprint"
                        ),
                        "serializer_config_fingerprint": sha256_json(
                            {
                                "codec": SERIALIZATION_CODEC,
                                "target_version": active_version,
                                "fixed_decimal_bits": config.fixed_decimal_bits
                                if active_version == 3
                                else None,
                                "frame_summary_required": active_version == 3,
                            }
                        ),
                        "fixed_decimal_bits": config.fixed_decimal_bits
                        if active_version == 3
                        else None,
                        "frame_set_digest": sha256_json(
                            [
                                frame.frame_fingerprint
                                for frame in record.frame_summaries
                            ]
                        )
                        if active_version == 3
                        else None,
                    }
                )
                serialized_cache_keys[file_id] = serial_key
                cached = cache.get("serialized", serial_key)
                if cached is not None:
                    serialized_payloads[file_id] = dict(cached)
                    decisions["serialization"][file_id] = "reuse"
                    degradation_dicts.extend(cached.get("degradations", []))
                    continue
                try:
                    converted, degradations, bypass = convert_one(
                        record, active_version, level_map, config.fixed_decimal_bits
                    )
                except LevelMapError as exc:
                    serialization_failures.append(
                        Failure(
                            code="E_LEVEL_UNMAPPED",
                            message=str(exc),
                            file_id=file_id,
                            stage="serialize",
                            task_key=f"{file_id}|serialize|v{active_version}",
                        )
                    )
                    decisions["serialization"][file_id] = "recompute"
                    continue
                if active_version == 1:
                    blob_data = write_format_bytes(1, [converted], config.fixed_decimal_bits)
                elif active_version == 2:
                    document = json.loads(
                        write_format_bytes(2, [converted], config.fixed_decimal_bits)
                    )
                    blob_data = canonical_json_bytes(document["results"][0])
                else:
                    blob_data = canonical_json_bytes(converted.to_dict())
                blob_digest = cache.store_blob(blob_data)
                cache_value = {
                    "file_id": file_id,
                    "target_version": active_version,
                    "converted": converted.to_dict(),
                    "blob": blob_digest,
                    "degradations": [item.to_dict() for item in degradations],
                    "bypass": bypass,
                    "cache_key": serial_key,
                }
                cache.put("serialized", serial_key, cache_value)
                serialized_payloads[file_id] = cache_value
                degradation_dicts.extend(item.to_dict() for item in degradations)
                decisions["serialization"][file_id] = "recompute"
                previous = cache.previous_file_state(file_id)
                serialized_payloads[file_id]["_reason"] = _reason_for_miss(
                    "serialization",
                    previous=previous,
                    current={
                        "target_version": active_version,
                        "fixed_decimal_bits": config.fixed_decimal_bits,
                        "level_entry_fingerprint": level_entry_fingerprints[file_id],
                    },
                )
            output_path, output_bytes, output_digest = self._assemble_active_output(
                active_version,
                serialized_payloads,
                cache,
                config.fixed_decimal_bits,
            )
            bypass_entries = [
                payload["bypass"]
                for _, payload in sorted(serialized_payloads.items())
                if payload.get("bypass") is not None
            ]
            bypass_document = {
                "schema": "aqeval.bypass.v1",
                "target_version": active_version,
                "entries": sorted(bypass_entries, key=lambda item: item["file_id"]),
            }
            atomic_write_json(self.workspace / "output" / "bypass_active.json", bypass_document)
            output_info = {
                "version": active_version,
                "path": str(output_path),
                "sha256": output_digest,
                "size": len(output_bytes),
                "record_count": len(serialized_payloads),
                "bypass_path": str(self.workspace / "output" / "bypass_active.json"),
                "bypass_sha256": sha256_json(bypass_document),
            }
            atomic_write_json(
                self.workspace / "output" / "active.json",
                {
                    "schema": "aqeval.active-output.v1",
                    "version": active_version,
                    "path": str(output_path),
                    "sha256": output_digest,
                    "record_count": len(serialized_payloads),
                },
            )
            conversion_payload = {
                "target_version": active_version,
                "serialized": {
                    file_id: {
                        key: value
                        for key, value in payload.items()
                        if not key.startswith("_")
                    }
                    for file_id, payload in sorted(serialized_payloads.items())
                },
                "degradations": sorted(
                    degradation_dicts,
                    key=lambda item: (
                        item["file_id"],
                        item["source_version"],
                        item["target_version"],
                        item["code"],
                    ),
                ),
                "failures": [
                    failure.to_dict()
                    for failure in sorted_failures(serialization_failures)
                ],
                "output": output_info,
                "serialized_cache_keys": serialized_cache_keys,
            }
            snapshot.commit_stage("conversion", conversion_payload, conversion_input)
        failures.extend(serialization_failures)
        for file_id in sorted(serialized_payloads):
            decisions["serialization"].setdefault(file_id, "reuse")
        cache.save()
        conversion_artifact = snapshot.stage_entry("conversion") or {}
        timings["conversion"] = {
            "elapsed_ms": round((time.perf_counter() - stage_started) * 1000.0, 3),
            "resumed": conversion_resumed,
            "artifact_version": conversion_artifact.get("artifact_version"),
        }
        self._interrupt_if_requested("conversion", cache)

        # Stable cache ledger.
        ledger_entries: list[dict[str, Any]] = []
        for file_id in sorted(feature_files):
            feature_info = feature_files[file_id]
            entry = {
                "file_id": file_id,
                "stage": "features",
                "artifact": "frame_features",
                "cache_key": feature_info["feature_key"],
                "status": decisions["features"].get(file_id, "recompute"),
                "reason": feature_info.get("_reason")
                or (
                    "cache_hit"
                    if decisions["features"].get(file_id) == "reuse"
                    else "computed"
                ),
                "frame_count": feature_info.get("frame_count", 0),
                "frame_refs": [
                    frame["frame_fingerprint"] for frame in feature_info.get("frames", [])
                ],
            }
            ledger_entries.append(entry)
            if file_id not in score_files:
                continue
            score_info = score_files[file_id]
            ledger_entries.append(
                {
                    "file_id": file_id,
                    "stage": "dimensions",
                    "artifact": "dimension_scores",
                    "cache_key": score_info["dimension_key"],
                    "status": decisions["dimensions"].get(file_id, "recompute"),
                    "reason": score_info["dimension_payload"].get("_dimension_reason")
                    or (
                        "cache_hit"
                        if decisions["dimensions"].get(file_id) == "reuse"
                        else "computed"
                    ),
                    "frame_refs": score_info["frame_refs"],
                    "dimension_sums_micro": score_info["dimension_sums_micro"],
                }
            )
            ledger_entries.append(
                {
                    "file_id": file_id,
                    "stage": "total",
                    "artifact": "fixed_total",
                    "cache_key": score_info["total_key"],
                    "status": decisions["total"].get(file_id, "recompute"),
                    "reason": score_info["total_payload"].get("_total_reason")
                    or (
                        "cache_hit"
                        if decisions["total"].get(file_id) == "reuse"
                        else "computed"
                    ),
                    "dimension_sums_micro": score_info["dimension_sums_micro"],
                    "weighted_contributions_micro": score_info["total_payload"].get(
                        "weighted_contributions_micro", {}
                    ),
                }
            )
            if file_id in serialized_payloads:
                payload = serialized_payloads[file_id]
                ledger_entries.append(
                    {
                        "file_id": file_id,
                        "stage": "serialization",
                        "artifact": f"exchange_v{active_version}",
                        "cache_key": payload.get("cache_key"),
                        "status": decisions["serialization"].get(file_id, "recompute"),
                        "reason": payload.get("_reason")
                        or (
                            "cache_hit"
                            if decisions["serialization"].get(file_id) == "reuse"
                            else "computed"
                        ),
                    }
                )
        ledger_entries.sort(
            key=lambda entry: (
                STAGE_INDEX.get(entry["stage"], 99),
                entry["file_id"],
                entry["artifact"],
            )
        )
        for index, entry in enumerate(ledger_entries, start=1):
            entry["order"] = index
            entry["task_key"] = (
                f"{entry['file_id']}|{entry['stage']}|{entry['artifact']}"
            )
        reuse = [
            {"order": entry["order"], "task_key": entry["task_key"]}
            for entry in ledger_entries
            if entry["status"] == "reuse"
        ]
        recompute = [
            {"order": entry["order"], "task_key": entry["task_key"]}
            for entry in ledger_entries
            if entry["status"] == "recompute"
        ]
        ledger = {
            "schema": "aqeval.cache-ledger.v1",
            "ordering_rule": "stage order then file_id then artifact",
            "stage_order": ["features", "dimensions", "total", "serialization"],
            "entries": ledger_entries,
            "reuse": reuse,
            "recompute": recompute,
            "reuse_count": len(reuse),
            "recompute_count": len(recompute),
            "config_hash": config.config_hash,
            "content_fingerprint": content_fingerprint,
        }
        atomic_write_json(self.workspace / "cache" / "ledger.json", ledger)

        # Persist per-file cache state so the next run can explain exact invalidations.
        for file_id in sorted(feature_files):
            feature_info = feature_files[file_id]
            score_info = score_files.get(file_id, {})
            serial_payload = serialized_payloads.get(file_id, {})
            cache.set_file_state(
                file_id,
                {
                    "audio_fingerprint": feature_info.get("audio_fingerprint"),
                    "frame_fingerprint": config.frame_fingerprint,
                    "feature_key": feature_info.get("feature_key"),
                    "threshold_fingerprint": config.threshold_fingerprint,
                    "quantization_levels": list(config.quantization_levels),
                    "dimension_key": score_info.get("dimension_key"),
                    "weights_fingerprint": config.weights_fingerprint,
                    "total_key": score_info.get("total_key"),
                    "target_version": active_version,
                    "fixed_decimal_bits": config.fixed_decimal_bits,
                    "level_entry_fingerprint": level_map.entry_fingerprint(
                        canonical.get(file_id).level_code
                        if canonical.get(file_id) is not None
                        else ""
                    ),
                    "serialization_key": serial_payload.get("cache_key"),
                },
            )
        cache.save()

        original_by_version = {
            version: {record.file_id: record for record in outcome.records}
            for version, outcome in outcomes.items()
        }
        deviations = deviation_report(canonical, original_by_version)
        consistency = consistency_report(canonical, level_map, config.fixed_decimal_bits)
        combined_degradations = list(degradation_dicts)
        combined_degradations.extend(consistency.get("degradations", []))
        combined_degradations.sort(
            key=lambda item: (
                item["file_id"],
                item["source_version"],
                item["target_version"],
                item["code"],
            )
        )
        failures.extend(Failure.from_dict(item) for item in consistency.get("failures", []))
        failures_sorted = sorted_failures(failures)
        boundary_sorted = sorted(
            boundaries,
            key=lambda item: (
                item["file_id"],
                DIMENSIONS.index(item["dimension"]) if item["dimension"] in DIMENSIONS else 4,
                item["kind"],
                float(item.get("threshold", 0.0)),
            ),
        )
        concurrency_match = feature_serial_verified and score_serial_verified
        injections = injection_hits(
            failures=[failure.to_dict() for failure in failures_sorted],
            degradations=combined_degradations,
            boundaries=boundary_sorted,
            concurrency_match=concurrency_match,
            snapshot_stale=snapshot.stale,
        )
        output_info = dict(output_info)
        output_info["path"] = str(output_info["path"])
        canonical_dicts = [canonical[file_id].to_dict() for file_id in sorted(canonical)]
        business_codes = sorted(
            {failure.code for failure in failures_sorted}
            | {str(item.get("code")) for item in combined_degradations}
            | ({E_SNAPSHOT_STALE} if snapshot.stale else set())
        )
        report = {
            "schema": "aqeval.diagnostic.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(self.workspace),
            "run": {
                "global_version": global_run + 1,
                "job_version": config_revision,
                "config_hash": config.config_hash,
                "level_map_hash": level_map.digest,
                "content_fingerprint": content_fingerprint,
                "version_fingerprint": snapshot.version_fingerprint,
                "active_version": active_version,
                "exchange_hashes": exchange_hashes,
                "audio_hashes": audio_hashes,
            },
            "config": config.to_dict(),
            "snapshot": {
                **snapshot.as_dict(),
                "stale": snapshot.stale,
                "stale_code": E_SNAPSHOT_STALE if snapshot.stale else None,
                "resumed_stages": [
                    stage
                    for stage, info in timings.items()
                    if bool(info.get("resumed"))
                ],
            },
            "stages": timings,
            "summary": {
                "canonical_count": len(canonical),
                "parsed_count": sum(len(outcome.records) for outcome in outcomes.values()),
                "failure_count": len(failures_sorted),
                "degradation_count": len(combined_degradations),
                "boundary_count": len(boundary_sorted),
                "reuse_count": len(reuse),
                "recompute_count": len(recompute),
            },
            "canonical_results": canonical_dicts,
            "deviations": deviations,
            "consistency": consistency,
            "degradations": {
                "entries": combined_degradations,
                "statistics": degradation_statistics(combined_degradations),
            },
            "boundary_samples": boundary_sorted,
            "failures": [failure.to_dict() for failure in failures_sorted],
            "business_codes": business_codes,
            "cache_ledger": ledger,
            "concurrency": {
                "configured_max": config.max_concurrency,
                "configured_cap": config.concurrency_cap,
                "workers": config.worker_count,
                "within_cap": config.worker_count <= config.concurrency_cap,
                "feature_serial_matches": feature_serial_verified,
                "score_serial_matches": score_serial_verified,
                "serial_matches_concurrent": concurrency_match,
                "feature_verification_mode": feature_verification_mode,
                "score_verification_mode": score_verification_mode,
            },
            "injections": injections,
            "outputs": output_info,
            "elapsed_ms": round((time.perf_counter() - run_started) * 1000.0, 3),
        }
        output_digest = sha256_json(
            {
                "output": output_info.get("sha256"),
                "canonical": canonical_dicts,
                "failures": [failure.to_dict() for failure in failures_sorted],
                "degradations": combined_degradations,
                "ledger_order": [
                    {
                        "order": entry["order"],
                        "task_key": entry["task_key"],
                    }
                    for entry in ledger_entries
                ],
            }
        )
        report_path = self.workspace / "reports" / "diagnostic.json"
        snapshot.complete(str(report_path), output_digest)
        report["snapshot"] = {
            **snapshot.as_dict(),
            "stale": snapshot.stale,
            "stale_code": E_SNAPSHOT_STALE if snapshot.stale else None,
            "resumed_stages": [
                stage for stage, info in timings.items() if bool(info.get("resumed"))
            ],
        }
        atomic_write_json(report_path, report)
        state.update(
            {
                "schema": "aqeval.state.v1",
                "active_version": active_version,
                "config_revision": config_revision,
                "global_run": global_run + 1,
                "last_run": {
                    "report": str(report_path),
                    "output_digest": output_digest,
                    "config_hash": config.config_hash,
                    "content_fingerprint": content_fingerprint,
                },
            }
        )
        atomic_write_json(self.workspace / "state.json", state)
        return report

    def _assemble_active_output(
        self,
        active_version: int,
        serialized_payloads: dict[str, dict[str, Any]],
        cache: ArtifactCache,
        fixed_bits: int,
    ) -> tuple[Path, bytes, str]:
        output_dir = self.workspace / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = OUTPUT_EXTENSIONS[active_version]
        output_path = output_dir / f"active_v{active_version}.{extension}"
        ordered = sorted(serialized_payloads)
        if active_version == 1:
            data = b"".join(
                cache.load_blob(serialized_payloads[file_id]["blob"]) for file_id in ordered
            )
        elif active_version == 2:
            items = [
                json.loads(cache.load_blob(serialized_payloads[file_id]["blob"]))
                for file_id in ordered
            ]
            data = pretty_json_bytes(
                {"schema": "aqeval.v2", "precision_tag": "float6", "results": items}
            )
        else:
            records = [
                UnifiedResult.from_dict(serialized_payloads[file_id]["converted"])
                for file_id in ordered
            ]
            data = write_format_bytes(3, records, fixed_bits)
        atomic_write_bytes(output_path, data)
        return output_path, data, sha256_bytes(data)
