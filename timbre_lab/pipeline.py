from __future__ import annotations

import copy
import time
from concurrent.futures import Executor, Future
from typing import Any, Callable

from .aggregation import aggregate_speaker, speaker_cache_key
from .assignment import adjacent_pairs, assignment_cache_key, decide_assignment, group_speakers
from .config import AnalysisConfig
from .features import extract_segment_features, segment_content_fingerprint
from .model import AudioDocument, Segment
from .quantization import build_description, description_cache_key


ProgressCallback = Callable[[str, int, int], None]


def _noop_progress(phase: str, completed: int, total: int) -> None:
    return None


def _cache_event(events: dict[str, Any], phase: str, key: str, reused: bool, task_key: str, affected: bool) -> None:
    bucket = events[phase]
    bucket["reused" if reused else "recomputed"].append(
        {"cache_key": key, "task_key": task_key, "affected": affected}
    )


def _task_key(phase: str, segment: Segment) -> str:
    return f"{segment.file}|{segment.start_frame:010d}|{segment.end_frame:010d}|{phase}"


class CachedPipeline:
    def __init__(
        self,
        config: AnalysisConfig,
        executor: Executor | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.progress = progress or _noop_progress
        self.feature_cache: dict[str, dict[str, Any]] = {}
        self.description_cache: dict[str, dict[str, Any]] = {}
        self.assignment_cache: dict[str, dict[str, Any]] = {}
        self.speaker_cache: dict[str, dict[str, Any]] = {}
        self.last_cache_events: dict[str, Any] = {}
        self.last_task_order: list[str] = []

    def _run_tasks(
        self,
        phase: str,
        tasks: dict[str, Callable[[], dict[str, Any]]],
        cache: dict[str, dict[str, Any]],
        progress_total: int,
        progress_done: int,
        affected_keys: set[str],
        cache_events: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], int]:
        results: dict[str, dict[str, Any]] = {}
        futures: dict[str, Future[dict[str, Any]]] = {}
        ordered_keys = sorted(tasks, key=lambda cache_key: getattr(tasks[cache_key], "sort_key", cache_key))
        for key in ordered_keys:
            sort_key = getattr(tasks[key], "sort_key", key)
            self.last_task_order.append(sort_key)
            if self.executor is None:
                cached = cache.get(key)
                if cached is not None:
                    results[key] = cached
                    _cache_event(cache_events, phase, key, True, sort_key, key in affected_keys)
                    progress_done += 1
                    self.progress(phase, progress_done, progress_total)
                else:
                    result = tasks[key]()
                    cache[key] = result
                    results[key] = result
                    _cache_event(cache_events, phase, key, False, sort_key, True)
                    progress_done += 1
                    self.progress(phase, progress_done, progress_total)
                continue
            futures[key] = self.executor.submit(tasks[key])
        for key in ordered_keys:
            if self.executor is None:
                continue
            future = futures[key]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - surfaced to caller with context
                for pending in futures.values():
                    pending.cancel()
                raise RuntimeError(f"{phase} task failed for cache key {key}") from exc
            cache[key] = result
            results[key] = result
            _cache_event(cache_events, phase, key, False, getattr(tasks[key], "sort_key", key), key in affected_keys)
            progress_done += 1
            self.progress(phase, progress_done, progress_total)
        return results, progress_done

    def compute(
        self,
        document: AudioDocument,
        overrides: dict[str, int] | None,
        failures: list[dict[str, Any]],
        affected_segments: set[str] | None = None,
        affected_stages: set[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self.last_task_order = []
        affected_segments = affected_segments or set()
        affected_stages = affected_stages or set()
        cache_events = {
            phase: {"reused": [], "recomputed": []}
            for phase in ("feature", "description", "assignment", "speaker")
        }
        ordered = document.ordered_segments()
        total_tasks = len(ordered) * 2 + max(0, len(ordered) - len(document.files)) + len(ordered)
        completed = 0

        feature_tasks: dict[str, Callable[[], dict[str, Any]]] = {}
        feature_keys: dict[str, str] = {}
        affected_feature_keys: set[str] = set()
        for segment in ordered:
            audio = document.files[segment.file]
            content_key = segment_content_fingerprint(audio, segment)
            from .features import feature_cache_key

            key = feature_cache_key(content_key, self.config, segment)
            feature_keys[segment.segment_id] = key
            if key in self.feature_cache:
                continue
            task_key = _task_key("feature", segment)

            def make_feature_task(audio=audio, segment=segment) -> Callable[[], dict[str, Any]]:
                def run_feature() -> dict[str, Any]:
                    return extract_segment_features(audio, segment, self.config)

                return run_feature

            task = make_feature_task()
            task.__name__ = f"feature:{task_key}"
            task.sort_key = task_key
            feature_tasks[key] = task
            if "feature" in affected_stages or segment.segment_id in affected_segments:
                affected_feature_keys.add(key)
        all_features_by_key = dict(self.feature_cache)
        computed_features, completed = self._run_tasks(
            "feature",
            feature_tasks,
            self.feature_cache,
            total_tasks,
            completed,
            affected_feature_keys,
            cache_events,
        )
        all_features_by_key.update(computed_features)
        features_by_segment: dict[str, dict[str, Any]] = {}
        for segment in ordered:
            key = feature_keys[segment.segment_id]
            features_by_segment[segment.segment_id] = all_features_by_key[key]
        # Record feature hits even when no miss task was submitted.
        for segment in ordered:
            key = feature_keys[segment.segment_id]
            if key in self.feature_cache and key not in computed_features:
                _cache_event(cache_events, "feature", key, True, _task_key("feature", segment), segment.segment_id in affected_segments)
                completed += 1
                self.progress("feature", completed, total_tasks)

        description_tasks: dict[str, Callable[[], dict[str, Any]]] = {}
        description_keys: dict[str, str] = {}
        affected_description_keys: set[str] = set()
        for segment in ordered:
            feature = features_by_segment[segment.segment_id]
            key = description_cache_key(feature["cache_key"], self.config, overrides)
            description_keys[segment.segment_id] = key
            if key in self.description_cache:
                continue
            task_key = _task_key("description", segment)

            def make_description_task(feature=feature) -> Callable[[], dict[str, Any]]:
                def run_description() -> dict[str, Any]:
                    return build_description(feature, self.config, overrides)

                return run_description

            task = make_description_task()
            task.__name__ = f"description:{task_key}"
            task.sort_key = task_key
            description_tasks[key] = task
            if (
                "description" in affected_stages
                or segment.segment_id in affected_segments
                or "feature" in affected_stages
            ):
                affected_description_keys.add(key)
        all_descriptions = dict(self.description_cache)
        computed_descriptions, completed = self._run_tasks(
            "description",
            description_tasks,
            self.description_cache,
            total_tasks,
            completed,
            affected_description_keys,
            cache_events,
        )
        all_descriptions.update(computed_descriptions)
        descriptions_by_segment: dict[str, dict[str, Any]] = {}
        for segment in ordered:
            key = description_keys[segment.segment_id]
            descriptions_by_segment[segment.segment_id] = all_descriptions[key]
        for segment in ordered:
            key = description_keys[segment.segment_id]
            if key in self.description_cache and key not in computed_descriptions:
                _cache_event(
                    cache_events,
                    "description",
                    key,
                    True,
                    _task_key("description", segment),
                    segment.segment_id in affected_segments or "description" in affected_stages,
                )
                completed += 1
                self.progress("description", completed, total_tasks)

        assignment_tasks: dict[str, Callable[[], dict[str, Any]]] = {}
        pair_keys: dict[tuple[str, str], str] = {}
        affected_assignment_keys: set[str] = set()
        for left, right in adjacent_pairs(document):
            left_desc = descriptions_by_segment[left.segment_id]
            right_desc = descriptions_by_segment[right.segment_id]
            key = assignment_cache_key(left_desc, right_desc, left, right, self.config)
            pair_keys[(left.segment_id, right.segment_id)] = key
            if key in self.assignment_cache:
                continue
            task_key = _task_key("assignment", left)

            def make_assignment_task(
                left=left,
                right=right,
                left_desc=left_desc,
                right_desc=right_desc,
                left_features=features_by_segment[left.segment_id],
                right_features=features_by_segment[right.segment_id],
            ) -> Callable[[], dict[str, Any]]:
                def run_assignment() -> dict[str, Any]:
                    return decide_assignment(
                        left,
                        right,
                        left_desc,
                        right_desc,
                        left_features,
                        right_features,
                        self.config,
                        overrides,
                    )

                return run_assignment

            task = make_assignment_task()
            task.__name__ = f"assignment:{task_key}"
            task.sort_key = task_key
            assignment_tasks[key] = task
            if (
                "assignment" in affected_stages
                or left.segment_id in affected_segments
                or right.segment_id in affected_segments
            ):
                affected_assignment_keys.add(key)
        all_assignments = dict(self.assignment_cache)
        computed_assignments, completed = self._run_tasks(
            "assignment",
            assignment_tasks,
            self.assignment_cache,
            total_tasks,
            completed,
            affected_assignment_keys,
            cache_events,
        )
        all_assignments.update(computed_assignments)
        assignments: list[dict[str, Any]] = []
        for left, right in adjacent_pairs(document):
            key = pair_keys[(left.segment_id, right.segment_id)]
            assignments.append(all_assignments[key])
        for (left_id, right_id), key in sorted(pair_keys.items()):
            if key in self.assignment_cache and key not in computed_assignments:
                left = document.segment_by_id(left_id)
                _cache_event(
                    cache_events,
                    "assignment",
                    key,
                    True,
                    _task_key("assignment", left),
                    left_id in affected_segments or right_id in affected_segments,
                )
                completed += 1
                self.progress("assignment", completed, total_tasks)

        groups, bridge_conflicts = group_speakers(document, assignments)
        speaker_aggregates: list[dict[str, Any]] = []
        affected_speaker_keys: set[str] = set()
        for group in groups:
            segment_ids = group["segment_ids"]
            weights = {segment_id: int(features_by_segment[segment_id]["effective_frame_count"]) for segment_id in segment_ids}
            key = speaker_cache_key(group["speaker"], segment_ids, descriptions_by_segment, weights, self.config, overrides)
            if key not in self.speaker_cache:
                self.speaker_cache[key] = aggregate_speaker(group, features_by_segment, descriptions_by_segment, self.config, overrides)
                _cache_event(cache_events, "speaker", key, False, f"speaker|{group['speaker']}", True)
            else:
                _cache_event(
                    cache_events,
                    "speaker",
                    key,
                    True,
                    f"speaker|{group['speaker']}",
                    bool(set(segment_ids) & affected_segments),
                )
            speaker_aggregates.append(self.speaker_cache[key])
            if "speaker" in affected_stages or set(segment_ids) & affected_segments:
                affected_speaker_keys.add(key)
            completed += 1
            self.progress("speaker", completed, total_tasks)

        segment_speaker: dict[str, str] = {}
        for group in groups:
            for segment_id in group["segment_ids"]:
                segment_speaker[segment_id] = group["speaker"]
        semantic = build_semantic_result(
            document,
            features_by_segment,
            descriptions_by_segment,
            assignments,
            groups,
            speaker_aggregates,
            segment_speaker,
            bridge_conflicts,
            failures,
            self.config,
            overrides,
        )
        semantic["cache_audit"] = {
            **cache_events,
            "affected_segments": sorted(affected_segments),
            "affected_stages": sorted(affected_stages),
            "affected_speaker_keys": sorted(affected_speaker_keys),
            "task_order": self.last_task_order,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        semantic["cache_refs"] = build_cache_refs(
            ordered, features_by_segment, descriptions_by_segment, pair_keys, speaker_aggregates
        )
        self.last_cache_events = semantic["cache_audit"]
        return semantic


def build_cache_refs(
    ordered: list[Segment],
    features: dict[str, dict[str, Any]],
    descriptions: dict[str, dict[str, Any]],
    pair_keys: dict[tuple[str, str], str],
    speakers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "segments": {
            segment.segment_id: {
                "feature": features[segment.segment_id]["cache_key"],
                "description": descriptions[segment.segment_id]["cache_key"],
            }
            for segment in ordered
        },
        "assignments": {f"{left}->{right}": key for (left, right), key in pair_keys.items()},
        "speakers": {speaker["speaker"]: speaker["cache_key"] for speaker in speakers},
    }


def build_semantic_result(
    document: AudioDocument,
    features: dict[str, dict[str, Any]],
    descriptions: dict[str, dict[str, Any]],
    assignments: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    segment_speaker: dict[str, str],
    bridge_conflicts: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
) -> dict[str, Any]:
    ordered = document.ordered_segments()
    segment_rows: list[dict[str, Any]] = []
    for segment in ordered:
        feature = features[segment.segment_id]
        description = descriptions[segment.segment_id]
        segment_rows.append(
            {
                **segment.as_dict(),
                "effective_frame_count": feature["effective_frame_count"],
                "frame_count": feature["frame_count"],
                "voiced_frame_count": feature["voiced_frame_count"],
                "content_fingerprint": feature["content_fingerprint"],
                "feature_cache_key": feature["cache_key"],
                "description_cache_key": description["cache_key"],
                "statistics": copy.deepcopy(feature["statistics"]),
                "pronunciation_style": feature["pronunciation_style"],
                "resolved_speaker": segment_speaker.get(segment.segment_id),
                "quantization": copy.deepcopy(description["dimensions"]),
                "description_vector": copy.deepcopy(description["vector"]),
                "description_integers": copy.deepcopy(description["integers"]),
            }
        )
    conflicts = [pair["conflict"] for pair in assignments if pair.get("conflict") is not None]
    boundaries = [pair["boundary_reference"] for pair in assignments if pair.get("boundary_reference") is not None]
    styles: dict[str, int] = {}
    for feature in features.values():
        style = feature["pronunciation_style"]
        styles[style] = styles.get(style, 0) + 1
    return {
        "schema_version": 1,
        "config_hash": config.config_hash,
        "quantization_plan": config.quantization_plan(overrides),
        "segments": segment_rows,
        "segment_statistics": [
            {
                "segment_id": segment["segment_id"],
                "effective_frame_count": segment["effective_frame_count"],
                **segment["statistics"],
                "pronunciation_style": segment["pronunciation_style"],
            }
            for segment in segment_rows
        ],
        "descriptions": [
            {
                "segment_id": segment["segment_id"],
                "bit_widths": {item["name"]: item["bit_width"] for item in segment["quantization"]},
                "integers": segment["description_integers"],
                "vector": segment["description_vector"],
                "dimensions": segment["quantization"],
            }
            for segment in segment_rows
        ],
        "assignments": assignments,
        "speaker_groups": groups,
        "speaker_vectors": speakers,
        "assignment_conflicts": conflicts,
        "boundary_samples": boundaries,
        "bridge_conflicts": bridge_conflicts,
        "style_summary": dict(sorted(styles.items())),
        "failures": copy.deepcopy(failures),
    }


def compute_full(
    document: AudioDocument,
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    pipeline = CachedPipeline(config)
    result = pipeline.compute(document, overrides, failures)
    result.pop("cache_audit", None)
    result.pop("cache_refs", None)
    return result


def compare_incremental_with_full(
    incremental: dict[str, Any],
    document: AudioDocument,
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    full = compute_full(document, config, overrides, failures)
    operational_keys = {
        "cache_audit",
        "cache_refs",
        "command_history",
        "full_rebuild_comparison",
        "initialization",
        "segmentation_summaries",
    }
    left = {key: value for key, value in incremental.items() if key not in operational_keys}
    if left == full:
        return {"equal": True, "differences": []}
    differences = _diff_paths(left, full)
    return {"equal": False, "differences": differences[:100]}


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "incremental": repr(left), "full": repr(right)}]
    if isinstance(left, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append({"path": f"{path}.{key}", "incremental": "<missing>", "full": repr(right[key])})
            elif key not in right:
                differences.append({"path": f"{path}.{key}", "incremental": repr(left[key]), "full": "<missing>"})
            else:
                differences.extend(_diff_paths(left[key], right[key], f"{path}.{key}"))
            if len(differences) >= 200:
                break
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": f"{path}.length", "incremental": len(left), "full": len(right)}]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_diff_paths(left_item, right_item, f"{path}[{index}]"))
            if len(differences) >= 200:
                break
        return differences
    if isinstance(left, float) and isinstance(right, float):
        if left == right:
            return []
        return [{"path": path, "incremental": left, "full": right}]
    if left != right:
        return [{"path": path, "incremental": repr(left), "full": repr(right)}]
    return []
