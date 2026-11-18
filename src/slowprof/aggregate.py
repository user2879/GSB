"""Full-fold and fixed-depth streaming aggregation paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .models import (
    ContextStat,
    FrameStat,
    Interval,
    PathSelection,
    ProfilePolicy,
    Sample,
    choose_path,
)
from .ndjson import NDJSONMetadata, iter_validated_samples, scan_ndjson


@dataclass(frozen=True, slots=True)
class ContextEdge:
    target_frame: str
    relation: str
    context_frame: str
    inclusive_time_ms: int
    participant_weight: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_frame": self.target_frame,
            "relation": self.relation,
            "context_frame": self.context_frame,
            "inclusive_time_ms": self.inclusive_time_ms,
            "participant_weight": self.participant_weight,
        }


@dataclass(frozen=True, slots=True)
class AggregationResult:
    selection: PathSelection
    frame_stats: tuple[FrameStat, ...]
    context_edges: tuple[ContextEdge, ...]
    total_weight: int
    total_time_ms: int
    folded_stacks: tuple[tuple[tuple[str, ...], int], ...] | None
    memory_estimate: dict[str, Any]
    metadata: NDJSONMetadata | None = None

    def frame_stat_map(self) -> dict[str, FrameStat]:
        return {stat.frame: stat for stat in self.frame_stats}

    def context_map(self) -> dict[str, dict[str, tuple[ContextStat, ...]]]:
        grouped: dict[str, dict[str, list[ContextStat]]] = {}
        for edge in self.context_edges:
            by_relation = grouped.setdefault(edge.target_frame, {"callers": [], "callees": []})
            key = "callers" if edge.relation == "caller" else "callees"
            by_relation[key].append(
                ContextStat(
                    frame=edge.context_frame,
                    inclusive_time_ms=edge.inclusive_time_ms,
                    participant_weight=edge.participant_weight,
                )
            )
        return {
            target: {
                relation: tuple(
                    sorted(
                        stats,
                        key=lambda item: (
                            -item.inclusive_time_ms,
                            -item.participant_weight,
                            item.frame,
                        ),
                    )
                )
                for relation, stats in relations.items()
            }
            for target, relations in grouped.items()
        }


class _ContextAccumulator:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str, str], list[int]] = {}

    def add(self, target: str, relation: str, context: str, weighted_duration: int, weight: int) -> None:
        key = (target, relation, context)
        value = self._values.get(key)
        if value is None:
            self._values[key] = [weighted_duration, weight]
        else:
            value[0] += weighted_duration
            value[1] += weight

    def finish(self) -> tuple[ContextEdge, ...]:
        return tuple(
            ContextEdge(
                target_frame=target,
                relation=relation,
                context_frame=context,
                inclusive_time_ms=value[0],
                participant_weight=value[1],
            )
            for (target, relation, context), value in sorted(self._values.items())
        )


def _add_context_for_edges(
    accumulator: _ContextAccumulator,
    edges: Iterable[tuple[str, str]],
    weighted_duration: int,
    weight: int,
) -> None:
    for caller, callee in edges:
        accumulator.add(callee, "caller", caller, weighted_duration, weight)
        accumulator.add(caller, "callee", callee, weighted_duration, weight)


def _stats_from_dicts(
    frame_names: Iterable[str],
    self_time: dict[str, int],
    cumulative: dict[str, int],
    participant: dict[str, int],
    occurrences: dict[str, int],
) -> tuple[FrameStat, ...]:
    return tuple(
        FrameStat(
            frame=frame,
            self_time_ms=self_time.get(frame, 0),
            cumulative_time_ms=cumulative.get(frame, 0),
            participant_weight=participant.get(frame, 0),
            occurrence_count=occurrences.get(frame, 0),
        )
        for frame in sorted(set(frame_names))
    )


def _aggregate_full(samples: Iterable[Sample]) -> tuple[
    tuple[FrameStat, ...],
    tuple[ContextEdge, ...],
    int,
    int,
    tuple[tuple[tuple[str, ...], int], ...],
]:
    folds: dict[tuple[str, ...], int] = {}
    self_time: dict[str, int] = {}
    cumulative: dict[str, int] = {}
    participant: dict[str, int] = {}
    occurrences: dict[str, int] = {}
    contexts = _ContextAccumulator()
    total_weight = 0
    total_time = 0
    all_frames: set[str] = set()

    for sample in samples:
        weighted_duration = sample.weighted_duration_ms
        total_weight += sample.weight
        total_time += weighted_duration
        folds[sample.frames] = folds.get(sample.frames, 0) + sample.weight

        unique_frames = set(sample.frames)
        all_frames.update(unique_frames)
        for frame in unique_frames:
            cumulative[frame] = cumulative.get(frame, 0) + weighted_duration
            participant[frame] = participant.get(frame, 0) + sample.weight
            occurrences[frame] = occurrences.get(frame, 0) + 1
        leaf = sample.frames[-1]
        self_time[leaf] = self_time.get(leaf, 0) + weighted_duration

        edges = set(zip(sample.frames, sample.frames[1:]))
        _add_context_for_edges(contexts, edges, weighted_duration, sample.weight)

    stats = _stats_from_dicts(
        all_frames,
        self_time,
        cumulative,
        participant,
        occurrences,
    )
    folded_stacks = tuple(sorted(folds.items(), key=lambda item: item[0]))
    return stats, contexts.finish(), total_weight, total_time, folded_stacks


def _aggregate_streaming(
    samples: Iterable[Sample],
    *,
    policy: ProfilePolicy,
    frame_names: Sequence[str],
) -> tuple[tuple[FrameStat, ...], tuple[ContextEdge, ...], int, int]:
    names = tuple(sorted(set(frame_names)))
    frame_to_id = {frame: index for index, frame in enumerate(names)}
    frame_count = len(names)
    self_time = [0] * frame_count
    cumulative = [0] * frame_count
    participant = [0] * frame_count
    occurrences = [0] * frame_count
    contexts = _ContextAccumulator()
    total_weight = 0
    total_time = 0

    stack_ids = np.empty(policy.max_depth, dtype=np.int64)

    for sample in samples:
        depth = len(sample.frames)
        weighted_duration = sample.weighted_duration_ms
        total_weight += sample.weight
        total_time += weighted_duration
        for index, frame in enumerate(sample.frames):
            stack_ids[index] = frame_to_id[frame]

        unique_ids = np.unique(stack_ids[:depth])
        for frame_id_value in unique_ids:
            frame_id = int(frame_id_value)
            cumulative[frame_id] += weighted_duration
            participant[frame_id] += sample.weight
            occurrences[frame_id] += 1
        leaf_id = int(stack_ids[depth - 1])
        self_time[leaf_id] += weighted_duration

        edge_ids = {
            (int(caller), int(callee))
            for caller, callee in zip(stack_ids[: depth - 1], stack_ids[1:depth])
        }
        _add_context_for_edges(
            contexts,
            ((names[caller], names[callee]) for caller, callee in edge_ids),
            weighted_duration,
            sample.weight,
        )

    stats = tuple(
        FrameStat(
            frame=names[index],
            self_time_ms=self_time[index],
            cumulative_time_ms=cumulative[index],
            participant_weight=participant[index],
            occurrence_count=occurrences[index],
        )
        for index in range(frame_count)
    )
    return stats, contexts.finish(), total_weight, total_time


def _estimate_memory(
    *,
    path: str,
    unique_stack_count: int,
    frame_count: int,
    frame_name_bytes: int,
    max_depth: int,
    average_depth: int,
    context_edge_count: int,
) -> dict[str, Any]:
    metadata_digest_set = unique_stack_count * (16 + 33)
    frame_registry = frame_name_bytes + frame_count * 49
    frame_arrays = frame_count * 5 * 8
    working_stack = max_depth * 8 + 160
    context_edges = context_edge_count * 160
    full_fold = (
        unique_stack_count * (96 + average_depth * 16) if path == "full" else 0
    )
    components = {
        "metadata_unique_stack_digest_set": metadata_digest_set,
        "frame_registry": frame_registry,
        "frame_stat_arrays": frame_arrays,
        "fixed_depth_working_buffer": working_stack,
        "context_edge_index": context_edges,
        "full_stack_fold_table": full_fold,
    }
    return {
        "estimated_peak_bytes": sum(components.values()),
        "components": components,
        "method": "deterministic structural estimate; excludes interpreter and allocator overhead",
        "streaming_no_path_strings": path == "streaming",
        "fixed_depth_working_buffer_bytes": max_depth * 8,
    }


def _run_aggregation(
    samples: Iterable[Sample],
    *,
    policy: ProfilePolicy,
    selection: PathSelection,
    metadata: NDJSONMetadata | None,
    frame_names: Sequence[str] | None = None,
) -> AggregationResult:
    if selection.path == "full":
        stats, contexts, total_weight, total_time, folded = _aggregate_full(samples)
    else:
        if frame_names is None:
            if metadata is not None:
                frame_names = metadata.frame_names
            else:
                raise ValueError("frame_names are required for in-memory streaming aggregation")
        stats, contexts, total_weight, total_time = _aggregate_streaming(
            samples,
            policy=policy,
            frame_names=frame_names,
        )
        folded = None

    if total_weight != selection.total_weight:
        raise ValueError(
            f"aggregation weight {total_weight} differs from validation weight {selection.total_weight}"
        )
    if sum(stat.self_time_ms for stat in stats) != total_time:
        raise AssertionError("self time must sum to total weighted elapsed time")

    frame_name_bytes = metadata.frame_name_bytes if metadata is not None else sum(
        len(stat.frame.encode("utf-8")) for stat in stats
    )
    average_depth = (
        metadata.selected_total_depth // metadata.selected_row_count
        if metadata is not None and metadata.selected_row_count
        else 0
    )
    memory = _estimate_memory(
        path=selection.path,
        unique_stack_count=selection.unique_stack_count,
        frame_count=len(stats),
        frame_name_bytes=frame_name_bytes,
        max_depth=policy.max_depth,
        average_depth=average_depth,
        context_edge_count=len(contexts),
    )
    return AggregationResult(
        selection=selection,
        frame_stats=stats,
        context_edges=contexts,
        total_weight=total_weight,
        total_time_ms=total_time,
        folded_stacks=folded,
        memory_estimate=memory,
        metadata=metadata,
    )


def aggregate_file(
    path: str | Path,
    policy: ProfilePolicy,
    interval: Interval,
    *,
    force_path: str | None = None,
) -> AggregationResult:
    metadata = scan_ndjson(path, policy, interval)
    selection = choose_path(
        policy,
        metadata.unique_stack_count,
        metadata.selected_total_weight,
        force_path=force_path,
    )
    samples = iter_validated_samples(path, policy, interval, expected=metadata)
    return _run_aggregation(
        samples,
        policy=policy,
        selection=selection,
        metadata=metadata,
        frame_names=metadata.frame_names,
    )


def aggregate_samples(
    samples: Sequence[Sample],
    policy: ProfilePolicy,
    *,
    path: str | None = None,
    unique_stack_count: int | None = None,
) -> AggregationResult:
    """In-memory aggregation primarily for order/concurrency parity tests.

    The same two implementations are used as the file API.
    """

    total_weight = sum(sample.weight for sample in samples)
    if unique_stack_count is None:
        unique_stack_count = len({sample.frames for sample in samples})
    selection = choose_path(
        policy,
        unique_stack_count,
        total_weight,
        force_path=path,
    )
    frame_names = tuple(sorted({frame for sample in samples for frame in sample.frames}))
    metadata = NDJSONMetadata(
        path="<memory>",
        file_sha256="0" * 64,
        selected_samples_sha256="0" * 64,
        line_count=len(samples),
        selected_row_count=len(samples),
        selected_total_weight=total_weight,
        unique_stack_count=unique_stack_count,
        frame_names=frame_names,
        frame_name_bytes=sum(len(frame.encode("utf-8")) for frame in frame_names),
        selected_total_depth=sum(len(sample.frames) for sample in samples),
        selected_max_depth=max((len(sample.frames) for sample in samples), default=0),
        first_timestamp_ms=min((sample.timestamp_ms for sample in samples), default=None),
        last_timestamp_ms=max((sample.timestamp_ms for sample in samples), default=None),
    )
    return _run_aggregation(
        samples,
        policy=policy,
        selection=selection,
        metadata=metadata,
        frame_names=frame_names,
    )
