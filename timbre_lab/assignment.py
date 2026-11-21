from __future__ import annotations

from typing import Any

from .config import AnalysisConfig, digest_json
from .model import AudioDocument, Segment
from .quantization import dual_level_reference, similarity


def adjacent_pairs(document: AudioDocument) -> list[tuple[Segment, Segment]]:
    pairs: list[tuple[Segment, Segment]] = []
    for file_name in sorted(document.files):
        segments = document.segments_for_file(file_name)
        for left, right in zip(segments, segments[1:]):
            pairs.append((left, right))
    pairs.sort(key=lambda pair: (pair[0].stable_key(), pair[1].stable_key()))
    return pairs


def assignment_cache_key(
    left: dict[str, Any],
    right: dict[str, Any],
    left_segment: Segment,
    right_segment: Segment,
    config: AnalysisConfig,
) -> str:
    return digest_json(
        {
            "left_description": left["cache_key"],
            "right_description": right["cache_key"],
            "left_segment_id": left_segment.segment_id,
            "right_segment_id": right_segment.segment_id,
            "left_label": left_segment.speaker,
            "right_label": right_segment.speaker,
            "threshold": config.similarity_threshold,
            "comparison": "inclusive",
            "assignment_config": config.assignment_fingerprint(),
            "order": [left_segment.start_frame, right_segment.start_frame],
        }
    )


def decide_assignment(
    left_segment: Segment,
    right_segment: Segment,
    left_description: dict[str, Any],
    right_description: dict[str, Any],
    left_features: dict[str, Any],
    right_features: dict[str, Any],
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
) -> dict[str, Any]:
    score, difference_breakdown = similarity(left_description, right_description, config)
    threshold = config.similarity_threshold
    vector_same = score >= threshold
    boundary = score == threshold
    left_label = left_segment.speaker
    right_label = right_segment.speaker
    annotation_same: bool | None = None
    if left_label is not None and right_label is not None:
        annotation_same = left_label == right_label
    conflict: dict[str, Any] | None = None
    if annotation_same is None:
        final_same = vector_same
        resolution = "vector_fallback"
    else:
        final_same = annotation_same
        resolution = "annotation_priority"
        if annotation_same != vector_same:
            conflict = {
                "code": "E_ANNOTATION_VECTOR_CONFLICT",
                "message": "speaker annotations and vector assignment disagree; annotation is authoritative",
                "left_speaker": left_label,
                "right_speaker": right_label,
                "vector_same": vector_same,
                "annotation_same": annotation_same,
                "fixed_rule": "explicit_annotation_priority_then_start_frame_order",
            }
    boundary_reference = None
    if boundary:
        boundary_reference = {
            "similarity": score,
            "threshold": threshold,
            "inclusive_same": True,
            "exclusive_same": False,
            "decision_rule": "inclusive_and_start_frame_ascending",
            "ordered_segments": [left_segment.segment_id, right_segment.segment_id],
            "ordered_start_frames": [left_segment.start_frame, right_segment.start_frame],
        }
    dual_reference = None
    if boundary or conflict is not None:
        dual_reference = {
            "left": dual_level_reference(left_features, left_description, config, overrides),
            "right": dual_level_reference(right_features, right_description, config, overrides),
        }
    return {
        "pair_id": f"{left_segment.segment_id}->{right_segment.segment_id}",
        "cache_key": assignment_cache_key(left_description, right_description, left_segment, right_segment, config),
        "left_segment_id": left_segment.segment_id,
        "right_segment_id": right_segment.segment_id,
        "left_start_frame": left_segment.start_frame,
        "right_start_frame": right_segment.start_frame,
        "left_speaker": left_label,
        "right_speaker": right_label,
        "similarity": score,
        "threshold": threshold,
        "vector_same": vector_same,
        "annotation_same": annotation_same,
        "final_same": final_same,
        "resolution": resolution,
        "boundary": boundary,
        "boundary_reference": boundary_reference,
        "conflict": conflict,
        "dual_level_reference": dual_reference,
        "difference_breakdown": difference_breakdown,
        "upstream": {
            "left_description": left_description["cache_key"],
            "right_description": right_description["cache_key"],
        },
    }


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


def group_speakers(
    document: AudioDocument,
    assignments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = document.ordered_segments()
    segment_by_id = {segment.segment_id: segment for segment in ordered}
    dsu = _DisjointSet([segment.segment_id for segment in ordered])
    label_sets: dict[str, set[str]] = {}
    for segment in ordered:
        if segment.speaker is not None:
            label_sets[segment.segment_id] = {segment.speaker}
        else:
            label_sets[segment.segment_id] = set()
    by_label: dict[str, list[str]] = {}
    for segment in ordered:
        if segment.speaker is not None:
            by_label.setdefault(segment.speaker, []).append(segment.segment_id)
    for segment_ids in by_label.values():
        base = segment_ids[0]
        for segment_id in segment_ids[1:]:
            root = dsu.union(base, segment_id)
            label_sets[root] = label_sets[base] | label_sets[segment_id]
            label_sets[base] = label_sets[root]

    assignment_conflicts: list[dict[str, Any]] = []
    for assignment in sorted(assignments, key=lambda item: (item["left_start_frame"], item["left_segment_id"])):
        if not assignment["final_same"]:
            continue
        left_root = dsu.find(assignment["left_segment_id"])
        right_root = dsu.find(assignment["right_segment_id"])
        if left_root == right_root:
            continue
        left_labels = label_sets.get(left_root, set())
        right_labels = label_sets.get(right_root, set())
        if left_labels and right_labels and left_labels != right_labels:
            assignment_conflicts.append(
                {
                    "code": "E_SPEAKER_BRIDGE_CONFLICT",
                    "pair_id": assignment["pair_id"],
                    "left_labels": sorted(left_labels),
                    "right_labels": sorted(right_labels),
                    "fixed_rule": "do_not_merge_distinct_explicit_labels",
                }
            )
            continue
        root = dsu.union(left_root, right_root)
        label_sets[root] = left_labels | right_labels

    groups: dict[str, list[str]] = {}
    for segment in ordered:
        root = dsu.find(segment.segment_id)
        groups.setdefault(root, []).append(segment.segment_id)
    result: list[dict[str, Any]] = []
    for segment_ids in groups.values():
        group_segments = sorted((segment_by_id[segment_id] for segment_id in segment_ids), key=lambda item: item.stable_key())
        labels = sorted({segment.speaker for segment in group_segments if segment.speaker is not None})
        if len(labels) == 1:
            speaker_id = labels[0]
        elif len(labels) > 1:
            speaker_id = "conflicted:" + "|".join(labels)
        else:
            speaker_id = f"auto:{group_segments[0].segment_id}"
        result.append(
            {
                "speaker": speaker_id,
                "segment_ids": [segment.segment_id for segment in group_segments],
                "first_stable_key": list(group_segments[0].stable_key()),
                "explicit_labels": labels,
                "conflicted": len(labels) > 1,
            }
        )
    result.sort(key=lambda item: (item["first_stable_key"], item["speaker"]))
    for item in result:
        del item["first_stable_key"]
    return result, assignment_conflicts
