"""Streaming fixed-point KMeans with explicit eviction reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import sha256_value
from .config import AppConfig
from .fingerprints import assignment_cache_key, centroid_cache_key


def _round_centroid(values: list[int], count: int) -> list[float]:
    if count <= 0:
        return [0.0 for _ in values]
    return [round(value / count, 6) for value in values]


@dataclass
class ClusterState:
    cluster_id: str
    slot: int
    status: str = "active"
    count: int = 0
    integer_sum: list[int] = field(default_factory=list)
    member_windows: set[str] = field(default_factory=set)
    merged_into: str | None = None
    deducted_count: int = 0
    deducted_integer_sum: list[int] = field(default_factory=list)

    def ensure_width(self, width: int) -> None:
        if not self.integer_sum:
            self.integer_sum = [0] * width
        if not self.deducted_integer_sum:
            self.deducted_integer_sum = [0] * width

    def centroid(self) -> list[float]:
        return _round_centroid(self.integer_sum, self.count)

    def to_dict(self, config: AppConfig, member_observation_keys: list[str]) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "slot": self.slot,
            "status": self.status,
            "merged_into": self.merged_into,
            "member_count": self.count,
            "integer_sum": list(self.integer_sum),
            "centroid": self.centroid(),
            "member_windows": sorted(self.member_windows),
            "deducted_count": self.deducted_count,
            "deducted_integer_sum": list(self.deducted_integer_sum),
            "cache_key": centroid_cache_key(self.cluster_id, member_observation_keys, config),
        }


def _state_digest(clusters: Iterable[ClusterState]) -> str:
    payload = [
        {
            "id": cluster.cluster_id,
            "status": cluster.status,
            "count": cluster.count,
            "sum": cluster.integer_sum,
        }
        for cluster in sorted(clusters, key=lambda item: item.cluster_id)
        if cluster.status == "active"
    ]
    return sha256_value({"kind": "cluster-state", "clusters": payload})


def _distance_to_centroid(observation: list[int], cluster: ClusterState) -> float:
    if cluster.count <= 0:
        return float("inf")
    count = float(cluster.count)
    total = 0.0
    for value, accumulator in zip(observation, cluster.integer_sum):
        difference = float(value) - float(accumulator) / count
        total += difference * difference
    return round(total, 12)


def _nearest_cluster(observation: list[int], clusters: Iterable[ClusterState]) -> ClusterState | None:
    active = [cluster for cluster in clusters if cluster.status == "active" and cluster.count > 0]
    if not active:
        return None
    return min(active, key=lambda cluster: (_distance_to_centroid(observation, cluster), cluster.cluster_id))


def _distance_between_clusters(left: ClusterState, right: ClusterState) -> float:
    if left.count <= 0 or right.count <= 0:
        return float("inf")
    total = 0.0
    for left_sum, right_sum in zip(left.integer_sum, right.integer_sum):
        difference = float(left_sum) / float(left.count) - float(right_sum) / float(right.count)
        total += difference * difference
    return round(total, 12)


def _cluster_prefix(config: AppConfig) -> str:
    return f"c{config.clustering.count}-{config.cluster_fingerprint[:6]}"


def _make_cluster(prefix: str, slot: int, width: int) -> ClusterState:
    cluster = ClusterState(cluster_id=f"{prefix}-s{slot:02d}", slot=slot)
    cluster.ensure_width(width)
    return cluster


def _transfer(source: ClusterState, target: ClusterState, assignments: dict[str, str]) -> None:
    for window_id in sorted(source.member_windows):
        assignments[window_id] = target.cluster_id
    target.member_windows.update(source.member_windows)
    target.count += source.count
    target.integer_sum = [left + right for left, right in zip(target.integer_sum, source.integer_sum)]
    target.deducted_count += source.deducted_count
    target.deducted_integer_sum = [
        left + right for left, right in zip(target.deducted_integer_sum, source.deducted_integer_sum)
    ]
    source.status = "merged"
    source.merged_into = target.cluster_id
    source.member_windows.clear()
    source.count = 0
    source.integer_sum = [0 for _ in source.integer_sum]


def run_streaming_clustering(
    *,
    introductions: Iterable[dict[str, Any]],
    evictions: Iterable[dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    retained_window_ids: set[str],
    config: AppConfig,
) -> dict[str, Any]:
    width = len(next(iter(observations.values()))["integer_values"]) if observations else 0
    prefix = _cluster_prefix(config)
    clusters: list[ClusterState] = [
        _make_cluster(prefix, slot, width) for slot in range(config.clustering.count)
    ]
    clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    assignments: dict[str, str] = {}
    assignment_keys: dict[str, str] = {}
    trajectory: list[dict[str, Any]] = []
    deduction_records: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []
    annihilation_records: list[dict[str, Any]] = []
    seeded = False
    next_slot = config.clustering.count
    event_number = 0
    introductions_by_block: dict[str, list[dict[str, Any]]] = {}
    evictions_by_block: dict[str, list[dict[str, Any]]] = {}
    for event in introductions:
        introductions_by_block.setdefault(event["trigger_block"], []).append(event)
    for event in evictions:
        evictions_by_block.setdefault(event["trigger_block"], []).append(event)

    def active_clusters() -> list[ClusterState]:
        return [cluster for cluster in clusters if cluster.status == "active"]

    def choose_cluster(observation: list[int]) -> tuple[ClusterState | None, bool]:
        nonlocal seeded, next_slot
        empty_slots = [cluster for cluster in active_clusters() if cluster.count == 0]
        if not seeded and empty_slots:
            if len(empty_slots) == 1:
                seeded = True
            return min(empty_slots, key=lambda item: item.slot), True
        nearest = _nearest_cluster(observation, active_clusters())
        if nearest is not None:
            seeded = True
            return nearest, False
        cluster = _make_cluster(prefix, next_slot, width)
        next_slot += 1
        clusters.append(cluster)
        clusters_by_id[cluster.cluster_id] = cluster
        seeded = True
        return cluster, True

    def record_add(cluster: ClusterState, window_id: str, before_count: int, before_sum: list[int], trigger: str) -> None:
        nonlocal event_number
        event_number += 1
        trajectory.append(
            {
                "event": "add_window",
                "event_number": event_number,
                "window_id": window_id,
                "cluster_id": cluster.cluster_id,
                "trigger_block": trigger,
                "member_count_before": before_count,
                "member_count_after": cluster.count,
                "integer_sum_before": before_sum,
                "integer_sum_after": list(cluster.integer_sum),
                "centroid_after": cluster.centroid(),
                "added_window_ids": [window_id],
            }
        )

    def reconcile_low_cluster(cluster: ClusterState, trigger_window: str, trigger_block: str) -> None:
        if cluster.status != "active" or cluster.count >= config.clustering.min_members:
            return
        others = [item for item in active_clusters() if item is not cluster and item.count > 0]
        if cluster.count == 0 or (cluster.count < config.clustering.min_members and others):
            if cluster.count == 0:
                cluster.status = "annihilated"
                annihilation_records.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "trigger_window": trigger_window,
                        "trigger_block": trigger_block,
                        "member_count": 0,
                        "deducted_count": cluster.deducted_count,
                        "cumulated_deduction": list(cluster.deducted_integer_sum),
                        "basis": "member_count_below_minimum_after_eviction",
                    }
                )
                return
            target = min(
                others,
                key=lambda item: (
                    _distance_between_clusters(cluster, item),
                    item.cluster_id,
                ),
            )
            merge_records.append(
                {
                    "source_cluster_id": cluster.cluster_id,
                    "target_cluster_id": target.cluster_id,
                    "trigger_window": trigger_window,
                    "trigger_block": trigger_block,
                    "source_member_count": cluster.count,
                    "deducted_count": cluster.deducted_count,
                    "cumulated_deduction": list(cluster.deducted_integer_sum),
                    "basis": "nearest_l1_to_adjacent_cluster_after_low_member_count",
                }
            )
            transferred = sorted(cluster.member_windows)
            _transfer(cluster, target, assignments)
            event_number_local = len(trajectory) + 1
            trajectory.append(
                {
                    "event": "merge_cluster",
                    "event_number": event_number_local,
                    "cluster_id": cluster.cluster_id,
                    "target_cluster_id": target.cluster_id,
                    "trigger_window": trigger_window,
                    "trigger_block": trigger_block,
                    "transferred_window_ids": transferred,
                    "deducted_count": cluster.deducted_count,
                    "cumulated_deduction": list(cluster.deducted_integer_sum),
                }
            )
        elif cluster.count > 0 and not others:
            cluster.status = "active"
            trajectory.append(
                {
                    "event": "floor_exception",
                    "event_number": len(trajectory) + 1,
                    "cluster_id": cluster.cluster_id,
                    "trigger_window": trigger_window,
                    "trigger_block": trigger_block,
                    "member_count": cluster.count,
                    "basis": "sole_active_cluster_cannot_merge",
                }
            )

    blocks = sorted(
        set(introductions_by_block) | set(evictions_by_block),
        key=lambda block_id: (
            min(
                [event["trigger_arrival_seq"] for event in introductions_by_block.get(block_id, [])]
                + [event["trigger_arrival_seq"] for event in evictions_by_block.get(block_id, [])]
            ),
            block_id,
        ),
    )
    for block_id in blocks:
        for event in introductions_by_block.get(block_id, []):
            window_id = event["window_id"]
            observation = observations[window_id]["integer_values"]
            cluster, _seeded_here = choose_cluster(observation)
            assert cluster is not None
            cluster.ensure_width(width)
            before_count = cluster.count
            before_sum = list(cluster.integer_sum)
            state_key = _state_digest(clusters)
            assignment_key = assignment_cache_key(observations[window_id]["cache_key"], state_key, config)
            assignments[window_id] = cluster.cluster_id
            assignment_keys[window_id] = assignment_key
            cluster.count += 1
            cluster.integer_sum = [
                left + right for left, right in zip(cluster.integer_sum, observation)
            ]
            cluster.member_windows.add(window_id)
            record_add(cluster, window_id, before_count, before_sum, block_id)

        for event in evictions_by_block.get(block_id, []):
            window_id = event["window_id"]
            if window_id not in assignments:
                continue
            cluster = clusters_by_id[assignments[window_id]]
            if cluster.status != "active" or window_id not in cluster.member_windows:
                continue
            observation = observations[window_id]["integer_values"]
            cluster.member_windows.remove(window_id)
            cluster.count -= 1
            cluster.integer_sum = [
                left - right for left, right in zip(cluster.integer_sum, observation)
            ]
            cluster.deducted_count += 1
            cluster.deducted_integer_sum = [
                left + right for left, right in zip(cluster.deducted_integer_sum, observation)
            ]
            event_number += 1
            deduction_records.append(
                {
                    "window_id": window_id,
                    "window_index": event["window_index"],
                    "cluster_id": cluster.cluster_id,
                    "trigger_block": block_id,
                    "deducted_integer_values": list(observation),
                    "member_count_after": cluster.count,
                    "integer_sum_after": list(cluster.integer_sum),
                    "cluster_deducted_count": cluster.deducted_count,
                    "cluster_cumulated_deduction": list(cluster.deducted_integer_sum),
                }
            )
            trajectory.append(
                {
                    "event": "evict_window",
                    "event_number": event_number,
                    "window_id": window_id,
                    "cluster_id": cluster.cluster_id,
                    "trigger_block": block_id,
                    "member_count_after": cluster.count,
                    "integer_sum_after": list(cluster.integer_sum),
                    "centroid_after": cluster.centroid(),
                    "removed_window_ids": [window_id],
                }
            )
            reconcile_low_cluster(cluster, window_id, block_id)

    for cluster in active_clusters():
        if cluster.count == 0:
            cluster.status = "annihilated"
            annihilation_records.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "trigger_window": None,
                    "trigger_block": None,
                    "member_count": 0,
                    "deducted_count": cluster.deducted_count,
                    "cumulated_deduction": list(cluster.deducted_integer_sum),
                    "basis": "unseeded_cluster_after_stream_end",
                }
            )

    retained_ids = set(retained_window_ids)
    final_assignments = {
        window_id: assignments[window_id]
        for window_id in sorted(retained_ids)
        if window_id in assignments
    }
    final_cluster_ids = sorted({cluster_id for cluster_id in final_assignments.values()})
    for cluster_id in final_cluster_ids:
        cluster = clusters_by_id[cluster_id]
        expected_members = {
            window_id
            for window_id, assigned in final_assignments.items()
            if assigned == cluster_id
        }
        if cluster.member_windows != expected_members:
            raise RuntimeError(f"cluster/window correspondence mismatch for {cluster_id}")

    member_obs_keys_by_cluster: dict[str, list[str]] = {}
    for window_id, cluster_id in final_assignments.items():
        member_obs_keys_by_cluster.setdefault(cluster_id, []).append(observations[window_id]["cache_key"])

    cluster_dicts = [
        cluster.to_dict(config, member_obs_keys_by_cluster.get(cluster.cluster_id, []))
        for cluster in sorted(clusters, key=lambda item: item.cluster_id)
    ]
    return {
        "clusters": cluster_dicts,
        "assignments": final_assignments,
        "assignment_cache_keys": {window_id: assignment_keys[window_id] for window_id in sorted(final_assignments)},
        "centroid_trajectory": trajectory,
        "deduction_records": deduction_records,
        "merge_records": merge_records,
        "annihilation_records": annihilation_records,
        "member_observation_keys": {
            cluster_id: sorted(keys)
            for cluster_id, keys in sorted(member_obs_keys_by_cluster.items())
        },
    }
