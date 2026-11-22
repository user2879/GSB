from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import METRICS, MergedResult, Observation, Snapshot


def _stable_sources(observations: Iterable[Observation]) -> list[Observation]:
    ordered = sorted(observations, key=lambda item: (-item.observation_seq, item.agent_id))
    selected: dict[str, Observation] = {}
    for observation in ordered:
        selected.setdefault(observation.agent_id, observation)
    return sorted(selected.values(), key=lambda item: (-item.observation_seq, item.agent_id))


def merge_metric(
    snapshot_seq: int,
    host_id: str,
    metric_id: str,
    observations: Iterable[Observation],
    min_sources: int,
) -> MergedResult:
    sources = _stable_sources(observations)
    source_count = len(sources)
    if source_count < min_sources:
        rendered = [
            {
                "agent_id": observation.agent_id,
                "observation_seq": observation.observation_seq,
                "value": observation.value,
                "aggregation_version": observation.aggregation_version,
                "monotonic_ms": observation.monotonic_ms,
                "deviation": None,
            }
            for observation in sources
        ]
        return MergedResult(
            snapshot_seq=snapshot_seq,
            host_id=host_id,
            metric_id=metric_id,
            status="SPARSE",
            authority_agent=None,
            authority_observation_seq=None,
            authority_value=None,
            aggregation_version=None,
            source_count=source_count,
            sparse=True,
            sources=rendered,
            max_abs_deviation=0,
            error_code="E_SPARSE",
        )

    authority = sources[0]
    rendered = [
        {
            "agent_id": observation.agent_id,
            "observation_seq": observation.observation_seq,
            "value": observation.value,
            "aggregation_version": observation.aggregation_version,
            "monotonic_ms": observation.monotonic_ms,
            "deviation": observation.value - authority.value,
        }
        for observation in sources
    ]
    max_abs_deviation = max(abs(item["deviation"]) for item in rendered)
    return MergedResult(
        snapshot_seq=snapshot_seq,
        host_id=host_id,
        metric_id=metric_id,
        status="AUTHORITATIVE",
        authority_agent=authority.agent_id,
        authority_observation_seq=authority.observation_seq,
        authority_value=authority.value,
        aggregation_version=authority.aggregation_version,
        source_count=source_count,
        sparse=False,
        sources=rendered,
        max_abs_deviation=max_abs_deviation,
    )


def merge_snapshot(
    snapshot: Snapshot,
    observations: Iterable[Observation],
    min_sources: int,
) -> list[MergedResult]:
    by_metric: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.host_id in snapshot.hosts and observation.metric_id in METRICS:
            by_metric[(observation.host_id, observation.metric_id)].append(observation)
    merged: list[MergedResult] = []
    for host_id in sorted(snapshot.hosts):
        for metric_id in METRICS:
            merged.append(
                merge_metric(
                    snapshot.seq,
                    host_id,
                    metric_id,
                    by_metric.get((host_id, metric_id), []),
                    min_sources,
                )
            )
    return merged
