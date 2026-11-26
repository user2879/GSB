"""Canonical staged materializer.

The same function is used for an incremental command's final state and for a
from-blank replay.  It fixes phase order, ordering keys, cache-key derivation,
and scheduler ordering in one place.
"""

from __future__ import annotations

from typing import Any, Iterable

from .blocks import SignalBlock
from .cache import CacheRegistry
from .canonical import sha256_value
from .clustering import run_streaming_clustering
from .config import AppConfig
from .features import observe_window
from .fingerprints import observation_cache_key
from .scheduler import BatchScheduler, ScheduledTask
from .segments import build_segments
from .windows import WindowBuild, build_windows


def _observation_payloads(
    window_build: WindowBuild,
    config: AppConfig,
    blocks: Iterable[SignalBlock],
    cache: CacheRegistry,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    block_fingerprints = {block.block_id: block.content_fingerprint for block in blocks}
    reused_before: list[str] = []
    observations: dict[str, dict[str, Any]] = {}
    for window in window_build.candidate_windows:
        fingerprints = [
            {"block_id": block_id, "fingerprint": block_fingerprints[block_id]}
            for block_id in sorted(window.block_refs)
        ]
        key = observation_cache_key(
            window_geometry=window.geometry(config),
            block_fingerprints=fingerprints,
            config=config,
        )
        if cache.has(key):
            observations[window.window_id] = cache.get(key)
            reused_before.append(key)
            continue
        payload = observe_window(
            window,
            config,
            window_build.sample_map,
            block_fingerprints,
        )
        observations[window.window_id] = payload
        cache.put(
            stage="observation",
            key=payload["cache_key"],
            payload=payload,
            block_refs=window.block_refs,
            window_refs=(window.window_id,),
            config_tags=("window", "fixed-point", "analysis", "bands"),
        )
    return observations, reused_before


def materialize(
    *,
    blocks: Iterable[SignalBlock],
    config: AppConfig,
    cache: CacheRegistry | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache = cache or CacheRegistry()
    initial_cache_keys = set(cache.entries)
    ordered_blocks = sorted(blocks, key=lambda item: (item.arrival_seq, item.block_id))
    window_build = build_windows(ordered_blocks, config)
    observations, observation_reused = _observation_payloads(
        window_build, config, ordered_blocks, cache
    )

    cluster_result = run_streaming_clustering(
        introductions=window_build.introductions,
        evictions=window_build.evictions,
        observations=observations,
        retained_window_ids={window.window_id for window in window_build.retained_windows},
        config=config,
    )

    assignment_reused: list[str] = []
    for window_id, assignment_key in cluster_result["assignment_cache_keys"].items():
        observation = observations[window_id]
        if assignment_key in initial_cache_keys:
            assignment_reused.append(assignment_key)
        cache.put(
            stage="assignment",
            key=assignment_key,
            payload={"window_id": window_id, "cluster_id": cluster_result["assignments"][window_id]},
            dependencies=(observation["cache_key"],),
            block_refs=tuple(
                item["block_id"] for item in observations[window_id]["block_fingerprints"]
            ),
            window_refs=(window_id,),
            cluster_refs=(cluster_result["assignments"][window_id],),
            config_tags=("clustering", "fixed-point"),
        )

    centroid_reused: list[str] = []
    for cluster in cluster_result["clusters"]:
        key = cluster["cache_key"]
        if key in initial_cache_keys:
            centroid_reused.append(key)
        cache.put(
            stage="centroid",
            key=key,
            payload=cluster,
            dependencies=tuple(
                observations[window_id]["cache_key"]
                for window_id in cluster["member_windows"]
                if window_id in observations
            ),
            window_refs=tuple(cluster["member_windows"]),
            cluster_refs=(cluster["cluster_id"],),
            config_tags=("clustering", "fixed-point"),
        )

    retained_windows = list(window_build.retained_windows)
    segments, smoothing_records = build_segments(
        retained_windows=retained_windows,
        assignments=cluster_result["assignments"],
        observations=observations,
        config=config,
    )
    segment_reused: list[str] = []
    for segment in segments:
        if segment["cache_key"] in initial_cache_keys:
            segment_reused.append(segment["cache_key"])
        cache.put(
            stage="segment",
            key=segment["cache_key"],
            payload=segment,
            dependencies=tuple(
                cluster_result["assignment_cache_keys"][window_id]
                for window_id in segment["window_ids"]
                if window_id in cluster_result["assignment_cache_keys"]
            ),
            block_refs=tuple(segment["block_refs"]),
            window_refs=tuple(segment["window_ids"]),
            cluster_refs=(segment["cluster_id"],),
            config_tags=("segment", "clustering", "fixed-point"),
        )

    retained_obs = [
        observations[window.window_id]
        for window in sorted(retained_windows, key=lambda item: item.index)
    ]
    window_by_id = {window.window_id: window for window in retained_windows}
    tasks = [
        ScheduledTask(
            window_id=observation["window_id"],
            anchor_arrival_seq=window_by_id[observation["window_id"]].anchor_arrival_seq,
            window_index=window_by_id[observation["window_id"]].index,
            point_cost=sum(
                end - start
                for start, end in zip(
                    window_by_id[observation["window_id"]].start_sample_by_channel.values(),
                    window_by_id[observation["window_id"]].end_sample_by_channel.values(),
                )
            ),
            cluster_id=cluster_result["assignments"].get(observation["window_id"]),
        )
        for observation in retained_obs
    ]
    scheduler = BatchScheduler(
        batch_point_budget=config.scheduler.batch_point_budget,
        cluster_point_quota=config.scheduler.cluster_point_quota,
        pending=tasks,
    )
    scheduler_result = scheduler.run_batch()

    roots = {
        observation["cache_key"] for observation in retained_obs
    } | set(cluster_result["assignment_cache_keys"].values()) | {
        cluster["cache_key"] for cluster in cluster_result["clusters"]
    } | {segment["cache_key"] for segment in segments}
    cache.prune_to(roots)

    cache_reused = sorted(
        set(observation_reused + assignment_reused + centroid_reused + segment_reused)
    )
    state = {
        "schema_version": 1,
        "config_hash": config.config_hash,
        "config": config.to_dict(),
        "phases": {
            "window": {
                "retained_windows": [window.to_dict() for window in retained_windows],
                "retained_window_count": len(retained_windows),
                "candidate_window_count": len(window_build.candidate_windows),
                "introductions": list(window_build.introductions),
                "evictions": list(window_build.evictions),
            },
            "observation": {
                "observations": retained_obs,
                "observation_count": len(retained_obs),
            },
            "cluster": cluster_result,
            "segment": {
                "segments": segments,
                "smoothing_records": smoothing_records,
                "segment_count": len(segments),
            },
            "scheduler": scheduler.to_dict(),
        },
        "dataset": {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "arrival_seq": block.arrival_seq,
                    "acquisition_ms": block.acquisition_ms,
                    "content_fingerprint": block.content_fingerprint,
                    "sample_counts": {
                        channel: len(values) for channel, values in sorted(block.signals.items())
                    },
                }
                for block in ordered_blocks
            ],
        },
        "cache": {
            "manifest": cache.current_manifest(),
            "keys": sorted(cache.entries),
            "dependency_manifest": cache.dependencies_manifest(),
            "reuse_links": sorted(
                [
                    {
                        "consumer": key,
                        "provider": dependency,
                    }
                    for key, entry in cache.entries.items()
                    for dependency in entry.dependencies
                ],
                key=lambda item: (item["consumer"], item["provider"]),
            ),
        },
        "failures": scheduler_result["errors"],
        "canonical_digest": "",
    }
    digest_payload = {key: value for key, value in state.items() if key != "canonical_digest"}
    state["canonical_digest"] = sha256_value(digest_payload)
    state["_runtime"] = {"reused_cache_keys": cache_reused}
    return state
