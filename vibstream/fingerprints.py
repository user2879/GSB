"""Stage cache-key derivation.

Every key is content-addressed.  Observations ignore clustering parameters;
assignment keys include the observation and the current centroid-state digest;
centroid and segment keys include only their direct dependencies.  This keeps
invalidation precise: changing K does not invalidate raw observations.
"""

from __future__ import annotations

from typing import Any

from .canonical import sha256_value
from .config import AppConfig


def analysis_fingerprint(config: AppConfig) -> str:
    return sha256_value(
        {
            "kind": "analysis",
            "channels": [channel.to_dict() for channel in config.channels],
            "bands": [band.to_dict() for band in config.bands],
        }
    )


def observation_cache_key(
    *,
    window_geometry: dict[str, Any],
    block_fingerprints: list[dict[str, str]],
    config: AppConfig,
) -> str:
    return sha256_value(
        {
            "stage": "observation",
            "geometry": window_geometry,
            "blocks": sorted(block_fingerprints, key=lambda item: item["block_id"]),
            "window_fingerprint": config.window_fingerprint,
            "fixed_point_fingerprint": config.fixed_point_fingerprint,
            "analysis_fingerprint": analysis_fingerprint(config),
        }
    )


def assignment_cache_key(observation_key: str, cluster_state_digest: str, config: AppConfig) -> str:
    return sha256_value(
        {
            "stage": "assignment",
            "observation": observation_key,
            "cluster_state_before": cluster_state_digest,
            "cluster_fingerprint": config.cluster_fingerprint,
            "fixed_point_fingerprint": config.fixed_point_fingerprint,
        }
    )


def centroid_cache_key(cluster_identity: str, member_observation_keys: list[str], config: AppConfig) -> str:
    return sha256_value(
        {
            "stage": "centroid",
            "cluster_identity": cluster_identity,
            "members": sorted(member_observation_keys),
            "cluster_fingerprint": config.cluster_fingerprint,
            "fixed_point_fingerprint": config.fixed_point_fingerprint,
        }
    )


def segment_cache_key(window_segment: list[dict[str, Any]], config: AppConfig) -> str:
    return sha256_value(
        {
            "stage": "segment",
            "windows": window_segment,
            "segment_fingerprint": config.segment.to_dict(),
            "cluster_fingerprint": config.cluster_fingerprint,
            "fixed_point_fingerprint": config.fixed_point_fingerprint,
        }
    )
