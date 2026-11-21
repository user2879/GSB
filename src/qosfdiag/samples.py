"""Bundled deterministic local samples; no network or external downloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


MAX_RATE_KBPS = 1_000_000


def single_stage_policy() -> dict[str, Any]:
    return {
        "policy_version": "single-stage-v1",
        "max_committed_rate_kbps": MAX_RATE_KBPS,
        "shapers": [
            {
                "shaper_id": "access",
                "committed_rate_kbps": 8000,
                "burst_bytes": 8000,
                "queue_limit_packets": 12,
                "match_dst_port_prefix": 530,
            }
        ],
        "edges": [],
    }


def single_stage_generator() -> dict[str, int]:
    return {
        "seed": 20261122,
        "count": 24,
        "packet_interval_ns": 10_000_000,
        "start_time_ns": 0,
        "min_packet_bytes": 220,
        "max_packet_bytes": 820,
        "dst_port_start": 53000,
        "dst_port_span": 8,
        "batch_size": 4,
    }


def three_stage_policy() -> dict[str, Any]:
    return {
        "policy_version": "three-stage-v1",
        "max_committed_rate_kbps": MAX_RATE_KBPS,
        "shapers": [
            {
                "shaper_id": "edge",
                "committed_rate_kbps": 2000,
                "burst_bytes": 3000,
                "queue_limit_packets": 8,
                "match_dst_port_prefix": 530,
            },
            {
                "shaper_id": "core",
                "committed_rate_kbps": 1200,
                "burst_bytes": 3000,
                "queue_limit_packets": 8,
                "match_dst_port_prefix": 530,
            },
            {
                "shaper_id": "tail",
                "committed_rate_kbps": 600,
                "burst_bytes": 3000,
                "queue_limit_packets": 8,
                "match_dst_port_prefix": 530,
            },
        ],
        "edges": [
            {"upstream": "edge", "downstream": "core"},
            {"upstream": "core", "downstream": "tail"},
        ],
    }


def three_stage_generator() -> dict[str, int]:
    return {
        "seed": 424242,
        "count": 40,
        "packet_interval_ns": 2_000_000,
        "start_time_ns": 0,
        "min_packet_bytes": 400,
        "max_packet_bytes": 1200,
        "dst_port_start": 53000,
        "dst_port_span": 8,
        "batch_size": 4,
    }


def three_stage_near_equal_policy() -> dict[str, Any]:
    policy = copy.deepcopy(three_stage_policy())
    policy["policy_version"] = "three-stage-near-equal-v1"
    for shaper in policy["shapers"]:
        shaper["committed_rate_kbps"] += 1
    return policy


SAMPLE_POLICIES = {
    "single_stage": single_stage_policy,
    "three_stage": three_stage_policy,
    "three_stage_near_equal": three_stage_near_equal_policy,
}

SAMPLE_GENERATORS = {
    "single_stage": single_stage_generator,
    "three_stage": three_stage_generator,
    "three_stage_near_equal": three_stage_generator,
}


def write_sample_files(directory: str | Path) -> tuple[Path, ...]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in ("single_stage", "three_stage", "three_stage_near_equal"):
        policy_path = output / f"{name}_policy.json"
        generator_path = output / f"{name}_generator.json"
        policy_path.write_text(
            json.dumps(SAMPLE_POLICIES[name](), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        generator_path.write_text(
            json.dumps(
                SAMPLE_GENERATORS[name](), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        paths.extend((policy_path, generator_path))
    return tuple(paths)
