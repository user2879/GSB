from __future__ import annotations

from typing import Any, Iterable, Mapping

from .models import AppConfig, Evidence, Turn
from .util import sha256_json


def evidence_content_fingerprint(evidence: Evidence) -> str:
    return sha256_json(
        {
            "kind": "evidence-content",
            "evidence_id": evidence.evidence_id,
            "text": evidence.text,
        }
    )


def evidence_weight_fingerprint(evidence: Evidence) -> str:
    return sha256_json(
        {
            "kind": "evidence-weight",
            "evidence_id": evidence.evidence_id,
            "weight": evidence.weight,
        }
    )


def turn_query_fingerprint(turn: Turn) -> str:
    return sha256_json({"kind": "turn-query", "query": turn.query})


def turn_content_fingerprint(
    turn: Turn, evidence_map: Mapping[str, Evidence]
) -> str:
    declarations = {}
    for channel, evidence_ids in sorted(turn.channels.items()):
        declarations[channel] = [
            {
                "evidence_id": evidence_id,
                "evidence_fingerprint": evidence_content_fingerprint(
                    evidence_map[evidence_id]
                ),
            }
            for evidence_id in evidence_ids
        ]
    return sha256_json(
        {
            "kind": "turn-content",
            "query": turn.query,
            "channels": declarations,
        }
    )


def channel_parameter_fingerprint(
    config: AppConfig, channels: Iterable[str]
) -> str:
    return sha256_json(
        {
            "kind": "channel-parameters",
            "weights": {
                channel: config.channel_weights[channel]
                for channel in sorted(set(channels))
            },
        }
    )


def cache_key(stage: str, payload: Any) -> str:
    return sha256_json({"kind": "stage-cache", "stage": stage, "payload": payload})


def evidence_set_fingerprint(
    dedupe_payload: Mapping[str, Any],
    evidence_map: Mapping[str, Evidence],
) -> str:
    return sha256_json(
        [
            {
                "evidence_id": item["evidence_id"],
                "content": evidence_content_fingerprint(
                    evidence_map[item["evidence_id"]]
                ),
                "authority_channel": item["authority_channel"],
            }
            for item in dedupe_payload["evidence"]
        ]
    )


def topic_assignment_fingerprint(
    *,
    turn_content_fp: str,
    topic_id: str,
    topic_start_content_fp: str,
    evidence_set_fp: str,
    evidence_fps: Iterable[str],
) -> str:
    return sha256_json(
        {
            "kind": "topic-assignment",
            "turn_content_fingerprint": turn_content_fp,
            "topic_id": topic_id,
            "topic_start_content_fingerprint": topic_start_content_fp,
            "evidence_set_fingerprint": evidence_set_fp,
            "evidence_content_fingerprints": sorted(evidence_fps),
        }
    )
