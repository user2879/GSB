"""Deterministic topic/subscription batching and stable batch identifiers."""

from __future__ import annotations

from dataclasses import dataclass

from .digest import canonical_text, sha256_hex, short_id
from .spec import Channel, Event, Spec, Subscriber


@dataclass(frozen=True)
class Batch:
    batch_id: str
    channel_id: str
    topic: str
    batch_index: int
    subscriber_id: str
    event_seqs: tuple[int, ...]
    payload_summary: dict
    digest: str


def _payload_summary(events: tuple[Event, ...]) -> dict:
    payloads = [event.payload for event in events]
    return {
        "event_count": len(events),
        "first_timestamp_ms": events[0].timestamp_ms,
        "last_timestamp_ms": events[-1].timestamp_ms,
        "payload_min": min(payloads),
        "payload_max": max(payloads),
        "payload_sum": sum(payloads),
    }


def derive_batch_id(channel_id, topic, batch_index, subscriber_id) -> str:
    """Stable 32-char id from an explicit sorted-key canonical document."""
    document = {
        "kind": "gsb.batch_id.v1",
        "channel_id": channel_id,
        "topic": topic,
        "batch_index": batch_index,
        "subscriber_id": subscriber_id,
    }
    return short_id(document, length=32)


def _batch_digest(batch_id, event_seqs, payload_summary, caliber_version):
    document = {
        "kind": "gsb.batch_digest.v1",
        "batch_id": batch_id,
        "event_seqs": list(event_seqs),
        "payload_summary": payload_summary,
        "caliber_version": caliber_version,
    }
    return sha256_hex(document)


def plan_batches(spec: Spec, caliber_version: str = "v1") -> tuple[Batch, ...]:
    """Pure function: identical input always yields byte-identical batches.

    Channels and subscribers are iterated in their declared (stable) order and
    events keep stream order, so the result never depends on dict hashing or on
    the order the scheduler happens to dispatch them.
    """
    planned: list[Batch] = []
    for channel in sorted(spec.channels, key=lambda c: c.channel_id):
        topic_events = tuple(
            event for event in spec.events if event.topic == channel.topic
        )
        for start in range(0, len(topic_events), channel.batch_size):
            window = topic_events[start : start + channel.batch_size]
            batch_index = start // channel.batch_size
            summary = _payload_summary(window)
            seqs = tuple(event.seq for event in window)
            for subscriber in _matching_subscribers(spec, channel):
                batch_id = derive_batch_id(
                    channel.channel_id,
                    channel.topic,
                    batch_index,
                    subscriber.subscriber_id,
                )
                digest = _batch_digest(batch_id, seqs, summary, caliber_version)
                planned.append(
                    Batch(
                        batch_id=batch_id,
                        channel_id=channel.channel_id,
                        topic=channel.topic,
                        batch_index=batch_index,
                        subscriber_id=subscriber.subscriber_id,
                        event_seqs=seqs,
                        payload_summary=summary,
                        digest=digest,
                    )
                )
    return tuple(planned)


def _matching_subscribers(spec: Spec, channel: Channel) -> tuple[Subscriber, ...]:
    matches = [
        subscriber
        for subscriber in spec.subscribers
        if channel.topic in subscriber.topics
    ]
    return tuple(sorted(matches, key=lambda s: s.subscriber_id))


def manifest_text(batches: tuple[Batch, ...]) -> str:
    documents = [
        {
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "channel_id": batch.channel_id,
            "digest": batch.digest,
            "event_seqs": list(batch.event_seqs),
            "payload_summary": batch.payload_summary,
            "subscriber_id": batch.subscriber_id,
            "topic": batch.topic,
        }
        for batch in batches
    ]
    return canonical_text(documents)
