"""Immutable domain objects used by validation, planning, and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import digest_value, stable_serialize


@dataclass(frozen=True)
class Subscriber:
    subscriber_id: str
    host: str
    port: int
    processing_budget_ms: int
    lease_limit: int
    seed: int
    topics: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None

    def accepts_topic(self, topic: str) -> bool:
        return not self.topics or topic in self.topics or "*" in self.topics

    def declaration_content(self) -> dict[str, Any]:
        return {
            "subscriber_id": self.subscriber_id,
            "host": self.host,
            "port": self.port,
            "processing_budget_ms": self.processing_budget_ms,
            "lease_limit": self.lease_limit,
            "seed": self.seed,
            "topics": sorted(self.topics),
        }

    def digest(self) -> str:
        return digest_value(self.declaration_content())


@dataclass(frozen=True)
class Channel:
    channel_id: str
    topic: str
    batch_size: int
    send_rate_limit: int
    subscriber_quota: int
    raw: dict[str, Any] | None = None

    def declaration_content(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "topic": self.topic,
            "batch_size": self.batch_size,
            "send_rate_limit": self.send_rate_limit,
            "subscriber_quota": self.subscriber_quota,
        }


@dataclass(frozen=True)
class Event:
    seq: int
    ts_ms: int
    topic: str
    payload_digest: str
    expected_order: int
    line: int
    raw: dict[str, Any] | None = None

    def batch_content(self) -> dict[str, Any]:
        return {
            "event_id": self.seq,
            "ts_ms": self.ts_ms,
            "topic": self.topic,
            "payload_digest": self.payload_digest,
            "expected_order": self.expected_order,
        }


@dataclass(frozen=True)
class Declaration:
    declaration_version: int
    global_lease_limit: int
    handshake_timeout_ms: int
    subscribers: tuple[Subscriber, ...]
    channels: tuple[Channel, ...]
    source_path: str
    raw: dict[str, Any] | None = None

    def subscriber_map(self) -> dict[str, Subscriber]:
        return {item.subscriber_id: item for item in self.subscribers}

    def channel_map(self) -> dict[str, Channel]:
        return {item.channel_id: item for item in self.channels}

    def channel_by_topic(self) -> dict[str, Channel]:
        return {item.topic: item for item in self.channels}

    def declaration_content(self) -> dict[str, Any]:
        return {
            "declaration_version": self.declaration_version,
            "global_lease_limit": self.global_lease_limit,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "subscribers": [item.declaration_content() for item in self.subscribers],
            "channels": [item.declaration_content() for item in self.channels],
        }

    def spec_digest(self) -> str:
        return digest_value(self.declaration_content())


@dataclass(frozen=True)
class WaitRecord:
    start_mono_ms: float
    end_mono_ms: float
    wait_ms: float
    reason: str


@dataclass(frozen=True)
class QuotaSnapshot:
    global_limit: int
    subscriber_limit: int
    channel_rate_limit: int
    global_active_after_grant: int
    subscriber_active_after_grant: int
    channel_sends_in_window: int

    def as_dict(self) -> dict[str, int]:
        return {
            "global_limit": self.global_limit,
            "subscriber_limit": self.subscriber_limit,
            "channel_rate_limit": self.channel_rate_limit,
            "global_active_after_grant": self.global_active_after_grant,
            "subscriber_active_after_grant": self.subscriber_active_after_grant,
            "channel_sends_in_window": self.channel_sends_in_window,
        }


@dataclass(frozen=True)
class Lease:
    lease_id: str
    batch_id: str
    subscriber_id: str
    channel_id: str
    granted_mono_ms: float
    expires_mono_ms: float
    version: int


@dataclass(frozen=True)
class Acquisition:
    lease: Lease
    wait: WaitRecord
    quota: QuotaSnapshot


@dataclass(frozen=True)
class PlanBatch:
    batch_id: str
    ordinal: int
    channel_id: str
    topic: str
    subscriber_id: str
    batch_index: int
    events: tuple[Event, ...]
    content: dict[str, Any]
    content_digest: str

    @property
    def event_ids(self) -> tuple[int, ...]:
        return tuple(event.seq for event in self.events)

    def canonical_content(self) -> str:
        return stable_serialize(self.content)


@dataclass(frozen=True)
class Ack:
    subscriber_id: str
    batch_id: str
    event_ids: tuple[int, ...]
    processing_ms: int
    ack_digest: str
    processed_order: tuple[int, ...]
    raw: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "subscriber_id": self.subscriber_id,
            "batch_id": self.batch_id,
            "event_ids": list(self.event_ids),
            "processing_ms": self.processing_ms,
            "ack_digest": self.ack_digest,
            "processed_order": list(self.processed_order),
            "protocol_version": int(self.raw.get("protocol_version", 1)),
        }
