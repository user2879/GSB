"""Deterministic topic-to-subscriber batching.

A channel owns exactly one topic.  Its eligible subscribers are the declared
subscribers that accept the topic (``topics`` may be omitted to mean all
topics).  The first ``subscriber_quota`` eligible subscriber identifiers in
lexical order participate.  That rule makes subscription evolution explicit
and avoids relying on JSON object or dictionary traversal order.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .canonical import digest_value, stable_serialize
from .errors import SpecError
from .models import Channel, Declaration, Event, PlanBatch


BATCH_SCHEMA = "eventbus.batch.v1"


@dataclass(frozen=True)
class SubscriptionPlan:
    batches: tuple[PlanBatch, ...]
    subscribers_by_channel: dict[str, tuple[str, ...]]
    subscribers_by_topic: dict[str, tuple[str, ...]]

    def by_id(self) -> dict[str, PlanBatch]:
        return {batch.batch_id: batch for batch in self.batches}

    def digest(self) -> str:
        return digest_value(
            [
                {
                    "batch_id": batch.batch_id,
                    "ordinal": batch.ordinal,
                    "content_digest": batch.content_digest,
                }
                for batch in self.batches
            ]
        )


def eligible_subscribers(
    declaration: Declaration, channel: Channel
) -> tuple[str, ...]:
    return tuple(
        sorted(
            subscriber.subscriber_id
            for subscriber in declaration.subscribers
            if subscriber.accepts_topic(channel.topic)
        )
    )


def selected_subscribers(
    declaration: Declaration, channel: Channel
) -> tuple[str, ...]:
    return eligible_subscribers(declaration, channel)[: channel.subscriber_quota]


def batch_identity(channel_id: str, topic: str, batch_index: int, subscriber_id: str) -> dict:
    return {
        "schema": "eventbus.batch-id.v1",
        "channel_id": channel_id,
        "topic": topic,
        "batch_index": int(batch_index),
        "subscriber_id": subscriber_id,
    }


def derive_batch_id(channel_id: str, topic: str, batch_index: int, subscriber_id: str) -> str:
    identity = batch_identity(channel_id, topic, batch_index, subscriber_id)
    return "b1_" + digest_value(identity)


def _chunked(items: list[Event], size: int) -> Iterable[tuple[int, list[Event]]]:
    for index, offset in enumerate(range(0, len(items), size)):
        yield index, items[offset : offset + size]


def build_plan(declaration: Declaration, events: Iterable[Event]) -> SubscriptionPlan:
    events = tuple(events)
    events_by_topic: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        events_by_topic[event.topic].append(event)
    for topic_events in events_by_topic.values():
        topic_events.sort(key=lambda item: item.seq)

    subscribers_by_channel: dict[str, tuple[str, ...]] = {}
    subscribers_by_topic: dict[str, tuple[str, ...]] = {}
    drafts: list[tuple[tuple[str, str, str, int], PlanBatch]] = []
    subscriber_map = declaration.subscriber_map()

    for channel in sorted(declaration.channels, key=lambda item: item.channel_id):
        recipients = selected_subscribers(declaration, channel)
        if not recipients:
            raise SpecError(
                f"channel {channel.channel_id!r} has no selected subscribers",
                field="subscriber_quota",
            )
        subscribers_by_channel[channel.channel_id] = recipients
        subscribers_by_topic[channel.topic] = recipients
        topic_events = sorted(events_by_topic.get(channel.topic, []), key=lambda item: item.seq)

        for subscriber_id in recipients:
            subscriber = subscriber_map[subscriber_id]
            for batch_index, chunk in _chunked(topic_events, channel.batch_size):
                if not chunk:
                    continue
                batch_id = derive_batch_id(
                    channel.channel_id, channel.topic, batch_index, subscriber_id
                )
                content = {
                    "schema": BATCH_SCHEMA,
                    "batch_id": batch_id,
                    "channel_id": channel.channel_id,
                    "topic": channel.topic,
                    "batch_index": batch_index,
                    "subscriber_id": subscriber_id,
                    "processing_contract": {
                        "seed": subscriber.seed,
                        "processing_budget_ms": subscriber.processing_budget_ms,
                        "ordering": "expected_order_then_event_id",
                    },
                    "event_ids": [event.seq for event in chunk],
                    "events": [event.batch_content() for event in chunk],
                    "canonical_serialization": "json-sort-keys-compact-v1",
                }
                sort_key = (channel.channel_id, channel.topic, subscriber_id, batch_index)
                drafts.append(
                    (
                        sort_key,
                        PlanBatch(
                            batch_id=batch_id,
                            ordinal=-1,
                            channel_id=channel.channel_id,
                            topic=channel.topic,
                            subscriber_id=subscriber_id,
                            batch_index=batch_index,
                            events=tuple(chunk),
                            content=content,
                            content_digest=digest_value(content),
                        ),
                    )
                )

    drafts.sort(key=lambda item: item[0])
    batches = tuple(
        PlanBatch(
            batch_id=batch.batch_id,
            ordinal=ordinal,
            channel_id=batch.channel_id,
            topic=batch.topic,
            subscriber_id=batch.subscriber_id,
            batch_index=batch.batch_index,
            events=batch.events,
            content=batch.content,
            content_digest=batch.content_digest,
        )
        for ordinal, (_, batch) in enumerate(drafts)
    )
    plan = SubscriptionPlan(
        batches=batches,
        subscribers_by_channel=subscribers_by_channel,
        subscribers_by_topic=subscribers_by_topic,
    )
    verify_plan_coverage(plan, declaration, events)
    return plan


def verify_plan_coverage(
    plan: SubscriptionPlan,
    declaration: Declaration,
    events: Iterable[Event],
) -> None:
    events = tuple(events)
    batch_ids: set[str] = set()
    seen_pairs: set[tuple[str, str, int]] = set()

    for batch in plan.batches:
        if batch.batch_id in batch_ids:
            raise SpecError(
                f"duplicate derived batch_id {batch.batch_id}", field="batch_id"
            )
        batch_ids.add(batch.batch_id)
        if not batch.events:
            raise SpecError(f"batch {batch.batch_id} is empty", field="batch_size")
        seqs = [event.seq for event in batch.events]
        if seqs != sorted(seqs):
            raise SpecError(
                f"batch {batch.batch_id} events are not sorted by seq", field="seq"
            )
        if len(seqs) != len(set(seqs)):
            raise SpecError(
                f"batch {batch.batch_id} contains duplicate event ids", field="seq"
            )
        for seq in seqs:
            key = (batch.channel_id, batch.subscriber_id, seq)
            if key in seen_pairs:
                raise SpecError(
                    f"event {seq} occurs more than once for subscriber {batch.subscriber_id}",
                    field="seq",
                )
            seen_pairs.add(key)

    channel_map = declaration.channel_map()
    for channel_id, recipients in plan.subscribers_by_channel.items():
        channel = channel_map[channel_id]
        topic_event_ids = sorted(
            event.seq for event in events if event.topic == channel.topic
        )
        for subscriber_id in recipients:
            actual: list[int] = []
            for batch in plan.batches:
                if (
                    batch.channel_id == channel_id
                    and batch.subscriber_id == subscriber_id
                ):
                    actual.extend(batch.event_ids)
            if sorted(actual) != topic_event_ids:
                missing = sorted(set(topic_event_ids) - set(actual))
                extra = sorted(set(actual) - set(topic_event_ids))
                raise SpecError(
                    "batch plan coverage mismatch",
                    field="batch_size",
                    details={
                        "channel_id": channel_id,
                        "subscriber_id": subscriber_id,
                        "missing_event_ids": missing,
                        "extra_event_ids": extra,
                    },
                )


def plan_signature(plan: SubscriptionPlan) -> str:
    return stable_serialize(
        {
            "schema": "eventbus.plan.v1",
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "ordinal": batch.ordinal,
                    "content_digest": batch.content_digest,
                    "event_ids": list(batch.event_ids),
                }
                for batch in plan.batches
            ],
        }
    )
