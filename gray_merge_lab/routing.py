from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import sha256_value
from .deterministic import stable_route_bucket
from .specs import RuleSpec, SampleSpec


DESTINATIONS = ("stable", "canary")


def canonical_sample_order(sample: SampleSpec) -> tuple[int, str, str]:
    return (sample.sample_seq, sample.sample_id, sample.client_key)


@dataclass(frozen=True)
class RoutedSample:
    sample: SampleSpec
    bucket: int
    destination: str
    canary_weight_bp: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample.sample_id,
            "sample_seq": self.sample.sample_seq,
            "client_key": self.sample.client_key,
            "rule_id": self.sample.rule_id,
            "bucket": self.bucket,
            "destination": self.destination,
            "canary_weight_bp": self.canary_weight_bp,
        }


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    rule_id: str
    destination: str
    samples: tuple[SampleSpec, ...]
    buckets: tuple[int, ...]
    canary_weight_bp: int
    batch_index: int

    def sample_ids(self) -> list[str]:
        return [item.sample_id for item in self.samples]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "rule_id": self.rule_id,
            "destination": self.destination,
            "sample_ids": self.sample_ids(),
            "buckets": list(self.buckets),
            "canary_weight_bp": self.canary_weight_bp,
        }


def route_sample(sample: SampleSpec, rule: RuleSpec) -> RoutedSample:
    bucket = stable_route_bucket(rule.rule_id, sample.client_key)
    destination = "canary" if bucket <= rule.canary_weight_bp else "stable"
    return RoutedSample(sample, bucket, destination, rule.canary_weight_bp)


def route_samples(samples: Iterable[SampleSpec], rules: Iterable[RuleSpec]) -> list[RoutedSample]:
    rule_map = {rule.rule_id: rule for rule in rules}
    routed: list[RoutedSample] = []
    for sample in sorted(samples, key=canonical_sample_order):
        rule = rule_map.get(sample.rule_id)
        if rule is None:
            continue
        routed.append(route_sample(sample, rule))
    return routed


def plan_batches(
    samples: Iterable[SampleSpec],
    rules: Iterable[RuleSpec],
) -> list[BatchPlan]:
    """Create canonical batches independent of dict/hash iteration or arrival order.

    Samples are ordered by (sample_seq, sample_id, client_key), then partitioned
    by rule, destination and client key, and finally into chunks of the rule's
    declared batch size. Client-homogeneous route batches keep every historical
    h2 receipt's singular client bucket semantically exact.
    """
    rule_map = {rule.rule_id: rule for rule in rules}
    grouped: dict[tuple[str, str, str], list[RoutedSample]] = {}
    for routed in route_samples(samples, rule_map.values()):
        key = (routed.sample.rule_id, routed.destination, routed.sample.client_key)
        grouped.setdefault(key, []).append(routed)

    plans: list[BatchPlan] = []
    for rule_id in sorted(rule_map):
        rule = rule_map[rule_id]
        for destination in DESTINATIONS:
            client_keys = sorted(key[2] for key in grouped if key[0] == rule_id and key[1] == destination)
            for client_key in client_keys:
                items = grouped[(rule_id, destination, client_key)]
                for offset in range(0, len(items), rule.batch_size):
                    chunk = items[offset : offset + rule.batch_size]
                    sample_ids = [entry.sample.sample_id for entry in chunk]
                    identity = {
                        "algorithm": "gray-merge-lab-batch-v1",
                        "rule_id": rule_id,
                        "destination": destination,
                        "client_key": client_key,
                        "sample_ids": sample_ids,
                        "sample_seqs": [entry.sample.sample_seq for entry in chunk],
                        "batch_size": rule.batch_size,
                    }
                    digest = sha256_value(identity).split(":", 1)[1]
                    first_seq = chunk[0].sample.sample_seq
                    last_seq = chunk[-1].sample.sample_seq
                    batch_id = f"b-{rule_id}-{destination}-{first_seq:06d}-{last_seq:06d}-{digest[:10]}"
                    plans.append(
                        BatchPlan(
                            batch_id=batch_id,
                            rule_id=rule_id,
                            destination=destination,
                            samples=tuple(entry.sample for entry in chunk),
                            buckets=tuple(entry.bucket for entry in chunk),
                            canary_weight_bp=rule.canary_weight_bp,
                            batch_index=0,
                        )
                    )
    plans.sort(key=lambda plan: (plan.rule_id, plan.destination, plan.samples[0].sample_seq, plan.batch_id))
    return [
        BatchPlan(
            batch_id=plan.batch_id,
            rule_id=plan.rule_id,
            destination=plan.destination,
            samples=plan.samples,
            buckets=plan.buckets,
            canary_weight_bp=plan.canary_weight_bp,
            batch_index=index,
        )
        for index, plan in enumerate(plans)
    ]


def batch_partition_digest(plans: Iterable[BatchPlan]) -> str:
    return sha256_value(
        [
            {
                "batch_id": plan.batch_id,
                "rule_id": plan.rule_id,
                "destination": plan.destination,
                "sample_ids": plan.sample_ids(),
            }
            for plan in plans
        ]
    )
