from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .models import METRICS, AffectedItem, AgentSpec, Criteria, SnapshotChain


@dataclass
class PropagationResult:
    affected: list[AffectedItem]
    differences: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "affected": [item.to_dict() for item in self.affected],
            "differences": self.differences,
        }


class _AffectedBuilder:
    def __init__(self) -> None:
        self._reasons: dict[tuple[int, str, str], set[str]] = {}

    def add(self, snapshot_seq: int, host_id: str, metric_id: str, reason: str) -> None:
        self._reasons.setdefault((snapshot_seq, host_id, metric_id), set()).add(reason)

    def add_agent(self, chain: SnapshotChain, agent: AgentSpec, reason: str) -> None:
        for snapshot in chain.snapshots.values():
            for host_id in sorted(snapshot.hosts):
                for metric_id in METRICS:
                    if agent.covers(host_id, metric_id):
                        self.add(snapshot.seq, host_id, metric_id, reason)

    def result(self) -> list[AffectedItem]:
        return [
            AffectedItem(
                snapshot_seq=snapshot_seq,
                host_id=host_id,
                metric_id=metric_id,
                reasons=tuple(sorted(reasons)),
            )
            for (snapshot_seq, host_id, metric_id), reasons in sorted(self._reasons.items())
        ]


def derive_affected(
    old_chain: SnapshotChain | None,
    new_chain: SnapshotChain,
    old_agents: Iterable[AgentSpec] | None,
    new_agents: Iterable[AgentSpec],
    old_criteria: Criteria | None,
    new_criteria: Criteria,
) -> PropagationResult:
    builder = _AffectedBuilder()
    differences: list[dict[str, object]] = []
    old_agent_map = {agent.agent_id: agent for agent in (old_agents or [])}
    new_agent_map = {agent.agent_id: agent for agent in new_agents}

    for agent_id in sorted(set(old_agent_map) - set(new_agent_map)):
        agent = old_agent_map[agent_id]
        builder.add_agent(new_chain, agent, f"agent_removed:{agent_id}")
        differences.append({"kind": "agent_removed", "agent_id": agent_id, "basis": "agent set changed"})
    for agent_id in sorted(set(new_agent_map) - set(old_agent_map)):
        agent = new_agent_map[agent_id]
        builder.add_agent(new_chain, agent, f"agent_added:{agent_id}")
        differences.append({"kind": "agent_added", "agent_id": agent_id, "basis": "agent set changed"})

    for agent_id in sorted(set(old_agent_map) & set(new_agent_map)):
        old_agent = old_agent_map[agent_id]
        new_agent = new_agent_map[agent_id]
        if old_agent.aggregation.version != new_agent.aggregation.version:
            reason = f"aggregation_version_changed:{agent_id}:{old_agent.aggregation.version}->{new_agent.aggregation.version}"
            builder.add_agent(new_chain, new_agent, reason)
            differences.append(
                {
                    "kind": "aggregation_version_changed",
                    "agent_id": agent_id,
                    "old": old_agent.aggregation.version,
                    "new": new_agent.aggregation.version,
                    "basis": "observation instrument changed",
                }
            )
        old_coverage = {(host, metric) for host in old_agent.hosts for metric in old_agent.metrics}
        new_coverage = {(host, metric) for host in new_agent.hosts for metric in new_agent.metrics}
        if old_coverage != new_coverage:
            for snapshot in new_chain.snapshots.values():
                for host_id in sorted(snapshot.hosts):
                    for metric_id in METRICS:
                        key = (host_id, metric_id)
                        if key in old_coverage and key not in new_coverage:
                            builder.add(snapshot.seq, host_id, metric_id, f"coverage_removed:{agent_id}")
                        if key in new_coverage and key not in old_coverage:
                            builder.add(snapshot.seq, host_id, metric_id, f"coverage_added:{agent_id}")
            differences.append(
                {
                    "kind": "agent_coverage_changed",
                    "agent_id": agent_id,
                    "old": sorted(f"{host}/{metric}" for host, metric in old_coverage),
                    "new": sorted(f"{host}/{metric}" for host, metric in new_coverage),
                    "basis": "agent host or metric scope changed",
                }
            )
        old_without_aggregation = old_agent.to_dict()
        new_without_aggregation = new_agent.to_dict()
        old_without_aggregation.pop("aggregation", None)
        new_without_aggregation.pop("aggregation", None)
        if old_without_aggregation != new_without_aggregation:
            builder.add_agent(new_chain, new_agent, f"agent_spec_changed:{agent_id}")
            differences.append(
                {
                    "kind": "agent_spec_changed",
                    "agent_id": agent_id,
                    "basis": "seed, profiled samples, listener, or quota changed",
                }
            )

    old_chain = old_chain or SnapshotChain(snapshots={})
    for seq in sorted(set(new_chain.snapshots) - set(old_chain.snapshots)):
        snapshot = new_chain.snapshots[seq]
        for host_id in sorted(snapshot.hosts):
            for metric_id in METRICS:
                builder.add(seq, host_id, metric_id, f"snapshot_added:{seq}")
        differences.append({"kind": "snapshot_added", "snapshot_seq": seq, "basis": "snapshot chain extended"})
    for seq in sorted(set(old_chain.snapshots) - set(new_chain.snapshots)):
        differences.append({"kind": "snapshot_removed", "snapshot_seq": seq, "basis": "declared snapshot chain changed"})

    for seq in sorted(set(old_chain.snapshots) & set(new_chain.snapshots)):
        old_snapshot = old_chain.snapshots[seq]
        new_snapshot = new_chain.snapshots[seq]
        if old_snapshot.parent_seq != new_snapshot.parent_seq:
            for host_id in sorted(set(old_snapshot.hosts) | set(new_snapshot.hosts)):
                for metric_id in METRICS:
                    builder.add(seq, host_id, metric_id, f"snapshot_parent_changed:{seq}")
            differences.append({"kind": "snapshot_parent_changed", "snapshot_seq": seq, "old": old_snapshot.parent_seq, "new": new_snapshot.parent_seq, "basis": "chain topology changed"})
        for host_id in sorted(set(old_snapshot.hosts) | set(new_snapshot.hosts)):
            old_host = old_snapshot.hosts.get(host_id)
            new_host = new_snapshot.hosts.get(host_id)
            if old_host is None or new_host is None:
                for metric_id in METRICS:
                    builder.add(seq, host_id, metric_id, f"snapshot_host_changed:{seq}:{host_id}")
                differences.append({"kind": "snapshot_host_changed", "snapshot_seq": seq, "host_id": host_id, "basis": "host set changed"})
                continue
            for metric_id in METRICS:
                if old_host.expected[metric_id] != new_host.expected[metric_id]:
                    builder.add(seq, host_id, metric_id, f"expected_changed:{seq}:{host_id}:{metric_id}")
                    differences.append(
                        {
                            "kind": "expected_changed",
                            "snapshot_seq": seq,
                            "host_id": host_id,
                            "metric_id": metric_id,
                            "old": old_host.expected[metric_id],
                            "new": new_host.expected[metric_id],
                            "basis": "snapshot expectation changed",
                        }
                    )
                if old_host.tolerance_bps[metric_id] != new_host.tolerance_bps[metric_id]:
                    builder.add(seq, host_id, metric_id, f"tolerance_changed:{seq}:{host_id}:{metric_id}")
                    differences.append(
                        {
                            "kind": "tolerance_changed",
                            "snapshot_seq": seq,
                            "host_id": host_id,
                            "metric_id": metric_id,
                            "old": old_host.tolerance_bps[metric_id],
                            "new": new_host.tolerance_bps[metric_id],
                            "basis": "snapshot tolerance changed",
                        }
                    )

    if old_criteria is None or old_criteria.scale_version() != new_criteria.scale_version():
        for snapshot in new_chain.snapshots.values():
            for host_id in sorted(snapshot.hosts):
                for metric_id in METRICS:
                    builder.add(snapshot.seq, host_id, metric_id, "scale_or_boundary_policy_changed")
        differences.append(
            {
                "kind": "scale_or_boundary_policy_changed",
                "old": old_criteria.scale_version() if old_criteria else None,
                "new": new_criteria.scale_version(),
                "basis": "fixed-point session policy changed",
            }
        )
    if old_criteria is None or old_criteria.min_sources != new_criteria.min_sources:
        for snapshot in new_chain.snapshots.values():
            for host_id in sorted(snapshot.hosts):
                for metric_id in METRICS:
                    builder.add(snapshot.seq, host_id, metric_id, "min_sources_changed")
        differences.append(
            {
                "kind": "min_sources_changed",
                "old": old_criteria.min_sources if old_criteria else None,
                "new": new_criteria.min_sources,
                "basis": "authority lower bound changed",
            }
        )

    differences.sort(
        key=lambda item: (
            str(item.get("kind")),
            int(item.get("snapshot_seq", -1)) if isinstance(item.get("snapshot_seq"), int) else -1,
            str(item.get("host_id", "")),
            str(item.get("metric_id", "")),
            str(item.get("agent_id", "")),
            str(item.get("old", "")),
            str(item.get("new", "")),
        )
    )
    return PropagationResult(affected=builder.result(), differences=differences)


def affected_key_payload(items: Iterable[AffectedItem]) -> list[dict[str, object]]:
    return [
        {
            "snapshot_seq": item.snapshot_seq,
            "host_id": item.host_id,
            "metric_id": item.metric_id,
        }
        for item in items
    ]
