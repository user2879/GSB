"""Immutable, versioned timeline snapshots for one choice sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .canonical import (
    double,
    edge_sequence_hash,
    replay_hash,
    sha256_json,
    state_hash,
)
from .engine import (
    Arbitration,
    BranchEvaluation,
    CausalChain,
    apply_edge,
    causal_chain_payload,
    condition_boundary_payload,
    evaluate_branches,
    initial_causal_chains,
    project_node_outgoing_effects,
)
from .errors import SequenceInvalid
from .model import NarrativeSpec, SequenceSample


@dataclass(frozen=True)
class Snapshot:
    schema: str
    sequence_id: str
    state_version: int
    step_index: int
    node_id: str
    node_title: str
    status: str
    state: tuple[float, ...]
    components: tuple[str, ...]
    state_hash: str
    edge_history: tuple[str, ...]
    incoming_edge_id: str | None
    requested_edge_id: str | None
    ending: str | None
    causal_chains: tuple[CausalChain, ...]
    branches: tuple[BranchEvaluation, ...]
    boundary_judgments: tuple[dict[str, Any], ...]
    arbitrations: tuple[Arbitration, ...]
    outgoing_projection: dict[str, Any]
    edge_application: dict[str, Any] | None
    content_hash: str
    version_stamp: str

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sequence_id": self.sequence_id,
            "state_version": self.state_version,
            "step_index": self.step_index,
            "node_id": self.node_id,
            "node_title": self.node_title,
            "status": self.status,
            "components": list(self.components),
            "state_vector": [double(value) for value in self.state],
            "state": {
                component: double(self.state[index])
                for index, component in enumerate(self.components)
            },
            "state_hash": self.state_hash,
            "edge_history": list(self.edge_history),
            "incoming_edge_id": self.incoming_edge_id,
            "requested_edge_id": self.requested_edge_id,
            "ending": self.ending,
            "causal_chains": causal_chain_payload(self.causal_chains),
            "branches": [branch.as_dict() for branch in self.branches],
            "boundary_judgments": [dict(item) for item in self.boundary_judgments],
            "arbitrations": [arbitration.as_dict() for arbitration in self.arbitrations],
            "outgoing_effect_projection": self.outgoing_projection,
            "edge_application": self.edge_application,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.core_payload()
        payload["content_hash"] = self.content_hash
        payload["version_stamp"] = self.version_stamp
        payload["boundary_manifest"] = {
            "state_version": self.state_version,
            "version_stamp": self.version_stamp,
            "content_hash": self.content_hash,
            "judgments": [dict(item) for item in self.boundary_judgments],
        }
        return payload

    def reachable_edges(self) -> tuple[BranchEvaluation, ...]:
        return tuple(branch for branch in self.branches if branch.status == "reachable")


@dataclass(frozen=True)
class TimelineResult:
    spec_id: str
    spec_hash: str
    sequence_id: str
    snapshots: tuple[Snapshot, ...]
    ending: str | None
    status: str
    final_state: tuple[float, ...]
    components: tuple[str, ...]
    edge_history: tuple[str, ...]
    unused_edges: tuple[str, ...]
    edge_sequence_hash: str
    state_hash: str
    replay_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "narrative-workbench/timeline/v1",
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "sequence_id": self.sequence_id,
            "status": self.status,
            "ending": self.ending,
            "components": list(self.components),
            "final_state": {
                component: double(self.final_state[index])
                for index, component in enumerate(self.components)
            },
            "final_state_vector": [double(value) for value in self.final_state],
            "edge_history": list(self.edge_history),
            "unused_edges": list(self.unused_edges),
            "edge_sequence_hash": self.edge_sequence_hash,
            "state_hash": self.state_hash,
            "replay_hash": self.replay_hash,
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "node_hashes": {
                f"snapshot:{snapshot.state_version:04d}": snapshot.content_hash
                for snapshot in self.snapshots
            },
        }


def make_snapshot(
    spec: NarrativeSpec,
    *,
    sequence_id: str,
    state_version: int,
    step_index: int,
    node_id: str,
    status: str,
    state_values: Sequence[float],
    edge_history: Sequence[str],
    incoming_edge_id: str | None,
    requested_edge_id: str | None,
    ending: str | None,
    causal_chains: Sequence[CausalChain],
    arbitrations: Sequence[Arbitration] = (),
    edge_application: dict[str, Any] | None = None,
    branches: Sequence[BranchEvaluation] | None = None,
    outgoing_projection: dict[str, Any] | None = None,
) -> Snapshot:
    node = spec.node_by_id(node_id)
    resolved_branches = tuple(branches) if branches is not None else evaluate_branches(spec, node_id, state_values)
    boundaries = tuple(condition_boundary_payload(resolved_branches))
    projection = (
        dict(outgoing_projection)
        if outgoing_projection is not None
        else project_node_outgoing_effects(spec, node_id, state_values)
    )
    provisional = Snapshot(
        schema="narrative-workbench/snapshot/v1",
        sequence_id=sequence_id,
        state_version=state_version,
        step_index=step_index,
        node_id=node_id,
        node_title=node.title,
        status=status,
        state=tuple(double(value) for value in state_values),
        components=tuple(spec.components),
        state_hash=state_hash(spec.components, state_values),
        edge_history=tuple(edge_history),
        incoming_edge_id=incoming_edge_id,
        requested_edge_id=requested_edge_id,
        ending=ending,
        causal_chains=tuple(causal_chains),
        branches=resolved_branches,
        boundary_judgments=boundaries,
        arbitrations=tuple(arbitrations),
        outgoing_projection=projection,
        edge_application=edge_application,
        content_hash="",
        version_stamp="",
    )
    digest = sha256_json(provisional.core_payload())
    stamp = f"state-v{state_version:04d}:{digest[:16]}"
    return Snapshot(
        schema=provisional.schema,
        sequence_id=provisional.sequence_id,
        state_version=provisional.state_version,
        step_index=provisional.step_index,
        node_id=provisional.node_id,
        node_title=provisional.node_title,
        status=provisional.status,
        state=provisional.state,
        components=provisional.components,
        state_hash=provisional.state_hash,
        edge_history=provisional.edge_history,
        incoming_edge_id=provisional.incoming_edge_id,
        requested_edge_id=provisional.requested_edge_id,
        ending=provisional.ending,
        causal_chains=provisional.causal_chains,
        branches=provisional.branches,
        boundary_judgments=provisional.boundary_judgments,
        arbitrations=provisional.arbitrations,
        outgoing_projection=provisional.outgoing_projection,
        edge_application=provisional.edge_application,
        content_hash=digest,
        version_stamp=stamp,
    )


class NarrativeSession:
    """Replay one sequence and expose only immutable versioned snapshots."""

    def __init__(self, spec: NarrativeSpec, sequence: SequenceSample | str) -> None:
        self.spec = spec
        if isinstance(sequence, str):
            self.sequence = spec.sample_by_id(sequence)
        else:
            self.sequence = sequence

    def run(self) -> TimelineResult:
        spec = self.spec
        state = list(spec.initial_state)
        chains = initial_causal_chains(spec.components, state)
        history: list[str] = []
        snapshots: list[Snapshot] = []
        current_node = spec.start
        current_node_obj = spec.node_by_id(current_node)
        version = 1
        step_index = 0
        status = "ended" if current_node_obj.ending is not None else "active"
        snapshots.append(
            make_snapshot(
                spec,
                sequence_id=self.sequence.id,
                state_version=version,
                step_index=step_index,
                node_id=current_node,
                status=status,
                state_values=state,
                edge_history=history,
                incoming_edge_id=None,
                requested_edge_id=None,
                ending=current_node_obj.ending,
                causal_chains=chains,
            )
        )
        if status == "ended":
            return self._result(snapshots, state, history, self.sequence.edges, "ended", current_node_obj.ending)

        cursor = 0
        while cursor < len(self.sequence.edges):
            chosen_id = self.sequence.edges[cursor]
            chosen = spec.edge_by_id(chosen_id)
            if chosen.source != current_node:
                raise SequenceInvalid(
                    f"sequence {self.sequence.id!r} takes edge {chosen_id!r} from {current_node!r}, expected {chosen.source!r}",
                    details={"sequence_id": self.sequence.id, "step_index": cursor},
                )

            branch = next(
                (candidate for candidate in evaluate_branches(spec, current_node, state) if candidate.edge_id == chosen_id),
                None,
            )
            if branch is None:
                raise SequenceInvalid(
                    f"edge {chosen_id!r} is not an outgoing edge of node {current_node!r}",
                    details={"sequence_id": self.sequence.id, "step_index": cursor},
                )
            if not branch.reachable:
                version += 1
                blocked_ending = f"blocked:{current_node}"
                snapshots.append(
                    make_snapshot(
                        spec,
                        sequence_id=self.sequence.id,
                        state_version=version,
                        step_index=cursor + 1,
                        node_id=current_node,
                        status="blocked",
                        state_values=state,
                        edge_history=history,
                        incoming_edge_id=history[-1] if history else None,
                        requested_edge_id=chosen_id,
                        ending=blocked_ending,
                        causal_chains=chains,
                    )
                )
                unused = self.sequence.edges[cursor:]
                return self._result(snapshots, state, history, unused, "blocked", blocked_ending)

            application = apply_edge(spec, chosen, state, chains)
            state = list(application.state)
            chains = application.causal_chains
            history.append(chosen.id)
            current_node = chosen.target
            cursor += 1
            version += 1
            step_index += 1
            target_node = spec.node_by_id(current_node)
            ending = target_node.ending
            snapshot_status = "ended" if ending is not None else "active"
            snapshots.append(
                make_snapshot(
                    spec,
                    sequence_id=self.sequence.id,
                    state_version=version,
                    step_index=step_index,
                    node_id=current_node,
                    status=snapshot_status,
                    state_values=state,
                    edge_history=history,
                    incoming_edge_id=chosen.id,
                    requested_edge_id=None,
                    ending=ending,
                    causal_chains=chains,
                    arbitrations=application.arbitrations,
                    edge_application=application.as_dict(),
                )
            )
            if ending is not None:
                unused = self.sequence.edges[cursor:]
                return self._result(snapshots, state, history, unused, "ended", ending)

        unresolved = f"unresolved:{current_node}"
        return self._result(snapshots, state, history, (), "unresolved", unresolved)

    def _result(
        self,
        snapshots: Sequence[Snapshot],
        state: Sequence[float],
        history: Sequence[str],
        unused_edges: Sequence[str],
        status: str,
        ending: str | None,
    ) -> TimelineResult:
        return TimelineResult(
            spec_id=self.spec.id,
            spec_hash=self.spec.spec_hash,
            sequence_id=self.sequence.id,
            snapshots=tuple(snapshots),
            ending=ending,
            status=status,
            final_state=tuple(double(value) for value in state),
            components=tuple(self.spec.components),
            edge_history=tuple(history),
            unused_edges=tuple(unused_edges),
            edge_sequence_hash=edge_sequence_hash(history),
            state_hash=state_hash(self.spec.components, state),
            replay_hash=replay_hash(history, self.spec.components, state),
        )


def replay_sequence(spec: NarrativeSpec, sequence_id: str) -> TimelineResult:
    return NarrativeSession(spec, sequence_id).run()
