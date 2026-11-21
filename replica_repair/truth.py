"""Deterministic replica collection and weighted-majority truth selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import BlockError, IntegrityError, StateError
from .model import Manifest
from .node_service import NodeClient


@dataclass(frozen=True)
class CopyState:
    node_id: str
    block_id: str
    size_bytes: int | None
    digest: str | None
    expected_size_bytes: int | None
    expected_digest: str | None
    priority: int
    status: str
    manifest_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "block_id": self.block_id,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
            "expected_size_bytes": self.expected_size_bytes,
            "expected_digest": self.expected_digest,
            "priority": self.priority,
            "status": self.status,
            "manifest_mismatch": self.manifest_mismatch,
        }


@dataclass(frozen=True)
class DigestCandidate:
    digest: str
    count: int
    priority_weight: int
    node_ids: tuple[str, ...]
    selected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "count": self.count,
            "priority_weight": self.priority_weight,
            "node_ids": list(self.node_ids),
            "selected": self.selected,
        }


@dataclass(frozen=True)
class BlockTruth:
    block_id: str
    size_bytes: int
    truth_digest: str
    truth_nodes: tuple[str, ...]
    copies: tuple[CopyState, ...]
    mismatches: tuple[CopyState, ...]
    ranking: tuple[DigestCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "size_bytes": self.size_bytes,
            "truth_digest": self.truth_digest,
            "truth_nodes": list(self.truth_nodes),
            "copies": [copy.to_dict() for copy in self.copies],
            "mismatches": [copy.to_dict() for copy in self.mismatches],
            "ranking": [candidate.to_dict() for candidate in self.ranking],
        }


@dataclass(frozen=True)
class TruthReport:
    group_id: str
    manifest_version: int
    manifest_fingerprint: str
    blocks: tuple[BlockTruth, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "manifest_version": self.manifest_version,
            "manifest_fingerprint": self.manifest_fingerprint,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    def block(self, block_id: str) -> BlockTruth:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise BlockError("block is absent from truth report", details={"block_id": block_id})

    @property
    def mismatch_count(self) -> int:
        return sum(len(block.mismatches) for block in self.blocks)


class TruthEngine:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def collect(
        self,
        clients: Mapping[str, NodeClient],
        *,
        strict_manifest: bool = True,
    ) -> tuple[CopyState, ...]:
        nodes = sorted(self.manifest.nodes, key=lambda item: item.node_id)
        block_ids = sorted(self.manifest.block_map())
        states: list[CopyState] = []
        for block_id in block_ids:
            for node in nodes:
                priority = node.priority if node.priority is not None else 1
                node_blocks = node.block_map()
                if block_id not in node_blocks:
                    states.append(
                        CopyState(
                            node_id=node.node_id,
                            block_id=block_id,
                            size_bytes=None,
                            digest=None,
                            expected_size_bytes=None,
                            expected_digest=None,
                            priority=priority,
                            status="missing",
                            manifest_mismatch=False,
                        )
                    )
                    continue
                if node.node_id not in clients:
                    raise StateError(
                        "no node client was supplied for manifest node",
                        path="$.nodes",
                        details={"node_id": node.node_id},
                    )
                block = node_blocks[block_id]
                assert block.digest is not None
                observation = clients[node.node_id].fetch_observation(
                    block_id,
                    block.size_bytes,
                    block.digest,
                    strict_manifest=strict_manifest,
                )
                status = "ok"
                if observation.manifest_mismatch:
                    status = "manifest_mismatch"
                states.append(
                    CopyState(
                        node_id=node.node_id,
                        block_id=block_id,
                        size_bytes=observation.size_bytes,
                        digest=observation.digest,
                        expected_size_bytes=block.size_bytes,
                        expected_digest=block.digest,
                        priority=priority,
                        status=status,
                        manifest_mismatch=observation.manifest_mismatch,
                    )
                )
        states.sort(key=lambda item: (item.block_id, item.node_id))
        return tuple(states)

    def determine(self, states: Sequence[CopyState]) -> TruthReport:
        priorities = {
            node.node_id: (node.priority if node.priority is not None else 1)
            for node in self.manifest.nodes
        }
        by_block: dict[str, list[CopyState]] = {}
        for state in states:
            by_block.setdefault(state.block_id, []).append(state)

        block_truths: list[BlockTruth] = []
        for block_id in sorted(by_block):
            copies = sorted(by_block[block_id], key=lambda item: item.node_id)
            eligible = [
                copy
                for copy in copies
                if copy.digest is not None and copy.status == "ok"
            ]
            if not eligible:
                raise BlockError(
                    "no valid replica is available for truth selection",
                    path=f"$.blocks[{block_id}]",
                    details={
                        "block_id": block_id,
                        "statuses": [
                            {"node_id": copy.node_id, "status": copy.status}
                            for copy in copies
                        ],
                    },
                )
            groups: dict[str, list[CopyState]] = {}
            for copy in eligible:
                groups.setdefault(copy.digest or "", []).append(copy)

            ranked: list[tuple[int, int, tuple[str, ...], str, list[CopyState]]] = []
            for digest, group in groups.items():
                node_ids = tuple(sorted(copy.node_id for copy in group))
                weight = sum(priorities[node_id] for node_id in node_ids)
                ranked.append((-len(group), -weight, node_ids, digest, group))
            # Explicit stable order: frequency desc, priority weight desc,
            # lexicographic node-id vector asc, digest asc.
            ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            chosen = ranked[0]
            truth_digest = chosen[3]
            truth_nodes = tuple(sorted(copy.node_id for copy in chosen[4]))
            ranking = tuple(
                DigestCandidate(
                    digest=item[3],
                    count=-item[0],
                    priority_weight=-item[1],
                    node_ids=item[2],
                    selected=(item[3] == truth_digest),
                )
                for item in ranked
            )
            final_copies: list[CopyState] = []
            mismatches: list[CopyState] = []
            chosen_size = chosen[4][0].size_bytes
            for copy in copies:
                if copy.status != "ok":
                    final_copies.append(copy)
                    if copy.status == "manifest_mismatch":
                        mismatches.append(copy)
                    continue
                if copy.digest == truth_digest:
                    final_copies.append(copy)
                else:
                    mismatch = CopyState(
                        node_id=copy.node_id,
                        block_id=copy.block_id,
                        size_bytes=copy.size_bytes,
                        digest=copy.digest,
                        expected_size_bytes=copy.expected_size_bytes,
                        expected_digest=copy.expected_digest,
                        priority=copy.priority,
                        status="mismatch",
                        manifest_mismatch=copy.manifest_mismatch,
                    )
                    final_copies.append(mismatch)
                    mismatches.append(mismatch)
            if chosen_size is None:
                raise IntegrityError("truth block has no size", details={"block_id": block_id})
            block_truths.append(
                BlockTruth(
                    block_id=block_id,
                    size_bytes=int(chosen_size),
                    truth_digest=truth_digest,
                    truth_nodes=truth_nodes,
                    copies=tuple(final_copies),
                    mismatches=tuple(sorted(mismatches, key=lambda item: item.node_id)),
                    ranking=ranking,
                )
            )
        return TruthReport(
            group_id=self.manifest.group_id,
            manifest_version=self.manifest.manifest_version,
            manifest_fingerprint=self.manifest.fingerprint(),
            blocks=tuple(block_truths),
        )

    def run(
        self,
        clients: Mapping[str, NodeClient],
        *,
        strict_manifest: bool = True,
    ) -> TruthReport:
        return self.determine(self.collect(clients, strict_manifest=strict_manifest))


def determine_from_states(manifest: Manifest, states: Sequence[CopyState]) -> TruthReport:
    return TruthEngine(manifest).determine(states)
