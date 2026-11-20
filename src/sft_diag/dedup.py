"""近重复剔除与代表样本选择（子词集合 Jaccard，确定性）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .models import WorkspaceState
from .tokenizer import Tokenizer
from .utils import derive_cache_key, fingerprint, float64


@dataclass
class SimilarityEdge:
    left: str
    right: str
    similarity: float
    boundary: bool


class _UnionFind:
    def __init__(self, identifiers: list[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        root = identifier
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[identifier] != root:
            self.parent[identifier], identifier = root, self.parent[identifier]
        return root

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            if root_left < root_right:
                self.parent[root_right] = root_left
            else:
                self.parent[root_left] = root_right


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float64(len(left & right) / len(union))


class Deduplicator:
    def __init__(self, state: WorkspaceState, tokenizer: Tokenizer) -> None:
        self.state = state
        self.tokenizer = tokenizer
        fps = state.fingerprints()
        self.fingerprints = fps
        self.threshold = float64(state.config.dedup_threshold)
        self.boundary_rule = state.config.boundary_rule
        self.dedup_config_fp = fingerprint(
            {
                "dedup_threshold": state.config.dedup_threshold,
                "boundary_rule": state.config.boundary_rule,
            }
        )

    def group_cache_key(self, task_type: str, candidates: list[dict[str, Any]]) -> str:
        payload = [
            {
                "candidate_id": candidate["candidate_id"],
                "tokens": sorted(self.tokenizer.token_set(candidate["instruction"])),
                "total_score": candidate["total_score"],
            }
            for candidate in candidates
        ]
        return derive_cache_key(
            "dedup",
            fingerprint({"task_type": task_type, "candidates": payload}),
            self.dedup_config_fp,
        )

    def _pair_is_duplicate(self, similarity: float) -> tuple[bool, bool]:
        boundary = similarity == self.threshold
        if similarity > self.threshold:
            return True, False
        if boundary:
            return (self.boundary_rule == "merge"), True
        return False, False

    def _compute_edges(
        self,
        candidates: list[dict[str, Any]],
        sets: list[frozenset[str]],
        max_workers: int,
    ) -> list[SimilarityEdge]:
        pairs = [
            (i, j)
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
        ]
        edges: list[SimilarityEdge | None] = [None] * len(pairs)

        def worker(pair: tuple[int, int]) -> SimilarityEdge:
            i, j = pair
            similarity = jaccard(sets[i], sets[j])
            is_dup, boundary = self._pair_is_duplicate(similarity)
            edge = SimilarityEdge(
                left=candidates[i]["candidate_id"],
                right=candidates[j]["candidate_id"],
                similarity=similarity,
                boundary=boundary,
            )
            edge.is_duplicate = is_dup  # type: ignore[attr-defined]
            return edge

        if not pairs:
            return []
        workers = max(1, min(max_workers, len(pairs)))
        if workers == 1:
            for index, pair in enumerate(pairs):
                edges[index] = worker(pair)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(worker, pair): idx for idx, pair in enumerate(pairs)}
                for future in futures:
                    edges[futures[future]] = future.result()
        return [edge for edge in edges if edge is not None]

    def deduplicate_task(
        self,
        task_type: str,
        candidates: list[dict[str, Any]],
        max_workers: int,
    ) -> dict[str, Any]:
        ordered = sorted(candidates, key=lambda item: item["candidate_id"])
        identifiers = [candidate["candidate_id"] for candidate in ordered]
        sets = [self.tokenizer.token_set(candidate["instruction"]) for candidate in ordered]
        edges = self._compute_edges(ordered, sets, max_workers)
        union_find = _UnionFind(identifiers)
        duplicate_edges: list[SimilarityEdge] = []
        boundary_members: set[str] = set()
        for edge in sorted(edges, key=lambda e: (e.left, e.right)):
            if getattr(edge, "is_duplicate", False):
                union_find.union(edge.left, edge.right)
                duplicate_edges.append(edge)
            if edge.boundary:
                boundary_members.add(edge.left)
                boundary_members.add(edge.right)

        groups_map: dict[str, list[str]] = {}
        for identifier in identifiers:
            groups_map.setdefault(union_find.find(identifier), []).append(identifier)
        score_by_id = {candidate["candidate_id"]: candidate["total_score"] for candidate in ordered}
        groups: list[dict[str, Any]] = []
        representatives: set[str] = set()
        removals: dict[str, dict[str, Any]] = {}
        for root in sorted(groups_map):
            members = sorted(groups_map[root])
            representative = min(
                members,
                key=lambda identifier: (-float(score_by_id[identifier]), identifier),
            )
            representatives.add(representative)
            groups.append(
                {
                    "group_id": f"{task_type}:G{len(groups):06d}",
                    "members": members,
                    "representative_id": representative,
                    "boundary_member_ids": sorted(
                        identifier for identifier in members if identifier in boundary_members
                    ),
                }
            )
            for member in members:
                if member == representative:
                    continue
                sim_to_rep = next(
                    (
                        edge.similarity
                        for edge in duplicate_edges
                        if {edge.left, edge.right} == {member, representative}
                    ),
                    None,
                )
                if sim_to_rep is None:
                    sim_to_rep = self._similarity_to(
                        member, representative, ordered, sets
                    )
                removals[member] = {
                    "candidate_id": member,
                    "representative_id": representative,
                    "similarity": sim_to_rep,
                }
        retained = [
            identifier
            for identifier in identifiers
            if identifier in representatives
        ]
        return {
            "task_type": task_type,
            "cache_key": self.group_cache_key(task_type, ordered),
            "groups": groups,
            "retained_ids": retained,
            "removals": [removals[key] for key in sorted(removals)],
            "boundary_candidate_ids": sorted(boundary_members),
            "edges": [
                {
                    "left": edge.left,
                    "right": edge.right,
                    "similarity": edge.similarity,
                    "boundary": edge.boundary,
                    "duplicate": getattr(edge, "is_duplicate", False),
                }
                for edge in sorted(edges, key=lambda e: (e.left, e.right))
            ],
        }

    @staticmethod
    def _similarity_to(
        member: str,
        representative: str,
        ordered: list[dict[str, Any]],
        sets: list[frozenset[str]],
    ) -> float:
        id_to_index = {candidate["candidate_id"]: i for i, candidate in enumerate(ordered)}
        return jaccard(
            sets[id_to_index[member]], sets[id_to_index[representative]]
        )

    def deduplicate_all(
        self,
        scored_candidates: list[dict[str, Any]],
        max_workers: int,
        cached: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        cached = cached or {}
        results: list[dict[str, Any]] = []
        status: dict[str, str] = {}
        by_task: dict[str, list[dict[str, Any]]] = {}
        for candidate in scored_candidates:
            by_task.setdefault(candidate["task_type"], []).append(candidate)
        for task_type in sorted(by_task):
            task_candidates = sorted(
                by_task[task_type], key=lambda item: item["candidate_id"]
            )
            key = self.group_cache_key(task_type, task_candidates)
            if key in cached:
                results.append(cached[key])
                for candidate in task_candidates:
                    status[candidate["candidate_id"]] = "reused"
            else:
                results.append(
                    self.deduplicate_task(task_type, task_candidates, max_workers)
                )
                for candidate in task_candidates:
                    status[candidate["candidate_id"]] = "recomputed"
        return results, status
