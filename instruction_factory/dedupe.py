from __future__ import annotations

from decimal import Decimal
from typing import Any

from .tokenizer import SubwordTokenizer
from .util import as_decimal, canonical_json, decimal_text, stable_key_hash


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def dedupe_cache_key(
    *,
    task_type: str,
    score_keys: list[str],
    candidate_ids: list[str],
    threshold: Decimal,
    tokenizer: SubwordTokenizer,
) -> str:
    return stable_key_hash(
        "dedupe-v1",
        task_type,
        canonical_json(score_keys),
        canonical_json(candidate_ids),
        str(threshold),
        tokenizer.fingerprint,
    )


def dedupe_task(
    *,
    task_type: str,
    scored_candidates: list[dict[str, Any]],
    sampling_by_id: dict[str, dict[str, Any]],
    threshold: Decimal,
    tokenizer: SubwordTokenizer,
) -> dict[str, Any]:
    candidates = sorted(scored_candidates, key=lambda item: (item["sample_seq"], item["candidate_id"]))
    token_sets = {
        candidate["candidate_id"]: set(tokenizer.tokenize(sampling_by_id[candidate["candidate_id"]]["instruction"]))
        for candidate in candidates
    }
    union_find = _UnionFind(len(candidates))
    similarity_edges: list[dict[str, Any]] = []
    equal_pairs: list[tuple[str, str]] = []
    for left in range(len(candidates)):
        left_id = candidates[left]["candidate_id"]
        left_tokens = token_sets[left_id]
        for right in range(left + 1, len(candidates)):
            right_id = candidates[right]["candidate_id"]
            right_tokens = token_sets[right_id]
            union = left_tokens | right_tokens
            if not union:
                similarity = Decimal(0)
            else:
                similarity = Decimal(len(left_tokens & right_tokens)) / Decimal(len(union))
            if similarity >= threshold:
                union_find.union(left, right)
                is_equal = similarity == threshold
                similarity_edges.append(
                    {
                        "left_candidate_id": left_id,
                        "right_candidate_id": right_id,
                        "similarity": decimal_text(similarity, 6),
                        "threshold_equal": is_equal,
                    }
                )
                if is_equal:
                    equal_pairs.append((left_id, right_id))

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        members_by_root.setdefault(union_find.find(index), []).append(index)
    ordered_member_lists = sorted(
        members_by_root.values(),
        key=lambda members: min(candidates[index]["sample_seq"] for index in members),
    )
    groups: list[dict[str, Any]] = []
    retained: list[str] = []
    removed: list[dict[str, Any]] = []
    for members in ordered_member_lists:
        member_candidates = [candidates[index] for index in members]
        member_candidates.sort(key=lambda item: (item["sample_seq"], item["candidate_id"]))
        representative = min(
            member_candidates,
            key=lambda item: (-item["total_fixed"], item["candidate_id"]),
        )
        representative_id = representative["candidate_id"]
        representative_tokens = token_sets[representative_id]
        min_seq = member_candidates[0]["sample_seq"]
        group_id = f"grp-{task_type}-{min_seq:06d}-{stable_key_hash(task_type, representative_id, min_seq)[:8]}"
        member_records: list[dict[str, Any]] = []
        for member in member_candidates:
            member_id = member["candidate_id"]
            member_tokens = token_sets[member_id]
            union = member_tokens | representative_tokens
            similarity = (
                Decimal(len(member_tokens & representative_tokens)) / Decimal(len(union))
                if union
                else Decimal(0)
            )
            member_records.append(
                {
                    "candidate_id": member_id,
                    "sample_seq": member["sample_seq"],
                    "similarity_to_representative": decimal_text(similarity, 6),
                    "threshold_equal": similarity == threshold,
                    "is_representative": member_id == representative_id,
                }
            )
            if member_id == representative_id:
                retained.append(member_id)
            else:
                removed.append(
                    {
                        "candidate_id": member_id,
                        "task_type": task_type,
                        "sample_seq": member["sample_seq"],
                        "representative_candidate_id": representative_id,
                        "similarity": decimal_text(similarity, 6),
                        "threshold_equal": similarity == threshold,
                    }
                )
        groups.append(
            {
                "group_id": group_id,
                "task_type": task_type,
                "min_sample_seq": min_seq,
                "representative_candidate_id": representative_id,
                "representative_total_fixed": representative["total_fixed"],
                "member_candidates": member_records,
            }
        )
    retained.sort(key=lambda candidate_id: next(item for item in candidates if item["candidate_id"] == candidate_id)["sample_seq"])
    removed.sort(key=lambda item: (item["sample_seq"], item["candidate_id"]))
    score_keys = [candidate["cache_key"] for candidate in candidates]
    cache_key = dedupe_cache_key(
        task_type=task_type,
        score_keys=score_keys,
        candidate_ids=[candidate["candidate_id"] for candidate in candidates],
        threshold=threshold,
        tokenizer=tokenizer,
    )
    return {
        "stage": "dedupe",
        "task_type": task_type,
        "stable_sort_key": f"{task_type}|dedupe",
        "threshold": decimal_text(threshold),
        "threshold_fingerprint": stable_key_hash("threshold-v1", str(threshold)),
        "groups": groups,
        "retained_candidate_ids": retained,
        "removed_candidates": removed,
        "similarity_edges": similarity_edges,
        "equal_threshold_pairs": [
            {"left_candidate_id": left, "right_candidate_id": right}
            for left, right in equal_pairs
        ],
        "cache_key": cache_key,
    }
