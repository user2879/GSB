"""基于签名汉明距离与相似度阈值的近重复簇账本。

候选样本对必须至少共享一个哈希桶；候选对按 (汉明距离升序, 样本标识升序)
稳定排序后顺序合并。簇标识取成员中最小样本标识，合并过程保留被合并簇标识、
合并依据与成员的簇内位次/合并相似度。删除样本后按“新最小成员”固定规则
重新指定簇标识并记录来源，保证无悬空成员、无重复簇标识。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .signature import hamming


def distance_threshold(signature_bits: int, similarity_permille: int) -> int:
    """相似度千分比换算为最大允许汉明距离（向下取整）。"""
    return signature_bits * (1000 - similarity_permille) // 1000


def permille_similarity(distance: int, signature_bits: int) -> int:
    return (1000 * (signature_bits - distance)) // signature_bits


@dataclass
class MergeRecord:
    merged_cluster_id: str
    into_cluster_id: str
    sample_a: str
    sample_b: str
    hamming_distance: int
    similarity_permille: int
    bucket_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_cluster_id": self.merged_cluster_id,
            "into_cluster_id": self.into_cluster_id,
            "sample_a": self.sample_a,
            "sample_b": self.sample_b,
            "hamming_distance": self.hamming_distance,
            "similarity_permille": self.similarity_permille,
            "bucket_key": self.bucket_key,
        }


@dataclass
class MemberRecord:
    sample_id: str
    rank: int
    join_similarity_permille: int
    joined_with: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "rank": self.rank,
            "join_similarity_permille": self.join_similarity_permille,
            "joined_with": self.joined_with,
        }


@dataclass
class Cluster:
    cluster_id: str
    members: list[MemberRecord] = field(default_factory=list)
    merge_log: list[MergeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "members": [m.to_dict() for m in self.members],
            "merge_log": [m.to_dict() for m in self.merge_log],
        }


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> tuple[str, str]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, rb
        # 固定规则：以样本标识较小的根为存活簇
        keep, drop = (ra, rb) if ra <= rb else (rb, ra)
        self.parent[drop] = keep
        return keep, drop


def _candidate_pairs(signatures: dict[str, Any]) -> list[tuple[int, str, str, str]]:
    """由共享桶推导候选对，返回 (距离, 小id, 大id, 桶键) 的去重排序列表。"""
    buckets: dict[str, list[str]] = {}
    for sid in sorted(signatures):
        for key in signatures[sid].bucket_keys:
            buckets.setdefault(key, []).append(sid)
    pair_bucket: dict[tuple[str, str], str] = {}
    for key, members in buckets.items():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                # 同一样本对若在多个桶共现，保留字典序最小桶键以保持确定性
                if pair not in pair_bucket or key < pair_bucket[pair]:
                    pair_bucket[pair] = key
    candidates = []
    for (a, b), key in pair_bucket.items():
        dist = hamming(signatures[a].signature_int, signatures[b].signature_int)
        candidates.append((dist, a, b, key))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates


def build_cluster_ledger(
    signatures: dict[str, Any],
    signature_bits: int,
    similarity_permille: int,
) -> dict[str, Cluster]:
    """一次性确定性构建簇账本（增量与全量共用同一纯函数）。"""
    threshold = distance_threshold(signature_bits, similarity_permille)
    uf = _UnionFind(sorted(signatures))
    # 每个根维护成员加入顺序信息与合并日志
    join_order: dict[str, list[str]] = {sid: [sid] for sid in sorted(signatures)}
    join_sim: dict[str, list[int]] = {sid: [1000] for sid in sorted(signatures)}
    join_with: dict[str, list[str | None]] = {sid: [None] for sid in sorted(signatures)}
    merge_logs: dict[str, list[MergeRecord]] = {sid: [] for sid in sorted(signatures)}

    for dist, a, b, bucket_key in _candidate_pairs(signatures):
        if dist > threshold:
            continue
        root_a, root_b = uf.find(a), uf.find(b)
        if root_a == root_b:
            continue
        keep, drop = uf.union(a, b)
        sim = permille_similarity(dist, signature_bits)
        # 被合并簇成员整体并入存活簇，位次接在存活簇现有成员之后
        join_order[keep].extend(join_order[drop])
        join_sim[keep].extend(join_sim[drop])
        join_with[keep].extend(join_with[drop])
        # 合并依据中的样本对相似度记录到每个被并入成员上
        for idx in range(len(join_order[keep]) - len(join_order[drop]), len(join_order[keep])):
            join_sim[keep][idx] = sim
            join_with[keep][idx] = f"{a}|{b}"
        merge_logs[keep].append(
            MergeRecord(
                merged_cluster_id=drop,
                into_cluster_id=keep,
                sample_a=a,
                sample_b=b,
                hamming_distance=dist,
                similarity_permille=sim,
                bucket_key=bucket_key,
            )
        )
        merge_logs[keep].extend(merge_logs[drop])
        join_order.pop(drop, None)
        join_sim.pop(drop, None)
        join_with.pop(drop, None)
        merge_logs.pop(drop, None)

    clusters: dict[str, Cluster] = {}
    for root in sorted(join_order):
        members = [
            MemberRecord(sample_id=sid, rank=rank, join_similarity_permille=sim, joined_with=partner)
            for rank, (sid, sim, partner) in enumerate(
                zip(join_order[root], join_sim[root], join_with[root])
            )
        ]
        clusters[root] = Cluster(cluster_id=root, members=members, merge_log=merge_logs[root])
    return clusters


def reassign_cluster_ids(
    old_clusters: dict[str, Cluster],
    new_clusters: dict[str, Cluster],
    deleted_ids: set[str],
) -> list[dict[str, str]]:
    """对比删除前后的簇账本，输出簇标识重指定记录。

    固定规则：删除后簇标识为剩余成员的最小样本标识；若与旧标识不同，
    记录 old_cluster_id -> new_cluster_id 及触发删除的样本。
    """
    records: list[dict[str, str]] = []
    old_members = {cid: {m.sample_id for m in cluster.members} for cid, cluster in old_clusters.items()}
    for new_cid, cluster in sorted(new_clusters.items()):
        new_member_set = {m.sample_id for m in cluster.members}
        for old_cid, old_set in old_members.items():
            if new_member_set and new_member_set.issubset(old_set) and old_cid != new_cid:
                records.append(
                    {
                        "old_cluster_id": old_cid,
                        "new_cluster_id": new_cid,
                        "reason": "anchor_deleted",
                        "deleted_sample_id": sorted(deleted_ids)[0] if deleted_ids else "",
                    }
                )
                break
    return records


def clusters_to_dict(clusters: dict[str, Cluster]) -> dict[str, Any]:
    return {cid: cluster.to_dict() for cid, cluster in sorted(clusters.items())}
