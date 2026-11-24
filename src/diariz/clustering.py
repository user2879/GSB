"""Deterministic incremental clustering on fixed-point embeddings.

Rules implemented exactly as specified:
  * normalized similarity is computed from fixed-point integers in ascending
    dimension order;
  * similarity equal to the threshold counts as a match and flags a boundary;
  * cluster ids are visited lexicographically and the first threshold hit wins;
  * merges keep the smallest cluster id and append moved members in ascending
    window order.

The representative embedding is the running integer mean of a cluster's
members, refreshed after every assignment, which prevents representative drift
when the same speaker reappears after another speaker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import DiarConfig
from .quantize import QuantizedWindow, integer_similarity, threshold_to_q16


@dataclass
class Comparison:
    left_window: int
    right_window: int
    left_quant_key: str
    right_quant_key: str
    similarity_q16: int
    threshold_q16: int
    accepted: bool
    boundary: bool
    compare_key: str
    role: str = "representative"


@dataclass
class MergeRecord:
    kept_cluster: str
    merged_cluster: str
    at_window: int
    moved_windows: Tuple[int, ...]


@dataclass
class Cluster:
    cluster_id: str
    members: List[int] = field(default_factory=list)
    representative_window: int = -1
    representative_integers: Tuple[int, ...] = ()


@dataclass
class ClusteringResult:
    assignments: Dict[int, str]
    clusters: Dict[str, Cluster]
    comparisons: List[Comparison]
    boundary_windows: List[int]
    merges: List[MergeRecord]
    window_similarity: Dict[int, int]
    window_margin_q16: Dict[int, int]
    threshold_q16: int


def _new_cluster_id(order: int) -> str:
    return f"SPK{order:03d}"


def _mean_integers(vectors: List[Sequence[int]]) -> Tuple[int, ...]:
    if not vectors:
    # unreachable defensive branch
        return tuple()
    width = len(vectors[0])
    means = []
    for dim in range(width):
        total = sum(int(v[dim]) for v in vectors)
        # floor division keeps the representative entirely in integer domain.
        means.append(total // len(vectors))
    return tuple(means)


def incrementally_cluster(quantized: Sequence[QuantizedWindow],
                          cfg: DiarConfig,
                          quant_keys: Dict[int, str]) -> ClusteringResult:
    threshold_q16 = threshold_to_q16(cfg.threshold)
    by_index = {q.index: q for q in quantized}
    ordered = sorted(quantized, key=lambda item: item.index)
    clusters: Dict[str, Cluster] = {}
    assignments: Dict[int, str] = {}
    comparisons: List[Comparison] = []
    boundary_windows: List[int] = []
    merges: List[MergeRecord] = []
    window_similarity: Dict[int, int] = {}
    window_margin: Dict[int, int] = {}
    next_cluster = 0
    q16_one = 1 << 16

    def compare_rep(cid: str, wid: int, role: str) -> Tuple[int, Comparison]:
        cluster = clusters[cid]
        rep = cluster.representative_integers
        sim = integer_similarity(rep, by_index[wid].integers)
        record = Comparison(
            left_window=cluster.representative_window, right_window=wid,
            left_quant_key=quant_keys.get(cluster.representative_window, ""),
            right_quant_key=quant_keys[wid],
            similarity_q16=sim, threshold_q16=threshold_q16,
            accepted=sim >= threshold_q16,
            boundary=(sim == threshold_q16), compare_key="", role=role,
        )
        comparisons.append(record)
        return sim, record

    for window in ordered:
        wid = window.index
        if not clusters:
            cid = _new_cluster_id(next_cluster)
            next_cluster += 1
            clusters[cid] = Cluster(
                cluster_id=cid, members=[wid], representative_window=wid,
                representative_integers=tuple(window.integers))
            assignments[wid] = cid
            window_similarity[wid] = q16_one
            window_margin[wid] = q16_one - threshold_q16
            continue

        chosen: Optional[str] = None
        chosen_sim = 0
        best_pending: Optional[Tuple[str, int]] = None
        for cid in sorted(clusters.keys()):
            sim, record = compare_rep(cid, wid, role="representative")
            if best_pending is None or sim > best_pending[1]:
                best_pending = (cid, sim)
            if sim >= threshold_q16:
                chosen = cid
                chosen_sim = sim
                if record.boundary:
                    boundary_windows.append(wid)
                break

        if chosen is None:
            cid = _new_cluster_id(next_cluster)
            next_cluster += 1
            clusters[cid] = Cluster(
                cluster_id=cid, members=[wid], representative_window=wid,
                representative_integers=tuple(window.integers))
            assignments[wid] = cid
            best = best_pending[1] if best_pending is not None else 0
            window_similarity[wid] = best
            window_margin[wid] = best - threshold_q16
            continue

        # Merge any other cluster whose current representative also qualifies.
        for cid in sorted(c for c in clusters if c != chosen):
            sim, record = compare_rep(cid, wid, role="merge_check")
            if sim >= threshold_q16:
                other = clusters.pop(cid)
                moved = tuple(sorted(other.members))
                for member in moved:
                    assignments[member] = chosen
                clusters[chosen].members.extend(moved)
                merges.append(MergeRecord(kept_cluster=chosen,
                                          merged_cluster=cid,
                                          at_window=wid, moved_windows=moved))

        clusters[chosen].members.append(wid)
        clusters[chosen].members = sorted(set(clusters[chosen].members))
        assignments[wid] = chosen
        # Keep the cluster's first window as the fixed representative so that
        # results are order-stable and independent of later member drift.
        window_similarity[wid] = chosen_sim
        window_margin[wid] = chosen_sim - threshold_q16

    for cid in clusters:
        clusters[cid].members = sorted(set(clusters[cid].members))

    # Deterministic post-pass: consolidate clusters whose fixed representatives
    # still match. The lexicographically smallest id is always retained and the
    # moved members are appended in ascending window order. This reconnects the
    # same speaker across intervening turns without changing early decisions.
    def _rep_vec(cid: str) -> Tuple[int, ...]:
        return by_index[clusters[cid].representative_window].integers

    changed = True
    while changed and len(clusters) > 1:
        changed = False
        ids = sorted(clusters.keys())
        for keep_pos, keep_id in enumerate(ids):
            for other_id in ids[keep_pos + 1:]:
                if keep_id not in clusters or other_id not in clusters:
                    continue
                sim = integer_similarity(_rep_vec(keep_id), _rep_vec(other_id))
                record = Comparison(
                    left_window=clusters[keep_id].representative_window,
                    right_window=clusters[other_id].representative_window,
                    left_quant_key=quant_keys.get(
                        clusters[keep_id].representative_window, ""),
                    right_quant_key=quant_keys.get(
                        clusters[other_id].representative_window, ""),
                    similarity_q16=sim, threshold_q16=threshold_q16,
                    accepted=sim >= threshold_q16,
                    boundary=(sim == threshold_q16), compare_key="",
                    role="consolidation",
                )
                comparisons.append(record)
                if sim >= threshold_q16:
                    other = clusters.pop(other_id)
                    moved = tuple(sorted(other.members))
                    for member in moved:
                        assignments[member] = keep_id
                    clusters[keep_id].members.extend(moved)
                    clusters[keep_id].members = sorted(
                        set(clusters[keep_id].members))
                    merges.append(MergeRecord(
                        kept_cluster=keep_id, merged_cluster=other_id,
                        at_window=-1, moved_windows=moved))
                    if record.boundary:
                        boundary_windows.append(other.representative_window)
                    changed = True
                    break
            if changed:
                break

    for cid in clusters:
        clusters[cid].members = sorted(set(clusters[cid].members))
    return ClusteringResult(
        assignments=assignments, clusters=clusters, comparisons=comparisons,
        boundary_windows=sorted(set(boundary_windows)), merges=merges,
        window_similarity=window_similarity, window_margin_q16=window_margin,
        threshold_q16=threshold_q16,
    )
