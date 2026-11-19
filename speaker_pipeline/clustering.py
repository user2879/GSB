"""Deterministic agglomerative clustering and interval attribution."""

import numpy as np

from .errors import AttributionError, InvalidThresholdError
from .fingerprint import round6


def _euclidean(vec_a, vec_b):
    return float(np.linalg.norm(np.asarray(vec_a) - np.asarray(vec_b)))


def base_agglomerative(interval_vectors, min_frame_keys, threshold):
    """Deterministic agglomerative clustering with centroid linkage.

    *interval_vectors*: list of float64 mean voiceprints in global order.
    *min_frame_keys*: per-interval stable key (file rank, min frame index).
    Ties on inter-cluster distance break on the clusters' minimum frame key,
    then on lexicographic temporary cluster id.
    """
    if threshold <= 0:
        raise InvalidThresholdError(
            "cluster_distance_threshold must be > 0", stage="clustering"
        )
    clusters = [
        {
            "temp_id": f"tmp-{idx:06d}",
            "members": [idx],
            "centroid": np.asarray(vector, dtype=np.float64),
            "min_key": min_frame_keys[idx],
        }
        for idx, vector in enumerate(interval_vectors)
    ]

    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = _euclidean(
                    clusters[i]["centroid"], clusters[j]["centroid"]
                )
                candidate = (
                    round6(distance),
                    min(clusters[i]["min_key"], clusters[j]["min_key"]),
                    min(clusters[i]["temp_id"], clusters[j]["temp_id"]),
                    i,
                    j,
                )
                if best is None or candidate < best:
                    best = candidate
        distance = best[0]
        if distance > threshold:
            break
        i, j = best[3], best[4]
        merged_members = clusters[i]["members"] + clusters[j]["members"]
        merged_centroid = np.mean(
            np.asarray([interval_vectors[m] for m in merged_members]),
            axis=0,
        )
        merged = {
            "temp_id": min(clusters[i]["temp_id"], clusters[j]["temp_id"]),
            "members": merged_members,
            "centroid": merged_centroid,
            "min_key": min(clusters[i]["min_key"], clusters[j]["min_key"]),
        }
        clusters = [
            cluster
            for idx, cluster in enumerate(clusters)
            if idx not in (i, j)
        ]
        clusters.append(merged)

    # Renumber clusters by the smallest global frame key of their members.
    clusters.sort(key=lambda cluster: (cluster["min_key"], cluster["temp_id"]))
    assignments = {}
    for ordinal, cluster in enumerate(clusters):
        cluster_id = f"SPK-{ordinal:03d}"
        for member in cluster["members"]:
            assignments[member] = cluster_id
    return assignments


def apply_overrides(base_assignments, interval_fingerprints, override_ledger):
    """Apply deterministic merge/split overrides on top of base clustering.

    *base_assignments* maps the global interval ordinal (integer) to a base
    cluster id, exactly as produced by :func:`base_agglomerative`.  Both the
    incremental and full-rebuild paths call this function, so the final
    cluster identifiers are always allocated by the same rule: after every
    override, clusters are renumbered ``SPK-000...`` in ascending order of
    the minimum global frame key of their members.
    """
    fingerprint_by_ordinal = {
        idx: fp for idx, fp in enumerate(interval_fingerprints)
    }
    groups = {}
    for ordinal, base_id in base_assignments.items():
        fp = fingerprint_by_ordinal[ordinal]
        groups.setdefault(base_id, set()).add(fp)

    for entry in override_ledger:
        op = entry["op"]
        if op == "merge":
            targets = [
                cid for cid in entry["cluster_ids"] if cid in groups
            ]
            if len(targets) < 2:
                continue
            merged = set()
            for cid in targets:
                merged |= groups.pop(cid)
            groups[_fresh_id(groups)] = merged
        elif op == "split":
            source = entry["source_cluster_id"]
            peel = set(entry["interval_fingerprints"])
            if source not in groups:
                continue
            current = groups.pop(source)
            remaining = current - peel
            peeled = current & peel
            if remaining:
                groups[_fresh_id(groups)] = remaining
            if peeled:
                groups[_fresh_id(groups)] = peeled

    # Renumber deterministically by minimum global interval order key.
    order_index = {fp: idx for idx, fp in enumerate(interval_fingerprints)}
    cluster_mins = []
    for temp_id, members in groups.items():
        min_order = min(order_index[fp] for fp in members)
        cluster_mins.append((min_order, temp_id, members))
    cluster_mins.sort(key=lambda item: (item[0], item[1]))

    final_assignments = {}
    final_clusters = []
    for ordinal, (_, _, members) in enumerate(cluster_mins):
        cluster_id = f"SPK-{ordinal:03d}"
        ordered_members = sorted(members, key=lambda fp: order_index[fp])
        final_clusters.append(
            {
                "cluster_id": cluster_id,
                "interval_fingerprints": ordered_members,
                "member_count": len(ordered_members),
            }
        )
        for fp in ordered_members:
            final_assignments[fp] = cluster_id
    return final_assignments, final_clusters


def _fresh_id(groups):
    index = 0
    while f"ovr-{index:06d}" in groups:
        index += 1
    return f"ovr-{index:06d}"


def attribute_intervals(intervals, final_assignments, cluster_centroids,
                        interval_fingerprints, worker_pool=None):
    """Nearest-centroid attribution with lexicographic id tie-break."""

    def job(item):
        position, interval, fingerprint = item
        vector = np.asarray(interval["mean_voiceprint"], dtype=np.float64)
        forced_id = final_assignments.get(fingerprint)
        if forced_id is None:
            raise AttributionError(
                "interval missing cluster assignment",
                stage="attribution",
                interval_index=interval["interval_index"],
            )
        candidates = []
        for cluster_id, centroid in cluster_centroids.items():
            distance = _euclidean(vector, centroid)
            candidates.append((round6(distance), cluster_id))
        candidates.sort(key=lambda pair: (pair[0], pair[1]))
        nearest_distance, nearest_id = candidates[0]
        return {
            "sort_key": position,
            "interval_index": interval["interval_index"],
            "interval_fingerprint": fingerprint,
            "forced_cluster_id": forced_id,
            "nearest_cluster_id": nearest_id,
            "attribution_distance": nearest_distance,
        }

    items = list(
        zip(range(len(intervals)), intervals, interval_fingerprints)
    )
    if worker_pool is not None:
        rows = worker_pool.map_ordered(job, items, sort_key=lambda x: x[0])
    else:
        rows = [job(item) for item in items]
    return rows
