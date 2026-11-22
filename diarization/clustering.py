"""Window similarity and incremental clustering with merge-gate hysteresis."""
import math


def similarity(a, b):
    """Cosine similarity, accumulated in ascending dimension order."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a)):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _mean_embedding(members, embeddings):
    acc = None
    for idx in sorted(members):
        vec = embeddings[idx]
        acc = vec.copy() if acc is None else acc + vec
    return acc / float(len(members))


def cluster_windows(embeddings, threshold, hysteresis):
    """Incrementally assign windows to clusters.

    Clusters are probed in lexicographic id order and the first cluster
    meeting the threshold wins; similarity exactly equal to the threshold
    counts as same-speaker. Windows whose best similarity falls inside the
    hysteresis band [threshold - hysteresis, threshold) stay unassigned
    instead of being silently absorbed. A post-pass merge gate merges
    cluster pairs at or above the threshold and records pairs inside the
    hysteresis band as boundary events.
    """
    clusters = {}
    assignments = {}
    unassigned = []
    hysteresis_events = []
    decision_margins = {}
    counter = 0
    for idx in sorted(embeddings):
        emb = embeddings[idx]
        chosen = None
        best_sim = None
        best_cid = None
        for cid in sorted(clusters):
            sim = similarity(emb, clusters[cid]["rep"])
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_cid = cid
            if sim >= threshold:
                chosen = cid
                break
        decision_margins[idx] = best_sim
        if chosen is not None:
            assignments[idx] = chosen
            clusters[chosen]["members"].append(idx)
            clusters[chosen]["rep"] = _mean_embedding(
                clusters[chosen]["members"], embeddings)
        elif clusters and best_sim is not None and best_sim >= threshold - hysteresis:
            unassigned.append(idx)
            hysteresis_events.append({
                "window_index": idx,
                "nearest_cluster": best_cid,
                "similarity": best_sim,
            })
        else:
            cid = "SPK_%03d" % counter
            counter += 1
            clusters[cid] = {"rep": emb.copy(), "members": [idx]}
            assignments[idx] = cid

    merge_events = []
    merge_boundary_events = []
    ids = sorted(clusters)
    parent = {cid: cid for cid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            root_a = find(ids[i])
            root_b = find(ids[j])
            if root_a == root_b:
                continue
            sim = similarity(clusters[ids[i]]["rep"], clusters[ids[j]]["rep"])
            if sim >= threshold:
                high, low = max(root_a, root_b), min(root_a, root_b)
                parent[high] = low
                merge_events.append({"from": high, "into": low, "similarity": sim})
            elif sim >= threshold - hysteresis:
                merge_boundary_events.append({
                    "cluster_a": root_a, "cluster_b": root_b, "similarity": sim})

    for idx in assignments:
        assignments[idx] = find(assignments[idx])
    merged = {}
    for cid in ids:
        merged.setdefault(find(cid), []).extend(clusters[cid]["members"])
    cluster_info = {}
    for root in sorted(merged):
        members = sorted(merged[root])
        cluster_info[root] = {
            "members": members,
            "rep": _mean_embedding(members, embeddings),
        }
    return {
        "assignments": assignments,
        "clusters": cluster_info,
        "unassigned": unassigned,
        "hysteresis_events": hysteresis_events,
        "merge_events": merge_events,
        "merge_boundary_events": merge_boundary_events,
        "decision_margins": decision_margins,
    }
