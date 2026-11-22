"""确定性凝聚聚类、重叠区说话人归属裁决与定点置信度。

聚类（run_clustering）与归属裁决（run_attribution）拆为两个独立阶段，
使 overlap_rule 变更只令归属及其下游失效，聚类产物可直接复用。
"""

from .util import l1_distance, mean_quantized, quantize_scalar
from .injection import InjectionRegistry


def _cluster_segments(segments, config):
    """自底向上的确定性凝聚聚类。

    距离同分（全最小距离集合）时按簇内最小帧序号决断；
    距离恰等于阈值时不合并（严格小于才合并）。
    """
    clusters = []
    for segment in segments:
        clusters.append({
            "id": f"CL{len(clusters):02d}",
            "members": [segment["segment_index"]],
            "centroid": list(segment["mean_cepstrum_q"]),
            "min_frame": segment["start_frame"],
        })
    threshold_q = int(round(config.cluster_distance_threshold * (2 ** config.frac_bits)))
    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = l1_distance(clusters[i]["centroid"], clusters[j]["centroid"])
                if distance >= threshold_q:
                    continue
                tie_key = (min(clusters[i]["min_frame"], clusters[j]["min_frame"]),
                           clusters[i]["id"], clusters[j]["id"])
                candidate = (distance, tie_key, i, j)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            break
        _, _, i, j = best
        left, right = clusters[i], clusters[j]
        members = sorted(left["members"] + right["members"])
        matrix = [segments[m]["mean_cepstrum_q"] for m in members]
        merged = {
            "id": f"CL{len(clusters):02d}",
            "members": members,
            "centroid": [int(x) for x in mean_quantized(matrix)],
            "min_frame": min(left["min_frame"], right["min_frame"]),
        }
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    clusters.sort(key=lambda c: c["min_frame"])
    for new_index, cluster in enumerate(clusters):
        cluster["id"] = f"CL{new_index:02d}"
    segment_cluster = {}
    for cluster in clusters:
        for member in cluster["members"]:
            segment_cluster[member] = cluster["id"]
    return clusters, segment_cluster


def _frame_distance(frame, centroid):
    return l1_distance(frame["cepstrum_q"], centroid)


def _adjudicate(frame, candidate_ids, centroids, rule, fallback_cluster=None):
    distances = {cid: _frame_distance(frame, centroids[cid]) for cid in candidate_ids}
    ordered = sorted(candidate_ids, key=lambda cid: (distances[cid], cid))
    nearest = ordered[0]
    nearest_dist = distances[nearest]
    if rule == "max_margin":
        # 最大余量策略：重叠帧优先归属进入重叠前的前一说话人簇，
        # 仅当前一簇不在候选集时退化为最近簇；与 nearest 的逐帧最近
        # 裁决形成确定性、可验证的差异。距离仍完整记录用于审计。
        if fallback_cluster is not None and fallback_cluster in candidate_ids:
            chosen = fallback_cluster
        else:
            chosen = nearest
    else:  # nearest：距离相同按簇标识字典序决断
        chosen = nearest
    farthest = max(distances.values())
    confidence = 1.0 if farthest <= 0 else 1.0 - nearest_dist / (farthest + 1)
    return chosen, nearest_dist, confidence, distances


def run_clustering(segment_product, config, registry=None, audio_name=None):
    """阶段三：仅产出说话人簇与段->簇映射（不做逐帧归属）。"""
    audio_name = audio_name or segment_product["audio"]
    if registry is not None:
        config, item = registry.effective_config(audio_name, config)
        InjectionRegistry.validate_threshold(
            config, audio_name, injection_id=item["injection_id"] if item else None)
    else:
        InjectionRegistry.validate_threshold(config, audio_name)

    segments = segment_product["segments"]
    clusters, segment_cluster = _cluster_segments(segments, config)
    return {
        "audio": audio_name,
        "clusters": clusters,
        "segment_cluster": {str(k): v for k, v in sorted(segment_cluster.items())},
        "threshold_q": int(round(config.cluster_distance_threshold * (2 ** config.frac_bits))),
    }


def run_attribution(segment_product, cluster_product, config, registry=None, audio_name=None):
    """阶段四：重叠区逐帧归属裁决与每说话人定点置信度区间。"""
    audio_name = audio_name or segment_product["audio"]
    segments = segment_product["segments"]
    frames_by_index = {f["frame_index"]: f for f in segment_product.get("_frames", [])}
    clusters = cluster_product["clusters"]
    segment_cluster = {int(k): v for k, v in cluster_product["segment_cluster"].items()}
    centroids = {c["id"]: c["centroid"] for c in clusters}

    adjudications = []
    frame_owner = {}
    frame_confidence = {}
    for segment in segments:
        cluster_id = segment_cluster[segment["segment_index"]]
        for frame_no in range(segment["start_frame"], segment["end_frame"]):
            frame_owner[frame_no] = cluster_id
            frame_confidence[frame_no] = 1.0

    for segment in segments:
        if not segment["overlap_candidate"]:
            continue
        prev = segments[segment["segment_index"] - 1]
        candidate_ids = sorted({
            segment_cluster[prev["segment_index"]],
            segment_cluster[segment["segment_index"]],
        })
        fallback = segment_cluster[prev["segment_index"]]
        for frame_no in range(segment["start_frame"], segment["end_frame"]):
            frame = frames_by_index[frame_no]
            chosen, distance, confidence, all_distances = _adjudicate(
                frame, candidate_ids, centroids, config.overlap_rule,
                fallback_cluster=fallback,
            )
            frame_owner[frame_no] = chosen
            frame_confidence[frame_no] = confidence
            adjudications.append({
                "frame_index": frame_no,
                "sort_key": f"adjudication:{frame_no:08d}",
                "rule": config.overlap_rule,
                "candidate_clusters": candidate_ids,
                "distances": all_distances,
                "assigned_cluster": chosen,
                "confidence_float": round(confidence, 6),
            })

    speaker_intervals = {}
    quant_step = None
    for frame_no in sorted(frame_owner):
        confidence_float = frame_confidence[frame_no]
        confidence_q, quant_step, residual = quantize_scalar(confidence_float, config.frac_bits)
        cluster_id = frame_owner[frame_no]
        intervals = speaker_intervals.setdefault(cluster_id, [])
        if intervals and frame_no == intervals[-1]["end_frame"]:
            intervals[-1]["end_frame"] = frame_no + 1
            intervals[-1]["conf_q_sum"] += confidence_q
            intervals[-1]["conf_res_sum"] += residual
            intervals[-1]["conf_count"] += 1
        else:
            intervals.append({
                "start_frame": frame_no,
                "end_frame": frame_no + 1,
                "conf_q_sum": confidence_q,
                "conf_res_sum": residual,
                "conf_count": 1,
            })

    final_intervals = {}
    for cluster_id, intervals in speaker_intervals.items():
        final_intervals[cluster_id] = []
        for record in intervals:
            mean_q = record["conf_q_sum"] // max(1, record["conf_count"])
            final_intervals[cluster_id].append({
                "start_frame": record["start_frame"],
                "end_frame": record["end_frame"],
                "start_ms": _frame_ms(segment_product, record["start_frame"], "start_ms"),
                "end_ms": _frame_ms(segment_product, record["end_frame"] - 1, "end_ms"),
                "confidence_q": int(mean_q),
                "confidence_float": round(mean_q * quant_step, 6),
                "quant_step": quant_step,
                "frac_bits": config.frac_bits,
                "residual_sum": round(record["conf_res_sum"], 8),
                "frame_set": list(range(record["start_frame"], record["end_frame"])),
                "source_cluster": cluster_id,
            })

    return {
        "audio": audio_name,
        "clusters": clusters,
        "segment_cluster": cluster_product["segment_cluster"],
        "adjudications": adjudications,
        "speaker_intervals": final_intervals,
        "frame_owner": {str(k): v for k, v in sorted(frame_owner.items())},
        "threshold_q": cluster_product["threshold_q"],
        "overlap_rule": config.overlap_rule,
    }


def _frame_ms(segment_product, frame_no, key):
    frame_lookup = segment_product.get("_frame_lookup", {})
    if frame_no in frame_lookup:
        return frame_lookup[frame_no][key]
    return None
