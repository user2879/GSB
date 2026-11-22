"""与分离真值的定量核验。"""

from .codes import E_TRUTH_MISMATCH, PipelineError


def _boundary_deviation_ms(truth_regions, speaker_intervals, cluster_product,
                           frame_lookup, n_frames, truth):
    """按说话人映射后，对重叠的真值/产出区间计算起止边界绝对偏差均值（毫秒）。"""
    mapping = _speaker_cluster_mapping(truth, cluster_product, frame_lookup, n_frames)
    inverse = {}
    for spk, cluster_id in mapping.items():
        inverse.setdefault(cluster_id, []).append(spk)
    deviations = []
    for spk, regions in truth_regions.items():
        cluster_id = mapping.get(spk)
        if cluster_id is None or cluster_id not in speaker_intervals:
            continue
        produced = sorted((iv["start_ms"], iv["end_ms"])
                          for iv in speaker_intervals[cluster_id])
        for start_ms, end_ms in regions:
            overlaps = [iv for iv in produced if iv[0] < end_ms and iv[1] > start_ms]
            if not overlaps:
                continue
            closest = min(overlaps, key=lambda iv: abs(iv[0] - start_ms) + abs(iv[1] - end_ms))
            deviations.append(abs(closest[0] - start_ms))
            deviations.append(abs(closest[1] - end_ms))
    if not deviations:
        return None, 0
    return round(sum(deviations) / len(deviations), 4), len(deviations) // 2


def _overlap_frame_accuracy(truth, cluster_product, frame_lookup, n_frames):
    overlap_frames = set()
    for start_ms, end_ms in truth["overlap_regions"]:
        # 仅纳入窗口中点落在真值重叠区内的帧，与主导说话人判定口径一致
        for frame_no, meta in frame_lookup.items():
            midpoint = (meta["start_ms"] + meta["end_ms"]) / 2.0
            if start_ms <= midpoint < end_ms:
                overlap_frames.add(frame_no)
    if not overlap_frames:
        return None, 0, 0
    adjudicated = {a["frame_index"]: a["assigned_cluster"]
                   for a in cluster_product["adjudications"]}
    # 真值重叠区的主导说话人：重叠窗口内增益最高的说话人（与混音构造一致）
    truth_dominant = _dominant_truth_speakers(truth, frame_lookup, n_frames)
    mapping = _speaker_cluster_mapping(truth, cluster_product, frame_lookup, n_frames)
    correct = 0
    total = 0
    first_wrong = None
    for frame_no in sorted(overlap_frames):
        total += 1
        expected_spk = truth_dominant.get(frame_no)
        assigned = adjudicated.get(frame_no)
        expected_cluster = mapping.get(expected_spk)
        if assigned is not None and assigned == expected_cluster:
            correct += 1
        elif first_wrong is None:
            first_wrong = frame_no
    return round(correct / total, 6), total, first_wrong


def _ms_to_frame(ms, frame_lookup, end=False):
    candidates = [idx for idx, meta in frame_lookup.items()
                  if meta["start_ms"] <= ms < meta["end_ms"]]
    if candidates:
        return min(candidates)
    return max(frame_lookup) if frame_lookup else 0


def _dominant_truth_speakers(truth, frame_lookup, n_frames):
    """重叠帧的主导说话人：取该帧中点处活跃说话人里标识最小者。

    合成数据中重叠窗的淡入说话人增益更高且标识顺序与时间线一致，
    因此该确定性规则与混音能量主导关系对齐。
    """
    dominant = {}
    for frame_no in range(n_frames):
        meta = frame_lookup.get(frame_no)
        if meta is None:
            continue
        mid = (meta["start_ms"] + meta["end_ms"]) / 2
        covering = []
        for spk, regions in truth["speaker_regions"].items():
            for start, end in regions:
                if start <= mid < end:
                    covering.append((start, spk))
        if len(covering) > 1:
            # 增益更高的当前独奏段(0.6)起始更早，淡入说话人(0.55)仅覆盖重叠尾部
            covering.sort()
            dominant[frame_no] = covering[0][1]
    return dominant


def _speaker_cluster_mapping(truth, cluster_product, frame_lookup, n_frames):
    """用簇内非重叠帧的真值说话人多数票建立 说话人 -> 簇 的确定性映射。"""
    frame_owner = {int(k): v for k, v in cluster_product["frame_owner"].items()}
    votes = {}
    for frame_no in range(n_frames):
        meta = frame_lookup.get(frame_no)
        if meta is None or frame_no not in frame_owner:
            continue
        mid = (meta["start_ms"] + meta["end_ms"]) / 2
        covering = []
        for spk, regions in truth["speaker_regions"].items():
            for start, end in regions:
                if start <= mid < end:
                    covering.append((start, spk))
        if not covering:
            continue
        covering.sort()
        # 重叠帧把票投给起始更早的主导说话人，使一个说话人映射到其主导簇
        spk = covering[0][1]
        votes.setdefault(spk, {}).setdefault(frame_owner[frame_no], 0)
        votes[spk][frame_owner[frame_no]] += 1
    mapping = {}
    for spk in sorted(votes):
        cluster_id, _ = sorted(votes[spk].items(), key=lambda item: (-item[1], item[0]))[0]
        mapping[spk] = cluster_id
    return mapping


def _confidence_calibration(cluster_product):
    """置信度校准偏差。

    非重叠帧为高置信基线（1.0），重叠裁决帧的置信度随裁决距离归一化，
    天然低于基线；校准偏差定义为相对该确定性参考曲线的平均绝对偏差。
    """
    adjudicated = {a["frame_index"]: a["confidence_float"]
                   for a in cluster_product["adjudications"]}
    values = []
    for intervals in cluster_product["speaker_intervals"].values():
        for interval in intervals:
            for frame_no in interval["frame_set"]:
                reference = adjudicated.get(frame_no, 1.0)
                values.append(abs(interval["confidence_float"] - reference))
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _diagnose_count_deviation(truth, cluster_product, segment_product, n_frames):
    """区分说话人数量偏差来源：切分 / 聚类合并 / 重叠归属。"""
    expected = truth["expected_speakers"]
    n_segments = len(segment_product["segments"])
    n_clusters = len(cluster_product["clusters"])
    first_divergence = None
    if n_segments < expected:
        source = "segmentation"
        covered = {f for seg in segment_product["segments"]
                   for f in range(seg["start_frame"], seg["end_frame"])}
        first_divergence = next((f for f in range(n_frames) if f not in covered), 0)
    elif n_clusters < expected:
        source = "cluster_merge"
        first_divergence = min((c["min_frame"] for c in cluster_product["clusters"]),
                               default=0)
    else:
        source = "overlap_attribution"
        first_divergence = next((a["frame_index"] for a in
                                 cluster_product["adjudications"]), 0)
    return source, first_divergence


def verify_audio(truth_entry, cluster_product, segment_product, config,
                 registry=None, audio_name=None, frame_lookup=None, n_frames=0):
    audio_name = audio_name or cluster_product["audio"]
    mismatch_item = None
    if registry is not None:
        truth_entry, mismatch_item = registry.effective_truth(audio_name, truth_entry)
        actual_speakers = len(truth_entry["speaker_regions"])
        if mismatch_item and truth_entry["expected_speakers"] != actual_speakers:
            raise PipelineError(
                E_TRUTH_MISMATCH, "verification",
                f"真值期望 {truth_entry['expected_speakers']} 个说话人，"
                f"音轨实际 {actual_speakers} 个",
                injection_id=mismatch_item["injection_id"],
            )

    boundary_dev, pairs = _boundary_deviation_ms(
        truth_entry["speaker_regions"], cluster_product["speaker_intervals"],
        cluster_product, frame_lookup or {}, n_frames, truth_entry,
    )
    overlap_acc, overlap_total, first_overlap_wrong = _overlap_frame_accuracy(
        truth_entry, cluster_product, frame_lookup or {}, n_frames,
    )
    calibration = _confidence_calibration(cluster_product)
    actual_count = len(cluster_product["clusters"])
    count_deviation = actual_count - truth_entry["expected_speakers"]
    diagnosis = None
    if count_deviation != 0:
        source, first_frame = _diagnose_count_deviation(
            truth_entry, cluster_product, segment_product, n_frames,
        )
        diagnosis = {"source": source, "first_divergence_frame": first_frame}
    return {
        "audio": audio_name,
        "expected_speakers": truth_entry["expected_speakers"],
        "actual_speakers": actual_count,
        "speaker_count_deviation": count_deviation,
        "boundary_deviation_ms": boundary_dev,
        "boundary_pairs": pairs,
        "overlap_frame_accuracy": overlap_acc,
        "overlap_total_frames": overlap_total,
        "first_overlap_wrong_frame": first_overlap_wrong,
        "confidence_calibration_deviation": calibration,
        "within_tolerance": bool(
            boundary_dev is None
            or (boundary_dev <= config.boundary_tolerance_ms
                and (calibration is None
                     or calibration <= config.confidence_tolerance))
        ),
        "diagnosis": diagnosis,
    }

