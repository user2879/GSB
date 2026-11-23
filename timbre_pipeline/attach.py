"""段间归属一致判定。

相邻段使用归一化 L1 相似度（定点整数向量计算），相似度恰等于阈值时
按配置 ``threshold_tie_rule`` 固定决断并标记边界样本；预测归属与真值
说话人标注不一致时输出冲突记录，绝不静默覆盖。
"""

from .errors import PipelineError, E_INJECT_THRESHOLD_RANGE, E_SEGMENT_OVERFLOW
from .utils import canonical_json, round_fixed, sha256_text


TIMBRE_DIMENSIONS = {"f0_median", "f0_iqr", "f1_median", "f2_median", "f3_median"}
TIMBRE_WEIGHT = 0.96
STYLE_WEIGHT = 0.04


def normalized_l1_similarity(left, right, names=None):
    """[0,1] 加权相似度：说话人音色维主导，节奏风格维低权重。"""
    if len(left) != len(right):
        raise ValueError("描述向量维度不一致，无法计算相似度")
    if not left:
        return 1.0
    names = names or []
    timbre_total = 0.0
    timbre_count = 0
    style_total = 0.0
    style_count = 0
    for index, (a, b) in enumerate(zip(left, right)):
        scale = max(1.0, float(max(abs(a), abs(b))))
        score = 1.0 - abs(a - b) / scale
        name = names[index] if index < len(names) else ""
        if name in TIMBRE_DIMENSIONS:
            timbre_total += score
            timbre_count += 1
        else:
            style_total += score
            style_count += 1
    timbre_part = timbre_total / timbre_count if timbre_count else 1.0
    style_part = style_total / style_count if style_count else 1.0
    return round_fixed(TIMBRE_WEIGHT * timbre_part + STYLE_WEIGHT * style_part)


def _tie_decision(similarity, threshold, rule):
    if similarity > threshold:
        return True, False
    if similarity < threshold:
        return False, False
    return rule == "merge", True


def attach_segments(config, audio_id, quantized, truth_by_segment=None, injection=None):
    """对同一音频的段按时间顺序做归属聚类。"""
    threshold = config.get("similarity_threshold")
    if injection and injection.get("type") == "threshold_out_of_range":
        threshold = float(injection.get("threshold", threshold))
        if not 0.0 <= threshold <= 1.0:
            raise PipelineError(
                E_INJECT_THRESHOLD_RANGE,
                "归属阈值越界: {}".format(threshold),
                "attach", audio_id,
                extra={"injection_id": injection.get("id"), "threshold": threshold},
            )
    rule = config.get("threshold_tie_rule")
    max_consecutive = int(config.get("max_consecutive_segments"))
    truth_by_segment = truth_by_segment or {}
    assignments = []
    similarities = []
    merge_trace = []
    conflicts = []
    boundary_segments = []
    cluster_run = 0
    cluster_index = 0
    for index, item in enumerate(quantized):
        segment_id = item["segment_id"]
        if index == 0:
            speaker_id = "{}#S{:03d}".format(audio_id, cluster_index)
            similarity = None
            decision = "start"
            cluster_run = 1
        else:
            dim_names = [dim["name"] for dim in item["dimensions"]]
            similarity = normalized_l1_similarity(
                quantized[index - 1]["integer_vector"], item["integer_vector"], dim_names)
            should_merge, is_boundary = _tie_decision(similarity, threshold, rule)
            if is_boundary:
                boundary_segments.append(segment_id)
            if should_merge:
                speaker_id = assignments[-1]["speaker_id"]
                decision = "merge"
                cluster_run += 1
            else:
                cluster_index += 1
                speaker_id = "{}#S{:03d}".format(audio_id, cluster_index)
                decision = "split"
                cluster_run = 1
            merge_trace.append({
                "left": quantized[index - 1]["segment_id"],
                "right": segment_id,
                "similarity": similarity,
                "threshold": round_fixed(threshold),
                "decision": decision,
                "boundary": is_boundary,
            })
        if cluster_run > max_consecutive:
            raise PipelineError(
                E_SEGMENT_OVERFLOW,
                "连续同归属段数 {} 超过容许上限 {}".format(cluster_run, max_consecutive),
                "attach", speaker_id,
                extra={"segment_id": segment_id, "consecutive": cluster_run},
            )
        assignment = {
            "segment_id": segment_id,
            "speaker_id": speaker_id,
            "similarity_to_previous": similarity,
            "decision": decision,
            "boundary_sample": segment_id in boundary_segments,
        }
        truth_speaker = truth_by_segment.get(index)
        if truth_speaker is not None:
            assignment["truth_speaker"] = truth_speaker
            expected_speaker = truth_by_segment.get(0)
            consistent = (index == 0) or (truth_speaker == expected_speaker and decision == "merge") \
                or (truth_speaker != expected_speaker and decision == "split")
            assignment["label_consistent"] = bool(consistent)
            if not consistent:
                conflicts.append({
                    "segment_id": segment_id,
                    "predicted_decision": decision,
                    "truth_speaker": truth_speaker,
                    "previous_truth_speaker": truth_by_segment.get(index - 1),
                    "rule": "keep_prediction_emit_conflict",
                })
        assignments.append(assignment)
        similarities.append(similarity)
    result = {
        "audio_id": audio_id,
        "threshold": round_fixed(threshold),
        "tie_rule": rule,
        "assignments": assignments,
        "merge_trace": merge_trace,
        "conflicts": conflicts,
        "boundary_segments": boundary_segments,
        "speaker_ids": sorted({item["speaker_id"] for item in assignments}),
    }
    result["attach_fingerprint"] = attach_fingerprint(
        [item["quant_fingerprint"] for item in quantized],
        config.stage_fingerprint("attach"), result)
    return result


def attach_fingerprint(quant_hashes, stage_config_hash, result):
    payload = {
        "quant_fingerprints": quant_hashes,
        "stage_config_hash": stage_config_hash,
        "assignments": [
            {"segment_id": item["segment_id"], "speaker_id": item["speaker_id"],
             "similarity_to_previous": item["similarity_to_previous"],
             "decision": item["decision"], "boundary_sample": item["boundary_sample"]}
            for item in result["assignments"]
        ],
    }
    return "attach_" + sha256_text(canonical_json(payload))[:16]
