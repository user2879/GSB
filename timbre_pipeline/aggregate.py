"""说话人级描述向量加权聚合。

以段内有效帧数为权重，对量化整数向量做定点整数加权平均与确定性
半数向上舍入；逐维保留参与段清单与权重，单段说话人输出退化标记。
"""

from .utils import canonical_json, round_half_up, sha256_text


def aggregate_speakers(quantized, features, attach_result):
    """返回说话人级向量、权重明细与差分所需的旧值占位。"""
    feature_by_id = {item["segment_id"]: item for item in features}
    quant_by_id = {item["segment_id"]: item for item in quantized}
    groups = {}
    for assignment in attach_result["assignments"]:
        groups.setdefault(assignment["speaker_id"], []).append(assignment["segment_id"])
    speakers = []
    for speaker_id in sorted(groups):
        segment_ids = groups[speaker_id]
        weights = {segment_id: int(feature_by_id[segment_id]["valid_frames"]) for segment_id in segment_ids}
        total_weight = sum(weights.values()) or 1
        dim_count = len(quant_by_id[segment_ids[0]]["integer_vector"])
        integer_vector = []
        per_dimension = []
        for dim_index in range(dim_count):
            accumulator = 0
            for segment_id in segment_ids:
                accumulator += quant_by_id[segment_id]["integer_vector"][dim_index] * weights[segment_id]
            integer_value = int(round_half_up(accumulator / total_weight))
            integer_vector.append(integer_value)
            per_dimension.append({
                "dim_index": dim_index,
                "dim_name": quant_by_id[segment_ids[0]]["dimensions"][dim_index]["name"],
                "weights": [{"segment_id": sid, "weight": weights[sid],
                             "integer": quant_by_id[sid]["integer_vector"][dim_index]}
                            for sid in segment_ids],
                "aggregated_integer": integer_value,
            })
        reconstructed = [
            quant_by_id[segment_ids[0]]["dimensions"][i]["step"] * integer_vector[i]
            for i in range(dim_count)
        ]
        speakers.append({
            "speaker_id": speaker_id,
            "segment_ids": segment_ids,
            "weights": weights,
            "total_weight": total_weight,
            "integer_vector": integer_vector,
            "descriptor_vector": [round_safe(v) for v in reconstructed],
            "per_dimension": per_dimension,
            "degraded_single_segment": len(segment_ids) == 1,
        })
    result = {
        "audio_id": attach_result["audio_id"],
        "speakers": speakers,
    }
    result["aggregate_fingerprint"] = aggregate_fingerprint(
        attach_result["attach_fingerprint"], result)
    return result


def round_safe(value):
    from .utils import round_fixed
    return round_fixed(value)


def aggregate_fingerprint(attach_hash, result):
    payload = {
        "attach_fingerprint": attach_hash,
        "speakers": [
            {"speaker_id": item["speaker_id"],
             "segment_ids": item["segment_ids"],
             "weights": item["weights"],
             "integer_vector": item["integer_vector"]}
            for item in result["speakers"]
        ],
    }
    return "aggr_" + sha256_text(canonical_json(payload))[:16]
