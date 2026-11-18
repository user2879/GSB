"""跨模态对齐打分与检索索引。

图像块特征按 [均值(3), 梯度强度(1), 方向直方图(B)] 切片，文本向量按槽位
折叠到相同维度；按对齐权重计算余弦相似度后做减去最大值的稳定 softmax
归一化，再加权汇总为确定性标量对齐得分。
"""

import numpy as np

from .hashing import digest_array, digest_json

MEAN_DIM = 3
GRAD_DIM = 1


def _fold_text(text_vector, target_dim):
    """把 T 维文本特征确定性折叠到目标维度（保留向量结构而非截断）。"""
    text_vector = np.asarray(text_vector, dtype=np.float32)
    folded = np.zeros(target_dim, dtype=np.float32)
    for idx, value in enumerate(text_vector):
        folded[idx % target_dim] += value
    norm = float(np.linalg.norm(folded))
    if norm > 0.0:
        folded = folded / norm
    return folded


def _slice_components(image_blocks, orientation_bins):
    grad_end = MEAN_DIM + GRAD_DIM
    return {
        "mean": image_blocks[:, :MEAN_DIM],
        "gradient": image_blocks[:, MEAN_DIM:grad_end],
        "orientation": image_blocks[:, grad_end:grad_end + orientation_bins],
    }


def _stable_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros_like(values)
    return exp / total


def _cosine_rows(matrix, vector):
    mat = matrix.astype(np.float64)
    vec = vector.astype(np.float64)
    mat_norm = np.linalg.norm(mat, axis=1)
    vec_norm = np.linalg.norm(vec)
    denom = mat_norm * vec_norm
    out = np.zeros(mat.shape[0], dtype=np.float64)
    valid = denom > 0.0
    out[valid] = (mat[valid] @ vec) / denom[valid]
    return out


def alignment_score(record, orientation_bins):
    """返回 (rounded_score, 各组件明细)。"""
    weights = record.align_weights
    components = _slice_components(record.image_blocks, orientation_bins)
    component_scores = {}
    total_weight = 0.0
    weighted_sum = 0.0
    for comp_name in ("mean", "gradient", "orientation"):
        matrix = components[comp_name]
        folded = _fold_text(record.text_vector, matrix.shape[1])
        cosines = _cosine_rows(matrix, folded)
        block_weights = _stable_softmax(cosines)
        component_scores[comp_name] = float(
            np.dot(block_weights, cosines))
        weight = float(weights[comp_name])
        weighted_sum += weight * component_scores[comp_name]
        total_weight += weight
    text_norm = float(np.linalg.norm(record.text_vector.astype(np.float64)))
    text_score = text_norm / np.sqrt(max(1.0, record.text_vector.shape[0]))
    weighted_sum += float(weights["text"]) * text_score
    total_weight += float(weights["text"])
    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    score = float(np.clip(score, -1.0, 1.0))
    rounded = round(float(score), 9)
    detail = {k: round(v, 9) for k, v in component_scores.items()}
    detail["text"] = round(text_score, 9)
    return rounded, detail


def build_scores(records, orientation_bins):
    """records: 样本名 -> FeatureRecord，返回样本名 -> 得分。"""
    return {sample: alignment_score(record, orientation_bins)[0]
            for sample, record in sorted(records.items())}


def search(score_map, top_k, threshold, query_sample=None):
    """对单条文本查询在图像库中检索。

    每条文本查询天然对应同一样本的图像，因此该查询的目标命中得分取其
    同名样本的对齐得分；候选清单仍按全库得分降序排列，同分按样本名
    字典序决断，供审计使用。
    """
    ordered = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    candidates = [
        {"sample": sample, "score": round(float(score), 9)}
        for sample, score in ordered
    ]
    top = candidates[:top_k]
    if query_sample is not None and query_sample in score_map:
        top_score = round(float(score_map[query_sample]), 9)
    else:
        top_score = candidates[0]["score"] if candidates else 0.0
    low_score = top_score < threshold
    return {
        "candidates": top,
        "all_candidates": candidates,
        "top_score": top_score,
        "threshold": threshold,
        "low_score": low_score,
        "code": "E_RETRIEVAL_LOW_SCORE" if low_score else None,
    }


def score_fingerprint(score_map):
    return digest_json({k: round(float(v), 9)
                        for k, v in sorted(score_map.items())},
                       prefix="scores")


def record_vector_fingerprint(record):
    return digest_array(record.image_blocks, "v-img") + ":" \
        + digest_array(record.text_vector, "v-txt")
