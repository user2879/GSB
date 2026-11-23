"""段级声学特征汇聚。

帧严格按帧序号升序处理，只累加该段自身的帧；分位数使用配置中的
固定插值规则；段特征指纹由段内容指纹与 feature 阶段配置指纹派生。
"""

import numpy as np

from .errors import PipelineError, E_INJECT_NON_FINITE, E_STAGE_FAILED
from .utils import canonical_json, quantile, round_fixed, sha256_text

F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
VOICED_ENERGY_RATIO = 0.12
LPC_ORDER = 10
FORMANT_COUNT = 3


def _frame_energy(frame):
    energy = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return float(energy)


def _raw_autocorr_peak(frame, sample_rate):
    signal = frame.astype(np.float64)
    signal = signal - np.mean(signal)
    norm = np.dot(signal, signal)
    if norm <= 0.0:
        return None, 0.0
    min_lag = max(2, int(sample_rate / F0_MAX_HZ))
    max_lag = min(len(signal) - 2, int(sample_rate / F0_MIN_HZ))
    if max_lag <= min_lag:
        return None, 0.0
    corr = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(min_lag, max_lag + 1):
        corr[lag] = np.dot(signal[:-lag], signal[lag:]) / norm
    peak_index = int(np.argmax(corr[min_lag:max_lag + 1])) + min_lag
    return peak_index, float(corr[peak_index])


def estimate_f0(frame, sample_rate, anchor_lag=None):
    """带锚点的自相关基频估计，返回 (f0, strength, lag)。

    若提供段内锚点周期，则在锚点及其半/双周期邻域内选峰，从根源上
    消除 116Hz 与 380Hz 之间的八度翻转；无锚点时回退到全局最大峰。
    """
    peak_index, strength = _raw_autocorr_peak(frame, sample_rate)
    if peak_index is None or strength < 0.3:
        return None, strength, peak_index
    if anchor_lag is not None:
        min_lag = max(2, int(sample_rate / F0_MAX_HZ))
        max_lag = min(len(frame) - 2, int(sample_rate / F0_MIN_HZ))
        candidates = [anchor_lag, anchor_lag // 2, anchor_lag * 2]
        best_lag, best_strength = peak_index, strength
        for lag in candidates:
            for probe in range(max(min_lag, lag - 3), min(max_lag, lag + 3) + 1):
                corr_strength = _corr_at_lag(frame, probe)
                if corr_strength > best_strength * 0.85 and abs(probe - anchor_lag) <= abs(best_lag - anchor_lag):
                    best_lag, best_strength = probe, corr_strength
        peak_index = best_lag
        strength = best_strength
    return float(sample_rate / peak_index), strength, peak_index


def _corr_at_lag(frame, lag):
    signal = frame.astype(np.float64)
    signal = signal - np.mean(signal)
    norm = np.dot(signal, signal)
    if norm <= 0 or lag <= 0 or lag >= len(signal):
        return 0.0
    return float(np.dot(signal[:-lag], signal[lag:]) / norm)


def resolve_anchor_lag(frame_table, sample_rate):
    """加权直方图选最长周期主峰作为段锚点。

    周期簇按 ±2 采样归并；只保留落在最大簇周期 0.45 倍以上的候选，
    再取加权质量最高者，从规则上排除半周期（八度上跳）峰。
    """
    histogram = {}
    for record in frame_table:
        lag = record.get("raw_lag")
        strength = record.get("raw_strength", 0.0)
        if lag is None or strength < 0.3:
            continue
        cluster = min(histogram, key=lambda key: abs(key - lag), default=None)
        if cluster is not None and abs(cluster - lag) <= 2:
            histogram[cluster] += float(strength)
        else:
            histogram[int(lag)] = float(strength)
    if not histogram:
        return None
    longest = max(histogram)
    accepted = {lag: weight for lag, weight in histogram.items()
                if lag >= 0.45 * longest}
    return max(accepted, key=lambda key: (round(accepted[key], 9), key))


def burg_lpc(frame, order):
    """Burg 法估计 LPC 系数，全固定运算顺序、无随机分支。"""
    signal = frame.astype(np.float64)
    signal = signal - np.mean(signal)
    count = len(signal)
    if count <= order + 1:
        return np.array([1.0], dtype=np.float64)
    forward = signal[1:].copy()
    backward = signal[:-1].copy()
    coefficients = np.array([1.0], dtype=np.float64)
    for _ in range(order):
        denominator = float(np.dot(forward, forward) + np.dot(backward, backward))
        if denominator <= 0.0:
            break
        reflection = 2.0 * float(np.dot(forward, backward)) / denominator
        new_coeff = np.concatenate([coefficients, [0.0]])
        new_coeff = new_coeff - reflection * new_coeff[::-1]
        coefficients = new_coeff
        new_forward = forward - reflection * backward
        new_backward = backward - reflection * forward
        forward = new_forward[1:]
        backward = new_backward[:-1]
    return coefficients


def estimate_formants(frame, sample_rate, formant_count=FORMANT_COUNT):
    """由 LPC 多项式根解出共振峰中心频率，按频率升序固定输出。"""
    coefficients = burg_lpc(frame, LPC_ORDER)
    if len(components_roots(coefficients)) == 0:
        return [0.0] * formant_count
    roots = components_roots(coefficients)
    frequencies = []
    for root in roots:
        angle = np.angle(root)
        if angle <= 0.0:
            continue
        frequency = angle * sample_rate / (2.0 * np.pi)
        if F0_MIN_HZ <= frequency <= sample_rate / 2.0 - 100.0:
            frequencies.append(float(frequency))
    frequencies.sort()
    while len(frequencies) < formant_count:
        frequencies.append(0.0)
    return [round_fixed(v) for v in frequencies[:formant_count]]


def components_roots(coefficients):
    """companion 矩阵特征值求根，集中封装便于确定性校验。"""
    if len(coefficients) <= 1:
        return np.array([], dtype=np.complex128)
    companion = np.zeros((len(coefficients) - 1, len(coefficients) - 1), dtype=np.float64)
    companion[0, :] = -coefficients[1:] / coefficients[0]
    for index in range(1, len(coefficients) - 1):
        companion[index, index - 1] = 1.0
    return np.linalg.eigvals(companion)


def count_syllable_peaks(energies):
    """固定规则局部峰计数：幅度高于段均值 25% 且峰间距 >=3 帧。"""
    if not energies:
        return 0, []
    baseline = sum(energies) / len(energies)
    threshold = baseline * 1.25
    peaks = []
    last_peak = -3
    for index in range(1, len(energies) - 1):
        if energies[index] >= threshold and energies[index] > energies[index - 1] \
                and energies[index] >= energies[index + 1] and index - last_peak >= 3:
            peaks.append(index)
            last_peak = index
    return len(peaks), peaks


def _stats(values, quantile_method):
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "q25": 0.0,
                "median": 0.0, "q75": 0.0, "max": 0.0}
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    variance = sum((v - mean) ** 2 for v in ordered) / len(ordered)
    return {
        "mean": round_fixed(mean),
        "std": round_fixed(np.sqrt(variance)),
        "min": round_fixed(ordered[0]),
        "q25": round_fixed(quantile(ordered, 0.25, quantile_method)),
        "median": round_fixed(quantile(ordered, 0.50, quantile_method)),
        "q75": round_fixed(quantile(ordered, 0.75, quantile_method)),
        "max": round_fixed(ordered[-1]),
    }


def compute_segment_feature(config, segment, samples, injection=None):
    """计算单段特征；注入 non_finite 时抛出可隔离业务异常。"""
    frame_length = config.frame_length()
    hop_length = config.hop_length()
    method = config.get("quantile_method")
    segment_samples = samples[segment["start_sample"]:segment["end_sample"]]
    frame_windows = []
    frame_index = segment["frame_start"]
    local_index = 0
    for start in range(0, len(segment_samples) - frame_length + 1, hop_length):
        frame = segment_samples[start:start + frame_length].astype(np.float64)
        energy = _frame_energy(frame)
        raw_lag, raw_strength = _raw_autocorr_peak(frame, config.get("sample_rate"))
        frame_windows.append({
            "frame": frame, "energy": energy, "raw_lag": raw_lag,
            "raw_strength": raw_strength, "frame_index": frame_index,
            "local_index": local_index,
        })
        frame_index += 1
        local_index += 1
    anchor_lag = resolve_anchor_lag(frame_windows, config.get("sample_rate"))
    frame_records = []
    energies = []
    f0_values = []
    strengths = []
    formant_table = []
    for window in frame_windows:
        frame = window["frame"]
        energy = window["energy"]
        if injection and injection.get("type") == "non_finite" \
                and window["local_index"] == int(injection.get("target_frame", 0)):
            frame = frame.copy()
            frame[0] = np.nan
            energy = float(np.sqrt(np.mean(frame ** 2)))
        f0, strength, chosen_lag = estimate_f0(
            frame, config.get("sample_rate"), anchor_lag)
        if f0 is not None and anchor_lag is not None and (
                chosen_lag < 0.75 * anchor_lag or chosen_lag > 1.25 * anchor_lag):
            f0, strength = None, strength
        formants = estimate_formants(frame, config.get("sample_rate")) if energy > 0 else [0.0] * FORMANT_COUNT
        if not np.isfinite(energy) or any(not np.isfinite(v) for v in formants) \
                or (f0 is not None and not np.isfinite(f0)):
            raise PipelineError(
                E_INJECT_NON_FINITE,
                "段 {} 帧 {} 出现非有限采样值".format(segment["segment_id"], window["frame_index"]),
                "feature", segment["segment_id"],
                extra={"injection_id": injection.get("id"),
                       "frame_index": window["frame_index"]},
            )
        record = {
            "frame_index": window["frame_index"],
            "local_index": window["local_index"],
            "energy": round_fixed(energy),
            "f0_hz": None if f0 is None else round_fixed(f0),
            "f0_strength": round_fixed(strength),
            "raw_lag": window["raw_lag"],
            "raw_strength": round_fixed(window["raw_strength"]),
            "anchor_lag": anchor_lag,
            "formants_hz": formants,
            "voiced": bool(energy > 0 and f0 is not None),
        }
        frame_records.append(record)
        energies.append(energy)
        if f0 is not None:
            f0_values.append(f0)
            strengths.append(strength)
        formant_table.append(formants)
    if not frame_records:
        raise PipelineError(E_STAGE_FAILED, "段内没有可处理帧", "feature", segment["segment_id"])
    peak_energy = max(energies) or 1.0
    normalized = [energy / peak_energy for energy in energies]
    valid = [record for record in frame_records
             if record["voiced"] and normalized[record["local_index"]] >= VOICED_ENERGY_RATIO]
    duration_s = len(segment_samples) / config.get("sample_rate")
    syllable_count, peak_positions = count_syllable_peaks(normalized)
    formant_columns = list(zip(*formant_table)) if formant_table else []
    formant_stats = []
    for column in formant_columns[:FORMANT_COUNT]:
        nonzero = [v for v in column if v > 0]
        formant_stats.append(_stats(nonzero, method))
    f0_stats = _stats(f0_values, method)
    energy_stats = _stats(energies, method)
    energy_mean = energy_stats["mean"]
    energy_cv = round_fixed(energy_stats["std"] / energy_mean) if energy_mean > 0 else 0.0
    descriptor_inputs = {
        "f0_median": f0_stats["median"],
        "f0_iqr": round_fixed(f0_stats["q75"] - f0_stats["q25"]),
        "f1_median": formant_stats[0]["median"] if formant_stats else 0.0,
        "f2_median": formant_stats[1]["median"] if len(formant_stats) > 1 else 0.0,
        "f3_median": formant_stats[2]["median"] if len(formant_stats) > 2 else 0.0,
        "syllable_rate": round_fixed(syllable_count / duration_s) if duration_s > 0 else 0.0,
        "voiced_ratio": round_fixed(len(valid) / len(frame_records)),
        "energy_cv": energy_cv,
    }
    feature = {
        "segment_id": segment["segment_id"],
        "content_fingerprint": segment["content_fingerprint"],
        "valid_frames": len(valid),
        "total_frames": len(frame_records),
        "duration_s": round_fixed(duration_s),
        "f0_stats": f0_stats,
        "formant_stats": formant_stats,
        "energy_stats": energy_stats,
        "syllable_peak_count": syllable_count,
        "syllable_peak_frames": peak_positions,
        "descriptor_inputs": {key: round_fixed(value) for key, value in descriptor_inputs.items()},
        "frame_table": frame_records,
    }
    feature["feature_fingerprint"] = feature_fingerprint(
        segment["content_fingerprint"], config.stage_fingerprint("feature"), feature)
    return feature


def feature_fingerprint(content_fingerprint, stage_config_hash, feature):
    payload = {
        "content_fingerprint": content_fingerprint,
        "stage_config_hash": stage_config_hash,
        "valid_frames": feature["valid_frames"],
        "f0_stats": feature["f0_stats"],
        "formant_stats": feature["formant_stats"],
        "energy_stats": feature["energy_stats"],
        "syllable_peak_count": feature["syllable_peak_count"],
        "descriptor_inputs": feature["descriptor_inputs"],
    }
    return "feat_" + sha256_text(canonical_json(payload))[:16]
