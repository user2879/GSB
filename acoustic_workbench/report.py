"""Deterministic JSON and Markdown reports derived only from snapshots."""

from __future__ import annotations

from typing import Any

from .hashing import content_hash
from .snapshot import Snapshot, deep_thaw


def build_report(snapshot: Snapshot) -> dict[str, Any]:
    payload = snapshot.to_dict()
    report: dict[str, Any] = {
        "report_schema": "acoustic-workbench.report/1.0",
        "snapshot": payload,
        "source_of_truth": "measurement snapshot only; report performs no DSP recomputation",
    }
    report["report_hash"] = content_hash(report)
    return report


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    snapshot = report["snapshot"]
    delay = snapshot["delay"]
    mismatch = snapshot["mismatch"]
    verification = snapshot["verification"]
    calibration = snapshot["calibration"]
    lines: list[str] = []
    lines.append("# 声学测量与校准误差传播报告")
    lines.append("")
    lines.append(f"- 快照版本: `{snapshot['version_stamp']}`")
    lines.append(f"- 快照哈希: `{snapshot['content_hash']}`")
    lines.append(f"- 报告哈希: `{report['report_hash']}`")
    lines.append(f"- 音频: {snapshot['audio']['sample_rate']} Hz / {snapshot['audio']['bit_depth']} bit / {snapshot['audio']['channels']} ch / {snapshot['audio']['frames']} frames")
    lines.append("")
    lines.append("## 阶段哈希与复用")
    lines.append("")
    lines.append("| 阶段 | 内容哈希 | 复用来源 |")
    lines.append("|---|---|---|")
    for stage, record in snapshot["reuse"].items():
        lines.append(f"| {stage} | `{record['content_hash'][:16]}` | {record['compute_source']} |")
    lines.append("")
    lines.append("## 同步与延迟真值对照")
    lines.append("")
    lines.append("## 反卷积复算元数据")
    lines.append("")
    lines.append("| 通道 | FFT 长度 | 窗 | 截断点 | 原点偏移 | 窗起点 | 峰值采样 | 到达(采样) | 峰值(dB) |")
    lines.append("|---|---:|---|---:|---:|---|---:|---:|---:|")
    for item in snapshot["deconvolution"]["channels"]:
        lines.append(
            f"| {item['channel']} | {item['nfft']} | `{item['window']}` | {item['truncation_samples']} | "
            f"{item['origin_shift_samples']} | {item['window_starts']} | {item['peak_sample']} | "
            f"{_fmt(item['arrival_sample'])} | {_fmt(item['peak_level_db'])} |"
        )
    lines.append("")
    lines.append("| 通道 | 真值(采样) | 估计(采样) | 偏差 | 误差上界 | 状态 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for item in delay["channels"]:
        lines.append(
            f"| {item['channel']} | {_fmt(item['truth_relative_samples'])} | {_fmt(item['estimated_relative_samples'])} | "
            f"{_fmt(item['bias_samples'])} | {_fmt(item['error_bound_samples'])} | {item['status']} |"
        )
    lines.append("")
    lines.append("## 增益/频率响应失配与合成误差区间")
    lines.append("")
    lines.append("| 通道 | 加权估计(dB) | 真值(dB) | 偏差(dB) | 区间低(dB) | 区间高(dB) | 噪声项 | 泄漏项 | 限带项 | 真值相容 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for item in mismatch["channels"]:
        lines.append(
            f"| {item['channel']} | {_fmt(item['weighted_gain_db'])} | {_fmt(item['truth_gain_db'])} | "
            f"{_fmt(item['gain_bias_db'])} | {_fmt(item['interval_low_db'])} | {_fmt(item['interval_high_db'])} | "
            f"{_fmt(item['noise_bound_db'])} | {_fmt(item['leakage_bound_db'])} | {_fmt(item['band_limit_bound_db'])} | {item['truth_within_interval']} |"
        )
    lines.append("")
    lines.append("合成区间只由当前配置的噪声、截断/Tukey 泄漏和限带响应误差生成；分析频带收窄时 span 因子和频点误差同步缩小。")
    lines.append("")
    lines.append("## 校准 FIR 与残余失配")
    lines.append("")
    lines.append(f"- 滤波器长度: {calibration['filter_length']} taps，窗: `{calibration['window']}`")
    lines.append(f"- 重测状态: **{verification['status']}**")
    lines.append("")
    lines.append("| 通道 | 残余加权增益(dB) | 最大幅度残差(dB) | 最大相位残差(deg) | 超出区间(dB) | 超出区间(deg) | 相容 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for item in verification["channels"]:
        max_amp = max((abs(value) for value in item["residual_amplitude_db"]), default=0.0)
        max_phase = max((abs(value) for value in item["residual_phase_deg"]), default=0.0)
        lines.append(
            f"| {item['channel']} | {_fmt(item['residual_gain_db'])} | {_fmt(max_amp)} | {_fmt(max_phase)} | "
            f"{_fmt(item['max_amplitude_excess_db'])} | {_fmt(item['max_phase_excess_deg'])} | {item['compatible']} |"
        )
    lines.append("")
    if snapshot["errors"]:
        lines.append("## 业务错误")
        lines.append("")
        for error in snapshot["errors"]:
            lines.append(f"- `{error['code']}` ({error['stage']}): {error['message']}")
    else:
        lines.append("## 业务错误")
        lines.append("")
        lines.append("- 无。")
    lines.append("")
    lines.append("## 验收摘要")
    lines.append("")
    for key, value in snapshot["acceptance"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)
