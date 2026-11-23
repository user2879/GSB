"""固定种子合成本地样本目录。

生成内容：
- 16kHz 单声道 16bit PCM WAV（可控基频均值/方差、能量动态范围、语速、平稳底噪）
- 对话脚本 JSON（轮次、说话人、衔接关系、情感标注）
- 声学证据摘要 JSONL（由组件自身的帧特征提取产出，保证与流水线同源）
- 校准配置 JSON 与独立注入清单 JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voicecalib.audio import SAMPLE_RATE, write_wav  # noqa: E402
from voicecalib.features import FEATURE_NAMES, extract_features  # noqa: E402

SPEAKER_PLAN = [
    ("S1", 120.0), ("S2", 210.0), ("S1", 120.0), ("S2", 210.0),
    ("S1", 120.0), ("S2", 210.0), ("S2", 210.0), ("S1", 120.0),
    ("S1", 120.0), ("S2", 210.0), ("S1", 120.0), ("S2", 210.0),
]
EMOTION_PLAN = [
    "calm", "calm", "happy", "happy", "angry", "angry",
    "sad", "sad", "happy", "angry", "calm", "calm",
]
# 情感 -> (基频倍率, 能量倍率, 节奏倍率)，与证据原型方向一致。
EMOTION_KNOBS = {
    "calm": (0.95, 0.45, 0.55),
    "happy": (1.15, 0.70, 0.95),
    "angry": (1.30, 0.95, 1.15),
    "sad": (0.85, 0.35, 0.40),
}


def synthesize_round(rng: np.random.Generator, speaker_f0: float,
                     emotion: str, duration_s: float) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f0_mult, energy_mult, rhythm_mult = EMOTION_KNOBS[emotion]
    base_f0 = speaker_f0 * f0_mult
    # 基频方差通过缓变随机游走体现，固定 rng 保证可复现。
    jitter = rng.normal(0.0, base_f0 * 0.02, size=(n + 799) // 800)
    jitter_up = np.repeat(jitter, 800)[:n]
    phase = 2.0 * np.pi * np.cumsum(base_f0 + jitter_up) / SAMPLE_RATE
    signal = np.sin(phase) + 0.35 * np.sin(2.0 * phase)

    # 能量动态范围：包络调制 + 语速控制的静音间隙。
    envelope = 0.6 + 0.4 * np.sin(2.0 * np.pi * (1.8 * rhythm_mult) * t)
    gap_rate = max(0.5, 2.2 * rhythm_mult)  # 每秒间隙次数
    gap_centers = np.arange(0.5, duration_s, 1.0 / gap_rate)
    for center in gap_centers:
        center_sample = int(center * SAMPLE_RATE)
        width = int(0.035 * SAMPLE_RATE)
        lo = max(0, center_sample - width)
        hi = min(n, center_sample + width)
        envelope[lo:hi] *= 0.08
    signal *= envelope * energy_mult

    # 平稳底噪：固定种子高斯白噪声，噪声水平恒定。
    noise = rng.normal(0.0, 0.01, size=n)
    signal = signal + noise
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak * 0.9
    return signal.astype(np.float64)


def build_config() -> dict:
    return {
        "initial_weights": [0.45, 0.35, 0.20],
        "normalization": "softmax",
        "calibration_gain": 0.8,
        "decay": 0.35,
        "fixed_point": {"frac_bits": 12, "quant_gear": 256},
        "adjacent_tolerance": 25.0,
        "batch_segment_budget": 4,
        "max_workers": 4,
        "snapshot_interval_rounds": 4,
        "speaker_anchors_hz": {"S1": 120.0, "S2": 210.0},
        "transition_weight": 0.6,
        "same_speaker_weight": 0.9,
        "min_segment_rounds": 2,
        "max_gap_rounds": 1,
        "frame_length_ms": 25.0,
        "frame_shift_ms": 10.0,
        "injections": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples", help="样本输出目录")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--rounds", type=int, default=len(SPEAKER_PLAN))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_rounds = min(args.rounds, len(SPEAKER_PLAN))

    script_rounds = []
    evidence_lines = []
    prev_speaker = None
    for rnd in range(n_rounds):
        speaker, speaker_f0 = SPEAKER_PLAN[rnd]
        emotion = EMOTION_PLAN[rnd]
        duration = 1.0 + 0.12 * (rnd % 3)
        samples = synthesize_round(rng, speaker_f0, emotion, duration)
        file_name = f"round_{rnd:03d}.wav"
        write_wav(out / file_name, samples)

        if rnd == 0:
            link = "init"
        elif speaker == prev_speaker:
            link = "same_speaker"
        else:
            link = "speaker_switch"
        script_rounds.append({
            "round": rnd,
            "audio_file": file_name,
            "speaker": speaker,
            "link": link,
            "emotion": emotion,
        })

        features = extract_features(
            rnd, file_name, samples, 25.0, 10.0,
            sort_key=f"evidence|{file_name}")
        evidence_lines.append({
            "round": rnd,
            "file": file_name,
            "n_frames": features.n_frames,
            "voiced_ratio": features.voiced_ratio,
            "features": {
                FEATURE_NAMES[0]: features.vector[0],
                FEATURE_NAMES[1]: features.vector[1],
                FEATURE_NAMES[2]: features.vector[2],
            },
        })
        prev_speaker = speaker

    (out / "dialogue_script.json").write_text(
        json.dumps({"session_id": "synth-session-001",
                    "rounds": script_rounds}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    with (out / "acoustic_evidence.jsonl").open("w", encoding="utf-8") as fh:
        for entry in evidence_lines:
            fh.write(json.dumps(entry, sort_keys=True,
                                ensure_ascii=False) + "\n")
    (out / "calibration_config.json").write_text(
        json.dumps(build_config(), indent=2, ensure_ascii=False),
        encoding="utf-8")

    injections = [
        {"injection_id": "INJ-NONFINITE-01", "round": 3,
         "stage": "evidence", "type": "nonfinite_sample",
         "payload": {"value": "nan"}},
        {"injection_id": "INJ-DIM-01", "round": 5,
         "stage": "evidence", "type": "evidence_dim_mismatch",
         "payload": {"expected_dim": 2, "force_fail": True}},
        {"injection_id": "INJ-ZEROGAIN-01", "round": 7,
         "stage": "calibration", "type": "zero_gain", "payload": {}},
        {"injection_id": "INJ-REPLACE-01", "round": 9,
         "stage": "calibration", "type": "replace_round_out_of_range",
         "payload": {"target_round": 999}},
    ]
    (out / "injections.json").write_text(
        json.dumps({"injections": injections}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"generated {n_rounds} rounds into {out}")


if __name__ == "__main__":
    main()
