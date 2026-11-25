"""确定性样本生成：固定配置 -> WAV 片段、片段清单与命令流示例。

全局信号先整体合成再按片段切片；语音块刻意跨越片段边界，
使端点合并与跨片段分析窗引用在样本中真实发生。
"""

import json
from pathlib import Path

import numpy as np

from .audio import write_wav
from .config import PipelineConfig

FADE_SAMPLES = 200
SILENCE_NOISE_SCALE = 0.05


SPEECH_START_IN_FRAGMENT = 5200


def _speech_mask(total: int, frag_len: int, speech_len: int) -> np.ndarray:
    """每片段后段起音并延伸进入下一前段，语音区确定性跨越片段边界。"""
    mask = np.zeros(total, dtype=bool)
    for start in range(SPEECH_START_IN_FRAGMENT, total, frag_len):
        mask[start:min(total, start + speech_len)] = True
    return mask


def _synthesize_global(cfg: PipelineConfig) -> np.ndarray:
    n_frag = cfg.sample.fragment_count
    frag_len = cfg.sample.samples_per_fragment
    total = n_frag * frag_len
    speech_len = int(round(frag_len * cfg.sample.speech_ratio))
    rng = np.random.default_rng(cfg.sample.noise_seed)
    noise = rng.normal(0.0, cfg.sample.noise_std, total)
    signal = noise * SILENCE_NOISE_SCALE
    time_axis = np.arange(total, dtype=np.float64) / cfg.sample.sample_rate
    mask = _speech_mask(total, frag_len, speech_len)
    vocab = cfg.vocab
    codes = sorted(vocab)
    boundaries = np.flatnonzero(np.diff(mask.view(np.int8), prepend=0, append=0))
    for region_id in range(0, len(boundaries) - 1):
        start, end = int(boundaries[region_id]), int(boundaries[region_id + 1])
        if not mask[start]:
            continue
        code = codes[(start // frag_len) % len(codes)]
        freq = cfg.sample.tone_base_hz + code * cfg.sample.tone_step_hz
        tone = cfg.sample.tone_amplitude * np.sin(2.0 * np.pi * freq * time_axis[start:end])
        fade = np.ones(end - start, dtype=np.float64)
        ramp = min(FADE_SAMPLES, (end - start) // 2)
        fade[:ramp] = np.linspace(0.0, 1.0, ramp)
        fade[-ramp:] = np.linspace(1.0, 0.0, ramp)
        signal[start:end] += tone * fade + noise[start:end]
    return np.rint(signal).clip(-32768, 32767).astype(np.int16)


def generate_dataset(cfg: PipelineConfig, base_dir: str | Path) -> dict:
    base = Path(base_dir)
    wav_dir = base / cfg.paths.data_dir
    wav_dir.mkdir(parents=True, exist_ok=True)
    frag_len = cfg.sample.samples_per_fragment
    global_signal = _synthesize_global(cfg)
    vocab_words = set(cfg.vocab.values())
    words_sorted = sorted(vocab_words)
    entries = []
    for seq in range(cfg.sample.fragment_count):
        chunk = global_signal[seq * frag_len:(seq + 1) * frag_len]
        rel = f"fragment_{seq:04d}.wav"
        write_wav(wav_dir / rel, chunk)
        entries.append(
            {
                "seq": seq,
                "wav_file": rel,
                "capture_ms": seq * frag_len * 1000 // cfg.sample.sample_rate,
                "command_word": words_sorted[seq % len(words_sorted)],
            }
        )
    order_rng = np.random.default_rng(cfg.sample.noise_seed + 7)
    arrival_order = order_rng.permutation(len(entries)).tolist()
    for rank, entry_index in enumerate(arrival_order):
        jitter = int(order_rng.integers(0, 20))
        entries[entry_index]["arrival_ms"] = rank * 100 + jitter
    manifest = {"fragments": entries}
    manifest_path = base / cfg.paths.manifest_file
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    stream = [
        {
            "fragment_seq": entry["seq"],
            "word": entry["command_word"],
            "issued_ms": entry["capture_ms"],
        }
        for entry in entries
    ]
    stream_path = base / cfg.paths.command_stream_file
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.write_text(
        json.dumps({"commands": stream}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "manifest": str(manifest_path),
        "command_stream": str(stream_path),
        "wav_dir": str(wav_dir),
    }
