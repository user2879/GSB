"""Generate deterministic 16 kHz mono 16-bit PCM samples and exchange files.

Run with ``python generate_samples.py``.  Everything is derived from fixed
seeds and pure-stdlib + numpy math, so repeated runs produce byte-identical
WAV/JSON/text output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voice_conversion.audio import write_wav
from voice_conversion.config import PipelineConfig, canonical_json
from voice_conversion.model import Segment
from voice_conversion.v1 import write_v1
from voice_conversion.v2 import write_v2

SAMPLE_RATE = 16000
SEED = 20261124


class Lcg:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def normal(self, size: int) -> np.ndarray:
        # Box-Muller on a fixed-seed LCG: independent of numpy global RNG.
        out = np.empty(size, dtype=np.float64)
        i = 0
        while i < size:
            self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
            u1 = (self.state >> 8) / 16777216.0 + 1e-12
            self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
            u2 = (self.state >> 8) / 16777216.0
            radius = np.sqrt(-2.0 * np.log(u1))
            angle = 2.0 * np.pi * u2
            out[i] = radius * np.cos(angle)
            i += 1
            if i < size:
                out[i] = radius * np.sin(angle)
                i += 1
        return out


def harmonic_signal(duration_s: float, f0_profile, formants, noise_floor: float,
                    seed: int) -> np.ndarray:
    count = int(duration_s * SAMPLE_RATE)
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * np.pi * np.cumsum(f0_profile(t)) / SAMPLE_RATE
    signal = np.zeros(count, dtype=np.float64)
    for harmonic in range(1, 13):
        # Formant structure: each harmonic is weighted by the nearest formant.
        weight = np.zeros_like(t)
        for center, gain in formants:
            weight += gain * np.exp(
                -0.5 * ((harmonic - center) / 1.4) ** 2)
        signal += weight * np.sin(phase * harmonic)
    rng = Lcg(seed)
    signal += noise_floor * rng.normal(count)
    peak = np.max(np.abs(signal)) or 1.0
    return np.clip(np.rint(signal / peak * 26000.0), -32768, 32767)


def build_config() -> PipelineConfig:
    return PipelineConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        mel_bands=20,
        mel_fmax=7600.0,
        cepstrum_order=12,
        f0_min_hz=70.0,
        f0_max_hz=400.0,
        max_frames_per_segment=128,
        frac_bits=12,
        quant_levels=4,
        workers=4,
        snapshot_commit_interval=8,
        crossfade_ms=20.0,
        timbre_table={
            "alto": {
                "code": 101,
                "f0_shift_ratio": 0.08,
                "formant_shift_ratio": 0.05,
                "cep_bend": 0.004,
                "gain_delta": 0.03,
            },
            "tenor": {
                "code": 202,
                "f0_shift_ratio": -0.12,
                "formant_shift_ratio": -0.07,
                "cep_bend": -0.005,
                "gain_delta": -0.02,
            },
            "soprano": {
                "code": 303,
                "f0_shift_ratio": 0.22,
                "formant_shift_ratio": 0.12,
                "cep_bend": 0.008,
                "gain_delta": 0.05,
            },
        },
    )


def main(out_dir: str = "samples") -> Path:
    root = Path(out_dir)
    wav_dir = root / "wav"
    exchange_dir = root / "exchange"
    config_dir = root / "config"
    for directory in (wav_dir, exchange_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = harmonic_signal(
        duration_s=4.0,
        f0_profile=lambda tt: 130.0 + 18.0 * np.sin(2.0 * np.pi * 1.5 * tt),
        formants=((2.0, 1.0), (5.0, 0.6), (9.0, 0.35)),
        noise_floor=0.015,
        seed=SEED,
    )
    write_wav(wav_dir / "source.wav", source, SAMPLE_RATE)

    references = {
        "alto": (lambda tt: 190.0 + 10.0 * np.sin(2 * np.pi * 1.2 * tt),
                 ((2.4, 1.0), (5.6, 0.6), (9.8, 0.3)), SEED + 1),
        "tenor": (lambda tt: 110.0 + 8.0 * np.sin(2 * np.pi * 1.0 * tt),
                  ((1.7, 1.0), (4.2, 0.55), (7.6, 0.28)), SEED + 2),
        "soprano": (lambda tt: 260.0 + 14.0 * np.sin(2 * np.pi * 1.8 * tt),
                    ((3.0, 1.0), (6.6, 0.62), (11.0, 0.34)), SEED + 3),
    }
    for name, (profile, formants, seed) in references.items():
        ref = harmonic_signal(2.0, profile, formants, 0.012, seed)
        write_wav(wav_dir / f"ref_{name}.wav", ref, SAMPLE_RATE)

    config = build_config()
    config_path = config_dir / "acoustic.json"
    config_path.write_text(
        canonical_json(config.to_dict()) + "\n", encoding="utf-8")

    # Five canonical segments.  v2 uses half-frame float boundaries so the
    # downgrade sidecar has non-zero residuals to preserve.
    boundaries = [(0, 60), (60, 130), (130, 200), (200, 270), (270, 340)]
    timbres = ["alto", "tenor", "soprano", "alto", "tenor"]
    permilles = [825, 650, 910, 475, 730]
    v1_segments, v2_segments = [], []
    for idx, ((start, end), timbre, permille) in enumerate(
            zip(boundaries, timbres, permilles)):
        v1_segments.append(Segment(
            index=idx, start_frame=float(start), end_frame=float(end),
            intensity=permille / 1000.0, timbre=timbre,
            source_version="v1", intensity_permille=permille,
            precision_tag="exact-permille",
            timbre_code=config.name_to_code(timbre),
        ))
        v2_segments.append(Segment(
            index=idx, start_frame=start + 0.37, end_frame=end - 0.21,
            intensity=(permille + 3) / 1000.0, timbre=timbre,
            source_version="v2", precision_tag="float-native",
            start_residual=0.37, end_residual=-0.21,
            timbre_code=config.name_to_code(timbre),
            passthrough={
                "engine": "nova-7",
                "confidence": 0.91 + idx * 0.01,
                "notes": f"segment {idx} passthrough",
            },
        ))
    write_v1(exchange_dir / "segments.v1.txt", v1_segments, config)
    write_v2(exchange_dir / "segments.v2.json", v2_segments)

    # A deliberately malformed v1 line for parser failure tests.
    good = (exchange_dir / "segments.v1.txt").read_text(encoding="utf-8")
    truncated_line = "     5     270     340   730   202"  # width 30 < 36
    (exchange_dir / "segments_broken.v1.txt").write_text(
        good + truncated_line + "\n", encoding="utf-8")
    print(f"samples written under {root}/")
    return root


if __name__ == "__main__":
    main()
