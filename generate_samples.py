#!/usr/bin/env python3
"""Fixed-seed synthetic sample generator.

Synthesizes 16 kHz mono 16-bit PCM WAV files with two to three speakers
alternating, partial overlaps, inter-segment silence and a stationary noise
floor. Speakers differ in pitch mean and formant positions. Everything is
derived from a fixed seed so repeated runs produce byte-identical files.
"""
import os
import wave

import numpy as np

SR = 16000
SEED = 20261123
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

SPEAKERS = {
    "A": {"f0": 120.0, "formants": (500.0, 1500.0, 2500.0), "amp": 0.60},
    "B": {"f0": 210.0, "formants": (650.0, 1800.0, 2900.0), "amp": 0.55},
    "C": {"f0": 90.0, "formants": (450.0, 1250.0, 2350.0), "amp": 0.65},
}


def synth_segment(rng, speaker, duration_s):
    """Harmonic stack shaped by a formant envelope, with mild vibrato."""
    spec = SPEAKERS[speaker]
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    f0_mean = spec["f0"] * rng.uniform(0.97, 1.03)
    f0 = f0_mean * (1.0 + 0.01 * np.sin(2.0 * np.pi * 4.0 * t))
    phase = 2.0 * np.pi * np.cumsum(f0) / SR
    x = np.zeros(n)
    for harmonic in range(1, 25):
        freq = harmonic * f0_mean
        amp = 0.0
        for formant in spec["formants"]:
            amp += np.exp(-0.5 * ((freq - formant) / 180.0) ** 2)
        amp /= harmonic
        if amp < 1e-4:
            continue
        x += amp * np.sin(harmonic * phase + rng.uniform(0.0, 2.0 * np.pi))
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak
    attack = int(0.02 * SR)
    release = int(0.03 * SR)
    env = np.ones(n)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env[-release:] = np.linspace(1.0, 0.0, release)
    return spec["amp"] * 0.6 * x * env


def build_track(rng, placements, total_s, noise_amp=0.004):
    n = int(total_s * SR)
    mix = rng.normal(0.0, noise_amp, n)
    for speaker, start_s, dur_s in placements:
        seg = synth_segment(rng, speaker, dur_s)
        offset = int(start_s * SR)
        mix[offset:offset + len(seg)] += seg
    return np.clip(mix, -0.98, 0.98)


def write_wav(path, samples):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def write_bad_wav(path):
    """Stereo 8 kHz file: violates the mono/16 kHz contract."""
    rng = np.random.RandomState(7)
    pcm = (rng.normal(0.0, 0.1, 8000 * 2) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm.tobytes())


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    tracks = {
        "two_speakers.wav": ([
            ("A", 0.2, 1.2),
            ("B", 1.8, 1.2),
            ("A", 3.4, 1.2),
            ("A", 5.0, 1.4),
            ("B", 5.8, 1.4),
            ("B", 7.4, 1.2),
        ], 9.0),
        "three_speakers.wav": ([
            ("A", 0.2, 1.0),
            ("B", 1.6, 1.0),
            ("C", 3.0, 1.0),
            ("A", 4.4, 1.6),
            ("B", 5.4, 1.2),
            ("C", 7.0, 1.2),
            ("A", 8.6, 1.2),
            ("C", 9.4, 1.0),
        ], 11.0),
        "overlap_pair.wav": ([
            ("A", 0.3, 2.0),
            ("B", 1.5, 2.0),
            ("A", 4.2, 1.5),
        ], 6.2),
        "quiet_single.wav": ([
            ("A", 0.5, 1.0),
            ("A", 3.5, 1.0),
            ("A", 5.5, 0.8),
        ], 7.0),
    }
    inj_base = ([
        ("A", 0.3, 1.2),
        ("B", 2.0, 1.2),
        ("A", 3.6, 1.2),
        ("B", 4.6, 1.0),
    ], 6.0)
    for name in ("inj_target_nonfinite.wav", "inj_target_clip.wav",
                 "inj_target_silence.wav", "inj_target_truncate.wav"):
        tracks[name] = inj_base
    for name in sorted(tracks):
        placements, total_s = tracks[name]
        samples = build_track(rng, placements, total_s)
        write_wav(os.path.join(SAMPLES_DIR, name), samples)
        print("wrote %s (%.1f s)" % (name, total_s))
    write_bad_wav(os.path.join(SAMPLES_DIR, "bad_format.wav"))
    print("wrote bad_format.wav (stereo 8 kHz, invalid)")


if __name__ == "__main__":
    main()
