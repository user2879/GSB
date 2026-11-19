"""Audio alignment evaluation via fixed cross-correlation.

For each shot segment the decoded mono mixdown is cross-correlated against
the reference waveform over a fixed lag window. Offset convention: a positive
offset means the decoded audio is *delayed* by that many samples relative to
the reference (dec[t] ~= gain * ref[t - offset]).
"""
from __future__ import annotations

import numpy as np

from . import util


def mixdown(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float64).mean(axis=0)


def align_segment(ref: np.ndarray, dec: np.ndarray, max_lag: int):
    """Return (offset_samples, gain_ratio, similarity) for equal-length segments."""
    n = len(ref)
    nfft = 1 << (2 * n - 1).bit_length()
    spec_d = np.fft.rfft(dec, nfft)
    spec_r = np.fft.rfft(ref, nfft)
    corr = np.fft.irfft(spec_d * np.conj(spec_r), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = corr[lags % nfft]
    offset = int(lags[int(np.argmax(vals))])

    if offset >= 0:
        d = dec[offset:]
        r = ref[: n - offset] if offset else ref
    else:
        d = dec[: n + offset]
        r = ref[-offset:]
    rr = float(np.dot(r, r))
    dd = float(np.dot(d, d))
    dr = float(np.dot(d, r))
    gain = dr / rr if rr > 0 else 0.0
    sim = dr / np.sqrt(rr * dd) if rr > 0 and dd > 0 else 0.0
    return offset, util.round6(gain), util.round6(sim)


def evaluate_audio(ref_pcm: np.ndarray, dec_pcm: np.ndarray, cfg, shots, max_lag: int) -> dict:
    """Per-segment alignment plus joint A/V timeline offset."""
    rate = cfg["audio"]["rate"]
    fps = cfg["video"]["fps"]
    ref = mixdown(ref_pcm)
    dec = mixdown(dec_pcm)
    segments = []
    for shot in shots:
        start = int(round(shot["start"] / fps * rate))
        end = int(round(shot["end"] / fps * rate))
        ref_seg = ref[start:end]
        dec_seg = dec[start:end]
        if len(dec_seg) < len(ref_seg):
            dec_seg = np.pad(dec_seg, (0, len(ref_seg) - len(dec_seg)))
        offset, gain, sim = align_segment(ref_seg, dec_seg[: len(ref_seg)], max_lag)
        segments.append(
            {
                "shot": shot["index"],
                "sample_start": start,
                "sample_end": end,
                "offset_samples": offset,
                "gain_ratio": gain,
                "similarity": sim,
            }
        )
    video_offset_s = 0.0  # video timeline anchored at pts 0 by construction
    for seg in segments:
        seg["audio_offset_seconds"] = util.round6(seg["offset_samples"] / rate)
        seg["joint_av_offset_seconds"] = util.round6(seg["audio_offset_seconds"] - video_offset_s)
    return {
        "rate": rate,
        "max_lag_samples": max_lag,
        "segments": segments,
        "sha256_decoded_pcm": util.hash_array(dec_pcm),
    }
