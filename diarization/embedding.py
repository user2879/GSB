"""Per-window voice embedding extraction and fixed-point quantization.

All feature math is float64. Frame aggregation and normalization use fixed
accumulation orders so identical inputs produce bit-identical embeddings.
"""
import numpy as np

_PROJ_SEED = 0xC0FFEE


def _projection_matrix(feat_dim, embed_dim):
    rng = np.random.RandomState(_PROJ_SEED)
    return rng.normal(0.0, 1.0 / float(feat_dim) ** 0.5, size=(feat_dim, embed_dim))


def _dct_matrix(n_in, n_out):
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    return np.cos(np.pi * (n + 0.5) * k / float(n_in))


def _mel_filterbank(n_filters, n_fft, sr):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(40.0), hz2mel(sr / 2.0), n_filters + 2)
    freqs = mel2hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        b = max(b, a + 1)
        c = max(c, b + 1)
        for k in range(a, b):
            fb[i, k] = (k - a) / float(b - a)
        for k in range(b, c):
            fb[i, k] = (c - k) / float(c - b)
    return fb


_MEL_FILTERS = 26


def frame_slice_params(sr, config):
    flen = sr * int(config["frame_ms"]) // 1000
    fhop = sr * int(config["frame_hop_ms"]) // 1000
    return flen, fhop


def frame_energies(samples, sr, config):
    flen, fhop = frame_slice_params(sr, config)
    n_frames = 0 if len(samples) < flen else 1 + (len(samples) - flen) // fhop
    out = np.zeros(n_frames)
    for i in range(n_frames):
        frame = samples[i * fhop:i * fhop + flen]
        out[i] = float(np.dot(frame, frame)) / flen
    return out


def estimate_noise_floor(energies, max_frames):
    """Mean energy of the quietest positive-energy frames.

    Exactly-silent frames (e.g. injected zero regions) are excluded:
    digital silence is not a noise floor. Returns 0.0 when no positive
    frame exists.
    """
    positive = np.sort(energies[energies > 1e-12])
    k = min(int(max_frames), len(positive))
    return float(np.mean(positive[:k])) if k > 0 else 0.0


def frame_features(win, sr, config, noise_floor=None):
    """Per-frame log energy, pitch estimate and cepstral coefficients.

    The noise floor (mean energy of the quietest frames) is estimated
    per window when not supplied; the pipeline passes a global estimate
    computed over the whole audio so the voicing gate is comparable
    across windows.
    """
    flen, fhop = frame_slice_params(sr, config)
    n_frames = 0 if len(win) < flen else 1 + (len(win) - flen) // fhop
    cep_count = int(config["cepstral_coeffs"])
    frames = [win[i * fhop:i * fhop + flen] for i in range(n_frames)]
    energies = np.array([float(np.dot(f, f)) / flen for f in frames])
    if noise_floor is None:
        noise_floor = estimate_noise_floor(energies, config["noise_floor_frames"])
    mel = _mel_filterbank(_MEL_FILTERS, flen, sr)
    dct = _dct_matrix(_MEL_FILTERS, cep_count + 1)
    min_lag = max(1, sr // 400)
    max_lag = sr // 60
    feats = []
    for i in range(n_frames):
        frame = frames[i]
        energy = float(energies[i])
        log_e = float(np.log(energy + 1e-12))
        pitch = 0.0
        if energy > 4.0 * noise_floor + 1e-9:
            ac = np.correlate(frame, frame, "full")[flen - 1:]
            if ac[0] > 0.0:
                region = ac[min_lag:max_lag + 1] / ac[0]
                lag = int(np.argmax(region)) + min_lag
                pitch = sr / float(lag)
        spec = np.abs(np.fft.rfft(frame))
        cep = dct @ np.log(mel @ spec + 1e-12)
        # drop c0 (pure energy) and scale the rest to a comparable range
        feats.append(np.concatenate(([pitch / 400.0], cep[1:] / 10.0)))
    return feats, energies, noise_floor


def window_embedding(win, sr, config, noise_floor=None):
    """Aggregate frame features into one L2-normalized window embedding.

    Voiced frames (pitch > 0 under the noise gate) are averaged in
    ascending frame order; if no frame is voiced all frames are used.
    The averaged feature is mapped through a fixed projection to the
    configured embedding dimension.
    """
    cep_count = int(config["cepstral_coeffs"])
    embed_dim = int(config["embed_dim"])
    proj = _projection_matrix(1 + cep_count, embed_dim)
    weights = np.concatenate(([2.0], np.ones(cep_count)))
    feats, energies, noise_floor = frame_features(win, sr, config, noise_floor)
    voiced = [f for f in feats if f[0] > 0.0]
    use = voiced if voiced else feats
    acc = np.zeros(1 + cep_count)
    for feat in use:  # fixed accumulation order: ascending frame index
        acc = acc + feat
    acc = acc / len(use)
    emb = (acc * weights) @ proj
    norm = 0.0
    for value in emb:  # fixed accumulation order: ascending dimension
        norm += float(value) * float(value)
    norm = norm ** 0.5
    if norm > 0.0:
        emb = emb / norm
    return emb, energies, noise_floor


def quantize(emb, frac_bits, levels):
    """Convert a float64 embedding to fixed-point integers.

    Returns the quantized integers, the dequantized floats, the per-dim
    residual and the uniform quantization step (2 ** -frac_bits).
    """
    step = 2.0 ** (-int(frac_bits))
    q = np.round(emb / step)
    q = np.clip(q, -(levels // 2), levels // 2 - 1).astype(np.int64)
    deq = (q * step).astype(np.float64)
    residual = emb - deq
    return {"quantized": q, "dequantized": deq, "residual": residual, "step": step}
