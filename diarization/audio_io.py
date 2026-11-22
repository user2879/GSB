"""WAV loading, container validation and overlapped windowing."""
import wave

import numpy as np

SAMPLE_RATE = 16000


class AudioError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def load_wav(path):
    """Load a WAV file and validate the RIFF container contract.

    Raises AudioError("E_AUDIO_FORMAT") on any contract violation.
    Returns (float64 samples in [-1, 1), sample rate).
    """
    try:
        wf = wave.open(path, "rb")
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioError("E_AUDIO_FORMAT", "invalid RIFF container: %s" % exc)
    with wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if rate != SAMPLE_RATE:
        raise AudioError("E_AUDIO_FORMAT", "sample rate %d != %d" % (rate, SAMPLE_RATE))
    if channels != 1:
        raise AudioError("E_AUDIO_FORMAT", "channel count %d != 1" % channels)
    if width != 2:
        raise AudioError("E_AUDIO_FORMAT", "sample width %d != 2 bytes" % width)
    if len(raw) != nframes * 2:
        raise AudioError("E_AUDIO_FORMAT",
                         "data chunk length %d does not match %d samples"
                         % (len(raw), nframes))
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return samples, rate


def make_windows(n_samples, sr, config):
    """Slice the signal into overlapped analysis windows.

    Window indices start at zero and strictly increase; each window records
    its start/end sample offsets. The final short window is padded or
    cropped per config["last_window_policy"] and always flagged is_last.
    """
    win = sr * int(config["window_ms"]) // 1000
    hop = sr * int(config["hop_ms"]) // 1000
    policy = config.get("last_window_policy", "pad")
    windows = []
    if n_samples <= 0 or win <= 0 or hop <= 0:
        return windows
    index = 0
    start = 0
    while start < n_samples:
        end = start + win
        is_last = end >= n_samples
        if is_last and policy == "crop":
            end = n_samples
        windows.append({
            "window_index": index,
            "start_sample": start,
            "end_sample": end,
            "is_last": is_last,
        })
        if is_last:
            break
        start += hop
        index += 1
    return windows


def extract_window(samples, window):
    """Cut one window out of the signal, zero-padding past the end."""
    start = window["start_sample"]
    end = window["end_sample"]
    seg = samples[start:min(end, len(samples))]
    if len(seg) < end - start:
        seg = np.concatenate([seg, np.zeros(end - start - len(seg))])
    return seg
