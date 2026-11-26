"""Integer PCM reading via the standard library wave/struct modules."""

import struct
import wave


def read_pcm(path):
    """Return a list of signed 16-bit mono samples."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM is supported")
    count = len(raw) // 2
    samples = list(struct.unpack("<%dh" % count, raw))
    if channels == 1:
        return samples
    mono = []
    for i in range(0, len(samples), channels):
        mono.append(samples[i])
    return mono
