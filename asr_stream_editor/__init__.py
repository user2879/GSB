"""Deterministic streaming ASR data editor.

Pure-Python + NumPy implementation that decodes deterministic synthetic PCM
streams in chunks, supports interactive edits under a latency budget, and
guarantees incremental results equal to a one-shot full decode.
"""

__all__ = ["errors", "config", "audio", "features", "decoder", "streaming", "stability", "snapshot", "engine", "full_decode", "cli"]
