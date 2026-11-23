from __future__ import annotations

from typing import Iterable


def numpy_watermark_ratios(
    active: Iterable[int],
    shares: Iterable[int],
    scale: int,
) -> list[int]:
    """NumPy fixed-point floor helper used to independently verify event ratios."""

    import numpy as np

    active_array = np.asarray(list(active), dtype=np.int64)
    share_array = np.asarray(list(shares), dtype=np.int64)
    if active_array.shape != share_array.shape:
        raise ValueError("active and shares must have equal length")
    ratios = np.zeros_like(active_array, dtype=np.int64)
    mask = share_array > 0
    ratios[mask] = np.floor_divide(active_array[mask] * np.int64(scale), share_array[mask])
    return [int(value) for value in ratios.tolist()]


def numpy_request_delay_ms(seed: int, request_index: int, base_interval_ms: int) -> int:
    """Deterministic non-negative jitter for a source's request stream."""

    import numpy as np

    if request_index < 0:
        raise ValueError("request_index must be non-negative")
    jitter_span = max(1, base_interval_ms // 4)
    generator = np.random.Generator(np.random.PCG64(seed & 0xFFFFFFFFFFFFFFFF))
    jitter = int(generator.integers(0, jitter_span + 1, size=request_index + 1)[-1])
    return base_interval_ms + jitter
