"""Window slicing with a fixed short-tail rule.

Each period is sliced independently into ``k = min(requested, n_samples)``
windows.  The first ``k - 1`` windows all use the fixed base size
``floor(n / k)`` and the remainder is concentrated in the final window.
When that tail holds fewer than half of the target size it is merged into the
previous window (``merged_short``); a short tail kept standalone is tagged
``standalone_short``.  When fewer samples than requested windows exist, every
window is tagged ``standalone_short``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSlice:
    index: int
    start: int
    end: int
    tag: str  # regular | standalone_short | merged_short

    @property
    def size(self) -> int:
        return self.end - self.start


def slice_windows(n_samples: int, requested_windows: int) -> tuple[WindowSlice, ...]:
    if requested_windows < 1:
        raise ValueError("requested_windows must be >= 1")
    if n_samples <= 0:
        return ()
    k = min(requested_windows, n_samples)
    base, remainder = divmod(n_samples, k)
    if k < requested_windows:
        boundaries = [idx for idx in range(k + 1)]
        return tuple(
            WindowSlice(idx, boundaries[idx], boundaries[idx + 1],
                        "standalone_short")
            for idx in range(k)
        )
    boundaries = [0]
    for idx in range(k - 1):
        boundaries.append(boundaries[-1] + base)
    boundaries.append(n_samples)
    slices = [
        WindowSlice(idx, boundaries[idx], boundaries[idx + 1], "regular")
        for idx in range(k)
    ]
    tail_size = slices[-1].size
    # Short tail: the remainder adds fewer than half a base window.
    if base > 1 and 0 < remainder < base / 2.0 and len(slices) >= 2:
        last = slices[-1]
        prev = slices[-2]
        slices[-2] = WindowSlice(prev.index, prev.start, last.end, "merged_short")
        slices = slices[:-1]
        return tuple(
            WindowSlice(idx, item.start, item.end, item.tag)
            for idx, item in enumerate(slices)
        )
    if remainder != 0:
        slices[-1] = WindowSlice(slices[-1].index, slices[-1].start,
                                 slices[-1].end, "standalone_short")
    return tuple(slices)


def aligned_window_plan(n_ref: int, n_cur: int, requested_windows: int) -> tuple[
    tuple[WindowSlice, ...], tuple[WindowSlice, ...]
]:
    return slice_windows(n_ref, requested_windows), slice_windows(n_cur, requested_windows)
