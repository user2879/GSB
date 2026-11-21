"""Stack folding and self/cumulative weight derivation.

Two aggregation engines implement the same math:

* :func:`fold_full`     - materializes every sample, then folds. Used when the
  unique-stack count and total weight stay below explicit thresholds.
* :func:`fold_streaming` - a single streaming pass that keeps only the folded
  stack table and frame maps; individual samples are never retained. Used once
  a threshold is reached.

Both emit an order-independent, canonical :class:`FoldResult`. The canonical
serialization (:meth:`FoldResult.canonical_bytes`) is guaranteed byte-identical
across engines and across arbitrary input ordering / weight distributions.

Weight semantics
----------------
A sample contributes ``weight`` to every distinct frame on its stack for
*cumulative* weight, and contributes ``weight`` only to its leaf (last) frame
for *self* weight. Identical stack paths are folded by summing weights.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .samples import Sample

STACK_SEP = ";"
DEFAULT_MAX_UNIQUE_STACKS = 5_000
DEFAULT_MAX_TOTAL_WEIGHT = 200_000


@dataclass(frozen=True)
class StackEntry:
    """One fully folded stack path."""

    frames: Tuple[str, ...]
    weight: int
    leaf: str


@dataclass(frozen=True)
class FrameMetric:
    frame: str
    self_weight: int
    cumulative_weight: int


@dataclass(frozen=True)
class FoldResult:
    """Canonical, order-independent folding output."""

    stacks: Tuple[StackEntry, ...]
    frame_metrics: Tuple[FrameMetric, ...]
    total_weight: int
    unique_stack_count: int
    path_used: str
    path_reason: str
    unique_stack_threshold: int
    total_weight_threshold: int

    def metric_map(self) -> Dict[str, FrameMetric]:
        return {m.frame: m for m in self.frame_metrics}

    def canonical_bytes(self) -> bytes:
        """Deterministic byte representation shared by both engines."""

        lines: List[str] = []
        lines.append(f"@total_weight={self.total_weight}")
        lines.append(f"@unique_stacks={self.unique_stack_count}")
        for entry in self.stacks:  # stacks are already canonical-sorted
            lines.append(
                f"S\t{STACK_SEP.join(entry.frames)}\t{entry.weight}\t{entry.leaf}"
            )
        for metric in self.frame_metrics:  # frame_metrics are name-sorted
            lines.append(
                f"F\t{metric.frame}\t{metric.self_weight}\t"
                f"{metric.cumulative_weight}"
            )
        return ("\n".join(lines) + "\n").encode("utf-8")


def select_path(unique_stack_count: int, total_weight: int,
                max_unique_stacks: int = DEFAULT_MAX_UNIQUE_STACKS,
                max_total_weight: int = DEFAULT_MAX_TOTAL_WEIGHT) -> str:
    """Return ``"full"`` below both thresholds, otherwise ``"streaming"``.

    The boundary is explicit: a value *below* the threshold selects the full
    path; reaching or exceeding either threshold selects the streaming path.
    """

    if (unique_stack_count >= max_unique_stacks
            or total_weight >= max_total_weight):
        return "streaming"
    return "full"


def _build_result(stack_weights: Mapping[Tuple[str, ...], int],
                  total_weight: int, path_used: str, path_reason: str,
                  unique_stack_threshold: int,
                  total_weight_threshold: int) -> FoldResult:
    stacks: List[StackEntry] = []
    self_weight: Dict[str, int] = {}
    cumulative_weight: Dict[str, int] = {}

    for frames, weight in stack_weights.items():
        leaf = frames[-1]
        stacks.append(StackEntry(frames=frames, weight=weight, leaf=leaf))
        self_weight[leaf] = self_weight.get(leaf, 0) + weight
        for frame in set(frames):  # count a frame once even if repeated
            cumulative_weight[frame] = \
                cumulative_weight.get(frame, 0) + weight

    stacks.sort(key=lambda entry: entry.frames)
    frame_names = sorted(set(self_weight) | set(cumulative_weight))
    metrics = tuple(
        FrameMetric(
            frame=name,
            self_weight=self_weight.get(name, 0),
            cumulative_weight=cumulative_weight.get(name, 0),
        )
        for name in frame_names
    )
    return FoldResult(
        stacks=tuple(stacks),
        frame_metrics=metrics,
        total_weight=total_weight,
        unique_stack_count=len(stack_weights),
        path_used=path_used,
        path_reason=path_reason,
        unique_stack_threshold=unique_stack_threshold,
        total_weight_threshold=total_weight_threshold,
    )


def fold_full(samples: Sequence[Sample],
              max_unique_stacks: int = DEFAULT_MAX_UNIQUE_STACKS,
              max_total_weight: int = DEFAULT_MAX_TOTAL_WEIGHT) -> FoldResult:
    """Full folding path: retain all samples, group stack paths, then derive."""

    stack_weights: Dict[Tuple[str, ...], int] = {}
    total_weight = 0
    for sample in samples:
        key = sample.frames
        stack_weights[key] = stack_weights.get(key, 0) + sample.weight
        total_weight += sample.weight

    path = select_path(
        len(stack_weights), total_weight,
        max_unique_stacks, max_total_weight)
    reason = (
        f"unique_stacks={len(stack_weights)} < "
        f"{max_unique_stacks} and total_weight={total_weight} < "
        f"{max_total_weight}: materialize-and-fold path"
    )
    if path != "full":
        reason = (
            f"threshold reached (unique_stacks={len(stack_weights)} vs "
            f"{max_unique_stacks}, total_weight={total_weight} vs "
            f"{max_total_weight}): full materialization not recommended"
        )
    return _build_result(
        stack_weights, total_weight, "full", reason,
        max_unique_stacks, max_total_weight)


def fold_streaming(samples: Iterable[Sample],
                   max_unique_stacks: int = DEFAULT_MAX_UNIQUE_STACKS,
                   max_total_weight: int = DEFAULT_MAX_TOTAL_WEIGHT) -> FoldResult:
    """Single-pass streaming path with constant per-sample retained state.

    Only the folded stack table plus self/cumulative frame maps are retained;
    no sample object is stored, so memory is bounded by cardinality rather than
    by the number of input rows.
    """

    stack_weights: Dict[Tuple[str, ...], int] = {}
    self_weight: Dict[str, int] = {}
    cumulative_weight: Dict[str, int] = {}
    total_weight = 0
    row_count = 0

    for sample in samples:
        row_count += 1
        frames = sample.frames
        weight = sample.weight
        stack_weights[frames] = stack_weights.get(frames, 0) + weight
        self_weight[frames[-1]] = self_weight.get(frames[-1], 0) + weight
        for frame in set(frames):
            cumulative_weight[frame] = \
                cumulative_weight.get(frame, 0) + weight
        total_weight += weight

    stacks = [
        StackEntry(frames=frames, weight=weight, leaf=frames[-1])
        for frames, weight in stack_weights.items()
    ]
    stacks.sort(key=lambda entry: entry.frames)
    frame_names = sorted(set(self_weight) | set(cumulative_weight))
    metrics = tuple(
        FrameMetric(
            frame=name,
            self_weight=self_weight.get(name, 0),
            cumulative_weight=cumulative_weight.get(name, 0),
        )
        for name in frame_names
    )
    reason = (
        f"single streaming pass over {row_count} rows; retained state bounded "
        f"by unique_stacks={len(stack_weights)} (thresholds "
        f"{max_unique_stacks} stacks / {max_total_weight} weight)"
    )
    return FoldResult(
        stacks=tuple(stacks),
        frame_metrics=metrics,
        total_weight=total_weight,
        unique_stack_count=len(stack_weights),
        path_used="streaming",
        path_reason=reason,
        unique_stack_threshold=max_unique_stacks,
        total_weight_threshold=max_total_weight,
    )


def fold_adaptive(samples: Sequence[Sample],
                  max_unique_stacks: int = DEFAULT_MAX_UNIQUE_STACKS,
                  max_total_weight: int = DEFAULT_MAX_TOTAL_WEIGHT,
                  force_path: str | None = None) -> FoldResult:
    """Pick the engine from explicit thresholds and run it.

    A cheap pre-scan counts unique stacks and total weight without retaining
    samples. ``force_path`` ("full" | "streaming") bypasses selection, which
    the tests use to prove byte-identical outputs across engines.
    """

    pre_scan_stacks: set[Tuple[str, ...]] = set()
    total_weight = 0
    for sample in samples:
        pre_scan_stacks.add(sample.frames)
        total_weight += sample.weight

    chosen = force_path or select_path(
        len(pre_scan_stacks), total_weight,
        max_unique_stacks, max_total_weight)

    if chosen == "full":
        return fold_full(
            samples, max_unique_stacks, max_total_weight)
    if chosen == "streaming":
        return fold_streaming(
            samples, max_unique_stacks, max_total_weight)
    raise ValueError(f"unknown force_path={force_path!r}")


def cross_check(samples: Sequence[Sample],
                max_unique_stacks: int = DEFAULT_MAX_UNIQUE_STACKS,
                max_total_weight: int = DEFAULT_MAX_TOTAL_WEIGHT
                ) -> Tuple[bool, str]:
    """Run both engines and compare canonical bytes."""

    full_result = fold_full(
        samples, max_unique_stacks, max_total_weight)
    streaming_result = fold_streaming(
        samples, max_unique_stacks, max_total_weight)
    same = (full_result.canonical_bytes()
            == streaming_result.canonical_bytes())
    return same, ("identical canonical bytes" if same
                  else "canonical byte mismatch between engines")
