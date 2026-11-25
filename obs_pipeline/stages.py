"""Integer fixed-point stage transforms (requirement 4).

All arithmetic is integer-only. Each stage kind declares its truncation /
rounding / saturation / negative-value policy explicitly in SEMANTICS. Null
handling rules (requirement 3) are declared per stage in NULL_RULES and the
null count plus rule identifier are emitted with every stage result.

Determinism: a stage window output depends only on the input values inside
that window, so any parallelism degree or batch execution order yields
byte-identical intermediate summaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1

SEMANTICS = {
    "std_diff": {
        "division": "none (pure integer subtraction)",
        "saturation": "saturate-to-int64",
        "negative": "preserved (signed result kept as-is)",
    },
    "window_avg": {
        "division": "floor (toward negative infinity, Python //)",
        "saturation": "saturate-to-int64 on the running sum",
        "negative": "floor rounding applies to negative sums identically",
    },
    "quantile_bucket": {
        "division": "floor on (x-lo)*buckets/(hi-lo)",
        "saturation": "clamp bucket index to [0, buckets-1]",
        "negative": "values below lo clamp to bucket 0",
    },
    "threshold_norm": {
        "division": "floor on (x-threshold)*scale/span",
        "saturation": "clamp output to [0, scale]",
        "negative": "values below threshold saturate to 0",
    },
}

NULL_RULES = {
    "std_diff": "propagate: output null when the record or its predecessor is null; "
    "first record of a window is null (no predecessor)",
    "window_avg": "skip: nulls are excluded from sum and count; output null only when "
    "the window contains no non-null value",
    "quantile_bucket": "propagate: null input maps to null output and is excluded from "
    "bucket counts",
    "threshold_norm": "propagate: null input maps to null output",
}


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    window_id: str
    outputs: tuple  # int or None, aligned with the input slice
    null_count: int
    null_rule: str
    summary: str  # sha256 hex of canonical output encoding
    semantics: dict


def _sat(value):
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return value


def summarize_outputs(outputs):
    canonical = ",".join("-" if v is None else str(v) for v in outputs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_stage(stage, window, inputs):
    """Execute one stage over one window.

    inputs: tuple of int|None fixed-point values for the window slice.
    Returns a StageResult. Pure function of (stage, window, inputs).
    """
    kind = stage.kind
    if kind == "std_diff":
        outputs = _std_diff(inputs)
    elif kind == "window_avg":
        outputs = _window_avg(inputs, stage.params["window"])
    elif kind == "quantile_bucket":
        outputs = _quantile_bucket(inputs, stage.params["buckets"], stage.params["lo"], stage.params["hi"])
    elif kind == "threshold_norm":
        outputs = _threshold_norm(inputs, stage.params["threshold"], stage.params["span"], stage.scale)
    else:  # pragma: no cover - validated at spec load
        raise ValueError("unknown stage kind %r" % kind)
    outputs = tuple(outputs)
    return StageResult(
        stage_id=stage.id,
        window_id=window.id,
        outputs=outputs,
        null_count=sum(1 for v in outputs if v is None),
        null_rule=NULL_RULES[kind],
        summary=summarize_outputs(outputs),
        semantics=SEMANTICS[kind],
    )


def _std_diff(inputs):
    out = [None]
    for prev, cur in zip(inputs, inputs[1:]):
        if prev is None or cur is None:
            out.append(None)
        else:
            out.append(_sat(cur - prev))
    return out


def _window_avg(inputs, width):
    out = []
    running = 0
    count = 0
    queue = []
    for value in inputs:
        queue.append(value)
        if value is not None:
            running = _sat(running + value)
            count += 1
        if len(queue) > width:
            old = queue.pop(0)
            if old is not None:
                running = _sat(running - old)
                count -= 1
        if count == 0:
            out.append(None)
        else:
            out.append(running // count)  # floor division, declared
    return out


def _quantile_bucket(inputs, buckets, lo, hi):
    span = hi - lo
    out = []
    for value in inputs:
        if value is None:
            out.append(None)
            continue
        index = (value - lo) * buckets // span  # floor division, declared
        out.append(min(max(index, 0), buckets - 1))
    return out


def _threshold_norm(inputs, threshold, span, scale):
    out = []
    for value in inputs:
        if value is None:
            out.append(None)
            continue
        normalized = (value - threshold) * scale // span  # floor division, declared
        out.append(min(max(normalized, 0), scale))
    return out
