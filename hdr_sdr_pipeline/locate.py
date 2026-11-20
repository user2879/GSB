"""First-affected-pixel localization and minimal counterexamples.

Ordering is fixed: ascending frame number, then row-major (row, then col).
For every mismatch found by the detector we return the first pixel where the
clean recomputation and the committed output differ, with its observed value,
the declared error interval and the triggering injection fragment.

The minimal counterexample (MCE) is derived by delta debugging: starting from
the full injection config, items are removed in fixed order while the target
error code still reproduces. The result is minimal *by construction*:
removing any single remaining element makes the code no longer reproduce.
"""

import os

import numpy as np

from .detector import run_detection
from .injection import InjectionConfig
from .pipeline import InjectionContext, process_frame
from .quantize import quantization_step
from .config import STAGE_INDEX
from .synth import FRAME_HEIGHT, FRAME_WIDTH

# Declared per-stage error intervals (in normalized output units).
ERROR_INTERVALS = {
    "colorspace": 1e-6,
    "exposure": 1e-6,
    "tonemap": 1e-6,
    "transfer": 1e-6,
    "quantize": quantization_step(),
}


def _diff_pixel(clean, observed, frame_index, stage, item):
    diff = np.any(clean != observed, axis=2)
    if not np.any(diff):
        return None
    rows, cols = np.nonzero(diff)
    # np.nonzero already returns row-major order over a C-order array.
    row, col = int(rows[0]), int(cols[0])
    interval = ERROR_INTERVALS.get(stage, 1e-6)
    return {
        "frame": frame_index,
        "row": row,
        "col": col,
        "stage": stage,
        "error_code": "E_HASH_MISMATCH",
        "observed_value": [int(v) for v in observed[row, col]],
        "clean_value": [int(v) for v in clean[row, col]],
        "error_interval": interval,
        "injection_item": item.item_id if item is not None else None,
        "injection_fragment": _fragment(item, frame_index),
    }


def _fragment(item, frame_index):
    if item is None:
        return None
    return {
        "item_id": item.item_id,
        "kind": item.kind,
        "frame": item.frame if item.frame is not None else frame_index,
        "stage": item.stage,
        "payload": {k: v for k, v in item.payload.items()
                    if k != "replacement"},
    }


def locate_first_affected(work_dir, manifest, state, injection_config, *,
                          context=None):
    """Locate the first affected pixel across all hash-mismatch findings."""
    report = run_detection(
        work_dir, manifest, state, injection_config, context=context
    )
    width = manifest["generator"]["width"]
    height = manifest["generator"]["height"]
    frames_dir = os.path.join(work_dir, "frames")
    item_by_key = {}
    for item in (injection_config.items if injection_config else []):
        if item.kind == "stage_replacement":
            item_by_key[(item.frame, item.stage)] = item

    locations = []
    for finding in report.get("findings", []):
        if finding.get("scope") != "stage":
            continue
        frame_index = finding["frame"]
        stage = finding["stage"]
        clean_u8, _hashes, clean_stages = process_frame(
            frame_index, frames_dir, state, context=None,
            width=width, height=height,
        )
        # Committed (possibly injected) output bytes.
        from .persistence import read_ledger

        ledger = read_ledger(work_dir)
        rec = ledger["outputs"][str(frame_index)]
        observed = np.fromfile(
            os.path.join(work_dir, *rec["file"].split("/")),
            dtype=np.uint8,
        ).reshape(height, width, 3)
        item = item_by_key.get((frame_index, stage))
        if item is None:
            # downstream mismatch caused by an upstream replacement on the
            # same frame: attribute to the closest replaced upstream stage.
            upstream = [
                st for (fr, st) in item_by_key
                if fr == frame_index
                and STAGE_INDEX[st] <= STAGE_INDEX[stage]
            ]
            if upstream:
                nearest = max(upstream, key=lambda st: STAGE_INDEX[st])
                item = item_by_key[(frame_index, nearest)]
        location = _diff_pixel(
            clean_u8, observed, frame_index, stage, item,
        )
        if location is not None:
            locations.append(location)

    locations.sort(key=lambda loc: (loc["frame"], loc["row"], loc["col"]))
    return locations[0] if locations else None, report, locations


def minimal_counterexample(work_dir, manifest, state, injection_config, *,
                           context_builder, target_code="E_HASH_MISMATCH",
                           width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Reduce the injection config to the smallest set that reproduces code.

    ``context_builder(items)`` builds the InjectionContext for a candidate
    item list (keeps the runner free of storage assumptions).
    """

    def reproduces(items):
        if not items:
            return False
        candidate = InjectionConfig(
            config_id=injection_config.config_id + ".mce", items=list(items)
        )
        ctx = context_builder(items)
        report = run_detection(
            work_dir, manifest, state, candidate, context=ctx
        )
        codes = {f["code"] for f in report["findings"]}
        return target_code in codes

    items = list(injection_config.items)
    if not reproduces(items):
        raise AssertionError(
            "target error code does not reproduce with the full config"
        )

    # Fixed-order single-element elimination, repeated to a fixed point.
    changed = True
    while changed:
        changed = False
        for item in list(items):
            trial = [other for other in items if other.item_id != item.item_id]
            if reproduces(trial):
                items = trial
                changed = True
                break

    # Minimality proof: removing any remaining element must break reproduction.
    for item in items:
        trial = [other for other in items if other.item_id != item.item_id]
        if reproduces(trial):
            raise AssertionError(
                "counterexample is not minimal: %s is removable" % item.item_id
            )
    mce = InjectionConfig(
        config_id=injection_config.config_id + ".mce", items=items
    )
    return mce
