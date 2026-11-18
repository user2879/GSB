"""Diagnostic report assembly, consistency validation and atomic write."""

import json
import os

from .errors import ALL_CODES

REQUIRED_TOP_LEVEL = (
    "run_id", "config_hash", "concurrency", "recovered_from", "sequences",
    "cache", "budget", "track_decisions", "thread_peak", "failures",
    "business_codes",
)


def validate_report(report):
    """Field and business-code consistency checks; raises ValueError."""
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in report]
    if missing:
        raise ValueError(f"report missing fields: {missing}")
    for code in report["business_codes"]:
        if code not in ALL_CODES:
            raise ValueError(f"unknown business code in report: {code}")
    for failure in report["failures"]:
        if failure["code"] not in ALL_CODES:
            raise ValueError(f"failure with unknown code: {failure}")
        for key in ("sequence", "frame", "stage", "code", "exception"):
            if key not in failure:
                raise ValueError(f"failure entry missing {key}: {failure}")
    for name, seq in report["sequences"].items():
        for key in ("processed_frames", "truncated_at_frame", "event_segments",
                    "tracks", "quota_deductions"):
            if key not in seq:
                raise ValueError(f"sequence {name} missing {key}")
        if seq["processed_frames"] < len(seq["quota_deductions"]):
            raise ValueError(
                f"sequence {name}: processed_frames < quota deduction count")
    budget = report["budget"]
    if budget["global_consumed_this_run"] != len(budget["quota_deduction_order"]):
        raise ValueError("budget.global_consumed_this_run != deduction order length")
    cache = report["cache"]
    for key in ("hits", "misses", "evictions", "recomputes"):
        if key not in cache or cache[key] < 0:
            raise ValueError(f"cache.{key} invalid")
    return True


def write_report_atomic(report, path):
    validate_report(report)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
