from __future__ import annotations

import copy
import math
from typing import Any

from .fingerprint import stable_float


def compute_woe(
    column: str,
    column_type: str,
    bins: list[dict[str, Any]],
    smoothing: float,
) -> dict[str, Any]:
    """Compute additive-smoothed WOE and IV with bin-level audit flags."""
    adjusted_bad: list[float] = []
    adjusted_good: list[float] = []
    smoothed_flags: list[bool] = []
    for bin_value in bins:
        bad = int(bin_value["bad"])
        good = int(bin_value["good"])
        smoothed = bad == 0 or good == 0
        adjusted_bad.append(float(bad + smoothing) if bad == 0 else float(bad))
        adjusted_good.append(float(good + smoothing) if good == 0 else float(good))
        smoothed_flags.append(smoothed)

    total_bad = sum(adjusted_bad)
    total_good = sum(adjusted_good)
    output_bins: list[dict[str, Any]] = []
    iv = 0.0
    for bin_value, bad, good, smoothed in zip(bins, adjusted_bad, adjusted_good, smoothed_flags):
        bad_share = bad / total_bad if total_bad else 0.0
        good_share = good / total_good if total_good else 0.0
        if bad_share <= 0.0 or good_share <= 0.0:
            # The validator requires a positive smoothing coefficient. This
            # branch is a last-resort guard against numerically empty columns.
            woe = 0.0
        else:
            woe = stable_float(math.log(bad_share / good_share))
        contribution = stable_float((bad_share - good_share) * woe)
        iv += contribution
        item = copy.deepcopy(bin_value)
        item["bad_share"] = stable_float(bad_share)
        item["good_share"] = stable_float(good_share)
        item["woe"] = woe
        item["iv_contribution"] = contribution
        item["smoothed"] = bool(smoothed)
        output_bins.append(item)

    return {
        "column": column,
        "type": column_type,
        "smoothing": stable_float(smoothing),
        "bins": output_bins,
        "iv": stable_float(iv),
        "smoothed_bin_count": sum(1 for flag in smoothed_flags if flag),
        "total_bad": int(sum(int(item["bad"]) for item in bins)),
        "total_good": int(sum(int(item["good"]) for item in bins)),
        "adjusted_total_bad": stable_float(total_bad),
        "adjusted_total_good": stable_float(total_good),
    }
