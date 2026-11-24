from __future__ import annotations

from decimal import Decimal
from typing import Any

from .config import LoadedConfig
from .fixedpoint import decimal_to_q, round_div
from .samples import Sample


def build_imputation_index(
    config: LoadedConfig,
    samples: list[Sample],
    input_scale: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    raw = config.raw
    windows = raw["windows"]
    channel_count = len(config.channels)
    sums: dict[tuple[str, int], int] = {}
    counts: dict[tuple[str, int], int] = {}
    members: dict[tuple[str, int], list[int]] = {}
    result: dict[tuple[int, int], dict[str, Any]] = {}

    def scope_for_seq(seq: int) -> str:
        for name in ("reference", "observation"):
            window = windows[name]
            if int(window["start_seq"]) <= seq <= int(window["end_seq"]):
                return name
        return "all_loaded"

    for sample in sorted(samples, key=lambda item: item.seq):
        scope = scope_for_seq(sample.seq)
        for channel_index in range(channel_count):
            scope_key = (scope, channel_index)
            all_key = ("all_loaded", channel_index)
            if sample.missing[channel_index]:
                count = counts.get(scope_key, 0)
                if count > 0:
                    sum_q = sums[scope_key]
                    mean_q = round_div(sum_q, count)
                    source = {
                        "basis": "window_seen_mean",
                        "scope": scope,
                        "channel_index": channel_index,
                        "observed_count": count,
                        "sum_q": sum_q,
                        "mean_q": mean_q,
                        "scale": input_scale,
                        "sample_set": list(members.get(scope_key, [])),
                    }
                else:
                    count = counts.get(all_key, 0)
                    if count > 0:
                        sum_q = sums[all_key]
                        mean_q = round_div(sum_q, count)
                        source = {
                            "basis": "global_seen_mean_before_window_start",
                            "scope": "all_loaded",
                            "channel_index": channel_index,
                            "observed_count": count,
                            "sum_q": sum_q,
                            "mean_q": mean_q,
                            "scale": input_scale,
                            "sample_set": list(members.get(all_key, [])),
                        }
                    else:
                        source = {
                            "basis": "zero_before_first_seen_sample",
                            "scope": scope,
                            "channel_index": channel_index,
                            "observed_count": 0,
                            "sum_q": 0,
                            "mean_q": 0,
                            "scale": input_scale,
                            "sample_set": [],
                        }
                result[(sample.seq, channel_index)] = source
                continue

            value = sample.values[channel_index]
            if value is None:
                continue
            q = decimal_to_q(value, input_scale)
            sums[scope_key] = sums.get(scope_key, 0) + q
            counts[scope_key] = counts.get(scope_key, 0) + 1
            members.setdefault(scope_key, []).append(sample.seq)
            sums[all_key] = sums.get(all_key, 0) + q
            counts[all_key] = counts.get(all_key, 0) + 1
            members.setdefault(all_key, []).append(sample.seq)
    return result


def scope_for_seq(config: LoadedConfig, seq: int) -> str:
    windows = config.raw["windows"]
    for name in ("reference", "observation"):
        window = windows[name]
        if int(window["start_seq"]) <= seq <= int(window["end_seq"]):
            return name
    return "outside_windows"
