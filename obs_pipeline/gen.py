"""Deterministic sample generation from integer seeds (requirement 3).

Generation is keyed by (source.seed, record index) only: each source stream is
produced from random.Random(source.seed) consumed in a fixed order, so any
batch split or load order yields bit-identical record sequences, null
positions and digests. An out-of-order arrival permutation is generated per
source; loaders must sort by obs_seq before validation, which restores the
canonical order regardless of arrival order.
"""

from __future__ import annotations

import hashlib
import json
import random

from .records import NormalizedRecord, decimal_to_fixed, encode_line


def generate_source_records(source):
    """Return the canonical list of NormalizedRecord for a source declaration."""
    rng = random.Random(source.seed)
    records = []
    for index in range(source.record_count):
        seq = index + 1
        if rng.random() < 0.5:
            raw_value = rng.randint(-5000, 5000)
        else:
            raw_value = round(rng.uniform(-500.0, 500.0), 3)
        timestamp = index * 10 + rng.randint(0, 3)
        metric_key = source.metrics[rng.randrange(len(source.metrics))]
        is_null = source.null_bitmap[index] == 1
        value_fixed = None if is_null else decimal_to_fixed(str(raw_value), source.default_scale)
        records.append(
            NormalizedRecord(
                version=source.record_format,
                seq=seq,
                source_id=source.id,
                timestamp=timestamp,
                metric_key=metric_key,
                scale=source.default_scale,
                value_fixed=value_fixed,
                filled_fields=(),
                fill_rationale=(),
            )
        )
    return records


def generate_arrival_order(source):
    """Out-of-order arrival permutation (deterministic, distinct stream seed)."""
    rng = random.Random((source.seed << 16) ^ 0x5A5A)
    order = list(range(source.record_count))
    rng.shuffle(order)
    return order


def render_stream_lines(source, arrival_order=False):
    """Render on-disk lines; optionally in (deterministic) arrival order."""
    records = generate_source_records(source)
    if arrival_order:
        order = generate_arrival_order(source)
        records = [records[i] for i in order]
    return [encode_line(r) for r in records]


def stream_digest(lines):
    """Bit-stable digest of raw on-disk lines."""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def records_digest(records):
    """Bit-stable digest of normalized records."""
    return hashlib.sha256("\n".join(r.canonical() for r in records).encode("utf-8")).hexdigest()


def build_sample_spec(seed=42):
    """Build the deterministic sample spec dict used by the demo and tests."""
    rng = random.Random(seed)
    sources = []
    total = 0
    declarations = [
        ("src-a", 1, 60, ["m0", "m1"], 100),
        ("src-b", 2, 50, ["m0"], 1000),
        ("src-c", 3, 40, ["m1", "m2"], 10),
    ]
    for sid, fmt, count, metrics, scale in declarations:
        src_seed = rng.randint(1, 10 ** 6)
        null_rng = random.Random(src_seed + 7777)
        bitmap = [1 if null_rng.random() < 0.1 else 0 for _ in range(count)]
        sources.append(
            {
                "id": sid,
                "seed": src_seed,
                "record_count": count,
                "record_format": fmt,
                "metrics": metrics,
                "default_scale": scale,
                "null_bitmap": bitmap,
            }
        )
        total += count
    windows = []
    start = 0
    width = total // 5
    for index in range(5):
        end = total if index == 4 else start + width
        windows.append({"id": "w%d" % index, "start": start, "end": end})
        start = end
    return {
        "spec_version": 1,
        "task_interval": {"start": 0, "end": total},
        "stages": [
            {"id": "s_diff", "kind": "std_diff", "params": {}, "scale": 100, "upstream": []},
            {"id": "s_avg", "kind": "window_avg", "params": {"window": 4}, "scale": 100, "upstream": ["s_diff"]},
            {
                "id": "s_bucket",
                "kind": "quantile_bucket",
                "params": {"buckets": 8, "lo": -20000, "hi": 20000},
                "scale": 1,
                "upstream": ["s_avg"],
            },
            {
                "id": "s_norm",
                "kind": "threshold_norm",
                "params": {"threshold": 2, "span": 6},
                "scale": 1000,
                "upstream": ["s_bucket"],
            },
        ],
        "sources": sources,
        "windows": windows,
        "drift_threshold": 250,
        "dominance_threshold_ppm": 400000,
    }


def write_sample_tree(spec, root="samples"):
    """Write spec.json and one record file per source under root/."""
    import os

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "spec.json"), "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
    from .spec import validate_spec

    parsed = validate_spec(spec)
    paths = {}
    for source in parsed.sources:
        path = os.path.join(root, "%s.v%d.rec" % (source.id, source.record_format))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(render_stream_lines(source)) + "\n")
        paths[source.id] = path
    return paths
