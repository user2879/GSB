"""In-training / extrapolation layered statistics.

Aggregates successful score results per (tier, scheme, precision).  Scores at
the same relative distance are summarized separately for the ``inside`` and
``extrap`` regions.  To keep float summation order independent of whether the
data was just computed or resumed from snapshot objects, every value carries a
deterministic ``(sample_id, precision, key_index)`` ordering tuple and all
statistics reduce over the sorted list.  Failures never enter aggregation.
"""

from __future__ import annotations

import numpy as np


def _stats(entries: list[tuple]) -> dict:
    # entries: [(order_tuple, value), ...] already sorted by order_tuple
    values = np.asarray([v for _, v in sorted(entries, key=lambda e: e[0])],
                        dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def aggregate(score_results: list[dict], tier_id: str) -> dict:
    by_scheme: dict[str, dict] = {}
    for res in score_results:
        if res["tier_id"] != tier_id:
            continue
        scheme = res["scheme"]
        node = by_scheme.setdefault(scheme, {
            "per_sample": [],
            "region_buckets": {"inside": {}, "extrap": {}},
            "precisions": set(),
        })
        node["precisions"].add(res["precision"])
        node["per_sample"].append({
            "sample_id": res["sample_id"],
            "precision": res["precision"],
            "regions": res["regions"],
            "rel_distances": res["rel_distances"],
            "raw_scores": [float(x) for x in res["raw_scores"]],
            "weights": [float(x) for x in res["weights"]],
        })
        for i, rel in enumerate(res["rel_distances"]):
            region = res["regions"][i]
            order = (res["sample_id"], res["precision"], i)
            bucket = node["region_buckets"][region].setdefault(str(rel), [])
            bucket.append((order, float(res["scaled_scores"][i])))

    scheme_blocks = {}
    for scheme, node in by_scheme.items():
        distance_summary = []
        all_distances = set(node["region_buckets"]["inside"]) | \
            set(node["region_buckets"]["extrap"])
        for rel in sorted(all_distances, key=lambda x: int(x)):
            distance_summary.append({
                "rel_distance": int(rel),
                "inside": _stats(
                    node["region_buckets"]["inside"].get(rel, [])),
                "extrap": _stats(
                    node["region_buckets"]["extrap"].get(rel, [])),
            })
        region_counts = {"inside": 0, "extrap": 0}
        for region, bucket_map in node["region_buckets"].items():
            region_counts[region] = sum(len(v) for v in bucket_map.values())
        scheme_blocks[scheme] = {
            "precisions": sorted(node["precisions"]),
            "region_counts": region_counts,
            "distance_summary": distance_summary,
            "per_sample": sorted(node["per_sample"],
                                 key=lambda row: (row["sample_id"],
                                                  row["precision"])),
        }
    return {"tier_id": tier_id, "schemes": scheme_blocks}
