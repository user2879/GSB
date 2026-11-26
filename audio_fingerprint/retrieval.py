from __future__ import annotations

from collections import defaultdict
from typing import Any

from .buckets import bucket_index
from .canonical import fingerprint
from .config import vote_config_fingerprint


def collect_hits(
    query_pairs: list[dict[str, Any]],
    buckets: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    index = buckets if isinstance(buckets, dict) else bucket_index(buckets)
    hits: list[dict[str, Any]] = []
    for query_pair in sorted(query_pairs, key=lambda item: item["product_id"]):
        bucket = index.get(query_pair["fingerprint_key"])
        if bucket is None:
            continue
        for entry in bucket["entries"]:
            hits.append(
                {
                    "query_pair_id": query_pair["product_id"],
                    "query_cache_key": query_pair["cache_key"],
                    "fingerprint_key": query_pair["fingerprint_key"],
                    "clip_id": entry["clip_id"],
                    "track_id": entry["track_id"],
                    "version": int(entry["version"]),
                    "db_anchor_frame": int(entry["anchor_frame"]),
                    "query_anchor_frame": int(query_pair["anchor_frame"]),
                    "offset_frames": int(entry["anchor_frame"]) - int(query_pair["anchor_frame"]),
                    "strength_q": int(entry["strength_q"]),
                    "bucket_cache_key": bucket["cache_key"],
                }
            )
    hits.sort(key=lambda hit: (hit["track_id"], hit["version"], hit["clip_id"], hit["offset_frames"], hit["query_pair_id"]))
    return hits


def rank_candidates(hits: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        grouped[(hit["track_id"], int(hit["version"]), hit["clip_id"])].append(hit)

    candidates: list[dict[str, Any]] = []
    for (track_id, version, clip_id), candidate_hits in grouped.items():
        candidates.append(
            {
                "track_id": track_id,
                "version": version,
                "clip_id": clip_id,
                "hit_count": len(candidate_hits),
                "total_strength_q": sum(int(hit["strength_q"]) for hit in candidate_hits),
                "hits": candidate_hits,
            }
        )

    candidates.sort(key=lambda item: (-item["hit_count"], item["track_id"], item["version"], item["clip_id"]))
    limit = int(config["candidate_limit"])
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate["selected_for_vote"] = rank <= limit
        candidate["boundary_candidate"] = False
        candidate["boundary_detail"] = None

    if len(candidates) > limit and limit > 0:
        boundary = candidates[limit - 1]
        next_candidate = candidates[limit]
        boundary["boundary_candidate"] = True
        boundary["boundary_detail"] = {
            "boundary_rank": limit,
            "boundary_hit_count": boundary["hit_count"],
            "next_rank": next_candidate["rank"],
            "next_candidate": {
                "track_id": next_candidate["track_id"],
                "version": next_candidate["version"],
                "clip_id": next_candidate["clip_id"],
            },
            "next_hit_count": next_candidate["hit_count"],
            "hit_count_delta": boundary["hit_count"] - next_candidate["hit_count"],
            "strength_delta_q": boundary["total_strength_q"] - next_candidate["total_strength_q"],
            "at_cutoff_tie": boundary["hit_count"] == next_candidate["hit_count"],
        }
    return candidates


def vote_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    fraction_bits = int(config["vote_fraction_bits"])
    unit_q = 1 << fraction_bits
    hist: dict[int, dict[str, Any]] = {}
    for hit in candidate["hits"]:
        offset = int(hit["offset_frames"])
        bucket = hist.setdefault(
            offset,
            {
                "offset_frames": offset,
                "offset_q": offset * unit_q,
                "hit_count": 0,
                "votes_q": 0,
                "strength_sum_q": 0,
            },
        )
        bucket["hit_count"] += 1
        bucket["votes_q"] += unit_q
        bucket["strength_sum_q"] += int(hit["strength_q"])

    histogram = sorted(hist.values(), key=lambda item: item["offset_frames"])
    ranked_offsets = sorted(
        histogram,
        key=lambda item: (-int(item["votes_q"]), abs(int(item["offset_frames"])), int(item["offset_frames"])),
    )
    main = ranked_offsets[0] if ranked_offsets else None
    tied = []
    if main is not None:
        tied = [
            item["offset_frames"]
            for item in histogram
            if item["votes_q"] == main["votes_q"] and item["hit_count"] == main["hit_count"]
        ]
    return {
        "histogram": histogram,
        "main_offset_frames": None if main is None else int(main["offset_frames"]),
        "main_offset_q": None if main is None else int(main["offset_q"]),
        "main_votes_q": 0 if main is None else int(main["votes_q"]),
        "main_hit_count": 0 if main is None else int(main["hit_count"]),
        "main_strength_sum_q": 0 if main is None else int(main["strength_sum_q"]),
        "tied_offsets": tied,
        "tie_rule": "fixed_point_votes_q desc, abs(offset) asc, offset asc",
    }


def arbitrate_versions(conclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conclusion in conclusions:
        grouped[conclusion["track_id"]].append(conclusion)

    arbitrations: list[dict[str, Any]] = []
    for track_id in sorted(grouped):
        versions = grouped[track_id]
        ordered = sorted(
            versions,
            key=lambda item: (
                -int(item.get("main_hit_count", item["hit_count"])),
                -int(item.get("main_strength_sum_q", 0)),
                -int(item["main_votes_q"]),
                -int(item["version"]),
                int(item["version"]),
                item["clip_id"],
            ),
        )
        winner = ordered[0]
        offsets = {item["main_offset_frames"] for item in versions}
        covered: list[dict[str, Any]] = []
        for item in ordered[1:]:
            item_aligned = int(item.get("main_hit_count", item["hit_count"]))
            winner_aligned = int(winner.get("main_hit_count", winner["hit_count"]))
            if item_aligned < winner_aligned:
                basis = "lower_aligned_hit_count"
            elif int(item.get("main_strength_sum_q", 0)) < int(winner.get("main_strength_sum_q", 0)):
                basis = "lower_aligned_strength"
            elif int(item["version"]) < int(winner["version"]):
                basis = "lower_version_equal_strength"
            else:
                basis = "stable_clip_id_order"
            covered.append(
                {
                    "track_id": track_id,
                    "version": int(item["version"]),
                    "clip_id": item["clip_id"],
                    "main_offset_frames": item["main_offset_frames"],
                    "hit_count": int(item["hit_count"]),
                    "main_hit_count": item_aligned,
                    "main_votes_q": int(item["main_votes_q"]),
                    "main_strength_sum_q": int(item.get("main_strength_sum_q", 0)),
                    "covered_by_version": int(winner["version"]),
                    "coverage_reason": basis,
                }
            )

        if len(ordered) == 1:
            decision_basis = "single_version"
        elif int(winner.get("main_hit_count", winner["hit_count"])) > int(
            ordered[1].get("main_hit_count", ordered[1]["hit_count"])
        ):
            decision_basis = "higher_aligned_hit_count"
        elif int(winner.get("main_strength_sum_q", 0)) > int(ordered[1].get("main_strength_sum_q", 0)):
            decision_basis = "stronger_aligned_strength"
        else:
            decision_basis = "higher_version_equal_strength"

        arbitrations.append(
            {
                "track_id": track_id,
                "offset_conflict": len(offsets) > 1,
                "distinct_offsets": sorted(offset for offset in offsets if offset is not None),
                "authoritative_version": int(winner["version"]),
                "authoritative_clip_id": winner["clip_id"],
                "authoritative_offset_frames": winner["main_offset_frames"],
                "authoritative_votes_q": winner["main_votes_q"],
                "authoritative_hit_count": int(winner.get("main_hit_count", winner["hit_count"])),
                "authoritative_main_hit_count": int(winner.get("main_hit_count", winner["hit_count"])),
                "authoritative_candidate_hit_count": int(winner["hit_count"]),
                "authoritative_main_strength_q": int(winner.get("main_strength_sum_q", 0)),
                "authoritative_total_strength_q": int(winner.get("total_strength_q", 0)),
                "decision_basis": decision_basis,
                "covered_versions": covered,
            }
        )
    arbitrations.sort(key=lambda item: item["track_id"])
    return arbitrations


def run_query(
    query_clip: dict[str, Any],
    query_pairs: list[dict[str, Any]],
    buckets: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    hits = collect_hits(query_pairs, buckets)
    candidates = rank_candidates(hits, config)
    conclusions: list[dict[str, Any]] = []
    for candidate in candidates:
        vote = vote_candidate(candidate, config)
        conclusions.append(
            {
                "track_id": candidate["track_id"],
                "version": candidate["version"],
                "clip_id": candidate["clip_id"],
                "rank": candidate["rank"],
                "selected_for_vote": candidate["selected_for_vote"],
                "boundary_candidate": candidate["boundary_candidate"],
                "boundary_detail": candidate["boundary_detail"],
                "hit_count": candidate["hit_count"],
                "total_strength_q": candidate["total_strength_q"],
                **vote,
            }
        )

    voted_conclusions = [item for item in conclusions if item["selected_for_vote"]]
    arbitrations = arbitrate_versions(voted_conclusions)
    arbitration_by_track = {item["track_id"]: item for item in arbitrations}
    for conclusion in conclusions:
        conclusion["version_arbitration"] = arbitration_by_track.get(conclusion["track_id"])

    authoritative_tracks = sorted(
        arbitrations,
        key=lambda item: (
            -int(item["authoritative_hit_count"]),
            -int(item["authoritative_votes_q"]),
            item["track_id"],
        ),
    )
    final = authoritative_tracks[0] if authoritative_tracks else None
    touched_bucket_keys = sorted({hit["bucket_cache_key"] for hit in hits})
    result_cache_key = fingerprint(
        {
            "product_type": "query_result",
            "query_clip_id": query_clip["clip_id"],
            "query_content_fingerprint": query_clip.get("content_fingerprint"),
            "query_pair_cache_keys": sorted(pair["cache_key"] for pair in query_pairs),
            "touched_bucket_cache_keys": touched_bucket_keys,
            "config_fingerprint": vote_config_fingerprint(config),
        },
        "qresult_",
    )
    return {
        "product_type": "query_result",
        "product_id": f"query:{query_clip['clip_id']}",
        "cache_key": result_cache_key,
        "query_clip_id": query_clip["clip_id"],
        "query_track_hint": query_clip.get("track_id"),
        "hit_count_total": len(hits),
        "candidates": [
            {key: value for key, value in candidate.items() if key != "hits"} for candidate in candidates
        ],
        "conclusions": conclusions,
        "version_arbitrations": arbitrations,
        "final_conclusion": final,
        "touched_bucket_cache_keys": touched_bucket_keys,
        "upstream_refs": [{"type": "bucket", "key": key} for key in touched_bucket_keys]
        + [{"type": "vote_config", "key": vote_config_fingerprint(config)}],
    }
