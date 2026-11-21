"""Canonical state comparison used by tests and the acceptance demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .snapshot import RuntimeState, SnapshotStore
from .util import array_content_fingerprint, canonical_hash, canonical_json


def canonical_state_view(state: RuntimeState) -> dict[str, Any]:
    tiles = {
        tile_id: {
            "cache_key": tile.cache_key,
            "content_fingerprint": tile.content_fingerprint,
            "band_order_version": tile.band_order_version,
            "result_version": tile.result_version,
            "class_histogram": tile.class_histogram,
            "feature_summary": tile.feature_summary,
            "score_summary": tile.score_summary,
            "windows": tile.windows,
            "classes_digest": array_content_fingerprint(tile.classes),
            "scores_digest": array_content_fingerprint(tile.scores),
        }
        for tile_id, tile in sorted(state.tiles.items())
    }
    regions = {
        pair_id: {
            "fingerprint": region.fingerprint,
            "region_version": region.region_version,
            "fusion_version": region.fusion_version,
            "overlap_bounds": list(region.overlap_bounds),
            "votes": region.votes,
            "winner_histogram": region.winner_histogram,
        }
        for pair_id, region in sorted(state.regions.items())
    }
    if state.global_result is None:
        global_view = None
    else:
        global_view = {
            "bounds": list(state.global_result.bounds),
            "fusion_version": state.global_result.fusion_version,
            "class_histogram": state.global_result.class_histogram,
            "classes_digest": array_content_fingerprint(state.global_result.classes),
            "overlap_trace": state.global_result.overlap_trace,
            "postprocess_records": state.global_result.postprocess_records,
            "postprocess_components": state.global_result.postprocess_components,
        }
    ledger = {
        f"tile:{tile_id}": {
            "cache_key": tile.cache_key,
            "result_version": tile.result_version,
            "state": "valid",
        }
        for tile_id, tile in sorted(state.tiles.items())
    }
    ledger.update(
        {
            f"region:{pair_id}": {
                "fingerprint": region.fingerprint,
                "region_version": region.region_version,
                "fusion_version": region.fusion_version,
                "state": "valid",
            }
            for pair_id, region in sorted(state.regions.items())
        }
    )
    return {
        "tiles": tiles,
        "regions": regions,
        "global": global_view,
        "reuse_recompute_ledger": ledger,
    }


def state_view_hash(state: RuntimeState) -> str:
    return canonical_hash(canonical_state_view(state), prefix="state-view-v1|")


def compare_states(
    left: RuntimeState, right: RuntimeState
) -> dict[str, Any]:
    left_view = canonical_state_view(left)
    right_view = canonical_state_view(right)
    categories = {
        "global_classes": left_view["global"] is not None
        and right_view["global"] is not None
        and left_view["global"]["classes_digest"] == right_view["global"]["classes_digest"],
        "global_trace": left_view["global"] is not None
        and right_view["global"] is not None
        and canonical_json(left_view["global"]["overlap_trace"])
        == canonical_json(right_view["global"]["overlap_trace"]),
        "postprocess_backfills": left_view["global"] is not None
        and right_view["global"] is not None
        and canonical_json(left_view["global"]["postprocess_records"])
        == canonical_json(right_view["global"]["postprocess_records"]),
        "tile_summaries": canonical_json(
            {
                key: {field: value for field, value in item.items() if field not in {"classes_digest", "scores_digest"}}
                for key, item in left_view["tiles"].items()
            }
        )
        == canonical_json(
            {
                key: {field: value for field, value in item.items() if field not in {"classes_digest", "scores_digest"}}
                for key, item in right_view["tiles"].items()
            }
        ),
        "tile_pixels": left_view["tiles"].keys() == right_view["tiles"].keys()
        and all(
            left_view["tiles"][key]["classes_digest"] == right_view["tiles"][key]["classes_digest"]
            and left_view["tiles"][key]["scores_digest"] == right_view["tiles"][key]["scores_digest"]
            for key in left_view["tiles"]
        ),
        "fusion_region_votes": canonical_json(left_view["regions"])
        == canonical_json(right_view["regions"]),
        "reuse_recompute_ledger": canonical_json(left_view["reuse_recompute_ledger"])
        == canonical_json(right_view["reuse_recompute_ledger"]),
    }
    equal = all(categories.values()) and canonical_json(left_view) == canonical_json(right_view)
    return {
        "equal": equal,
        "categories": categories,
        "left_hash": canonical_hash(left_view, prefix="state-view-v1|"),
        "right_hash": canonical_hash(right_view, prefix="state-view-v1|"),
        "left_state_fingerprint": left.manifest.get("state_fingerprint") if left.manifest else None,
        "right_state_fingerprint": right.manifest.get("state_fingerprint") if right.manifest else None,
    }


def load_state(state_dir: Path) -> RuntimeState:
    state = SnapshotStore(state_dir).load()
    if state is None:
        raise FileNotFoundError(f"no snapshot at {state_dir}")
    return state
