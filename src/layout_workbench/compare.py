from __future__ import annotations

from typing import Any


def canonical_characters(characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for char in characters:
        result.append(
            {
                "index": char.get("index"),
                "segment_index": char.get("segment_index"),
                "bbox": char.get("bbox"),
                "text": char.get("text"),
                "score": char.get("score"),
                "metric": char.get("metric"),
                "status": char.get("status"),
                "tie_break": char.get("tie_break"),
                "candidates": char.get("candidates", []),
                "candidate_count": char.get("candidate_count"),
                "eliminated_beyond_top_k": char.get("eliminated_beyond_top_k"),
                "pending_code": char.get("pending_code"),
                "cache_key": char.get("cache_key"),
            }
        )
    return result


def canonical_budget(budget: dict[str, Any]) -> dict[str, Any]:
    return {
        "limit": budget.get("limit"),
        "point_cost_per_character": budget.get("point_cost_per_character"),
        "consumed": budget.get("consumed"),
        "remaining": budget.get("remaining"),
        "pending_count": budget.get("pending_count"),
    }


def _set_divergence(
    target: dict[str, Any] | None,
    *,
    image: str | None,
    line_index: int | None,
    stage: str,
    field: str,
    incremental: Any,
    full: Any,
) -> dict[str, Any] | None:
    if target is not None:
        return target
    if incremental == full:
        return None
    return {
        "image": image,
        "line_index": line_index,
        "stage": stage,
        "field": field,
        "incremental": incremental,
        "full": full,
    }


def first_divergence(incremental: dict[str, Any], full: dict[str, Any]) -> dict[str, Any] | None:
    divergence: dict[str, Any] | None = None
    inc_images = [image["name"] for image in incremental.get("images", [])]
    full_images = [image["name"] for image in full.get("images", [])]
    divergence = _set_divergence(
        divergence,
        image=None,
        line_index=None,
        stage="load",
        field="image_order",
        incremental=inc_images,
        full=full_images,
    )
    if divergence is not None:
        return divergence
    divergence = _set_divergence(
        divergence,
        image=None,
        line_index=None,
        stage="load",
        field="failures",
        incremental=incremental.get("failures", []),
        full=full.get("failures", []),
    )
    if divergence is not None:
        return divergence
    inc_by_name = {image["name"]: image for image in incremental.get("images", [])}
    full_by_name = {image["name"]: image for image in full.get("images", [])}
    for name in inc_images:
        inc = inc_by_name[name]
        ref = full_by_name[name]
        divergence = _set_divergence(
            divergence,
            image=name,
            line_index=None,
            stage="line_aggregation",
            field="suppressed_lines",
            incremental=inc.get("suppressed_lines", []),
            full=ref.get("suppressed_lines", []),
        )
        if divergence is not None:
            return divergence
        inc_lines = inc.get("lines", [])
        full_lines = ref.get("lines", [])
        divergence = _set_divergence(
            divergence,
            image=name,
            line_index=None,
            stage="line_aggregation",
            field="line_order",
            incremental=[
                {"line_index": line["line_index"], "bbox": line["bbox"]} for line in inc_lines
            ],
            full=[{"line_index": line["line_index"], "bbox": line["bbox"]} for line in full_lines],
        )
        if divergence is not None:
            return divergence
        for line_index, (inc_line, full_line) in enumerate(zip(inc_lines, full_lines)):
            for field in ("bbox", "score", "scale", "blocks"):
                divergence = _set_divergence(
                    divergence,
                    image=name,
                    line_index=line_index,
                    stage="line_aggregation",
                    field=field,
                    incremental=inc_line.get(field),
                    full=full_line.get(field),
                )
                if divergence is not None:
                    return divergence
            for field in ("segments", "segmentation_records", "initial_fragments"):
                divergence = _set_divergence(
                    divergence,
                    image=name,
                    line_index=line_index,
                    stage="segmentation",
                    field=field,
                    incremental=inc_line.get(field),
                    full=full_line.get(field),
                )
                if divergence is not None:
                    return divergence
            for field in ("text", "characters", "pending_count"):
                inc_value = inc_line.get(field)
                full_value = full_line.get(field)
                if field == "characters":
                    inc_value = canonical_characters(inc_value or [])
                    full_value = canonical_characters(full_value or [])
                divergence = _set_divergence(
                    divergence,
                    image=name,
                    line_index=line_index,
                    stage="recognition",
                    field=field,
                    incremental=inc_value,
                    full=full_value,
                )
                if divergence is not None:
                    return divergence
        divergence = _set_divergence(
            divergence,
            image=name,
            line_index=None,
            stage="recognition",
            field="pending_match_queue",
            incremental=inc.get("pending_match_queue", []),
            full=ref.get("pending_match_queue", []),
        )
        if divergence is not None:
            return divergence
    for field in ("pending_match_queue", "budget"):
        inc_value = incremental.get(field)
        full_value = full.get(field)
        if field == "budget":
            inc_value = canonical_budget(inc_value or {})
            full_value = canonical_budget(full_value or {})
        divergence = _set_divergence(
            divergence,
            image=None,
            line_index=None,
            stage="scheduler",
            field=field,
            incremental=inc_value,
            full=full_value,
        )
        if divergence is not None:
            return divergence
    return None


def semantic_projection(report: dict[str, Any]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for image in report.get("images", []):
        images.append(
            {
                "name": image["name"],
                "image_hash": image.get("image_hash"),
                "status": image.get("status"),
                "failures": [],
                "suppressed_lines": image.get("suppressed_lines", []),
                "lines": [
                    {
                        "line_index": line["line_index"],
                        "bbox": line.get("bbox"),
                        "score": line.get("score"),
                        "scale": line.get("scale"),
                        "blocks": line.get("blocks", []),
                        "segments": line.get("segments", []),
                        "segmentation_records": line.get("segmentation_records", []),
                        "initial_fragments": line.get("initial_fragments", []),
                        "text": line.get("text"),
                        "characters": canonical_characters(line.get("characters", [])),
                        "pending_count": line.get("pending_count"),
                    }
                    for line in image.get("lines", [])
                ],
            }
        )
    return {
        "images": images,
        "failures": report.get("failures", []),
        "pending_match_queue": report.get("pending_match_queue", []),
        "budget": canonical_budget(report.get("budget", {})),
    }
