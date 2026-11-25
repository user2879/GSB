from __future__ import annotations

from typing import Any


PRIORITY_RULE = (
    "authoritative_keypoint_geometry > block_center_geometry > "
    "validated_truth_reading_no > stable_block_id"
)


def choose_authoritative_keypoint(
    submissions: list[dict[str, Any]],
    image_name: str,
    block_id: str,
) -> dict[str, Any] | None:
    candidates = [s for s in submissions if s.get("image") == image_name and s.get("block_id") == block_id]
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda s: (
            -int(s["version_no"]),
            -int(s["annotated_ms"]),
            str(s["annotator"]),
        ),
    )
    winner = ordered[0]
    return {
        "image": image_name,
        "block_id": block_id,
        "x": int(winner["x"]),
        "y": int(winner["y"]),
        "version_no": int(winner["version_no"]),
        "version_name": winner["version_name"],
        "annotation_version": f"{winner['version_name']}#{winner['version_no']}/{winner['annotator']}@{winner['annotated_ms']}",
        "annotator": winner["annotator"],
        "annotated_ms": int(winner["annotated_ms"]),
        "rule": "version_no_desc,annotated_ms_desc,annotator_lex_asc",
        "candidate_order": [
            {
                "version_no": int(s["version_no"]),
                "annotated_ms": int(s["annotated_ms"]),
                "annotator": s["annotator"],
                "version_name": s["version_name"],
            }
            for s in ordered
        ],
    }


def iou(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / (area_a + area_b - intersection)


def associate_truth_block(
    detected_bbox: list[int],
    truth_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    best_score = 0.0
    cx = (detected_bbox[0] + detected_bbox[2]) / 2
    cy = (detected_bbox[1] + detected_bbox[3]) / 2
    for block in truth_blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        score = iou(detected_bbox, bbox)
        if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
            score += 0.25
        if score > best_score:
            best_score = score
            best = block
    return best if best_score >= 0.18 else None


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, int, int, str]:
    kp = entry.get("authoritative_keypoint")
    if kp:
        y = int(kp["y"])
    else:
        y = int((entry["bbox"][1] + entry["bbox"][3]) // 2)
    truth_no = entry.get("truth_reading_no")
    return (int(entry.get("rtl_index", 0)), y, int(truth_no) if truth_no else 10**9, str(entry.get("character_id", "")))


def detect_conflicts(entries: list[dict[str, Any]], pending_by_truth_no: dict[int, str]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    ordered = sorted(entries, key=_entry_sort_key)

    for prev, nxt in zip(ordered, ordered[1:]):
        prev_no = prev.get("truth_reading_no")
        next_no = nxt.get("truth_reading_no")
        if not isinstance(prev_no, int) or not isinstance(next_no, int):
            continue
        same_column = prev.get("truth_column_id") == nxt.get("truth_column_id")
        ids = [prev.get("character_id"), nxt.get("character_id")]
        if same_column:
            if next_no < prev_no:
                conflicts.append({
                    "type": "wrong_order",
                    "character_ids": ids,
                    "truth_reading_numbers": [prev_no, next_no],
                    "detail": "top-down keypoint geometry contradicts truth reading order",
                    "resolution": "geometry_priority",
                    "priority_rule": PRIORITY_RULE,
                })
            elif next_no > prev_no + 1:
                missing = list(range(prev_no + 1, next_no))
                conflicts.append({
                    "type": "missing_number",
                    "character_ids": ids,
                    "missing_reading_numbers": missing,
                    "pending_character_ids": [pending_by_truth_no.get(no) for no in missing],
                    "detail": "reading sequence has a gap",
                    "resolution": "keep_gap_and_pending_marker",
                    "priority_rule": PRIORITY_RULE,
                })
        else:
            rtl_advances = int(nxt.get("rtl_index", 0)) >= int(prev.get("rtl_index", 0))
            if next_no <= prev_no or next_no != prev_no + 1 or not rtl_advances:
                conflicts.append({
                    "type": "cross_column",
                    "character_ids": ids,
                    "truth_reading_numbers": [prev_no, next_no],
                    "columns": [prev.get("truth_column_id"), nxt.get("truth_column_id")],
                    "detail": "column attribution/RTL geometry conflicts with truth reading numbers",
                    "resolution": "rtl_geometry_priority",
                    "priority_rule": PRIORITY_RULE,
                })

    active = [e for e in ordered if e.get("match_status") != "E_MATCH_BUDGET_PENDING"]
    for prev, nxt in zip(active, active[1:]):
        prev_no = prev.get("truth_reading_no")
        next_no = nxt.get("truth_reading_no")
        if isinstance(prev_no, int) and isinstance(next_no, int) and next_no > prev_no + 1:
            missing = list(range(prev_no + 1, next_no))
            if not any(c.get("type") == "missing_number" and c.get("missing_reading_numbers") == missing for c in conflicts):
                conflicts.append({
                    "type": "missing_number",
                    "character_ids": [prev.get("character_id"), nxt.get("character_id")],
                    "missing_reading_numbers": missing,
                    "pending_character_ids": [pending_by_truth_no.get(no) for no in missing],
                    "detail": "budget-pending character creates an explicit reading gap",
                    "resolution": "next_budget_batch_repairs_gap",
                    "priority_rule": PRIORITY_RULE,
                })
    return conflicts


def build_reading_sequence(
    image_name: str,
    character_records: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    pending_by_truth_no: dict[int, str] = {}
    for record in character_records:
        block_id = record.get("truth_block_id") or record.get("block_key")
        authority = choose_authoritative_keypoint(submissions, image_name, block_id) if block_id else None
        bbox = record["bbox"]
        entry = {
            "image": image_name,
            "character_id": block_id or record.get("stable_char_key"),
            "detected_key": record.get("stable_char_key"),
            "truth_block_id": block_id,
            "truth_column_id": record.get("truth_column_id"),
            "truth_reading_no": record.get("truth_reading_no"),
            "rtl_index": record.get("rtl_index", 0),
            "bbox": bbox,
            "recognized_char": record.get("recognized_char"),
            "confidence": record.get("confidence"),
            "match_status": record.get("match_status", "matched"),
            "authoritative_keypoint": authority,
            "keypoint_coordinates": [authority["x"], authority["y"]] if authority else None,
            "keypoint_version": authority["version_name"] if authority else "truth_center_fallback",
            "annotation_version": authority["annotation_version"] if authority else "truth.keypoint",
        }
        if isinstance(entry["truth_reading_no"], int) and entry["match_status"] == "E_MATCH_BUDGET_PENDING":
            pending_by_truth_no[entry["truth_reading_no"]] = entry["character_id"]
        entries.append(entry)

    ordered = sorted(entries, key=_entry_sort_key)
    for index, entry in enumerate(ordered, 1):
        entry["reading_no"] = index
        entry["reading_order_name"] = (
            "rtl_top_to_bottom_conflict_resolved"
            if entry.get("truth_reading_no") != index
            else "rtl_top_to_bottom"
        )
    conflicts = detect_conflicts(ordered, pending_by_truth_no)
    conflicted_ids = {cid for c in conflicts for cid in c.get("character_ids", [])}
    for entry in ordered:
        entry["in_conflict"] = entry["character_id"] in conflicted_ids
    return {
        "image": image_name,
        "priority_rule": PRIORITY_RULE,
        "entries": ordered,
        "conflicts": conflicts,
        "conflict_counts": {
            kind: sum(1 for c in conflicts if c["type"] == kind)
            for kind in ("cross_column", "wrong_order", "missing_number")
        },
    }
