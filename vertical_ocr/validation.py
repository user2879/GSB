from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .codes import INPUT_INVALID
from .config import canonical_hash


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _failure(message: str, **context: Any) -> dict[str, Any]:
    return {"code": INPUT_INVALID, "message": message, "stage": "load_validation", **context}


def inspect_png(path: Path, config: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        header = path.read_bytes()[:8]
        if header != PNG_SIGNATURE:
            raise ValueError("bad PNG container signature")
        with Image.open(path) as img:
            if img.format != "PNG":
                raise ValueError(f"container format is {img.format}, expected PNG")
            img.verify()
        with Image.open(path) as img:
            img.load()
            expected_size = (int(config["layout"]["width"]), int(config["layout"]["height"]))
            if img.size != expected_size:
                raise ValueError(f"fixed resolution mismatch: got {img.size}, expected {expected_size}")
            if img.mode != config["layout"].get("mode", "L"):
                raise ValueError(f"pixel mode mismatch: got {img.mode}, expected L")
            arr = np.asarray(img, dtype=np.uint8).copy()
        fingerprint = hashlib.sha256(arr.shape[0].to_bytes(8, "little") + arr.shape[1].to_bytes(8, "little") + arr.tobytes()).hexdigest()
        return {
            "name": path.name,
            "path": str(path),
            "array": arr,
            "fingerprint": fingerprint,
            "file_fingerprint": hashlib.sha256(path.read_bytes()).hexdigest(),
        }, None
    except Exception as exc:
        return None, _failure(f"invalid PNG {path.name}: {exc}", image=path.name, field="png")


def _bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4 or not all(_is_int(v) for v in value):
        return None
    return value


def validate_truth(
    truth: dict[str, Any],
    images: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    block_index: dict[tuple[str, str], dict[str, Any]] = {}
    width = int(config["layout"]["width"])
    height = int(config["layout"]["height"])
    if not isinstance(truth, dict) or not isinstance(truth.get("images"), list):
        failures.append(_failure("layout_truth.json must contain images array", field="layout_truth"))
        return block_index, failures

    truth_names = set()
    seen_block_ids: set[str] = set()
    for img_pos, img_entry in enumerate(truth["images"]):
        ctx_image = img_entry.get("image", f"images[{img_pos}]") if isinstance(img_entry, dict) else f"images[{img_pos}]"
        if not isinstance(img_entry, dict):
            failures.append(_failure("truth image entry must be object", image=ctx_image))
            continue
        name = img_entry.get("image")
        truth_names.add(name)
        if not _is_int(img_entry.get("image_no")):
            failures.append(_failure("image_no must be integer", image=name, field="image_no"))
        if name not in images:
            failures.append(_failure("truth references missing/invalid PNG", image=name, field="image"))
        columns = img_entry.get("columns")
        if not isinstance(columns, list) or not columns:
            failures.append(_failure("truth image must have non-empty columns", image=name, field="columns"))
            continue

        x_intervals: list[tuple[int, int, str]] = []
        for col_pos, column in enumerate(columns):
            col_id = column.get("column_id", f"col-{col_pos}")
            if not isinstance(column, dict):
                failures.append(_failure("column must be object", image=name, column=col_id))
                continue
            col_bbox = _bbox4(column.get("bbox"))
            if col_bbox is None:
                failures.append(_failure("column bbox must be four integers", image=name, column=col_id, field="bbox"))
                continue
            x1, y1, x2, y2 = col_bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                failures.append(_failure("column bbox is outside image or malformed", image=name, column=col_id, field="bbox"))
            else:
                x_intervals.append((x1, x2, col_id))

            blocks = column.get("blocks")
            if not isinstance(blocks, list) or not blocks:
                failures.append(_failure("column must contain blocks", image=name, column=col_id, field="blocks"))
                continue
            reading_numbers: list[int] = []
            for block in blocks:
                block_id = block.get("block_id", "<missing>")
                if block_id in seen_block_ids:
                    failures.append(_failure("duplicate block_id", image=name, column=col_id, block_id=block_id))
                seen_block_ids.add(block_id)
                bbox = _bbox4(block.get("bbox"))
                if bbox is None:
                    failures.append(_failure("block bbox must be four integers", image=name, block_id=block_id, field="bbox"))
                    continue
                bx1, by1, bx2, by2 = bbox
                if not (0 <= bx1 < bx2 <= width and 0 <= by1 < by2 <= height):
                    failures.append(_failure("block bbox is outside image", image=name, block_id=block_id, field="bbox"))
                elif col_bbox and not (col_bbox[0] <= bx1 and bx2 <= col_bbox[2]):
                    failures.append(_failure("block x interval crosses its column interval", image=name, block_id=block_id, field="bbox"))
                kp = block.get("keypoint")
                if not isinstance(kp, list) or len(kp) != 2 or not all(_is_int(v) for v in kp):
                    failures.append(_failure("keypoint must be two integers", image=name, block_id=block_id, field="keypoint"))
                elif bbox and not (bx1 <= kp[0] <= bx2 and by1 <= kp[1] <= by2):
                    failures.append(_failure("keypoint is outside block bbox", image=name, block_id=block_id, field="keypoint"))
                rn = block.get("reading_no")
                if not _is_int(rn) or rn <= 0:
                    failures.append(_failure("reading_no must be positive integer", image=name, block_id=block_id, field="reading_no"))
                else:
                    reading_numbers.append(rn)
                block_index[(name, block_id)] = {
                    "image": name,
                    "block_id": block_id,
                    "column_id": col_id,
                    "bbox": bbox,
                    "keypoint": kp,
                    "reading_no": rn if _is_int(rn) else None,
                    "char": block.get("char"),
                }
            if reading_numbers:
                ordered = sorted(reading_numbers)
                if len(ordered) != len(set(ordered)) or ordered != list(range(ordered[0], ordered[0] + len(ordered))):
                    failures.append(_failure(
                        "same-column reading numbers are not continuous",
                        image=name,
                        column=col_id,
                        field="reading_no",
                        reading_numbers=reading_numbers,
                    ))

        x_intervals.sort()
        for left, right in zip(x_intervals, x_intervals[1:]):
            if left[1] > right[0]:
                failures.append(_failure(
                    "column x intervals overlap",
                    image=name,
                    columns=[left[2], right[2]],
                    field="column_bbox",
                ))

    for name in images:
        if name not in truth_names:
            failures.append(_failure("PNG has no layout truth entry", image=name, field="layout_truth"))
    return block_index, failures


def validate_keypoints(
    keypoint_data: dict[str, Any] | None,
    block_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    if keypoint_data is None:
        return [], canonical_hash(None), failures
    raw_subs = keypoint_data.get("submissions") if isinstance(keypoint_data, dict) else None
    if not isinstance(raw_subs, list):
        failures.append(_failure("keypoint_versions.json must contain submissions array", field="keypoint_versions"))
        return [], canonical_hash(None), failures
    for i, sub in enumerate(raw_subs):
        ctx = {"submission_index": i, "image": sub.get("image") if isinstance(sub, dict) else None,
               "block_id": sub.get("block_id") if isinstance(sub, dict) else None}
        if not isinstance(sub, dict):
            failures.append(_failure("keypoint submission must be object", **ctx))
            continue
        required = ["image", "block_id", "version_no", "version_name", "annotated_ms", "annotator", "x", "y"]
        missing = [key for key in required if key not in sub]
        if missing:
            failures.append(_failure(f"keypoint submission missing {missing}", field="keypoint", **ctx))
            continue
        if not _is_int(sub["version_no"]) or not _is_int(sub["annotated_ms"]):
            failures.append(_failure("version_no and annotated_ms must be integers", field="keypoint_version", **ctx))
        if not isinstance(sub["version_name"], str) or not isinstance(sub["annotator"], str):
            failures.append(_failure("version_name and annotator must be strings", field="keypoint_version", **ctx))
        if not _is_int(sub["x"]) or not _is_int(sub["y"]):
            failures.append(_failure("keypoint coordinates must be integers", field="keypoint", **ctx))
            continue
        block = block_index.get((sub["image"], sub["block_id"]))
        if block is None or block.get("bbox") is None:
            failures.append(_failure("keypoint references unknown block", field="block_id", **ctx))
            continue
        x1, y1, x2, y2 = block["bbox"]
        if not (x1 <= sub["x"] <= x2 and y1 <= sub["y"] <= y2):
            failures.append(_failure("keypoint coordinate is outside block bbox", field="keypoint", **ctx))
        submissions.append(dict(sub))
    fingerprint = canonical_hash(submissions)
    return submissions, fingerprint, failures


def load_dataset(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    input_dir = root / "input" if (root / "input").exists() else root
    failures: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    png_paths = sorted(input_dir.glob("*.png"), key=lambda p: p.name)
    for path in png_paths:
        image, failure = inspect_png(path, config)
        if failure:
            failures.append(failure)
        else:
            images[path.name] = image

    truth = None
    truth_path = input_dir / "layout_truth.json"
    if truth_path.exists():
        try:
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(_failure(f"layout_truth.json invalid JSON: {exc}", field="layout_truth"))
    else:
        failures.append(_failure("layout_truth.json is missing", field="layout_truth"))

    block_index, truth_failures = validate_truth(truth or {}, images, config)
    failures.extend(truth_failures)

    keypoint_data = None
    kp_path = input_dir / "keypoint_versions.json"
    if kp_path.exists():
        try:
            keypoint_data = json.loads(kp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(_failure(f"keypoint_versions.json invalid JSON: {exc}", field="keypoint_versions"))
    else:
        failures.append(_failure("keypoint_versions.json is missing", field="keypoint_versions"))
    submissions, kp_fingerprint, kp_failures = validate_keypoints(keypoint_data, block_index)
    failures.extend(kp_failures)

    return {
        "input_dir": str(input_dir),
        "images": images,
        "png_paths": png_paths,
        "truth": truth,
        "block_index": block_index,
        "keypoint_submissions": submissions,
        "keypoint_fingerprint": kp_fingerprint,
        "failures": failures,
    }


def fingerprint_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
