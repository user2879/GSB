from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .generator import IMAGE_HEIGHT, IMAGE_WIDTH, write_candidate_files
from .util import atomic_write_json, quantize_int, read_json
from .v2 import parse_v2


def copy_dataset(source: str | Path, destination: str | Path, *, force: bool = False) -> None:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        return
    if destination_path.exists():
        if not force:
            raise FileExistsError(str(destination_path))
        shutil.rmtree(destination_path)
    shutil.copytree(source_path, destination_path)


def switch_version(data_dir: str | Path, version: str) -> None:
    atomic_write_json(Path(data_dir) / "exchange_config.json", {"active_version": version})


def set_iou_threshold(data_dir: str | Path, value: float) -> None:
    path = Path(data_dir) / "match_config.json"
    config = read_json(path)
    config["iou_threshold"] = float(value)
    atomic_write_json(path, config)


def set_pool_capacity(data_dir: str | Path, value: int) -> None:
    path = Path(data_dir) / "detection_params.json"
    config = read_json(path)
    config["candidate_pool_capacity"] = int(value)
    atomic_write_json(path, config)


def replace_frame_candidates(data_dir: str | Path, series_name: str, frame_index: int) -> int:
    root = Path(data_dir)
    parsed = parse_v2(root / "candidates_v2.json")
    candidates = []
    changed = 0
    for key, items in sorted(parsed.frames.items()):
        for index, item in enumerate(sorted(items, key=lambda candidate: -candidate.score)):
            if key == (series_name, frame_index) and index == 0:
                shifted_x = min(IMAGE_WIDTH - 2, quantize_int(item.x1) + 5)
                shifted_y = min(IMAGE_HEIGHT - 2, quantize_int(item.y1) + 3)
                width = quantize_int(item.x2) - quantize_int(item.x1)
                height = quantize_int(item.y2) - quantize_int(item.y1)
                box = [
                    float(shifted_x),
                    float(shifted_y),
                    float(min(IMAGE_WIDTH, shifted_x + width)),
                    float(min(IMAGE_HEIGHT, shifted_y + height)),
                ]
                candidates.append(
                    type(item)(
                        **{
                            **item.__dict__,
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                            "score": min(1.0, item.score + 0.017),
                            "extra": {**item.extra, "edited": "replace_frame_candidates"},
                        }
                    )
                )
                changed += 1
            else:
                candidates.append(item)
    if changed == 0:
        raise ValueError(f"candidate target not found: {series_name}:{frame_index}")
    candidates.sort(
        key=lambda item: (
            item.series_name,
            item.frame_index,
            -int(round(item.score * 1000)),
            quantize_int(item.x1),
            quantize_int(item.y1),
            quantize_int(item.x2),
            quantize_int(item.y2),
            item.class_code,
            item.source_ordinal,
        )
    )
    write_candidate_files(root, candidates)
    return changed


def replace_image(data_dir: str | Path, series_name: str, frame_index: int) -> str:
    root = Path(data_dir)
    path = root / "images" / series_name / f"frame_{frame_index:04d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        modified = image.convert("RGB")
    draw = ImageDraw.Draw(modified)
    x = (frame_index * 23 + 17) % (IMAGE_WIDTH - 36)
    y = (frame_index * 17 + 11) % (IMAGE_HEIGHT - 28)
    draw.rectangle((x, y, x + 34, y + 25), fill=(245, 210, 64), outline=(255, 255, 255), width=2)
    draw.line((x, y, x + 34, y + 25), fill=(32, 32, 32), width=3)
    modified.save(path, format="PNG", optimize=False)
    return str(path)


def apply_operation(data_dir: str | Path, operation: dict[str, Any]) -> dict[str, Any]:
    operation_type = operation["type"]
    if operation_type == "switch_version":
        switch_version(data_dir, operation["active_version"])
        return {"changed": {"active_version": operation["active_version"]}}
    if operation_type == "replace_frame_candidates":
        count = replace_frame_candidates(
            data_dir, operation["series_name"], int(operation["frame_index"])
        )
        return {"changed_candidate_count": count}
    if operation_type == "replace_image":
        path = replace_image(data_dir, operation["series_name"], int(operation["frame_index"]))
        return {"image": path}
    if operation_type == "set_iou_threshold":
        set_iou_threshold(data_dir, float(operation["value"]))
        return {"iou_threshold": float(operation["value"])}
    if operation_type == "set_pool_capacity":
        set_pool_capacity(data_dir, int(operation["value"]))
        return {"candidate_pool_capacity": int(operation["value"])}
    raise ValueError(f"unknown edit operation: {operation_type}")
