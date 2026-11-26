from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .codes import E_INPUT_INVALID
from .utils import sha256_file


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class ValidatedDataset:
    root: Path
    views: list[dict[str, Any]] = field(default_factory=list)
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    image_sha256: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def valid_view_ids(self) -> list[str]:
        return [view["view_id"] for view in self.views]


def _error(ref: str, reason: str) -> dict[str, str]:
    return {"code": E_INPUT_INVALID, "ref": ref, "reason": reason}


def validate_png(path: Path, width: int, height: int, mode: str) -> dict[str, Any]:
    ref = str(path)
    if not path.is_file():
        return {"ok": False, "error": _error(ref, "image file does not exist")}
    try:
        with open(path, "rb") as handle:
            header = handle.read(8)
        if header != PNG_SIGNATURE:
            return {"ok": False, "error": _error(ref, "invalid PNG container header")}
        with Image.open(path) as image:
            if image.format != "PNG":
                return {"ok": False, "error": _error(ref, f"container format is {image.format}, expected PNG")}
            if image.size != (int(width), int(height)):
                return {
                    "ok": False,
                    "error": _error(ref, f"size {image.size} is incompatible with {(width, height)}"),
                }
            if image.mode != mode:
                return {"ok": False, "error": _error(ref, f"pixel mode {image.mode} != expected {mode}")}
            image.load()
        return {"ok": True, "sha256": sha256_file(path)}
    except Exception as exc:  # decoding failures must become input errors
        return {"ok": False, "error": _error(ref, f"undecodable PNG: {type(exc).__name__}: {exc}")}


def determinant3(matrix: list[list[float | int]]) -> float:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _as_3x3(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return None
        converted: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            converted.append(float(item))
        result.append(converted)
    return result


def validate_camera(camera: Any, ref: str) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(camera, dict):
        return [_error(ref, "camera must be an object")]

    k = _as_3x3(camera.get("K"))
    if k is None:
        errors.append(_error(f"{ref}.K", "K must be a 3x3 numeric matrix"))
    elif determinant3(k) == 0.0:
        errors.append(_error(f"{ref}.K", "intrinsic matrix is singular"))

    r = _as_3x3(camera.get("R"))
    if r is None:
        errors.append(_error(f"{ref}.R", "R must be a 3x3 numeric matrix"))
    else:
        rt_r = [[sum(r[k][i] * r[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                if abs(rt_r[i][j] - expected) > 1e-8:
                    errors.append(_error(f"{ref}.R", "rotation matrix is not orthogonal"))
                    break
            if errors and errors[-1]["ref"] == f"{ref}.R":
                break
        det_r = determinant3(r)
        if det_r <= 0 or abs(det_r - 1.0) > 1e-8:
            errors.append(_error(f"{ref}.R", "rotation determinant must be positive and equal to +1"))

    t = camera.get("t")
    if not isinstance(t, list) or len(t) != 3 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in t):
        errors.append(_error(f"{ref}.t", "translation must be a 3-vector of numbers"))
    return errors


def _resolve_cameras(raw: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if "cameras" in raw and isinstance(raw["cameras"], dict):
        return raw["cameras"], errors
    camera_ref = raw.get("camera_file") or raw.get("cameras_file")
    if camera_ref:
        path = (root / str(camera_ref)).resolve()
        try:
            from .utils import read_json

            data = read_json(path)
            if isinstance(data, dict) and "cameras" in data:
                data = data["cameras"]
            if not isinstance(data, dict):
                raise ValueError("camera file root must be an object or contain cameras object")
            return data, errors
        except Exception as exc:
            return {}, [_error(str(path), f"cannot load cameras: {exc}")]
    return {}, [_error(str(root), "dataset has no cameras mapping or camera_file")]


def validate_dataset(raw: dict[str, Any], root: str | Path, config: dict[str, Any]) -> ValidatedDataset:
    root_path = Path(root)
    result = ValidatedDataset(root=root_path)
    cameras, camera_errors = _resolve_cameras(raw, root_path)
    result.errors.extend(camera_errors)
    result.cameras = cameras

    raw_views = raw.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        result.errors.append(_error("dataset.views", "views must be a non-empty list"))
        return result

    seen_ids: set[str] = set()
    seen_sequences: set[int] = set()
    valid_view_ids: set[str] = set()

    for index, view in enumerate(raw_views):
        ref = f"dataset.views[{index}]"
        if not isinstance(view, dict):
            result.errors.append(_error(ref, "view must be an object"))
            continue
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            result.errors.append(_error(ref, "view_id must be a non-empty string"))
            continue
        ref = f"view:{view_id}"
        if view_id in seen_ids:
            result.errors.append(_error(ref, "duplicate view_id"))
            continue
        seen_ids.add(view_id)

        sequence = view.get("sequence", index)
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            result.errors.append(_error(ref, "sequence must be an integer"))
            continue
        if sequence in seen_sequences:
            result.errors.append(_error(ref, f"duplicate view sequence {sequence}"))
            continue
        seen_sequences.add(sequence)

        image_ref = view.get("image")
        if not isinstance(image_ref, str):
            result.errors.append(_error(ref, "image path is required"))
            continue
        image_path = (root_path / image_ref).resolve()
        image_info = validate_png(image_path, int(config["width"]), int(config["height"]), str(config["mode"]))
        if not image_info["ok"]:
            result.errors.append(image_info["error"])
            continue

        camera_id = view.get("camera_id", view_id)
        camera = cameras.get(camera_id)
        camera_errors = validate_camera(camera, f"camera:{camera_id}") if camera is not None else [
            _error(f"camera:{camera_id}", "camera reference is missing")
        ]
        if camera_errors:
            result.errors.extend(camera_errors)
            continue

        normalized = dict(view)
        normalized["sequence"] = sequence
        normalized["camera_id"] = camera_id
        normalized["image_abs"] = str(image_path)
        result.views.append(normalized)
        result.image_sha256[view_id] = image_info["sha256"]
        valid_view_ids.add(view_id)

    result.views.sort(key=lambda item: (int(item["sequence"]), str(item["view_id"])))

    raw_pairs = raw.get("pairs", [])
    if not isinstance(raw_pairs, list):
        result.errors.append(_error("dataset.pairs", "pairs must be a list"))
        raw_pairs = []
    sequence_by_id = {view["view_id"]: int(view["sequence"]) for view in result.views}
    seen_pairs: set[tuple[str, str]] = set()
    for item in raw_pairs:
        if not isinstance(item, list) or len(item) != 2 or not all(isinstance(x, str) for x in item):
            result.errors.append(_error("dataset.pairs", "pair must be [left_view_id, right_view_id]"))
            continue
        left, right = item
        if left not in valid_view_ids or right not in valid_view_ids:
            result.errors.append(
                _error(f"pair:{left},{right}", "unknown or invalid view reference; pair skipped")
            )
            continue
        if left == right:
            result.errors.append(_error(f"pair:{left},{right}", "self-pair is not allowed"))
            continue
        if sequence_by_id[left] > sequence_by_id[right]:
            left, right = right, left
        pair = (left, right)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        result.pairs.append(pair)

    result.pairs.sort(key=lambda pair: (sequence_by_id[pair[0]], sequence_by_id[pair[1]], pair[0], pair[1]))
    return result
