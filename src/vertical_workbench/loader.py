from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codes import E_SAMPLE_INVALID
from .config import WorkbenchConfig
from .hashing import fingerprint, sha256_bytes
from .pngcodec import PngDecodeError, decode_png


def failure_item(target: str, reasons: list[str], injection_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": E_SAMPLE_INVALID, "target": target, "reasons": reasons}
    if injection_id:
        item["injection_id"] = injection_id
    return item


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_truth_payload(
    payload: Any, width: int, height: int, require_bbox: bool = True
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["truth document must be a JSON object"]
    columns = payload.get("columns")
    points = payload.get("keypoints")
    if not isinstance(columns, list) or not columns:
        errors.append("columns must be a non-empty list")
        columns = []
    if not isinstance(points, list) or not points:
        errors.append("keypoints must be a non-empty list")
        points = []

    intervals: list[tuple[int, int, int]] = []
    ranks: set[int] = set()
    for index, column in enumerate(columns):
        if not isinstance(column, dict):
            errors.append(f"columns[{index}] must be an object")
            continue
        rank = column.get("column_rank")
        x_min = column.get("x_min")
        x_max = column.get("x_max")
        if not all(_is_int(v) for v in (rank, x_min, x_max)):
            errors.append(f"columns[{index}] rank/x_min/x_max must be integers")
            continue
        assert _is_int(rank) and _is_int(x_min) and _is_int(x_max)
        ranks.add(rank)
        if rank < 0:
            errors.append(f"columns[{index}] has negative rank")
        if x_min >= x_max:
            errors.append(f"column {rank} has empty/inverted x interval")
        elif x_min < 0 or x_max > width:
            errors.append(f"column {rank} x interval is outside image")
        else:
            intervals.append((x_min, x_max, rank))

    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            errors.append(f"column intervals overlap: rank {left[2]} and {right[2]}")
    if ranks and ranks != set(range(len(columns))):
        errors.append("column_rank values must be contiguous starting at 0")

    orders: list[int] = []
    ids: set[str] = set()
    for index, point in enumerate(points):
        where = f"keypoints[{index}]"
        if not isinstance(point, dict):
            errors.append(f"{where} must be an object")
            continue
        point_id = point.get("keypoint_id")
        if not isinstance(point_id, str) or not point_id:
            errors.append(f"{where}.keypoint_id must be a non-empty string")
        elif point_id in ids:
            errors.append(f"duplicate keypoint_id {point_id}")
        else:
            ids.add(point_id)

        x = point.get("x")
        y = point.get("y")
        order = point.get("reading_order")
        rank = point.get("column_rank")
        if not _is_int(x) or not _is_int(y):
            errors.append(f"{where} x/y must be integers")
        else:
            assert _is_int(x) and _is_int(y)
            if not (0 <= x < width and 0 <= y < height):
                errors.append(f"{where} x/y is outside the fixed image")
        if not _is_int(order) or order <= 0:
            errors.append(f"{where}.reading_order must be a positive integer")
        else:
            orders.append(order)
        if not _is_int(rank) or rank not in ranks:
            errors.append(f"{where}.column_rank is missing from columns")

        if require_bbox or "bbox" in point:
            bbox = point.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_int(v) for v in bbox):
                if require_bbox or "bbox" in point:
                    errors.append(f"{where}.bbox must be four integers")
            else:
                x0, y0, x1, y1 = bbox
                if x0 >= x1 or y0 >= y1 or x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    errors.append(f"{where}.bbox is invalid or outside image")

    if orders:
        expected = list(range(1, len(orders) + 1))
        if sorted(orders) != expected:
            errors.append("reading_order must be continuous 1..N without duplicates or gaps")
    return errors


def validate_annotation_document(doc: Any, width: int, height: int) -> list[str]:
    if not isinstance(doc, dict):
        return ["annotation document must be an object"]
    points = doc.get("keypoints")
    if not isinstance(points, list) or not points:
        return ["annotation keypoints must be a non-empty list"]
    wrapped = {"columns": [{"column_rank": i, "x_min": 0, "x_max": width} for i in range(1)], "keypoints": []}
    # Annotation points may use any column rank; construct broad non-overlapping columns.
    max_rank = max((p.get("column_rank", 0) for p in points if isinstance(p, dict)), default=0)
    wrapped["columns"] = [
        {"column_rank": rank, "x_min": 0, "x_max": width}
        for rank in range(max(1, int(max_rank) + 1) if _is_int(max_rank) else 1)
    ]
    wrapped["keypoints"] = points
    errors = validate_truth_payload(wrapped, width, height, require_bbox=False)
    return [e for e in errors if not e.startswith("column intervals overlap")]


@dataclass
class LoadedWorkspace:
    root: Path
    config: WorkbenchConfig
    samples: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    annotation_bundle: dict[str, Any] = field(default_factory=dict)
    annotation_failure: dict[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "config_hash": self.config.config_hash,
            "sample_ids": [s["image_id"] for s in self.samples],
            "failures": self.failures,
            "annotation_version": self.annotation_bundle.get("version_id"),
            "annotation_digest": self.annotation_bundle.get("content_digest"),
        }


def load_config_file(path: str | Path) -> tuple[WorkbenchConfig | None, dict[str, Any] | None]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = WorkbenchConfig.from_dict(raw)
        errors = config.validate()
        if errors:
            return None, failure_item(str(path), errors)
        return config, None
    except FileNotFoundError:
        return None, failure_item(str(path), ["config file not found"])
    except json.JSONDecodeError as exc:
        return None, failure_item(str(path), [f"invalid config JSON: {exc}"])


def load_workspace(
    root: str | Path,
    config: WorkbenchConfig | None = None,
    injection_ids: dict[str, str] | None = None,
) -> LoadedWorkspace:
    root = Path(root)
    injection_ids = injection_ids or {}
    config_path = root / "configs" / "workbench.json"
    config_failure: dict[str, Any] | None = None
    if config is None and config_path.exists():
        config, config_failure = load_config_file(config_path)
    if config is None:
        config = WorkbenchConfig()
        errors = config.validate()
        if errors:
            config_failure = failure_item(str(config_path), errors)

    loaded = LoadedWorkspace(root=root, config=config)
    if config_failure:
        loaded.failures.append(config_failure)
        return loaded

    sample_dir = root / "samples"
    truth_dir = root / "truth"
    if not sample_dir.exists():
        loaded.failures.append(failure_item("samples", ["sample directory not found"]))
        return loaded

    png_paths = sorted(sample_dir.glob("*.png"), key=lambda p: p.name)
    for png_path in png_paths:
        rel_png = png_path.relative_to(root).as_posix() if png_path.is_relative_to(root) else png_path.name
        injection_id = injection_ids.get(png_path.name)
        try:
            data = png_path.read_bytes()
            image = decode_png(data)
            if image.width != config.width or image.height != config.height:
                raise PngDecodeError(
                    f"fixed resolution must be {config.width}x{config.height}, "
                    f"got {image.width}x{image.height}"
                )
            if image.mode != "L":
                raise PngDecodeError(f"pixel mode must be 8-bit grayscale L, got {image.mode}")
        except (OSError, PngDecodeError) as exc:
            loaded.failures.append(failure_item(rel_png, [str(exc)], injection_id))
            continue

        truth_path = truth_dir / f"{png_path.stem}.json"
        rel_truth = truth_path.relative_to(root).as_posix() if truth_path.is_relative_to(root) else truth_path.name
        try:
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            loaded.failures.append(failure_item(rel_truth, [f"cannot load truth: {exc}"], injection_ids.get(truth_path.name)))
            continue

        errors = validate_truth_payload(truth, config.width, config.height)
        if errors:
            loaded.failures.append(failure_item(rel_truth, errors, injection_ids.get(truth_path.name)))
            continue

        expected_digest = truth.get("content_digest")
        digest_body = {k: v for k, v in truth.items() if k != "content_digest"}
        actual_digest = fingerprint(digest_body)
        if expected_digest != actual_digest:
            loaded.failures.append(
                failure_item(rel_truth, ["truth content_digest does not match payload"], injection_ids.get(truth_path.name))
            )
            continue

        loaded.samples.append(
            {
                "image_id": truth.get("image_id", png_path.stem),
                "png_path": png_path,
                "truth_path": truth_path,
                "png_rel": rel_png,
                "truth_rel": rel_truth,
                "png_bytes": data,
                "pixels": image.pixels,
                "image_fingerprint": sha256_bytes(image.pixels.tobytes()),
                "png_fingerprint": sha256_bytes(data),
                "truth": truth,
            }
        )

    annotation_path = root / "annotations" / "keypoint_versions.json"
    if annotation_path.exists():
        try:
            bundle = json.loads(annotation_path.read_text(encoding="utf-8"))
            docs = bundle.get("documents", {}) if isinstance(bundle, dict) else {}
            for image_id in sorted(docs):
                doc = docs[image_id]
                errors = validate_annotation_document(doc, config.width, config.height)
                if errors:
                    inj = injection_ids.get(f"annotation:{image_id}")
                    loaded.failures.append(
                        failure_item(f"annotations/keypoint_versions.json#{image_id}", errors, inj)
                    )
            if isinstance(bundle, dict):
                documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}
                body_digest = fingerprint(
                    {
                        image_id: doc.get("content_digest") if isinstance(doc, dict) else None
                        for image_id, doc in sorted(documents.items())
                    }
                )
                if bundle.get("content_digest") != body_digest:
                    loaded.annotation_failure = failure_item(
                        "annotations/keypoint_versions.json",
                        ["bundle content_digest does not match documents"],
                        injection_ids.get("annotations/keypoint_versions.json"),
                    )
                    loaded.failures.append(loaded.annotation_failure)
                loaded.annotation_bundle = bundle
        except (OSError, json.JSONDecodeError) as exc:
            loaded.annotation_failure = failure_item(
                "annotations/keypoint_versions.json", [f"invalid annotation JSON: {exc}"]
            )
            loaded.failures.append(loaded.annotation_failure)

    loaded.samples.sort(key=lambda s: s["image_id"])
    return loaded
