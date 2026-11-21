from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .generator import write_candidate_files
from .unify import discover_series_images
from .util import atomic_write_json, atomic_write_text, quantize_int, read_json, stable_hash
from .v1 import V1_ROW_WIDTH
from .v2 import parse_v2


def _copy_dataset(source: Path, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"injection output exists: {destination}")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _find_image(root: Path, series_name: str, frame_index: int) -> Path:
    images = discover_series_images(root / "images")
    try:
        return images[series_name][frame_index]
    except KeyError as exc:
        raise BusinessError(
            "E_IMAGE_INVALID",
            f"cannot inject fault: image not found for {series_name}:{frame_index}",
            "inject",
            series_name=series_name,
            frame_index=frame_index,
        ) from exc


def _rewrite_candidate_set(
    root: Path,
    *,
    series_name: str,
    frame_index: int,
    transform: Any,
) -> int:
    parsed = parse_v2(root / "candidates_v2.json")
    candidates = []
    changed = 0
    for key, items in parsed.frames.items():
        for item in items:
            if key == (series_name, frame_index):
                updated = transform(item)
                if updated is not None:
                    candidates.append(updated)
                    changed += 1
                else:
                    candidates.append(item)
            else:
                candidates.append(item)
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
    if changed == 0:
        raise BusinessError(
            "E_INJECTION_TARGET_MISSING",
            f"no candidate found for injection target {series_name}:{frame_index}",
            "inject",
            series_name=series_name,
            frame_index=frame_index,
        )
    write_candidate_files(root, candidates)
    return changed


def _hit_v1_field_missing(root: Path, series_name: str, frame_index: int) -> None:
    path = root / "candidates_v1.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = False
    result: list[str] = []
    for line in lines:
        if not line or line.startswith("#"):
            result.append(line)
            continue
        same_series = line[:16].strip() == series_name
        same_frame = line[16:24].strip() == f"{frame_index:08d}"
        if same_series and same_frame and not changed:
            chars = list(line)
            for offset in range(24, 30):
                chars[offset] = " "
            for offset in range(67, 75):
                chars[offset] = " "
            mask = "C-------"
            for index, char in enumerate(mask):
                chars[67 + index] = char
            line = "".join(chars)
            assert len(line) == V1_ROW_WIDTH
            changed = True
        result.append(line)
    if not changed:
        raise BusinessError(
            "E_INJECTION_TARGET_MISSING",
            f"no v1 row for injection target {series_name}:{frame_index}",
            "inject",
            series_name=series_name,
            frame_index=frame_index,
        )
    atomic_write_text(path, "\n".join(result) + "\n")


def apply_injections(
    data_dir: str | Path,
    output_dir: str | Path,
    injections_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(data_dir).resolve()
    destination = Path(output_dir).resolve()
    if source == destination:
        raise ValueError("injection output must differ from source dataset")
    _copy_dataset(source, destination, force)
    payload = read_json(injections_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("injections"), list):
        raise BusinessError(
            "E_CONFIG_INVALID", "injection file must contain injections list", "inject"
        )
    hits: list[dict[str, Any]] = []
    language_level_injections = sorted(
        payload["injections"],
        key=lambda item: (
            1 if item.get("type") == "v1_field_missing" else 0,
            str(item.get("id", "")),
        ),
    )
    for injection in language_level_injections:
        if not isinstance(injection, dict):
            continue
        injection_type = injection.get("type")
        series_name = str(injection.get("series_name", ""))
        frame_index = int(injection.get("frame_index", -1))
        detail: dict[str, Any] = {}
        if injection_type == "truncated_png":
            image_path = _find_image(destination, series_name, frame_index)
            original = image_path.read_bytes()
            truncated = original[: max(8, len(original) // 3)]
            image_path.write_bytes(truncated)
            detail = {
                "image": image_path.relative_to(destination).as_posix(),
                "original_bytes": len(original),
                "injected_bytes": len(truncated),
            }
        elif injection_type == "v1_field_missing":
            _hit_v1_field_missing(destination, series_name, frame_index)
            detail = {"file": "candidates_v1.txt", "missing_mask": "C-------"}
        elif injection_type == "coordinate_out_of_bounds":
            def transform(item: Any) -> Any:
                if item.class_code == 1:
                    return type(item)(
                        **{
                            **item.__dict__,
                            "x2": 260.0,
                            "y2": 220.0,
                            "precision_label": "injected_out_of_bounds",
                        }
                    )
                return item

            _rewrite_candidate_set(
                destination,
                series_name=series_name,
                frame_index=frame_index,
                transform=transform,
            )
            detail = {"box": [item for item in (0, 0, 260, 220)]}
        elif injection_type == "non_finite_weight":
            config_path = destination / "detection_params.json"
            config = read_json(config_path)
            override = (
                config.setdefault("frame_overrides", {})
                .setdefault(series_name, {})
                .setdefault(str(frame_index), {})
            )
            override["fusion_weights"] = {"color": math.nan, "edge": 0.42}
            atomic_write_text(
                config_path,
                json.dumps(config, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            )
            detail = {"fusion_weights": {"color": "NaN", "edge": 0.42}}
        elif injection_type == "class_code_not_in_map":
            def transform(item: Any) -> Any:
                return type(item)(
                    **{
                        **item.__dict__,
                        "class_code": 99,
                        "class_name": "ghost",
                        "precision_label": "injected_unknown_class",
                    }
                )

            _rewrite_candidate_set(
                destination,
                series_name=series_name,
                frame_index=frame_index,
                transform=transform,
            )
            detail = {"class_code": 99, "class_name": "ghost"}
        elif injection_type in {
            "matching_forced_error",
            "serialization_forced_error",
        }:
            detail = {"runtime_only": True}
        else:
            raise BusinessError(
                "E_CONFIG_INVALID",
                f"unsupported injection type: {injection_type}",
                "inject",
                series_name=series_name,
                frame_index=frame_index,
            )
        hits.append({"id": injection.get("id"), "type": injection_type, "applied": True, **detail})
    active = dict(payload)
    active["applied"] = True
    active["applied_at_data_fingerprint"] = stable_hash(
        {series: sorted(frames) for series, frames in discover_series_images(destination / "images").items()}
    )
    active["application_log"] = hits
    atomic_write_json(destination / "active_injections.json", active)
    return {
        "output_dir": str(destination),
        "injection_count": len(hits),
        "applications": hits,
        "active_injections_path": str(destination / "active_injections.json"),
    }
