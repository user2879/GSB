from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import read_json, sha256_bytes
from .composite import composite_scene
from .diagnostics import classification_key, classification_keys, classification_records
from .errors import ErrorCode, sorted_issues
from .flatten import flatten_scene
from .inject import apply_injections, attach_injection_ids, default_render_params
from .scene import (
    normalized_scene_for_render,
    scene_payload_hash,
    segment_fragment,
    validate_scene,
    validate_tolerance,
)


def _used_tolerance(requested: Any) -> float:
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return 0.25
    if not np.isfinite(value) or value <= 0.0:
        return float(abs(value)) if np.isfinite(value) and value != 0.0 else 0.25
    return value


def _mismatch(
    violation_type: str,
    layer_id: str | None,
    x: int | None,
    y: int | None,
    path_id: str | None = None,
    channel: int | None = None,
    expected: Any = None,
    actual: Any = None,
    message: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "violation_type": violation_type,
        "layer_id": layer_id,
        "path_id": path_id,
        "x": x,
        "y": y,
    }
    if channel is not None:
        payload["channel"] = channel
    if expected is not None:
        payload["expected"] = expected
    if actual is not None:
        payload["actual"] = actual
    if message:
        payload["message"] = message
    return payload


def _first_byte_difference(expected: bytes, actual: bytes, width: int, height: int) -> tuple[int, int, int] | None:
    expected_array = np.frombuffer(expected, dtype=np.uint8)
    actual_array = np.frombuffer(actual, dtype=np.uint8)
    limit = min(expected_array.size, actual_array.size)
    if expected_array.size != actual_array.size:
        limit = min(limit, expected_array.size, actual_array.size)
    if limit == 0:
        return (0, 0, 0) if expected_array.size != actual_array.size else None
    difference = expected_array[:limit] != actual_array[:limit]
    indices = np.flatnonzero(difference)
    if indices.size == 0:
        if expected_array.size != actual_array.size:
            return (0, 0, 0)
        return None
    byte_index = int(indices[0])
    pixel_index, channel = divmod(byte_index, 4)
    y, x = divmod(pixel_index, width)
    return int(x), int(y), int(channel)


def _classification_mismatch_pixel(record: dict[str, Any], layers: list[dict[str, Any]]) -> tuple[str | None, str | None, int | None, int | None]:
    location = record.get("location", {})
    layer_id = location.get("layer_id")
    path_id = location.get("path_id")
    layer = next((item for item in layers if item["layer_id"] == layer_id), None)
    if layer is None and path_id is not None:
        layer = next((item for item in layers if item.get("path_id") == path_id), None)
    if layer is None:
        return layer_id, path_id, None, None
    coverage = np.maximum(layer["fill"][layer["fill"]["selected_rule"]], layer["stroke"]["coverage"])
    indices = np.flatnonzero(coverage.reshape(-1))
    if indices.size == 0:
        return layer["layer_id"], layer.get("path_id"), 0, 0
    y, x = divmod(int(indices[0]), coverage.shape[1])
    return layer["layer_id"], layer.get("path_id"), int(x), int(y)


def verify_case_dir(case_dir: Path) -> dict[str, Any]:
    case_dir = Path(case_dir)
    checks: dict[str, Any] = {
        "scene_json_read": False,
        "injection_json_read": False,
        "layer_manifest_read": False,
        "scene_hash_valid": False,
        "injection_classification_match": False,
        "layer_headers_match": False,
        "layer_hashes_match": False,
        "layer_pixels_match": False,
        "fill_rule_coverage_match": False,
        "final_buffer_match": False,
    }
    errors: list[dict[str, Any]] = []
    try:
        scene = read_json(case_dir / "scene.json")
        checks["scene_json_read"] = True
    except Exception as exc:
        return {
            "schema": "vgsynth.verification/v1",
            "status": "mismatch",
            "code": str(ErrorCode.VERIFIER_MISMATCH),
            "checks": checks,
            "first_mismatch": _mismatch("scene_json", None, None, None, message=str(exc)),
            "errors": [{"code": str(ErrorCode.VERIFIER_MISMATCH), "message": str(exc)}],
        }
    try:
        injection = read_json(case_dir / "injection.json")
        checks["injection_json_read"] = True
    except Exception as exc:
        return {
            "schema": "vgsynth.verification/v1",
            "status": "mismatch",
            "code": str(ErrorCode.VERIFIER_MISMATCH),
            "checks": checks,
            "first_mismatch": _mismatch("injection_json", None, None, None, message=str(exc)),
            "errors": [{"code": str(ErrorCode.VERIFIER_MISMATCH), "message": str(exc)}],
        }
    try:
        layer_manifest = read_json(case_dir / "layers.json")
        checks["layer_manifest_read"] = True
    except Exception as exc:
        return {
            "schema": "vgsynth.verification/v1",
            "status": "mismatch",
            "code": str(ErrorCode.VERIFIER_MISMATCH),
            "checks": checks,
            "first_mismatch": _mismatch("layer_manifest", None, None, None, message=str(exc)),
            "errors": [{"code": str(ErrorCode.VERIFIER_MISMATCH), "message": str(exc)}],
        }

    checks["scene_hash_valid"] = scene.get("content_hash") == scene_payload_hash(scene)
    render_params = copy.deepcopy(injection.get("render_params", default_render_params()))
    injection_result = apply_injections(scene, injection, render_params)
    tolerance = _used_tolerance(injection_result.render_params.get("tolerance", 0.25))
    expected_issues = injection_result.issues + validate_scene(injection_result.scene, check_content_hash=False) + validate_tolerance(
        injection_result.render_params.get("tolerance", 0.25)
    )
    expected_issues = sorted_issues(expected_issues)
    attach_injection_ids(expected_issues, injection_result.applications)
    expected_classification = classification_records(expected_issues)
    observed_classification = layer_manifest.get("injection_classification", [])
    expected_keys = sorted(classification_keys(expected_classification))
    observed_keys = sorted(classification_keys(observed_classification))
    checks["injection_classification_match"] = expected_keys == observed_keys

    normalized = normalized_scene_for_render(injection_result.scene)
    flat = flatten_scene(normalized, tolerance)
    recomputed = composite_scene(normalized, flat)
    layers = recomputed["layers"]
    records = {record["layer_id"]: record for record in layer_manifest.get("layers", [])}
    first_mismatch: dict[str, Any] | None = None
    if not checks["scene_hash_valid"] and first_mismatch is None:
        first_mismatch = _mismatch("scene_hash", None, None, None, expected=scene.get("content_hash"), actual=scene_payload_hash(scene))
    if not checks["injection_classification_match"] and first_mismatch is None:
        expected_set = set(expected_keys)
        observed_set = set(observed_keys)
        differing_key = next(iter(sorted(expected_set.symmetric_difference(observed_set))), None)
        record = next((item for item in expected_classification + observed_classification if classification_key(item) == differing_key), None)
        if record is None and observed_classification:
            record = observed_classification[0]
        layer_id, path_id, x, y = _classification_mismatch_pixel(record or {}, layers)
        first_mismatch = _mismatch(
            "injection_classification",
            layer_id,
            x,
            y,
            path_id=path_id,
            expected=expected_classification,
            actual=observed_classification,
        )
        errors.append({"code": str(ErrorCode.VERIFIER_MISMATCH), "message": "注入异常分类与验证器复算不一致"})

    headers_ok = True
    hashes_ok = True
    pixels_ok = True
    fill_ok = True
    for layer in sorted(layers, key=lambda item: item["layer_id"]):
        layer_id = layer["layer_id"]
        record = records.get(layer_id)
        if record is None:
            headers_ok = hashes_ok = pixels_ok = fill_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch("missing_layer_manifest", layer_id, 0, 0, path_id=layer.get("path_id"))
            errors.append({"code": str(ErrorCode.LAYER_HASH), "message": "层清单缺少层", "layer_id": layer_id})
            continue
        header = layer["header"]
        selected = layer["fill"]["selected_rule"]
        header_pairs = (
            ("path_id", record.get("path_id")),
            ("declared_fill_rule", record.get("declared_fill_rule")),
            ("fill_rule_used", record.get("fill_rule_used")),
            ("fill_hash_nonzero", record.get("fill_hash_nonzero")),
            ("fill_hash_evenodd", record.get("fill_hash_evenodd")),
            ("fill_hash_selected", record.get("fill_hash_selected")),
            ("stroke_coverage_hash", record.get("stroke_coverage_hash")),
        )
        for key, actual in header_pairs:
            expected = header.get(key)
            if expected != actual:
                headers_ok = False
                if first_mismatch is None:
                    first_mismatch = _mismatch(
                        f"layer_header:{key}",
                        layer_id,
                        0,
                        0,
                        path_id=layer.get("path_id"),
                        expected=expected,
                        actual=actual,
                    )
                errors.append(
                    {
                        "code": str(ErrorCode.LAYER_HASH),
                        "message": f"层头字段 {key} 不一致",
                        "layer_id": layer_id,
                    }
                )
                break
        recomputed_hashes = {
            "nonzero": layer["fill"]["nonzero_info"]["content_hash"],
            "evenodd": layer["fill"]["evenodd_info"]["content_hash"],
            "selected": layer["fill"][selected + "_info"]["content_hash"],
            "stroke": layer["stroke"]["info"]["content_hash"],
        }
        if record.get("fill_rule_used") not in ("nonzero", "evenodd") or record.get("fill_rule_used") != selected:
            fill_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch(
                    "fill_rule",
                    layer_id,
                    0,
                    0,
                    path_id=layer.get("path_id"),
                    expected=selected,
                    actual=record.get("fill_rule_used"),
                )
            errors.append({"code": str(ErrorCode.FILL_RULE), "message": "填充规则与覆盖率不一致", "layer_id": layer_id})
        if record.get("fill_hash_nonzero") != recomputed_hashes["nonzero"] or record.get("fill_hash_evenodd") != recomputed_hashes["evenodd"]:
            fill_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch(
                    "fill_coverage_hash",
                    layer_id,
                    0,
                    0,
                    path_id=layer.get("path_id"),
                    expected=recomputed_hashes,
                    actual={
                        "nonzero": record.get("fill_hash_nonzero"),
                        "evenodd": record.get("fill_hash_evenodd"),
                    },
                )
            errors.append({"code": str(ErrorCode.FILL_RULE), "message": "填充覆盖率哈希不一致", "layer_id": layer_id})
        expected_layer_hash = layer["layer_hash"]
        actual_layer_hash = record.get("layer_hash")
        if record.get("pixel_hash") != layer["pixel_hash"]:
            hashes_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch(
                    "layer_pixel_hash",
                    layer_id,
                    0,
                    0,
                    path_id=layer.get("path_id"),
                    expected=layer["pixel_hash"],
                    actual=record.get("pixel_hash"),
                )
            errors.append({"code": str(ErrorCode.LAYER_HASH), "message": "层像素哈希清单不一致", "layer_id": layer_id})
        if expected_layer_hash != actual_layer_hash:
            hashes_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch(
                    "layer_hash",
                    layer_id,
                    0,
                    0,
                    path_id=layer.get("path_id"),
                    expected=expected_layer_hash,
                    actual=actual_layer_hash,
                )
            errors.append({"code": str(ErrorCode.LAYER_HASH), "message": "层内容哈希不一致", "layer_id": layer_id})
        layer_path = case_dir / record.get("bgra_file", f"layers/{layer_id}.bgra")
        expected_bytes = layer["pixels"].tobytes()
        try:
            actual_bytes = layer_path.read_bytes()
        except Exception:
            actual_bytes = b""
            pixels_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch("missing_layer_buffer", layer_id, 0, 0, path_id=layer.get("path_id"))
        difference = _first_byte_difference(expected_bytes, actual_bytes, layer["pixels"].shape[1], layer["pixels"].shape[0])
        if difference is not None:
            pixels_ok = False
            x, y, channel = difference
            if first_mismatch is None:
                first_mismatch = _mismatch(
                    "layer_pixel",
                    layer_id,
                    x,
                    y,
                    path_id=layer.get("path_id"),
                    channel=channel,
                    expected=int(layer["pixels"][y, x, channel]),
                    actual=int(actual_bytes[y * layer["pixels"].shape[1] * 4 + x * 4 + channel]) if len(actual_bytes) > y * layer["pixels"].shape[1] * 4 + x * 4 + channel else None,
                )
            errors.append(
                {
                    "code": str(ErrorCode.LAYER_HASH),
                    "message": "层像素与复算缓冲不一致",
                    "layer_id": layer_id,
                    "x": x,
                    "y": y,
                }
            )
        if sha256_bytes(actual_bytes) != record.get("stored_pixel_hash"):
            hashes_ok = False
            if first_mismatch is None:
                first_mismatch = _mismatch(
                    "stored_pixel_hash",
                    layer_id,
                    0,
                    0,
                    path_id=layer.get("path_id"),
                    expected=record.get("stored_pixel_hash"),
                    actual=sha256_bytes(actual_bytes),
                )
            errors.append({"code": str(ErrorCode.LAYER_HASH), "message": "落盘层像素哈希不一致", "layer_id": layer_id})

    checks["layer_headers_match"] = headers_ok
    checks["layer_hashes_match"] = hashes_ok
    checks["layer_pixels_match"] = pixels_ok
    checks["fill_rule_coverage_match"] = fill_ok

    final_path = case_dir / layer_manifest.get("final_bgra_file", "final.bgra")
    expected_final = recomputed["canvas"].tobytes()
    try:
        actual_final = final_path.read_bytes()
    except Exception:
        actual_final = b""
    final_difference = _first_byte_difference(expected_final, actual_final, recomputed["width"], recomputed["height"])
    checks["final_buffer_match"] = final_difference is None and sha256_bytes(actual_final) == layer_manifest.get("canvas_hash")
    if final_difference is not None:
        x, y, channel = final_difference
        if first_mismatch is None:
            first_mismatch = _mismatch(
                "final_canvas_pixel",
                None,
                x,
                y,
                channel=channel,
                expected=int(recomputed["canvas"][y, x, channel]),
                actual=int(actual_final[y * recomputed["width"] * 4 + x * 4 + channel]) if len(actual_final) > y * recomputed["width"] * 4 + x * 4 + channel else None,
            )
        errors.append({"code": str(ErrorCode.LAYER_HASH), "message": "最终 BGRA 缓冲不一致", "x": x, "y": y})
    elif sha256_bytes(actual_final) != layer_manifest.get("canvas_hash"):
        checks["final_buffer_match"] = False
        if first_mismatch is None:
            first_mismatch = _mismatch(
                "final_canvas_hash",
                None,
                0,
                0,
                expected=layer_manifest.get("canvas_hash"),
                actual=sha256_bytes(actual_final),
            )
        errors.append({"code": str(ErrorCode.LAYER_HASH), "message": "最终 BGRA 哈希不一致"})

    if first_mismatch is not None and first_mismatch.get("path_id"):
        fragment = segment_fragment(injection_result.scene, first_mismatch["path_id"], None, None)
        if fragment is not None:
            first_mismatch["path_fragment"] = fragment
    status = "ok" if first_mismatch is None else "mismatch"
    if status == "mismatch":
        errors.insert(0, {"code": str(ErrorCode.VERIFIER_MISMATCH), "message": "合成器与独立验证器结论不一致"})
    return {
        "schema": "vgsynth.verification/v1",
        "status": status,
        "code": str(ErrorCode.VERIFIER_MISMATCH) if status == "mismatch" else None,
        "checks": checks,
        "first_mismatch": first_mismatch,
        "errors": errors,
        "classification": {
            "expected": expected_classification,
            "observed": observed_classification,
            "match": checks["injection_classification_match"],
        },
        "recomputed": {
            "flatten_hash": flat["flatten_hash"],
            "canvas_hash": recomputed["canvas_hash"],
            "composite_chain_hash": recomputed["composite_chain_hash"],
            "layer_hashes": {layer["layer_id"]: layer["layer_hash"] for layer in layers},
        },
    }
