from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fixedpoint import SCALE, as_q
from .png_codec import PngError, read_png_rgb


FIXED_WIDTH = 64
FIXED_HEIGHT = 48
FIXED_MODE = "RGB"

BUSINESS_CODES = {
    "OK": 0,
    "VALIDATION_FAILED": 4000,
    "E_JSON": 4001,
    "E_CONFIG_SCHEMA": 4002,
    "E_CONFIG_VALUE": 4003,
    "E_PNG_SIGNATURE": 4101,
    "E_PNG_HEADER": 4102,
    "E_PNG_DIMENSIONS": 4103,
    "E_PNG_MODE": 4104,
    "E_PNG_CRC": 4105,
    "E_PNG_CONTAINER": 4106,
    "E_IMAGE_MISSING": 4110,
    "E_IMAGE_VIEW_MISMATCH": 4111,
    "E_VIEW_INDEX_DUPLICATE": 4112,
    "E_CAMERA_SCHEMA": 4201,
    "E_INTRINSIC_SINGULAR": 4202,
    "E_EXTRINSIC_ORTHOGONAL": 4203,
    "E_COMMAND_SCHEMA": 4301,
    "E_COMMAND_UNSUPPORTED": 4302,
}


@dataclass
class Failure:
    code: str
    target: str
    message: str

    @property
    def numeric_code(self) -> int:
        return BUSINESS_CODES.get(self.code, 4000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "business_code": self.numeric_code,
            "target": self.target,
            "message": self.message,
        }


@dataclass
class LoadedProject:
    root: Path
    ok: bool
    config: dict[str, Any] | None = None
    cameras: dict[str, Any] | None = None
    commands: dict[str, Any] | None = None
    views: list[dict[str, Any]] = field(default_factory=list)
    images: dict[int, Any] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)

    @property
    def code(self) -> str:
        return "OK" if self.ok else "VALIDATION_FAILED"

    @property
    def business_code(self) -> int:
        return BUSINESS_CODES[self.code]

    def report(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "business_code": self.business_code,
            "failure_count": len(self.failures),
            "failures": [item.to_dict() for item in self.failures],
            "view_count": len(self.views),
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_json(path: Path, failures: list[Failure]) -> Any | None:
    target = str(path)
    if not path.exists():
        failures.append(Failure("E_JSON", target, "required JSON file does not exist"))
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(Failure("E_JSON", target, f"invalid JSON: {exc.msg} at line {exc.lineno}"))
        return None


def _integer_matrix(value: Any, rows: int, cols: int, target: str, failures: list[Failure]) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) != rows:
        failures.append(Failure("E_CAMERA_SCHEMA", target, f"expected {rows}x{cols} matrix"))
        return None
    result: list[list[int]] = []
    for r, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            failures.append(Failure("E_CAMERA_SCHEMA", f"{target}[{r}]", f"expected {cols} columns"))
            return None
        qrow: list[int] = []
        for c, item in enumerate(row):
            try:
                qrow.append(as_q(item))
            except (TypeError, ValueError, ArithmeticError):
                failures.append(Failure("E_CAMERA_SCHEMA", f"{target}[{r}][{c}]", "numeric value required"))
                return None
        result.append(qrow)
    return result


def _det3(matrix: list[list[int]]) -> int:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_config(config: Any, failures: list[Failure]) -> dict[str, Any] | None:
    target = "config/reconstruction.json"
    if not isinstance(config, dict):
        failures.append(Failure("E_CONFIG_SCHEMA", target, "configuration must be an object"))
        return None
    image = config.get("image")
    if not isinstance(image, dict):
        failures.append(Failure("E_CONFIG_SCHEMA", "config.image", "image descriptor required"))
    else:
        if image.get("width") != FIXED_WIDTH or image.get("height") != FIXED_HEIGHT:
            failures.append(
                Failure("E_PNG_DIMENSIONS", "config.image", f"fixed resolution is {FIXED_WIDTH}x{FIXED_HEIGHT}")
            )
        if image.get("pixel_mode") != FIXED_MODE:
            failures.append(Failure("E_PNG_MODE", "config.image", f"fixed pixel mode is {FIXED_MODE}"))

    positive_int_keys = ("min_parallax", "residual_threshold_q", "voxel_size", "depth_quantization")
    for key in positive_int_keys:
        value = config.get(key)
        if not _is_int(value) or value <= 0:
            failures.append(Failure("E_CONFIG_VALUE", f"config.{key}", "must be a positive integer"))

    for key in ("max_subdivision_depth", "global_point_budget", "view_point_quota"):
        value = config.get(key)
        if not _is_int(value) or value < 0:
            failures.append(Failure("E_CONFIG_VALUE", f"config.{key}", "must be a non-negative integer"))

    workers = config.get("workers", 1)
    if not _is_int(workers) or workers <= 0:
        failures.append(Failure("E_CONFIG_VALUE", "config.workers", "must be a positive integer"))

    bounds = config.get("world_bounds")
    if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
        failures.append(Failure("E_CONFIG_SCHEMA", "config.world_bounds", "must contain min and max"))
    else:
        if not all(isinstance(v, list) and len(v) == 3 for v in (bounds.get("min"), bounds.get("max"))):
            failures.append(Failure("E_CONFIG_SCHEMA", "config.world_bounds", "min/max must be length 3"))
        elif any(not _is_int(x) for x in bounds["min"] + bounds["max"]):
            failures.append(Failure("E_CONFIG_VALUE", "config.world_bounds", "bounds must be integers"))
        elif any(bounds["min"][i] >= bounds["max"][i] for i in range(3)):
            failures.append(Failure("E_CONFIG_VALUE", "config.world_bounds", "min must be less than max"))

    if failures:
        return None
    return config


def _validate_camera_matrices(
    camera: dict[str, Any], target: str, failures: list[Failure]
) -> dict[str, Any] | None:
    k_q = _integer_matrix(camera.get("K"), 3, 3, f"{target}.K", failures)
    r_q = _integer_matrix(camera.get("R"), 3, 3, f"{target}.R", failures)
    t_raw = camera.get("t")
    if not isinstance(t_raw, list) or len(t_raw) != 3:
        failures.append(Failure("E_CAMERA_SCHEMA", f"{target}.t", "expected length 3 vector"))
        return None
    try:
        t_q = [as_q(v) for v in t_raw]
    except (TypeError, ValueError):
        failures.append(Failure("E_CAMERA_SCHEMA", f"{target}.t", "translation must be numeric"))
        return None
    if k_q is None or r_q is None:
        return None

    if _det3(k_q) == 0:
        failures.append(Failure("E_INTRINSIC_SINGULAR", f"{target}.K", "intrinsic matrix must be invertible"))

    scale = int(camera.get("fixed_point_scale", SCALE))
    tolerance = max(64, scale * scale // 100)
    for i in range(3):
        norm = sum(r_q[i][k] * r_q[i][k] for k in range(3))
        if abs(norm - scale * scale) > 2 * scale * scale // 100:
            failures.append(Failure("E_EXTRINSIC_ORTHOGONAL", f"{target}.R[{i}]", "rotation row is not unit length"))
        for j in range(i + 1, 3):
            dot = sum(r_q[i][k] * r_q[j][k] for k in range(3))
            if abs(dot) > tolerance:
                failures.append(
                    Failure("E_EXTRINSIC_ORTHOGONAL", f"{target}.R[{i},{j}]", "rotation rows are not orthogonal")
                )
    if _det3(r_q) <= 0:
        failures.append(Failure("E_EXTRINSIC_ORTHOGONAL", f"{target}.R", "rotation determinant must be positive"))

    return {"K_q": k_q, "R_q": r_q, "t_q": t_q}


def validate_commands(commands: Any, failures: list[Failure]) -> bool:
    target = "config/edit_commands.json"
    if not isinstance(commands, dict) or not isinstance(commands.get("commands"), list):
        failures.append(Failure("E_COMMAND_SCHEMA", target, "commands must contain a commands list"))
        return False
    seen_ids: set[str] = set()
    supported = {"set_config", "set_extrinsic", "replace_view", "set_budget"}
    for pos, command in enumerate(commands["commands"]):
        name = f"{target}.commands[{pos}]"
        if not isinstance(command, dict):
            failures.append(Failure("E_COMMAND_SCHEMA", name, "command must be object"))
            continue
        command_id = command.get("id")
        if not isinstance(command_id, str) or not command_id:
            failures.append(Failure("E_COMMAND_SCHEMA", f"{name}.id", "non-empty string id required"))
        elif command_id in seen_ids:
            failures.append(Failure("E_COMMAND_SCHEMA", f"{name}.id", f"duplicate command id {command_id}"))
        else:
            seen_ids.add(command_id)
        op = command.get("op")
        if op not in supported:
            failures.append(Failure("E_COMMAND_UNSUPPORTED", f"{name}.op", f"unsupported op {op!r}"))
            continue
        if op == "set_config" and (not isinstance(command.get("key"), str) or "value" not in command):
            failures.append(Failure("E_COMMAND_SCHEMA", name, "set_config requires key and value"))
        if op == "set_extrinsic":
            if not _is_int(command.get("view_index")) or not isinstance(command.get("t"), list):
                failures.append(Failure("E_COMMAND_SCHEMA", name, "set_extrinsic requires integer view_index and t"))
        if op == "replace_view":
            if not _is_int(command.get("view_index")) or not isinstance(command.get("source"), str):
                failures.append(Failure("E_COMMAND_SCHEMA", name, "replace_view requires view_index and source"))
        if op == "set_budget" and not isinstance(command.get("budget"), dict):
            failures.append(Failure("E_COMMAND_SCHEMA", name, "set_budget requires budget object"))
    return not any(f.code.startswith("E_COMMAND") for f in failures)


def load_project(root: str | Path, require_commands: bool = True) -> LoadedProject:
    root = Path(root)
    failures: list[Failure] = []
    config = _load_json(root / "config" / "reconstruction.json", failures)
    config_clean = _validate_config(config, failures) if config is not None else None
    cameras_doc = _load_json(root / "data" / "cameras.json", failures)
    commands_doc = None
    command_path = root / "config" / "edit_commands.json"
    if require_commands or command_path.exists():
        commands_doc = _load_json(command_path, failures)
        if commands_doc is not None:
            validate_commands(commands_doc, failures)

    loaded = LoadedProject(root=root, ok=False, config=config_clean, cameras=cameras_doc, commands=commands_doc)
    if not isinstance(cameras_doc, dict) or not isinstance(cameras_doc.get("views"), list):
        failures.append(Failure("E_CAMERA_SCHEMA", "data/cameras.json.views", "views list required"))
        loaded.failures = failures
        return loaded

    seen_indices: set[int] = set()
    seen_images: set[str] = set()
    seen_image_paths: set[Path] = set()
    image_parent_dirs: set[Path] = set()
    for pos, camera in enumerate(cameras_doc["views"]):
        target = f"data/cameras.json.views[{pos}]"
        if not isinstance(camera, dict):
            failures.append(Failure("E_CAMERA_SCHEMA", target, "camera must be object"))
            continue
        index = camera.get("index")
        if not _is_int(index) or index < 0:
            failures.append(Failure("E_VIEW_INDEX_DUPLICATE", f"{target}.index", "non-negative integer required"))
            continue
        if index in seen_indices:
            failures.append(Failure("E_VIEW_INDEX_DUPLICATE", f"{target}.index", f"duplicate view index {index}"))
        seen_indices.add(index)

        image_rel = camera.get("image")
        if not isinstance(image_rel, str) or not image_rel:
            failures.append(Failure("E_IMAGE_VIEW_MISMATCH", f"{target}.image", "image path required"))
            continue
        if image_rel in seen_images:
            failures.append(Failure("E_IMAGE_VIEW_MISMATCH", f"{target}.image", "image is assigned to multiple views"))
        seen_images.add(image_rel)

        image_path = ((root / "data" / "cameras.json").parent / image_rel)
        resolved_image_path = image_path.resolve()
        seen_image_paths.add(resolved_image_path)
        image_parent_dirs.add(resolved_image_path.parent)
        try:
            resolved_image_path.relative_to((root / "data").resolve())
        except ValueError:
            failures.append(Failure("E_IMAGE_VIEW_MISMATCH", f"{target}.image", "image must stay below data/"))
            continue
        if not image_path.exists():
            failures.append(Failure("E_IMAGE_MISSING", str(image_path), "image file does not exist"))
            continue
        try:
            pixels = read_png_rgb(image_path)
        except PngError as exc:
            failures.append(Failure(exc.code, image_rel, str(exc)))
            continue
        if pixels.shape[1] != FIXED_WIDTH or pixels.shape[0] != FIXED_HEIGHT:
            failures.append(
                Failure(
                    "E_PNG_DIMENSIONS",
                    image_rel,
                    f"PNG is {pixels.shape[1]}x{pixels.shape[0]}, expected {FIXED_WIDTH}x{FIXED_HEIGHT}",
                )
            )
            continue

        matrices = _validate_camera_matrices(camera, target, failures)
        if matrices is None:
            continue
        view = dict(camera)
        view.update(matrices)
        view["image_abs"] = str(image_path)
        view["pixels"] = pixels
        loaded.views.append(view)
        loaded.images[index] = pixels

    for image_dir in image_parent_dirs:
        if not image_dir.exists():
            continue
        for png_path in sorted(image_dir.glob("*.png")):
            if png_path.resolve() not in seen_image_paths:
                failures.append(
                    Failure(
                        "E_IMAGE_VIEW_MISMATCH",
                        str(png_path),
                        "PNG in active view directory is not assigned to exactly one camera view",
                    )
                )

    loaded.views.sort(key=lambda item: item.get("index", 0))
    if [v.get("index") for v in loaded.views] != sorted(seen_indices):
        pass
    if len({v.get("index") for v in loaded.views}) != len(cameras_doc["views"]):
        pass
    loaded.failures = failures
    loaded.ok = not failures and config_clean is not None and len(loaded.views) == len(cameras_doc.get("views", []))
    if loaded.ok:
        loaded.config = config_clean
        loaded.cameras = cameras_doc
    return loaded
