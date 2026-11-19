from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .canonical import canonical_json_bytes, sha256_bytes
from .errors import PipelineError


def _err(code: str, path: str, message: str, **details: Any) -> "NoReturn":  # type: ignore[name-defined]
    raise PipelineError(code=code, path=path, message=message, details=details)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, path: str, code: str = "E_TYPE") -> float:
    if not _is_number(value):
        _err(code, path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        _err(code, path, "expected a finite number")
    return result


def _positive_int(value: Any, path: str, code: str = "E_TYPE") -> int:
    if not _is_int(value) or value <= 0:
        _err(code, path, "expected a positive integer")
    return int(value)


def _vec3(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        _err("E_TYPE", path, "expected an array of three finite numbers")
    return (
        _finite_number(value[0], f"{path}[0]"),
        _finite_number(value[1], f"{path}[1]"),
        _finite_number(value[2], f"{path}[2]"),
    )


def _int3(value: Any, path: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        _err("E_TYPE", path, "expected an array of three integers")
    result: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item):
            _err("E_TYPE", f"{path}[{index}]", "expected an integer")
        result.append(int(item))
    return result[0], result[1], result[2]


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class VolumeSpec:
    path: Path
    dims: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    scalar_min: float
    scalar_max: float


@dataclass(frozen=True)
class TransferPoint:
    scalar: float
    opacity: float
    gray: float
    extras: tuple[tuple[str, Any], ...] = ()

    def normalized(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "scalar": self.scalar,
            "opacity": self.opacity,
            "gray": self.gray,
        }
        for key, item in self.extras:
            value[key] = item
        return value


@dataclass(frozen=True)
class TransferFunction:
    points: tuple[TransferPoint, ...]
    options: tuple[tuple[str, Any], ...] = ()

    @property
    def scalars(self) -> np.ndarray:
        return np.asarray([point.scalar for point in self.points], dtype=np.float64)

    @property
    def opacities(self) -> np.ndarray:
        return np.asarray([point.opacity for point in self.points], dtype=np.float64)

    @property
    def grays(self) -> np.ndarray:
        return np.asarray([point.gray for point in self.points], dtype=np.float64)

    def normalized(self) -> list[dict[str, Any]]:
        return [point.normalized() for point in self.points]

    @property
    def digest(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {"points": self.normalized(), "options": dict(self.options)}
            )
        )

    @property
    def options_dict(self) -> dict[str, Any]:
        return dict(self.options)


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_y_deg: float
    aspect: float

    def normalized(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_deg": self.fov_y_deg,
            "aspect": self.aspect,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.normalized()))


@dataclass(frozen=True)
class Edit:
    center: tuple[int, int, int]
    size: tuple[int, int, int]
    value: float

    @property
    def start(self) -> tuple[int, int, int]:
        return tuple(
            self.center[axis] - (self.size[axis] - 1) // 2 for axis in range(3)
        )  # type: ignore[return-value]

    @property
    def stop_exclusive(self) -> tuple[int, int, int]:
        start = self.start
        return tuple(start[axis] + self.size[axis] for axis in range(3))  # type: ignore[return-value]

    def normalized(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "size": list(self.size),
            "value": self.value,
        }


@dataclass(frozen=True)
class TaskSpec:
    task_path: Path
    volume: VolumeSpec
    transfer_function: TransferFunction
    camera: Camera
    width: int
    height: int
    block_width: int
    block_height: int
    concurrency: int
    memory_budget_bytes: int
    sampling_step_mm: float
    background_gray: float
    edits: tuple[Edit, ...]
    state_dir: Path
    gray_path: Path
    depth_path: Path
    report_path: Path

    @property
    def block_cols(self) -> int:
        return self.width // self.block_width

    @property
    def block_rows(self) -> int:
        return self.height // self.block_height

    @property
    def edits_digest(self) -> str:
        return sha256_bytes(
            canonical_json_bytes([edit.normalized() for edit in self.edits])
        )


def _parse_transfer(raw: Any) -> TransferFunction:
    if not isinstance(raw, dict):
        _err("E_TYPE", "transfer_function", "expected an object")
    points_raw = raw.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        _err("E_TYPE", "transfer_function.points", "expected a non-empty array")
    points: list[TransferPoint] = []
    previous_scalar: float | None = None
    for index, raw_point in enumerate(points_raw):
        path = f"transfer_function.points[{index}]"
        if not isinstance(raw_point, dict):
            _err("E_TYPE", path, "expected an object")
        scalar = _finite_number(raw_point.get("scalar"), f"{path}.scalar")
        opacity = _finite_number(raw_point.get("opacity"), f"{path}.opacity")
        gray = _finite_number(raw_point.get("gray"), f"{path}.gray")
        if not (0.0 <= opacity <= 1.0):
            _err("E_TRANSFER", f"{path}.opacity", "opacity must be in [0, 1]")
        if not (0.0 <= gray <= 1.0):
            _err("E_TRANSFER", f"{path}.gray", "gray must be in [0, 1]")
        if previous_scalar is not None and not scalar > previous_scalar:
            _err(
                "E_TRANSFER",
                f"{path}.scalar",
                "transfer-function scalar values must be strictly increasing",
            )
        previous_scalar = scalar
        extras = tuple(
            sorted(
                (key, extra_value)
                for key, extra_value in raw_point.items()
                if key not in {"scalar", "opacity", "gray"}
            )
        )
        points.append(
            TransferPoint(scalar=scalar, opacity=opacity, gray=gray, extras=extras)
        )
    options = tuple(
        sorted(
            (key, value)
            for key, value in raw.items()
            if key != "points"
        )
    )
    return TransferFunction(points=tuple(points), options=options)


def _parse_camera(raw: Any, width: int, height: int) -> Camera:
    if not isinstance(raw, dict):
        _err("E_TYPE", "camera", "expected an object")
    position = _vec3(raw.get("position"), "camera.position")
    target = _vec3(raw.get("target"), "camera.target")
    up = _vec3(raw.get("up"), "camera.up")
    fov = _finite_number(raw.get("fov_y_deg"), "camera.fov_y_deg")
    if not (0.0 < fov < 180.0):
        _err("E_CAMERA", "camera.fov_y_deg", "fov_y_deg must be in (0, 180)")
    if "aspect" in raw:
        aspect = _finite_number(raw.get("aspect"), "camera.aspect")
        if aspect <= 0.0:
            _err("E_CAMERA", "camera.aspect", "aspect must be positive")
    else:
        aspect = width / height

    position_np = np.asarray(position, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    up_np = np.asarray(up, dtype=np.float64)
    forward = target_np - position_np
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-12:
        _err("E_CAMERA", "camera.target", "camera target must differ from position")
    up_norm = float(np.linalg.norm(up_np))
    if up_norm <= 1e-12:
        _err("E_CAMERA", "camera.up", "up vector must be non-zero")
    if float(np.linalg.norm(np.cross(forward, up_np))) <= 1e-12 * max(1.0, forward_norm * up_norm):
        _err("E_CAMERA", "camera.up", "up vector must not be parallel to the view direction")
    return Camera(position=position, target=target, up=up, fov_y_deg=fov, aspect=aspect)


def _parse_edits(raw: Any, dims: Sequence[int], scalar_min: float, scalar_max: float) -> tuple[Edit, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        _err("E_TYPE", "edits", "expected an array")
    edits: list[Edit] = []
    for index, item in enumerate(raw):
        path = f"edits[{index}]"
        if not isinstance(item, dict):
            _err("E_TYPE", path, "expected an object")
        center = _int3(item.get("center"), f"{path}.center")
        size = _int3(item.get("size"), f"{path}.size")
        value = _finite_number(item.get("value"), f"{path}.value")
        edit = Edit(center=center, size=size, value=value)
        for axis in range(3):
            if size[axis] <= 0:
                _err("E_EDIT", f"{path}.size[{axis}]", "edit size must be a positive integer")
            if not (0 <= edit.start[axis] < edit.stop_exclusive[axis] <= dims[axis]):
                _err(
                    "E_EDIT",
                    f"{path}.size[{axis}]",
                    "edit region must be fully inside the volume",
                    start=edit.start[axis],
                    stop_exclusive=edit.stop_exclusive[axis],
                    dimension=dims[axis],
                )
        if not math.isfinite(value) or not (scalar_min <= value <= scalar_max):
            _err(
                "E_EDIT",
                f"{path}.value",
                "edit value must be inside the volume scalar domain",
                scalar_min=scalar_min,
                scalar_max=scalar_max,
            )
        edits.append(edit)
    return tuple(edits)


def load_task(
    task_path: str | Path,
    *,
    state_dir_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> TaskSpec:
    path = Path(task_path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise PipelineError("E_IO", "$", f"cannot read task file: {exc}", {"task": str(path)})
    try:
        raw = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError("E_JSON", "$", f"invalid JSON: {exc}")
    if not isinstance(raw, dict):
        raise PipelineError("E_TYPE", "$", "task description must be a JSON object")
    schema_version = raw.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version != 1:
        _err("E_SCHEMA", "schema_version", "unsupported schema version")

    base_dir = path.resolve().parent
    volume_raw = raw.get("volume")
    if not isinstance(volume_raw, dict):
        _err("E_TYPE", "volume", "expected an object")
    volume_path_value = volume_raw.get("path")
    if not isinstance(volume_path_value, str) or not volume_path_value:
        _err("E_TYPE", "volume.path", "expected a non-empty string")
    dims = _int3(volume_raw.get("dims"), "volume.dims")
    for axis, dimension in enumerate(dims):
        if dimension <= 0:
            _err("E_DIMENSIONS", f"volume.dims[{axis}]", "dimension must be a positive integer")
    spacing = _vec3(volume_raw.get("spacing_mm"), "volume.spacing_mm")
    for axis, value in enumerate(spacing):
        if value <= 0.0:
            _err("E_SPACING", f"volume.spacing_mm[{axis}]", "spacing must be positive")
    domain_raw = volume_raw.get("scalar_domain")
    if not isinstance(domain_raw, dict):
        _err("E_TYPE", "volume.scalar_domain", "expected an object")
    scalar_min = _finite_number(domain_raw.get("min"), "volume.scalar_domain.min")
    scalar_max = _finite_number(domain_raw.get("max"), "volume.scalar_domain.max")
    if not scalar_min < scalar_max:
        _err("E_VOLUME", "volume.scalar_domain", "scalar_domain.min must be less than max")

    transfer_function = _parse_transfer(raw.get("transfer_function"))
    for index, point in enumerate(transfer_function.points):
        if not (scalar_min <= point.scalar <= scalar_max):
            _err(
                "E_TRANSFER",
                f"transfer_function.points[{index}].scalar",
                "transfer-function scalar must be inside the volume scalar domain",
            )

    output_raw = raw.get("output")
    if not isinstance(output_raw, dict):
        _err("E_TYPE", "output", "expected an object")
    width = _positive_int(output_raw.get("width"), "output.width", "E_OUTPUT")
    height = _positive_int(output_raw.get("height"), "output.height", "E_OUTPUT")

    blocks_raw = raw.get("blocks")
    if not isinstance(blocks_raw, dict):
        _err("E_TYPE", "blocks", "expected an object")
    block_width = _positive_int(blocks_raw.get("width"), "blocks.width", "E_BLOCK")
    block_height = _positive_int(blocks_raw.get("height"), "blocks.height", "E_BLOCK")
    if width % block_width != 0:
        _err("E_BLOCK", "blocks.width", "block width must exactly cover the output width")
    if height % block_height != 0:
        _err("E_BLOCK", "blocks.height", "block height must exactly cover the output height")

    concurrency = _positive_int(raw.get("concurrency"), "concurrency", "E_CONCURRENCY")
    memory_budget = _positive_int(
        raw.get("memory_budget_bytes"), "memory_budget_bytes", "E_MEMORY"
    )

    if "sampling_step_mm" in raw:
        sampling_step = _finite_number(raw.get("sampling_step_mm"), "sampling_step_mm")
        if sampling_step <= 0.0:
            _err("E_TYPE", "sampling_step_mm", "sampling step must be positive")
    else:
        sampling_step = min(spacing) * 0.5

    background_gray_raw = raw.get("background_gray", 0.0)
    background_gray = _finite_number(background_gray_raw, "background_gray")
    if not (0.0 <= background_gray <= 1.0):
        _err("E_TYPE", "background_gray", "background_gray must be in [0, 1]")

    camera = _parse_camera(raw.get("camera"), width, height)
    edits = _parse_edits(raw.get("edits"), dims, scalar_min, scalar_max)

    state_dir_value = raw.get("state_dir")
    if not isinstance(state_dir_value, str) or not state_dir_value:
        _err("E_TYPE", "state_dir", "expected a non-empty string")
    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, dict):
        _err("E_TYPE", "outputs", "expected an object")
    output_paths: dict[str, Path] = {}
    for key in ("gray", "depth", "report"):
        value = outputs_raw.get(key)
        if not isinstance(value, str) or not value:
            _err("E_TYPE", f"outputs.{key}", "expected a non-empty string")
        output_paths[key] = _resolve(base_dir, value)

    if state_dir_override is not None:
        state_dir = Path(state_dir_override)
        if not state_dir.is_absolute():
            state_dir = base_dir / state_dir
        state_dir = state_dir.resolve()
    else:
        state_dir = _resolve(base_dir, state_dir_value)

    if output_dir_override is not None:
        output_dir = Path(output_dir_override)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        output_dir = output_dir.resolve()
        output_paths = {
            "gray": output_dir / "gray.npy",
            "depth": output_dir / "depth.npy",
            "report": output_dir / "report.json",
        }

    if len({path for path in output_paths.values()}) != 3:
        _err("E_OUTPUT", "outputs", "gray, depth, and report paths must be distinct")
    volume_path = _resolve(base_dir, volume_path_value)
    if volume_path in set(output_paths.values()):
        _err("E_OUTPUT", "outputs", "output paths must not overwrite the input volume")
    if state_dir / "manifest.json" in set(output_paths.values()):
        _err("E_OUTPUT", "outputs", "output paths must not overwrite the state manifest")

    return TaskSpec(
        task_path=path.resolve(),
        volume=VolumeSpec(
            path=volume_path,
            dims=dims,
            spacing_mm=spacing,
            scalar_min=scalar_min,
            scalar_max=scalar_max,
        ),
        transfer_function=transfer_function,
        camera=camera,
        width=width,
        height=height,
        block_width=block_width,
        block_height=block_height,
        concurrency=concurrency,
        memory_budget_bytes=memory_budget,
        sampling_step_mm=sampling_step,
        background_gray=background_gray,
        edits=edits,
        state_dir=state_dir,
        gray_path=output_paths["gray"],
        depth_path=output_paths["depth"],
        report_path=output_paths["report"],
    )


def task_render_key(task: TaskSpec) -> dict[str, Any]:
    return {
        "dims": list(task.volume.dims),
        "spacing_mm": list(task.volume.spacing_mm),
        "scalar_domain": [task.volume.scalar_min, task.volume.scalar_max],
        "sampling_step_mm": task.sampling_step_mm,
        "background_gray": task.background_gray,
        "width": task.width,
        "height": task.height,
        "block_width": task.block_width,
        "block_height": task.block_height,
    }
