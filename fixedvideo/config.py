"""Configuration, sample-array, vocabulary, and command-stream validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (
    E_CONFIG_INVALID,
    E_NOISE_TABLE_MISMATCH,
    E_SEED_OUT_OF_RANGE,
    RenderError,
)
from .util import (
    array_sha256,
    hash_object,
    read_json,
    require_mapping,
    resolve_relative,
)


QUANTIZATION_STEPS = {"q1": 1, "q2": 32, "q4": 256, "q8": 1024}
SUPPORTED_COMMANDS = {
    "set_prompt",
    "set_seed",
    "set_guidance",
    "set_steps",
    "set_frame_count",
    "set_initial_image",
    "add_patch",
    "remove_patch",
    "set_chunk_rows",
    "set_quantization_level",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_int(mapping: dict[str, Any], key: str, where: str) -> int:
    if key not in mapping or not _is_int(mapping[key]):
        raise RenderError(E_CONFIG_INVALID, f"{where}.{key} must be an integer")
    return int(mapping[key])


def _require_number(mapping: dict[str, Any], key: str, where: str) -> float:
    if key not in mapping or not _is_number(mapping[key]):
        raise RenderError(E_CONFIG_INVALID, f"{where}.{key} must be a number")
    return float(mapping[key])


def _require_string(mapping: dict[str, Any], key: str, where: str) -> str:
    if key not in mapping or not isinstance(mapping[key], str):
        raise RenderError(E_CONFIG_INVALID, f"{where}.{key} must be a string")
    return mapping[key]


def _require_int_list(mapping: dict[str, Any], key: str, where: str) -> tuple[int, ...]:
    raw = mapping.get(key)
    if not isinstance(raw, list) or not raw or any(not _is_int(item) for item in raw):
        raise RenderError(E_CONFIG_INVALID, f"{where}.{key} must be a non-empty integer array")
    return tuple(int(item) for item in raw)


def _require_string_list(mapping: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    raw = mapping.get(key)
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise RenderError(E_CONFIG_INVALID, f"{where}.{key} must be a non-empty string array")
    return tuple(raw)


@dataclass(frozen=True)
class InitialLatent:
    path: Path
    array: np.ndarray
    raw_fingerprint: str
    shape: tuple[int, ...]
    dtype: str

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "raw_fingerprint": self.raw_fingerprint,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class RenderConfig:
    raw: dict[str, Any]
    base_dir: Path
    source_hash: str
    name: str
    seed_min: int
    seed_max: int
    seed: int
    default_prompt: str
    steps: int
    guidance: float
    frame_count: int
    height: int
    width: int
    chunk_rows: int
    latent_shape: tuple[int, int, int]
    embedding_dim: int
    fixed_point_bits: int
    allowed_fixed_point_bits: tuple[int, ...]
    quantization_level: str
    allowed_quantization_levels: tuple[str, ...]
    quantization_step: int
    interaction_latency_budget_ms: int
    worker_count: int
    snapshot_interval_commands: int
    recompute_tile_budget: int
    initial_skip_steps: int
    noise_table_version: str
    noise_table_versions: tuple[str, ...]
    noise_table_path: Path
    noise_table: np.ndarray
    noise_shape: tuple[int, ...]
    noise_version_in_file: str
    vocabulary_path: Path
    vocabulary: tuple[str, ...]
    vocabulary_hash: str
    initial_image_spec: dict[str, Any] | None
    initial_image: InitialLatent | None
    projection_seed: int
    condition_seed: int
    output_gain: int

    @property
    def scale(self) -> int:
        return 1 << self.fixed_point_bits

    @property
    def latent_size(self) -> int:
        return int(np.prod(self.latent_shape))

    @property
    def max_noise_steps(self) -> int:
        return int(self.noise_table.shape[0])

    @property
    def max_noise_frames(self) -> int:
        return int(self.noise_table.shape[1])

    def latent_parameter_fingerprint(self, quantization_level: str) -> str:
        return hash_object(
            {
                "vocabulary_hash": self.vocabulary_hash,
                "embedding_dim": self.embedding_dim,
                "latent_shape": list(self.latent_shape),
                "fixed_point_bits": self.fixed_point_bits,
                "quantization_level": quantization_level,
                "condition_seed": self.condition_seed,
            },
            domain="fixedvideo-latent-config",
        )

    def static_model_fingerprint(self) -> str:
        return hash_object(
            {
                "vocabulary_hash": self.vocabulary_hash,
                "embedding_dim": self.embedding_dim,
                "latent_shape": list(self.latent_shape),
                "resolution": [self.height, self.width],
                "fixed_point_bits": self.fixed_point_bits,
                "projection_seed": self.projection_seed,
                "condition_seed": self.condition_seed,
                "output_gain": self.output_gain,
                "noise_version": self.noise_table_version,
                "noise_shape": list(self.noise_shape),
            },
            domain="fixedvideo-static-model",
        )


def _load_vocabulary(path: Path) -> tuple[tuple[str, ...], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(E_CONFIG_INVALID, f"cannot read vocabulary: {path}", {"error": str(exc)}) from exc
    tokens = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not tokens:
        raise RenderError(E_CONFIG_INVALID, "vocabulary must contain at least one token")
    if len(set(tokens)) != len(tokens):
        raise RenderError(E_CONFIG_INVALID, "vocabulary contains duplicate tokens")
    for token in tokens:
        if any(ch.isspace() for ch in token) or not token:
            raise RenderError(E_CONFIG_INVALID, f"invalid vocabulary token: {token!r}")
    return tuple(tokens), hash_object(list(tokens), domain="fixedvideo-vocabulary")


def _load_noise_table(path: Path) -> tuple[np.ndarray, tuple[int, ...], str]:
    if not path.exists():
        raise RenderError(E_NOISE_TABLE_MISMATCH, f"noise table does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"noise", "version", "shape", "dtype"}
            missing = required.difference(archive.files)
            if missing:
                raise RenderError(
                    E_NOISE_TABLE_MISMATCH,
                    "noise table is missing descriptors",
                    {"missing": sorted(missing)},
                )
            noise = np.asarray(archive["noise"])
            version_raw = archive["version"]
            shape = tuple(int(item) for item in np.asarray(archive["shape"]).reshape(-1).tolist())
            dtype_raw = archive["dtype"]
    except RenderError:
        raise
    except (OSError, ValueError) as exc:
        raise RenderError(E_NOISE_TABLE_MISMATCH, f"cannot load noise table: {path}", {"error": str(exc)}) from exc
    version = str(version_raw.item() if getattr(version_raw, "shape", ()) == () else version_raw.tolist())
    dtype = dtype_raw.item() if getattr(dtype_raw, "shape", ()) == () else str(dtype_raw.tolist())
    dtype = str(dtype)
    if not shape or any(item <= 0 for item in shape):
        raise RenderError(E_NOISE_TABLE_MISMATCH, "noise table shape descriptor is invalid")
    if noise.dtype.kind not in {"i", "u"}:
        raise RenderError(E_NOISE_TABLE_MISMATCH, "noise table must contain integer fixed-point entries")
    if int(np.prod(shape)) != int(noise.size):
        raise RenderError(
            E_NOISE_TABLE_MISMATCH,
            "noise table shape descriptor does not match data length",
            {"shape": list(shape), "size": int(noise.size)},
        )
    if tuple(noise.shape) != shape:
        raise RenderError(
            E_NOISE_TABLE_MISMATCH,
            "noise table ndarray shape does not match its descriptor",
            {"array_shape": list(noise.shape), "descriptor": list(shape)},
        )
    if dtype != str(noise.dtype):
        raise RenderError(
            E_NOISE_TABLE_MISMATCH,
            "noise table dtype descriptor does not match data dtype",
            {"descriptor": dtype, "actual": str(noise.dtype)},
        )
    return noise.astype(np.int64, copy=False), shape, version


def load_initial_latent(path: str | Path, expected_shape: tuple[int, ...] | None = None) -> InitialLatent:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise RenderError(E_CONFIG_INVALID, f"initial latent file does not exist: {resolved}")
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            required = {"latent", "shape", "dtype"}
            missing = required.difference(archive.files)
            if missing:
                raise RenderError(
                    E_CONFIG_INVALID,
                    "initial latent file is missing descriptors",
                    {"file": str(resolved), "missing": sorted(missing)},
                )
            latent = np.asarray(archive["latent"])
            shape = tuple(int(item) for item in np.asarray(archive["shape"]).reshape(-1).tolist())
            dtype_raw = archive["dtype"]
    except RenderError:
        raise
    except (OSError, ValueError) as exc:
        raise RenderError(
            E_CONFIG_INVALID,
            f"cannot load initial latent file: {resolved}",
            {"error": str(exc)},
        ) from exc
    dtype = dtype_raw.item() if getattr(dtype_raw, "shape", ()) == () else str(dtype_raw.tolist())
    dtype = str(dtype)
    if latent.dtype != np.float32:
        raise RenderError(
            E_CONFIG_INVALID,
            "initial latent data must be float32",
            {"actual_dtype": str(latent.dtype)},
        )
    if dtype != "float32":
        raise RenderError(
            E_CONFIG_INVALID,
            "initial latent dtype descriptor must be float32",
            {"descriptor": dtype},
        )
    if not shape or any(item <= 0 for item in shape):
        raise RenderError(E_CONFIG_INVALID, "initial latent shape descriptor is invalid")
    if int(np.prod(shape)) != int(latent.size):
        raise RenderError(
            E_CONFIG_INVALID,
            "initial latent shape descriptor does not match data length",
            {"shape": list(shape), "size": int(latent.size)},
        )
    if tuple(latent.shape) != shape:
        raise RenderError(
            E_CONFIG_INVALID,
            "initial latent ndarray shape does not match its descriptor",
            {"array_shape": list(latent.shape), "descriptor": list(shape)},
        )
    if expected_shape is not None and tuple(latent.shape) != tuple(expected_shape):
        raise RenderError(
            E_CONFIG_INVALID,
            "initial latent shape does not match configured latent shape",
            {"expected": list(expected_shape), "actual": list(latent.shape)},
        )
    contiguous = np.ascontiguousarray(latent, dtype=np.float32)
    return InitialLatent(
        path=resolved,
        array=contiguous,
        raw_fingerprint=array_sha256(contiguous),
        shape=tuple(contiguous.shape),
        dtype="float32",
    )


def _validate_initial_image_spec(raw: dict[str, Any], base_dir: Path, latent_shape: tuple[int, int, int]) -> tuple[dict[str, Any] | None, InitialLatent | None]:
    spec_raw = raw.get("initial_image")
    if spec_raw is None:
        return None, None
    if not isinstance(spec_raw, dict):
        raise RenderError(E_CONFIG_INVALID, "initial_image must be an object or null")
    path_text = spec_raw.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise RenderError(E_CONFIG_INVALID, "initial_image.path must be a non-empty string")
    expected = spec_raw.get("shape", list(latent_shape))
    if not isinstance(expected, list) or any(not _is_int(item) for item in expected):
        raise RenderError(E_CONFIG_INVALID, "initial_image.shape must be an integer array")
    expected_shape = tuple(int(item) for item in expected)
    if expected_shape != latent_shape:
        raise RenderError(
            E_CONFIG_INVALID,
            "initial_image.shape must match latent_shape",
            {"initial_image_shape": list(expected_shape), "latent_shape": list(latent_shape)},
        )
    path = resolve_relative(base_dir, path_text)
    initial = load_initial_latent(path, expected_shape=latent_shape)
    spec = {
        "path": str(path),
        "shape": list(latent_shape),
        "dtype": "float32",
        "raw_fingerprint": initial.raw_fingerprint,
    }
    return spec, initial


def load_render_config(path: str | Path) -> RenderConfig:
    resolved = Path(path).resolve()
    try:
        raw_value = read_json(resolved)
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(E_CONFIG_INVALID, f"cannot read render config: {resolved}", {"error": str(exc)}) from exc
    if not isinstance(raw_value, dict):
        raise RenderError(E_CONFIG_INVALID, "render config root must be an object")
    raw: dict[str, Any] = raw_value
    base_dir = resolved.parent

    name = raw.get("name", resolved.stem)
    if not isinstance(name, str):
        raise RenderError(E_CONFIG_INVALID, "name must be a string")

    seed_range = raw.get("seed_range", [0, 2**32 - 1])
    if (
        not isinstance(seed_range, list)
        or len(seed_range) != 2
        or any(not _is_int(item) for item in seed_range)
    ):
        raise RenderError(E_CONFIG_INVALID, "seed_range must be [min_integer, max_integer]")
    seed_min, seed_max = int(seed_range[0]), int(seed_range[1])
    if seed_min < 0 or seed_max < seed_min:
        raise RenderError(E_CONFIG_INVALID, "seed_range is invalid")
    seed = _require_int(raw, "seed", "config")
    if not seed_min <= seed <= seed_max:
        raise RenderError(
            E_SEED_OUT_OF_RANGE,
            "configured seed is outside seed_range",
            {"seed": seed, "seed_range": [seed_min, seed_max]},
        )

    default_prompt = raw.get("prompt", "")
    if not isinstance(default_prompt, str):
        raise RenderError(E_CONFIG_INVALID, "config.prompt must be a string")
    steps = _require_int(raw, "steps", "config")
    if steps <= 0:
        raise RenderError(E_CONFIG_INVALID, "config.steps must be a positive integer")
    guidance = _require_number(raw, "guidance", "config")
    if guidance < 0 or guidance > 100:
        raise RenderError(E_CONFIG_INVALID, "config.guidance must be between 0 and 100")
    frame_count = _require_int(raw, "frame_count", "config")
    if frame_count <= 0:
        raise RenderError(E_CONFIG_INVALID, "config.frame_count must be a positive integer")

    resolution = require_mapping(raw.get("frame_resolution"), "config.frame_resolution")
    height = _require_int(resolution, "height", "config.frame_resolution")
    width = _require_int(resolution, "width", "config.frame_resolution")
    if height <= 0 or width <= 0:
        raise RenderError(E_CONFIG_INVALID, "frame resolution must be positive")
    chunk_rows = _require_int(raw, "chunk_rows", "config")
    if chunk_rows <= 0:
        raise RenderError(E_CONFIG_INVALID, "config.chunk_rows must be a positive integer")

    latent_shape_raw = raw.get("latent_shape")
    if (
        not isinstance(latent_shape_raw, list)
        or len(latent_shape_raw) != 3
        or any(not _is_int(item) or int(item) <= 0 for item in latent_shape_raw)
    ):
        raise RenderError(E_CONFIG_INVALID, "config.latent_shape must contain three positive integers")
    latent_shape = tuple(int(item) for item in latent_shape_raw)

    embedding_dim = _require_int(raw, "embedding_dim", "config")
    if embedding_dim <= 0:
        raise RenderError(E_CONFIG_INVALID, "config.embedding_dim must be positive")

    allowed_fixed = _require_int_list(raw, "allowed_fixed_point_bits", "config")
    if any(item <= 0 or item > 24 for item in allowed_fixed) or len(set(allowed_fixed)) != len(allowed_fixed):
        raise RenderError(E_CONFIG_INVALID, "allowed_fixed_point_bits must be unique values in 1..24")
    fixed_point_bits = _require_int(raw, "fixed_point_bits", "config")
    if fixed_point_bits not in allowed_fixed:
        raise RenderError(
            E_CONFIG_INVALID,
            "fixed_point_bits is not in allowed_fixed_point_bits",
            {"fixed_point_bits": fixed_point_bits, "allowed": list(allowed_fixed)},
        )

    allowed_quant = _require_string_list(raw, "allowed_quantization_levels", "config")
    if any(level not in QUANTIZATION_STEPS for level in allowed_quant):
        raise RenderError(E_CONFIG_INVALID, "allowed_quantization_levels contains an unknown level")
    quantization_level = _require_string(raw, "quantization_level", "config")
    if quantization_level not in allowed_quant:
        raise RenderError(
            E_CONFIG_INVALID,
            "quantization_level is not in allowed_quantization_levels",
            {"quantization_level": quantization_level, "allowed": list(allowed_quant)},
        )
    quantization_step = QUANTIZATION_STEPS[quantization_level]
    if quantization_step > (1 << fixed_point_bits):
        raise RenderError(E_CONFIG_INVALID, "quantization step exceeds fixed-point scale")

    latency_budget = _require_int(raw, "interaction_latency_budget_ms", "config")
    if latency_budget < 0:
        raise RenderError(E_CONFIG_INVALID, "interaction_latency_budget_ms must be non-negative")
    worker_count = _require_int(raw, "worker_count", "config")
    if worker_count <= 0:
        raise RenderError(E_CONFIG_INVALID, "worker_count must be a positive integer")
    snapshot_interval = _require_int(raw, "snapshot_interval_commands", "config")
    if snapshot_interval <= 0:
        raise RenderError(E_CONFIG_INVALID, "snapshot_interval_commands must be a positive integer")
    tile_budget = _require_int(raw, "recompute_tile_budget", "config")
    if tile_budget <= 0:
        raise RenderError(E_CONFIG_INVALID, "recompute_tile_budget must be a positive integer")
    initial_skip = _require_int(raw, "initial_skip_steps", "config")
    if initial_skip < 0 or initial_skip > steps:
        raise RenderError(E_CONFIG_INVALID, "initial_skip_steps must be between 0 and steps")

    noise_versions = _require_string_list(raw, "noise_table_versions", "config")
    noise_spec = require_mapping(raw.get("noise_table"), "config.noise_table")
    noise_version = _require_string(noise_spec, "version", "config.noise_table")
    if noise_version not in noise_versions:
        raise RenderError(
            E_NOISE_TABLE_MISMATCH,
            "configured noise table version is not declared",
            {"version": noise_version, "declared": list(noise_versions)},
        )
    noise_path_text = _require_string(noise_spec, "path", "config.noise_table")
    noise_path = resolve_relative(base_dir, noise_path_text)
    noise_table, noise_shape, noise_version_in_file = _load_noise_table(noise_path)
    if noise_version_in_file != noise_version:
        raise RenderError(
            E_NOISE_TABLE_MISMATCH,
            "noise table file version does not match configuration",
            {"configured": noise_version, "file": noise_version_in_file},
        )
    declared_noise_shape = noise_spec.get("shape")
    if declared_noise_shape is not None:
        if (
            not isinstance(declared_noise_shape, list)
            or len(declared_noise_shape) != 5
            or any(not _is_int(item) or int(item) <= 0 for item in declared_noise_shape)
        ):
            raise RenderError(E_CONFIG_INVALID, "config.noise_table.shape must contain five positive integers")
        if tuple(int(item) for item in declared_noise_shape) != noise_shape:
            raise RenderError(
                E_NOISE_TABLE_MISMATCH,
                "configured noise table shape does not match file descriptor",
                {"configured": declared_noise_shape, "file": list(noise_shape)},
            )
    if len(noise_shape) != 5 or tuple(noise_shape[2:]) != latent_shape:
        raise RenderError(
            E_NOISE_TABLE_MISMATCH,
            "noise table latent dimensions do not match latent_shape",
            {"noise_shape": list(noise_shape), "latent_shape": list(latent_shape)},
        )
    if steps > noise_shape[0]:
        raise RenderError(
            E_CONFIG_INVALID,
            "steps exceed noise table length",
            {"steps": steps, "noise_steps": noise_shape[0]},
        )
    if frame_count > noise_shape[1]:
        raise RenderError(
            E_CONFIG_INVALID,
            "frame_count exceeds noise table frame capacity",
            {"frame_count": frame_count, "noise_frames": noise_shape[1]},
        )

    vocab_spec = require_mapping(raw.get("vocabulary"), "config.vocabulary")
    vocab_path_text = _require_string(vocab_spec, "path", "config.vocabulary")
    vocab_path = resolve_relative(base_dir, vocab_path_text)
    vocabulary, vocabulary_hash = _load_vocabulary(vocab_path)

    initial_spec, initial_image = _validate_initial_image_spec(raw, base_dir, latent_shape)  # type: ignore[arg-type]

    projection_seed = _require_int(raw, "projection_seed", "config")
    condition_seed = _require_int(raw, "condition_seed", "config")
    output_gain = _require_int(raw, "output_gain", "config")
    if output_gain <= 0 or output_gain > 64:
        raise RenderError(E_CONFIG_INVALID, "output_gain must be in 1..64")

    return RenderConfig(
        raw=raw,
        base_dir=base_dir,
        source_hash=hash_object(raw, domain="fixedvideo-source-config"),
        name=name,
        seed_min=seed_min,
        seed_max=seed_max,
        seed=seed,
        default_prompt=default_prompt,
        steps=steps,
        guidance=guidance,
        frame_count=frame_count,
        height=height,
        width=width,
        chunk_rows=chunk_rows,
        latent_shape=latent_shape,  # type: ignore[arg-type]
        embedding_dim=embedding_dim,
        fixed_point_bits=fixed_point_bits,
        allowed_fixed_point_bits=allowed_fixed,
        quantization_level=quantization_level,
        allowed_quantization_levels=allowed_quant,
        quantization_step=quantization_step,
        interaction_latency_budget_ms=latency_budget,
        worker_count=worker_count,
        snapshot_interval_commands=snapshot_interval,
        recompute_tile_budget=tile_budget,
        initial_skip_steps=initial_skip,
        noise_table_version=noise_version,
        noise_table_versions=noise_versions,
        noise_table_path=noise_path,
        noise_table=noise_table,
        noise_shape=noise_shape,
        noise_version_in_file=noise_version_in_file,
        vocabulary_path=vocab_path,
        vocabulary=vocabulary,
        vocabulary_hash=vocabulary_hash,
        initial_image_spec=initial_spec,
        initial_image=initial_image,
        projection_seed=projection_seed,
        condition_seed=condition_seed,
        output_gain=output_gain,
    )


def validate_command_shape(command: Any, index: int) -> dict[str, Any]:
    where = f"commands[{index}]"
    if not isinstance(command, dict):
        raise RenderError(E_CONFIG_INVALID, f"{where} must be an object")
    command_id = command.get("command_id")
    if not isinstance(command_id, str) or not command_id:
        raise RenderError(E_CONFIG_INVALID, f"{where}.command_id must be a non-empty string")
    op = command.get("op")
    if not isinstance(op, str) or op not in SUPPORTED_COMMANDS:
        raise RenderError(
            E_CONFIG_INVALID,
            f"{where}.op is unsupported",
            {"op": op, "supported": sorted(SUPPORTED_COMMANDS)},
        )

    if op == "set_prompt":
        if not isinstance(command.get("prompt"), str):
            raise RenderError(E_CONFIG_INVALID, f"{where}.prompt must be a string")
    elif op == "set_seed":
        if not _is_int(command.get("seed")):
            raise RenderError(E_CONFIG_INVALID, f"{where}.seed must be an integer")
    elif op == "set_guidance":
        if not _is_number(command.get("guidance")):
            raise RenderError(E_CONFIG_INVALID, f"{where}.guidance must be a number")
    elif op in {"set_steps", "set_frame_count", "set_chunk_rows"}:
        key = op.removeprefix("set_")
        if not _is_int(command.get(key)):
            raise RenderError(E_CONFIG_INVALID, f"{where}.{key} must be an integer")
    elif op == "set_quantization_level":
        if not isinstance(command.get("quantization_level"), str):
            raise RenderError(E_CONFIG_INVALID, f"{where}.quantization_level must be a string")
    elif op == "set_initial_image":
        if "path" not in command:
            raise RenderError(E_CONFIG_INVALID, f"{where}.path is required")
        if command["path"] is not None and not isinstance(command["path"], str):
            raise RenderError(E_CONFIG_INVALID, f"{where}.path must be a string or null")
    elif op == "remove_patch":
        if not isinstance(command.get("patch_id"), str) or not command["patch_id"]:
            raise RenderError(E_CONFIG_INVALID, f"{where}.patch_id must be a non-empty string")
    elif op == "add_patch":
        patch = command.get("patch")
        if not isinstance(patch, dict):
            raise RenderError(E_CONFIG_INVALID, f"{where}.patch must be an object")
        if not isinstance(patch.get("patch_id"), str) or not patch["patch_id"]:
            raise RenderError(E_CONFIG_INVALID, f"{where}.patch.patch_id must be a non-empty string")
        for key in ("step", "frame", "x", "y", "width", "height"):
            if not _is_int(patch.get(key)):
                raise RenderError(E_CONFIG_INVALID, f"{where}.patch.{key} must be an integer")
        if not _is_number(patch.get("delta")):
            raise RenderError(E_CONFIG_INVALID, f"{where}.patch.delta must be a number")
    return command


def load_command_stream(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    resolved = Path(path).resolve()
    try:
        raw = read_json(resolved)
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(E_CONFIG_INVALID, f"cannot read command stream: {resolved}", {"error": str(exc)}) from exc
    if not isinstance(raw, dict):
        raise RenderError(E_CONFIG_INVALID, "command stream root must be an object")
    commands_raw = raw.get("commands")
    if not isinstance(commands_raw, list):
        raise RenderError(E_CONFIG_INVALID, "command stream commands must be an array")
    commands: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, command in enumerate(commands_raw):
        validated = validate_command_shape(command, index)
        command_id = validated["command_id"]
        if command_id in seen_ids:
            raise RenderError(
                E_CONFIG_INVALID,
                "command_id values must be unique",
                {"command_id": command_id, "index": index},
            )
        seen_ids.add(command_id)
        commands.append(validated)
    return commands, hash_object(raw, domain="fixedvideo-command-stream")
