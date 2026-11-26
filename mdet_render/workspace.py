"""Editable workspace state and transactional edit-command application."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .conditions import (
    MotionTable,
    PromptSpec,
    Vocabulary,
    load_image_condition,
    load_motion_table,
    load_prompt,
    load_vocabulary,
    parse_prompt,
)
from .config import FrameSpec, RenderConfig, load_render_config
from .errors import BusinessError
from .fixed import ALLOWED_FRACTION_BITS, ALLOWED_QUANT_LEVELS, ALLOWED_STORAGE_BITS, FixedPoint
from .util import read_json_file


@dataclass
class FrameInput:
    frame: int
    prompt: PromptSpec
    image_path: Path | None
    prompt_source: str


@dataclass
class Workspace:
    config: RenderConfig
    vocabulary: Vocabulary
    motion: MotionTable
    frames: dict[int, FrameInput]

    @classmethod
    def load(cls, config_path: str | Path) -> "Workspace":
        config = load_render_config(config_path)
        vocabulary = load_vocabulary(config.vocabulary_path, config.condition_dim)
        motion = load_motion_table(config.motion_path, config.motion_block_rows,
                                   config.motion_block_cols, config.condition_dim)
        frames: dict[int, FrameInput] = {}
        for spec in config.frames:
            prompt = load_prompt(spec.prompt_path, vocabulary)
            if spec.image_path is not None:
                load_image_condition(spec.image_path, config.width, config.height, config.pixel_mode)
            frames[spec.frame] = FrameInput(frame=spec.frame, prompt=prompt, image_path=spec.image_path,
                                            prompt_source=str(spec.prompt_path))
        workspace = cls(config=config, vocabulary=vocabulary, motion=motion, frames=frames)
        workspace.validate_assets()
        return workspace

    def clone(self) -> "Workspace":
        return copy.deepcopy(self)

    def frame(self, index: int) -> FrameInput:
        try:
            return self.frames[int(index)]
        except (KeyError, ValueError, TypeError) as exc:
            raise BusinessError("E_FRAME", "unknown frame index", {"frame": index,
                                                                   "known": sorted(self.frames)}) from exc

    def validate_assets(self) -> None:
        self.config.validate()
        if self.vocabulary.dimension != self.config.condition_dim:
            raise BusinessError("E_VECTOR_DIM", "vocabulary no longer matches condition_dim",
                                {"vocabulary_dim": self.vocabulary.dimension,
                                 "condition_dim": self.config.condition_dim})
        if self.motion.block_rows != self.config.motion_block_rows:
            raise BusinessError("E_MOTION_ROWS", "motion table row count changed incompatibly")
        if self.motion.block_cols != self.config.motion_block_cols:
            raise BusinessError("E_MOTION_ROWS", "motion table column count changed incompatibly")
        motion_dim = self.motion.block_rows * self.motion.block_cols * self.motion.vector_dim
        if motion_dim != self.config.condition_dim:
            raise BusinessError("E_MOTION_DIM", "motion vector table dimension is incompatible with config",
                                {"motion_dim": motion_dim, "condition_dim": self.config.condition_dim})
        for index in sorted(self.frames):
            frame = self.frames[index]
            if frame.frame != index:
                raise BusinessError("E_FRAME", "frame map key mismatch", {"key": index, "frame": frame.frame})
            if frame.image_path is not None:
                load_image_condition(frame.image_path, self.config.width, self.config.height,
                                     self.config.pixel_mode)

    def image_for(self, frame: int):
        item = self.frame(frame)
        if item.image_path is None:
            return None
        return load_image_condition(item.image_path, self.config.width, self.config.height,
                                    self.config.pixel_mode)

    def command_snapshot(self) -> dict[str, Any]:
        return {
            "weights": dict(sorted(self.config.weights.items())),
            "denoise_steps": self.config.denoise_steps,
            "decode_block_rows": self.config.decode_block_rows,
            "quantization": self.config.fixed.as_dict(),
            "lease_ttl_ms": self.config.lease_ttl_ms,
            "point_budget": self.config.point_budget,
            "max_workers": self.config.max_workers,
            "guidance_strength": self.config.guidance_strength,
            "frames": {
                str(index): {
                    "prompt": item.prompt.as_dict(),
                    "image": None if item.image_path is None else str(item.image_path),
                }
                for index, item in sorted(self.frames.items())
            },
        }

    def apply_command(self, command: dict[str, Any]) -> "Workspace":
        if not isinstance(command, dict):
            raise BusinessError("E_COMMAND_SCHEMA", "command must be a JSON object")
        clone = self.clone()
        clone._apply_command_in_place(command)
        clone.validate_assets()
        return clone

    def _require_frame(self, command: dict[str, Any]) -> FrameInput:
        if "frame" not in command:
            raise BusinessError("E_COMMAND_SCHEMA", "command requires frame", {"command": command})
        value = command["frame"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise BusinessError("E_COMMAND_SCHEMA", "frame must be an integer",
                                {"frame": value})
        return self.frame(value)

    def _apply_command_in_place(self, command: dict[str, Any]) -> None:
        op = command.get("op")
        if not isinstance(op, str):
            raise BusinessError("E_COMMAND_SCHEMA", "command requires a string op", {"command": command})
        config = self.config
        if op == "replace_prompt":
            frame = self._require_frame(command)
            if "prompt" in command:
                prompt_payload = command["prompt"]
                prompt_id = (str(prompt_payload.get("prompt_id", f"frame-{frame.frame}"))
                             if isinstance(prompt_payload, dict) else f"frame-{frame.frame}")
                prompt = parse_prompt(prompt_payload, self.vocabulary,
                                      prompt_id=prompt_id, source="<command-inline>")
                source = "<command-inline>"
            elif "path" in command or "prompt_path" in command:
                value = command.get("path", command.get("prompt_path"))
                if not isinstance(value, str) or not value:
                    raise BusinessError("E_COMMAND_SCHEMA", "replace_prompt path must be a non-empty string")
                prompt_path = Path(value)
                if not prompt_path.is_absolute():
                    prompt_path = config.base_dir / prompt_path
                prompt = load_prompt(prompt_path, self.vocabulary)
                source = str(prompt_path)
            else:
                raise BusinessError("E_COMMAND_SCHEMA", "replace_prompt requires prompt or path")
            frame.prompt = prompt
            frame.prompt_source = source
            return
        if op == "replace_image":
            frame = self._require_frame(command)
            value = command.get("path", command.get("image_path"))
            if not isinstance(value, str) or not value:
                raise BusinessError("E_COMMAND_SCHEMA", "replace_image requires a non-empty path")
            image_path = Path(value)
            if not image_path.is_absolute():
                image_path = config.base_dir / image_path
            load_image_condition(image_path, config.width, config.height, config.pixel_mode)
            frame.image_path = image_path
            return
        if op == "delete_image":
            frame = self._require_frame(command)
            frame.image_path = None
            return
        if op == "set_weight":
            name = command.get("condition", command.get("name"))
            if name not in {"text", "image", "motion"}:
                raise BusinessError("E_COMMAND_SCHEMA", "set_weight condition must be text, image or motion",
                                    {"condition": name})
            value = _number(command.get("weight", command.get("value")), "weight", minimum=0.0, maximum=8.0)
            config.weights[str(name)] = value
            return
        if op == "set_denoise_steps":
            value = _integer(command.get("steps", command.get("value")), "steps", minimum=1,
                             maximum=config.max_denoise_steps)
            config.denoise_steps = value
            return
        if op == "set_decode_block_rows":
            value = _integer(command.get("block_rows", command.get("value")), "block_rows", minimum=1,
                             maximum=config.height)
            config.decode_block_rows = value
            return
        if op == "set_quantization":
            fraction_bits = _integer(command.get("fraction_bits", config.fixed.fraction_bits),
                                     "fraction_bits")
            storage_bits = _integer(command.get("storage_bits", config.fixed.storage_bits), "storage_bits")
            quant_level = _integer(command.get("quant_level", config.fixed.quant_level), "quant_level")
            fixed = FixedPoint(fraction_bits=fraction_bits, storage_bits=storage_bits,
                               quant_level=quant_level)
            fixed.validate()
            if fraction_bits not in ALLOWED_FRACTION_BITS:
                raise BusinessError("E_ENUM_FRACTION_BITS", "fraction_bits is outside the allowed enum",
                                    {"allowed": list(ALLOWED_FRACTION_BITS)})
            if storage_bits not in ALLOWED_STORAGE_BITS:
                raise BusinessError("E_ENUM_STORAGE_BITS", "storage_bits is outside the allowed enum",
                                    {"allowed": list(ALLOWED_STORAGE_BITS)})
            if quant_level not in ALLOWED_QUANT_LEVELS:
                raise BusinessError("E_ENUM_QUANT_LEVEL", "quant_level is outside the allowed enum",
                                    {"allowed": list(ALLOWED_QUANT_LEVELS)})
            config.fixed = fixed
            return
        if op == "set_lease":
            value = _integer(command.get("ttl_ms", command.get("value")), "ttl_ms", minimum=1, maximum=86_400_000)
            config.lease_ttl_ms = value
            return
        if op == "set_point_budget":
            value = _integer(command.get("point_budget", command.get("value")), "point_budget", minimum=1)
            if value < config.denoise_points_per_step:
                raise BusinessError("E_POINT_BUDGET", "point budget must cover one denoise step",
                                    {"point_budget": value, "step_points": config.denoise_points_per_step})
            config.point_budget = value
            return
        if op == "set_max_workers":
            value = _integer(command.get("max_workers", command.get("value")), "max_workers", minimum=1,
                             maximum=32)
            config.max_workers = value
            return
        if op == "set_guidance_strength":
            value = _number(command.get("guidance_strength", command.get("value")), "guidance_strength",
                            minimum=0.0, maximum=8.0)
            config.guidance_strength = value
            return
        raise BusinessError("E_COMMAND_UNKNOWN", "unknown edit command", {"op": op})


def _integer(value: Any, name: str, *, minimum: int | None = None,
             maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} must be an integer", {"field": name})
    if minimum is not None and value < minimum:
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} is below its minimum",
                            {"field": name, "minimum": minimum, "value": value})
    if maximum is not None and value > maximum:
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} is above its maximum",
                            {"field": name, "maximum": maximum, "value": value})
    return value


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} must be a number", {"field": name})
    number = float(value)
    if not math.isfinite(number):
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} must be finite", {"field": name})
    if minimum is not None and number < minimum:
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} is below its minimum",
                            {"field": name, "minimum": minimum, "value": number})
    if maximum is not None and number > maximum:
        raise BusinessError("E_COMMAND_SCHEMA", f"{name} is above its maximum",
                            {"field": name, "maximum": maximum, "value": number})
    return number


def load_commands(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot read command file {path}", {"path": str(path)}) from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        commands = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BusinessError("E_COMMANDS_JSON", "invalid JSONL command",
                                    {"path": str(path), "line": line_number, "error": exc.msg}) from exc
            commands.append(item)
    else:
        if isinstance(parsed, list):
            commands = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("commands"), list):
            commands = parsed["commands"]
        else:
            raise BusinessError("E_COMMANDS_SCHEMA", "command document must be an array or an object with commands",
                                {"path": str(path)})
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise BusinessError("E_COMMAND_SCHEMA", "command must be an object", {"index": index})
    return commands
