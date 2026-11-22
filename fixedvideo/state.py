"""Effective render state and command validation/application."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import QUANTIZATION_STEPS, RenderConfig, load_initial_latent
from .errors import E_CONFIG_INVALID, E_PATCH_OUT_OF_RANGE, E_SEED_OUT_OF_RANGE, RenderError
from .util import hash_object, resolve_relative


@dataclass(frozen=True)
class LatentPatch:
    patch_id: str
    step: int
    frame: int
    x: int
    y: int
    width: int
    height: int
    delta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "step": self.step,
            "frame": self.frame,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "delta": self.delta,
        }

    def fingerprint(self) -> str:
        return hash_object(self.as_dict(), domain="fixedvideo-patch")


@dataclass(frozen=True)
class RenderState:
    prompt: str
    seed: int
    steps: int
    guidance: float
    frame_count: int
    chunk_rows: int
    quantization_level: str
    initial_image_path: str | None
    initial_image_raw_fp: str | None
    patches: tuple[LatentPatch, ...]
    version: int = 0
    command_cursor: int = 0
    last_command_id: str | None = None

    @property
    def quantization_step(self) -> int:
        return QUANTIZATION_STEPS[self.quantization_level]

    def patch_for(self, step: int, frame: int) -> tuple[LatentPatch, ...]:
        return tuple(
            patch
            for patch in self.patches
            if patch.step == step and patch.frame == frame
        )

    def patch_fingerprint(self, step: int, frame: int) -> str:
        patches = self.patch_for(step, frame)
        return hash_object([patch.as_dict() for patch in patches], domain="fixedvideo-patch-set")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance,
            "frame_count": self.frame_count,
            "chunk_rows": self.chunk_rows,
            "quantization_level": self.quantization_level,
            "initial_image_path": self.initial_image_path,
            "initial_image_raw_fp": self.initial_image_raw_fp,
            "patches": [patch.as_dict() for patch in self.patches],
            "version": self.version,
            "command_cursor": self.command_cursor,
            "last_command_id": self.last_command_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RenderState":
        patches = tuple(
            LatentPatch(
                patch_id=str(item["patch_id"]),
                step=int(item["step"]),
                frame=int(item["frame"]),
                x=int(item["x"]),
                y=int(item["y"]),
                width=int(item["width"]),
                height=int(item["height"]),
                delta=float(item["delta"]),
            )
            for item in raw.get("patches", [])
        )
        return cls(
            prompt=str(raw["prompt"]),
            seed=int(raw["seed"]),
            steps=int(raw["steps"]),
            guidance=float(raw["guidance"]),
            frame_count=int(raw["frame_count"]),
            chunk_rows=int(raw["chunk_rows"]),
            quantization_level=str(raw["quantization_level"]),
            initial_image_path=raw.get("initial_image_path"),
            initial_image_raw_fp=raw.get("initial_image_raw_fp"),
            patches=patches,
            version=int(raw.get("version", 0)),
            command_cursor=int(raw.get("command_cursor", 0)),
            last_command_id=raw.get("last_command_id"),
        )


@dataclass(frozen=True)
class CommandEffect:
    affected_frames: tuple[int, ...]
    affected_stages: tuple[str, ...]
    reason: str
    dropped_patch_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "affected_frames": list(self.affected_frames),
            "affected_frame_range": (
                [min(self.affected_frames), max(self.affected_frames)]
                if self.affected_frames
                else []
            ),
            "affected_stages": list(self.affected_stages),
            "reason": self.reason,
            "dropped_patch_ids": list(self.dropped_patch_ids),
        }


def initial_state(config: RenderConfig) -> RenderState:
    return RenderState(
        prompt=config.default_prompt,
        seed=config.seed,
        steps=config.steps,
        guidance=config.guidance,
        frame_count=config.frame_count,
        chunk_rows=config.chunk_rows,
        quantization_level=config.quantization_level,
        initial_image_path=(
            str(config.initial_image.path)
            if config.initial_image is not None
            else None
        ),
        initial_image_raw_fp=(
            config.initial_image.raw_fingerprint
            if config.initial_image is not None
            else None
        ),
        patches=(),
    )


def effective_config_fingerprint(config: RenderConfig, state: RenderState) -> str:
    return hash_object(
        {
            "source_config_hash": config.source_hash,
            "render_state": {
                "prompt": state.prompt,
                "seed": state.seed,
                "steps": state.steps,
                "guidance": state.guidance,
                "frame_count": state.frame_count,
                "chunk_rows": state.chunk_rows,
                "quantization_level": state.quantization_level,
                "initial_image_path": state.initial_image_path,
                "initial_image_raw_fp": state.initial_image_raw_fp,
                "patches": [patch.as_dict() for patch in state.patches],
            },
            "static_model": config.static_model_fingerprint(),
        },
        domain="fixedvideo-effective-config",
    )


def _validate_seed(config: RenderConfig, seed: int) -> int:
    if not config.seed_min <= seed <= config.seed_max:
        raise RenderError(
            E_SEED_OUT_OF_RANGE,
            "seed is outside the configured seed range",
            {"seed": seed, "seed_range": [config.seed_min, config.seed_max]},
        )
    return seed


def _all_frames(state: RenderState) -> tuple[int, ...]:
    return tuple(range(state.frame_count))


def apply_command(
    config: RenderConfig,
    state: RenderState,
    command: dict[str, Any],
    *,
    command_index: int,
) -> tuple[RenderState, CommandEffect]:
    """Apply one already shape-validated command and return an immutable new state."""
    command_id = str(command["command_id"])
    op = str(command["op"])

    if op == "set_prompt":
        new_state = replace(state, prompt=str(command["prompt"]))
        effect = CommandEffect(
            _all_frames(state),
            ("encoding", "initialization", "denoise", "decode", "composition"),
            "prompt changed",
        )
    elif op == "set_seed":
        seed = _validate_seed(config, int(command["seed"]))
        new_state = replace(state, seed=seed)
        effect = CommandEffect(
            _all_frames(state),
            ("initialization", "denoise", "decode", "composition"),
            "seed changed",
        )
    elif op == "set_guidance":
        guidance = float(command["guidance"])
        if not math.isfinite(guidance) or guidance < 0 or guidance > 100:
            raise RenderError(E_CONFIG_INVALID, "guidance must be finite and between 0 and 100")
        new_state = replace(state, guidance=guidance)
        effect = CommandEffect(
            _all_frames(state),
            ("denoise", "decode", "composition"),
            "guidance changed",
        )
    elif op == "set_steps":
        steps = int(command["steps"])
        if steps <= 0:
            raise RenderError(E_CONFIG_INVALID, "steps must be a positive integer")
        if steps > config.max_noise_steps:
            raise RenderError(
                E_CONFIG_INVALID,
                "steps exceed noise table capacity",
                {"steps": steps, "max_steps": config.max_noise_steps},
            )
        if steps < config.initial_skip_steps:
            raise RenderError(
                E_CONFIG_INVALID,
                "steps must be at least initial_skip_steps",
                {"steps": steps, "initial_skip_steps": config.initial_skip_steps},
            )
        new_state = replace(state, steps=steps)
        effect = CommandEffect(
            _all_frames(state),
            ("denoise", "decode", "composition"),
            "denoise steps changed",
        )
    elif op == "set_frame_count":
        frame_count = int(command["frame_count"])
        if frame_count <= 0:
            raise RenderError(E_CONFIG_INVALID, "frame_count must be a positive integer")
        if frame_count > config.max_noise_frames:
            raise RenderError(
                E_CONFIG_INVALID,
                "frame_count exceeds noise table frame capacity",
                {"frame_count": frame_count, "max_frames": config.max_noise_frames},
            )
        dropped = tuple(sorted(p.patch_id for p in state.patches if p.frame >= frame_count))
        kept = tuple(p for p in state.patches if p.frame < frame_count)
        old_count = state.frame_count
        new_state = replace(state, frame_count=frame_count, patches=kept)
        if frame_count >= old_count:
            affected = tuple(range(old_count, frame_count))
        else:
            affected = tuple(range(frame_count))
        effect = CommandEffect(
            affected,
            ("decode", "composition"),
            "frame_count changed",
            dropped_patch_ids=dropped,
        )
    elif op == "set_chunk_rows":
        chunk_rows = int(command["chunk_rows"])
        if chunk_rows <= 0:
            raise RenderError(E_CONFIG_INVALID, "chunk_rows must be a positive integer")
        new_state = replace(state, chunk_rows=chunk_rows)
        effect = CommandEffect(
            _all_frames(state),
            ("decode", "composition"),
            "chunk_rows changed",
        )
    elif op == "set_quantization_level":
        level = str(command["quantization_level"])
        if level not in config.allowed_quantization_levels:
            raise RenderError(
                E_CONFIG_INVALID,
                "quantization_level is not allowed",
                {"quantization_level": level, "allowed": list(config.allowed_quantization_levels)},
            )
        if QUANTIZATION_STEPS[level] > config.scale:
            raise RenderError(E_CONFIG_INVALID, "quantization step exceeds fixed-point scale")
        new_state = replace(state, quantization_level=level)
        effect = CommandEffect(
            _all_frames(state),
            ("initialization", "denoise", "decode", "composition"),
            "quantization_level changed",
        )
    elif op == "set_initial_image":
        path_value = command.get("path")
        if path_value is None:
            new_state = replace(state, initial_image_path=None, initial_image_raw_fp=None)
            affected_stages = ("initialization", "denoise", "decode", "composition")
        else:
            if not isinstance(path_value, str) or not path_value:
                raise RenderError(E_CONFIG_INVALID, "initial image path must be a non-empty string or null")
            resolved = resolve_relative(config.base_dir, Path(path_value))
            initial = load_initial_latent(resolved, expected_shape=config.latent_shape)
            new_state = replace(
                state,
                initial_image_path=str(initial.path),
                initial_image_raw_fp=initial.raw_fingerprint,
            )
            affected_stages = ("image_encoding", "denoise", "decode", "composition")
        effect = CommandEffect(_all_frames(state), affected_stages, "initial image changed")
    elif op == "add_patch":
        raw_patch = command["patch"]
        patch = LatentPatch(
            patch_id=str(raw_patch["patch_id"]),
            step=int(raw_patch["step"]),
            frame=int(raw_patch["frame"]),
            x=int(raw_patch["x"]),
            y=int(raw_patch["y"]),
            width=int(raw_patch["width"]),
            height=int(raw_patch["height"]),
            delta=float(raw_patch["delta"]),
        )
        if patch.step < 0 or patch.step >= state.steps:
            raise RenderError(
                E_CONFIG_INVALID,
                "patch.step is outside the active denoise sequence",
                {"step": patch.step, "steps": state.steps},
            )
        if patch.frame < 0 or patch.frame >= state.frame_count:
            raise RenderError(
                E_CONFIG_INVALID,
                "patch.frame is outside the active frame range",
                {"frame": patch.frame, "frame_count": state.frame_count},
            )
        latent_channels, latent_height, latent_width = config.latent_shape
        if (
            patch.width <= 0
            or patch.height <= 0
            or patch.x < 0
            or patch.y < 0
            or patch.x + patch.width > latent_width
            or patch.y + patch.height > latent_height
        ):
            raise RenderError(
                E_PATCH_OUT_OF_RANGE,
                "patch rectangle is outside the latent spatial bounds",
                {
                    "patch": patch.as_dict(),
                    "latent_spatial_shape": [latent_height, latent_width],
                    "latent_channels": latent_channels,
                },
            )
        if not math.isfinite(patch.delta):
            raise RenderError(E_CONFIG_INVALID, "patch.delta must be finite")
        patches = {item.patch_id: item for item in state.patches}
        patches[patch.patch_id] = patch
        new_state = replace(state, patches=tuple(sorted(patches.values(), key=lambda item: item.patch_id)))
        effect = CommandEffect(
            (patch.frame,),
            ("denoise", "decode", "composition"),
            f"patch {patch.patch_id} added at step {patch.step}",
        )
    elif op == "remove_patch":
        patch_id = str(command["patch_id"])
        remaining = tuple(item for item in state.patches if item.patch_id != patch_id)
        if len(remaining) == len(state.patches):
            raise RenderError(E_CONFIG_INVALID, f"patch does not exist: {patch_id}")
        removed = next(item for item in state.patches if item.patch_id == patch_id)
        new_state = replace(state, patches=remaining)
        effect = CommandEffect(
            (removed.frame,),
            ("denoise", "decode", "composition"),
            f"patch {patch_id} removed from step {removed.step}",
        )
    else:  # pragma: no cover - loader prevents this path
        raise RenderError(E_CONFIG_INVALID, f"unsupported command op: {op}")

    new_state = replace(
        new_state,
        version=state.version + 1,
        command_cursor=command_index + 1,
        last_command_id=command_id,
    )
    return new_state, effect


def touch_rejected_command(state: RenderState, command: dict[str, Any], command_index: int) -> RenderState:
    return replace(
        state,
        command_cursor=command_index + 1,
        last_command_id=str(command.get("command_id", f"command-{command_index}")),
    )
