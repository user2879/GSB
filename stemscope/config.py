from __future__ import annotations

import copy
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .hashing import canonical_json_bytes, sha256_bytes


def _round_half_up(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


@dataclass
class TrackSettings:
    gain: float = 1.0
    muted: bool = False
    start_ms: float = 0.0
    end_ms: float = 0.0

    def validate(self, track_name: str) -> None:
        if self.gain < 0 or self.gain > 4:
            raise DiagnosticError(
                "E_CONFIG_INVALID",
                f"track {track_name} gain must be in [0, 4]",
                stage="config",
                track=track_name,
            )
        if self.start_ms < 0 or self.end_ms < 0:
            raise DiagnosticError(
                "E_CONFIG_INVALID",
                f"track {track_name} crop values must be non-negative",
                stage="config",
                track=track_name,
            )
        if self.end_ms < self.start_ms:
            raise DiagnosticError(
                "E_CONFIG_INVALID",
                f"track {track_name} end_ms precedes start_ms",
                stage="config",
                track=track_name,
            )

    def crop_frames(self, sample_rate: int, total_frames: int) -> tuple[int, int]:
        start = _round_half_up(Decimal(str(self.start_ms)) * Decimal(sample_rate) / Decimal(1000))
        end = _round_half_up(Decimal(str(self.end_ms)) * Decimal(sample_rate) / Decimal(1000))
        return max(0, min(start, total_frames)), max(0, min(end, total_frames))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain": self.gain,
            "muted": self.muted,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    def clone(self) -> "TrackSettings":
        return TrackSettings(
            gain=float(self.gain),
            muted=bool(self.muted),
            start_ms=float(self.start_ms),
            end_ms=float(self.end_ms),
        )


@dataclass
class RemixConfig:
    sample_rate: int = 16000
    gain_quantization_steps: int = 1000
    render_frames: int = 2048
    workers: int = 4
    interaction_budget_ms: float = 40.0
    snapshot_interval_segments: int = 4
    format_version: str = "v1"
    tracks: dict[str, TrackSettings] = field(default_factory=dict)
    simulated_compute_ms: float = 0.0
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RemixConfig":
        track_values = value.get("tracks", {})
        if not isinstance(track_values, dict) or not track_values:
            raise DiagnosticError(
                "E_CONFIG_INVALID",
                "config.tracks must be a non-empty object",
                stage="config",
            )
        tracks: dict[str, TrackSettings] = {}
        for name, settings in track_values.items():
            if not isinstance(name, str) or not name:
                raise DiagnosticError("E_CONFIG_INVALID", "track names must be non-empty strings", stage="config")
            if not isinstance(settings, dict):
                raise DiagnosticError("E_CONFIG_INVALID", f"track {name} settings must be an object", stage="config")
            tracks[name] = TrackSettings(
                gain=float(settings.get("gain", 1.0)),
                muted=bool(settings.get("muted", False)),
                start_ms=float(settings.get("start_ms", 0.0)),
                end_ms=float(settings.get("end_ms", 0.0)),
            )
        config = cls(
            sample_rate=int(value.get("sample_rate", 16000)),
            gain_quantization_steps=int(value.get("gain_quantization_steps", 1000)),
            render_frames=int(value.get("render_frames", 2048)),
            workers=int(value.get("workers", 4)),
            interaction_budget_ms=float(value.get("interaction_budget_ms", 40.0)),
            snapshot_interval_segments=int(value.get("snapshot_interval_segments", 4)),
            format_version=str(value.get("format_version", "v1")),
            tracks=tracks,
            simulated_compute_ms=float(value.get("simulated_compute_ms", 0.0)),
            schema_version=int(value.get("schema_version", 1)),
        )
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "RemixConfig":
        import json

        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise DiagnosticError("E_CONFIG_INVALID", "sample_rate must be positive", stage="config")
        if self.gain_quantization_steps <= 0:
            raise DiagnosticError(
                "E_CONFIG_INVALID", "gain_quantization_steps must be positive", stage="config"
            )
        if self.render_frames <= 0:
            raise DiagnosticError("E_CONFIG_INVALID", "render_frames must be positive", stage="config")
        if self.workers <= 0:
            raise DiagnosticError("E_CONFIG_INVALID", "workers must be positive", stage="config")
        if self.interaction_budget_ms <= 0:
            raise DiagnosticError(
                "E_CONFIG_INVALID", "interaction_budget_ms must be positive", stage="config"
            )
        if self.snapshot_interval_segments <= 0:
            raise DiagnosticError(
                "E_CONFIG_INVALID", "snapshot_interval_segments must be positive", stage="config"
            )
        if self.format_version not in {"v1", "v2", "v3"}:
            raise DiagnosticError("E_CONFIG_INVALID", f"unknown format {self.format_version}", stage="config")
        for name, settings in self.tracks.items():
            settings.validate(name)

    def quantized_gain_int(self, track_name: str) -> int:
        settings = self.tracks[track_name]
        value = Decimal(str(settings.gain)) * Decimal(self.gain_quantization_steps)
        return _round_half_up(value)

    def effective_gain(self, track_name: str) -> float:
        return self.quantized_gain_int(track_name) / self.gain_quantization_steps

    def track_param_fingerprint(self, track_name: str) -> str:
        settings = self.tracks[track_name]
        payload = {
            "gain_int": self.quantized_gain_int(track_name),
            "gain_source": str(settings.gain),
            "muted": settings.muted,
            "start_ms": str(settings.start_ms),
            "end_ms": str(settings.end_ms),
            "quantization_steps": self.gain_quantization_steps,
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def remix_fingerprint(self) -> str:
        payload = {
            "tracks": {name: self.track_param_fingerprint(name) for name in sorted(self.tracks)},
            "quantization_steps": self.gain_quantization_steps,
            "render_frames": self.render_frames,
            "sample_rate": self.sample_rate,
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def config_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_rate": self.sample_rate,
            "gain_quantization_steps": self.gain_quantization_steps,
            "render_frames": self.render_frames,
            "workers": self.workers,
            "interaction_budget_ms": self.interaction_budget_ms,
            "snapshot_interval_segments": self.snapshot_interval_segments,
            "format_version": self.format_version,
            "tracks": {name: settings.to_dict() for name, settings in sorted(self.tracks.items())},
            "simulated_compute_ms": self.simulated_compute_ms,
        }

    def clone(self) -> "RemixConfig":
        return RemixConfig(
            sample_rate=self.sample_rate,
            gain_quantization_steps=self.gain_quantization_steps,
            render_frames=self.render_frames,
            workers=self.workers,
            interaction_budget_ms=self.interaction_budget_ms,
            snapshot_interval_segments=self.snapshot_interval_segments,
            format_version=self.format_version,
            tracks={name: settings.clone() for name, settings in self.tracks.items()},
            simulated_compute_ms=self.simulated_compute_ms,
            schema_version=self.schema_version,
        )

    def apply_command(self, command: dict[str, Any]) -> "RemixConfig":
        updated = self.clone()
        command_type = command.get("type")
        track = command.get("track")
        if command_type in {"set_gain", "set_mute", "set_crop"}:
            if not isinstance(track, str) or track not in updated.tracks:
                raise DiagnosticError(
                    "E_CONFIG_INVALID",
                    f"command references unknown track {track!r}",
                    stage="interaction",
                    details={"command": command},
                )
        if command_type == "set_gain":
            updated.tracks[track].gain = float(command["gain"])
        elif command_type == "set_mute":
            updated.tracks[track].muted = bool(command["muted"])
        elif command_type == "set_crop":
            updated.tracks[track].start_ms = float(command["start_ms"])
            updated.tracks[track].end_ms = float(command["end_ms"])
        elif command_type == "set_quantization":
            updated.gain_quantization_steps = int(command["steps"])
        elif command_type == "switch_format":
            updated.format_version = str(command["format"])
        elif command_type not in {"replace_track", "noop"}:
            raise DiagnosticError(
                "E_CONFIG_INVALID",
                f"unknown interaction command {command_type!r}",
                stage="interaction",
                details={"command": command},
            )
        updated.validate()
        return updated

    def copy_without_runtime(self) -> "RemixConfig":
        value = copy.deepcopy(self)
        value.simulated_compute_ms = 0.0
        return value
