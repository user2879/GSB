from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest, read_json
from .font import FONT_VERSION

MATCH_METRICS = ("l1", "xor", "cosine", "ink_balance")


@dataclass(frozen=True)
class RecognitionConfig:
    image_width: int = 360
    image_height: int = 220
    allowed_image_modes: tuple[str, ...] = ("L",)
    window_scales: tuple[int, ...] = (10, 12, 16)
    window_step: int = 4
    binarization_threshold: int = 165
    stroke_density_threshold: float = 0.035
    edge_strength_threshold: float = 0.012
    edge_difference_threshold: int = 24
    line_merge_projection_gap: int = 10
    line_connectivity_min_pixels: int = 2
    candidate_score_threshold: float = 0.36
    min_char_height: int = 8
    char_min_width: int = 8
    char_max_width: int = 24
    fragment_merge_gap: int = 4
    fragment_merge_preference: str = "nearest_then_left"
    match_metric: str = "l1"
    top_k_candidates: int = 5
    thread_pool_workers: int = 4
    char_match_point_budget: int = 1200
    snapshot_commit_interval_commands: int = 2
    fault_injections: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not self.allowed_image_modes:
            raise ValueError("allowed_image_modes cannot be empty")
        if not self.window_scales or any(int(scale) <= 0 for scale in self.window_scales):
            raise ValueError("window_scales must be a non-empty positive list")
        if self.window_step <= 0:
            raise ValueError("window_step must be positive")
        if not 0 <= self.binarization_threshold <= 255:
            raise ValueError("binarization_threshold must be in [0, 255]")
        if not 0 <= self.stroke_density_threshold <= 1:
            raise ValueError("stroke_density_threshold must be in [0, 1]")
        if not 0 <= self.edge_strength_threshold <= 1:
            raise ValueError("edge_strength_threshold must be in [0, 1]")
        if self.line_merge_projection_gap < 0:
            raise ValueError("line_merge_projection_gap must be non-negative")
        if self.line_connectivity_min_pixels <= 0:
            raise ValueError("line_connectivity_min_pixels must be positive")
        if not 0 <= self.candidate_score_threshold <= 1:
            raise ValueError("candidate_score_threshold must be in [0, 1]")
        if self.min_char_height <= 0:
            raise ValueError("min_char_height must be positive")
        if self.char_min_width <= 0 or self.char_max_width < self.char_min_width:
            raise ValueError("character width constraints are invalid")
        if self.char_max_width < 2 * self.char_min_width:
            raise ValueError("char_max_width must allow two minimum-width fragments")
        if self.fragment_merge_gap < 0:
            raise ValueError("fragment_merge_gap must be non-negative")
        if self.fragment_merge_preference not in {
            "nearest_then_left",
            "nearest_then_right",
        }:
            raise ValueError("unsupported fragment_merge_preference")
        if self.match_metric not in MATCH_METRICS:
            raise ValueError(f"match_metric must be one of {MATCH_METRICS}")
        if self.top_k_candidates <= 0:
            raise ValueError("top_k_candidates must be positive")
        if self.thread_pool_workers <= 0:
            raise ValueError("thread_pool_workers must be positive")
        if self.char_match_point_budget < 0:
            raise ValueError("char_match_point_budget must be non-negative")
        if self.snapshot_commit_interval_commands <= 0:
            raise ValueError("snapshot_commit_interval_commands must be positive")
        for item in self.fault_injections:
            if not isinstance(item, dict) or not item.get("id") or not item.get("type"):
                raise ValueError("fault injections require id and type")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecognitionConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        data = dict(value)
        data["window_scales"] = tuple(int(v) for v in data.get("window_scales", (10, 12, 16)))
        data["allowed_image_modes"] = tuple(
            str(v).upper() for v in data.get("allowed_image_modes", ("L",))
        )
        data["fault_injections"] = tuple(data.get("fault_injections", ()))
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "RecognitionConfig":
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("config root must be an object")
        return cls.from_dict(value)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, tuple):
                value = list(value)
            result[name] = value
        return result

    def with_updates(self, **updates: Any) -> "RecognitionConfig":
        if "window_scales" in updates:
            updates["window_scales"] = tuple(int(v) for v in updates["window_scales"])
        if "fault_injections" in updates:
            updates["fault_injections"] = tuple(updates["fault_injections"])
        return replace(self, **updates)

    @property
    def detection_params(self) -> dict[str, Any]:
        return {
            "window_scales": list(self.window_scales),
            "window_step": self.window_step,
            "binarization_threshold": self.binarization_threshold,
            "stroke_density_threshold": self.stroke_density_threshold,
            "edge_strength_threshold": self.edge_strength_threshold,
            "edge_difference_threshold": self.edge_difference_threshold,
        }

    @property
    def line_params(self) -> dict[str, Any]:
        return {
            "line_merge_projection_gap": self.line_merge_projection_gap,
            "line_connectivity_min_pixels": self.line_connectivity_min_pixels,
            "candidate_score_threshold": self.candidate_score_threshold,
            "min_char_height": self.min_char_height,
        }

    @property
    def segmentation_params(self) -> dict[str, Any]:
        return {
            "binarization_threshold": self.binarization_threshold,
            "min_char_height": self.min_char_height,
            "char_min_width": self.char_min_width,
            "char_max_width": self.char_max_width,
            "fragment_merge_gap": self.fragment_merge_gap,
            "fragment_merge_preference": self.fragment_merge_preference,
        }

    @property
    def recognition_params(self) -> dict[str, Any]:
        return {
            "match_metric": self.match_metric,
            "top_k_candidates": self.top_k_candidates,
            "font_version": FONT_VERSION,
        }

    @property
    def fingerprint(self) -> str:
        return digest(self.as_dict())

    @property
    def config_hash(self) -> str:
        return digest(
            {
                "config": self.as_dict(),
                "font_version": FONT_VERSION,
            }
        )


def apply_config_update(config: RecognitionConfig, command: dict[str, Any]) -> RecognitionConfig:
    kind = command.get("type")
    payload = command.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("command payload must be an object")
    if kind == "set_threshold":
        value = payload.get("value", payload.get("binarization_threshold"))
        if value is None:
            raise ValueError("set_threshold requires value")
        return config.with_updates(binarization_threshold=int(value))
    if kind == "set_window":
        updates: dict[str, Any] = {}
        if "window_scales" in payload:
            updates["window_scales"] = tuple(int(v) for v in payload["window_scales"])
        if "window_step" in payload:
            updates["window_step"] = int(payload["window_step"])
        if not updates:
            raise ValueError("set_window requires window_scales or window_step")
        return config.with_updates(**updates)
    if kind == "set_line_merge":
        updates = {}
        for name in (
            "line_merge_projection_gap",
            "line_connectivity_min_pixels",
            "candidate_score_threshold",
        ):
            if name in payload:
                updates[name] = payload[name]
        if not updates:
            raise ValueError("set_line_merge requires at least one merge parameter")
        return config.with_updates(**updates)
    if kind == "set_char_width_constraints":
        updates = {}
        for name in ("char_min_width", "char_max_width", "fragment_merge_gap"):
            if name in payload:
                updates[name] = int(payload[name])
        if not updates:
            raise ValueError("set_char_width_constraints requires width constraints")
        return config.with_updates(**updates)
    if kind == "set_match_metric":
        metric = payload.get("value", payload.get("match_metric"))
        return config.with_updates(match_metric=str(metric))
    if kind == "set_point_budget":
        value = payload.get("value", payload.get("char_match_point_budget"))
        return config.with_updates(char_match_point_budget=int(value))
    if kind == "set_snapshot_interval":
        value = payload.get(
            "value", payload.get("snapshot_commit_interval_commands")
        )
        return config.with_updates(snapshot_commit_interval_commands=int(value))
    if kind in {"replace_image", "delete_image"}:
        return config
    raise ValueError(f"unsupported command type: {kind}")
