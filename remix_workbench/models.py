from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fingerprint import array_fingerprint, digest_json

V1_TRACK_ORDER: tuple[str, ...] = ("bass", "drums", "vocal")
DEFAULT_TRACKS: tuple[str, ...] = V1_TRACK_ORDER
FORMAT_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")


@dataclass
class TrackIssue:
    clip_id: str
    track: str
    stage: str
    code: str
    message: str
    injection_id: str | None = None
    fragment_index: int | None = None

    @property
    def stable_key(self) -> tuple:
        stage_rank = {"parse": 0, "convert": 1, "gain": 2, "mix": 3}.get(self.stage, 9)
        return (
            stage_rank,
            self.clip_id,
            -1 if self.fragment_index is None else self.fragment_index,
            self.track,
            self.code,
            self.injection_id or "",
        )

    @property
    def stable_key_text(self) -> str:
        stage_rank = {"parse": 0, "convert": 1, "gain": 2, "mix": 3}.get(self.stage, 9)
        fragment = (
            "-" if self.fragment_index is None else f"{self.fragment_index:08d}"
        )
        return "|".join(
            [
                str(stage_rank),
                self.clip_id,
                fragment,
                self.track,
                self.code,
                self.injection_id or "",
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "track": self.track,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
            "fragment_index": self.fragment_index,
        }


@dataclass
class TrackPlane:
    clip_id: str
    format_version: str
    sample_rate: int
    bit_depth: int
    frames: int
    channel_map: dict[str, int]
    tracks: dict[str, np.ndarray]
    track_frames: dict[str, int]
    source_metadata: dict[str, Any] = field(default_factory=dict)
    missing_tracks: list[str] = field(default_factory=list)
    issues: list[TrackIssue] = field(default_factory=list)
    package_path: str = ""

    def __post_init__(self) -> None:
        self.tracks = {
            name: np.asarray(values, dtype=np.int32) for name, values in self.tracks.items()
        }

    @property
    def track_names(self) -> list[str]:
        return sorted(self.tracks)

    @property
    def source_frame_annotations(self) -> dict[str, Any]:
        return self.source_metadata.get("source_frame_annotations", {})

    @property
    def track_gain_metadata(self) -> dict[str, Any]:
        return self.source_metadata.get("track_gain_metadata", {})

    def content_fingerprint(self) -> str:
        metadata = {
            "clip_id": self.clip_id,
            "format_version": self.format_version,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "frames": self.frames,
            "channel_map": self.channel_map,
            "track_frames": self.track_frames,
            "missing_tracks": self.missing_tracks,
            "source_metadata": self.source_metadata,
            "issues": [issue.as_dict() for issue in self.issues],
            "package_path": self.package_path,
        }
        return array_fingerprint(
            {name: self.tracks[name] for name in self.track_names}, metadata
        )

    def plane_metadata_fingerprint(self) -> str:
        return digest_json(
            {
                "format_version": self.format_version,
                "frames": self.frames,
                "channel_map": self.channel_map,
                "missing_tracks": self.missing_tracks,
                "source_metadata": self.source_metadata,
            }
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "format_version": self.format_version,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "frames": self.frames,
            "channel_map": self.channel_map,
            "track_frames": self.track_frames,
            "track_names": self.track_names,
            "missing_tracks": self.missing_tracks,
            "source_metadata": self.source_metadata,
            "issues": [issue.as_dict() for issue in self.issues],
            "package_path": self.package_path,
        }


@dataclass
class CacheEvent:
    stage: str
    clip_id: str
    track: str | None
    fragment_index: int | None
    cache_key: str
    status: str
    injection_id: str | None = None

    @property
    def stable_key(self) -> tuple:
        stage_rank = {"parse": 0, "convert": 1, "gain": 2, "mix": 3}.get(self.stage, 9)
        return (
            stage_rank,
            self.clip_id,
            -1 if self.fragment_index is None else self.fragment_index,
            self.track or "",
            self.cache_key,
        )

    @property
    def stable_key_text(self) -> str:
        stage_rank = {"parse": 0, "convert": 1, "gain": 2, "mix": 3}.get(self.stage, 9)
        fragment = (
            "-" if self.fragment_index is None else f"{self.fragment_index:08d}"
        )
        return "|".join(
            [
                str(stage_rank),
                self.clip_id,
                fragment,
                self.track or "",
                self.cache_key,
            ]
        )

    def as_dict(self, *, include_status: bool = True) -> dict[str, Any]:
        data = {
            "stable_key": self.stable_key_text,
            "stage": self.stage,
            "clip_id": self.clip_id,
            "track": self.track,
            "fragment_index": self.fragment_index,
            "cache_key": self.cache_key,
            "injection_id": self.injection_id,
        }
        if include_status:
            data["status"] = self.status
        return data
