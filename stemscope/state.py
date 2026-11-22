from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RemixConfig
from .model import ParsedPackage, Track


@dataclass
class ClipRuntime:
    clip_id: str
    package: ParsedPackage
    tracks: dict[str, Track]
    frames: int
    sample_rate: int

    @classmethod
    def from_package(cls, clip_id: str, package: ParsedPackage) -> "ClipRuntime":
        return cls(
            clip_id=clip_id,
            package=package,
            tracks=dict(package.tracks),
            frames=package.frames,
            sample_rate=package.sample_rate,
        )

    def clone(self) -> "ClipRuntime":
        return ClipRuntime(
            clip_id=self.clip_id,
            package=self.package,
            tracks=dict(self.tracks),
            frames=self.frames,
            sample_rate=self.sample_rate,
        )

    def with_track(self, track: Track) -> "ClipRuntime":
        updated_tracks = dict(self.tracks)
        updated_tracks[track.name] = track
        package = self.package.clone()
        package.tracks = updated_tracks
        package.recompute_fingerprint()
        package.parse_key = ""
        return ClipRuntime(
            clip_id=self.clip_id,
            package=package,
            tracks=updated_tracks,
            frames=self.frames,
            sample_rate=self.sample_rate,
        )

    def with_package(self, package: ParsedPackage) -> "ClipRuntime":
        return ClipRuntime.from_package(self.clip_id, package)


@dataclass
class WorkbenchState:
    config: RemixConfig
    clips: dict[str, ClipRuntime]
    param_version: int = 0
    format_history: list[str] = field(default_factory=list)
    conversion_records: list[dict[str, Any]] = field(default_factory=list)
    replacement_records: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "WorkbenchState":
        return WorkbenchState(
            config=self.config.clone(),
            clips={clip_id: clip.clone() for clip_id, clip in self.clips.items()},
            param_version=self.param_version,
            format_history=list(self.format_history),
            conversion_records=[dict(record) for record in self.conversion_records],
            replacement_records=[dict(record) for record in self.replacement_records],
            failures=[dict(item) for item in self.failures],
        )

    @property
    def format_versions(self) -> list[str]:
        values = set(self.format_history)
        values.add(self.config.format_version)
        return sorted(values)

    def to_info(self) -> dict[str, Any]:
        return {
            "param_version": self.param_version,
            "format_version": self.config.format_version,
            "format_history": list(self.format_history),
            "config_hash": self.config.config_hash(),
            "remix_fingerprint": self.config.remix_fingerprint(),
            "clips": {
                clip_id: {
                    "frames": clip.frames,
                    "sample_rate": clip.sample_rate,
                    "package_fingerprint": clip.package.package_fingerprint,
                    "tracks": {
                        name: {
                            "frames": track.frames,
                            "max_abs_amplitude": track.max_abs_amplitude,
                            "content_fingerprint": track.content_fingerprint(),
                        }
                        for name, track in sorted(clip.tracks.items())
                    },
                }
                for clip_id, clip in sorted(self.clips.items())
            },
            "conversion_records": list(self.conversion_records),
            "replacement_records": list(self.replacement_records),
        }
