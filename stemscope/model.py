from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import DiagnosticError
from .hashing import sha256_bytes, sha256_json


CANONICAL_TRACK_ORDER: tuple[str, ...] = ("bass", "drums", "vocals")
FORMAT_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")


@dataclass
class Track:
    name: str
    pcm: np.ndarray
    sample_rate: int
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pcm = np.asarray(self.pcm, dtype="<i2")
        if pcm.ndim != 1:
            raise ValueError(f"track {self.name!r} PCM must be mono")
        self.pcm = np.ascontiguousarray(pcm)

    @property
    def frames(self) -> int:
        return int(self.pcm.shape[0])

    @property
    def max_abs_amplitude(self) -> int:
        if self.frames == 0:
            return 0
        return int(np.max(np.abs(self.pcm.astype(np.int32))))

    def content_fingerprint(self) -> str:
        header = {
            "name": self.name,
            "frames": self.frames,
            "sample_rate": self.sample_rate,
        }
        return sha256_bytes(sha256_json(header).encode("ascii"), self.pcm.tobytes(order="C"))

    def clone(self, *, pcm: np.ndarray | None = None, metadata: dict[str, Any] | None = None) -> "Track":
        return Track(
            name=self.name,
            pcm=np.array(self.pcm if pcm is None else pcm, dtype="<i2", copy=True),
            sample_rate=self.sample_rate,
            source_format=self.source_format,
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def to_info(self, clip_id: str = "") -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self.name,
            "frames": self.frames,
            "max_abs_amplitude": self.max_abs_amplitude,
            "source_format_version": self.source_format,
            "content_fingerprint": self.content_fingerprint(),
        }
        if clip_id:
            info["clip_id"] = clip_id
        if self.metadata:
            info["metadata"] = self.metadata
        return info


@dataclass
class ParsedPackage:
    format_version: str
    sample_rate: int
    tracks: dict[str, Track]
    track_order: tuple[str, ...]
    source_path: str
    sidecar: dict[str, Any] = field(default_factory=dict)
    sidecar_bytes: bytes = b""
    package_fingerprint: str = ""
    source_fingerprint: str = ""
    parse_key: str = ""
    conversion_ledger: list[dict[str, Any]] = field(default_factory=list)

    @property
    def frames(self) -> int:
        if not self.tracks:
            return 0
        return max(track.frames for track in self.tracks.values())

    def track_names_sorted(self) -> list[str]:
        return sorted(self.tracks)

    def clone(self) -> "ParsedPackage":
        return ParsedPackage(
            format_version=self.format_version,
            sample_rate=self.sample_rate,
            tracks={name: track.clone() for name, track in self.tracks.items()},
            track_order=tuple(self.track_order),
            source_path=self.source_path,
            sidecar=dict(self.sidecar),
            sidecar_bytes=bytes(self.sidecar_bytes),
            package_fingerprint=self.package_fingerprint,
            source_fingerprint=self.source_fingerprint,
            parse_key=self.parse_key,
            conversion_ledger=[dict(item) for item in self.conversion_ledger],
        )

    def recompute_fingerprint(self) -> str:
        records = []
        for name in sorted(self.tracks):
            track = self.tracks[name]
            records.append(
                {
                    "name": name,
                    "frames": track.frames,
                    "sample_rate": track.sample_rate,
                    "pcm_sha256": track.content_fingerprint(),
                    "metadata": track.metadata,
                }
            )
        payload = {
            "format_version": self.format_version,
            "sample_rate": self.sample_rate,
            "track_order": list(self.track_order),
            "tracks": records,
            "sidecar_sha256": sha256_bytes(self.sidecar_bytes),
        }
        self.package_fingerprint = sha256_json(payload)
        return self.package_fingerprint

    def to_package_info(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "sample_rate": self.sample_rate,
            "frames": self.frames,
            "track_order": list(self.track_order),
            "package_fingerprint": self.package_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "sidecar_sha256": sha256_bytes(self.sidecar_bytes),
            "tracks": [self.tracks[name].to_info() for name in sorted(self.tracks)],
        }


@dataclass
class PackageParseResult:
    format_version: str
    package: ParsedPackage | None
    failures: list[DiagnosticError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.package is not None and not self.failures
