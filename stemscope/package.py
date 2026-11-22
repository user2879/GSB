from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, sort_errors
from .hashing import cache_key, hash_directory, sha256_bytes, sha256_json
from .model import (
    CANONICAL_TRACK_ORDER,
    FORMAT_VERSIONS,
    PackageParseResult,
    ParsedPackage,
    Track,
)
from .v3codec import decode_v3
from .wavio import read_pcm16_wav


def package_paths(segment_dir: str | Path, format_version: str) -> dict[str, Path]:
    root = Path(segment_dir) / "packages"
    if format_version == "v1":
        return {"root": root / "v1", "wav": root / "v1" / "stems.wav"}
    if format_version == "v2":
        return {
            "root": root / "v2",
            "manifest": root / "v2" / "manifest.json",
            "sidecar": root / "v2" / "manifest.sidecar.json",
        }
    if format_version == "v3":
        return {
            "root": root / "v3",
            "package": root / "v3" / "stems.v3",
            "sidecar": root / "v3" / "stems.v3.sidecar.json",
        }
    raise DiagnosticError("E_CONFIG_INVALID", f"unknown format version {format_version!r}", stage="parse")


def _load_sidecar(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.exists():
        return {}, b""
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("sidecar root must be an object")
        return value, raw
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            f"invalid sidecar metadata {path}: {exc}",
            stage="parse",
            details={"path": str(path)},
        ) from exc


def _package_fingerprint(
    format_version: str,
    sample_rate: int,
    tracks: dict[str, Track],
    track_order: tuple[str, ...],
    sidecar_bytes: bytes,
) -> str:
    tracks_info = []
    for name in sorted(tracks):
        track = tracks[name]
        tracks_info.append(
            {
                "name": name,
                "frames": track.frames,
                "sample_rate": track.sample_rate,
                "pcm_sha256": track.content_fingerprint(),
                "metadata": track.metadata,
            }
        )
    return sha256_json(
        {
            "format_version": format_version,
            "sample_rate": sample_rate,
            "track_order": list(track_order),
            "tracks": tracks_info,
            "sidecar_sha256": sha256_bytes(sidecar_bytes),
        }
    )


def _make_package(
    format_version: str,
    sample_rate: int,
    tracks: dict[str, Track],
    order: tuple[str, ...],
    source_path: str,
    sidecar: dict[str, Any],
    sidecar_bytes: bytes,
    source_fingerprint: str,
) -> ParsedPackage:
    if tracks:
        frame_values = {track.frames for track in tracks.values()}
        if len(frame_values) != 1:
            raise DiagnosticError(
                "E_TRACK_PARSE_FAILED",
                "tracks in a package must have equal frame counts",
                stage="parse",
                format_version=format_version,
                details={"frame_counts": {name: track.frames for name, track in tracks.items()}},
            )
        rates = {track.sample_rate for track in tracks.values()}
        if rates != {sample_rate}:
            raise DiagnosticError(
                "E_TRACK_PARSE_FAILED",
                "track sample rates do not match package sample rate",
                stage="parse",
                format_version=format_version,
            )
    fingerprint = _package_fingerprint(format_version, sample_rate, tracks, order, sidecar_bytes)
    return ParsedPackage(
        format_version=format_version,
        sample_rate=sample_rate,
        tracks=tracks,
        track_order=order,
        source_path=source_path,
        sidecar=sidecar,
        sidecar_bytes=sidecar_bytes,
        package_fingerprint=fingerprint,
        source_fingerprint=source_fingerprint,
    )


def package_source_fingerprint(segment_dir: str | Path, format_version: str) -> str:
    segment_dir = Path(segment_dir)
    paths = package_paths(segment_dir, format_version)
    root = paths["root"]
    return hash_directory(root) if root.exists() else ""


def parse_cache_key(format_version: str, source_fingerprint: str) -> str:
    return cache_key(
        "parse",
        format_version,
        source_fingerprint,
        schema="stemscope-parse-v1",
    )


def parse_package(
    segment_dir: str | Path,
    format_version: str,
    *,
    clip_id: str = "",
) -> PackageParseResult:
    if format_version not in FORMAT_VERSIONS:
        return PackageParseResult(
            format_version,
            None,
            [
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"unknown format version {format_version!r}",
                    stage="parse",
                    format_version=format_version,
                    clip_id=clip_id,
                )
            ],
        )
    segment = Path(segment_dir)
    try:
        source_fp = package_source_fingerprint(segment, format_version)
        if format_version == "v1":
            package = _parse_v1(segment, source_fp, clip_id)
        elif format_version == "v2":
            package = _parse_v2(segment, source_fp, clip_id)
        else:
            package = _parse_v3(segment, source_fp, clip_id)
        package.parse_key = parse_cache_key(format_version, source_fp)
        return PackageParseResult(format_version, package, [])
    except AggregateParseErrors as exc:
        return PackageParseResult(format_version, None, sort_errors(exc.failures))
    except DiagnosticError as exc:
        if not exc.clip_id:
            exc = DiagnosticError(
                exc.code,
                exc.message,
                stage=exc.stage or "parse",
                format_version=exc.format_version or format_version,
                clip_id=clip_id,
                track=exc.track,
                details=exc.details,
            )
        return PackageParseResult(format_version, None, [exc])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return PackageParseResult(
            format_version,
            None,
            [
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"cannot parse {format_version} package: {exc}",
                    stage="parse",
                    format_version=format_version,
                    clip_id=clip_id,
                )
            ],
        )


def _parse_v1(segment: Path, source_fp: str, clip_id: str) -> ParsedPackage:
    paths = package_paths(segment, "v1")
    wav = read_pcm16_wav(paths["wav"])
    sidecar, sidecar_bytes = _load_sidecar(Path(str(paths["wav"]) + ".sidecar.json"))
    channel_names_value = sidecar.get("channel_names", list(CANONICAL_TRACK_ORDER))
    if not isinstance(channel_names_value, list):
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            "v1 sidecar channel_names must be a list",
            stage="parse",
            format_version="v1",
            clip_id=clip_id,
        )
    channel_names = tuple(str(name) for name in channel_names_value)
    if len(set(channel_names)) != len(channel_names) or not all(channel_names):
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            "v1 sidecar channel_names must be unique and non-empty",
            stage="parse",
            format_version="v1",
            clip_id=clip_id,
        )
    if wav.channels != len(channel_names):
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            f"v1 expected {len(channel_names)} channels, got {wav.channels}",
            stage="parse",
            format_version="v1",
            clip_id=clip_id,
        )
    tracks = {
        name: Track(
            name=name,
            pcm=wav.pcm[index].copy(),
            sample_rate=wav.sample_rate,
            source_format="v1",
        )
        for index, name in enumerate(channel_names)
    }
    return _make_package(
        "v1",
        wav.sample_rate,
        tracks,
        channel_names,
        str(paths["wav"]),
        sidecar,
        sidecar_bytes,
        source_fp,
    )


def _parse_v2(segment: Path, source_fp: str, clip_id: str) -> ParsedPackage:
    paths = package_paths(segment, "v2")
    manifest_path = paths["manifest"]
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or int(manifest.get("version", 0)) != 2:
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            "v2 manifest version must be 2",
            stage="parse",
            format_version="v2",
            clip_id=clip_id,
        )
    sample_rate = int(manifest.get("sample_rate", 0))
    frame_count = int(manifest.get("frame_count", -1))
    entries = manifest.get("tracks")
    if sample_rate <= 0 or frame_count < 0 or not isinstance(entries, list) or not entries:
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            "v2 manifest has invalid sample_rate/frame_count/tracks",
            stage="parse",
            format_version="v2",
            clip_id=clip_id,
        )
    sidecar_path = paths["sidecar"]
    if manifest.get("sidecar"):
        sidecar_path = manifest_path.parent / str(manifest["sidecar"])
    sidecar, sidecar_bytes = _load_sidecar(sidecar_path)
    tracks: dict[str, Track] = {}
    failures: list[DiagnosticError] = []
    order: list[tuple[int, str]] = []
    seen_channels: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    "v2 track entry must be an object",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                )
            )
            continue
        name = entry.get("name")
        filename = entry.get("file")
        channel = int(entry.get("channel", index))
        declared_frames = int(entry.get("frame_count", entry.get("frames", -1)))
        if not isinstance(name, str) or not name or name in tracks:
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"invalid or duplicate v2 track name at entry {index}",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=str(name or ""),
                )
            )
            continue
        if channel in seen_channels or channel < 0 or channel >= len(entries):
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"invalid channel mapping {channel} for track {name}",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=name,
                )
            )
            continue
        seen_channels.add(channel)
        if not isinstance(filename, str) or not filename:
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"missing file for v2 track {name}",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=name,
                )
            )
            continue
        try:
            wav = read_pcm16_wav(manifest_path.parent / filename)
        except DiagnosticError as exc:
            failures.append(
                DiagnosticError(
                    exc.code,
                    exc.message,
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=name,
                    details=exc.details,
                )
            )
            continue
        if wav.channels != 1:
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"v2 track {name} must be mono, got {wav.channels} channels",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=name,
                )
            )
            continue
        if wav.sample_rate != sample_rate:
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"v2 track {name} sample rate {wav.sample_rate} != manifest {sample_rate}",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=name,
                )
            )
            continue
        if declared_frames != wav.frames or wav.frames != frame_count:
            failures.append(
                DiagnosticError(
                    "E_TRACK_PARSE_FAILED",
                    f"v2 track {name} frame mismatch declared={declared_frames} actual={wav.frames} package={frame_count}",
                    stage="parse",
                    format_version="v2",
                    clip_id=clip_id,
                    track=name,
                    details={
                        "declared_frames": declared_frames,
                        "actual_frames": wav.frames,
                        "package_frames": frame_count,
                    },
                )
            )
            continue
        tracks[name] = Track(
            name=name,
            pcm=wav.pcm[0].copy(),
            sample_rate=wav.sample_rate,
            source_format="v2",
            metadata={"channel": channel, "file": filename},
        )
        order.append((channel, name))
    if failures:
        raise AggregateParseErrors(failures)
    if len(tracks) != len(entries) or seen_channels != set(range(len(entries))):
        raise DiagnosticError(
            "E_TRACK_PARSE_FAILED",
            "v2 channel mapping is incomplete",
            stage="parse",
            format_version="v2",
            clip_id=clip_id,
        )
    track_order = tuple(name for _, name in sorted(order))
    return _make_package(
        "v2",
        sample_rate,
        tracks,
        track_order,
        str(manifest_path),
        sidecar,
        sidecar_bytes,
        source_fp,
    )


def _parse_v3(segment: Path, source_fp: str, clip_id: str) -> ParsedPackage:
    paths = package_paths(segment, "v3")
    package_path = paths["package"]
    raw = package_path.read_bytes()
    tracks, order, failures, sample_rate = decode_v3(raw, str(package_path), clip_id)
    if failures:
        raise AggregateParseErrors(failures)
    sidecar, sidecar_bytes = _load_sidecar(paths["sidecar"])
    return _make_package(
        "v3",
        sample_rate,
        tracks,
        order,
        str(package_path),
        sidecar,
        sidecar_bytes,
        source_fp,
    )


class AggregateParseErrors(DiagnosticError):
    def __init__(self, failures: list[DiagnosticError]):
        super().__init__(
            code="E_TRACK_PARSE_FAILED",
            message="; ".join(error.message for error in sort_errors(failures)),
            stage="parse",
            details={"failures": [error.to_dict() for error in sort_errors(failures)]},
        )
        self.failures = sort_errors(failures)


def parse_package_all(segment_dir: str | Path, clip_id: str = "") -> dict[str, PackageParseResult]:
    return {
        version: parse_package(segment_dir, version, clip_id=clip_id)
        for version in FORMAT_VERSIONS
    }
