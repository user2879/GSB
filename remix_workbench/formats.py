from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import TrackFormatError
from .models import (
    DEFAULT_TRACKS,
    V1_TRACK_ORDER,
    TrackIssue,
    TrackPlane,
)
from .pcm import decode_pcm_bytes, read_wav

V1_AUDIO_FILE = "mix.wav"
V2_MANIFEST_FILE = "manifest.json"
V3_PACKAGE_FILE = "tracks.r3p"
V3_MAGIC = b"RMX3PK01"
V3_HEADER = struct.Struct("<8sHHIIHH")
V3_TABLE_PREFIX = struct.Struct("<H")
V3_TABLE_ENTRY = struct.Struct("<II")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackFormatError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TrackFormatError(f"JSON manifest must be an object: {path}")
    return data


def _zero_track(frames: int) -> np.ndarray:
    return np.zeros(frames, dtype=np.int32)


def _issue(
    clip_id: str,
    track: str,
    message: str,
    *,
    injection_id: str | None = None,
) -> TrackIssue:
    return TrackIssue(
        clip_id=clip_id,
        track=track,
        stage="parse",
        code="E_TRACK_FORMAT",
        message=message,
        injection_id=injection_id,
    )


def _validate_channel_map(channel_map: Any, context: str) -> dict[str, int]:
    if not isinstance(channel_map, dict):
        raise TrackFormatError(f"{context}: channel_map must be an object")
    result: dict[str, int] = {}
    used: set[int] = set()
    for name, channel in channel_map.items():
        if not isinstance(name, str) or not name:
            raise TrackFormatError(f"{context}: invalid track name in channel_map")
        if not isinstance(channel, int) or isinstance(channel, bool) or channel < 0:
            raise TrackFormatError(f"{context}: invalid channel for {name}")
        if channel in used:
            raise TrackFormatError(f"{context}: duplicate channel {channel}")
        used.add(channel)
        result[name] = channel
    return result


def _overlay_overrides(
    plane: TrackPlane,
    overrides: dict[str, str | Path],
    *,
    injection_id: str | None,
    injection_track: str | None,
) -> None:
    for track in sorted(overrides):
        target_issue_injection = (
            injection_id if injection_track in (None, track) else None
        )
        try:
            rate, channels, width, frames, raw = read_wav(overrides[track])
            if channels != 1:
                raise ValueError("track replacement must be mono")
            if rate != plane.sample_rate:
                raise ValueError(
                    f"track replacement sample rate {rate} != {plane.sample_rate}"
                )
            if frames != plane.frames:
                raise ValueError(
                    f"track replacement frames {frames} != {plane.frames}"
                )
            values = decode_pcm_bytes(raw, width)
            if values.size != frames:
                raise ValueError("track replacement payload length mismatch")
            plane.tracks[track] = values.astype(np.int32)
            plane.track_frames[track] = frames
            plane.source_metadata.setdefault("track_overrides", {})[track] = str(
                overrides[track]
            )
        except (OSError, ValueError) as exc:
            message = f"replacement PCM for {track} is invalid: {exc}"
            plane.issues.append(
                _issue(
                    plane.clip_id,
                    track,
                    message,
                    injection_id=target_issue_injection,
                )
            )
            plane.tracks[track] = _zero_track(plane.frames)
            plane.track_frames[track] = plane.frames


def _finish_plane(
    plane: TrackPlane,
    *,
    expected_tracks: Iterable[str],
    overrides: dict[str, str | Path],
    injection_id: str | None,
    injection_track: str | None,
) -> TrackPlane:
    for track in sorted(set(expected_tracks)):
        if track not in plane.tracks:
            plane.tracks[track] = _zero_track(plane.frames)
            plane.track_frames[track] = plane.frames
            plane.missing_tracks.append(track)
            plane.issues.append(
                _issue(
                    plane.clip_id,
                    track,
                    f"expected track {track!r} is absent; inserted deterministic silence",
                    injection_id=injection_id if injection_track in (None, track) else None,
                )
            )
    _overlay_overrides(
        plane,
        overrides,
        injection_id=injection_id,
        injection_track=injection_track,
    )
    if plane.format_version == "v1":
        plane.channel_map = {
            track: plane.channel_map[track]
            for track in plane.track_names
            if track in plane.channel_map
        }
    plane.issues.sort(key=lambda item: item.stable_key)
    plane.missing_tracks = sorted(set(plane.missing_tracks))
    return plane


def parse_v1(
    clip_id: str,
    package_path: Path,
    *,
    expected_tracks: Iterable[str] = DEFAULT_TRACKS,
    overrides: dict[str, str | Path] | None = None,
    injection_id: str | None = None,
    injection_track: str | None = None,
) -> TrackPlane:
    path = package_path / V1_AUDIO_FILE
    try:
        rate, channels, width, frames, raw = read_wav(path)
    except (OSError, ValueError) as exc:
        raise TrackFormatError(
            f"v1 container header/payload is damaged: {exc}",
            track=injection_track,
            injection_id=injection_id,
        ) from exc
    if channels <= 0:
        raise TrackFormatError("v1 WAV has no channels", injection_id=injection_id)
    values = decode_pcm_bytes(raw, width)
    if values.size != frames * channels:
        raise TrackFormatError(
            "v1 interleaved payload does not contain the declared frame count",
            injection_id=injection_id,
        )
    matrix = values.reshape(frames, channels)
    tracks: dict[str, np.ndarray] = {}
    channel_map: dict[str, int] = {}
    for channel, track in enumerate(V1_TRACK_ORDER):
        if channel < channels:
            tracks[track] = np.ascontiguousarray(matrix[:, channel])
            channel_map[track] = channel
    source_metadata: dict[str, Any] = {
        "container": "RIFF/WAVE",
        "channel_order": list(V1_TRACK_ORDER[:channels]),
        "channel_count": channels,
        "sample_width_bytes": width,
    }
    if channels > len(V1_TRACK_ORDER):
        source_metadata["ignored_channels"] = list(
            range(len(V1_TRACK_ORDER), channels)
        )
    plane = TrackPlane(
        clip_id=clip_id,
        format_version="v1",
        sample_rate=rate,
        bit_depth=width * 8,
        frames=frames,
        channel_map=channel_map,
        tracks=tracks,
        track_frames={name: frames for name in tracks},
        source_metadata=source_metadata,
        package_path=str(package_path),
    )
    return _finish_plane(
        plane,
        expected_tracks=expected_tracks,
        overrides=overrides or {},
        injection_id=injection_id,
        injection_track=injection_track,
    )


def parse_v2(
    clip_id: str,
    package_path: Path,
    *,
    expected_tracks: Iterable[str] = DEFAULT_TRACKS,
    overrides: dict[str, str | Path] | None = None,
    injection_id: str | None = None,
    injection_track: str | None = None,
) -> TrackPlane:
    manifest_path = package_path / V2_MANIFEST_FILE
    if not manifest_path.is_file():
        raise TrackFormatError(f"v2 manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("format_version") not in (None, "v2", 2):
        raise TrackFormatError("v2 manifest declares a different format_version")
    try:
        sample_rate = int(manifest["sample_rate"])
        frames = int(manifest["frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackFormatError(f"v2 manifest lacks valid sample rate/frames: {exc}") from exc
    if sample_rate <= 0 or frames <= 0:
        raise TrackFormatError("v2 manifest sample rate/frames must be positive")
    channel_map = _validate_channel_map(manifest.get("channel_map", {}), "v2")
    entries = manifest.get("tracks")
    if not isinstance(entries, dict) or not entries:
        raise TrackFormatError("v2 manifest tracks must be a non-empty object")

    metadata = manifest.get("metadata")
    source_metadata: dict[str, Any] = {
        "container": "JSON manifest + mono WAV files",
        "manifest_format_version": manifest.get("format_version", "v2"),
        "declared_bit_depth": manifest.get("bit_depth"),
        "manifest_metadata": metadata if isinstance(metadata, dict) else {},
        "track_entries": {},
    }
    if isinstance(metadata, dict):
        if isinstance(metadata.get("track_gain_metadata"), dict):
            source_metadata["track_gain_metadata"] = metadata[
                "track_gain_metadata"
            ]
        if isinstance(metadata.get("source_frame_annotations"), dict):
            source_metadata["source_frame_annotations"] = metadata[
                "source_frame_annotations"
            ]

    tracks: dict[str, np.ndarray] = {}
    track_frames: dict[str, int] = {}
    issues: list[TrackIssue] = []
    gain_metadata = dict(source_metadata.get("track_gain_metadata", {}))
    frame_annotations = dict(source_metadata.get("source_frame_annotations", {}))
    for track in sorted(entries):
        entry = entries[track]
        inject_here = injection_id if injection_track in (None, track) else None
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ValueError("track entry must contain a file path")
            declared_frames = int(entry.get("frames", frames))
            if declared_frames != frames:
                raise ValueError(
                    f"declared track frames {declared_frames} != manifest frames {frames}"
                )
            track_path = package_path / entry["file"]
            rate, channels, width, actual_frames, raw = read_wav(track_path)
            if rate != sample_rate:
                raise ValueError(f"sample rate {rate} != manifest {sample_rate}")
            if channels != 1:
                raise ValueError("v2 individual track WAV must be mono")
            if actual_frames != declared_frames:
                raise ValueError(
                    f"actual frames {actual_frames} != declared {declared_frames}"
                )
            values = decode_pcm_bytes(raw, width)
            if values.size != frames:
                raise ValueError("decoded track sample count mismatch")
            channel = entry.get("channel", channel_map.get(track))
            if not isinstance(channel, int) or isinstance(channel, bool):
                raise ValueError("track channel is missing or invalid")
            if track in channel_map and channel_map[track] != channel:
                raise ValueError(
                    f"channel_map declares {channel_map[track]} but entry declares {channel}"
                )
            channel_map[track] = channel
            tracks[track] = values.astype(np.int32)
            track_frames[track] = frames
            source_metadata["track_entries"][track] = {
                "file": entry["file"],
                "channel": channel,
                "frames": frames,
                "bit_depth": width * 8,
                "gain_db": entry.get("gain_db"),
                "source_frames": entry.get("source_frames"),
            }
            if "gain_db" in entry:
                gain_metadata[track] = entry["gain_db"]
            if "source_frames" in entry:
                frame_annotations[track] = entry["source_frames"]
        except (OSError, ValueError, TypeError) as exc:
            message = f"v2 track {track!r} failed frame/content validation: {exc}"
            issues.append(_issue(clip_id, track, message, injection_id=inject_here))
            tracks[track] = _zero_track(frames)
            track_frames[track] = frames
    if gain_metadata:
        source_metadata["track_gain_metadata"] = gain_metadata
    if frame_annotations:
        source_metadata["source_frame_annotations"] = frame_annotations
    if not channel_map:
        raise TrackFormatError("v2 manifest has no usable channel mapping")
    widths = [
        int(item["bit_depth"]) for item in source_metadata["track_entries"].values()
    ]
    bit_depth = max(widths) if widths else int(manifest.get("bit_depth", 16))
    plane = TrackPlane(
        clip_id=clip_id,
        format_version="v2",
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        frames=frames,
        channel_map=channel_map,
        tracks=tracks,
        track_frames=track_frames,
        source_metadata=source_metadata,
        issues=issues,
        package_path=str(package_path),
    )
    return _finish_plane(
        plane,
        expected_tracks=expected_tracks,
        overrides=overrides or {},
        injection_id=injection_id,
        injection_track=injection_track,
    )


def parse_v3(
    clip_id: str,
    package_path: Path,
    *,
    expected_tracks: Iterable[str] = DEFAULT_TRACKS,
    overrides: dict[str, str | Path] | None = None,
    injection_id: str | None = None,
    injection_track: str | None = None,
) -> TrackPlane:
    path = package_path / V3_PACKAGE_FILE
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise TrackFormatError(
            f"v3 package cannot be read: {exc}", injection_id=injection_id
        ) from exc
    if len(blob) < V3_HEADER.size:
        raise TrackFormatError(
            "v3 header is truncated",
            track=injection_track,
            injection_id=injection_id,
        )
    try:
        magic, version, track_count, frames, sample_rate, bit_depth, _reserved = (
            V3_HEADER.unpack_from(blob, 0)
        )
    except struct.error as exc:
        raise TrackFormatError("v3 header cannot be unpacked") from exc
    if magic != V3_MAGIC:
        raise TrackFormatError("v3 magic/header is corrupt", injection_id=injection_id)
    if version != 3:
        raise TrackFormatError(f"v3 header version is {version}, expected 3")
    if track_count <= 0 or track_count > 1024:
        raise TrackFormatError(f"invalid v3 track count {track_count}")
    if frames <= 0 or sample_rate <= 0 or bit_depth != 16:
        raise TrackFormatError(
            f"invalid v3 header frame/rate/bit-depth values: {frames}/{sample_rate}/{bit_depth}"
        )
    cursor = V3_HEADER.size
    entries: list[tuple[str, int, int]] = []
    seen_names: set[str] = set()
    for _ in range(track_count):
        if cursor + V3_TABLE_PREFIX.size > len(blob):
            raise TrackFormatError("v3 offset table is truncated")
        (name_len,) = V3_TABLE_PREFIX.unpack_from(blob, cursor)
        cursor += V3_TABLE_PREFIX.size
        if name_len <= 0 or name_len > 255 or cursor + name_len > len(blob):
            raise TrackFormatError("v3 track name length is invalid")
        raw_name = blob[cursor : cursor + name_len]
        cursor += name_len
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrackFormatError("v3 track name is not UTF-8") from exc
        if name in seen_names:
            raise TrackFormatError(f"duplicate v3 track name {name!r}")
        seen_names.add(name)
        if cursor + V3_TABLE_ENTRY.size > len(blob):
            raise TrackFormatError("v3 offset table is truncated")
        offset, byte_length = V3_TABLE_ENTRY.unpack_from(blob, cursor)
        cursor += V3_TABLE_ENTRY.size
        expected_length = frames * 2
        if byte_length != expected_length:
            raise TrackFormatError(
                f"v3 offset table declares {byte_length} bytes for {name}, "
                f"expected {expected_length}",
                track=name,
                injection_id=injection_id if injection_track in (None, name) else None,
            )
        if offset < cursor or offset + byte_length > len(blob):
            raise TrackFormatError(
                f"v3 offset table is out of bounds for {name}: "
                f"offset={offset}, length={byte_length}, file={len(blob)}",
                track=name,
                injection_id=injection_id if injection_track in (None, name) else None,
            )
        entries.append((name, offset, byte_length))
    ranges = sorted((offset, offset + length) for _, offset, length in entries)
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise TrackFormatError("v3 offset table contains overlapping track blocks")

    tracks: dict[str, np.ndarray] = {}
    track_frames: dict[str, int] = {}
    channel_map: dict[str, int] = {}
    source_entries: dict[str, Any] = {}
    for channel, (name, offset, byte_length) in enumerate(entries):
        raw = blob[offset : offset + byte_length]
        values = np.frombuffer(raw, dtype="<i2").astype(np.int32) << 8
        if values.size != frames:
            raise TrackFormatError(
                f"v3 track block for {name} has {values.size} frames, expected {frames}",
                track=name,
                injection_id=injection_id if injection_track in (None, name) else None,
            )
        tracks[name] = values
        track_frames[name] = frames
        channel_map[name] = channel
        source_entries[name] = {
            "offset": offset,
            "byte_length": byte_length,
            "channel": channel,
        }
    source_metadata: dict[str, Any] = {
        "container": "RMX3 binary stem package",
        "magic": V3_MAGIC.decode("ascii"),
        "header_size_bytes": V3_HEADER.size,
        "table_end_bytes": cursor,
        "track_entries": source_entries,
        "reserved": _reserved,
    }
    plane = TrackPlane(
        clip_id=clip_id,
        format_version="v3",
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        frames=frames,
        channel_map=channel_map,
        tracks=tracks,
        track_frames=track_frames,
        source_metadata=source_metadata,
        package_path=str(package_path),
    )
    return _finish_plane(
        plane,
        expected_tracks=expected_tracks,
        overrides=overrides or {},
        injection_id=injection_id,
        injection_track=injection_track,
    )


def parse_package(
    clip_id: str,
    package_path: str | Path,
    format_version: str,
    *,
    expected_tracks: Iterable[str] = DEFAULT_TRACKS,
    overrides: dict[str, str | Path] | None = None,
    injection_id: str | None = None,
    injection_track: str | None = None,
) -> TrackPlane:
    path = Path(package_path)
    if not path.is_dir():
        raise TrackFormatError(f"package directory does not exist: {path}")
    if format_version == "v1":
        return parse_v1(
            clip_id,
            path,
            expected_tracks=expected_tracks,
            overrides=overrides,
            injection_id=injection_id,
            injection_track=injection_track,
        )
    if format_version == "v2":
        return parse_v2(
            clip_id,
            path,
            expected_tracks=expected_tracks,
            overrides=overrides,
            injection_id=injection_id,
            injection_track=injection_track,
        )
    if format_version == "v3":
        return parse_v3(
            clip_id,
            path,
            expected_tracks=expected_tracks,
            overrides=overrides,
            injection_id=injection_id,
            injection_track=injection_track,
        )
    raise TrackFormatError(
        f"unsupported exchange format version {format_version!r}",
        track=injection_track,
        injection_id=injection_id,
    )


def package_files(package_path: str | Path, format_version: str) -> list[Path]:
    path = Path(package_path)
    if format_version == "v1":
        return [path / V1_AUDIO_FILE]
    if format_version == "v2":
        manifest = path / V2_MANIFEST_FILE
        files = [manifest]
        if manifest.is_file():
            try:
                data = _load_json(manifest)
                entries = data.get("tracks", {})
                if isinstance(entries, dict):
                    for entry in entries.values():
                        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                            files.append(path / entry["file"])
            except TrackFormatError:
                pass
        return files
    if format_version == "v3":
        return [path / V3_PACKAGE_FILE]
    return []

