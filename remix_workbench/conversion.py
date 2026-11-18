from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import canonical_json, digest_json
from .formats import (
    V1_AUDIO_FILE,
    V2_MANIFEST_FILE,
    V3_HEADER,
    V3_MAGIC,
    V3_PACKAGE_FILE,
    V3_TABLE_ENTRY,
    V3_TABLE_PREFIX,
)
from .models import V1_TRACK_ORDER, TrackPlane
from .pcm import (
    q23_to_i16,
    q23_to_i24_bytes,
    i16_to_bytes,
    write_pcm_wav,
    write_wav,
)

CONVERTER_VERSION = "conversion-v1"
STRUCTURAL_METADATA_KEYS = {
    "container",
    "channel_order",
    "channel_count",
    "sample_width_bytes",
    "ignored_channels",
    "magic",
    "header_size_bytes",
    "table_end_bytes",
    "track_entries",
    "reserved",
    "manifest_format_version",
    "declared_bit_depth",
    "manifest_metadata",
}


def _sorted_tracks(plane: TrackPlane) -> list[str]:
    ordered = [name for name in V1_TRACK_ORDER if name in plane.tracks]
    ordered.extend(name for name in sorted(plane.tracks) if name not in ordered)
    return ordered


def _target_mapping(plane: TrackPlane, target: str) -> dict[str, int]:
    if target == "v1":
        return {name: index for index, name in enumerate(V1_TRACK_ORDER)}
    return {name: index for index, name in enumerate(_sorted_tracks(plane))}


def _interleaved_i16(plane: TrackPlane) -> bytes:
    matrix = np.zeros((plane.frames, len(V1_TRACK_ORDER)), dtype=np.int16)
    for channel, name in enumerate(V1_TRACK_ORDER):
        if name in plane.tracks:
            matrix[:, channel] = q23_to_i16(plane.tracks[name])
    return matrix.astype("<i2").tobytes(order="C")


def _write_v1(plane: TrackPlane, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    raw = _interleaved_i16(plane)
    write_wav(
        directory / V1_AUDIO_FILE,
        sample_rate=plane.sample_rate,
        channels=len(V1_TRACK_ORDER),
        sample_width=2,
        raw_pcm=raw,
    )
    descriptor = {
        "format_version": "v1",
        "sample_rate": plane.sample_rate,
        "bit_depth": 16,
        "frames": plane.frames,
        "channel_map": {name: index for index, name in enumerate(V1_TRACK_ORDER)},
        "note": "v1 audio is authoritative; descriptor is diagnostic only",
    }
    (directory / "package.json").write_text(
        canonical_json(descriptor) + "\n", encoding="utf-8"
    )
    return {
        "audio": V1_AUDIO_FILE,
        "descriptor": "package.json",
        "channels": len(V1_TRACK_ORDER),
        "bit_depth": 16,
    }


def _write_v2(
    plane: TrackPlane, directory: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    tracks_dir = directory / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    mapping = _target_mapping(plane, "v2")
    entries: dict[str, Any] = {}
    for name in sorted(plane.tracks):
        file_name = f"tracks/{name}.wav"
        write_pcm_wav(
            directory / file_name,
            plane.tracks[name],
            sample_rate=plane.sample_rate,
            sample_width=3,
        )
        entry: dict[str, Any] = {
            "file": file_name,
            "channel": mapping[name],
            "frames": plane.frames,
            "bit_depth": 24,
        }
        gains = plane.track_gain_metadata
        annotations = plane.source_frame_annotations
        if name in gains:
            entry["gain_db"] = gains[name]
        if name in annotations:
            entry["source_frames"] = annotations[name]
        entries[name] = entry
    manifest = {
        "format_version": "v2",
        "sample_rate": plane.sample_rate,
        "bit_depth": 24,
        "frames": plane.frames,
        "channel_map": mapping,
        "tracks": entries,
        "metadata": metadata,
    }
    (directory / V2_MANIFEST_FILE).write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return {
        "manifest": V2_MANIFEST_FILE,
        "tracks": {name: f"tracks/{name}.wav" for name in sorted(entries)},
        "bit_depth": 24,
        "channel_map": mapping,
    }


def _write_v3(plane: TrackPlane, directory: Path) -> dict[str, Any]:
    names = _sorted_tracks(plane)
    table = bytearray()
    payloads: list[bytes] = []
    for name in names:
        payload = i16_to_bytes(q23_to_i16(plane.tracks[name]))
        payloads.append(payload)
        encoded = name.encode("utf-8")
        table.extend(V3_TABLE_PREFIX.pack(len(encoded)))
        table.extend(encoded)
        table.extend(V3_TABLE_ENTRY.pack(0, len(payload)))
    header_size = V3_HEADER.size + len(table)
    cursor = header_size
    table = bytearray()
    for name, payload in zip(names, payloads):
        encoded = name.encode("utf-8")
        table.extend(V3_TABLE_PREFIX.pack(len(encoded)))
        table.extend(encoded)
        table.extend(V3_TABLE_ENTRY.pack(cursor, len(payload)))
        cursor += len(payload)
    header = V3_HEADER.pack(
        V3_MAGIC,
        3,
        len(names),
        plane.frames,
        plane.sample_rate,
        16,
        0,
    )
    blob = header + bytes(table) + b"".join(payloads)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / V3_PACKAGE_FILE).write_bytes(blob)
    return {
        "package": V3_PACKAGE_FILE,
        "header_size": header_size,
        "bit_depth": 16,
        "channel_map": {name: index for index, name in enumerate(names)},
        "offsets": [
            {
                "track": name,
                "offset": (
                    V3_HEADER.size
                    + sum(
                        V3_TABLE_PREFIX.size
                        + len(item.encode("utf-8"))
                        + V3_TABLE_ENTRY.size
                        for item in names[:index]
                    )
                ),
                "bytes": len(payload),
            }
            for index, (name, payload) in enumerate(zip(names, payloads))
        ],
    }


def _quantization_loss(plane: TrackPlane, target_bit_depth: int) -> list[dict[str, Any]]:
    if target_bit_depth >= 24:
        return []
    losses: list[dict[str, Any]] = []
    for name in plane.track_names:
        values = plane.tracks[name].astype(np.int64)
        reconstructed = q23_to_i16(values).astype(np.int64) << 8
        residual = values - reconstructed
        nonzero = int(np.count_nonzero(residual))
        if nonzero == 0:
            continue
        losses.append(
            {
                "field": f"tracks.{name}.quantization_residual",
                "track": name,
                "reason": (
                    f"target {target_bit_depth}-bit PCM cannot retain Q23 low bits"
                ),
                "sampled_frames": int(values.size),
                "nonzero_residual_frames": nonzero,
                "max_abs_residual_q23": int(np.max(np.abs(residual))),
                "residual_sha256": digest_json(
                    {
                        "track": name,
                        "nonzero": nonzero,
                        "max": int(np.max(np.abs(residual))),
                    }
                ),
            }
        )
    return losses


def _metadata_losses(
    plane: TrackPlane,
    target_version: str,
    target_map: dict[str, int],
) -> list[dict[str, Any]]:
    losses: list[dict[str, Any]] = []
    if target_version == "v2":
        return losses
    for track, value in sorted(plane.track_gain_metadata.items()):
        losses.append(
            {
                "field": f"track_gain_metadata.{track}",
                "value": value,
                "track": track,
                "reason": f"target {target_version} has no per-track gain metadata channel",
            }
        )
    for track, value in sorted(plane.source_frame_annotations.items()):
        losses.append(
            {
                "field": f"source_frame_annotations.{track}",
                "value": value,
                "track": track,
                "reason": f"target {target_version} has no source-frame annotation channel",
            }
        )
    for key, value in sorted(plane.source_metadata.items()):
        if key in STRUCTURAL_METADATA_KEYS:
            continue
        if key in {"track_gain_metadata", "source_frame_annotations", "track_overrides"}:
            continue
        losses.append(
            {
                "field": f"source_metadata.{key}",
                "value": value,
                "reason": f"target {target_version} cannot encode arbitrary source metadata",
            }
        )
    losses.append(
        {
            "field": "source_format_version",
            "value": plane.format_version,
            "reason": f"target {target_version} payload has no source-format field",
        }
    )
    if target_version == "v1":
        source_mapping = dict(plane.channel_map)
        canonical = {name: index for index, name in enumerate(V1_TRACK_ORDER)}
        if source_mapping != canonical:
            losses.append(
                {
                    "field": "channel_map",
                    "value": source_mapping,
                    "reason": "v1 uses fixed bass/drums/vocal channel order",
                }
            )
        for name in plane.track_names:
            if name not in V1_TRACK_ORDER:
                losses.append(
                    {
                        "field": f"tracks.{name}",
                        "value": plane.frames,
                        "reason": "v1 has only three fixed channels; extra track omitted",
                    }
                )
    return losses


def _sidecar_metadata(
    plane: TrackPlane,
    *,
    target_version: str,
    losses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_format_version": plane.format_version,
        "target_format_version": target_version,
        "source_clip_id": plane.clip_id,
        "source_frames": plane.frames,
        "source_bit_depth": plane.bit_depth,
        "source_channel_map": plane.channel_map,
        "source_metadata": plane.source_metadata,
        "source_track_gain_metadata": plane.track_gain_metadata,
        "source_frame_annotations": plane.source_frame_annotations,
        "unexpressed_fields": losses,
    }


def _replace_directory(temp: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + f".old.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temp, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def convert_plane(
    plane: TrackPlane,
    target_version: str,
    destination: str | Path,
    *,
    source_package: str | Path | None = None,
) -> dict[str, Any]:
    if target_version not in {"v1", "v2", "v3"}:
        raise ValueError(f"unsupported target format version {target_version!r}")
    destination = Path(destination)
    temp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    target_map = _target_mapping(plane, target_version)
    losses = _metadata_losses(plane, target_version, target_map)
    target_depth = 24 if target_version == "v2" else 16
    losses.extend(_quantization_loss(plane, target_depth))
    losses.sort(key=lambda item: (item["field"], json.dumps(item.get("value"), sort_keys=True)))
    if target_version == "v1":
        outputs = _write_v1(plane, temp)
    elif target_version == "v2":
        metadata = {
            "source_format_version": plane.format_version,
            "source_frame_annotations": plane.source_frame_annotations,
            "track_gain_metadata": plane.track_gain_metadata,
            "source_metadata": plane.source_metadata,
            "source_channel_map": plane.channel_map,
            "source_bit_depth": plane.bit_depth,
        }
        outputs = _write_v2(plane, temp, metadata)
    else:
        outputs = _write_v3(plane, temp)

    sidecar = _sidecar_metadata(plane, target_version=target_version, losses=losses)
    (temp / "loss_metadata.json").write_text(
        canonical_json(sidecar) + "\n", encoding="utf-8"
    )
    report = {
        "source_format_version": plane.format_version,
        "target_format_version": target_version,
        "source_clip_id": plane.clip_id,
        "source_package": str(source_package) if source_package else plane.package_path,
        "source_frames": plane.frames,
        "source_bit_depth": plane.bit_depth,
        "target_bit_depth": target_depth,
        "target_channel_map": target_map,
        "outputs": outputs,
        "sidecar_metadata_file": "loss_metadata.json",
        "losses": losses,
        "loss_count": len(losses),
    }
    report_text = canonical_json(report) + "\n"
    (temp / "conversion_report.json").write_text(report_text, encoding="utf-8")
    report["report_sha256"] = digest_json(report)
    _replace_directory(temp, destination)
    report["destination"] = str(destination)
    return report


def report_path(destination: str | Path) -> Path:
    return Path(destination) / "conversion_report.json"

