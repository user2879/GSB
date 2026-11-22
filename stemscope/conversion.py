from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DiagnosticError
from .hashing import cache_key, canonical_json_bytes, sha256_bytes
from .model import FORMAT_VERSIONS, ParsedPackage, Track
from .package import parse_package
from .v3codec import encode_v3
from .wavio import write_multichannel_wav, write_pcm16_wav


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class ConversionResult:
    source_package: ParsedPackage
    target_package: ParsedPackage | None
    target_format: str
    conversion_key: str
    record: dict[str, Any]
    failures: list[DiagnosticError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.target_package is not None and not self.failures


def conversion_cache_key(package: ParsedPackage, target_format: str) -> str:
    return cache_key(
        "convert",
        package.package_fingerprint,
        target_format,
        sha256_bytes(package.sidecar_bytes),
        schema="stemscope-convert-v1",
    )


def lost_fields_for_target(package: ParsedPackage, target_format: str) -> list[dict[str, Any]]:
    lost: list[dict[str, Any]] = []
    for name in sorted(package.tracks):
        track = package.tracks[name]
        for field_name, value in sorted(track.metadata.items()):
            representable = False
            if target_format == "v1":
                representable = field_name in {"channel", "file"}
            elif target_format == "v2":
                representable = field_name in {"channel", "file"}
            elif target_format == "v3":
                representable = field_name in {"block_checksum_crc32"}
            if not representable:
                lost.append(
                    {
                        "field": field_name,
                        "source_track": name,
                        "source_format_version": package.format_version,
                        "value": value,
                    }
                )
    return lost


def _sidecar_payload(
    package: ParsedPackage,
    target_format: str,
    lost_fields: list[dict[str, Any]],
    channel_names: tuple[str, ...] | None,
) -> dict[str, Any]:
    prior = package.sidecar if isinstance(package.sidecar, dict) else {}
    return {
        "schema": "stemscope/sidecar/v1",
        "target_format": target_format,
        "channel_names": list(channel_names) if channel_names is not None else None,
        "lost_fields": sorted(
            lost_fields,
            key=lambda item: (
                item.get("source_track", ""),
                item.get("field", ""),
                str(item.get("value", "")),
            ),
        ),
        "carried_sidecar": prior,
        "conversion_ledger": [
            {
                "field": item["field"],
                "source_track": item["source_track"],
                "source_format_version": item["source_format_version"],
                "target_format_version": target_format,
            }
            for item in lost_fields
        ],
    }


def _write_sidecar(path: Path, payload: dict[str, Any]) -> bytes:
    raw = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _format_tracks(package: ParsedPackage) -> list[Track]:
    return [package.tracks[name] for name in package.track_order]


def _write_v1(package: ParsedPackage, root: Path) -> bytes:
    tracks = _format_tracks(package)
    if not tracks:
        raise DiagnosticError("E_CONVERSION_FAILED", "cannot convert empty package", stage="convert")
    frames = {track.frames for track in tracks}
    if len(frames) != 1:
        raise DiagnosticError(
            "E_CONVERSION_FAILED", "v1 target requires equal track frame counts", stage="convert"
        )
    channel_matrix = np.stack([track.pcm for track in tracks], axis=0)
    wav_path = root / "packages" / "v1" / "stems.wav"
    write_multichannel_wav(wav_path, channel_matrix, package.sample_rate)
    lost_fields = lost_fields_for_target(package, "v1")
    sidecar = _sidecar_payload(package, "v1", lost_fields, tuple(track.name for track in tracks))
    _write_sidecar(Path(str(wav_path) + ".sidecar.json"), sidecar)
    return sha256_bytes(canonical_json_bytes(sidecar))


def _write_v2(package: ParsedPackage, root: Path) -> bytes:
    target_root = root / "packages" / "v2"
    target_root.mkdir(parents=True, exist_ok=True)
    tracks = _format_tracks(package)
    entries: list[dict[str, Any]] = []
    for channel, track in enumerate(tracks):
        safe_name = _SAFE_NAME.sub("_", track.name) or f"track_{channel}"
        filename = f"{channel:02d}_{safe_name}.wav"
        write_pcm16_wav(target_root / filename, track.pcm, package.sample_rate)
        entries.append(
            {
                "name": track.name,
                "channel": channel,
                "file": filename,
                "frame_count": track.frames,
            }
        )
    lost_fields = lost_fields_for_target(package, "v2")
    sidecar = _sidecar_payload(package, "v2", lost_fields, tuple(track.name for track in tracks))
    sidecar_bytes = _write_sidecar(target_root / "manifest.sidecar.json", sidecar)
    manifest = {
        "format": "stemscope/v2",
        "version": 2,
        "sample_rate": package.sample_rate,
        "frame_count": package.frames,
        "sidecar": "manifest.sidecar.json",
        "tracks": entries,
    }
    (target_root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return sha256_bytes(sidecar_bytes)


def _write_v3(package: ParsedPackage, root: Path) -> bytes:
    target_root = root / "packages" / "v3"
    target_root.mkdir(parents=True, exist_ok=True)
    tracks = _format_tracks(package)
    raw = encode_v3(tracks, package.sample_rate, package.frames)
    package_path = target_root / "stems.v3"
    package_path.write_bytes(raw)
    lost_fields = lost_fields_for_target(package, "v3")
    sidecar = _sidecar_payload(package, "v3", lost_fields, tuple(track.name for track in tracks))
    _write_sidecar(target_root / "stems.v3.sidecar.json", sidecar)
    return sha256_bytes(raw)


def convert_package(
    package: ParsedPackage,
    target_format: str,
    destination_segment_dir: str | Path,
    *,
    clip_id: str = "",
) -> ConversionResult:
    if target_format not in FORMAT_VERSIONS:
        error = DiagnosticError(
            "E_CONVERSION_FAILED",
            f"unknown target format {target_format!r}",
            stage="convert",
            format_version=target_format,
            clip_id=clip_id,
        )
        return ConversionResult(package, None, target_format, "", {"failures": [error.to_dict()]}, [error])
    key = conversion_cache_key(package, target_format)
    root = Path(destination_segment_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        if target_format == "v1":
            target_digest = _write_v1(package, root)
        elif target_format == "v2":
            target_digest = _write_v2(package, root)
        else:
            target_digest = _write_v3(package, root)
        parsed = parse_package(root, target_format, clip_id=clip_id)
        if not parsed.ok or parsed.package is None:
            failures = list(parsed.failures)
            record = {
                "conversion_key": key,
                "source_format": package.format_version,
                "target_format": target_format,
                "source_package_fingerprint": package.package_fingerprint,
                "status": "failed",
                "losses": [],
                "failures": [item.to_dict() for item in failures],
            }
            return ConversionResult(package, None, target_format, key, record, failures)
        target_package = parsed.package
        losses = lost_fields_for_target(package, target_format)
        record = {
            "conversion_key": key,
            "source_format": package.format_version,
            "target_format": target_format,
            "source_package_fingerprint": package.package_fingerprint,
            "target_package_fingerprint": target_package.package_fingerprint,
            "source_sidecar_sha256": sha256_bytes(package.sidecar_bytes),
            "target_sidecar_sha256": sha256_bytes(target_package.sidecar_bytes),
            "target_digest": target_digest,
            "status": "ok",
            "losses": sorted(
                losses,
                key=lambda item: (item["source_track"], item["field"], str(item["value"])),
            ),
            "failures": [],
        }
        return ConversionResult(package, target_package, target_format, key, record, [])
    except (OSError, ValueError, DiagnosticError) as exc:
        error = exc if isinstance(exc, DiagnosticError) else DiagnosticError(
            "E_CONVERSION_FAILED",
            f"conversion {package.format_version}->{target_format} failed: {exc}",
            stage="convert",
            clip_id=clip_id,
        )
        return ConversionResult(
            package,
            None,
            target_format,
            key,
            {
                "conversion_key": key,
                "source_format": package.format_version,
                "target_format": target_format,
                "status": "failed",
                "losses": [],
                "failures": [error.to_dict()],
            },
            [error],
        )


def write_conversion_record(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "stemscope/conversion-records/v1", "conversions": records}
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    os.replace(temporary, output)
