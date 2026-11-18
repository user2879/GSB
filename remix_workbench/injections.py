from __future__ import annotations

import json
import shutil
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .formats import (
    V1_AUDIO_FILE,
    V2_MANIFEST_FILE,
    V3_HEADER,
    V3_PACKAGE_FILE,
    V3_TABLE_ENTRY,
    V3_TABLE_PREFIX,
)


@dataclass(frozen=True)
class Injection:
    id: str
    stage: str
    kind: str
    clip_id: str | None
    track: str | None = None
    format_version: str | None = None
    value: Any = None
    byte_count: int | None = None
    expected_code: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Injection":
        return cls(
            id=str(data["id"]),
            stage=str(data["stage"]),
            kind=str(data["kind"]),
            clip_id=data.get("clip_id"),
            track=data.get("track"),
            format_version=data.get("format_version"),
            value=data.get("value"),
            byte_count=data.get("bytes"),
            expected_code=data.get("expected_code"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "kind": self.kind,
            "clip_id": self.clip_id,
            "track": self.track,
            "format_version": self.format_version,
            "value": self.value,
            "bytes": self.byte_count,
            "expected_code": self.expected_code,
        }


def load_injections(path: str | Path | None) -> list[Injection]:
    if path is None:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = raw.get("injections", [])
    else:
        raise ValueError("injection config must be an object or array")
    if not isinstance(entries, list):
        raise ValueError("injection config .injections must be an array")
    return [Injection.from_dict(item) for item in entries]


def _clone_package(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _truncate_wav(package: Path, version: str, track: str | None, count: int) -> Path:
    if version == "v1":
        target = package / V1_AUDIO_FILE
    elif version == "v2":
        manifest = json.loads((package / V2_MANIFEST_FILE).read_text(encoding="utf-8"))
        entry = manifest.get("tracks", {}).get(track or "")
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError(f"cannot resolve v2 track {track!r} for truncation")
        target = package / entry["file"]
    else:
        target = package / V3_PACKAGE_FILE
    data = target.read_bytes()
    keep = max(0, len(data) - max(1, count))
    target.write_bytes(data[:keep])
    return target


def _v3_offset_out_of_bounds(package: Path, track: str | None) -> None:
    path = package / V3_PACKAGE_FILE
    blob = bytearray(path.read_bytes())
    if len(blob) < V3_HEADER.size:
        return
    _, _, track_count, _, _, _, _ = V3_HEADER.unpack_from(blob, 0)
    cursor = V3_HEADER.size
    for _ in range(track_count):
        (name_len,) = V3_TABLE_PREFIX.unpack_from(blob, cursor)
        cursor += V3_TABLE_PREFIX.size
        name = bytes(blob[cursor : cursor + name_len]).decode("utf-8")
        cursor += name_len
        entry_offset = cursor
        old_offset, byte_length = V3_TABLE_ENTRY.unpack_from(blob, cursor)
        cursor += V3_TABLE_ENTRY.size
        if track is None or name == track:
            new_offset = len(blob) + 4096
            struct.pack_into("<I", blob, entry_offset, new_offset)
            break
    path.write_bytes(bytes(blob))


def _v2_frame_count_mismatch(package: Path, track: str | None) -> None:
    path = package / V2_MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("tracks", {})
    if not isinstance(entries, dict) or track not in entries:
        return
    entry = entries[track]
    if isinstance(entry, dict):
        entry["frames"] = int(entry.get("frames", manifest.get("frames", 0))) + 1
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def apply_parse_faults(
    source_package: str | Path,
    format_version: str,
    injections: Iterable[Injection],
    destination: str | Path,
) -> tuple[Path, list[Injection]]:
    injections = [item for item in injections if item.stage == "parse"]
    if not injections:
        return Path(source_package), []
    destination = Path(destination)
    _clone_package(Path(source_package), destination)
    applied: list[Injection] = []
    for item in sorted(injections, key=lambda value: value.id):
        if item.format_version and item.format_version != format_version:
            continue
        if item.kind == "truncate_wav_bytes":
            _truncate_wav(
                destination,
                format_version,
                item.track,
                int(item.byte_count or 1),
            )
        elif item.kind == "offset_table_out_of_bounds":
            if format_version != "v3":
                continue
            _v3_offset_out_of_bounds(destination, item.track)
        elif item.kind == "frame_count_mismatch":
            if format_version != "v2":
                continue
            _v2_frame_count_mismatch(destination, item.track)
        else:
            raise ValueError(f"unsupported parse injection kind {item.kind!r}")
        applied.append(item)
    (destination / "injection_applied.json").write_text(
        json.dumps(
            {
                "injections": [item.as_dict() for item in applied],
                "fault_id": uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "|".join(sorted(item.id for item in applied)),
                ).hex,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination, applied


def injections_for_clip(injections: Iterable[Injection], clip_id: str) -> list[Injection]:
    return sorted(
        [item for item in injections if item.clip_id in (None, clip_id)],
        key=lambda item: item.id,
    )


def gain_injection(injections: Iterable[Injection], clip_id: str, track: str) -> Injection | None:
    for item in injections:
        if (
            item.stage == "gain"
            and item.kind in {"gain_nonfinite", "gain_negative"}
            and item.clip_id in (None, clip_id)
            and item.track in (None, track)
        ):
            return item
    return None


def config_injections(injections: Iterable[Injection]) -> list[Injection]:
    return sorted(
        [
            item
            for item in injections
            if item.stage in {"config", "mix-config"}
        ],
        key=lambda item: item.id,
    )

