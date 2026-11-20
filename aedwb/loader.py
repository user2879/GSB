"""Packet discovery, injection application, and strict raw validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import SCORE_TYPES
from .util import sha256_json, sha256_text


@dataclass
class PacketSource:
    path: Path
    relative_path: str
    raw_text: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    applied_injections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fallback_packet_id(self) -> str:
        if isinstance(self.parsed, dict) and isinstance(self.parsed.get("packet_id"), str):
            return self.parsed["packet_id"]
        for injection in self.applied_injections:
            value = injection.get("packet_id")
            if isinstance(value, str) and value:
                return value
        return f"<invalid:{self.path.stem}>"


def _relative_path(sample_dir: Path, path: Path) -> str:
    return path.relative_to(sample_dir).as_posix()


def _injection_matches(
    injection: dict[str, Any], relative_path: str, parsed: dict[str, Any] | None
) -> bool:
    target_file = injection.get("packet_file")
    if isinstance(target_file, str) and target_file.replace("\\", "/").lstrip("./") == relative_path:
        return True
    target_id = injection.get("packet_id")
    return bool(parsed and isinstance(target_id, str) and parsed.get("packet_id") == target_id)


def _apply_post_parse_injection(packet: dict[str, Any], kind: str) -> None:
    if kind == "frame_count_mismatch":
        frames = packet.get("frames")
        if isinstance(frames, list) and frames:
            packet["frames"] = frames[:-1]
        return
    if kind == "missing_score_type":
        packet.pop("score_type", None)
        return
    if kind == "nonfinite_logit":
        frames = packet.get("frames")
        if isinstance(frames, list) and frames and isinstance(frames[0], list) and frames[0]:
            frames[0][0] = float("inf")
        return
    if kind == "unmappable_codebook":
        packet["class_codebook_version"] = "__injected_missing_codebook__"
        return
    raise ValueError(f"unsupported post-parse injection: {kind}")


def discover_packet_sources(
    sample_dir: str | Path,
    injections: list[dict[str, Any]],
) -> list[PacketSource]:
    """Read every packet file in stable relative-path order and apply injections."""

    sample_dir = Path(sample_dir)
    packet_dir = sample_dir / "packets"
    sources: list[PacketSource] = []
    consumed: set[str] = set()
    for path in sorted(packet_dir.glob("*.json"), key=lambda item: item.name):
        relative_path = _relative_path(sample_dir, path)
        raw_text = path.read_text(encoding="utf-8")
        matching = [
            item
            for item in injections
            if item.get("packet_file", "").replace("\\", "/").lstrip("./") == relative_path
        ]
        pre_parse = [item for item in matching if item.get("kind") == "truncated_packet"]
        if pre_parse:
            injection = pre_parse[0]
            consumed.add(injection["injection_id"])
            raw_text = raw_text[: max(16, len(raw_text) // 2)]
            sources.append(
                PacketSource(
                    path=path,
                    relative_path=relative_path,
                    raw_text=raw_text,
                    parsed=None,
                    parse_error="truncated packet JSON",
                    applied_injections=[injection],
                )
            )
            continue

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as error:
            sources.append(
                PacketSource(
                    path=path,
                    relative_path=relative_path,
                    raw_text=raw_text,
                    parsed=None,
                    parse_error=f"invalid JSON: {error.msg} at line {error.lineno}",
                )
            )
            continue
        if not isinstance(parsed, dict):
            sources.append(
                PacketSource(
                    path=path,
                    relative_path=relative_path,
                    raw_text=raw_text,
                    parsed=None,
                    parse_error="packet JSON root must be an object",
                )
            )
            continue

        applied: list[dict[str, Any]] = []
        for injection in injections:
            if injection["injection_id"] in consumed:
                continue
            if not _injection_matches(injection, relative_path, parsed):
                continue
            _apply_post_parse_injection(parsed, str(injection["kind"]))
            consumed.add(injection["injection_id"])
            applied.append(injection)

        sources.append(
            PacketSource(
                path=path,
                relative_path=relative_path,
                raw_text=raw_text,
                parsed=parsed,
                parse_error=None,
                applied_injections=applied,
            )
        )
    return sources


def packet_content_hash(source: PacketSource) -> str:
    # The original text is intentionally retained as the content hash source.
    # In-memory injections are represented by their stable IDs because one of
    # them deliberately introduces a non-finite float (not valid strict JSON).
    return sha256_json(
        {
            "raw_text_sha256": sha256_text(source.raw_text),
            "applied_injection_ids": sorted(
                item["injection_id"] for item in source.applied_injections
            ),
        }
    )


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _invalid_payload(
    source: PacketSource,
    errors: list[dict[str, str]],
    duplicate_ids: dict[str, int],
) -> dict[str, Any]:
    return {
        "valid": False,
        "packet_id": source.fallback_packet_id,
        "source_file": source.relative_path,
        "raw_text_sha256": sha256_text(source.raw_text),
        "content_sha256": packet_content_hash(source),
        "raw_score_type": None,
        "frames": [],
        "frame_count": 0,
        "header": {},
        "audio_file": None,
        "start_ms": None,
        "end_ms": None,
        "frame_shift_ms": None,
        "class_codebook_version": None,
        "frame_grid_version": None,
        "detector_version": None,
        "errors": sorted(errors, key=lambda item: (item["code"], item["message"])),
        "injection_ids": sorted(item["injection_id"] for item in source.applied_injections),
        "duplicate_packet_id_count": duplicate_ids.get(source.fallback_packet_id, 0),
    }


def parse_packet(
    source: PacketSource,
    audio_metadata: dict[str, dict[str, Any]],
    duplicate_ids: dict[str, int],
    class_count: int,
) -> dict[str, Any]:
    """Validate a packet and retain its original scores and score-type provenance."""

    errors: list[dict[str, str]] = []
    if source.parsed is None:
        errors.append({"code": "E_PACKET_INVALID", "message": source.parse_error or "invalid packet"})
        return _invalid_payload(source, errors, duplicate_ids)

    packet = source.parsed
    required = (
        "packet_id",
        "audio_file",
        "start_ms",
        "end_ms",
        "frame_shift_ms",
        "score_type",
        "class_codebook_version",
        "frame_grid_version",
        "frames",
    )
    missing = [name for name in required if name not in packet]
    for name in missing:
        errors.append({"code": "E_PACKET_INVALID", "message": f"missing required field: {name}"})

    packet_id = packet.get("packet_id")
    if not isinstance(packet_id, str) or not packet_id:
        errors.append({"code": "E_PACKET_INVALID", "message": "packet_id must be a non-empty string"})
    elif duplicate_ids.get(packet_id, 0) > 1:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": f"duplicate packet_id appears {duplicate_ids[packet_id]} times",
            }
        )

    score_type = packet.get("score_type")
    if score_type not in SCORE_TYPES:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": "score_type must be exactly 'logit' or 'probability'",
            }
        )

    audio_file = packet.get("audio_file")
    audio = audio_metadata.get(audio_file) if isinstance(audio_file, str) else None
    if audio is None:
        errors.append(
            {"code": "E_PACKET_INVALID", "message": f"unknown audio_file: {audio_file!r}"}
        )

    start_ms = packet.get("start_ms")
    end_ms = packet.get("end_ms")
    shift_ms = packet.get("frame_shift_ms")
    integer_header_fields = {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "frame_shift_ms": shift_ms,
    }
    for name, value in integer_header_fields.items():
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(
                {"code": "E_PACKET_INVALID", "message": f"{name} must be an integer"}
            )

    if (
        isinstance(start_ms, int)
        and not isinstance(start_ms, bool)
        and isinstance(end_ms, int)
        and not isinstance(end_ms, bool)
        and isinstance(shift_ms, int)
        and not isinstance(shift_ms, bool)
    ):
        if start_ms < 0 or end_ms <= start_ms:
            errors.append(
                {"code": "E_PACKET_INVALID", "message": "start_ms/end_ms must form a positive interval"}
            )
        if shift_ms <= 0:
            errors.append(
                {"code": "E_PACKET_INVALID", "message": "frame_shift_ms must be positive"}
            )
        if audio is not None:
            if end_ms > audio["duration_ms"] + 1e-9 or start_ms < 0:
                errors.append(
                    {
                        "code": "E_PACKET_INVALID",
                        "message": "packet interval lies outside audio duration",
                    }
                )
        if end_ms > start_ms and shift_ms > 0:
            duration_ms = end_ms - start_ms
            if duration_ms % shift_ms != 0:
                errors.append(
                    {
                        "code": "E_PACKET_INVALID",
                        "message": "interval duration is not divisible by frame_shift_ms",
                    }
                )

    frames = packet.get("frames")
    expected_frames: int | None = None
    if (
        isinstance(start_ms, int)
        and not isinstance(start_ms, bool)
        and isinstance(end_ms, int)
        and not isinstance(end_ms, bool)
        and isinstance(shift_ms, int)
        and not isinstance(shift_ms, bool)
        and shift_ms > 0
        and end_ms > start_ms
        and (end_ms - start_ms) % shift_ms == 0
    ):
        expected_frames = (end_ms - start_ms) // shift_ms
    if not isinstance(frames, list):
        errors.append({"code": "E_PACKET_INVALID", "message": "frames must be a list"})
        frames = []
    elif expected_frames is not None and len(frames) != expected_frames:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": f"frame count {len(frames)} != expected {expected_frames}",
            }
        )

    bad_width = False
    nonfinite_count = 0
    non_integer_logit_count = 0
    out_of_range_probability_count = 0
    for frame_index, row in enumerate(frames):
        if not isinstance(row, list) or len(row) != class_count:
            bad_width = True
            continue
        for class_index, value in enumerate(row):
            if not _finite_number(value):
                nonfinite_count += 1
                continue
            if score_type == "logit" and (isinstance(value, bool) or not isinstance(value, int)):
                non_integer_logit_count += 1
            if score_type == "probability" and not (0.0 <= float(value) <= 1.0):
                out_of_range_probability_count += 1
    if bad_width:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": f"every frame must contain exactly {class_count} class values",
            }
        )
    if nonfinite_count:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": f"{nonfinite_count} score value(s) are non-finite",
            }
        )
    if non_integer_logit_count:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": f"{non_integer_logit_count} logit value(s) are not integers",
            }
        )
    if out_of_range_probability_count:
        errors.append(
            {
                "code": "E_PACKET_INVALID",
                "message": f"{out_of_range_probability_count} probability value(s) outside [0,1]",
            }
        )

    for name in ("class_codebook_version", "frame_grid_version"):
        if not isinstance(packet.get(name), str) or not packet.get(name):
            errors.append(
                {"code": "E_PACKET_INVALID", "message": f"{name} must be a non-empty string"}
            )

    if errors:
        payload = _invalid_payload(source, errors, duplicate_ids)
        frame_count = len(frames) if isinstance(frames, list) else 0
        payload.update(
            {
                "packet_id": packet_id if isinstance(packet_id, str) else source.fallback_packet_id,
                "raw_score_type": score_type if isinstance(score_type, str) else None,
                "header": {
                    name: packet.get(name)
                    for name in (
                        "audio_file",
                        "start_ms",
                        "end_ms",
                        "frame_shift_ms",
                        "class_codebook_version",
                        "frame_grid_version",
                        "detector_version",
                    )
                },
                "audio_file": audio_file if isinstance(audio_file, str) else None,
                "start_ms": start_ms if isinstance(start_ms, int) else None,
                "end_ms": end_ms if isinstance(end_ms, int) else None,
                "frame_shift_ms": shift_ms if isinstance(shift_ms, int) else None,
                "class_codebook_version": packet.get("class_codebook_version"),
                "frame_grid_version": packet.get("frame_grid_version"),
                "detector_version": packet.get("detector_version"),
                # Invalid raw values (including injected non-finite floats) are
                # represented by the validation error and frame count, not
                # persisted as non-strict JSON.
                "frames": [],
                "frame_count": frame_count,
            }
        )
        return payload

    return {
        "valid": True,
        "packet_id": packet_id,
        "source_file": source.relative_path,
        "raw_text_sha256": sha256_text(source.raw_text),
        "content_sha256": packet_content_hash(source),
        "raw_score_type": score_type,
        "frames": frames,
        "frame_count": len(frames),
        "header": {
            name: packet.get(name)
            for name in (
                "audio_file",
                "start_ms",
                "end_ms",
                "frame_shift_ms",
                "class_codebook_version",
                "frame_grid_version",
                "detector_version",
            )
        },
        "audio_file": audio_file,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "frame_shift_ms": shift_ms,
        "class_codebook_version": packet["class_codebook_version"],
        "frame_grid_version": packet["frame_grid_version"],
        "detector_version": packet.get("detector_version"),
        "errors": [],
        "injection_ids": sorted(item["injection_id"] for item in source.applied_injections),
        "duplicate_packet_id_count": 1,
    }
