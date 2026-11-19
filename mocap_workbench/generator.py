"""Deterministic synthetic mocap generation and artifact loading."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    DEFECT_JITTER,
    DEFECT_JUMP,
    DEFECT_MISSING,
    DEFECT_ROOT_SLIP,
    FPS,
    FRAME_COUNTS,
    INVALID_QUAT,
    ROOT_NAME,
    SCHEMA_DEFECTS,
    SCHEMA_RECORD,
    SCHEMA_TRUTH,
    SKELETON,
)
from .errors import BusinessError, ErrorCode, require
from .io import (
    atomic_write_json,
    canonical_json_bytes,
    clean_float,
    clean_vector,
    read_json,
    sha256_json,
)
from .model import DefectEntry, Motion


class SplitMix64:
    """Small platform-independent PRNG used only for deterministic scheduling."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int | str) -> None:
        if isinstance(seed, str):
            seed = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
        self.state = int(seed) & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (value ^ (value >> 31)) & self.MASK

    def randrange(self, start: int, stop: int) -> int:
        if stop <= start:
            raise ValueError("empty PRNG range")
        return start + self.next_u64() % (stop - start)

    def uniform(self, low: float, high: float) -> float:
        unit = self.next_u64() / float(1 << 64)
        return low + (high - low) * unit

    def choice(self, values: tuple[Any, ...]) -> Any:
        return values[self.randrange(0, len(values))]


@dataclass(frozen=True)
class GeneratedBundle:
    record: Motion
    truth: Motion
    defects: tuple[DefectEntry, ...]
    record_path: Path
    truth_path: Path
    defects_path: Path

    def paths(self) -> dict[str, str]:
        return {
            "record": str(self.record_path),
            "truth": str(self.truth_path),
            "defects": str(self.defects_path),
        }


def _unit_axis(seed: int, joint: str) -> np.ndarray:
    digest = hashlib.sha256(f"axis:{seed}:{joint}".encode("utf-8")).digest()
    raw = np.array(
        [int.from_bytes(digest[i : i + 4], "big") / 2**32 * 2.0 - 1.0 for i in (0, 4, 8)]
    )
    length = float(np.linalg.norm(raw))
    if length < 1e-8:
        return np.array([0.0, 1.0, 0.0])
    return raw / length


def _joint_params(seed: int, joint: str) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(f"motion:{seed}:{joint}".encode("utf-8")).digest()
    unit = [int.from_bytes(digest[i : i + 4], "big") / 2**32 for i in (0, 4, 8, 12)]
    # Slow, non-commensurate local oscillations keep the truth smooth but not trivial.
    base_angle = 0.08 + 0.24 * (unit[0] - 0.5)
    amplitude = 0.22 + 0.32 * unit[1]
    # Keep the base motion comfortably band-limited relative to sparse keyframe
    # spacing. Faster local oscillations would make truthful SLERP between
    # defect endpoints inherently exceed the declared 2-degree evaluation budget.
    frequency = 0.10 + 0.18 * unit[2]
    phase = 2.0 * math.pi * unit[3]
    return base_angle, amplitude, frequency, phase


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    half = angle * 0.5
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _normalize_sign(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return q.copy()
    q = q / norm
    # Stable canonical storage choice for generated truth.
    if q[0] < 0.0 or (q[0] == 0.0 and q[1] < 0.0):
        q = -q
    return q


def _base_motion(seed: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(frame_count, dtype=np.float64) / float(FPS)
    quats = np.empty((frame_count, len(SKELETON), 4), dtype=np.float64)
    for joint_index, joint in enumerate(SKELETON):
        base_angle, amplitude, frequency, phase = _joint_params(seed, joint)
        secondary = 0.08 * np.sin(2.0 * math.pi * (frequency * 2.3) * times + phase * 0.7)
        angle = base_angle + amplitude * np.sin(2.0 * math.pi * frequency * times + phase) + secondary
        axis = _unit_axis(seed, joint)
        half = angle * 0.5
        quats[:, joint_index, 0] = np.cos(half)
        quats[:, joint_index, 1:] = np.sin(half)[:, None] * axis[None, :]
        norm = np.linalg.norm(quats[:, joint_index, :], axis=1, keepdims=True)
        quats[:, joint_index, :] /= norm
        flip = quats[:, joint_index, 0] < 0.0
        quats[flip, joint_index, :] *= -1.0

    roots = np.empty((frame_count, 3), dtype=np.float64)
    roots[:, 0] = 0.0045 * times + 0.065 * np.sin(2.0 * math.pi * 0.17 * times + 0.3)
    roots[:, 1] = 0.94 + 0.018 * np.sin(2.0 * math.pi * 0.31 * times + 1.1)
    roots[:, 2] = 0.12 * np.sin(2.0 * math.pi * 0.23 * times + 2.0)
    return quats, roots


def _overlaps(occupied: set[tuple[int, int]], frames: range, joints: list[int], buffer: int = 1) -> bool:
    for frame in frames:
        for joint in joints:
            for delta in range(-buffer, buffer + 1):
                if (frame + delta, joint) in occupied:
                    return True
    return False


def _choose_joint_block(
    rng: SplitMix64,
    frame_count: int,
    occupied: set[tuple[int, int]],
    length_range: tuple[int, int],
    joint_range: tuple[int, int],
    margin: int = 8,
) -> tuple[int, int, list[int]] | None:
    length = rng.randrange(length_range[0], length_range[1] + 1)
    joint_count = rng.randrange(joint_range[0], joint_range[1] + 1)
    joints = [rng.randrange(0, len(SKELETON)) for _ in range(joint_count)]
    joints = list(dict.fromkeys(joints))
    for _attempt in range(1200):
        start = rng.randrange(margin, max(margin + 1, frame_count - margin - length))
        frames = range(start, start + length)
        if not _overlaps(occupied, frames, joints, buffer=2):
            return start, length, joints
    return None


def _inject_defects(
    quats: np.ndarray, roots: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple[DefectEntry, ...]]:
    frame_count = quats.shape[0]
    dirty_q = quats.copy()
    dirty_root = roots.copy()
    entries: list[DefectEntry] = []
    occupied: set[tuple[int, int]] = set()
    root_occupied: set[int] = set()
    segment_id = 1

    event_count = max(2, frame_count // 900)
    rng = SplitMix64(seed ^ 0xD3FEC7A9)

    # Single-frame quaternion jumps. A jump manifests as a one-frame outlier;
    # the detector resolves its two transition shoulders back to the peak frame.
    created = 0
    attempts = 0
    while created < event_count and attempts < event_count * 80:
        attempts += 1
        frame = rng.randrange(8, frame_count - 8)
        joint = rng.randrange(0, len(SKELETON))
        if any((frame + d, joint) in occupied for d in range(-3, 4)):
            continue
        axis = _unit_axis(seed ^ frame, SKELETON[joint])
        angle = math.radians(rng.uniform(48.0, 72.0))
        extra = _quat_from_axis_angle(axis, angle)
        dirty_q[frame, joint, :] = _quat_multiply(extra, quats[frame, joint, :])
        occupied.add((frame, joint))
        entries.append(DefectEntry(frame, SKELETON[joint], DEFECT_JUMP, segment_id))
        segment_id += 1
        created += 1

    # Continuous high-frequency, small-amplitude jitter blocks.
    created = 0
    attempts = 0
    while created < event_count and attempts < event_count * 80:
        attempts += 1
        chosen = _choose_joint_block(rng, frame_count, occupied, (8, 13), (1, 2))
        if chosen is None:
            continue
        start, length, joints = chosen
        for joint in joints:
            axis = _unit_axis(seed ^ 0x7147, SKELETON[joint])
            amplitude = math.radians(rng.uniform(2.1, 2.9))
            phase = rng.uniform(0.0, math.pi)
            for offset in range(length):
                frame = start + offset
                # Sample a high-frequency sine at Nyquist: +A, -A, +A...
                # A random phase can land on zero every integer frame, which
                # would make the "jitter" block physically absent.
                angle = amplitude * math.sin(math.pi * (offset + 0.5))
                extra = _quat_from_axis_angle(axis, angle)
                dirty_q[frame, joint, :] = _quat_multiply(extra, quats[frame, joint, :])
                occupied.add((frame, joint))
                entries.append(DefectEntry(frame, SKELETON[joint], DEFECT_JITTER, segment_id))
        segment_id += 1
        created += 1

    # Missing markers are explicit invalid placeholders, never silently repaired.
    created = 0
    attempts = 0
    while created < event_count and attempts < event_count * 80:
        attempts += 1
        chosen = _choose_joint_block(rng, frame_count, occupied, (3, 8), (1, 2))
        if chosen is None:
            continue
        start, length, joints = chosen
        for joint in joints:
            for offset in range(length):
                frame = start + offset
                dirty_q[frame, joint, :] = INVALID_QUAT
                occupied.add((frame, joint))
                entries.append(DefectEntry(frame, SKELETON[joint], DEFECT_MISSING, segment_id))
        segment_id += 1
        created += 1

    # Instantaneous root displacement slips are single-frame offsets. Midpoint
    # residual attribution resolves the surrounding transition shoulders to the
    # true slipped frame.
    created = 0
    attempts = 0
    while created < event_count and attempts < event_count * 80:
        attempts += 1
        frame = rng.randrange(8, frame_count - 8)
        if any(abs(frame - existing) <= 4 for existing in root_occupied):
            continue
        offset = np.array(
            [rng.uniform(0.24, 0.46), rng.uniform(-0.12, 0.12), rng.uniform(-0.34, 0.34)]
        )
        dirty_root[frame, :] = roots[frame, :] + offset
        root_occupied.add(frame)
        entries.append(DefectEntry(frame, ROOT_NAME, DEFECT_ROOT_SLIP, segment_id))
        segment_id += 1
        created += 1

    entries.sort(key=lambda entry: (entry.frame, entry.joint, entry.defect_type))
    return dirty_q, dirty_root, tuple(entries)


def _motion_payload(
    schema: str, seed: int, frame_count: int, quats: np.ndarray, roots: np.ndarray
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame in range(frame_count):
        joints = {
            joint: clean_vector(quats[frame, joint_index, :])
            for joint_index, joint in enumerate(SKELETON)
        }
        frames.append(
            {
                "frame": frame,
                "root": clean_vector(roots[frame, :]),
                "joints": joints,
            }
        )
    return {
        "schema": schema,
        "seed": int(seed),
        "fps": FPS,
        "frame_count": int(frame_count),
        "skeleton": list(SKELETON),
        "quaternion_order": "wxyz",
        "root_units": "metres",
        "frames": frames,
    }


def write_motion(path: Path, schema: str, seed: int, quats: np.ndarray, roots: np.ndarray) -> str:
    core = _motion_payload(schema, seed, quats.shape[0], quats, roots)
    content_hash = sha256_json(core)
    payload = dict(core)
    payload["content_sha256"] = content_hash
    atomic_write_json(path, payload)
    return content_hash


def load_motion(path: Path, expected_schema: str | None = None) -> Motion:
    path = Path(path)
    if not path.exists():
        raise BusinessError(ErrorCode.E_FRAME_MISSING, "motion file is missing", {"path": str(path)})
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID, "motion file is not valid JSON", {"path": str(path)}
        ) from exc
    schema = payload.get("schema")
    if expected_schema and schema != expected_schema:
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID,
            "unexpected motion schema",
            {"path": str(path), "expected": expected_schema, "actual": schema},
        )
    frame_count = int(payload.get("frame_count", -1))
    if frame_count not in FRAME_COUNTS:
        raise BusinessError(
            ErrorCode.E_FRAME_MISSING,
            "record frame count is not one of the supported tiers",
            {"path": str(path), "frame_count": frame_count},
        )
    names = tuple(payload.get("skeleton", []))
    if names != SKELETON:
        raise BusinessError(
            ErrorCode.E_FRAME_MISSING,
            "record skeleton does not match the fixed skeleton",
            {"path": str(path), "skeleton": list(names)},
        )
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise BusinessError(
            ErrorCode.E_FRAME_MISSING,
            "declared frame count does not match stored frames",
            {"path": str(path), "declared": frame_count, "stored": len(frames) if isinstance(frames, list) else None},
        )
    quats = np.empty((frame_count, len(SKELETON), 4), dtype=np.float64)
    roots = np.empty((frame_count, 3), dtype=np.float64)
    for expected_frame, frame_data in enumerate(frames):
        if int(frame_data.get("frame", -1)) != expected_frame:
            raise BusinessError(
                ErrorCode.E_FRAME_MISSING,
                "record has a missing or out-of-order frame",
                {"path": str(path), "expected": expected_frame, "actual": frame_data.get("frame")},
            )
        root = frame_data.get("root")
        if not isinstance(root, list) or len(root) != 3:
            raise BusinessError(
                ErrorCode.E_QUAT_INVALID,
                "root displacement is missing or malformed",
                {"path": str(path), "frame": expected_frame},
            )
        roots[expected_frame, :] = root
        joints = frame_data.get("joints", {})
        for joint_index, joint in enumerate(SKELETON):
            q = joints.get(joint)
            if not isinstance(q, list) or len(q) != 4:
                raise BusinessError(
                    ErrorCode.E_QUAT_INVALID,
                    "joint quaternion is missing or malformed",
                    {"path": str(path), "frame": expected_frame, "joint": joint},
                )
            quats[expected_frame, joint_index, :] = q

    if not np.all(np.isfinite(quats)) or not np.all(np.isfinite(roots)):
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID,
            "record contains non-finite values",
            {"path": str(path)},
        )
    norms = np.linalg.norm(quats, axis=2)
    unit = np.abs(norms - 1.0) <= 2e-7
    missing = norms <= 1e-12
    if expected_schema == SCHEMA_TRUTH and not np.all(unit):
        frame, joint = np.argwhere(~unit)[0]
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID,
            "truth contains a non-unit quaternion",
            {"path": str(path), "frame": int(frame), "joint": SKELETON[int(joint)]},
        )
    if expected_schema != SCHEMA_TRUTH and not np.all(unit | missing):
        frame, joint = np.argwhere(~(unit | missing))[0]
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID,
            "record contains a non-unit non-placeholder quaternion",
            {
                "path": str(path),
                "frame": int(frame),
                "joint": SKELETON[int(joint)],
                "norm": float(norms[frame, joint]),
            },
        )
    core = {key: payload[key] for key in payload if key != "content_sha256"}
    content_hash = sha256_json(core)
    if payload.get("content_sha256") != content_hash:
        raise BusinessError(
            ErrorCode.E_QUAT_INVALID,
            "motion content hash mismatch",
            {"path": str(path), "stored": payload.get("content_sha256"), "actual": content_hash},
        )
    return Motion(
        seed=int(payload.get("seed", 0)),
        frame_count=frame_count,
        fps=int(payload.get("fps", FPS)),
        joint_names=SKELETON,
        quats=quats,
        roots=roots,
        content_sha256=content_hash,
    )


def defects_payload(seed: int, frame_count: int, entries: tuple[DefectEntry, ...]) -> dict[str, Any]:
    body = {
        "schema": SCHEMA_DEFECTS,
        "seed": seed,
        "frame_count": frame_count,
        "entries": [entry.to_dict() for entry in entries],
    }
    body["entries_sha256"] = sha256_json(
        [{k: v for k, v in entry.items() if k != "segment_id"} for entry in body["entries"]]
    )
    return body


def load_defects(path: Path) -> tuple[DefectEntry, ...]:
    payload = read_json(path)
    require(
        payload.get("schema") == SCHEMA_DEFECTS,
        ErrorCode.E_FRAME_MISSING,
        "unexpected defect manifest schema",
        path=str(path),
    )
    entries = []
    for item in payload.get("entries", []):
        entries.append(
            DefectEntry(
                frame=int(item["frame"]),
                joint=str(item["joint"]),
                defect_type=str(item["type"]),
                segment_id=int(item.get("segment_id", 0)),
            )
        )
    return tuple(entries)


def generate_bundle(out_dir: Path, frame_count: int, seed: int) -> GeneratedBundle:
    require(
        frame_count in FRAME_COUNTS,
        ErrorCode.E_FRAME_MISSING,
        "frame count must be one of 1200, 6000 or 12000",
        frame_count=frame_count,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_quats, base_roots = _base_motion(seed, frame_count)
    dirty_quats, dirty_roots, entries = _inject_defects(base_quats, base_roots, seed)
    record_path = out_dir / "record.json"
    truth_path = out_dir / "truth.json"
    defects_path = out_dir / "defects.json"
    record_hash = write_motion(record_path, SCHEMA_RECORD, seed, dirty_quats, dirty_roots)
    truth_hash = write_motion(truth_path, SCHEMA_TRUTH, seed, base_quats, base_roots)
    atomic_write_json(defects_path, defects_payload(seed, frame_count, entries))
    record = Motion(seed, frame_count, FPS, SKELETON, dirty_quats, dirty_roots, record_hash)
    truth = Motion(seed, frame_count, FPS, SKELETON, base_quats, base_roots, truth_hash)
    return GeneratedBundle(record, truth, entries, record_path, truth_path, defects_path)


def bundle_hashes(bundle: GeneratedBundle) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("record", bundle.record_path),
            ("truth", bundle.truth_path),
            ("defects", bundle.defects_path),
        )
    }


def load_bundle(directory: Path) -> GeneratedBundle:
    directory = Path(directory)
    record_path = directory / "record.json"
    truth_path = directory / "truth.json"
    defects_path = directory / "defects.json"
    record = load_motion(record_path, SCHEMA_RECORD)
    truth = load_motion(truth_path, SCHEMA_TRUTH)
    if record.frame_count != truth.frame_count:
        raise BusinessError(
            ErrorCode.E_FRAME_MISSING,
            "record and truth frame counts differ",
            {"record": record.frame_count, "truth": truth.frame_count},
        )
    defects = load_defects(defects_path)
    return GeneratedBundle(record, truth, defects, record_path, truth_path, defects_path)


def canonical_bundle_bytes(bundle: GeneratedBundle) -> dict[str, bytes]:
    return {
        "record": canonical_json_bytes(read_json(bundle.record_path)),
        "truth": canonical_json_bytes(read_json(bundle.truth_path)),
        "defects": canonical_json_bytes(read_json(bundle.defects_path)),
    }
