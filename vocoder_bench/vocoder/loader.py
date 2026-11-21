"""Input scanning and validation for reference WAVs and feature frame JSONL."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hashing import canonical_hash
from .wav_io import AudioFormatError, read_wav


@dataclass
class UtteranceInput:
    utterance_id: str
    wav_path: Path
    frames_path: Path
    frames: list[dict] = field(default_factory=list)
    frame_fingerprints: list[str] = field(default_factory=list)
    reference: np.ndarray | None = None
    feature_fingerprint: str = ""


@dataclass
class LoadFailure:
    utterance_id: str
    code: str
    message: str
    stage: str = "load"


def _pair_files(input_dir: Path) -> list[tuple[str, Path | None, Path]]:
    pairs = []
    for frames_path in sorted(input_dir.glob("*.frames.jsonl")):
        utter_id = frames_path.name[: -len(".frames.jsonl")]
        wav_path = input_dir / f"{utter_id}.wav"
        pairs.append((utter_id, wav_path if wav_path.exists() else None, frames_path))
    return pairs


def _validate_frames(frames: list[dict]) -> tuple[list[dict], str, list[str]]:
    if not frames:
        raise ValueError("empty frame file")
    harmonic_limit = len(frames[0].get("harmonic_amplitudes", []))
    if harmonic_limit < 1:
        raise ValueError("frame 0: empty harmonic_amplitudes")
    expected = 0
    seen = set()
    for frame in frames:
        index = frame.get("frame_index")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid frame index {index!r}")
        if index in seen:
            raise ValueError(f"duplicate frame index {index}")
        if index != expected:
            raise ValueError(
                f"frame index {index} != expected {expected} (gap or reordering)"
            )
        seen.add(index)
        amps = frame.get("harmonic_amplitudes")
        if not isinstance(amps, list) or len(amps) != harmonic_limit:
            raise ValueError(
                f"frame {index}: harmonic_amplitudes length "
                f"{len(amps) if isinstance(amps, list) else 'n/a'} "
                f"!= declared width {harmonic_limit}"
            )
        if not all(isinstance(v, (int, float)) for v in amps):
            raise ValueError(f"frame {expected}: non-numeric harmonic amplitude")
        f0 = frame.get("f0")
        gain = frame.get("energy_gain")
        if not isinstance(f0, (int, float)) or not math.isfinite(f0) or f0 <= 0:
            raise ValueError(f"frame {index}: invalid f0 {f0!r}")
        if not isinstance(gain, (int, float)) or not math.isfinite(gain):
            raise ValueError(f"frame {index}: invalid energy_gain {gain!r}")
        expected += 1
    normalized = [{
        "frame_index": f["frame_index"],
        "f0": float(f["f0"]),
        "harmonic_amplitudes": [float(v) for v in f["harmonic_amplitudes"]],
        "energy_gain": float(f["energy_gain"]),
    } for f in frames]
    raw_frame_fps = [canonical_hash(frame) for frame in frames]
    bundle_fp = canonical_hash(raw_frame_fps)
    return normalized, bundle_fp, raw_frame_fps


def load_inputs(input_dir: str | Path, harmonic_limit: int | None = None) -> tuple[list[UtteranceInput], list[LoadFailure]]:
    input_dir = Path(input_dir)
    utterances: list[UtteranceInput] = []
    failures: list[LoadFailure] = []
    for utter_id, wav_path, frames_path in _pair_files(input_dir):
        try:
            raw_frames = [
                json.loads(line)
                for line in frames_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            frames, bundle_fp, raw_frame_fps = _validate_frames(raw_frames)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            failures.append(
                LoadFailure(utter_id, "E_FEATURE_INVALID", f"{frames_path.name}: {exc}")
            )
            continue
        if wav_path is None:
            failures.append(LoadFailure(
                utter_id, "E_AUDIO_FORMAT", f"{utter_id}.wav: reference wav missing"
            ))
            continue
        try:
            reference, _sr = read_wav(wav_path)
        except AudioFormatError as exc:
            failures.append(LoadFailure(utter_id, exc.code, str(exc)))
            continue
        frame_fps = raw_frame_fps
        utterances.append(UtteranceInput(
            utterance_id=utter_id,
            wav_path=wav_path,
            frames_path=frames_path,
            frames=frames,
            frame_fingerprints=frame_fps,
            reference=reference,
            feature_fingerprint=bundle_fp,
        ))
    utterances.sort(key=lambda item: item.utterance_id)
    failures.sort(key=lambda item: item.utterance_id)
    return utterances, failures
