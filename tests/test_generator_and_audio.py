from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from tools.generate_samples import generate_samples
from vc_workbench.audio import validate_and_decode_wav
from vc_workbench.errors import AudioFormatError

from conftest import SAMPLES


def _digest_tree(root: Path) -> dict[str, str]:
    import hashlib

    result: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".generated"):
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_sample_generator_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_samples(first, force=True)
    generate_samples(second, force=True)
    assert _digest_tree(first) == _digest_tree(second)


def test_wav_validation_accepts_generated_pcm_and_rejects_truncation() -> None:
    audio = validate_and_decode_wav(SAMPLES / "sources" / "session.wav", "sources/session.wav")
    assert audio.sample_rate == 16000
    assert audio.channels == 1
    assert audio.sample_width == 2
    assert audio.samples.dtype == np.int16
    assert audio.frame_count == 22880
    with pytest.raises(AudioFormatError, match="RIFF size mismatch"):
        validate_and_decode_wav(SAMPLES / "errors" / "truncated.wav", "errors/truncated.wav")


def test_wav_validation_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "8k.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(np.zeros(80, dtype="<i2").tobytes())
    with pytest.raises(AudioFormatError, match="sample rate"):
        validate_and_decode_wav(path, "8k.wav")


def test_sample_manifest_matches_files() -> None:
    manifest = json.loads((SAMPLES / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fixed_seed"] == 20261121
    for rel in manifest["references"]:
        assert (SAMPLES / rel).is_file()
    for value in manifest["sources"].values():
        assert (SAMPLES / value["path"]).is_file()
