from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from voice_diag.audio import parse_wav_bytes, read_wav, write_pcm16_wav
from voice_diag.config import load_config
from voice_diag.errors import AudioFormatError, E_AUDIO_FORMAT, E_SCHEMA, SchemaError
from voice_diag.pipeline import DiagnosticPipeline, PipelineOptions
from voice_diag.schema import load_session
from voice_diag.synthesize import generate_demo


def _demo(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = generate_demo(tmp_path / "sample")
    return paths["config"], paths["manifest"], paths["root"]


def test_strict_wav_validation_and_expected_samples(tmp_path: Path) -> None:
    pcm = np.arange(160, dtype="<i2") * 10
    path = tmp_path / "ok.wav"
    write_pcm16_wav(path, pcm)
    info = read_wav(path, expected_samples=160)
    assert info.sample_count == 160
    assert info.sample_rate == 16000
    assert info.content_hash
    with pytest.raises(AudioFormatError) as caught:
        read_wav(path, expected_samples=159)
    assert caught.value.code == E_AUDIO_FORMAT


def test_wrong_sample_rate_is_audio_format(tmp_path: Path) -> None:
    pcm = np.zeros(160, dtype="<i2")
    path = tmp_path / "bad-rate.wav"
    write_pcm16_wav(path, pcm)
    data = bytearray(path.read_bytes())
    # fmt payload starts at chunk offset 20: audio_format(2), channels(2), rate(4).
    struct.pack_into("<I", data, 24, 8000)
    with pytest.raises(AudioFormatError) as caught:
        parse_wav_bytes(bytes(data), expected_samples=160, where="bad-rate.wav")
    assert caught.value.code == E_AUDIO_FORMAT


def test_bad_audio_is_skipped_with_business_code(tmp_path: Path) -> None:
    config_path, manifest_path, root = _demo(tmp_path)
    config = load_config(config_path)
    data = bytearray((root / "audio/F002.wav").read_bytes())
    struct.pack_into("<I", data, 24, 8000)
    (root / "audio/F002.wav").write_bytes(data)
    session = load_session(config_path, manifest_path, config)
    assert len(session.fragments) == 3
    assert session.skipped_fragment_ids == ("F002",)
    assert [issue.code for issue in session.issues] == [E_AUDIO_FORMAT]


def test_schema_error_creates_no_snapshot(tmp_path: Path) -> None:
    config_path, manifest_path, root = _demo(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["intervals"][0]["id"] = raw["intervals"][1]["id"]
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "out-schema-error"
    with pytest.raises(SchemaError) as caught:
        DiagnosticPipeline(config_path, manifest_path, output).run()
    assert caught.value.code == E_SCHEMA
    assert not output.exists()


def test_unordered_char_offsets_are_schema_error(tmp_path: Path) -> None:
    config_path, manifest_path, root = _demo(tmp_path)
    config = load_config(config_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["intervals"][0]["chars"][1]["start_sample"] = raw["intervals"][0]["chars"][0]["start_sample"] - 1
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        load_session(config_path, manifest_path, config)
    assert caught.value.code == E_SCHEMA


def test_command_stream_missing_char_is_schema_error(tmp_path: Path) -> None:
    config_path, manifest_path, root = _demo(tmp_path)
    config = load_config(config_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["command_stream"][0]["char_indices"] = [999]
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        load_session(config_path, manifest_path, config)
    assert caught.value.code == E_SCHEMA


def test_missing_receipt_interval_is_schema_error(tmp_path: Path) -> None:
    config_path, manifest_path, root = _demo(tmp_path)
    config = load_config(config_path)
    receipt_path = root / "receipts" / "evidence.a1"
    lines = [line for line in receipt_path.read_text(encoding="utf-8").splitlines() if not line.startswith("I004")]
    receipt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_session(config_path, manifest_path, config)


def test_missing_receipt_file_is_schema_error(tmp_path: Path) -> None:
    config_path, manifest_path, root = _demo(tmp_path)
    config = load_config(config_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["receipts"]["a1"] = "receipts/does-not-exist.a1"
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        load_session(config_path, manifest_path, config)
    assert caught.value.code == E_SCHEMA


def test_non_finite_endpoint_parameter_is_schema_error(tmp_path: Path) -> None:
    config_path, _, _ = _demo(tmp_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["endpoint"]["high_energy"] = float("nan")
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        load_config(config_path)
    assert caught.value.code == E_SCHEMA


def test_command_phrase_must_use_controlled_character_table(tmp_path: Path) -> None:
    config_path, manifest_path, _ = _demo(tmp_path)
    config = load_config(config_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["commands"][0]["phrase"] = "打开门窗"
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        load_session(config_path, manifest_path, config)
    assert caught.value.code == E_SCHEMA


def test_audio_path_escape_is_schema_error_not_skippable_audio(tmp_path: Path) -> None:
    config_path, manifest_path, _ = _demo(tmp_path)
    config = load_config(config_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["fragments"][0]["audio_file"] = "../outside.wav"
    manifest_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        load_session(config_path, manifest_path, config)
    assert caught.value.code == E_SCHEMA
