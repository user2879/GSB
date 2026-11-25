import json
import wave

import pytest

from voicecmd.audio import read_wav, write_wav
from voicecmd.config import load_config
from voicecmd.errors import E_AUDIO_FORMAT, E_FRAGMENT_INVALID
from voicecmd.samplegen import generate_dataset
from voicecmd.audio import load_fragments, load_command_stream
import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_manifest_deterministic_and_out_of_order(workspace, cfg):
    manifest = json.loads((workspace / cfg.paths.manifest_file).read_text("utf-8"))
    seqs = [f["seq"] for f in manifest["fragments"]]
    arrivals = [f["arrival_ms"] for f in manifest["fragments"]]
    assert seqs == sorted(seqs)
    assert arrivals != sorted(arrivals)
    assert all(a >= 0 for a in arrivals)
    assert any(f["arrival_ms"] != f["capture_ms"] for f in manifest["fragments"])


def test_wav_format_validation(workspace, cfg):
    wav = workspace / cfg.paths.data_dir / "fragment_0000.wav"
    samples = read_wav(wav)
    assert samples.dtype == np.int16
    raw = wav.read_bytes()
    from voicecmd.audio import read_wav_bytes
    with pytest.raises(Exception) as exc:
        read_wav_bytes(raw[:-100])
    assert exc.value.code == E_AUDIO_FORMAT
    bad = bytearray(raw)
    bad[22] = 8  # 声道数改为 8
    with pytest.raises(Exception) as exc:
        read_wav_bytes(bytes(bad))
    assert exc.value.code == E_AUDIO_FORMAT
    bad2 = bytearray(raw)
    bad2[24:28] = (8000).to_bytes(4, "little")
    with pytest.raises(Exception) as exc:
        read_wav_bytes(bytes(bad2))
    assert exc.value.code == E_AUDIO_FORMAT


def test_manifest_rejects_duplicate_seq_and_bad_word(workspace, cfg):
    manifest = workspace / cfg.paths.manifest_file
    payload = json.loads(manifest.read_text("utf-8"))
    payload["fragments"][1]["seq"] = payload["fragments"][0]["seq"]
    manifest.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    with pytest.raises(Exception) as exc:
        load_fragments(manifest, workspace / cfg.paths.data_dir, set(cfg.vocab.values()))
    assert exc.value.code == E_FRAGMENT_INVALID

    generate_dataset(cfg, workspace)
    payload = json.loads(manifest.read_text("utf-8"))
    payload["fragments"][0]["command_word"] = "不在词表"
    manifest.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    with pytest.raises(Exception) as exc:
        load_fragments(manifest, workspace / cfg.paths.data_dir, set(cfg.vocab.values()))
    assert exc.value.code == E_FRAGMENT_INVALID


def test_negative_arrival_rejected(workspace, cfg):
    manifest = workspace / cfg.paths.manifest_file
    payload = json.loads(manifest.read_text("utf-8"))
    payload["fragments"][0]["arrival_ms"] = -1
    manifest.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    with pytest.raises(Exception) as exc:
        load_fragments(manifest, workspace / cfg.paths.data_dir, set(cfg.vocab.values()))
    assert exc.value.code == E_FRAGMENT_INVALID


def test_command_stream_vocab_validation(workspace, cfg):
    stream_path = workspace / cfg.paths.command_stream_file
    stream = json.loads(stream_path.read_text("utf-8"))
    assert load_command_stream(stream_path, set(cfg.vocab.values()))
    stream["commands"][0]["word"] = "非法词"
    stream_path.write_text(json.dumps(stream, ensure_ascii=False), "utf-8")
    with pytest.raises(Exception) as exc:
        load_command_stream(stream_path, set(cfg.vocab.values()))
    assert exc.value.code == E_FRAGMENT_INVALID


def test_config_rejects_bad_quant_level(tmp_path):
    cfg = load_config(ROOT / "configs" / "pipeline.json")
    payload = cfg.model_dump()
    payload["quant"]["quant_level"] = 13
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(Exception):
        load_config(path)


def test_wave_standard_library_roundtrip(tmp_path):
    samples = np.arange(-100, 100, dtype=np.int16)
    path = tmp_path / "x.wav"
    write_wav(path, samples)
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
    assert np.array_equal(read_wav(path), samples)
