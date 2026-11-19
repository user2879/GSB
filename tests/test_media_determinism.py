"""要求 1：确定性素材生成，两次逐字节一致，格式合规。"""
from __future__ import annotations

import os
import wave

from syncbench.config import FRAME_H, FRAME_W, default_config
from syncbench.engine import Engine
from syncbench.hashing import sha256_hex


def _generate(ws: str) -> dict:
    eng = Engine(ws, default_config())
    return eng.prepare_media()


def test_media_byte_identical(tmp_path):
    ws_a = str(tmp_path / "a")
    ws_b = str(tmp_path / "b")
    man_a = _generate(ws_a)
    man_b = _generate(ws_b)
    assert man_a == man_b
    for rel in ("manifest.json", "media/ch0_audio.wav", "media/ch1_video.raw",
                "media/ch3_audio.wav"):
        ha = sha256_hex(open(os.path.join(ws_a, rel), "rb").read())
        hb = sha256_hex(open(os.path.join(ws_b, rel), "rb").read())
        assert ha == hb


def test_wav_format_24bit_mono_48k(tmp_path):
    man = _generate(str(tmp_path / "wav"))
    entry = man["channels"][0]
    with wave.open(os.path.join(str(tmp_path / "wav"),
                                entry["audio"]["path"]), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getsampwidth() == 3
        assert wf.getnchannels() == 1


def test_video_frame_dims(tmp_path):
    man = _generate(str(tmp_path / "vid"))
    entry = man["channels"][2]
    assert entry["video"]["frames"] == 240
    raw_size = os.path.getsize(os.path.join(str(tmp_path / "vid"),
                                            entry["video"]["path"]))
    assert raw_size == 240 * FRAME_H * FRAME_W


def test_four_channels_with_truth(engine):
    assert len(engine.manifest["channels"]) == 4
    for ch, entry in enumerate(engine.manifest["channels"]):
        truth = entry["truth"]
        assert "ppm" in truth and "delay_samples" in truth
        assert len(entry["pulses"]) == 10
        assert len(entry["local_sync_frames"]) == 10
