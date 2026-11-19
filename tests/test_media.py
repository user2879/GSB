from __future__ import annotations

import wave
from pathlib import Path

import av

from hls_workbench.canonical import sha256_file
from hls_workbench.config import EncoderConfig, MediaConfig
from hls_workbench.media import SyntheticMediaGenerator


def test_manifest_and_source_are_deterministic(tmp_path: Path, media_config: MediaConfig) -> None:
    first = SyntheticMediaGenerator(media_config, EncoderConfig(), tmp_path / "first").generate()
    second = SyntheticMediaGenerator(media_config, EncoderConfig(), tmp_path / "second").generate()

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest["slices"][0]["content_sha256"] == second.manifest["slices"][0]["content_sha256"]
    assert first.manifest["source"]["content_sha256"] == second.manifest["source"]["content_sha256"]
    assert sha256_file(first.source_video) == sha256_file(second.source_video)
    assert sha256_file(first.source_audio) == sha256_file(second.source_audio)


def test_fixed_seed_source_shape_and_pcm(generated_media) -> None:
    manifest = generated_media.manifest
    assert manifest["config"]["seed"] == 20240517
    assert manifest["source"]["frame_count"] == 60
    assert manifest["source"]["audio_samples"] == 96000
    assert len(manifest["slices"]) == 2
    assert manifest["slices"][0]["frame_count"] == 30
    assert manifest["slices"][1]["start_pts"] == 30
    assert manifest["slices"][1]["start_ms"] == 1000

    container = av.open(str(generated_media.source_video))
    decoded = list(container.decode(video=0))
    container.close()
    assert len(decoded) == 60
    assert [frame.pts for frame in decoded[:3]] == [0, 3000, 6000]

    with wave.open(str(generated_media.source_audio), "rb") as handle:
        assert handle.getframerate() == 48000
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 96000
