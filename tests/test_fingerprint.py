from __future__ import annotations

from dataclasses import replace

import pytest

from fingerprint_workbench.audio import audio_content_hash, read_wav_bytes
from fingerprint_workbench.config import FingerprintConfig
from fingerprint_workbench.errors import WorkbenchError
from fingerprint_workbench.fingerprint import extract_fingerprint


def test_fingerprint_is_byte_deterministic_and_config_sensitive(generated: dict) -> None:
    data, samples = read_wav_bytes(generated["tracks"]["track_01"])
    config = FingerprintConfig()
    first = extract_fingerprint(samples, config, audio_sha256=audio_content_hash(data))
    second = extract_fingerprint(samples, config, audio_sha256=audio_content_hash(data))
    assert first.content_hash == second.content_hash
    assert first.document == second.document
    variants = [
        replace(config, window_samples=1024),
        replace(config, hop_samples=256),
        replace(config, peak_density=18.0),
        replace(config, band_edges_hz=(90.0, 300.0, 900.0, 2200.0, 5200.0, 10000.0)),
    ]
    hashes = {first.content_hash}
    for variant in variants:
        result = extract_fingerprint(samples, variant, audio_sha256=audio_content_hash(data))
        assert result.content_hash not in hashes
        hashes.add(result.content_hash)
        assert result.document["config_digest"] == variant.digest


def test_invalid_fingerprint_config_uses_business_error() -> None:
    with pytest.raises(WorkbenchError) as caught:
        FingerprintConfig(window_samples=64)
    assert caught.value.code == "E_CONFIG_RANGE"
