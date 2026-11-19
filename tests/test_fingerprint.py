from __future__ import annotations

import hashlib

import numpy as np

from audiofp.config import DEFAULT_CONFIG, LEGACY_CONFIG
from audiofp.fingerprint import extract_fingerprints, match_fingerprint_sequences


def _tone(seconds: int = 4) -> np.ndarray:
    rate = DEFAULT_CONFIG.sample_rate
    time = np.arange(rate * seconds) / rate
    rng = np.random.default_rng(77)
    return (
        0.42 * np.sin(2 * np.pi * 220.0 * time)
        + 0.21 * np.sin(2 * np.pi * 554.37 * time + 0.2)
        + 0.08 * np.sin(2 * np.pi * 1318.5 * time + 0.7)
        + 0.01 * rng.standard_normal(len(time))
    )


def test_fingerprints_are_deterministic_ordered_and_recomputable() -> None:
    signal = _tone()
    first = extract_fingerprints(signal, DEFAULT_CONFIG)
    second = extract_fingerprints(signal.copy(), DEFAULT_CONFIG)
    assert first.canonical_rows() == second.canonical_rows()
    assert first.content_hash() == second.content_hash()
    assert first.fingerprints == sorted(
        first.fingerprints,
        key=lambda item: (
            item.anchor_frame,
            item.target_frame,
            item.band1,
            item.band2,
            item.hash,
        ),
    )
    assert first.fingerprints
    for item in first.fingerprints[:40]:
        payload = DEFAULT_CONFIG.hash_payload(item.band1, item.band2, item.delta_frames)
        expected = hashlib.blake2b(payload.encode("ascii"), digest_size=8).hexdigest()
        assert item.hash == expected
    assert match_fingerprint_sequences(first, second).hit_rate == 1.0


def test_config_change_has_explicit_encoding_space_difference() -> None:
    signal = _tone(2)
    v2 = extract_fingerprints(signal, DEFAULT_CONFIG)
    v1 = extract_fingerprints(signal, LEGACY_CONFIG)
    assert v1.peaks == v2.peaks
    assert v1.content_hash() != v2.content_hash()
    assert DEFAULT_CONFIG.migration_compatible_from(LEGACY_CONFIG)
    assert any(a.hash != b.hash for a, b in zip(v1.fingerprints, v2.fingerprints))
