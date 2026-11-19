from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest

from spatial_diag.projection import project_tier
from spatial_diag.synth import generate_tier


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_is_byte_deterministic_and_emits_truth(config, tmp_path: Path):
    tier = config.tier("small")
    first = generate_tier(config, tier, tmp_path / "first")
    second = generate_tier(config, tier, tmp_path / "second")

    assert first["step_hash"] == second["step_hash"]
    assert first["wav_hash_set"] == second["wav_hash_set"]
    assert (tmp_path / "first" / "manifest.json").read_bytes() == (
        tmp_path / "second" / "manifest.json"
    ).read_bytes()

    manifest = first["manifest"]
    assert manifest["truth_policy"].startswith("path directions")
    assert len(manifest["sources"]) == 8
    assert manifest["path_count"] == sum(len(source["paths"]) for source in manifest["sources"])
    for source in manifest["sources"]:
        wav_path = tmp_path / "first" / source["wav"]["path"]
        assert _sha(wav_path) == source["wav"]["sha256"]
        with wave.open(str(wav_path), "rb") as handle:
            assert handle.getnchannels() == 4
            assert handle.getframerate() == 48000
            assert handle.getsampwidth() == 2
            assert handle.getnframes() == 4096
        assert len(source["first_arrival_seconds"]) == 4
        assert any(value is not None for value in source["first_arrival_seconds"])
        for path in source["paths"]:
            direction = np.asarray(path["direction"], dtype=np.float64)
            assert np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-10)
            assert path["arrival_seconds"] == pytest.approx(
                path["arrival_samples"] / 48000.0, abs=1.0e-12
            )
            assert path["rendered_energy"] >= 0.0


def test_projection_conserves_events_and_excludes_low_confidence(config, tmp_path: Path):
    tier = config.tier("small")
    generated = generate_tier(config, tier, tmp_path / "small")
    projection = project_tier(config, tier, generated["manifest"], tmp_path / "small")
    counts = np.asarray(projection["effective_sample_count"], dtype=np.int64)
    low = np.asarray(projection["low_confidence"], dtype=np.int8).astype(bool)
    weights = np.asarray(projection["sample_weights"], dtype=np.float64)

    assert projection["event_total"] > 0
    assert int(np.sum(counts)) == projection["event_total"]
    assert projection["binned_event_total"] == projection["event_total"]
    assert projection["conservation"]["events_equal_bin_counts"] is True
    assert projection["fit_eligible_direction_count"] == int(np.sum(~low))
    assert projection["low_confidence_direction_count"] == int(np.sum(low))
    assert np.all(counts[low] < projection["min_effective_samples"])
    assert np.all(counts[~low] >= projection["min_effective_samples"])
    assert np.allclose(weights, 4.0 * np.pi / tier["sphere_points"])
