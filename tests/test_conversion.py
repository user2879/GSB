from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from remix_workbench.conversion import convert_plane
from remix_workbench.formats import parse_package
from remix_workbench.pcm import q23_to_i16


def _package(samples_root: Path, version: str) -> Path:
    return samples_root / "clip_000" / "packages" / version


def test_v1_to_v3_roundtrip_preserves_16bit_tracks(
    samples_root: Path, tmp_path: Path
) -> None:
    source = parse_package("clip_000", _package(samples_root, "v1"), "v1")
    report = convert_plane(source, "v3", tmp_path / "v3", source_package=_package(samples_root, "v1"))
    target = parse_package("clip_000", tmp_path / "v3", "v3")
    assert report["target_format_version"] == "v3"
    assert target.frames == source.frames
    assert target.track_names == source.track_names
    for track in source.track_names:
        expected = q23_to_i16(source.tracks[track]).astype(np.int32) << 8
        np.testing.assert_array_equal(target.tracks[track], expected)
    assert (tmp_path / "v3" / "conversion_report.json").is_file()
    assert (tmp_path / "v3" / "loss_metadata.json").is_file()


def test_v3_to_v2_manifest_and_24bit_tracks(
    samples_root: Path, tmp_path: Path
) -> None:
    source = parse_package("clip_000", _package(samples_root, "v3"), "v3")
    report = convert_plane(source, "v2", tmp_path / "v2", source_package=_package(samples_root, "v3"))
    target = parse_package("clip_000", tmp_path / "v2", "v2")
    manifest = json.loads((tmp_path / "v2" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bit_depth"] == 24
    for track, entry in manifest["tracks"].items():
        assert entry["frames"] == source.frames
        assert entry["file"] == f"tracks/{track}.wav"
    assert target.channel_map == {"bass": 0, "drums": 1, "vocal": 2}
    for track in source.track_names:
        np.testing.assert_array_equal(target.tracks[track], source.tracks[track])
    assert report["loss_count"] == 0
    assert json.loads(
        (tmp_path / "v2" / "loss_metadata.json").read_text(encoding="utf-8")
    )["unexpressed_fields"] == []


def test_v2_to_v1_reorders_channels_and_reports_all_losses(
    samples_root: Path, tmp_path: Path
) -> None:
    source = parse_package("clip_000", _package(samples_root, "v2"), "v2")
    report = convert_plane(source, "v1", tmp_path / "v1", source_package=_package(samples_root, "v2"))
    target = parse_package("clip_000", tmp_path / "v1", "v1")
    assert target.channel_map == {"bass": 0, "drums": 1, "vocal": 2}
    fields = {loss["field"] for loss in report["losses"]}
    assert "track_gain_metadata.drums" in fields
    assert "source_frame_annotations.drums" in fields
    assert "tracks.drums.quantization_residual" in fields
    for track in source.track_names:
        expected = q23_to_i16(source.tracks[track]).astype(np.int32) << 8
        np.testing.assert_array_equal(target.tracks[track], expected)
    sidecar = json.loads(
        (tmp_path / "v1" / "loss_metadata.json").read_text(encoding="utf-8")
    )
    assert sidecar["source_channel_map"] == {"bass": 0, "drums": 2, "vocal": 1}
    assert len(sidecar["unexpressed_fields"]) == report["loss_count"]
    assert len(report["losses"]) == len(fields)

