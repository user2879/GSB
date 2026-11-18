from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from remix_workbench.errors import TrackFormatError
from remix_workbench.formats import parse_package
from remix_workbench.injections import Injection, apply_parse_faults


def _package(samples_root: Path, clip: str, version: str) -> Path:
    return samples_root / clip / "packages" / version


def test_three_formats_parse_to_unified_plane(samples_root: Path) -> None:
    planes = {
        version: parse_package("clip_000", _package(samples_root, "clip_000", version), version)
        for version in ("v1", "v2", "v3")
    }
    for plane in planes.values():
        assert plane.clip_id == "clip_000"
        assert plane.sample_rate == 16000
        assert plane.frames == 12800
        assert plane.track_names == ["bass", "drums", "vocal"]
        assert set(plane.track_frames.values()) == {12800}
    assert planes["v1"].bit_depth == 16
    assert planes["v2"].bit_depth == 24
    assert planes["v3"].bit_depth == 16
    assert planes["v1"].channel_map == {"bass": 0, "drums": 1, "vocal": 2}
    assert planes["v2"].channel_map == {"bass": 0, "drums": 2, "vocal": 1}
    assert planes["v3"].channel_map == {"bass": 0, "vocal": 1, "drums": 2}
    assert planes["v1"].content_fingerprint() != planes["v2"].content_fingerprint()
    assert planes["v2"].source_frame_annotations["drums"]["original_frames"] == 12800
    assert planes["v2"].track_gain_metadata["drums"] == -1.0
    # 24-bit v2 retains lower Q23 bits that 16-bit v1/v3 cannot express.
    assert np.any((planes["v2"].tracks["drums"] & 0xFF) != 0)
    assert np.all((planes["v1"].tracks["drums"] & 0xFF) == 0)


def test_v1_truncated_container_maps_to_track_format(
    samples_root: Path, tmp_path: Path
) -> None:
    source = _package(samples_root, "clip_000", "v1")
    destination = tmp_path / "v1"
    shutil.copytree(source, destination)
    wav = destination / "mix.wav"
    wav.write_bytes(wav.read_bytes()[:-9])
    with pytest.raises(TrackFormatError) as caught:
        parse_package("clip_000", destination, "v1")
    assert caught.value.code == "E_TRACK_FORMAT"


def test_v2_frame_count_mismatch_is_isolated(
    samples_root: Path, tmp_path: Path
) -> None:
    source = _package(samples_root, "clip_001", "v2")
    destination = tmp_path / "v2"
    shutil.copytree(source, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tracks"]["drums"]["frames"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plane = parse_package(
        "clip_001",
        destination,
        "v2",
        expected_tracks=("bass", "drums", "vocal"),
    )
    assert plane.track_names == ["bass", "drums", "vocal"]
    assert any(issue.track == "drums" and issue.code == "E_TRACK_FORMAT" for issue in plane.issues)
    assert np.count_nonzero(plane.tracks["drums"]) == 0
    assert np.count_nonzero(plane.tracks["bass"]) > 0


def test_v3_offset_table_out_of_bounds(samples_root: Path, tmp_path: Path) -> None:
    source = _package(samples_root, "clip_002", "v3")
    injection = Injection(
        id="TEST_V3_OFFSET",
        stage="parse",
        kind="offset_table_out_of_bounds",
        clip_id="clip_002",
        track="vocal",
        format_version="v3",
        expected_code="E_TRACK_FORMAT",
    )
    prepared, applied = apply_parse_faults(
        source, "v3", [injection], tmp_path / "injected"
    )
    assert [item.id for item in applied] == ["TEST_V3_OFFSET"]
    with pytest.raises(TrackFormatError) as caught:
        parse_package(
            "clip_002",
            prepared,
            "v3",
            injection_id="TEST_V3_OFFSET",
            injection_track="vocal",
        )
    assert caught.value.code == "E_TRACK_FORMAT"
    assert caught.value.injection_id == "TEST_V3_OFFSET"


def test_v2_missing_track_is_deterministic_silence(
    samples_root: Path, tmp_path: Path
) -> None:
    source = _package(samples_root, "clip_000", "v2")
    destination = tmp_path / "v2_missing"
    shutil.copytree(source, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tracks"].pop("vocal")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plane = parse_package(
        "clip_000", destination, "v2", expected_tracks=("bass", "drums", "vocal")
    )
    assert "vocal" in plane.missing_tracks
    assert np.count_nonzero(plane.tracks["vocal"]) == 0
    assert any(issue.track == "vocal" and "absent" in issue.message for issue in plane.issues)

