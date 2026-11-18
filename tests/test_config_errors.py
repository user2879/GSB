from __future__ import annotations

import copy
from pathlib import Path

from remix_workbench.engine import RemixEngine


def _run(config: dict, samples_root: Path, workdir: Path):
    result = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        force_workers=2,
    ).execute()
    assert result.report is not None
    return result.report


def test_negative_track_gain_blocks_only_that_clip(
    config: dict, samples_root: Path, tmp_path: Path
) -> None:
    edited = copy.deepcopy(config)
    edited["clips"][0]["tracks"]["drums"]["gain"] = -0.1
    report = _run(edited, samples_root, tmp_path / "negative")
    failures = [item for item in report["failures"] if item["clip_id"] == "clip_000"]
    assert failures
    assert all(item["code"] == "E_MIX_CONFIG" for item in failures)
    clip0 = next(item for item in report["clips"] if item["clip_id"] == "clip_000")
    assert clip0["status"] == "failed"
    assert clip0["output_wav"] is None
    assert not (tmp_path / "negative" / "mixes" / "clip_000.wav").exists()
    assert any(
        item["status"] == "rendered"
        for item in report["clips"]
        if item["clip_id"] != "clip_000"
    )


def test_invalid_quantization_prevents_all_mix_writes(
    config: dict, samples_root: Path, tmp_path: Path
) -> None:
    edited = copy.deepcopy(config)
    edited["gain_quantization"] = 0
    report = _run(edited, samples_root, tmp_path / "quantization")
    assert report["global_mix_blocked"] is True
    assert all(clip["status"] == "failed" for clip in report["clips"])
    assert all(item["code"] == "E_MIX_CONFIG" for item in report["failures"])
    assert not list((tmp_path / "quantization" / "mixes").glob("*.wav"))


def test_out_of_bounds_trim_blocks_clip_without_writing(
    config: dict, samples_root: Path, tmp_path: Path
) -> None:
    edited = copy.deepcopy(config)
    edited["clips"][0]["end_ms"] = 900.0
    report = _run(edited, samples_root, tmp_path / "trim")
    clip0 = next(item for item in report["clips"] if item["clip_id"] == "clip_000")
    assert clip0["status"] == "failed"
    assert any(
        item["clip_id"] == "clip_000"
        and item["stage"] == "mix"
        and item["code"] == "E_MIX_CONFIG"
        for item in report["failures"]
    )
    assert not (tmp_path / "trim" / "mixes" / "clip_000.wav").exists()

