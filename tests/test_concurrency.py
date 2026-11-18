from __future__ import annotations

from pathlib import Path

from remix_workbench.engine import RemixEngine, compare_signatures, render_signature
from remix_workbench.pcm import read_wav


def test_thread_pool_and_serial_renders_are_frame_identical(
    config: dict, samples_root: Path, tmp_path: Path
) -> None:
    parallel = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=tmp_path / "parallel",
        force_workers=4,
    ).execute()
    serial = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=tmp_path / "serial",
        force_workers=1,
    ).execute()
    assert parallel.report is not None and serial.report is not None
    comparison = compare_signatures(
        render_signature(parallel.report, parallel.workdir),
        render_signature(serial.report, serial.workdir),
    )
    assert comparison["matches"] is True
    assert parallel.report["cache_event_order"] == serial.report["cache_event_order"]
    assert parallel.report["failure_order"] == serial.report["failure_order"]
    for clip in parallel.report["clips"]:
        pcm_parallel = read_wav(clip["output_wav"])[4]
        serial_clip = next(
            item for item in serial.report["clips"] if item["clip_id"] == clip["clip_id"]
        )
        pcm_serial = read_wav(serial_clip["output_wav"])[4]
        assert pcm_parallel == pcm_serial

