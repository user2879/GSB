from __future__ import annotations

from pathlib import Path

from fixedvideo.verify import verify


def test_end_to_end_acceptance(sample_dir: Path) -> None:
    report = verify(
        config_path=sample_dir / "render_config.json",
        commands_path=sample_dir / "commands.json",
        workdir=sample_dir.parent / "verification",
        force=True,
    )
    assert report["status"] == "ok"
    assert all(report["checks"].values())
    assert report["incremental_vs_full"]["pixels_equal"] is True
    assert report["incremental_vs_full"]["required_artifact_content_equal"] is True
    assert report["recovery"]["resumed_from_cursor"] == 6
    assert report["stale"]["all_rejected"] is True
