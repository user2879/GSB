from __future__ import annotations

from pathlib import Path

import pytest

from audiofp.demo import _run_pipeline, _tree_hashes, run_demo


@pytest.mark.slow
def test_full_demo_is_byte_reproducible(tmp_path: Path) -> None:
    report = run_demo(tmp_path / "demo", seed=20261119, verify_repeat=True)
    assert report["ok"] is True
    assert report["repeatability"]["ok"] is True
    assert (tmp_path / "demo" / "reports" / "demo_report.json").is_file()
    assert (tmp_path / "demo" / "reports" / "repeatability.json").is_file()


@pytest.mark.slow
def test_pipeline_tree_is_byte_identical_without_nested_repeat_check(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _run_pipeline(first, 20261119)
    _run_pipeline(second, 20261119)
    assert _tree_hashes(first) == _tree_hashes(second)
