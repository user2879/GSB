from __future__ import annotations

import json
from pathlib import Path

from acoustic_workbench.demo import run_demo
from acoustic_workbench.gl_preview import check_offscreen_fbo


def test_repeatable_demo_and_full_rebuild_equivalence(tmp_path: Path) -> None:
    lines: list[str] = []
    result = run_demo(tmp_path / "demo", print_fn=lines.append)
    assert result["accepted"] is True
    assert result["identical_sequential_manifests"] is True
    assert result["incremental_manifest_equals_full_rebuild"] is True
    assert result["node_hash_mismatches"] == []
    assert result["tile_hash_mismatches"] == []
    assert set(result["numeric_max_abs_differences"].values()) == {0.0}
    assert (tmp_path / "demo" / "manifest_run_a.json").read_bytes() == (tmp_path / "demo" / "manifest_run_b.json").read_bytes()
    assert any("closure=" in line for line in lines)
    assert any("acceptance: PASS" in line for line in lines)


def test_offscreen_gl_probe_returns_structured_result() -> None:
    result = check_offscreen_fbo()
    assert isinstance(result["available"], bool)
    assert "backend" in result
    if not result["available"]:
        assert "reason" in result
