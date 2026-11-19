from __future__ import annotations

import hashlib
from pathlib import Path

from glyphforge.demo import run_demo


def _file_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def test_demo_outputs_are_byte_identical_across_independent_runs(tmp_path):
    first = run_demo(tmp_path / "first", threads=3)
    second = run_demo(tmp_path / "second", threads=3)
    assert first["byte_identical"] is True
    assert first["scenario_sha256"] == second["scenario_sha256"]
    assert _file_hashes(tmp_path / "first") == _file_hashes(tmp_path / "second")
    assert all(first["validation"].values())
