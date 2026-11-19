from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_demo_stdout_is_byte_reproducible(root: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    command = [sys.executable, "-m", "dns_local.demo"]
    first = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=30)
    second = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stderr.decode("utf-8", "replace")
    assert second.returncode == 0, second.stderr.decode("utf-8", "replace")
    assert first.stdout == second.stdout
    lines = first.stdout.decode("utf-8").splitlines()
    assert any(line.startswith("SCENARIO 1") for line in lines)
    assert any(line.startswith("SCENARIO 2") for line in lines)
    assert any(line.startswith("SCENARIO 3") for line in lines)
    assert any(line.startswith("SNAPSHOT_RESTORE") and "mode=whole" in line for line in lines)
    assert lines[-1].startswith("FINAL scenarios=3")


def test_gui_module_imports_without_requiring_tk() -> None:
    import dns_local.gui as gui

    assert callable(gui.main)
    assert callable(gui.run_gui)

