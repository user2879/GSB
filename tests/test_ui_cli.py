from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from audiofp.config import DEFAULT_CONFIG
from audiofp.ui import portable_session


def test_gui_offscreen_smoke(main_index: Path) -> None:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    process = subprocess.run(
        [sys.executable, "-m", "audiofp", "ui", "--index", str(main_index), "--smoke"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout.splitlines()[-1])
    assert payload["ok"] is True
    assert payload["platform"] == "offscreen"


def test_portable_ui_emits_explicit_notice(
    main_index: Path, capsys: object
) -> None:
    code = portable_session(
        str(main_index),
        DEFAULT_CONFIG,
        reason="test headless environment",
        commands=["list", "quit"],
    )
    assert code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "PORTABLE" in captured.err
    assert "portable" in captured.out


def test_cli_doctor_is_machine_readable() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "audiofp", "doctor", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["offline_only"] is True
    assert payload["ffmpeg"]["available"] is True
