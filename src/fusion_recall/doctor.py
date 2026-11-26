from __future__ import annotations

import os
import platform
import socket
import tempfile
from pathlib import Path
from typing import Any


def environment_report(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    staging = root / "workspace" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    temp_writable = False
    temp_path: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".doctor-", dir=str(staging))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("ok")
        Path(temp_name).unlink(missing_ok=True)
        temp_writable = True
        temp_path = str(staging)
    except OSError:
        temp_writable = False

    loopback_port: int | None = None
    loopback_ok = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            loopback_port = sock.getsockname()[1]
            loopback_ok = True
    except OSError:
        loopback_ok = False

    try:
        import numpy as np

        numpy_version = np.__version__
        numpy_ok = numpy_version == "1.26.4"
    except ImportError:
        numpy_version = None
        numpy_ok = False

    python_version = platform.python_version()
    return {
        "python_version": python_version,
        "python_3_11_ok": python_version.startswith("3.11"),
        "numpy_version": numpy_version,
        "numpy_1_26_4_ok": numpy_ok,
        "staging_dir": temp_path,
        "staging_writable": temp_writable,
        "loopback_host": "127.0.0.1",
        "loopback_ephemeral_port": loopback_port,
        "loopback_bind_ok": loopback_ok,
        "offline_only": True,
        "ready": bool(
            python_version.startswith("3.11")
            and numpy_ok
            and temp_writable
            and loopback_ok
        ),
    }
