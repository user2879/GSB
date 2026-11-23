from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
RUNTIME = ROOT / "runtime" / "tests"
RUNTIME.mkdir(parents=True, exist_ok=True)


def _remove_readonly_on_windows(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except OSError:
        pass


@contextmanager
def local_temp(prefix: str) -> Iterator[Path]:
    # TemporaryDirectory can fail to remove SQLite snapshot backups on Windows
    # because those files are deliberately read-only.  Clear the bit first.
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(RUNTIME)))
    try:
        yield path
    finally:
        shutil.rmtree(path, onerror=_remove_readonly_on_windows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
