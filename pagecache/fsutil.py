from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat


def remove_tree(path: str | Path) -> None:
    """Remove a tree even if Windows read-only bits are present."""

    target = Path(path)
    if not target.exists():
        return
    if not target.is_dir():
        target.unlink()
        return

    def clear_readonly_and_retry(function: object, failing_path: str, _error: object) -> None:
        os.chmod(failing_path, stat.S_IREAD | stat.S_IWRITE)
        function(failing_path)  # type: ignore[operator]

    shutil.rmtree(target, onerror=clear_readonly_and_retry)
