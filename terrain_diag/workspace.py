from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .canonical import write_json
from .errors import ErrorCode, fail


LAYOUT_VERSION = 1
MINIMUM_FREE_BYTES = 2 * 1024 * 1024 * 1024
SUBDIRECTORIES = (
    "terrain",
    "manifests",
    "hiz",
    "reports",
    "ledgers",
    "meta",
)


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def terrain_dir(self) -> Path:
        return self.root / "terrain"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def hiz_dir(self) -> Path:
        return self.root / "hiz"

    @property
    def report_dir(self) -> Path:
        return self.root / "reports"

    @property
    def ledger_dir(self) -> Path:
        return self.root / "ledgers"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str],
        minimum_free_bytes: int = MINIMUM_FREE_BYTES,
        enforce_disk: bool = True,
    ) -> "Workspace":
        path = Path(root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            fail(ErrorCode.E_BUDGET_DENIED, "workspace root is not a directory", root=str(path))

        free_bytes = shutil.disk_usage(path).free
        if enforce_disk and free_bytes < minimum_free_bytes:
            fail(
                ErrorCode.E_BUDGET_DENIED,
                "insufficient persistent workspace space",
                free_bytes=free_bytes,
                required_bytes=minimum_free_bytes,
            )

        for name in SUBDIRECTORIES:
            (path / name).mkdir(parents=True, exist_ok=True)

        probe = path / "meta" / ".write_probe"
        try:
            probe.write_bytes(b"probe")
            if probe.read_bytes() != b"probe":
                raise OSError("write probe mismatch")
        except OSError as exc:
            fail(ErrorCode.E_BUDGET_DENIED, "workspace is not writable", root=str(path), error=str(exc))
        finally:
            probe.unlink(missing_ok=True)

        workspace = cls(path)
        layout = {
            "layout_version": LAYOUT_VERSION,
            "directories": list(SUBDIRECTORIES),
            "minimum_free_bytes": int(minimum_free_bytes),
            "disk_check_enforced": bool(enforce_disk),
        }
        write_json(workspace.meta_dir / "workspace_layout.json", layout)
        return workspace

    def relative(self, path: str | os.PathLike[str]) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()
