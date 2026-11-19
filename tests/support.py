from __future__ import annotations

import tempfile
from pathlib import Path

from forensic_workbench.source import SourceSpec, generate_source


SMALL_SPEC = SourceSpec(
    width=160,
    height=90,
    fps=30,
    duration_seconds=3,
    seed=7,
    gop_size=10,
)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name)
        return self.path

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._temporary.cleanup()


def make_workspace_source(root: Path, spec: SourceSpec = SMALL_SPEC):
    return generate_source(root / "source", spec)

