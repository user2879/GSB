"""Child process used by acceptance to simulate a hard termination mid-commit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npr_editor.editing import Editor
from npr_editor.render import RenderConfig
from npr_editor.samples import build_sample_project
from npr_editor.snapshots import SnapshotStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args()
    project = build_sample_project()
    project.version_stamp = args.stamp
    editor = Editor(project, config=RenderConfig(width=240, height=160, supersample=1))
    editor.edit_control_point("tree_crown", 0, 0, 1, 151.0, 83.0)
    result = editor.render("lineart")
    store = SnapshotStore(args.workspace)

    def crash(stage: str) -> None:
        if stage == "after_payload":
            os._exit(23)

    store.commit(
        editor.project,
        mode="lineart",
        result=result,
        engine=editor.renderer,
        fault_hook=crash,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
