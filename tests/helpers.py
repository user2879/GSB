from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from mesh_audit.engine import RunEngine, RunResult
from mesh_audit.sample import deterministic_mesh_payload, deterministic_repair_script
from mesh_audit.schema import (
    MeshInput,
    RepairScript,
    validate_mesh_payload,
    validate_repair_script,
)


def make_mesh(payload: dict[str, Any] | None = None) -> MeshInput:
    return validate_mesh_payload(
        deepcopy(payload if payload is not None else deterministic_mesh_payload()),
        "<test-mesh>",
    )


def make_script(overrides: dict[str, Any] | None = None) -> RepairScript:
    payload = deepcopy(deterministic_repair_script())
    if overrides:
        payload.update(deepcopy(overrides))
    return validate_repair_script(payload, "<test-script>")


def run_audit(
    root: Path,
    mesh: MeshInput | None = None,
    script: RepairScript | None = None,
    state_name: str = "state",
    output_name: str = "output",
    restart: bool = False,
) -> tuple[RunResult, RunEngine]:
    actual_mesh = mesh or make_mesh()
    actual_script = script or make_script()
    engine = RunEngine(
        actual_mesh,
        actual_script,
        root / state_name,
        root / output_name,
    )
    result = engine.run(restart=restart)
    return result, engine


def write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
