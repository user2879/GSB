from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from log_sampler.pipeline import Pipeline, RuntimeConfig

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_bundle() -> dict[str, Any]:
    return {
        "root": ROOT,
        "config_path": SAMPLES / "config.json",
        "batches_path": SAMPLES / "batches.json",
        "injections_path": SAMPLES / "injections.json",
        "commands_path": SAMPLES / "commands.json",
        "config": _read(SAMPLES / "config.json"),
        "batches": _read(SAMPLES / "batches.json")["batches"],
        "injections": _read(SAMPLES / "injections.json")["injections"],
        "commands": _read(SAMPLES / "commands.json")["commands"],
    }


@pytest.fixture
def run_in_state(tmp_path):
    def run(
        config: dict[str, Any],
        batches: list[dict[str, Any]],
        injections: list[dict[str, Any]] | None = None,
        *,
        state_name: str = "run",
        crash_after: str | None = None,
    ):
        state = tmp_path / state_name
        return Pipeline(
            copy.deepcopy(config),
            copy.deepcopy(batches),
            copy.deepcopy(injections or []),
            RuntimeConfig(state_root=state, crash_after=crash_after),
        ).run()

    return run

