from __future__ import annotations

import json
from pathlib import Path

import pytest

from fcdiag.loader import load_commands, load_config, load_pipeline, load_samples
from fcdiag.receipts import parse_receipts
from fcdiag.state import WorkState

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session")
def context():
    samples = load_samples((SAMPLES / "samples.jsonl").read_text(encoding="utf-8"))
    pipeline = load_pipeline((SAMPLES / "pipeline.yaml").read_text(encoding="utf-8"))
    config = load_config(json.loads((SAMPLES / "config.json").read_text(encoding="utf-8")))
    commands = load_commands((SAMPLES / "commands.jsonl").read_text(encoding="utf-8"))
    receipts = {
        version: parse_receipts(
            version, (SAMPLES / "receipts" / f"v{version}.txt").read_text(encoding="utf-8")
        )
        for version in (1, 2, 3)
    }
    state = WorkState.initial(samples, pipeline, config)
    return {
        "samples": samples,
        "pipeline": pipeline,
        "config": config,
        "commands": commands,
        "receipts": receipts,
        "state": state,
    }
