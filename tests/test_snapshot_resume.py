from __future__ import annotations

import json
import threading
from pathlib import Path

from iidx.config import load_config
from iidx.engine import Engine
from iidx.indexer import canonical_projection
from iidx.loader import discover_inputs
from iidx.util import first_difference


def _run(data: Path, state: Path, stop_event=None):
    paths = discover_inputs(data)
    config = load_config(paths.config)
    return Engine(
        data_dir=data,
        state_dir=state,
        config=config.config,
        config_failures=config.failures,
        stop_event=stop_event,
    ).run()


def test_graceful_stop_commits_checkpoint_and_resume_matches_full(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "resume-state"
    stop_event = threading.Event()
    stop_event.set()
    interrupted = _run(sample_dir, state, stop_event)
    assert interrupted.exit_code == 0
    assert interrupted.interrupted
    assert interrupted.report["run"]["status"] == "interrupted"
    assert len(interrupted.report["budget_ledger"]) == 1
    assert (state / "checkpoints" / "current.json").exists()

    resumed = _run(sample_dir, state)
    full = _run(sample_dir, tmp_path / "full-state")
    assert resumed.report["diagnostics"]["checkpoint"]["resumed"] is True
    assert first_difference(
        canonical_projection(resumed.report), canonical_projection(full.report)
    ) is None


def test_config_change_marks_snapshot_stale(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = _run(sample_dir, state)
    config_path = sample_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["compression_tier"] = 2
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    second = _run(sample_dir, state)
    assert any(
        event["code"] == "E_SNAPSHOT_STALE"
        for event in second.report["diagnostics"]["snapshot_events"]
    )
    assert first.report["canonical_digest"] != second.report["canonical_digest"]
