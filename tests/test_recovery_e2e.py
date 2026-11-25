import json
import subprocess
import sys
from pathlib import Path

from voicecmd.config import load_config
from voicecmd.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_resume_after_forced_stop(workspace, cfg):
    config = load_config(workspace / "config.json")
    runner = ROOT / "scripts" / "run_batches.py"
    first = subprocess.run(
        [sys.executable, str(runner), "--base-dir", str(workspace),
         "--config", "config.json", "--budget", "7", "--max-batches", "2"],
        cwd=workspace, capture_output=True, text=True, check=True,
    )
    snapshot = workspace / config.paths.snapshot_dir / "state.json"
    assert snapshot.exists()
    state = json.loads(snapshot.read_text("utf-8"))
    assert state["global_version"] == 2
    assert len(state["pending"]) > 0

    resumed = Pipeline(config, str(workspace), snapshot=True)
    assert resumed.resume_if_possible() == "resume"
    while resumed.pending or resumed.invalid_windows:
        resumed.run_batch(budget=7)
    assert len(resumed.features) == len(resumed.windows)
    assert resumed.conclusions

    fresh = Pipeline(config, str(workspace), snapshot=False)
    fresh.load_all()
    fresh.run_batch(budget=1000)
    assert {i: f.fixed_point for i, f in resumed.features.items()} == \
           {i: f.fixed_point for i, f in fresh.features.items()}
    assert [c.cache_key for c in resumed.conclusions] == \
           [c.cache_key for c in fresh.conclusions]
