from __future__ import annotations

import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from ner_review.engine import IncrementalEngine
from ner_review.snapshot import SnapshotStore
from ner_review.synth import write_demo_bundle
from ner_review.validation import load_package


def test_sigkill_before_snapshot_replays_and_matches_full(tmp_path: Path):
    data = tmp_path / "data"
    workspace = tmp_path / "state"
    write_demo_bundle(data)
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    killed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ner_review.cli",
            "kill-target",
            "--data",
            str(data),
            "--workspace",
            str(workspace),
            "--mode",
            "before-save",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert killed.returncode == -9, killed.stderr

    recovered = SnapshotStore(workspace).load()
    assert recovered["ok"]
    assert recovered["engine"].command_cursor == 0

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ner_review.cli",
            "run",
            "--data",
            str(data),
            "--workspace",
            str(workspace),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    final_loaded = SnapshotStore(workspace).load()
    assert final_loaded["ok"]
    incremental = final_loaded["engine"]
    package_result = load_package(data / "model_params.json")
    full = IncrementalEngine(
        package=copy.deepcopy(incremental.package),
        config=replace(incremental.config),
        samples=copy.deepcopy(list(incremental.samples.values())),
        lexicon=copy.deepcopy(incremental.lexicon_list),
        corpus_failures=copy.deepcopy(incremental.corpus_failures),
        lexicon_failures=copy.deepcopy(incremental.lexicon_failures),
    )
    full.run_incremental()
    assert incremental.authoritative_result() == full.authoritative_result()
    assert package_result["ok"]
