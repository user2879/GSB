from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from haptic_diag.compiler import compile_graph
from haptic_diag.engine import apply_update, initial_state
from haptic_diag.generator import load_config, generate_sample
from haptic_diag.persistence import AtomicWorkspaceStore, InjectedCommitCrash

ROOT = Path(__file__).resolve().parents[1]


def _sample_graph(index: int = 2):
    config = load_config(ROOT / "configs" / "samples.json")
    return generate_sample(config["samples"][index])


def test_atomic_commit_and_recovery_restore_hashes_digests_and_ledger(tmp_path: Path) -> None:
    state = initial_state(_sample_graph(), 2)
    primitive = state.graph.primitives[0]
    state, _ = apply_update(
        state,
        {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.05)}},
    )
    store = AtomicWorkspaceStore(tmp_path / "workspace")
    commit = store.commit(state)
    recovery = store.recover()
    assert recovery.recovered
    assert recovery.current_generation == commit.generation
    assert recovery.state is not None
    assert [frame.content_hash for frame in recovery.state.frames] == [
        frame.content_hash for frame in state.frames
    ]
    assert [digest.digest for digest in recovery.state.digests] == [
        digest.digest for digest in state.digests
    ]
    assert recovery.state.ledger == state.ledger
    assert recovery.state.protocol_version == 2


def test_incomplete_snapshot_is_orphaned_and_previous_commit_is_restored(tmp_path: Path) -> None:
    baseline = initial_state(_sample_graph(), 2)
    store = AtomicWorkspaceStore(tmp_path / "workspace")
    first = store.commit(baseline)
    primitive = baseline.graph.primitives[0]
    pending, _ = apply_update(
        baseline,
        {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.07)}},
    )
    with pytest.raises(InjectedCommitCrash):
        store.commit(pending, crash_after_files=3)
    recovery = store.recover()
    assert recovery.recovered
    assert recovery.current_generation == first.generation
    assert recovery.orphans
    assert all(item.classification == "orphan" for item in recovery.orphans)
    assert recovery.recovery_log_path is not None
    assert Path(recovery.recovery_log_path).exists()
    assert recovery.state is not None
    assert [frame.content_hash for frame in recovery.state.frames] == [
        frame.content_hash for frame in baseline.frames
    ]


def test_complete_snapshot_newer_than_current_is_orphaned(tmp_path: Path) -> None:
    baseline = initial_state(_sample_graph(), 2)
    store = AtomicWorkspaceStore(tmp_path / "workspace")
    first = store.commit(baseline)
    primitive = baseline.graph.primitives[0]
    newer, _ = apply_update(
        baseline,
        {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.04)}},
    )
    second = store.commit(newer)
    # Simulate a crash between snapshot-directory rename and CURRENT replace by
    # restoring the old pointer after the newer complete directory exists.
    store.current_path.write_text(first.generation + "\n", encoding="ascii")
    recovery = store.recover()
    assert recovery.recovered
    assert recovery.current_generation == first.generation
    assert recovery.orphans
    assert any(item.generation == second.generation for item in recovery.orphans)
    assert recovery.state is not None
    assert [frame.content_hash for frame in recovery.state.frames] == [
        frame.content_hash for frame in baseline.frames
    ]


def test_resume_after_crash_matches_full_recompile(tmp_path: Path) -> None:
    baseline = initial_state(_sample_graph(), 2)
    store = AtomicWorkspaceStore(tmp_path / "workspace")
    store.commit(baseline)
    primitive = baseline.graph.primitives[0]
    change = {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.07)}}
    pending, _ = apply_update(baseline, change)
    with pytest.raises(InjectedCommitCrash):
        store.commit(pending, crash_after_files=2)
    recovery = store.recover()
    assert recovery.state is not None
    resumed, update = apply_update(recovery.state, change)
    full = compile_graph(resumed.graph, resumed.protocol_version)
    assert update.recompiled_frames
    assert [frame.content_hash for frame in resumed.frames] == [
        frame.content_hash for frame in full.frames
    ]
    assert [digest.digest for digest in resumed.digests] == [
        digest.digest for digest in full.digests
    ]


def test_protocol_switch_archived_ledger_is_not_reused_after_recovery(tmp_path: Path) -> None:
    baseline = initial_state(_sample_graph(), 2)
    primitive = baseline.graph.primitives[0]
    changed, _ = apply_update(
        baseline,
        {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.04)}},
    )
    v1_state, _ = apply_update(changed, {}, new_protocol_version=1)
    store = AtomicWorkspaceStore(tmp_path / "workspace")
    store.commit(v1_state)
    recovery = store.recover()
    assert recovery.state is not None
    assert recovery.state.ledger.protocol_version == 1
    assert recovery.state.ledger_history
    assert all(not ledger.reusable for ledger in recovery.state.ledger_history)
    assert all(entry.protocol_version == 1 for entry in recovery.state.ledger.entries)


def test_real_subprocess_forced_termination_then_resume_matches_full(tmp_path: Path) -> None:
    workspace = tmp_path / "crash-workspace"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "haptic_diag.cli",
            "crash-worker",
            "--workspace",
            str(workspace),
            "--crash-after-batches",
            "2",
            "--commit-crash-after-files",
            "3",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode != 0, completed.stderr.decode("utf-8", errors="replace")
    store = AtomicWorkspaceStore(workspace)
    recovery = store.recover()
    assert recovery.recovered
    assert recovery.state is not None
    assert recovery.orphans
    assert all(item.classification == "orphan" for item in recovery.orphans)
    baseline = initial_state(_sample_graph(), 2)
    assert [frame.content_hash for frame in recovery.state.frames] == [
        frame.content_hash for frame in baseline.frames
    ]
    primitive = recovery.state.graph.primitives[0]
    change = {primitive.id: {"amplitude": min(1.0, primitive.amplitude + 0.07)}}
    resumed, _ = apply_update(recovery.state, change)
    full = compile_graph(resumed.graph, 2)
    assert [frame.content_hash for frame in resumed.frames] == [
        frame.content_hash for frame in full.frames
    ]
