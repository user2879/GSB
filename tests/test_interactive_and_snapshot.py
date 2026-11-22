from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fixedvideo.cache import ArtifactCache
from fixedvideo.config import load_command_stream, load_render_config
from fixedvideo.engine import InteractiveEngine
from fixedvideo.errors import E_INTERACT_BUDGET_PENDING, E_SNAPSHOT_STALE, RenderError
from fixedvideo.renderer import DeterministicRenderer
from fixedvideo.session import SessionRunner
from fixedvideo.snapshot import SnapshotManager
from fixedvideo.state import initial_state


def test_interactive_budget_returns_pending_then_latest_preview(sample_dir: Path) -> None:
    config = replace(load_render_config(sample_dir / "render_config.json"), interaction_latency_budget_ms=1)
    commands, _ = load_command_stream(sample_dir / "commands.json")
    renderer = DeterministicRenderer(config, ArtifactCache())
    engine = InteractiveEngine(config, renderer, state=initial_state(config))
    try:
        first = engine.submit(commands[0])
        assert first["business_code"] == E_INTERACT_BUDGET_PENDING
        assert first["response_latency_ms"] <= 3.0
        assert first["preview"] is None
        engine.wait_idle()
        first_complete = engine.latest_preview
        assert first_complete is not None

        second = engine.submit(commands[1])
        assert second["business_code"] == E_INTERACT_BUDGET_PENDING
        assert second["preview"] is not None
        assert second["preview_corresponds_to_version"] == first_complete["version"]
        assert second["result_corresponds_to"]["config_fingerprint"] is not None
        engine.wait_idle()
    finally:
        engine.shutdown()


def test_snapshot_stale_rejection_and_resume_cursor(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    commands, command_hash = load_command_stream(sample_dir / "commands.json")
    workdir = sample_dir.parent / "snapshot-run"
    first = SessionRunner(
        config=config,
        commands=commands,
        command_stream_hash=command_hash,
        workdir=workdir,
        resume=False,
    )
    run_info = first.run(stop_after=3, commit_final=True)
    assert run_info["final_cursor"] == 3

    manager = SnapshotManager(workdir)
    stale_config = replace(config, source_hash="wrong-config-hash")
    try:
        manager.load_for_resume(config=stale_config, command_stream_hash=command_hash)
        raise AssertionError("stale snapshot was not rejected")
    except RenderError as exc:
        assert exc.code == E_SNAPSHOT_STALE

    resumed = SessionRunner(
        config=config,
        commands=commands,
        command_stream_hash=command_hash,
        workdir=workdir,
        resume=True,
    )
    assert resumed.engine.state.command_cursor == 3
    resumed.run(stop_after=1, commit_final=True)
    assert resumed.engine.state.command_cursor == 4
