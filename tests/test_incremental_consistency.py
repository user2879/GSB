from __future__ import annotations

import json
from pathlib import Path

from vc_workbench.state import load_current, load_snapshot_directory

from conftest import SAMPLES, run_cli


def _completed_snapshot_for_cursor(state_root: Path, cursor: int):
    matches = []
    for snapshot_dir in sorted((state_root / "snapshots").iterdir()):
        if not snapshot_dir.is_dir():
            continue
        manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("command_complete") is True
            and manifest.get("active_command_seq") is None
            and int(manifest.get("command_cursor", -1)) == cursor
        ):
            matches.append(snapshot_dir)
    assert len(matches) == 1, f"expected one completed snapshot at cursor {cursor}, found {len(matches)}"
    return load_snapshot_directory(matches[0])


def test_each_incremental_command_equals_full_replay_snapshot(baseline: dict[str, object], tmp_path: Path) -> None:
    incremental_state = tmp_path / "incremental_state"
    incremental_output = tmp_path / "incremental.wav"
    incremental_report = tmp_path / "incremental.json"
    for cursor in range(1, 9):
        result = run_cli(
            "run",
            "--samples",
            SAMPLES,
            "--config",
            SAMPLES / "config.json",
            "--commands",
            SAMPLES / "commands.json",
            "--state",
            incremental_state,
            "--output",
            incremental_output,
            "--report",
            incremental_report,
            "--stop-after-command",
            cursor,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        current = load_current(incremental_state)
        full = _completed_snapshot_for_cursor(Path(baseline["state"]), cursor)
        assert current is not None
        assert current.snapshot_id == full.snapshot_id
        assert current.manifest["params"] == full.manifest["params"]
        assert current.analysis == full.analysis
        assert current.stats == full.stats
        assert current.mappings == full.mappings
        assert current.assembly == full.assembly
        assert current.ledger == full.ledger
        assert current.pcm_by_segment == full.pcm_by_segment
        assert current.assembly_pcm == full.assembly_pcm
        report = json.loads(incremental_report.read_text(encoding="utf-8"))
        assert report["reuse"] == current.ledger["decision"]["reuse"]
        assert report["recompute"] == current.ledger["decision"]["recompute"]
        assert report["failures"] == current.ledger["failures"]
        assert report["budget"]["consumed"] == current.manifest["consumed_frame_budget"]


def test_command_stream_tail_extension_and_unreferenced_wav_do_not_invalidate(
    samples_copy: Path, tmp_path: Path
) -> None:
    full_plan = json.loads((samples_copy / "commands.json").read_text(encoding="utf-8"))
    prefix_plan = dict(full_plan)
    prefix_plan["commands"] = full_plan["commands"][:1]
    prefix_path = samples_copy / "prefix.json"
    prefix_path.write_text(json.dumps(prefix_plan, indent=2, sort_keys=True), encoding="utf-8")
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report = tmp_path / "report.json"
    first = run_cli(
        "run",
        "--samples",
        samples_copy,
        "--config",
        samples_copy / "config.json",
        "--commands",
        prefix_path,
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
        "--fresh",
    )
    assert first.returncode == 0, first.stderr + first.stdout
    import shutil

    shutil.copyfile(samples_copy / "sources" / "append.wav", samples_copy / "sources" / "unreferenced.wav")
    extended_plan = dict(full_plan)
    extended_plan["commands"] = full_plan["commands"][:2]
    extended_path = samples_copy / "extended.json"
    extended_path.write_text(json.dumps(extended_plan, indent=2, sort_keys=True), encoding="utf-8")
    second = run_cli(
        "run",
        "--samples",
        samples_copy,
        "--config",
        samples_copy / "config.json",
        "--commands",
        extended_path,
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
    )
    assert second.returncode == 0, second.stderr + second.stdout
    resumed = json.loads(report.read_text(encoding="utf-8"))
    assert resumed["command_cursor"] == 2
    assert resumed["startup_events"] == []
    assert resumed["recovery"]["recovered"] is True


def test_segment_override_changes_only_that_segment_pcm(tmp_path: Path) -> None:
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report_path = tmp_path / "report.json"
    snapshots = []
    for cursor in (1, 2):
        result = run_cli(
            "run",
            "--samples",
            SAMPLES,
            "--config",
            SAMPLES / "config.json",
            "--commands",
            SAMPLES / "commands.json",
            "--state",
            state,
            "--output",
            output,
            "--report",
            report_path,
            "--stop-after-command",
            cursor,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        snapshots.append(json.loads(report_path.read_text(encoding="utf-8")))
    first = {item["segment_id"]: item for item in snapshots[0]["segments"]}
    second = {item["segment_id"]: item for item in snapshots[1]["segments"]}
    assert first["seg_001"]["pcm_sha256"] != second["seg_001"]["pcm_sha256"]
    assert first["seg_002"]["pcm_sha256"] == second["seg_002"]["pcm_sha256"]
    assert first["seg_003"]["pcm_sha256"] == second["seg_003"]["pcm_sha256"]
    assert snapshots[0]["assembly"]["pcm_sha256"] != snapshots[1]["assembly"]["pcm_sha256"]
    assert [item["segment_id"] for item in snapshots[1]["reuse"]] == ["seg_002", "seg_003"]
    assert [item["segment_id"] for item in snapshots[1]["recompute"]] == ["seg_001"]
