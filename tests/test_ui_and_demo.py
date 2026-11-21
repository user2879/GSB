from __future__ import annotations

from pathlib import Path

from qosfdiag.demo import format_demo_summary, run_demo
from qosfdiag.engine import CascadeEngine
from qosfdiag.error_propagation import propagate_fixed_errors
from qosfdiag.generator import DeterministicUdpGenerator
from qosfdiag.policy import load_policy
from qosfdiag.samples import three_stage_generator, three_stage_policy
from qosfdiag.transport import InMemoryTransport
from qosfdiag.ui_model import build_view


def test_view_model_is_stable_and_complete() -> None:
    policy = load_policy(three_stage_policy())
    generator = DeterministicUdpGenerator(three_stage_generator())
    with CascadeEngine(
        policy,
        generator,
        scale=3,
        transport=InMemoryTransport([s.shaper_id for s in policy.shapers]),
    ) as engine:
        run = engine.run()
    errors = propagate_fixed_errors(
        policy,
        run.summary,
        run.events,
        scale=3,
        generator_digest=generator.sequence_digest,
    )
    snapshots = [
        {
            "filename": "b.json",
            "policy_version": "v2",
            "scale": 3,
            "cursor_seq": 4,
            "content_digest": "b",
        },
        {
            "filename": "a.json",
            "policy_version": "v1",
            "scale": 3,
            "cursor_seq": 2,
            "content_digest": "a",
        },
    ]
    first = build_view(
        policy=policy, run=run, errors=errors, snapshots=snapshots
    )
    second = build_view(
        policy=policy, run=run, errors=errors, snapshots=list(reversed(snapshots))
    )
    assert first == second
    assert [row["filename"] for row in first.snapshot_rows] == ["a.json", "b.json"]
    assert [row["stage_id"] for row in first.stage_rows] == ["edge", "core", "tail"]
    assert first.error_heat_rows[-1]["stage_id"] == "tail"
    assert first.error_heat_rows[0]["bytes_share_value"] > 0
    assert first.error_heat_rows[0]["heat"]
    assert "root edge" in first.topology_lines[0]
    assert "dominant=" in first.as_text()


def test_demo_three_scenarios_are_repeatable(tmp_path: Path) -> None:
    first = run_demo(tmp_path / "first")
    second = run_demo(tmp_path / "second")
    assert format_demo_summary(first) == format_demo_summary(second)
    names = [item["name"] for item in first["scenarios"]]
    assert names == ["single_stage", "three_stage", "snapshot_restore_replay"]
    cascade = first["scenarios"][1]
    assert cascade["summary"]["final_released_packets"] < cascade["summary"]["total_generated_packets"]
    assert cascade["summary"]["average_delay_ns"] > 0
    assert cascade["indistinguishability"]["indistinguishable"] is True
    replay = first["scenarios"][2]
    assert replay["replay"]["ok"] is True
    assert replay["backup_read_only"] is True
