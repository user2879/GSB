from __future__ import annotations

from mmfm.renderer import Renderer


def test_budget_pending_and_resume_match_full_render(workspace, tmp_path):
    full = Renderer(workspace, tmp_path / "full", budget_override=100).render()
    limited = Renderer(workspace, tmp_path / "limited", budget_override=2).render()
    assert limited["code"] == "E_BUDGET_PENDING"
    assert limited["budget"]["batch_spent"] == 2
    assert limited["pending"]
    assert all(item["code"] == "E_BUDGET_PENDING" for item in limited["pending"])
    resumed = Renderer(workspace, tmp_path / "limited", budget_override=100).render()
    assert resumed["code"] == "OK"
    assert resumed["budget"]["batch_spent"] == 7
    assert [frame["pixel_digest"] for frame in full["frames"]] == [
        frame["pixel_digest"] for frame in resumed["frames"]
    ]
    assert [frame["png_sha256"] for frame in full["frames"]] == [
        frame["png_sha256"] for frame in resumed["frames"]
    ]


def test_worker_count_does_not_change_budget_sequence_or_pending(workspace, tmp_path):
    serial_ws, _ = workspace.apply_commands([{"op": "set_max_workers", "value": 1}])
    parallel_ws, _ = workspace.apply_commands([{"op": "set_max_workers", "value": 4}])
    serial = Renderer(serial_ws, tmp_path / "serial", budget_override=5).render()
    parallel = Renderer(parallel_ws, tmp_path / "parallel", budget_override=5).render()
    assert serial["code"] == parallel["code"] == "E_BUDGET_PENDING"
    assert serial["budget"]["deduction_sequence"] == parallel["budget"]["deduction_sequence"]
    assert serial["pending"] == parallel["pending"]
    assert serial["max_observed_concurrency"] == 1
    assert 1 <= parallel["max_observed_concurrency"] <= 4
    serial_done = {frame["frame"]: frame for frame in serial["frames"] if frame["status"] == "complete"}
    parallel_done = {frame["frame"]: frame for frame in parallel["frames"] if frame["status"] == "complete"}
    assert serial_done.keys() == parallel_done.keys()
    for frame in serial_done:
        assert serial_done[frame]["pixel_digest"] == parallel_done[frame]["pixel_digest"]
        assert serial_done[frame]["chunks"] == parallel_done[frame]["chunks"]

    serial_full = Renderer(serial_ws, tmp_path / "serial_full", budget_override=100).render()
    parallel_full = Renderer(parallel_ws, tmp_path / "parallel_full", budget_override=100).render()
    assert serial_full["code"] == parallel_full["code"] == "OK"
    assert serial_full["frames"] == parallel_full["frames"]
    assert serial_full["ledger_events"] == parallel_full["ledger_events"]
    assert serial_full["cursor"] == parallel_full["cursor"]
    assert serial_full["budget"]["deduction_sequence"] == parallel_full["budget"]["deduction_sequence"]
    assert serial_full["pending"] == parallel_full["pending"] == []
