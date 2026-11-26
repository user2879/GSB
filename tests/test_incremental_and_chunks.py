from __future__ import annotations

from mmfm.renderer import Renderer


def render(workspace, snapshot, budget=100):
    return Renderer(workspace, snapshot, budget_override=budget).render()


def events(result, stage):
    return [event for event in result["ledger_events"] if event["stage"] == stage]


def test_selective_image_edit_invalidates_only_referencing_frames(workspace, tmp_path):
    initial = render(workspace, tmp_path / "snap")
    edited, applied = workspace.apply_commands("samples/edits/selective_image.json")
    assert applied[0]["id"] == "img_shared"
    changed = render(edited, tmp_path / "snap")
    assert changed["budget"]["batch_spent"] == 6
    frame1_steps = [event for event in events(changed, "denoise") if event["frame"] == 1]
    frame2_steps = [event for event in events(changed, "denoise") if event["frame"] == 2]
    frame3_steps = [event for event in events(changed, "denoise") if event["frame"] == 3]
    assert frame1_steps and all(event["event"] == "reuse_hit" for event in frame1_steps)
    assert all(event["hit_origin"] == "committed" for event in frame1_steps)
    assert len(frame2_steps) == 3 and all(event["event"] == "recompute" for event in frame2_steps)
    assert len(frame3_steps) == 3 and all(event["event"] == "recompute" for event in frame3_steps)
    assert initial["frames"][0]["pixel_digest"] == changed["frames"][0]["pixel_digest"]
    assert initial["frames"][1]["pixel_digest"] != changed["frames"][1]["pixel_digest"]
    assert initial["frames"][2]["pixel_digest"] != changed["frames"][2]["pixel_digest"]
    bad = [event for event in changed["ledger_events"] if event["event"] == "recompute"]
    assert all(event["invalidation_basis"]["reason"] for event in bad)
    shared_new = [
        event
        for event in changed["ledger_events"]
        if event.get("logical_id") == "condition:3:image"
    ][0]
    assert shared_new["hit_origin"] == "current_run"
    assert shared_new["invalidation_basis"]["reason"] == "current_run_content_reuse"


def test_fusion_weight_edit_invalidates_all_frames(workspace, tmp_path):
    render(workspace, tmp_path / "snap")
    edited, _ = workspace.apply_commands("samples/edits/fusion_weight.json")
    changed = render(edited, tmp_path / "snap")
    assert changed["budget"]["batch_spent"] == 9
    denoise = events(changed, "denoise")
    assert len(denoise) == 9
    assert all(event["event"] == "recompute" for event in denoise)
    assert all(event["invalidation_basis"]["reason"] == "upstream_content_changed" for event in denoise)


def test_chunk_rows_change_reuses_steps_but_repartitions_rows(workspace, tmp_path):
    initial = render(workspace, tmp_path / "snap")
    edited, _ = workspace.apply_commands("samples/edits/chunk_rows.json")
    changed = render(edited, tmp_path / "snap")
    assert changed["budget"]["batch_spent"] == 0
    assert all(event["event"] == "reuse_hit" for event in events(changed, "denoise"))
    assert all(event["event"] == "reuse_hit" for event in events(changed, "frame_decode"))
    chunk_events = events(changed, "chunk")
    assert len(chunk_events) == 9
    assert all(event["event"] == "recompute" for event in chunk_events)
    for before, after in zip(initial["frames"], changed["frames"]):
        assert before["pixel_digest"] == after["pixel_digest"]
        assert len(before["chunks"]) == 2
        assert len(after["chunks"]) == 3
        assert [chunk["row_range"] for chunk in after["chunks"]] == [[0, 10], [10, 21], [21, 32]]


def test_denoise_steps_edit_only_computes_new_step(workspace, tmp_path):
    render(workspace, tmp_path / "snap")
    edited, _ = workspace.apply_commands("samples/edits/denoise_steps.json")
    changed = render(edited, tmp_path / "snap")
    denoise = events(changed, "denoise")
    assert changed["budget"]["batch_spent"] == 3
    assert sum(event["event"] == "recompute" for event in denoise) == 3
    assert all(event["step"] == 4 for event in denoise if event["event"] == "recompute")
