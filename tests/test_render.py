from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mdet_render.cli import main
from mdet_render.errors import BusinessError, E_POINT_BUDGET_PENDING
from mdet_render.png import parse_png
from mdet_render.renderer import Renderer
from mdet_render.util import decode_array
from mdet_render.workspace import load_commands

from conftest import computed_keys, ledger_entries, ledger_tasks, load_workspace, task_by_stable


def _renderer(root: Path, tmp_path: Path, name: str = "render") -> Renderer:
    return Renderer(load_workspace(root), cache_dir=tmp_path / f"{name}-cache",
                    ledger_path=tmp_path / f"{name}-ledger.jsonl")


def test_png_output_is_rgb_uint8_and_repeatable(sample_root: Path, tmp_path: Path) -> None:
    first = _renderer(sample_root, tmp_path, "first")
    result1 = first.run(output_dir=tmp_path / "out1")
    first.close()
    assert result1.complete
    frame_path = tmp_path / "out1" / "frames" / "frame_0000.png"
    info, image = parse_png(frame_path.read_bytes())
    assert (info.width, info.height, info.pixel_mode) == (32, 24, "RGB")
    assert image.pixels.dtype == np.uint8
    expected_digest = hashlib.sha256(b"RGB8|" + image.pixels.tobytes(order="C")).hexdigest()
    assert result1.frames[0]["pixel_digest"] == expected_digest

    second = _renderer(sample_root, tmp_path, "second")
    result2 = second.run(output_dir=tmp_path / "out2")
    second.close()
    assert result1.task_keys == result2.task_keys
    assert {frame: {key: value for key, value in data.items() if key != "png_path"}
            for frame, data in result1.frames.items()} == {
                frame: {key: value for key, value in data.items() if key != "png_path"}
                for frame, data in result2.frames.items()
            }
    for frame in result1.frames:
        assert (tmp_path / "out1" / "frames" / f"frame_{frame:04d}.png").read_bytes() == (
            tmp_path / "out2" / "frames" / f"frame_{frame:04d}.png").read_bytes()


def test_concurrency_limit_and_determinism(sample_root: Path, tmp_path: Path) -> None:
    workspace1 = load_workspace(sample_root)
    workspace1.config.max_workers = 1
    renderer1 = Renderer(workspace1, cache_dir=tmp_path / "c1", ledger_path=tmp_path / "l1.jsonl")
    result1 = renderer1.run(output_dir=tmp_path / "o1")
    renderer1.close()

    workspace2 = load_workspace(sample_root)
    workspace2.config.max_workers = 3
    renderer2 = Renderer(workspace2, cache_dir=tmp_path / "c2", ledger_path=tmp_path / "l2.jsonl")
    result2 = renderer2.run(output_dir=tmp_path / "o2")
    renderer2.close()

    assert result1.task_keys == result2.task_keys
    assert {k: v["pixel_digest"] for k, v in result1.frames.items()} == {
        k: v["pixel_digest"] for k, v in result2.frames.items()
    }
    assert result1.max_observed_workers <= 1
    assert result2.max_observed_workers <= 3
    assert result2.max_observed_workers >= 2


def test_incremental_command_replay_matches_full_render(sample_root: Path, tmp_path: Path) -> None:
    commands = load_commands(sample_root / "commands.json")
    workspace = load_workspace(sample_root)
    ledger = tmp_path / "incremental.jsonl"
    renderer = Renderer(workspace, cache_dir=tmp_path / "incremental-cache", ledger_path=ledger)
    result = renderer.run(output_dir=tmp_path / "incremental-out")
    final = result
    for command in commands:
        workspace = workspace.apply_command(command)
        renderer.workspace = workspace
        final = renderer.run(output_dir=tmp_path / "incremental-out")
        assert final.complete
    final_tasks = ledger_tasks(ledger, final.run_id)
    assert {entry["outcome"] for entry in final_tasks} == {"reused"}
    renderer.close()

    full_workspace = load_workspace(sample_root)
    for command in commands:
        full_workspace = full_workspace.apply_command(command)
    full_renderer = Renderer(full_workspace, cache_dir=tmp_path / "full-cache",
                             ledger_path=tmp_path / "full.jsonl")
    full = full_renderer.run(output_dir=tmp_path / "full-out")
    full_renderer.close()

    assert final.task_keys == full.task_keys
    for frame in final.frames:
        assert final.frames[frame]["pixel_digest"] == full.frames[frame]["pixel_digest"]
        assert final.frames[frame]["png_digest"] == full.frames[frame]["png_digest"]
        assert [block["pixel_digest"] for block in final.frames[frame]["blocks"]] == [
            block["pixel_digest"] for block in full.frames[frame]["blocks"]
        ]
        assert (tmp_path / "incremental-out" / "frames" / f"frame_{frame:04d}.png").read_bytes() == (
            tmp_path / "full-out" / "frames" / f"frame_{frame:04d}.png").read_bytes()
    assert [batch["selected"] for batch in final.scheduler_batches] == [
        batch["selected"] for batch in full.scheduler_batches
    ]
    assert [batch["pending"] for batch in final.scheduler_batches] == [
        batch["pending"] for batch in full.scheduler_batches
    ]


def test_targeted_invalidation_is_exact(sample_root: Path, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=ledger)
    renderer.run(output_dir=tmp_path / "out")

    workspace = workspace.apply_command({
        "op": "replace_prompt",
        "frame": 1,
        "prompt": {"entities": ["cloud", "gear"], "attributes": ["violet", "glass", "soft"]},
    })
    renderer.workspace = workspace
    prompt_result = renderer.run(output_dir=tmp_path / "out")
    computed = computed_keys(ledger, prompt_result.run_id)
    assert "encode/frame=0001/text" in computed
    assert [key for key in computed if key.startswith("encode/")] == ["encode/frame=0001/text"]
    assert all("frame=0001" in key for key in computed if not key.startswith("encode/"))
    assert not any("frame=0000" in key or "frame=0002" in key for key in computed)
    task_map = task_by_stable(ledger, prompt_result.run_id)
    assert "upstream_changed:condition:text" in task_map["fusion/frame=0001"]["invalidation_basis"]

    workspace = workspace.apply_command({"op": "set_weight", "condition": "image", "weight": 1.25})
    renderer.workspace = workspace
    weight_result = renderer.run(output_dir=tmp_path / "out")
    weight_computed = computed_keys(ledger, weight_result.run_id)
    assert not [key for key in weight_computed if key.startswith("encode/")]
    assert {key.split("/")[0] for key in weight_computed} == {
        "fusion", "latent", "denoise", "decode", "block"
    }
    fusion_frames = {int(key.split("frame=")[1].split("/")[0])
                     for key in weight_computed if key.startswith("fusion/")}
    assert fusion_frames == {0, 1, 2}

    workspace = workspace.apply_command({"op": "set_decode_block_rows", "block_rows": 5})
    renderer.workspace = workspace
    block_result = renderer.run(output_dir=tmp_path / "out")
    block_computed = computed_keys(ledger, block_result.run_id)
    assert block_computed
    assert {key.split("/")[0] for key in block_computed} == {"decode", "block"}

    workspace = workspace.apply_command({"op": "set_lease", "ttl_ms": 20_000})
    renderer.workspace = workspace
    lease_result = renderer.run(output_dir=tmp_path / "out")
    lease_tasks = ledger_tasks(ledger, lease_result.run_id)
    assert lease_tasks
    assert {entry["outcome"] for entry in lease_tasks} == {"reused"}
    renderer.close()


def test_block_rows_only_read_their_own_latent_slice_and_do_not_change_pixels(
        sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")
    first = renderer.run(output_dir=tmp_path / "out")
    matrix, bias = renderer._projection()
    final_state = decode_array(renderer._products["denoise/frame=0000/step=0003"]["payload"]["state"])
    for block in first.frames[0]["blocks"]:
        start, end = block["row_start"], block["row_end"]
        expected = renderer._decode_rows(final_state[start:end], matrix, bias)
        product = renderer._products[f"block/frame=0000/row={block['block']:04d}"]
        actual = decode_array(product["payload"]["pixels"])
        assert np.array_equal(actual, expected)
        tampered = final_state.copy()
        tampered[:start] += 500
        tampered[end:] += 500
        assert np.array_equal(renderer._decode_rows(tampered[start:end], matrix, bias), actual)

    workspace = workspace.apply_command({"op": "set_decode_block_rows", "block_rows": 5})
    renderer.workspace = workspace
    second = renderer.run(output_dir=tmp_path / "out")
    assert len(first.frames[0]["blocks"]) == 4
    assert len(second.frames[0]["blocks"]) == 5
    assert first.frames[0]["pixel_digest"] == second.frames[0]["pixel_digest"]
    assert [block["cache_key"] for block in first.frames[0]["blocks"]] != [
        block["cache_key"] for block in second.frames[0]["blocks"]
    ]
    renderer.close()


def test_point_budget_pending_and_next_batch_priority(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    workspace.config.point_budget = workspace.config.denoise_points_per_step
    ledger = tmp_path / "ledger.jsonl"
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=ledger)
    partial = renderer.run(output_dir=tmp_path / "out", max_batches=3)
    assert not partial.complete
    assert partial.code == E_POINT_BUDGET_PENDING
    assert partial.pending_tasks
    batches = partial.scheduler_batches
    assert batches[0]["selected"] == ["denoise/frame=0000/step=0000"]
    assert batches[1]["selected"] == ["denoise/frame=0001/step=0000"]
    assert batches[2]["selected"] == ["denoise/frame=0002/step=0000"]
    assert batches[0]["deferred"][0] == batches[1]["selected"][0]
    assert batches[0]["code"] == E_POINT_BUDGET_PENDING

    completed = renderer.run(output_dir=tmp_path / "out")
    assert completed.complete

    reference = _renderer(sample_root, tmp_path, "reference")
    full = reference.run(output_dir=tmp_path / "reference-out")
    reference.close()
    assert completed.task_keys == full.task_keys
    assert {key: value["pixel_digest"] for key, value in completed.frames.items()} == {
        key: value["pixel_digest"] for key, value in full.frames.items()
    }
    renderer.close()


def test_invalid_command_is_transactional_and_cli_fails_nonzero(sample_root: Path, tmp_path: Path,
                                                                capsys) -> None:
    workspace = load_workspace(sample_root)
    before = workspace.command_snapshot()
    with pytest.raises(BusinessError) as caught:
        workspace.apply_command({
            "op": "replace_prompt",
            "frame": 0,
            "prompt": {"entities": ["not_allowed"], "attributes": []},
        })
    assert caught.value.code == "E_PROMPT_TERM"
    assert workspace.command_snapshot() == before

    bad_commands = tmp_path / "bad_commands.json"
    bad_commands.write_text(json.dumps([{
        "op": "replace_prompt",
        "frame": 0,
        "prompt": {"entities": ["not_allowed"], "attributes": []},
    }]), encoding="utf-8")
    output_dir = tmp_path / "never-written"
    code = main([
        "render", "--config", str(sample_root / "config.json"),
        "--commands", str(bad_commands), "--cache-dir", str(tmp_path / "cache"),
        "--ledger", str(tmp_path / "ledger.jsonl"), "--out", str(output_dir),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code != 0
    assert payload["code"] == "E_PROMPT_TERM"
    assert not (output_dir / "render_manifest.json").exists()


def test_denoise_step_count_reuses_unchanged_prefix(sample_root: Path, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=ledger)
    renderer.run(output_dir=tmp_path / "out")

    workspace = workspace.apply_command({"op": "set_denoise_steps", "steps": 3})
    renderer.workspace = workspace
    shorter = renderer.run(output_dir=tmp_path / "out")
    shorter_computed = computed_keys(ledger, shorter.run_id)
    assert {key.split("/")[0] for key in shorter_computed} == {"decode", "block"}

    workspace = workspace.apply_command({"op": "set_denoise_steps", "steps": 4})
    renderer.workspace = workspace
    longer = renderer.run(output_dir=tmp_path / "out")
    longer_tasks = task_by_stable(ledger, longer.run_id)
    for frame in range(3):
        for step in range(4):
            assert longer_tasks[f"denoise/frame={frame:04d}/step={step:04d}"]["outcome"] == "reused"
    assert computed_keys(ledger, longer.run_id) == []
    renderer.close()


def test_malformed_command_payload_is_a_business_error(sample_root: Path) -> None:
    workspace = load_workspace(sample_root)
    before = workspace.command_snapshot()
    with pytest.raises(BusinessError) as caught:
        workspace.apply_command({"op": "replace_prompt", "frame": 0, "prompt": "not-an-object"})
    assert caught.value.code == "E_PROMPT_SCHEMA"
    assert workspace.command_snapshot() == before
    with pytest.raises(BusinessError) as caught:
        workspace.apply_command({"op": "delete_image", "frame": "0"})
    assert caught.value.code == "E_COMMAND_SCHEMA"
    assert workspace.command_snapshot() == before
