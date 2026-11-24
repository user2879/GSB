from __future__ import annotations

from pathlib import Path

from gsscene.pipeline import ScenePipeline

from conftest import copy_sample, write_edits


def _baseline(sample: Path, work: Path) -> ScenePipeline:
    pipeline = ScenePipeline(sample, work, replay_edits=True)
    pipeline.run(0)
    return pipeline


def test_camera_tweak_invalidates_only_that_view(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(
        sample,
        [
            {
                "seq": 0,
                "type": "camera_tweak",
                "params": {"view_index": 1, "rotation_delta": [[0, 0.01, 0], [0, 0, 0], [0, 0, 0]]},
            }
        ],
    )
    pipeline = _baseline(sample, tmp_path / "work")
    report = pipeline.run(1)
    actions = report["actual_invalidation"]
    assert actions["recompute_kernel_groups"] == [[1, 0], [1, 1], [1, 2]]
    assert actions["recompute_composites"] == [[1, 0], [1, 1], [1, 2]]
    assert all(tile[0] == 1 for tile in actions["recompute_tiles"])
    assert actions["reused"] > 0


def test_quant_switch_reuses_kernels_and_invalidates_render(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(sample, [{"seq": 0, "type": "quant_switch", "params": {"frac_bits": 10, "quant_gear": "FLOOR"}}])
    pipeline = _baseline(sample, tmp_path / "work")
    report = pipeline.run(1)
    actions = report["actual_invalidation"]
    assert actions["recompute_kernels"] == []
    assert actions["reuse_kernels"]
    assert actions["recompute_tiles"]
    assert len(actions["recompute_composites"]) == 6 * 3


def test_sh_modify_invalidates_exact_kernel_and_downstream(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(
        sample,
        [
            {
                "seq": 0,
                "type": "sh_modify",
                "params": {
                    "view_index": 0,
                    "levels": [0, 1],
                    "kernel_index": 0,
                    "coefficient": 1,
                    "channel": 0,
                    "delta": 0.05,
                },
            }
        ],
    )
    pipeline = _baseline(sample, tmp_path / "work")
    report = pipeline.run(1)
    actions = report["actual_invalidation"]
    assert actions["recompute_kernels"] == [[0, 0, 0], [0, 1, 0]]
    assert len(actions["recompute_tiles"]) == 2
    assert actions["recompute_composites"] == [[0, 0], [0, 1], [0, 2]]


def test_level_scale_change_stays_in_level_and_composite_closure(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(sample, [{"seq": 0, "type": "level_scale", "params": {"level": 1, "value": 0.55}}])
    pipeline = _baseline(sample, tmp_path / "work")
    report = pipeline.run(1)
    actions = report["actual_invalidation"]
    assert actions["recompute_kernel_groups"]
    assert all(group[1] == 1 for group in actions["recompute_kernel_groups"])
    assert actions["recompute_composites"] == [[view, level] for view in [0, 1, 2, 3, 4, 5] for level in [1, 2]]


def test_view_remove_reuses_remaining_and_deletes_orphans(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(sample, [{"seq": 0, "type": "view_remove", "params": {"view_index": 1}}])
    pipeline = _baseline(sample, tmp_path / "work")
    report = pipeline.run(1)
    assert report["active_views"] == [0, 2, 3, 4, 5]
    actions = report["actual_invalidation"]
    assert actions["recompute_kernels"] == []
    assert actions["recompute_tiles"] == []
    assert actions["recompute_composites"] == []
    assert not (tmp_path / "work" / "cache" / "kernels" / "0001_L00.npz").exists()


def test_view_add_only_computes_new_view(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(sample, [{"seq": 0, "type": "view_add", "params": {"view_index": 6, "asset_dir": "assets/add_0006"}}])
    pipeline = _baseline(sample, tmp_path / "work")
    report = pipeline.run(1)
    assert 6 in report["active_views"]
    actions = report["actual_invalidation"]
    assert all(group[0] == 6 for group in actions["recompute_kernel_groups"])
    assert all(tile[0] == 6 for tile in actions["recompute_tiles"])
    assert actions["recompute_composites"] == [[6, 0], [6, 1], [6, 2]]
