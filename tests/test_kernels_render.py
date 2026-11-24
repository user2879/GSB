from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gsscene.compare import compare_cache_arrays
from gsscene.config import SceneConfig
from gsscene.pipeline import ScenePipeline
from gsscene.rendering import render_tile_arrays
from gsscene.util import read_json, read_npz

from conftest import copy_sample


def _first_kernel_and_config(work: Path) -> tuple[SceneConfig, dict[str, np.ndarray]]:
    state = read_json(work / "state" / "state.json")
    config = SceneConfig.from_dict(state["config"])
    arrays = read_npz(work / "cache" / "kernels" / "0000_L00.npz")
    return config, arrays


def test_kernel_and_render_are_deterministic(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    work_a = tmp_path / "a"
    work_b = tmp_path / "b"
    report_a = ScenePipeline(sample, work_a, replay_edits=False).run(0)
    report_b = ScenePipeline(sample, work_b, replay_edits=False).run(0)
    assert report_a["failures"] == report_b["failures"] == []
    assert compare_cache_arrays(work_a, work_b)["equal"]
    assert report_a["key_manifest"] == report_b["key_manifest"]


def test_repeat_run_reuses_every_artifact(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    work = tmp_path / "work"
    report = ScenePipeline(sample, work, replay_edits=False).run(0)
    assert report["actual_invalidation"]["reused"] == 0
    second = ScenePipeline(sample, work, replay_edits=False).run(0)
    actions = second["actual_invalidation"]
    assert actions["recompute_kernels"] == []
    assert actions["recompute_tiles"] == []
    assert actions["recompute_composites"] == []
    assert actions["reused"] > 0
    assert second["resume_mode"] == "RESUME"


def test_quantization_gear_and_width_change_integer_paths(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    work = tmp_path / "work"
    report = ScenePipeline(sample, work, replay_edits=False).run(0)
    config, kernels = _first_kernel_and_config(work)
    refs = range(len(next(iter([kernels["opacity"]]))))
    outputs = []
    for gear, bits in (("NEAREST", 12), ("FLOOR", 12), ("CEIL", 12), ("NEAREST", 10)):
        arrays = render_tile_arrays(config, kernels, refs, 0, 0, 0, frac_bits=bits, quant_gear=gear)
        outputs.append((gear, bits, arrays))
        assert arrays["residual"].shape == arrays["round_dir"].shape
    base = outputs[0][2]
    for gear, bits, arrays in outputs[1:]:
        assert not np.array_equal(base["pixels_q"], arrays["pixels_q"]), (gear, bits)
        assert not np.array_equal(base["weight_q"], arrays["weight_q"]), (gear, bits)
        assert arrays["round_dir"].dtype == np.int8


def test_reference_path_has_extra_fixed_bit(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    work = tmp_path / "work"
    report = ScenePipeline(sample, work, replay_edits=False).run(0)
    assert report["diagnostics"]
    for diagnostic in report["diagnostics"]:
        assert diagnostic["stage_values"]["kernel_init"] >= 0.0
        assert diagnostic["stage_values"]["layer_render"] >= 0.0
        assert diagnostic["stage_values"]["layer_composite"] >= 0.0
        assert set(diagnostic["tolerance"]) == {"kernel", "render", "composite"}
