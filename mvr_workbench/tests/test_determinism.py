"""确定性、并发/串行一致性与定点量化测试。"""

from __future__ import annotations

import copy

from conftest import deterministic_core, run_pipeline, write_config
from mvrwb.util import quantize_scalar


def test_fixed_point_quantize_is_half_away_from_zero():
    assert quantize_scalar(0.5, 0) == 1
    assert quantize_scalar(-0.5, 0) == -1
    assert quantize_scalar(0.25, 2) == 1
    assert quantize_scalar(-0.25, 2) == -1
    assert quantize_scalar(1.0, 3) == 8


def test_repeated_runs_are_byte_identical(sample_dir, work_dirs, tmp_path):
    first = run_pipeline(sample_dir, work_dirs)
    second_dirs = {
        "work": str(tmp_path / "work2"),
        "snapshot": str(tmp_path / "snap2"),
        "report": str(tmp_path / "report2.json"),
    }
    second = run_pipeline(sample_dir, second_dirs)
    assert deterministic_core(first) == deterministic_core(second)
    assert first["cloud"]["cloud_fingerprint"] == second["cloud"]["cloud_fingerprint"]


def test_worker_count_does_not_change_results(sample_dir, work_dirs, tmp_path):
    baseline = run_pipeline(sample_dir, work_dirs)

    serial_dirs = {
        "work": str(tmp_path / "work_serial"),
        "snapshot": str(tmp_path / "snap_serial"),
        "report": str(tmp_path / "serial.json"),
    }
    # 串行（单线程）工作副本
    from mvrwb.editor import apply_edit_stream
    apply_edit_stream(sample_dir, serial_dirs["work"], None)
    write_config(serial_dirs["work"], {"thread_workers": 1})
    serial = run_pipeline(sample_dir, serial_dirs)

    parallel_dirs = {
        "work": str(tmp_path / "work_par"),
        "snapshot": str(tmp_path / "snap_par"),
        "report": str(tmp_path / "par.json"),
    }
    from mvrwb.editor import apply_edit_stream as copy_stream
    copy_stream(sample_dir, parallel_dirs["work"], None)
    write_config(parallel_dirs["work"], {"thread_workers": 8})
    parallel = run_pipeline(sample_dir, parallel_dirs)

    assert deterministic_core(serial) == deterministic_core(baseline)
    assert deterministic_core(parallel) == deterministic_core(baseline)


def test_output_order_uses_stable_sort_keys(sample_dir, work_dirs):
    report = run_pipeline(sample_dir, work_dirs)
    view_ids = [view["view_id"] for view in report["views"]]
    assert view_ids == sorted(view_ids)
    pair_ids = [pair["pair_id"] for pair in report["pairs"]]
    assert pair_ids == sorted(pair_ids)
    # 点云按体素键空间稳定排序（代表点顺序），从快照读取体素键校验
    import os
    from mvrwb.util import read_json
    cloud_path = os.path.join(
        work_dirs["snapshot"], "cloud", "cloud-global.json"
    )
    cloud = read_json(cloud_path)["content"]["representatives"]
    voxel_keys = [tuple(item["voxel_key"]) for item in cloud]
    assert voxel_keys == sorted(voxel_keys)
    assert [item["point_key"] for item in cloud] == report["cloud"]["point_order"]
