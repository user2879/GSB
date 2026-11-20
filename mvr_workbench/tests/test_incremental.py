"""任意编辑序列后，增量运行与从空白全量重建逐字段一致。"""

from __future__ import annotations

import copy
import os
import shutil

from conftest import deterministic_core, reconstruction_artifacts, run_pipeline, write_config
from mvrwb.util import read_json, write_json_atomic


def _build_final_state(sample_dir: str, target_dir: str) -> str:
    """不经过增量机制，直接构造编辑后的最终视图集与配置，用于全量对照。"""
    shutil.copytree(sample_dir, target_dir)
    # 与 edit_stream.json 等价的最终状态：替换 view2 图像、view3 相机、改阈值与体素
    shutil.copyfile(
        os.path.join(sample_dir, "alt_view_0002.png"),
        os.path.join(target_dir, "views", "view_0002.png"),
    )
    shutil.copyfile(
        os.path.join(sample_dir, "alt_view_0003.cam.json"),
        os.path.join(target_dir, "views", "view_0003.cam.json"),
    )
    write_config(target_dir, {"feature_threshold": 170.0, "voxel_size": 0.28})
    return target_dir


def test_incremental_equals_clean_full_rebuild(sample_dir, work_dirs, tmp_path):
    edits_path = os.path.join(sample_dir, "edit_stream.json")

    # 1) 增量序列：在同一快照上反复运行，逐条施加编辑
    incremental_reports = []
    partial_streams = _split_edit_stream(sample_dir, tmp_path)
    for index, stream in enumerate(partial_streams):
        dirs = {
            "work": str(tmp_path / f"inc_work_{index}"),
            "snapshot": work_dirs["snapshot"],  # 同一持久化快照
            "report": str(tmp_path / f"inc_{index}.json"),
        }
        report = run_pipeline(sample_dir, dirs, edits=stream)
        incremental_reports.append(report)

    # 2) 从空白对最终状态一次性全量重建
    final_dir = _build_final_state(sample_dir, str(tmp_path / "final_state"))
    full_dirs = {
        "work": str(tmp_path / "full_work"),
        "snapshot": str(tmp_path / "full_snapshot"),
        "report": str(tmp_path / "full.json"),
    }
    full_report = run_pipeline(final_dir, full_dirs)

    final_incremental = incremental_reports[-1]
    assert reconstruction_artifacts(final_incremental) == reconstruction_artifacts(full_report)


def test_only_affected_artifacts_are_recomputed(sample_dir, work_dirs, tmp_path):
    # 首次全量
    first = run_pipeline(sample_dir, work_dirs)
    first_recomputed = {(x["stage"], x.get("artifact_id")) for x in first["invalidation"]["recomputed"]}

    # 无编辑二次运行：全部复用、零重算
    rerun_dirs = {
        "work": str(tmp_path / "rerun_work"),
        "snapshot": work_dirs["snapshot"],
        "report": str(tmp_path / "rerun.json"),
    }
    rerun = run_pipeline(sample_dir, rerun_dirs)
    assert rerun["invalidation"]["recomputed"] == []
    reused = {(x["stage"], x.get("artifact_id")) for x in rerun["invalidation"]["reused"]}
    # 首次产物全部进入复用清单
    assert first_recomputed.issubset(reused)

    # 仅替换 view2 图像：只有 view2 特征及其下游匹配/深度/点云失效
    edit = _single_edit_stream(tmp_path, [
        {"command": "replace_image", "view_id": 2, "source_image": "alt_view_0002.png"}
    ])
    edit_dirs = {
        "work": str(tmp_path / "edit_work"),
        "snapshot": work_dirs["snapshot"],
        "report": str(tmp_path / "edit.json"),
    }
    edited = run_pipeline(sample_dir, edit_dirs, edits=edit)
    recomputed_stages = {}
    for item in edited["invalidation"]["recomputed"]:
        recomputed_stages.setdefault(item["stage"], []).append(item)
    # view1 / view3-6 的特征必须直接复用，不得重算
    reused_features = {x.get("view_id") for x in edited["invalidation"]["reused"] if x["stage"] == "feature"}
    assert {1, 3, 4, 5, 6}.issubset(reused_features)
    recomputed_features = {x.get("view_id") for x in edited["invalidation"]["recomputed"] if x["stage"] == "feature"}
    assert recomputed_features == {2}


def test_voxel_size_change_repartitions_cloud(sample_dir, work_dirs, tmp_path):
    first = run_pipeline(sample_dir, work_dirs)
    baseline_cloud = first["cloud"]

    edit = _single_edit_stream(tmp_path, [
        {"command": "update_config", "parameters": {"voxel_size": 0.2}}
    ])
    dirs = {
        "work": str(tmp_path / "vox_work"),
        "snapshot": work_dirs["snapshot"],
        "report": str(tmp_path / "vox.json"),
    }
    changed = run_pipeline(sample_dir, dirs, edits=edit)
    # 体素变小 => 划分更细 => 代表点数量只增不减，且点云指纹真实改变
    assert changed["cloud"]["point_count"] >= baseline_cloud["point_count"]
    assert changed["cloud"]["cloud_fingerprint"] != baseline_cloud["cloud_fingerprint"]
    # 深度阶段不受体素尺寸影响，必须复用
    assert not any(x["stage"] == "depth" for x in changed["invalidation"]["recomputed"])
    assert any(x["stage"] == "cloud" for x in changed["invalidation"]["recomputed"])


def _split_edit_stream(sample_dir, tmp_path):
    stream = read_json(os.path.join(sample_dir, "edit_stream.json"))
    commands = stream["commands"]
    paths = []
    for index in range(1, len(commands) + 1):
        path = str(tmp_path / f"stream_{index}.json")
        write_json_atomic(path, {"commands": commands[:index]})
        paths.append(path)
    return paths


def _single_edit_stream(tmp_path, commands):
    path = str(tmp_path / "single_edit.json")
    write_json_atomic(path, {"commands": commands})
    return path


# 延迟导入，避免顶部循环
from mvrwb.util import read_json  # noqa: E402
