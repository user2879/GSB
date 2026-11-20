"""异常注入隔离：注入项编号命中并映射到业务码，其余视图继续处理。"""

from __future__ import annotations

import os

from conftest import run_pipeline, write_injection
from mvrwb.errors import (
    E_NON_FINITE_FEATURE,
    E_PARALLAX_OUT_OF_RANGE,
    E_VIEW_INVALID,
)


def _dirs(tmp_path, name):
    return {
        "work": str(tmp_path / f"{name}_work"),
        "snapshot": str(tmp_path / f"{name}_snap"),
        "report": str(tmp_path / f"{name}.json"),
    }


def test_corrupt_png_is_view_invalid_and_others_continue(sample_dir, tmp_path):
    injection_path = write_injection(str(tmp_path / "inj.json"), [
        {"id": "INJ-CORRUPT", "type": "corrupt_png", "view_id": 5, "stage": "load"}
    ])
    report = run_pipeline(sample_dir, _dirs(tmp_path, "corrupt"), injections=injection_path)
    failure = next(f for f in report["failures"] if f["code"] == E_VIEW_INVALID)
    assert failure["injection_id"] == "INJ-CORRUPT"
    # 命中记录写入报告
    hit_ids = {hit["injection_id"] for hit in report["injections"]}
    assert "INJ-CORRUPT" in hit_ids
    # 其余视图仍完成特征提取
    processed = {view["view_id"] for view in report["views"]}
    assert {1, 2, 3, 4, 6}.issubset(processed)
    assert 5 not in processed


def test_missing_camera_is_view_invalid(sample_dir, tmp_path):
    # 先复制工作副本并删除 view6 相机，再启用 missing_camera 注入
    from mvrwb.editor import apply_edit_stream
    dirs = _dirs(tmp_path, "missing")
    apply_edit_stream(sample_dir, dirs["work"], None)
    camera_path = os.path.join(dirs["work"], "views", "view_0006.cam.json")
    os.remove(camera_path)
    injection_path = write_injection(str(tmp_path / "inj_missing.json"), [
        {"id": "INJ-MISSING", "type": "missing_camera", "view_id": 6, "stage": "load"}
    ])
    report = run_pipeline(sample_dir, dirs, injections=injection_path)
    failure = next(f for f in report["failures"]
                   if f["code"] == E_VIEW_INVALID and f.get("view_id") == 6)
    assert failure["injection_id"] == "INJ-MISSING"


def test_parallax_out_of_range_maps_to_business_code(sample_dir, tmp_path):
    injection_path = write_injection(str(tmp_path / "inj_par.json"), [
        {"id": "INJ-PARALLAX", "type": "parallax_out_of_range",
         "pair_id": "vp-0001-0002", "stage": "matching"}
    ])
    report = run_pipeline(sample_dir, _dirs(tmp_path, "parallax"), injections=injection_path)
    failure = next(f for f in report["failures"] if f["code"] == E_PARALLAX_OUT_OF_RANGE)
    assert failure["injection_id"] == "INJ-PARALLAX"
    assert failure["pair_id"] == "vp-0001-0002"
    # 其他视图对不受影响，仍产出匹配
    accepted_pairs = [p["pair_id"] for p in report["pairs"] if p["accepted"]]
    assert "vp-0002-0003" in accepted_pairs


def test_non_finite_feature_is_isolated(sample_dir, tmp_path):
    injection_path = write_injection(str(tmp_path / "inj_nan.json"), [
        {"id": "INJ-NAN", "type": "non_finite_feature", "view_id": 4, "stage": "feature"}
    ])
    report = run_pipeline(sample_dir, _dirs(tmp_path, "nan"), injections=injection_path)
    failure = next(f for f in report["failures"] if f["code"] == E_NON_FINITE_FEATURE)
    assert failure["injection_id"] == "INJ-NAN" and failure["view_id"] == 4
    processed = {view["view_id"] for view in report["views"]}
    assert 4 not in processed and 1 in processed


def test_all_injection_ids_recorded_in_report(sample_dir, tmp_path):
    injection_path = os.path.join(sample_dir, "injection_config.json")
    report = run_pipeline(sample_dir, _dirs(tmp_path, "all"), injections=injection_path)
    hit_ids = {hit["injection_id"] for hit in report["injections"]}
    # 样例注入配置中的四项都应被命中并映射业务码
    assert {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}.issubset(hit_ids)
    codes = {hit["code"] for hit in report["injections"]}
    assert E_VIEW_INVALID in codes
    assert E_PARALLAX_OUT_OF_RANGE in codes
    assert E_NON_FINITE_FEATURE in codes
