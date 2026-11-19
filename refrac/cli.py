"""命令行入口：端到端可复现验证流水线。

子命令：
- clean      清理 samples/reports/artifacts 旧产物
- generate   确定性生成场景 JSON 与元数据清单
- render     渲染一帧并落盘 RGBA/报告/哈希
- suite      运行完整验收流水线（双跑字节比对、变体、注入套件、反例）
- replay     按反例片段单独复现单像素错误码
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Optional

import numpy as np

from .camera import analytic_coverage, build_camera_frame
from .diagnostics import (
    build_minimal_counterexample,
    first_affected_pixel,
    serialize_reports,
    verify_counterexample_minimality,
)
from .errors import ERROR_CODES, RefracError, classify_exception
from .injection import InjectionConfig, injection_from_dict
from .inject_cases import build_injection_suite
from .persistence import save_json, save_rgba, sha256_bytes
from .scene import generate_scene, load_scene
from .tracer import render_frame

DEFAULT_DIRS = ("samples", "reports", "artifacts")


def clean_outputs(workspace: str) -> list[str]:
    removed = []
    for name in DEFAULT_DIRS:
        target = os.path.join(workspace, name)
        if os.path.exists(target):
            shutil.rmtree(target)
            removed.append(target)
    for name in DEFAULT_DIRS:
        os.makedirs(os.path.join(workspace, name), exist_ok=True)
    return removed


def _write_scene(workspace: str, scene) -> dict[str, str]:
    frame = build_camera_frame(scene.camera)
    coverage = analytic_coverage(scene, frame)
    scene_path = os.path.join(workspace, "samples", "scene.json")
    meta_path = os.path.join(workspace, "samples", "scene_metadata.json")
    save_json(scene_path, scene.to_dict())
    metadata = {
        "seed": scene.seed,
        "generator_tag": scene.generator_tag,
        "camera": scene.camera.__dict__,
        "bodies": [
            {
                "body_id": body.body_id,
                "kind": body.kind,
                "center": list(body.center) if body.center else None,
                "radius": body.radius,
                "axis": body.axis,
                "thickness": body.thickness,
                "ior": body.ior,
                "absorption": list(body.absorption),
                "surface_normal_truth": _surface_normal_truth(body),
                "projected_coverage": coverage[body.body_id],
            }
            for body in scene.bodies
        ],
        "background": scene.background.__dict__,
        "light": scene.light.__dict__,
    }
    save_json(meta_path, metadata)
    return {"scene": scene_path, "metadata": meta_path}


def _surface_normal_truth(body) -> Optional[list[list[float]]]:
    if body.kind == "sphere":
        return "unit radial normal n=(p-center)/radius"
    if getattr(body, "slab_normal", None) is not None:
        normal = list(body.slab_normal)
        return [[-v for v in normal], normal]
    axis = int(body.axis)
    plus = [0.0, 0.0, 0.0]
    minus = [0.0, 0.0, 0.0]
    plus[axis] = 1.0
    minus[axis] = -1.0
    return [minus, plus]


def _render_and_persist(workspace: str, scene, injection: Optional[InjectionConfig],
                        label: str) -> dict[str, Any]:
    result = render_frame(scene, injection)
    effective = result["scene"]
    rgba = result["rgba"]
    rgba_hash = save_rgba(os.path.join(workspace, "artifacts", f"{label}_rgba"), rgba)
    report_text = serialize_reports(result["reports"])
    report_path, report_hash = save_json(
        os.path.join(workspace, "reports", f"{label}_path_report.json"), report_text
    )
    counterexample = build_minimal_counterexample(scene, injection, result["isolated"])
    minimality = None
    if counterexample is not None:
        minimality = verify_counterexample_minimality(scene, injection, counterexample)
    summary = {
        "label": label,
        "injection": injection.to_dict() if injection else None,
        "resolution": [scene.camera.height, scene.camera.width],
        "rgba_dtype": "float32",
        "rgba_shape": list(rgba.shape),
        "rgba_hash_sha256": rgba_hash,
        "path_report_hash_sha256": report_hash,
        "termination_counts": result["termination_counts"],
        "coverage_hits": result["coverage_hits"],
        "isolated_count": len(result["isolated"]),
        "isolated_records": result["isolated"],
        "first_affected_pixel": first_affected_pixel(result["isolated"]),
        "minimal_counterexample": counterexample,
        "counterexample_minimality": minimality,
        "effective_max_depth": effective.max_depth,
        "effective_sample_count": effective.sample_count,
    }
    summary_path, summary_hash = save_json(
        os.path.join(workspace, "reports", f"{label}_summary.json"), summary
    )
    summary["summary_hash_sha256"] = summary_hash
    return summary


def cmd_clean(args) -> int:
    removed = clean_outputs(args.workspace)
    print(json.dumps({"removed": removed, "recreated": list(DEFAULT_DIRS)}, indent=2))
    return 0


def cmd_generate(args) -> int:
    scene = generate_scene(seed=args.seed, width=args.width, height=args.height,
                           sample_count=args.sample_count, max_depth=args.max_depth)
    paths = _write_scene(args.workspace, scene)
    print(json.dumps(paths, indent=2))
    return 0


def cmd_render(args) -> int:
    scene = load_scene(args.scene)
    injection = injection_from_dict(load_json_file(args.injection)) if args.injection else None
    summary = _render_and_persist(args.workspace, scene, injection, args.label)
    print(json.dumps({"label": summary["label"],
                      "rgba_hash_sha256": summary["rgba_hash_sha256"],
                      "isolated_count": summary["isolated_count"]}, indent=2))
    return 0


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _compare_rgba_exclusive(a: np.ndarray, b: np.ndarray, allowed: set[tuple[int, int]]) -> dict[str, Any]:
    diff_pixels = np.argwhere(np.any(a != b, axis=2))
    unexpected = [(int(r), int(c)) for r, c in diff_pixels if (int(r), int(c)) not in allowed]
    return {
        "different_pixel_count": int(len(diff_pixels)),
        "all_differences_within_allowed_set": not unexpected,
        "unexpected_pixels": unexpected[:20],
    }


def _compare_reports(reports_a, reports_b) -> dict[str, Any]:
    """深度上限变体比对。

    新被隔离的像素（触及深度上限）允许 termination 与隔离相关字段变化；
    其余像素的全部字段必须逐字段不变。
    """
    changed_termination: list[list[int]] = []
    field_violations: list[dict[str, Any]] = []
    isolation_keys = {"termination", "isolated", "error_code", "rgba", "events",
                      "interface_crossings", "internal_length", "tir_count"}
    for trace_a, trace_b in zip(reports_a, reports_b):
        key = (trace_a.row, trace_a.col)
        became_isolated = (not trace_a.isolated) and trace_b.isolated
        if trace_a.termination != trace_b.termination:
            changed_termination.append([trace_a.row, trace_a.col])
        if became_isolated:
            continue
        fields_a = serialize_report_fields(trace_a)
        fields_b = serialize_report_fields(trace_b)
        differing = sorted(k for k in fields_a if fields_a[k] != fields_b[k])
        unexpected = [k for k in differing if k not in isolation_keys or k == "termination"]
        if differing and unexpected:
            field_violations.append({"pixel": list(key), "keys": differing})
    return {
        "changed_termination_pixels": changed_termination,
        "changed_termination_count": len(changed_termination),
        "non_termination_field_violations": field_violations[:20],
        "field_wise_identical_else": not field_violations,
    }


def serialize_report_fields(trace) -> dict[str, Any]:
    return {
        "row": trace.row,
        "col": trace.col,
        "rgba": [float(v) for v in trace.color],
        "termination": trace.termination,
        "interface_crossings": trace.interface_crossings,
        "tir_count": trace.tir_count,
        "internal_length": trace.internal_length,
        "events": trace.events,
        "isolated": trace.isolated,
        "error_code": trace.error_code,
    }


def cmd_suite(args) -> int:
    import platform
    import numpy as np

    workspace = args.workspace
    clean_outputs(workspace)

    # 1) 确定性场景生成：两次生成逐字节一致。
    scene_a = generate_scene(seed=args.seed, width=args.width, height=args.height)
    scene_b = generate_scene(seed=args.seed, width=args.width, height=args.height)
    generation_byte_identical = scene_a.to_json() == scene_b.to_json()
    _write_scene(workspace, scene_a)
    scene_hash = sha256_bytes(scene_a.to_json().encode("utf-8"))

    # 2) 基线双跑：RGBA 与路径报告逐字节一致。
    run_1 = _render_and_persist(workspace, scene_a, None, "baseline_run1")
    run_2 = _render_and_persist(workspace, scene_a, None, "baseline_run2")
    baseline_identical = (
        run_1["rgba_hash_sha256"] == run_2["rgba_hash_sha256"]
        and run_1["path_report_hash_sha256"] == run_2["path_report_hash_sha256"]
    )

    # 3) 吸收采样数变体：仅内部长度非零的像素允许变化。
    scene_s16 = generate_scene(seed=args.seed, width=args.width, height=args.height,
                               sample_count=16, max_depth=scene_a.max_depth)
    _render_and_persist(workspace, scene_s16, None, "samplecount16")
    rgba_base = np.load(os.path.join(workspace, "artifacts", "baseline_run1_rgba.npy"))
    rgba_s16 = np.load(os.path.join(workspace, "artifacts", "samplecount16_rgba.npy"))
    absorbing_pixels = {(t.row, t.col) for t in _reports(workspace, "baseline_run1")
                        if t.internal_length > 0.0}
    sample_check = _compare_rgba_exclusive(rgba_base, rgba_s16, absorbing_pixels)
    sample_check["absorbing_pixel_count"] = len(absorbing_pixels)
    # 常量 sigma 下采样积分与解析式数学等价：差异像素必须是吸收像素的子集（含零差异）。
    analytic_ok = all(
        event["absorption_deviation"] <= 1e-10
        for trace in _reports(workspace, "baseline_run1")
        for event in trace.events if event["type"] == "absorption"
    )

    # 4) 深度上限变体：仅触及上限的像素终止原因变化，其余逐字段不变。
    scene_d2 = generate_scene(seed=args.seed, width=args.width, height=args.height,
                              sample_count=scene_a.sample_count, max_depth=2)
    _render_and_persist(workspace, scene_d2, None, "depthlimit2")
    depth_check = _compare_reports(_reports(workspace, "baseline_run1"),
                                   _reports(workspace, "depthlimit2"))

    # 5) 注入套件：错误归类、像素隔离、首个受影响像素、最小反例。
    # 选用确定性的零穿越角像素，保证像素级注入不被基线深度预算掩盖。
    center = (0, 0)
    suite = build_injection_suite(center)
    # 折射率等于空气（1.0）属于合法边界：临界角退化为不存在，
    # 光线按 NO_CRITICAL_ANGLE 传播，不产生隔离（由 test_optics 单元测试断言）。
    # 该用例在足够高的深度预算下单独验证，避免与基线深度触顶混淆。
    degenerate = InjectionConfig(
        name="inj_ior_one_degenerate_critical",
        body_overrides={"sphere_glass": {"ior": 1.0}},
        pixel_overrides=[], render_overrides={"max_depth": 64},
    )
    suite["inj_ior_one_degenerate_critical"] = degenerate
    injection_results: dict[str, Any] = {}
    for label, injection in suite.items():
        inject_path = os.path.join(workspace, "samples", f"{label}.json")
        save_json(inject_path, injection.to_dict())
        try:
            summary = _render_and_persist(workspace, scene_a, injection, label)
        except RefracError as exc:
            summary = {
                "label": label,
                "run_level_error": True,
                "code": classify_exception(exc),
                "message": str(exc),
                "context": exc.context,
            }
            save_json(os.path.join(workspace, "reports", f"{label}_summary.json"), summary)
        injection_results[label] = summary

    manifest = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "seed": args.seed,
        "error_codes_declared": list(ERROR_CODES),
        "scene_generation_byte_identical": generation_byte_identical,
        "scene_hash_sha256": scene_hash,
        "baseline_double_run_byte_identical": baseline_identical,
        "baseline_rgba_hash_sha256": run_1["rgba_hash_sha256"],
        "baseline_report_hash_sha256": run_1["path_report_hash_sha256"],
        "sample_count_variant_check": sample_check,
        "absorption_analytic_tolerance_ok": analytic_ok,
        "depth_variant_check": {
            "changed_termination_count": depth_check["changed_termination_count"],
            "changed_termination_first20": depth_check["changed_termination_pixels"][:20],
            "field_wise_identical_else": depth_check["field_wise_identical_else"],
            "non_termination_field_violations": depth_check["non_termination_field_violations"],
        },
        "baseline_termination_counts": run_1["termination_counts"],
        "baseline_tir_total": sum(t.tir_count for t in _reports(workspace, "baseline_run1")),
        "injection_runs": _compact_injection_results(injection_results),
    }
    save_json(os.path.join(workspace, "reports", "run_manifest.json"), manifest)
    print(json.dumps({"manifest": os.path.join(workspace, "reports", "run_manifest.json"),
                      "baseline_byte_identical": baseline_identical,
                      "generation_byte_identical": generation_byte_identical}, indent=2))
    return 0


def _reports(workspace: str, label: str):
    data = load_json_file(os.path.join(workspace, "reports", f"{label}_path_report.json"))
    return [_DictTrace(item) for item in data]


class _DictTrace:
    def __init__(self, data: dict[str, Any]):
        self.__dict__.update(data)
        self.row = data["row"]
        self.col = data["col"]
        self.termination = data["termination"]
        self.internal_length = data["internal_length"]
        self.tir_count = data["tir_count"]
        self.events = data["events"]
        self.isolated = data["isolated"]
        self.error_code = data["error_code"]
        self.color = np.asarray(data["rgba"], dtype=np.float32)
        self.interface_crossings = data["interface_crossings"]


def _compact_injection_results(results: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for label, summary in results.items():
        if summary.get("run_level_error"):
            compact[label] = {"run_level_error": True, "code": summary["code"]}
            continue
        compact[label] = {
            "isolated_count": summary["isolated_count"],
            "first_affected_pixel": summary["first_affected_pixel"],
            "minimal_counterexample_code": (summary["minimal_counterexample"] or {}).get("code"),
            "counterexample_minimal": (summary["counterexample_minimality"] or {}).get("minimal"),
            "termination_counts": summary["termination_counts"],
        }
    return compact


def cmd_replay(args) -> int:
    from .diagnostics import replay_pixel

    scene = load_scene(args.scene)
    injection = injection_from_dict(load_json_file(args.injection)) if args.injection else None
    result = replay_pixel(scene, injection, args.row, args.col)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refrac", description="透明介质折射合成与异常注入验证")
    parser.add_argument("--workspace", default=os.getcwd())
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("clean").set_defaults(func=cmd_clean)

    gen = sub.add_parser("generate")
    gen.add_argument("--seed", type=int, default=20261119)
    gen.add_argument("--width", type=int, default=120)
    gen.add_argument("--height", type=int, default=90)
    gen.add_argument("--sample-count", type=int, default=8)
    gen.add_argument("--max-depth", type=int, default=6)
    gen.set_defaults(func=cmd_generate)

    render = sub.add_parser("render")
    render.add_argument("--scene", required=True)
    render.add_argument("--injection")
    render.add_argument("--label", default="custom")
    render.set_defaults(func=cmd_render)

    suite = sub.add_parser("suite")
    suite.add_argument("--seed", type=int, default=20261119)
    suite.add_argument("--width", type=int, default=120)
    suite.add_argument("--height", type=int, default=90)
    suite.set_defaults(func=cmd_suite)

    replay = sub.add_parser("replay")
    replay.add_argument("--scene", required=True)
    replay.add_argument("--injection")
    replay.add_argument("--row", type=int, required=True)
    replay.add_argument("--col", type=int, required=True)
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
