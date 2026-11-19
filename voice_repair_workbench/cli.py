"""一键可重复演示命令行。

子命令：
  generate  生成确定性合成素材（含本地有损样本）
  demo      完整验收演示：跨格式/跨版本一致性、迁移幂等、
            失效闭包、崩溃恢复、错误码
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import audio_io, config, lossy_codec, metrics as metrics_mod
from .engine import RepairEngine
from .errors import ALL_ERROR_CODES, RepairError
from .persistence import Workspace
from .project import (
    default_v2_project,
    legacy_v1_project,
    load_and_normalize,
    migrate_v1_to_v2,
)
from .synthesis import file_sha256, generate_material


def _print(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)


def generate(output_dir: str) -> dict:
    _print(f"[1] 生成确定性合成素材 -> {output_dir}")
    paths = generate_material(output_dir)
    # 由单声道 16bit 素材经本地离线有损编码器转出 .lrx
    parsed = audio_io.parse_audio(paths["mono16"])
    lossy_path = os.path.join(output_dir, "speech_mono_lossy.lrx")
    lossy_codec.encode_file(parsed.mono_analysis, parsed.sample_rate, lossy_path)
    paths["lossy"] = lossy_path
    for name, path in paths.items():
        print(f"  {name:10s} {os.path.basename(path):32s} sha256={file_sha256(path)[:16]}...")
    return paths


def _load_segments(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["segments"]


def _pcm16_of_ws(ws_root: str) -> np.ndarray:
    import soundfile as sf

    wav = os.path.join(ws_root, "repaired.wav")
    data, _ = sf.read(wav, dtype="int16")
    return data


def demo(work_root: str) -> dict:
    # 一键可重复：每次演示从干净工作区开始，避免上一轮残留快照被判定陈旧。
    import shutil

    if os.path.exists(work_root):
        shutil.rmtree(work_root)
    material_dir = os.path.join(work_root, "materials")
    paths = generate(material_dir)
    segments = _load_segments(paths["segments"])
    results: dict = {"steps": []}

    # ---- 步骤 2：解析三套素材并打印格式识别结果 ----
    _print("[2] 多格式解析与归一（采样率/位深/声道/有损标记）")
    metas = {}
    for label, key, expect_ch in (
        ("mono16", "mono16", 1),
        ("stereo24", "stereo24", 2),
        ("lossy", "lossy", 1),
    ):
        parsed = audio_io.parse_audio(paths[key], expected_channels=expect_ch)
        metas[label] = parsed.to_meta()
        print(
            f"  {label:9s} sr={parsed.sample_rate} bit={parsed.bit_depth} "
            f"ch={parsed.channels} lossy={parsed.lossy}({parsed.lossy_format}) "
            f"content={parsed.content_hash()[:16]}"
        )
    results["steps"].append({"step": "parse", "metas": metas})

    # ---- 步骤 3：v1 -> v2 迁移与幂等 ----
    _print("[3] schema v1 -> v2 迁移（固定迁移表 + 幂等）")
    v1 = legacy_v1_project("legacy_demo")
    migrated_once = migrate_v1_to_v2(v1)
    migrated_twice = migrate_v1_to_v2(migrated_once)
    idempotent = migrated_once == migrated_twice
    print(f"  迁移幂等: {idempotent}")
    print(f"  迁移后 gate={migrated_once['gate']}")
    print(f"  迁移后 normalize={migrated_once['normalize']}")
    print(f"  迁移后 limit={migrated_once['limit']}")
    results["steps"].append(
        {"step": "migration", "idempotent": idempotent, "migrated": migrated_once}
    )

    # ---- 步骤 4：同一 v1 工程在三种素材上的修复结论一致性 ----
    _print("[4] 跨格式 + 跨版本修复（v1 迁移工程，三套素材）")
    run_roots = {}
    pcm_outputs = {}
    metric_reports = {}
    for label, key, expect_ch in (
        ("mono16", "mono16", 1),
        ("stereo24", "stereo24", 2),
        ("lossy", "lossy", 1),
    ):
        ws_root = os.path.join(work_root, f"ws_{label}_v1")
        run_roots[label] = ws_root
        engine = RepairEngine(Workspace(ws_root))
        snap = engine.execute(
            paths[key], v1, segments, expected_channels=expect_ch
        )
        pcm_outputs[label] = _pcm16_of_ws(ws_root)
        metric_reports[label] = snap["metrics"]
        print(f"  {label:9s} pcm16={snap['output']['pcm16_sha256'][:16]} closure={snap['recomputed_closure']}")

    cmp_ms = metrics_mod.compare_conclusions(
        metric_reports["mono16"], metric_reports["stereo24"]
    )
    cmp_ml = metrics_mod.compare_conclusions(
        metric_reports["mono16"], metric_reports["lossy"]
    )
    print(f"  指标结论 mono16 vs stereo24 一致(容差内): {cmp_ms['consistent']}")
    print(f"  指标结论 mono16 vs lossy    一致(容差内): {cmp_ml['consistent']}")
    results["steps"].append(
        {
            "step": "cross_format",
            "mono_vs_stereo": cmp_ms,
            "mono_vs_lossy": cmp_ml,
        }
    )

    # ---- 步骤 5：同参数重复运行的确定性 + 第二次仅命中缓存 ----
    _print("[5] 确定性重跑（第二次应全部复用缓存，闭包为空）")
    engine = RepairEngine(Workspace(run_roots["mono16"]))
    snap2 = engine.execute(paths["mono16"], v1, segments, expected_channels=1)
    pcm2 = _pcm16_of_ws(run_roots["mono16"])
    deterministic = bool(np.array_equal(pcm_outputs["mono16"], pcm2))
    actions = [(t["stage"], t["action"]) for t in snap2["execution_trace"]]
    print(f"  两次 int16 PCM 逐样本一致: {deterministic}")
    print(f"  第二次阶段动作: {actions}")
    print(f"  实际重算集合 = 闭包清单 {snap2['recomputed_closure']} (空=全复用)")
    results["steps"].append(
        {"step": "determinism", "pcm_equal": deterministic, "trace": snap2["execution_trace"]}
    )

    # ---- 步骤 6：参数变化的失效闭包（只重算下游）----
    _print("[6] 失效闭包演示（仅修改 gate.threshold_db）")
    v2 = load_and_normalize(default_v2_project("param_demo"))
    ws6 = Workspace(os.path.join(work_root, "ws_closure"))
    eng6 = RepairEngine(ws6)
    snap6a = eng6.execute(paths["mono16"], v2, segments, expected_channels=1)
    v2_b = json.loads(json.dumps(v2))
    v2_b["gate"]["threshold_db"] = -30.0
    snap6b = eng6.execute(paths["mono16"], v2_b, segments, expected_channels=1)
    closure6 = snap6b["recomputed_closure"]
    expected6 = ["gate", "normalize", "limit"]
    print(f"  期望闭包: {expected6}")
    print(f"  实际闭包: {closure6}")
    print(f"  闭包外(denoise)动作: "
          f"{[t for t in snap6b['execution_trace'] if t['stage'] == 'denoise'][0]['action']}")
    results["steps"].append(
        {
            "step": "invalidation_closure",
            "expected": expected6,
            "actual": closure6,
            "exact_match": closure6 == expected6,
            "trace": snap6b["execution_trace"],
        }
    )

    # ---- 步骤 7：通道顺序变化只影响 normalize/limit ----
    _print("[7] 通道处理顺序影响面（stereo24，denoise/gate 哈希保持不变）")
    # 构造非对称双声道素材（右声道减半），通道顺序差异才可观测
    import soundfile as sf
    asym_path = os.path.join(material_dir, "speech_stereo_asym_48k_24bit.wav")
    stereo_data, sr7 = sf.read(paths["stereo24"], dtype="float64", always_2d=True)
    stereo_data[:, 1] *= 0.5
    sf.write(asym_path, stereo_data, sr7, subtype="PCM_24")
    order_a = json.loads(json.dumps(v2))
    order_a["channel_order"] = config.ORDER_MONO_FIRST
    order_b = json.loads(json.dumps(v2))
    order_b["channel_order"] = config.ORDER_PER_CHANNEL
    ws7 = Workspace(os.path.join(work_root, "ws_order"))
    eng7 = RepairEngine(ws7)
    snap7a = eng7.execute(asym_path, order_a, segments, expected_channels=2)
    snap7b = eng7.execute(asym_path, order_b, segments, expected_channels=2)
    hash_a = {e["stage"]: e["output_hash"] for e in snap7a["stages"]}
    hash_b = {e["stage"]: e["output_hash"] for e in snap7b["stages"]}
    prefix_same = hash_a["denoise"] == hash_b["denoise"] and hash_a["gate"] == hash_b["gate"]
    downstream_changed = (
        hash_a["normalize"] != hash_b["normalize"] or hash_a["limit"] != hash_b["limit"]
    )
    print(f"  denoise/gate 中间结果哈希不变: {prefix_same}")
    print(f"  normalize/limit 发生变化: {downstream_changed}")
    results["steps"].append(
        {
            "step": "channel_order",
            "prefix_hashes_equal": prefix_same,
            "downstream_changed": downstream_changed,
            "mono_first": hash_a,
            "per_channel": hash_b,
        }
    )

    # ---- 步骤 8：强制终止恢复（未提交 .tmp 作孤儿回收，恢复后逐项一致）----
    _print("[8] 强制终止恢复（注入未提交缓存 + 移除关闭标记）")
    crash_root = os.path.join(work_root, "ws_crash")
    ws8 = Workspace(crash_root)
    eng8 = RepairEngine(ws8)
    snap8 = eng8.execute(paths["mono16"], v2, segments, expected_channels=1)
    pcm_before = _pcm16_of_ws(crash_root)
    # 模拟强制终止：留下 .tmp 未提交文件、删掉关闭标记（本来也没写）
    with open(os.path.join(ws8.cache_dir, "denoise_FAKE.tmp"), "wb") as fh:
        fh.write(b"uncommitted")
    assert not os.path.exists(ws8.closed_flag)
    restored = eng8.restore(v2)
    audit = restored["recovery_audit"]
    pcm_after = _pcm16_of_ws(crash_root)
    recover_equal = bool(np.array_equal(pcm_before, pcm_after))
    metric_equal = restored["snapshot"]["metrics"] == snap8["metrics"]
    print(f"  回收未提交 .tmp: {audit['uncommitted_tmp_reclaimed']}")
    print(f"  未伪造退出报告(无 exit_report): {not audit['exit_report_present']}")
    print(f"  恢复后 PCM 与中断前最后提交逐项一致: {recover_equal}")
    print(f"  恢复后逐段指标逐项一致: {metric_equal}")
    # 受控退出对比
    report = eng8.graceful_shutdown(snap8)
    print(f"  受控退出报告 exit_kind={report['exit_kind']} 并写入 CLOSED.flag")
    results["steps"].append(
        {
            "step": "crash_recovery",
            "audit": audit,
            "pcm_equal": recover_equal,
            "metrics_equal": metric_equal,
            "graceful_report": report,
        }
    )

    # ---- 步骤 9：业务错误码链路 ----
    _print("[9] 业务错误码覆盖演示")
    error_demos = _error_code_demos(paths)
    for item in error_demos:
        print(f"  {item['code']:22s} field={item['field']} -> {item['message']}")
    results["steps"].append({"step": "error_codes", "items": error_demos})

    # ---- 逐段指标报告落盘 ----
    report_path = os.path.join(work_root, "segment_metrics_report.json")
    _dump(
        report_path,
        {
            "tolerance": metrics_mod.TOLERANCE,
            "reports": {label: rows for label, rows in metric_reports.items()},
            "cross_format_conclusions": {
                "mono16_vs_stereo24": cmp_ms,
                "mono16_vs_lossy": cmp_ml,
            },
        },
    )
    results["segment_metrics_report"] = report_path
    _dump(os.path.join(work_root, "demo_results.json"), results)
    print(f"\n逐段指标报告: {report_path}")
    print("演示完成。")
    return results


def _error_code_demos(paths: dict) -> list[dict]:
    items: list[dict] = []

    def capture(fn):
        try:
            fn()
        except RepairError as exc:
            items.append(exc.to_dict())

    # E_FORMAT_UNSUPPORTED：非法扩展名
    capture(lambda: audio_io.parse_audio(paths["mono16"] + ".xyz"))

    # E_SCHEMA_VERSION：未知 schema 版本
    capture(lambda: load_and_normalize({"schema_version": 99}))

    # E_SCHEMA_VERSION：旧版存在不可迁移字段
    capture(
        lambda: migrate_v1_to_v2(
            {
                "schema_version": config.SCHEMA_V1,
                "denoise": {
                    "alpha": 2.0,
                    "floor": 0.02,
                    "noise_seed": 7,
                    "noise_seconds": 0.5,
                    "experimental_magic": 1,
                },
            }
        )
    )

    # E_FIELD_MISSING：v2 工程缺字段
    bad_v2 = default_v2_project()
    del bad_v2["gate"]
    capture(lambda: load_and_normalize(bad_v2))

    # E_CHANNEL_MISMATCH：把双声道文件当单声道声明
    capture(
        lambda: audio_io.parse_audio(paths["stereo24"], expected_channels=1)
    )

    # E_PARAM_RANGE：参数越界
    bad_range = default_v2_project()
    bad_range["gate"]["threshold_db"] = 6.0
    capture(lambda: load_and_normalize(bad_range))

    # E_CACHE_HASH：篡改缓存文件后读取
    def tamper_cache():
        from .hashing import sha256_json
        from .persistence import Workspace as WS
        from .engine import RepairEngine as RE

        tmp_ws = WS(os.path.join(os.path.dirname(paths["mono16"]), "..", "ws_tamper"))
        eng = RE(tmp_ws)
        proj = default_v2_project("tamper")
        seg = _load_segments(paths["segments"])
        snap = eng.execute(paths["mono16"], proj, seg, expected_channels=1)
        target = os.path.join(tmp_ws.cache_dir, snap["stages"][0]["key"] + ".npy")
        data = np.load(target)
        data = data.copy()
        data[:] = 0.0
        np.save(target, data)
        tmp_ws.verify_stage_hashes(snap)

    capture(tamper_cache)

    # E_SNAPSHOT_STALE：用不同工程指纹恢复
    def stale_snapshot():
        from .persistence import Workspace as WS
        from .engine import RepairEngine as RE

        tmp_ws = WS(os.path.join(os.path.dirname(paths["mono16"]), "..", "ws_stale"))
        eng = RE(tmp_ws)
        proj_a = default_v2_project("stale_a")
        seg = _load_segments(paths["segments"])
        eng.execute(paths["mono16"], proj_a, seg, expected_channels=1)
        proj_b = default_v2_project("stale_b")
        proj_b["gate"]["threshold_db"] = -60.0
        eng.restore(proj_b)

    capture(stale_snapshot)

    seen = {item["code"] for item in items}
    assert seen == set(ALL_ERROR_CODES), f"missing codes: {set(ALL_ERROR_CODES) - seen}"
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="语音录音修复工作台（确定性离线演示）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="生成合成素材与本地有损样本")
    gen.add_argument("--out", default="demo_workspace/materials")
    dm = sub.add_parser("demo", help="运行完整验收演示")
    dm.add_argument("--work", default="demo_workspace")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "generate":
            generate(args.out)
        else:
            demo(args.work)
    except RepairError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
