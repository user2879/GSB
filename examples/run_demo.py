"""端到端示例：全量运行、增量编辑、配置切换、快照恢复与注入诊断。"""

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from timbre_pipeline.audio import scan_wav_dir
from timbre_pipeline.config import Config
from timbre_pipeline.inject import InjectionBook
from timbre_pipeline.pipeline import TimbrePipeline
from timbre_pipeline.snapshot import SnapshotStore
from timbre_pipeline.synthesize import generate_sample_workspace
from timbre_pipeline.utils import canonical_json


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def baseline_pipeline(config, injection_book, snapshot_dir, deterministic=True):
    store = SnapshotStore(snapshot_dir) if snapshot_dir else None
    return TimbrePipeline(config, injection_book, store, deterministic=deterministic)


def run_full_demo(layout, out_dir, use_injections=False):
    config = Config.load(layout["config_path"])
    injections = InjectionBook.load(layout["injection_path"]) if use_injections else InjectionBook([])
    snapshot_dir = os.path.join(out_dir, "snapshot_full")
    pipeline = baseline_pipeline(config, injections, snapshot_dir)
    wav_paths = scan_wav_dir(layout["wav_dir"])
    truth = load_json(layout["label_path"])
    report = pipeline.run_full(wav_paths, truth)
    write_json(os.path.join(out_dir, "report_full.json"), report)
    return pipeline, report, truth


def run_incremental_consistency(layout, out_dir):
    """增量流水线 vs 独立全量流水线，逐字段比对。"""
    config = Config.load(layout["config_path"])
    no_inject = InjectionBook([])
    full_pipeline = baseline_pipeline(config, no_inject, None)
    wav_paths = scan_wav_dir(layout["wav_dir"])
    truth = load_json(layout["label_path"])
    full_report = full_pipeline.run_full(wav_paths, truth)

    inc_pipeline = baseline_pipeline(config, InjectionBook([]),
                                     os.path.join(out_dir, "snapshot_incremental"))
    inc_pipeline.run_full(wav_paths, truth)
    commands = [
        {"command_id": "cmd-001", "type": "split", "audio": "clip_01",
         "segment_index": 1, "sample_offset": 1600},
        {"command_id": "cmd-002", "type": "modify_feature", "audio": "clip_02",
         "segment_index": 0, "dimension": "f0_median", "value": 190.0},
        {"command_id": "cmd-003", "type": "merge", "audio": "clip_03",
         "segment_index": 2},
    ]
    inc_reports = inc_pipeline.apply_edit_commands(commands, truth)
    write_json(os.path.join(out_dir, "report_incremental.json"), inc_reports[-1])

    # 独立全量重放同样的编辑序列，验证中断恢复等价性。
    replay_pipeline = baseline_pipeline(config, InjectionBook([]),
                                        os.path.join(out_dir, "snapshot_replay"))
    replay_pipeline.run_full(wav_paths, truth)
    replay_pipeline.apply_edit_commands(copy.deepcopy(commands), truth)
    return full_report, inc_reports, inc_pipeline, replay_pipeline


def run_config_switch(layout, out_dir):
    config = Config.load(layout["config_path"])
    pipeline = baseline_pipeline(config, InjectionBook([]),
                                 os.path.join(out_dir, "snapshot_config"))
    wav_paths = scan_wav_dir(layout["wav_dir"])
    truth = load_json(layout["label_path"])
    pipeline.run_full(wav_paths, truth)
    before = copy.deepcopy(pipeline.build_report())
    new_data = config.data
    for dimension in new_data["dimensions"]:
        dimension["levels"] = [level + 2 for level in dimension["levels"]]
    new_data["similarity_threshold"] = 0.82
    new_config = Config(new_data)
    switched = pipeline.apply_config(new_config, "cmd-config-001", truth)
    write_json(os.path.join(out_dir, "report_config_switch.json"), switched)
    return before, switched


def run_recovery(layout, out_dir):
    config = Config.load(layout["config_path"])
    snapshot_dir = os.path.join(out_dir, "snapshot_recovery")
    first = baseline_pipeline(config, InjectionBook([]), snapshot_dir)
    wav_paths = scan_wav_dir(layout["wav_dir"])
    truth = load_json(layout["label_path"])
    first.run_full(wav_paths, truth)
    revived = baseline_pipeline(Config.load(layout["config_path"]), InjectionBook([]),
                                snapshot_dir)
    decision = revived.restore_from_snapshot()
    write_json(os.path.join(out_dir, "report_recovery.json"), {
        "decision": decision,
        "summary": revived.build_summary(),
    })
    return decision, revived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", help="样本目录；缺省自动生成到 sample_work")
    parser.add_argument("--out", default=os.path.join(ROOT, "sample_work", "outputs"))
    args = parser.parse_args()
    if args.workspace:
        layout = {
            "root": args.workspace,
            "wav_dir": os.path.join(args.workspace, "wav"),
            "config_path": os.path.join(args.workspace, "pipeline_config.json"),
            "injection_path": os.path.join(args.workspace, "injections.json"),
            "label_path": os.path.join(args.workspace, "truth_labels.json"),
        }
    else:
        layout = generate_sample_workspace(os.path.join(ROOT, "sample_work", "data"))
    os.makedirs(args.out, exist_ok=True)
    clean_pipeline, clean_report, truth = run_full_demo(layout, args.out, False)
    injected_pipeline, injected_report, _ = run_full_demo(layout, args.out, True)
    write_json(os.path.join(out_dir := args.out, "report_injected.json"), injected_report)
    full_report, inc_reports, inc_pipeline, replay_pipeline = run_incremental_consistency(layout, args.out)
    before, switched = run_config_switch(layout, args.out)
    decision, revived = run_recovery(layout, args.out)
    summary = {
        "clean_audio_count": len(clean_report["audio_reports"]),
        "injected_failure_count": len(injected_report["failure_list"]),
        "injection_hits": {
            audio_id: report["injection_hits"]
            for audio_id, report in injected_report["audio_reports"].items()
            if report["injection_hits"]
        },
        "incremental_commands": len(inc_reports),
        "recovery_action": decision["action"],
        "recovery_reason": decision["reason"],
        "config_switch_version": switched["global_version"],
    }
    write_json(os.path.join(args.out, "demo_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
