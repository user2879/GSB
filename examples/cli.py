"""命令行入口：自检、生成样本、全量运行与增量编辑。"""

import argparse
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


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="说话人音色分析组件 CLI")
    parser.add_argument("command", choices=("generate", "run", "edit"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--commands-json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-injections", action="store_true")
    args = parser.parse_args()

    if args.command == "generate":
        layout = generate_sample_workspace(args.workspace)
        print(json.dumps(layout, ensure_ascii=False, indent=2, sort_keys=True))
        return
    config = Config.load(os.path.join(args.workspace, "pipeline_config.json"))
    injection_path = os.path.join(args.workspace, "injections.json")
    injections = InjectionBook.load(injection_path) if args.with_injections else InjectionBook([])
    store = SnapshotStore(args.snapshot) if args.snapshot else None
    pipeline = TimbrePipeline(config, injections, store, deterministic=True)
    truth_path = os.path.join(args.workspace, "truth_labels.json")
    truth = json.load(open(truth_path, encoding="utf-8"))
    if args.command == "run":
        report = pipeline.run_full(scan_wav_dir(os.path.join(args.workspace, "wav")), truth)
    else:
        pipeline.run_full(scan_wav_dir(os.path.join(args.workspace, "wav")), truth)
        commands = json.load(open(args.commands_json, encoding="utf-8"))
        reports = pipeline.apply_edit_commands(commands, truth)
        report = reports[-1]
    write_json(args.out, report)
    print("REPORT_WRITTEN", args.out)


if __name__ == "__main__":
    main()
