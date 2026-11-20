#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamparse.app import IncrementalApp
from streamparse.loader import load_config, load_extra_scores
from streamparse.reporting import write_report_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic interactive edit sequence")
    parser.add_argument("--sample-dir", default="sample_data")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()

    demo_config = replace(
        load_config(Path(args.sample_dir) / "decode_config.json"),
        report_path="artifacts/diagnostic_report_demo.json",
        snapshot_root="artifacts/snapshots_demo",
    )
    app = IncrementalApp(
        args.sample_dir,
        workspace_root=args.workspace_root,
        config=demo_config,
    )
    new_rows = load_extra_scores(
        Path(args.sample_dir) / "extras" / "new_session_scores.jsonl",
        app.dataset.vocab,
    )
    commands = [
        {"cmd": "set_strategy", "session_id": "S01", "strategy": "greedy"},
        {"cmd": "set_param", "session_id": "S01", "key": "lookahead_frames", "value": 2},
        {"cmd": "set_strategy", "session_id": "S01", "strategy": "lookahead"},
        {"cmd": "set_param", "session_id": "S01", "key": "overlap_frames", "value": 1},
        {
            "cmd": "append_shard",
            "session_id": "S01",
            "shard_seq": 3,
            "start_frame": 8,
            "end_frame": 11,
            "score_file": "extras/append_scores.jsonl",
        },
        {
            "cmd": "replace_shard",
            "session_id": "S01",
            "shard_seq": 1,
            "start_frame": 3,
            "end_frame": 6,
            "score_file": "extras/replace_scores.jsonl",
        },
        {
            "cmd": "add_session",
            "session_id": "S05",
            "shards": [
                {
                    "shard_seq": 0,
                    "start_frame": 0,
                    "end_frame": 4,
                    "scores": new_rows[:5],
                }
            ],
        },
        {"cmd": "delete_session", "session_id": "S03"},
        {"cmd": "wait"},
        {"cmd": "report"},
    ]
    try:
        for command in commands:
            response = app.execute(command)
            if command["cmd"] == "wait" and response.get("status") != "ok":
                response = app.execute({"cmd": "wait"})
            print(json.dumps(response, ensure_ascii=False, sort_keys=True)[:600])
        app.wait()
        write_report_atomic(app.report_path, app.latest_report or {})
        print(json.dumps({"report": str(app.report_path), "version": app.latest_version}))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
