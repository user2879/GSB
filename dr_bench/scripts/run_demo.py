"""Cross-platform demo entry point used by the Windows wrapper scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_bench.demo import run_demo
from dr_bench.canonical import canonical_text


if __name__ == "__main__":
    workspace = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    summary = run_demo(workspace)
    print("=" * 72)
    print("DISASTER RECOVERY DECISION BENCH - DETERMINISTIC DEMO SUMMARY")
    print("=" * 72)
    print(f"scene        : {summary['scene']['scene_id']} v{summary['scene']['scene_version']}")
    print(f"scale        : {summary['scene']['scale']}")
    print(f"planner pick : {summary['cold_start']['planner']['selected_path']}")
    print(f"cold rounds  : {summary['cold_start']['ledger']['committed_rounds']}")
    print("fault codes  :")
    for kind, result in summary["fault_injection"].items():
        code = result["error"]["code"] if result.get("error") else "OK"
        print(f"  - {kind:<28} {code}")
    print(f"crash        : pending={summary['crash_recovery']['crash']['pending_round']}")
    print(f"replay verify: {summary['replay']['verified_rounds']}")
    print(f"replay rerun : {summary['replay']['rerun_rounds']}")
    print(f"summary file : {workspace / 'data' / 'demo_summary.json'}")
    print(f"log file     : {workspace / 'data' / 'structured.log.jsonl'}")
    print(canonical_text(summary["planner"]) if False else "")
    raise SystemExit(0)
