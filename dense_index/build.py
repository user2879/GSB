from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .commands import CommandState, load_all_receipts, load_queries
from .engine import IndexEngine


def run_full_build(root: str | Path, budget: Optional[int] = None, write: bool = True) -> Dict[str, Any]:
    root = Path(root)
    state = CommandState.from_sample_dir(root)
    receipts = load_all_receipts(root)
    queries = load_queries(root)
    engine = IndexEngine(state.config, receipts=receipts, queries=queries, receipt_preference=state.receipt_version)
    report = engine.run_batch(state.blocks, budget=budget or 10**9)

    if write:
        output = root / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        serializable_view = report["view"]
        (output / "full_build_report.json").write_text(
            json.dumps({"status": report["status"], "view": serializable_view}, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (output / "unified_ledger.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for entry in report["ledger"]:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        (output / "recall_report.json").write_text(
            json.dumps(report["recall"], ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (output / "invalidation_report_full_build.json").write_text(
            json.dumps(report["invalidation"], ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return report
