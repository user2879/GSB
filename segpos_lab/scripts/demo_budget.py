"""首响预算语义演示：预算内返回部分结果与 E_INTERACT_BUDGET_PENDING，
待办句子保留且不进入快照，多次排空后得到与无预算运行一致的最终结果。"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segpos.errors import E_INTERACT_BUDGET_PENDING  # noqa: E402
from segpos.runner import build_engine, prepare_bundle  # noqa: E402
from scripts.verify_equivalence import comparable  # noqa: E402


def main(sample_dir: str) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="segpos-budget-"))
    try:
        tight = prepare_bundle(
            sample_dir,
            {
                "first_response_budget_ms": 4.0,
                "simulated_task_delay_ms": 3.0,
                "snapshot_commit_interval_commands": 100,
            },
        )[0]
        engine = build_engine(tight, tmp / "tight")
        first = engine.run_due(set(engine.states))
        print("first response:", json.dumps({k: first[k] for k in
              ("code", "completed_sentences", "total_sentences", "progress", "pending_sentences")}, ensure_ascii=False))
        assert first["code"] == E_INTERACT_BUDGET_PENDING
        rounds = 1
        while engine.pending:
            response = engine.run_due(set())
            rounds += 1
            assert response["code"] == E_INTERACT_BUDGET_PENDING or not engine.pending
        print(f"drained after {rounds} rounds; pending={engine.pending}")
        engine.commit_snapshot(False)
        manifest = engine.snapshot.load_manifest()
        assert manifest["pending_sentences"] == []
        budget_report = engine.build_report(0)

        relaxed = prepare_bundle(sample_dir, {"simulated_task_delay_ms": 0.0})[0]
        full_engine = build_engine(relaxed, tmp / "full")
        full_engine.run_due(set(full_engine.states))
        full_report = full_engine.build_report(0)
        assert comparable(budget_report) == comparable(full_report)
        print("final result identical to unbounded full run: OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "samples/base"))
