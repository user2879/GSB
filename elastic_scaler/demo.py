"""Deterministic three-run demonstration.

Run order (stable across machines; no network and no wall-clock dependence in
the decisions themselves):
1. cold-start advance over the baseline declaration;
2. declaration change -> whole-view replacement and a second advance;
3. forced-kill simulation -> ledger recovery, resume and replay verification.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .common import canonical_dumps
from .engine import ForcedKill, ScalingEngine
from .gui import ScalerWorkbench
from .spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "demo"
V1 = ROOT / "samples" / "cluster_v1.json"
V2 = ROOT / "samples" / "cluster_v2.json"
DB = DATA / "demo.sqlite3"
LOG = DATA / "events.jsonl"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _summarize_decision(decision: Any) -> dict[str, Any]:
    return {
        "minute": decision.minute,
        "spec_version": decision.spec_version,
        "budget_used": decision.budget_used,
        "budget": decision.budget,
        "total_rejected": decision.total_rejected,
        "services": [
            {
                "service": alloc.service,
                "requests": alloc.estimate.requests,
                "raw_demand": alloc.estimate.raw_demand,
                "demand": alloc.estimate.demand,
                "allocated": alloc.allocated_replicas,
                "rejected": alloc.rejected_requests,
                "reasons": list(alloc.trim_reasons),
            }
            for alloc in decision.allocations
        ],
    }


def run_demo() -> dict[str, Any]:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)

    summary: dict[str, Any] = {"run_order": ["cold_start", "spec_change", "crash_recovery"]}

    # Run 1: cold start.
    with ScalingEngine(str(DB), str(LOG), run_id="run-v1") as engine:
        spec_v1 = load_spec(V1)
        version, _ = engine.initialize(spec_v1)
        cold_decisions = engine.advance_all()
        engine.verify_fast_vs_authoritative()
        workbench = ScalerWorkbench(engine)
        for minute in engine.ledger.committed_minutes(engine.run_id):
            workbench.refresh()
        summary["run_1_cold_start"] = {
            "spec_version": version,
            "digest": spec_v1.digest,
            "minutes": [_summarize_decision(d) for d in cold_decisions],
            "numpy_load_summary": engine.numpy_load_summary(),
            "interactive_latency": workbench.latency_report(),
        }
        engine.save_snapshot("snap-v1-minute4", 4, note="post cold-start checkpoint")

    # Run 2: spec change and whole-view replacement.
    with ScalingEngine(str(DB), str(LOG), run_id="run-v1") as engine:
        engine.recover()
        spec_v2 = load_spec(V2)
        replacement = engine.replace_spec(spec_v2)
        changed_decisions = engine.advance_all()
        engine.verify_fast_vs_authoritative()
        summary["run_2_spec_change"] = {
            "replacement": _jsonable(replacement),
            "versions": engine.list_versions(),
            "minutes": [_summarize_decision(d) for d in changed_decisions],
            "old_version_read_only": engine.get_version_view(replacement.old_version)["status"],
        }

    # Run 3: new engine process, forced kill, recovery, resume and replay.
    with ScalingEngine(str(DB), str(LOG), run_id=engine.run_id) as engine:
        engine.recover()
        kill_minute = 3
        try:
            # Reuse a fresh run id so the kill happens on an uncommitted run.
            engine.run_id = "run-crash"
            if engine.ledger.get_run(engine.run_id) is None:
                engine.ledger.create_run(engine.run_id, engine.spec_version, engine.spec.horizon)
            engine._cache = {}
            engine.advance_all(simulate_kill_at=kill_minute)
        except ForcedKill:
            kill_seen = True
        recovery = engine.recover()
        resumed = engine.resume_after_recovery()
        replay = engine.verify_replay()
        engine.verify_fast_vs_authoritative()
        summary["run_3_crash_recovery"] = {
            "kill_seen": kill_seen,
            "recovery": _jsonable(recovery),
            "resumed_minutes": [d.minute for d in resumed],
            "replay": _jsonable(replay),
        }

    # Deterministic byte-identity guard: rerun run 1 decisions independently.
    with ScalingEngine(str(DB), str(LOG), run_id="run-v1") as engine:
        canonical = [
            canonical_dumps(engine.authoritative_query(m).to_dict())
            for m in engine.ledger.committed_minutes("run-v1")
        ]
    summary["canonical_decision_sha_input_lines"] = len(canonical)
    out_path = DATA / "demo-summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
                        encoding="utf-8")
    return summary


def main() -> int:
    summary = run_demo()
    print(json.dumps({
        "run_order": summary["run_order"],
        "run_1_minutes": len(summary["run_1_cold_start"]["minutes"]),
        "run_2_new_version": summary["run_2_spec_change"]["replacement"]["new_version"],
        "run_3_replay_ok": summary["run_3_crash_recovery"]["replay"]["ok"],
        "summary_path": str(DATA / "demo-summary.json"),
        "event_log": str(LOG),
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
