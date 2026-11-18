"""End-to-end deterministic demonstration for tier_mover.

Sessions, in fixed order:
  1. initial hot/cold placement and first migration cycle;
  2. capacity tightening (new versions) -> whole-snapshot replacement + remigrate;
  3. forced termination mid-transfer -> recovery -> deterministic replay.

Then the three fault injections are exercised and reported by business code:
E_CAPACITY, E_INTEGRITY, E_VERSION.  Everything is written under artifacts/demo
and a stable ordered summary is printed.  No network or external storage is used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tier_mover.engine import Engine  # noqa: E402
from tier_mover.faults import Injection  # noqa: E402
from tier_mover.ledger import Ledger  # noqa: E402
from tier_mover.logging_util import JsonlLogger  # noqa: E402
from tier_mover.manifest import load_manifest  # noqa: E402
from tier_mover.replay import replay  # noqa: E402
from tier_mover.samplegen import build_sample  # noqa: E402
from tier_mover.storage import staging_dir  # noqa: E402

ART = ROOT / "artifacts" / "demo"


def _reset(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine(workdir: Path, injection: Injection | None = None) -> Engine:
    manifest = load_manifest(workdir / "manifest.json")
    logger = JsonlLogger(workdir / "logs" / "demo.jsonl")
    return Engine(workdir, manifest, logger, injection)


def _summary_line(title: str, conclusions: dict) -> dict:
    return {
        "stage": title,
        "tiers": [
            {
                "tier_id": t["tier_id"],
                "used_bytes": t["used_bytes"],
                "free_bytes": t["free_bytes"],
                "object_count": t["object_count"],
                "storage_cost": t["storage_cost"],
            }
            for t in conclusions["tiers"]
        ],
        "migration_records": len(conclusions["migrations"]),
        "total_move_cost": conclusions["total_move_cost"],
    }


def main() -> int:
    _reset(ART)
    summary: list[dict] = []

    # ------------------------------------------------------- session 1
    s1 = ART / "session1"
    build_sample(s1, policy_version="policy-v1", tier_param_version="tier-v1",
                 hot_capacity=16384, cold_capacity=65536,
                 hot_watermark=12288, cold_reclaim=32768, batch_size=4)
    engine = _engine(s1)
    initial = engine.run_initial()
    engine.logger.close()
    summary.append(_summary_line("session1_initial_placement", initial))

    engine = _engine(s1)
    cycle1 = engine.run_cycle()
    engine.logger.close()
    summary.append(_summary_line("session1_migration_cycle", cycle1))
    replay1 = replay(str(s1), load_manifest(s1 / "manifest.json"))

    # ------------------------------------------------------- session 2
    s2 = ART / "session2"
    shutil.copytree(s1, s2)
    # Whole-snapshot replacement discards old tier ownership/state: clear the
    # physical tier directories so the fresh ledger rebuilds from staging only.
    shutil.rmtree(s2 / "tiers", ignore_errors=True)
    # New manifest: bumped versions + tightened hot capacity and watermark.
    build_sample(s2, policy_version="policy-v2", tier_param_version="tier-v2",
                 hot_capacity=12288, cold_capacity=65536,
                 hot_watermark=8192, cold_reclaim=40960, batch_size=4)
    ledger = Ledger.open(s2)
    replace_info = ledger.replace_snapshot(load_manifest(s2 / "manifest.json"))
    ledger.close()
    engine = _engine(s2)
    # Whole-state replacement starts a fresh ledger: re-place objects from the
    # staging inputs, then run the tightened-policy remigration cycle.
    engine.run_initial(allow_existing=True)
    engine.logger.close()
    engine = _engine(s2)
    cycle2 = engine.run_cycle()
    engine.logger.close()
    summary.append(_summary_line("session2_replaced_and_remigrated", cycle2))
    replay2 = replay(str(s2), load_manifest(s2 / "manifest.json"))

    # ------------------------------------------------------- session 3
    s3 = ART / "session3"
    shutil.copytree(s2, s3)
    # A new session version with a relaxed watermark: replace the snapshot as a
    # whole unit, clear physical tiers, and re-place from staging.  The relaxed
    # watermark guarantees a non-empty promotion cycle for forced termination.
    build_sample(s3, policy_version="policy-v2", tier_param_version="tier-v2",
                 hot_capacity=12288, cold_capacity=65536,
                 hot_watermark=8192, cold_reclaim=24576, batch_size=4)
    manifest_path = s3 / "manifest.json"
    shutil.rmtree(s3 / "tiers", ignore_errors=True)
    ledger3 = Ledger.open(s3)
    ledger3.replace_snapshot(load_manifest(manifest_path))
    ledger3.close()
    _engine(s3).run_initial(allow_existing=True)
    # Initial placement already exists inside the copied snapshot; run a fresh
    # cycle in a subprocess and hard-kill it mid transfer.
    proc = subprocess.run(
        [sys.executable, "-m", "tier_mover.cli", "cycle", str(s3),
         "--manifest", str(manifest_path), "--inject", "kill",
         "--kill-after-commit", "1"],
        cwd=ROOT, capture_output=True, text=True,
    )
    kill_rc = proc.returncode
    engine = _engine(s3)
    recovery = engine.recover()
    engine.logger.close()
    replay3 = replay(str(s3), load_manifest(manifest_path))

    # ------------------------------------------------------- determinism
    da = ART / "det_a"
    db = ART / "det_b"
    for folder in (da, db):
        build_sample(folder, hot_capacity=16384, cold_capacity=65536,
                     hot_watermark=12288, cold_reclaim=32768, batch_size=4)
        eng = _engine(folder)
        eng.run_initial()
        eng.run_cycle()
        eng.logger.close()
    determinism = {
        "initial_plan_equal": _sha(da / "initial_plan.json") == _sha(db / "initial_plan.json"),
        "cycle_plan_equal": _sha(da / "cycle_plan.json") == _sha(db / "cycle_plan.json"),
        "initial_conclusions_equal": _sha(da / "initial_conclusions.json")
        == _sha(db / "initial_conclusions.json"),
        "cycle_conclusions_equal": _sha(da / "cycle_conclusions.json")
        == _sha(db / "cycle_conclusions.json"),
    }

    # ------------------------------------------------------- fault injections
    faults: list[dict] = []

    cap_dir = ART / "fault_capacity"
    build_sample(cap_dir)
    try:
        _engine(cap_dir, Injection(kind="capacity", capacity_override=1)).run_initial()
    except Exception as exc:  # noqa: BLE001
        faults.append({"injection": "capacity", "code": exc.code, "context": exc.context})

    int_dir = ART / "fault_integrity"
    build_sample(int_dir)
    target = load_manifest(int_dir / "manifest.json").objects[0]
    p = staging_dir(int_dir) / target.digest
    p.write_bytes(p.read_bytes()[: max(1, target.size_bytes // 2)])
    try:
        _engine(int_dir).run_initial()
    except Exception as exc:  # noqa: BLE001
        faults.append({"injection": "truncate", "code": exc.code,
                       "context": {"key": exc.context.get("key")}})

    ver_dir = ART / "fault_version"
    build_sample(ver_dir)
    _engine(ver_dir).run_initial()
    from dataclasses import replace as dc_replace
    ver_manifest = dc_replace(load_manifest(ver_dir / "manifest.json"),
                              policy_version="policy-INJECTED",
                              tier_param_version="tier-INJECTED")
    logger = JsonlLogger(ver_dir / "logs" / "ver.jsonl")
    try:
        Engine(ver_dir, ver_manifest, logger).run_cycle()
    except Exception as exc:  # noqa: BLE001
        faults.append({"injection": "version", "code": exc.code,
                       "context": {
                           "snapshot_policy_version": exc.context["snapshot_policy_version"],
                           "session_policy_version": exc.context["session_policy_version"],
                       }})
    finally:
        logger.close()

    report = {
        "sessions": summary,
        "replay": {
            "session1": {"status": replay1["status"],
                         "migrations_replayed": replay1["migrations_replayed"]},
            "session2": {"status": replay2["status"],
                         "migrations_replayed": replay2["migrations_replayed"]},
            "session3": {"kill_exit_code": kill_rc,
                         "resolved_unknown": recovery["resolved_unknown_objects"],
                         "reran_migrations": recovery["reran_migrations"],
                         "status": replay3["status"],
                         "migrations_replayed": replay3["migrations_replayed"]},
        },
        "determinism": determinism,
        "fault_injections": faults,
        "snapshot_replacement": {
            "archived_snapshot": Path(replace_info["archived_snapshot"]).name,
            "tier_changes": replace_info["diff"]["tier_changes"],
            "object_changes": replace_info["diff"]["object_changes"],
        },
    }
    out = ART / "demo_summary.json"
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False))
    print(f"\nWrote stable summary to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
