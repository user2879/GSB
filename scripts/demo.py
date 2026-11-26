from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckptsim.canonical import canonical_json, sha256_value  # noqa: E402
from ckptsim.impact import expected_batch_records, selective_recompute  # noqa: E402
from ckptsim.ledger import Ledger, SnapshotIdentity  # noqa: E402
from ckptsim.plan import build_event_plan  # noqa: E402
from ckptsim.runner import Runner, RunnerOptions  # noqa: E402
from ckptsim.spec import load_bundle, load_layers  # noqa: E402
from ckptsim.verify import verify_ledger  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def deterministic_projection(result: dict[str, object]) -> dict[str, object]:
    batches = result["batches"]
    checkpoints = result["checkpoints"]
    return {
        "batches": [
            {
                "shard_id": item["shard_id"],
                "batch_id": item["batch_id"],
                "parameter_version": item["parameter_version"],
                "generation": item["generation"],
                "parameter_digest": item["parameter_digest"],
                "samples_digest": item["samples_digest"],
                "loss_digest": item["loss_digest"],
                "checkpoint_id": item["checkpoint_id"],
                "checkpoint_status": item["checkpoint_status"],
            }
            for item in batches
        ],
        "checkpoints": [
            {
                "checkpoint_id": item["checkpoint_id"],
                "parameter_version": item["parameter_version"],
                "generation": item["generation"],
                "parameter_digests": item["parameter_digests"],
                "batch_keys": item["batch_keys"],
                "loss_baseline": item["loss_baseline"],
                "baseline_digest": item["baseline_digest"],
                "status": item["status"],
            }
            for item in checkpoints
        ],
        "rollbacks": result["rollbacks"],
    }


def main() -> int:
    demo_dir = ROOT / "artifacts" / "demo"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True)
    bundle = load_bundle(
        ROOT / "samples" / "shards.json",
        ROOT / "samples" / "layers.json",
        ROOT / "samples" / "checkpoints.json",
        ROOT / "samples" / "events.json",
    )
    plan = build_event_plan(bundle)
    ledger = demo_dir / "ledger.sqlite"
    first = Runner(
        bundle,
        RunnerOptions(ledger, lease_ttl_ms=30_000, client_timeout=5.0),
    ).run()
    write_json(demo_dir / "run.json", first)
    first_projection = deterministic_projection(first)
    first_digest = sha256_value(first_projection)
    write_json(demo_dir / "deterministic-run.json", first_projection)

    second_dir = demo_dir / "repeat"
    second = Runner(
        bundle,
        RunnerOptions(second_dir / "ledger.sqlite", lease_ttl_ms=30_000, client_timeout=5.0),
    ).run()
    second_projection = deterministic_projection(second)
    second_digest = sha256_value(second_projection)
    write_json(second_dir / "run.json", second)
    write_json(demo_dir / "deterministic-repeat.json", second_projection)

    verified = verify_ledger(bundle, ledger, raise_on_error=False)
    write_json(demo_dir / "verify.json", verified.to_dict())

    baseline = expected_batch_records(bundle, plan)
    edited = load_layers(ROOT / "samples" / "layers_edited.json")
    impact = selective_recompute(bundle, edited, plan, baseline)
    write_json(demo_dir / "impact.json", impact)
    verification_ledger = Ledger.open(
        ledger,
        identity=SnapshotIdentity.from_bundle(bundle),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    try:
        impact_digest = verification_ledger.record_impact(impact["impact"])
    finally:
        verification_ledger.close()
    verified_with_impact = verify_ledger(
        bundle,
        ledger,
        edited_layers_path=ROOT / "samples" / "layers_edited.json",
        raise_on_error=False,
    )
    write_json(demo_dir / "verify-impact.json", verified_with_impact.to_dict())

    crash_dir = demo_dir / "crash"
    killed: list[str] = []

    def kill_once(batch, manager) -> None:
        if batch.shard_id == "shard-a" and not killed:
            killed.append(batch.batch_id)
            manager.kill("shard-a")

    crash = Runner(
        bundle,
        RunnerOptions(
            crash_dir / "ledger.sqlite",
            lease_ttl_ms=30_000,
            client_timeout=5.0,
        ),
        on_batch_started=kill_once,
    ).run()
    write_json(crash_dir / "run.json", crash)
    crash_verify = verify_ledger(bundle, crash_dir / "ledger.sqlite", raise_on_error=False)
    write_json(crash_dir / "verify.json", crash_verify.to_dict())

    changed_bundle = load_bundle(
        ROOT / "samples" / "shards.json",
        ROOT / "samples" / "layers_edited.json",
        ROOT / "samples" / "checkpoints.json",
        ROOT / "samples" / "events.json",
    )
    changed_ledger = Ledger.open(
        ledger,
        identity=SnapshotIdentity.from_bundle(changed_bundle),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    try:
        snapshot_report = changed_ledger.snapshot_report.to_dict()
        snapshot_batches = changed_ledger.list_batches()
    finally:
        changed_ledger.close()
    write_json(
        demo_dir / "snapshot-replacement.json",
        {
            "snapshot": snapshot_report,
            "batches_after_replacement": len(snapshot_batches),
        },
    )

    summary = {
        "ok": all(
            [
                first["ok"],
                second["ok"],
                verified.verified,
                verified_with_impact.verified,
                crash["ok"],
                crash_verify.verified,
                first_digest == second_digest,
                snapshot_report.get("rebuilt"),
            ]
        ),
        "deterministic_digest_first": first_digest,
        "deterministic_digest_second": second_digest,
        "deterministic_repeat_match": first_digest == second_digest,
        "verification": verified.to_dict(),
        "verification_with_impact": verified_with_impact.to_dict(),
        "impact": {
            "affected_batches": impact["impact"]["affected_batches"],
            "unchanged_batches": impact["impact"]["unchanged_batches"],
            "affected_checkpoints": impact["impact"]["affected_checkpoints"],
            "recomputed_batch_count": impact["recomputed_batch_count"],
            "reused_batch_count": impact["reused_batch_count"],
            "recomputed_checkpoints": impact["recomputed_checkpoints"],
            "unchanged_verified": impact["unchanged_verified"],
            "impact_digest": impact_digest,
        },
        "forced_termination": {
            "killed_batch": killed[0] if killed else None,
            "unknown_attempts": crash["unknown_attempts"],
            "recovery_events": crash["recovery_events"],
            "verification": crash_verify.to_dict(),
            "forced_exit_report": crash["forced_exit_report"],
        },
        "snapshot_replacement": snapshot_report,
        "artifacts_dir": str(demo_dir),
    }
    write_json(demo_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
