"""Three-scenario deterministic demonstration used by CLI and Tkinter UI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .engine import CascadeEngine, EngineRunResult
from .error_propagation import (
    compare_configurations,
    propagate_fixed_errors,
)
from .generator import DeterministicUdpGenerator
from .ledger import DecisionLedger
from .models import ErrorPropagationResult, IndistinguishabilityResult, PolicySet
from .policy import load_policy
from .replay import verify_snapshot_replay
from .samples import (
    single_stage_generator,
    single_stage_policy,
    three_stage_generator,
    three_stage_near_equal_policy,
    three_stage_policy,
)
from .snapshot import (
    SnapshotDocument,
    SnapshotManager,
    current_session_descriptor,
)
from .transport import InMemoryTransport


DEMO_SCALE = 3


def _clean_ledger(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _run_policy(
    policy: PolicySet,
    generator: DeterministicUdpGenerator,
    *,
    ledger_path: Path,
    use_loopback: bool,
    base_port: int,
) -> tuple[EngineRunResult, dict[str, Any]]:
    _clean_ledger(ledger_path)
    with DecisionLedger(ledger_path) as ledger:
        with CascadeEngine(
            policy,
            generator,
            scale=DEMO_SCALE,
            use_loopback=use_loopback,
            base_port=base_port,
        ) as engine:
            run = engine.run(ledger=ledger)
        errors = propagate_fixed_errors(
            policy,
            run.summary,
            run.events,
            scale=DEMO_SCALE,
            generator_digest=generator.sequence_digest,
        )
        ledger_info = {
            "cursor_seq": ledger.progress().cursor_seq,
            "committed_batches": ledger.progress().committed_batches,
            "content_digest": ledger.progress().content_digest,
            "decision_count": len(ledger.decisions()),
        }
    return run, {"errors": errors, "ledger": ledger_info}


def _run_without_ledger(
    policy: PolicySet, generator: DeterministicUdpGenerator
) -> tuple[EngineRunResult, ErrorPropagationResult]:
    transport = InMemoryTransport([shaper.shaper_id for shaper in policy.shapers])
    with CascadeEngine(
        policy,
        generator,
        scale=DEMO_SCALE,
        transport=transport,
    ) as engine:
        run = engine.run()
    errors = propagate_fixed_errors(
        policy,
        run.summary,
        run.events,
        scale=DEMO_SCALE,
        generator_digest=generator.sequence_digest,
    )
    return run, errors


def _snapshot_current(
    policy: PolicySet,
    generator: DeterministicUdpGenerator,
    document: SnapshotDocument,
) -> dict[str, Any]:
    descriptor = current_session_descriptor(
        policy=policy, scale=DEMO_SCALE, generator=generator
    )
    descriptor["content_digest"] = document.content_digest
    descriptor["ledger_digest"] = document.ledger.get("content_digest")
    descriptor["snapshot_path"] = str(document.path) if document.path else None
    return descriptor


def run_demo(output_dir: str | Path = "artifacts/demo") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_directory = output / "snapshots"
    if snapshot_directory.exists():
        shutil.rmtree(snapshot_directory)
    snapshots = SnapshotManager(snapshot_directory)

    single_policy = load_policy(single_stage_policy())
    single_generator = DeterministicUdpGenerator(single_stage_generator())
    single_run, single_extra = _run_policy(
        single_policy,
        single_generator,
        ledger_path=output / "single_stage.sqlite",
        use_loopback=True,
        base_port=54100,
    )
    with DecisionLedger(output / "single_stage.sqlite") as single_ledger:
        single_ledger.initialize(
            policy=single_policy, scale=DEMO_SCALE, generator=single_generator
        )
        single_document = snapshots.save(
            filename="single-stage.json",
            policy=single_policy,
            scale=DEMO_SCALE,
            generator=single_generator,
            ledger=single_ledger,
            errors=single_extra["errors"],
        )
    single_extra["snapshot"] = single_document

    cascade_policy = load_policy(three_stage_policy())
    cascade_generator = DeterministicUdpGenerator(three_stage_generator())
    cascade_run, cascade_extra = _run_policy(
        cascade_policy,
        cascade_generator,
        ledger_path=output / "three_stage.sqlite",
        use_loopback=True,
        base_port=54110,
    )
    with DecisionLedger(output / "three_stage.sqlite") as cascade_ledger:
        cascade_ledger.initialize(
            policy=cascade_policy, scale=DEMO_SCALE, generator=cascade_generator
        )
        cascade_document = snapshots.save(
            filename="three-stage.json",
            policy=cascade_policy,
            scale=DEMO_SCALE,
            generator=cascade_generator,
            ledger=cascade_ledger,
            errors=cascade_extra["errors"],
        )

    near_policy = load_policy(three_stage_near_equal_policy())
    near_generator = DeterministicUdpGenerator(three_stage_generator())
    near_run, near_errors = _run_without_ledger(near_policy, near_generator)
    comparison = compare_configurations(cascade_extra["errors"], near_errors)

    single_current = _snapshot_current(
        single_policy, single_generator, single_document
    )
    restore = snapshots.open(cascade_document.path, current=single_current)
    restored_ledger_path = restore.document.restore_ledger(
        output / "restored_three_stage.sqlite", overwrite=True
    )
    with DecisionLedger(restored_ledger_path) as restored_ledger:
        restored_ledger.initialize(
            policy=restore.document.policy,
            scale=restore.document.scale,
            generator=DeterministicUdpGenerator(restore.document.generator_config),
        )
        restored_ledger_info = {
            "cursor_seq": restored_ledger.progress().cursor_seq,
            "committed_batches": restored_ledger.progress().committed_batches,
            "content_digest": restored_ledger.progress().content_digest,
            "decision_count": len(restored_ledger.decisions()),
        }
    replay = verify_snapshot_replay(
        restore.document,
        current_policy_version=cascade_policy.policy_version,
        current_scale=DEMO_SCALE,
        current_generator_digest=cascade_generator.sequence_digest,
    )
    backup_read_only = False
    if restore.backup_path is not None:
        backup_read_only = not (restore.backup_path.stat().st_mode & 0o222)

    result: dict[str, Any] = {
        "scale": DEMO_SCALE,
        "scenarios": [
            {
                "name": "single_stage",
                "policy_version": single_policy.policy_version,
                "summary": single_run.summary.as_dict(),
                "errors": single_extra["errors"].as_dict(),
                "ledger": single_extra["ledger"],
            },
            {
                "name": "three_stage",
                "policy_version": cascade_policy.policy_version,
                "topology": list(cascade_policy.topological_order()),
                "summary": cascade_run.summary.as_dict(),
                "errors": cascade_extra["errors"].as_dict(),
                "indistinguishability": comparison.as_dict(),
                "near_equal": near_run.summary.as_dict(),
                "ledger": cascade_extra["ledger"],
            },
            {
                "name": "snapshot_restore_replay",
                "snapshot": cascade_document.path.name if cascade_document.path else "",
                "content_digest": cascade_document.content_digest,
                "restore": restore.as_dict(),
                "backup_read_only": backup_read_only,
                "restored_ledger": restored_ledger_info,
                "replay": replay.as_dict(),
            },
        ],
        "snapshots": list(snapshots.list_snapshots()),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(summary_path)
    return result


def format_demo_summary(result: dict[str, Any]) -> str:
    lines: list[str] = ["QoS fixed-point diagnostic demo"]
    for scenario in result["scenarios"]:
        name = scenario["name"]
        if name in {"single_stage", "three_stage"}:
            summary = scenario["summary"]
            errors = scenario["errors"]
            lines.append(f"SCENARIO {name}")
            lines.append(f"  policy_version={scenario['policy_version']}")
            lines.append(
                f"  generated={summary['total_generated_packets']} "
                f"released={summary['final_released_packets']} "
                f"released_bytes={summary['final_released_bytes']}"
            )
            lines.append(
                f"  bytes_error_bound={errors['bytes_error_bound']} "
                f"avg_delay_ns={errors['average_delay_ns']} "
                f"delay_error_bound_ns={errors['average_delay_error_bound_ns']}"
            )
            lines.append(f"  dominant_stage={errors['dominant_stage']}")
            if name == "three_stage":
                decision = scenario["indistinguishability"]
                lines.append(
                    "  indistinguishable="
                    f"{str(decision['indistinguishable']).lower()} "
                    f"bytes_diff={decision['released_bytes_difference']} "
                    f"combined_bound={decision['combined_bytes_error_bound']}"
                )
        else:
            restore = scenario["restore"]
            replay = scenario["replay"]
            lines.append("SCENARIO snapshot_restore_replay")
            lines.append(f"  snapshot={scenario['snapshot']}")
            lines.append(
                f"  applied={str(restore['applied']).lower()} "
                f"replaced_fields={','.join(restore['replaced_fields'])}"
            )
            lines.append(f"  backup_read_only={str(scenario['backup_read_only']).lower()}")
            restored = scenario["restored_ledger"]
            lines.append(
                f"  restored_ledger_cursor={restored['cursor_seq']} "
                f"decisions={restored['decision_count']}"
            )
            lines.append(
                f"  replay_ok={str(replay['ok']).lower()} "
                f"events_compared={replay['events_compared']} "
                f"classification={replay['classification']}"
            )
    lines.append("SNAPSHOTS " + ",".join(item["filename"] for item in result["snapshots"]))
    lines.append("DONE")
    return "\n".join(lines)
