"""Recovery, replay verification and crash-injection session runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_bytes, first_diff
from .engine import (
    PATH_INCREMENTAL,
    EngineState,
    advance_round,
    initial_runtime,
)
from .errors import DivergenceError, ReplayError
from .ledger import Ledger, RecoveryState, conclusion_digest, read_snapshot
from .statemachine import initial_states


@dataclass
class ReplayReport:
    verified_rounds: list[int] = field(default_factory=list)
    rerun_rounds: list[int] = field(default_factory=list)
    digests: dict[int, str] = field(default_factory=dict)


def replay_from_ledger(scene, ledger: Ledger, path_name: str = PATH_INCREMENTAL) -> ReplayReport:
    """Re-advance remaining rounds from the last committed cursor and verify."""

    recovery = ledger.initialize_session(scene, path_name)
    engine = EngineState(
        scene=scene,
        instance_states=initial_states(scene),
        runtime=initial_runtime(scene),
        last_switch_at_ms={service_id: None for service_id in scene.services},
    )
    report = ReplayReport()
    previous = None
    for round_index in scene.rounds:
        conclusion = advance_round(
            scene,
            engine,
            round_index,
            observed_only=True,
            previous_conclusion=previous,
        )
        previous = conclusion
        if round_index in recovery.recovered_rounds:
            stored = ledger.load_round(round_index)
            if canonical_bytes(stored) != canonical_bytes(conclusion):
                diff = first_diff(stored, conclusion)
                raise ReplayError(
                    f"replay cannot reproduce round {round_index}",
                    f"rounds[{round_index}]",
                    {"first_diff": diff},
                )
            ledger.verify_digest(round_index, conclusion)
            report.verified_rounds.append(round_index)
            report.digests[round_index] = conclusion_digest(conclusion)
        else:
            ledger.begin_round(round_index, path_name)
            digest = ledger.commit_round(round_index, path_name, conclusion)
            report.rerun_rounds.append(round_index)
            report.digests[round_index] = digest
    return report


def compare_path_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    names = sorted(outputs)
    baseline_name = names[0]
    baseline = outputs[baseline_name]
    for other in names[1:]:
        candidate = outputs[other]
        if len(baseline) != len(candidate):
            raise DivergenceError(
                "round count differs",
                "$",
                {"paths": [baseline_name, other], "lengths": [len(baseline), len(candidate)]},
            )
        for index, (left, right) in enumerate(zip(baseline, candidate)):
            if canonical_bytes(left) != canonical_bytes(right):
                diff = first_diff(left, right)
                raise DivergenceError(
                    f"paths {baseline_name} and {other} diverge at round {left['round_index']}",
                    f"rounds[{index}]",
                    {"first_diff": diff, "paths": [baseline_name, other]},
                )


def verify_snapshot_version(snapshot_dir, scene) -> dict[str, Any]:
    payload = read_snapshot(snapshot_dir, scene.scene_id)
    from .ledger import scene_digest

    if payload.get("scene_version") != scene.scene_version:
        raise SnapshotVersionMismatch(payload, "scene_version mismatch")
    if payload.get("scene_digest") != scene_digest(scene):
        raise SnapshotVersionMismatch(payload, "scene_digest mismatch")
    return payload


class SnapshotVersionMismatch(Exception):
    def __init__(self, payload: dict[str, Any], reason: str):
        super().__init__(reason)
        self.payload = payload
        self.reason = reason


def run_session_with_crash(
    scene,
    ledger: Ledger,
    path_name: str,
    crash_round: int | None,
    *,
    crash_before_commit: bool = True,
) -> dict[str, Any]:
    """Drive a session; optionally leave a round pending to simulate kill -9."""

    recovery = ledger.initialize_session(scene, path_name)
    engine = EngineState(
        scene=scene,
        instance_states=initial_states(scene),
        runtime=initial_runtime(scene),
        last_switch_at_ms={service_id: None for service_id in scene.services},
    )
    committed: list[int] = []
    pending: int | None = None
    previous = None
    for round_index in scene.rounds:
        if round_index in recovery.recovered_rounds:
            conclusion = ledger.load_round(round_index)
            previous = conclusion
            committed.append(round_index)
            continue
        conclusion = advance_round(
            scene,
            engine,
            round_index,
            observed_only=True,
            previous_conclusion=previous,
        )
        previous = conclusion
        ledger.begin_round(round_index, path_name)
        if crash_round == round_index and crash_before_commit:
            pending = round_index
            break
        digest = ledger.commit_round(round_index, path_name, conclusion)
        committed.append(round_index)
    return {
        "committed_rounds": committed,
        "pending_round": pending,
        "recovery": recovery,
    }
