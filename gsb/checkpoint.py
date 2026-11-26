"""Deterministic checkpoint commit, rollback and generation semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from .canonical import sha256_hex
from .errors import DiagnosticError
from .model import EventSpec, SessionSpec


@dataclass(frozen=True)
class CheckpointState:
    checkpoint_id: str
    shard_id: str
    declared_parameter_version: int
    parameter_version: int
    parameter_digest: str
    loss_baseline_digest: str
    generation: int
    committed: bool
    commit_seq: int | None = None
    batch_sequences: tuple[tuple[int, str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "shard_id": self.shard_id,
            "declared_parameter_version": self.declared_parameter_version,
            "parameter_version": self.parameter_version,
            "parameter_digest": self.parameter_digest,
            "loss_baseline_digest": self.loss_baseline_digest,
            "generation": self.generation,
            "committed": self.committed,
            "commit_seq": self.commit_seq,
            "batch_sequences": [list(row) for row in self.batch_sequences],
        }

    @property
    def state_digest(self) -> str:
        return sha256_hex(self.to_dict())


ParameterDigestLookup = Callable[[str, int], str]


class CheckpointManager:
    """A deterministic event state machine independent of worker completion order."""

    def __init__(
        self,
        session: SessionSpec,
        parameter_digest_lookup: ParameterDigestLookup,
    ) -> None:
        self.session = session
        self._lookup = parameter_digest_lookup
        self._current_version = {
            shard.shard_id: shard.parameter_version for shard in session.shards
        }
        self._generation = {shard.shard_id: 0 for shard in session.shards}
        self._batch_losses: dict[str, dict[str, tuple[int, str]]] = {
            shard.shard_id: {} for shard in session.shards
        }
        self._checkpoints: dict[str, CheckpointState] = {}
        for declared in session.checkpoints:
            self._checkpoints[declared.checkpoint_id] = CheckpointState(
                checkpoint_id=declared.checkpoint_id,
                shard_id=declared.shard_id,
                declared_parameter_version=declared.parameter_version,
                parameter_version=declared.parameter_version,
                parameter_digest=self._lookup(
                    declared.shard_id, declared.parameter_version
                ),
                loss_baseline_digest=declared.loss_baseline_digest,
                generation=0,
                committed=False,
            )
        self._restored_from: str | None = None

    def current_version(self, shard_id: str) -> int:
        return self._current_version[shard_id]

    def generation(self, shard_id: str) -> int:
        return self._generation[shard_id]

    def current_parameter_digest(self, shard_id: str) -> str:
        return self._lookup(shard_id, self._current_version[shard_id])

    def checkpoint(self, checkpoint_id: str) -> CheckpointState:
        return self._checkpoints[checkpoint_id]

    def checkpoints(self) -> tuple[CheckpointState, ...]:
        return tuple(
            self._checkpoints[key]
            for key in sorted(self._checkpoints)
        )

    def record_batch(
        self, *, sequence: int, shard_id: str, batch_id: str, loss_digest: str
    ) -> None:
        self._batch_losses[shard_id][batch_id] = (int(sequence), loss_digest)

    def _loss_baseline(
        self, shard_id: str, parameter_version: int, upto_seq: int
    ) -> tuple[str, tuple[tuple[int, str, str], ...]]:
        rows = tuple(
            (sequence, batch_id, loss_digest)
            for batch_id, (sequence, loss_digest) in sorted(
                self._batch_losses[shard_id].items(),
                key=lambda item: (item[1][0], item[0]),
            )
            if sequence < upto_seq
        )
        digest = sha256_hex(
            {
                "shard_id": shard_id,
                "parameter_version": parameter_version,
                "batches": [list(row) for row in rows],
            }
        )
        return digest, rows

    def commit(self, checkpoint_id: str, *, seq: int) -> CheckpointState:
        current = self._checkpoints[checkpoint_id]
        shard_id = current.shard_id
        version = self._current_version[shard_id]
        parameter_digest = self._lookup(shard_id, version)
        baseline_digest, rows = self._loss_baseline(shard_id, version, seq)
        updated = replace(
            current,
            parameter_version=version,
            parameter_digest=parameter_digest,
            loss_baseline_digest=baseline_digest,
            generation=self._generation[shard_id],
            committed=True,
            commit_seq=seq,
            batch_sequences=rows,
        )
        self._checkpoints[checkpoint_id] = updated
        return updated

    def rollback(self, checkpoint_id: str) -> CheckpointState:
        checkpoint = self._checkpoints[checkpoint_id]
        shard_id = checkpoint.shard_id
        # A declared (not yet committed) checkpoint is still a legitimate immutable
        # version reference. Its parameter digest was computed at load time.
        self._current_version[shard_id] = checkpoint.parameter_version
        self._generation[shard_id] += 1
        self._restored_from = checkpoint_id
        return replace(checkpoint, generation=self._generation[shard_id])

    def apply_event(self, event: EventSpec) -> CheckpointState | None:
        if event.type == "commit":
            assert event.checkpoint_id is not None
            return self.commit(event.checkpoint_id, seq=event.seq)
        if event.type == "rollback":
            assert event.checkpoint_id is not None
            return self.rollback(event.checkpoint_id)
        return None

    def snapshot(self) -> dict[str, Any]:
        versions = {key: self._current_version[key] for key in sorted(self._current_version)}
        generations = {key: self._generation[key] for key in sorted(self._generation)}
        checkpoints = {cp.checkpoint_id: cp.to_dict() for cp in self.checkpoints()}
        payload = {
            "current_versions": versions,
            "generations": generations,
            "checkpoints": checkpoints,
            "restored_from": self._restored_from,
        }
        return {**payload, "state_digest": sha256_hex(payload)}

    def state_for_shard(self, shard_id: str) -> dict[str, Any]:
        return {
            "parameter_version": self._current_version[shard_id],
            "generation": self._generation[shard_id],
            "checkpoints": {
                cp.checkpoint_id: cp.to_dict()
                for cp in self.checkpoints()
                if cp.shard_id == shard_id
            },
        }


def checkpoint_state_from_snapshot(
    checkpoint_id: str, payload: Mapping[str, Any]
) -> CheckpointState:
    try:
        return CheckpointState(
            checkpoint_id=checkpoint_id,
            shard_id=str(payload["shard_id"]),
            declared_parameter_version=int(payload["declared_parameter_version"]),
            parameter_version=int(payload["parameter_version"]),
            parameter_digest=str(payload["parameter_digest"]),
            loss_baseline_digest=str(payload["loss_baseline_digest"]),
            generation=int(payload["generation"]),
            committed=bool(payload["committed"]),
            commit_seq=int(payload["commit_seq"]) if payload.get("commit_seq") is not None else None,
            batch_sequences=tuple(
                (int(row[0]), str(row[1]), str(row[2]))
                for row in payload.get("batch_sequences", [])
            ),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise DiagnosticError(
            "E_REPLAY", f"invalid checkpoint snapshot for {checkpoint_id}: {exc}"
        ) from exc

