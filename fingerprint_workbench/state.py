from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import digest_object
from .config import FingerprintConfig
from .errors import WorkbenchError


STATE_SCHEMA = 1


@dataclass(frozen=True)
class Member:
    track_id: str
    audio_sha256: str
    duration_samples: int
    display_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "audio_sha256": self.audio_sha256,
            "duration_samples": self.duration_samples,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Member":
        return cls(
            track_id=str(data["track_id"]),
            audio_sha256=str(data["audio_sha256"]),
            duration_samples=int(data["duration_samples"]),
            display_name=str(data.get("display_name", data["track_id"])),
        )


def _state_payload(
    members: Iterable[Member],
    fingerprint_config: FingerprintConfig,
    similarity_threshold: float,
    fingerprints: dict[str, str],
) -> dict[str, Any]:
    ordered_members = sorted(members, key=lambda member: member.track_id)
    ordered_fingerprints = {key: fingerprints[key] for key in sorted(fingerprints)}
    return {
        "members": [member.to_dict() for member in ordered_members],
        "fingerprint_config": fingerprint_config.to_dict(),
        "fingerprint_config_digest": fingerprint_config.digest,
        "similarity_threshold": round(float(similarity_threshold), 6),
        "fingerprints": ordered_fingerprints,
    }


@dataclass(frozen=True)
class LibraryState:
    version: int
    members: tuple[Member, ...]
    fingerprint_config: FingerprintConfig
    similarity_threshold: float
    fingerprints: tuple[tuple[str, str], ...]
    manifest_hash: str
    state_digest: str

    @classmethod
    def create(
        cls,
        *,
        version: int,
        members: Iterable[Member],
        fingerprint_config: FingerprintConfig,
        similarity_threshold: float,
        fingerprints: dict[str, str],
        manifest_hash: str,
    ) -> "LibraryState":
        if not 0.0 <= similarity_threshold <= 1.0:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "similarity_threshold must be between 0 and 1",
                {"similarity_threshold": similarity_threshold},
            )
        ordered_members = tuple(sorted(members, key=lambda member: member.track_id))
        if len({member.track_id for member in ordered_members}) != len(ordered_members):
            raise WorkbenchError("E_STATE_CONFLICT", "duplicate track ids in state")
        if set(fingerprints) != {member.track_id for member in ordered_members}:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "fingerprint mapping does not cover library members",
            )
        payload = _state_payload(
            ordered_members,
            fingerprint_config,
            similarity_threshold,
            fingerprints,
        )
        return cls(
            version=version,
            members=ordered_members,
            fingerprint_config=fingerprint_config,
            similarity_threshold=round(float(similarity_threshold), 6),
            fingerprints=tuple((key, fingerprints[key]) for key in sorted(fingerprints)),
            manifest_hash=manifest_hash,
            state_digest=digest_object(payload),
        )

    @property
    def fingerprint_map(self) -> dict[str, str]:
        return dict(self.fingerprints)

    @property
    def member_map(self) -> dict[str, Member]:
        return {member.track_id: member for member in self.members}

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "version": self.version,
            "members": [member.to_dict() for member in self.members],
            "fingerprint_config": self.fingerprint_config.to_dict(),
            "fingerprint_config_digest": self.fingerprint_config.digest,
            "similarity_threshold": self.similarity_threshold,
            "fingerprints": self.fingerprint_map,
            "manifest_hash": self.manifest_hash,
            "state_digest": self.state_digest,
        }

    @classmethod
    def from_document(cls, data: dict[str, Any]) -> "LibraryState":
        if data.get("schema") != STATE_SCHEMA:
            raise WorkbenchError(
                "E_INDEX_VERSION",
                "unsupported state schema",
                {"schema": data.get("schema"), "expected": STATE_SCHEMA},
            )
        state = cls.create(
            version=int(data["version"]),
            members=(Member.from_dict(item) for item in data["members"]),
            fingerprint_config=FingerprintConfig.from_dict(data["fingerprint_config"]),
            similarity_threshold=float(data["similarity_threshold"]),
            fingerprints={key: str(value) for key, value in data["fingerprints"].items()},
            manifest_hash=str(data["manifest_hash"]),
        )
        if state.state_digest != data.get("state_digest"):
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "state digest mismatch",
                {"stored": data.get("state_digest"), "actual": state.state_digest},
            )
        return state


@dataclass(frozen=True)
class Operation:
    version: int
    parent_version: int
    operation: dict[str, Any]
    state: LibraryState

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "version": self.version,
            "parent_version": self.parent_version,
            "operation": self.operation,
            "state": self.state.to_document(),
        }

    @classmethod
    def from_document(cls, data: dict[str, Any]) -> "Operation":
        if data.get("schema") != 1:
            raise WorkbenchError(
                "E_INDEX_VERSION",
                "unsupported operation schema",
                {"schema": data.get("schema")},
            )
        state = LibraryState.from_document(data["state"])
        operation = cls(
            version=int(data["version"]),
            parent_version=int(data["parent_version"]),
            operation=dict(data["operation"]),
            state=state,
        )
        if operation.version != state.version:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "operation/state version mismatch",
                {"operation_version": operation.version, "state_version": state.version},
            )
        return operation
