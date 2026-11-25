from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .util import atomic_write_json, fingerprint, read_json


@dataclass
class SnapshotValidation:
    snapshot_present: bool
    checksum_valid: bool | None
    restoreable: bool
    errors: list[BusinessError]
    compared: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_present": self.snapshot_present,
            "checksum_valid": self.checksum_valid,
            "restoreable": self.restoreable,
            "errors": [error.to_dict() for error in self.errors],
            "compared": self.compared,
        }


def _stale(path: str, message: str) -> BusinessError:
    return BusinessError(
        code="E_SNAPSHOT_STALE",
        path=path,
        message=message,
        stage="snapshot",
    )


class SnapshotStore:
    def __init__(self, state_dir: Path, clock=None):
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "snapshot.json"
        self.clock = clock

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            value = read_json(self.path)
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def checksum_valid(snapshot: dict[str, Any]) -> bool:
        checksum = snapshot.get("checksum")
        if not isinstance(checksum, str):
            return False
        body = {key: value for key, value in snapshot.items() if key != "checksum"}
        return checksum == fingerprint(body)

    def current_global_version(self) -> int:
        snapshot = self.read()
        if snapshot is None:
            return 0
        value = snapshot.get("global_version", 0)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

    def evaluate_restore(
        self,
        *,
        expected_global_version: int,
        expected_config_hash: str,
        expected_data_fingerprint: str,
        expected_lease_generation: int,
        expected_lease_map: dict[str, int] | None = None,
    ) -> SnapshotValidation:
        snapshot = self.read()
        if snapshot is None:
            return SnapshotValidation(
                snapshot_present=False,
                checksum_valid=None,
                restoreable=False,
                errors=[],
                compared={
                    "expected_global_version": expected_global_version,
                    "expected_config_hash": expected_config_hash,
                    "expected_data_fingerprint": expected_data_fingerprint,
                    "expected_lease_generation": expected_lease_generation,
                },
            )
        checksum_valid = self.checksum_valid(snapshot)
        errors: list[BusinessError] = []
        if not checksum_valid:
            errors.append(
                BusinessError(
                    code="E_SNAPSHOT_CORRUPT",
                    path="snapshot.checksum",
                    message="snapshot checksum does not match its canonical payload",
                    stage="snapshot",
                )
            )
        comparisons = {
            "global_version": (snapshot.get("global_version"), expected_global_version),
            "config_hash": (snapshot.get("config_hash"), expected_config_hash),
            "data_fingerprint": (snapshot.get("data_fingerprint"), expected_data_fingerprint),
            "lease_generation": (snapshot.get("lease_generation"), expected_lease_generation),
        }
        for field_name, (actual, expected) in comparisons.items():
            if actual != expected:
                errors.append(
                    _stale(
                        f"snapshot.{field_name}",
                        f"snapshot {field_name} mismatch: persisted={actual!r}, current={expected!r}",
                    )
                )
        if expected_lease_map is not None:
            persisted_map = snapshot.get("lease_map")
            if persisted_map != expected_lease_map:
                errors.append(
                    _stale(
                        "snapshot.lease_map",
                        "snapshot lease map does not match persisted lease generations",
                    )
                )
        return SnapshotValidation(
            snapshot_present=True,
            checksum_valid=checksum_valid,
            restoreable=not errors,
            errors=errors,
            compared={
                "persisted": {key: snapshot.get(key) for key in comparisons},
                "expected": {
                    "global_version": expected_global_version,
                    "config_hash": expected_config_hash,
                    "data_fingerprint": expected_data_fingerprint,
                    "lease_generation": expected_lease_generation,
                    "lease_map": expected_lease_map,
                },
            },
        )

    def audit_supersession(
        self,
        *,
        new_global_version: int,
        new_config_hash: str,
        new_data_fingerprint: str,
        new_lease_generation: int,
        new_lease_map: dict[str, int],
    ) -> SnapshotValidation:
        snapshot = self.read()
        if snapshot is None:
            return SnapshotValidation(
                snapshot_present=False,
                checksum_valid=None,
                restoreable=False,
                errors=[],
                compared={},
            )
        errors: list[BusinessError] = []
        if not self.checksum_valid(snapshot):
            errors.append(
                BusinessError(
                    code="E_SNAPSHOT_CORRUPT",
                    path="snapshot.checksum",
                    message="existing snapshot checksum is invalid",
                    stage="snapshot",
                )
            )
        expected_previous_version = new_global_version - 1
        comparisons = {
            "global_version": (snapshot.get("global_version"), expected_previous_version),
            "config_hash": (snapshot.get("config_hash"), new_config_hash),
            "data_fingerprint": (snapshot.get("data_fingerprint"), new_data_fingerprint),
            "lease_generation": (snapshot.get("lease_generation"), new_lease_generation),
            "lease_map": (snapshot.get("lease_map"), new_lease_map),
        }
        for field_name, (actual, expected) in comparisons.items():
            if actual != expected:
                errors.append(
                    _stale(
                        f"snapshot.{field_name}",
                        "old snapshot is not current; new snapshot will supersede it",
                    )
                )
        return SnapshotValidation(
            snapshot_present=True,
            checksum_valid=self.checksum_valid(snapshot),
            restoreable=not errors,
            errors=errors,
            compared={
                "old": {key: snapshot.get(key) for key in comparisons},
                "new": {
                    "global_version": new_global_version,
                    "config_hash": new_config_hash,
                    "data_fingerprint": new_data_fingerprint,
                    "lease_generation": new_lease_generation,
                    "lease_map": new_lease_map,
                },
            },
        )

    def commit(
        self,
        *,
        global_version: int,
        config_hash: str,
        data_fingerprint: str,
        lease_generation: int,
        lease_map: dict[str, int],
        payload: dict[str, Any],
        cache_digest: str,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "schema_version": 1,
            "global_version": int(global_version),
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "lease_generation": int(lease_generation),
            "lease_map": dict(sorted(lease_map.items())),
            "cache_digest": cache_digest,
            "created_monotonic_ms": (
                self.clock.now_ms() if self.clock is not None else None
            ),
            "payload": payload,
        }
        snapshot["checksum"] = fingerprint(snapshot)
        atomic_write_json(self.path, snapshot)
        return snapshot

