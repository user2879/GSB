from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BusinessError:
    code: str
    path: str
    message: str
    stage: str = "runtime"
    window_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def sort_key(self) -> tuple[str, int, str, str, str]:
        stage_order = {
            "config": 0,
            "edit": 1,
            "window": 2,
            "lease": 3,
            "snapshot": 4,
            "cluster": 5,
            "density": 6,
            "runtime": 7,
        }
        return (
            f"{stage_order.get(self.stage, 99):02d}",
            0,
            self.path,
            self.code,
            self.window_id or "",
        )


class InsDiagError(Exception):
    code = "E_RUNTIME"
    stage = "runtime"

    def __init__(self, message: str, path: str = "$", window_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.window_id = window_id

    def business_error(self) -> BusinessError:
        return BusinessError(
            code=self.code,
            path=self.path,
            message=self.message,
            stage=self.stage,
            window_id=self.window_id,
        )


class ConfigValidationError(InsDiagError):
    code = "E_CONFIG_INVALID"
    stage = "config"

    def __init__(self, issues: list[BusinessError]):
        super().__init__("configuration validation failed")
        self.issues = sorted(issues, key=lambda item: item.sort_key)


class WindowProcessingError(InsDiagError):
    stage = "window"

    def __init__(
        self,
        message: str,
        path: str = "$",
        window_id: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message, path=path, window_id=window_id)
        self.code = code or "E_WINDOW_PROCESSING"


class LeaseError(InsDiagError):
    stage = "lease"


class LeaseExpiredError(LeaseError):
    code = "E_LEASE_EXPIRED"


class LeaseGenerationRollbackError(LeaseError):
    code = "E_LEASE_GENERATION_ROLLBACK"


class SnapshotError(InsDiagError):
    stage = "snapshot"


class SnapshotStaleError(SnapshotError):
    code = "E_SNAPSHOT_STALE"


class SnapshotCorruptError(SnapshotError):
    code = "E_SNAPSHOT_CORRUPT"


class CacheError(InsDiagError):
    pass


@dataclass
class FailureCollector:
    items: list[BusinessError] = field(default_factory=list)

    def add(self, error: BusinessError) -> None:
        self.items.append(error)

    def extend(self, errors: list[BusinessError]) -> None:
        self.items.extend(errors)

    def sorted(self) -> list[BusinessError]:
        unique: dict[tuple[str, str, str, str | None], BusinessError] = {}
        for item in self.items:
            unique[(item.code, item.path, item.stage, item.window_id)] = item
        return sorted(unique.values(), key=lambda item: item.sort_key)
