"""Stable error codes shared by validation, scheduling, and reporting."""

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_BLOCK_NOT_FOUND = "E_BLOCK_NOT_FOUND"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_CLUSTER_QUOTA_EXCEEDED = "E_CLUSTER_QUOTA_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"


class ValidationError(ValueError):
    """A validation failure carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class ConfigError(ValidationError):
    """One or more configuration errors; the first error determines the code."""

    def __init__(self, issues: list[ValidationError]) -> None:
        if not issues:
            raise ValueError("ConfigError requires at least one issue")
        self.issues = issues
        first = issues[0]
        super().__init__(first.code, first.message, first.path)

    def as_list(self) -> list[dict[str, str]]:
        return [issue.as_dict() for issue in self.issues]


class SnapshotError(RuntimeError):
    """Snapshot loading or validation failed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
