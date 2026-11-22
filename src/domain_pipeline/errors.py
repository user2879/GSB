from __future__ import annotations


class DomainPipelineError(Exception):
    code = "E_PIPELINE"


class SnapshotStaleError(DomainPipelineError):
    code = "E_SNAPSHOT_STALE"

    def __init__(self, message: str, *, expected: str | None = None, actual: str | None = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual

    def as_failure(self) -> dict[str, object]:
        failure: dict[str, object] = {
            "scope": "snapshot",
            "codes": [self.code],
            "message": str(self),
        }
        if self.expected is not None:
            failure["expected"] = self.expected
        if self.actual is not None:
            failure["actual"] = self.actual
        return failure


class ConfigPatchError(DomainPipelineError):
    code = "E_EDIT_CONFIG_PATH"
