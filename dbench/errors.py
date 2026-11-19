"""Business error codes and the unified WorkbenchError exception.

Every recoverable validation / injection / snapshot failure is represented by
a stable string code so reports and CLI output can be audited deterministically.
"""

from __future__ import annotations


class WorkbenchError(Exception):
    """An error carrying a stable business code plus structured context."""

    def __init__(self, code: str, message: str, context: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})

    def to_dict(self) -> dict:
        data = {"code": self.code, "message": self.message}
        if self.context:
            data["context"] = self.context
        return data


# Validation codes (requirement 5)
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SEED_OUT_OF_RANGE = "E_SEED_OUT_OF_RANGE"
E_GUIDE_NOT_FINITE = "E_GUIDE_NOT_FINITE"
E_STEP_ZERO = "E_STEP_ZERO"
E_NOISE_TABLE_MISMATCH = "E_NOISE_TABLE_MISMATCH"

# Patch codes (requirement 4)
E_PATCH_OUT_OF_RANGE = "E_PATCH_OUT_OF_RANGE"
E_PATCH_SHAPE_MISMATCH = "E_PATCH_SHAPE_MISMATCH"

# Snapshot codes (requirement 1 / 8)
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"

# Injection codes (requirement 6)
E_NON_FINITE_LATENT = "E_NON_FINITE_LATENT"
E_IMAGE_SIZE_MISMATCH = "E_IMAGE_SIZE_MISMATCH"
E_NOISE_ENTRY_TRUNCATED = "E_NOISE_ENTRY_TRUNCATED"
E_INJECTED_PATCH_OUT_OF_RANGE = "E_INJECTED_PATCH_OUT_OF_RANGE"

# Image decoding failure (used by the initial image encoding stage)
E_IMAGE_DECODE_FAILED = "E_IMAGE_DECODE_FAILED"

VALIDATION_CODES = {
    E_CONFIG_INVALID,
    E_SEED_OUT_OF_RANGE,
    E_GUIDE_NOT_FINITE,
    E_STEP_ZERO,
    E_NOISE_TABLE_MISMATCH,
    E_PATCH_OUT_OF_RANGE,
    E_PATCH_SHAPE_MISMATCH,
}

INJECTION_CODES = {
    E_NON_FINITE_LATENT,
    E_IMAGE_SIZE_MISMATCH,
    E_NOISE_ENTRY_TRUNCATED,
    E_INJECTED_PATCH_OUT_OF_RANGE,
}
