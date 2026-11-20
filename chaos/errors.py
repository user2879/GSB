"""Business error codes used across the pipeline."""

E_SPEC = "E_SPEC"          # declaration invalid
E_SEED = "E_SEED"          # seed / timeline digest mismatch
E_SNAPSHOT = "E_SNAPSHOT"  # ledger content digest mismatch
E_REVERT = "E_REVERT"      # revert verification failed
E_REPLAY = "E_REPLAY"      # replay recomputation mismatch

ALL_CODES = (E_SPEC, E_SEED, E_SNAPSHOT, E_REVERT, E_REPLAY)

EXIT_CODES = {E_SPEC: 2, E_SEED: 3, E_SNAPSHOT: 4, E_REVERT: 5, E_REPLAY: 6}


class ChaosError(Exception):
    """Structured business error.

    Attributes:
        code: one of ALL_CODES.
        field: first offending field path (for E_SPEC) or mismatch locator.
        evidence: opaque dict preserved in ledger / logs for diagnosis.
    """

    def __init__(self, code, message, field=None, evidence=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.field = field
        self.evidence = evidence or {}

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "evidence": self.evidence,
        }
